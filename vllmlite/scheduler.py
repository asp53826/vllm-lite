"""Continuous batching scheduler.

The whole point: instead of running a fixed batch to completion, the batch gets
rebuilt every single forward pass. A sequence that finishes frees its slot
immediately and a queued request takes it on the very next step, so we never
sit half idle waiting for the slowest member of a static batch.

Prefill and decode share one token budget (chunked prefill), which stops a long
prompt from stalling everyone else's decoding for several steps in a row.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .config import CacheConfig, SchedulerConfig
from .sequence import Sequence, Status


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int  # tokens to push through the model this step
    is_prefill: bool

    @property
    def start_pos(self) -> int:
        return self.seq.num_computed


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    cow: list[tuple[int, int]] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled if s.is_prefill)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, sched_cfg: SchedulerConfig, cache_cfg: CacheConfig):
        self.cfg = sched_cfg
        self.blocks = BlockManager(cache_cfg)
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        # kept for debugging only, bounded so a long lived server doesn't grow
        self.finished: deque[Sequence] = deque(maxlen=2048)
        # seqs that died outside a normal sampling step, engine drains these
        self.pending_reports: list[Sequence] = []
        self.watermark_blocks = int(sched_cfg.watermark * cache_cfg.num_blocks)
        self.total_preemptions = 0

    def add(self, seq: Sequence) -> None:
        if len(self.waiting) >= self.cfg.max_waiting:
            raise RuntimeError("queue is full")
        self.waiting.append(seq)

    def abort(self, request_id: str) -> bool:
        for seq in list(self.running):
            if seq.request_id == request_id:
                self.running.remove(seq)
                self._retire(seq, Status.ABORTED, "abort")
                return True
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                self._retire(seq, Status.ABORTED, "abort")
                return True
        return False

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def drain_reports(self) -> list[Sequence]:
        out, self.pending_reports = self.pending_reports, []
        return out

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_batched_tokens
        slots = self.cfg.max_seqs

        # running sequences first. they already hold cache blocks and somebody
        # is watching their tokens stream in.
        queue = deque(self.running)
        keep: list[Sequence] = []

        while queue:
            seq = queue.popleft()
            if budget <= 0 or slots <= 0:
                keep.append(seq)
                continue

            want = min(seq.num_uncomputed, budget)
            if seq.draft_token_ids and want < seq.num_uncomputed:
                # not enough budget for the full guess. drop the tail of it
                # rather than splitting a speculation across two steps.
                seq.trim_drafts(want - 1)
                want = seq.num_uncomputed

            # free room by dropping the newest sequences that haven't been
            # scheduled yet this step. oldest-first keeps latency fair.
            while not self.blocks.can_allocate(seq, want) and queue:
                self._preempt(queue.pop(), out)

            if not self.blocks.can_allocate(seq, want):
                if not keep and out.is_empty:
                    # alone in the cache and it still doesn't fit, this request
                    # can never make progress
                    self._retire(seq, Status.FINISHED_LENGTH, "cache_full")
                else:
                    self._preempt(seq, out)
                continue

            out.cow += self.blocks.append_slots(seq, want)
            out.scheduled.append(ScheduledSeq(seq, want, seq.num_uncomputed > 1))
            keep.append(seq)
            budget -= want
            slots -= 1

        self.running = keep

        # then admit new work with whatever budget is left over
        while self.waiting and budget > 0 and slots > 0:
            seq = self.waiting[0]
            if seq.num_computed == 0 and not seq.table.block_ids:
                self.blocks.reuse_prefix(seq)

            want = seq.num_uncomputed
            if self.cfg.chunked_prefill:
                want = min(want, budget)
            elif want > budget:
                break  # whole prompt won't fit this step, try again next one

            if not self.blocks.can_allocate(seq, want, self.watermark_blocks):
                if not self.running and out.is_empty and not self.blocks.can_allocate(seq, want):
                    # idle engine, no watermark, and it still doesn't fit. this
                    # prompt is longer than the entire cache so waiting for room
                    # would just spin forever.
                    self.waiting.popleft()
                    self._retire(seq, Status.FINISHED_LENGTH, "cache_full")
                    continue
                break

            self.waiting.popleft()
            out.cow += self.blocks.append_slots(seq, want)
            out.scheduled.append(ScheduledSeq(seq, want, True))
            seq.status = Status.RUNNING
            self.running.append(seq)
            budget -= want
            slots -= 1

        return out

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        """Drop this sequence's KV and put it back at the front of the queue.

        Recompute rather than swap: on the way back in the prefix cache usually
        still holds most of its blocks, so the redo is cheaper than it looks.
        """
        self.blocks.free(seq)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)
        out.preempted.append(seq.request_id)
        self.total_preemptions += 1

    def _retire(self, seq: Sequence, status: Status, reason: str) -> None:
        self.blocks.free(seq)
        seq.finish(status, reason)
        self.finished.append(seq)
        self.pending_reports.append(seq)

    def finish(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self.blocks.free(seq)
        self.finished.append(seq)

    def stats(self) -> dict:
        d = self.blocks.stats()
        d.update(
            running=len(self.running),
            waiting=len(self.waiting),
            preemptions=self.total_preemptions,
        )
        return d
