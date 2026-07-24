from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from .block import BlockTable


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOP = "finished_stop"
    FINISHED_LENGTH = "finished_length"
    ABORTED = "aborted"

    @property
    def finished(self) -> bool:
        return self in (Status.FINISHED_STOP, Status.FINISHED_LENGTH, Status.ABORTED)


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables
    max_tokens: int = 64
    min_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    repetition_penalty: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    status: Status = Status.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    # guessed tokens waiting to be confirmed or thrown away by the real model
    draft_token_ids: list[int] = field(default_factory=list)
    table: BlockTable = field(default_factory=BlockTable)
    # how much of this sequence already has KV in the cache. drives chunked
    # prefill and tells us how many tokens the next forward pass needs.
    num_computed: int = 0
    arrival: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    preempt_count: int = 0
    cached_prefix_tokens: int = 0
    stop_reason: str | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids + self.draft_token_ids

    @property
    def num_tokens(self) -> int:
        return (
            len(self.prompt_token_ids) + len(self.output_token_ids) + len(self.draft_token_ids)
        )

    def trim_drafts(self, keep: int) -> None:
        del self.draft_token_ids[keep:]

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def is_prefill(self) -> bool:
        # a step that pushes more than one token through is prefill work,
        # whether that's the original prompt or a recompute after preemption
        return self.num_uncomputed > 1

    @property
    def num_uncomputed(self) -> int:
        return self.num_tokens - self.num_computed

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def reset_for_recompute(self) -> None:
        """Preemption by recompute: drop the KV, keep the tokens."""
        self.draft_token_ids.clear()
        self.num_computed = 0
        self.cached_prefix_tokens = 0
        self.table = BlockTable()
        self.preempt_count += 1
        self.status = Status.PREEMPTED

    def finish(self, status: Status, reason: str | None = None) -> None:
        self.status = status
        self.stop_reason = reason
        self.finished_at = time.perf_counter()

    # timing helpers used by the benchmark
    @property
    def ttft(self) -> float | None:
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.arrival

    @property
    def e2e(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first one."""
        if self.first_token_at is None or self.finished_at is None:
            return None
        n = len(self.output_token_ids) - 1
        if n <= 0:
            return None
        return (self.finished_at - self.first_token_at) / n


@dataclass
class RequestOutput:
    request_id: str
    token_ids: list[int]
    new_token_ids: list[int]
    text_delta: str
    finished: bool
    finish_reason: str | None = None
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    cached_prefix_tokens: int = 0
