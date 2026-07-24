"""Sits between the scheduler and the raw block allocator.

Scheduler asks "can this sequence grow?", this decides yes/no and hands back
the physical slots plus any copy-on-write work the runner has to do before the
forward pass.
"""

from __future__ import annotations

from .block import BlockAllocator, BlockTable, hash_block
from .config import CacheConfig
from .sequence import Sequence


class BlockManager:
    def __init__(self, cfg: CacheConfig):
        self.block_size = cfg.block_size
        self.allocator = BlockAllocator(cfg.num_blocks, cfg.block_size, cfg.enable_prefix_caching)
        self.prefix_caching = cfg.enable_prefix_caching
        self.peak_blocks = 0
        self.live_tokens = 0
        self.peak_live_tokens = 0

    def _blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence, num_new_tokens: int, watermark_blocks: int = 0) -> bool:
        have = self._blocks_needed(seq.table.num_filled)
        want = self._blocks_needed(seq.table.num_filled + num_new_tokens)
        return self.allocator.num_free >= (want - have) + watermark_blocks

    def reuse_prefix(self, seq: Sequence) -> int:
        """Grab any cached blocks that match this prompt's prefix.

        Only whole blocks can be reused since a partial block's contents are
        still in flux. Returns how many prompt tokens we skipped.
        """
        if not self.prefix_caching or seq.table.block_ids:
            return 0

        toks = seq.prompt_token_ids
        prev_hash = None
        reused = 0
        # leave at least one token to actually run through the model, otherwise
        # there are no logits to sample from
        max_full = (len(toks) - 1) // self.block_size
        for i in range(max_full):
            chunk = tuple(toks[i * self.block_size : (i + 1) * self.block_size])
            h = hash_block(prev_hash, chunk)
            bid = self.allocator.cached.get(h)
            if bid is None:
                break
            self.allocator.cached.move_to_end(h)
            self.allocator.blocks[bid].ref_count += 1
            self.allocator.hits += 1
            seq.table.block_ids.append(bid)
            prev_hash = h
            reused += self.block_size

        seq.table.num_filled = reused
        seq.num_computed = reused
        seq.cached_prefix_tokens = reused
        return reused

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> list[tuple[int, int]]:
        """Make room for num_new_tokens and return the copy-on-write pairs.

        A CoW pair is (src_block, dst_block): the runner has to memcpy before
        writing, because someone else is still reading the original.
        """
        cows: list[tuple[int, int]] = []
        table = seq.table

        # if the block we're about to write into is shared, fork it first.
        # index by position rather than taking the last block, a sequence can
        # hold blocks past num_filled after speculative tokens get rejected.
        if table.num_filled % self.block_size != 0:
            idx = table.num_filled // self.block_size
            if idx < len(table.block_ids):
                tail = table.block_ids[idx]
                if self.allocator.is_shared(tail):
                    new_id, _ = self.allocator.allocate()
                    cows.append((tail, new_id))
                    self.allocator.free(tail)
                    table.block_ids[idx] = new_id

        target = table.num_filled + num_new_tokens
        while len(table.block_ids) * self.block_size < target:
            bid, _ = self.allocator.allocate()
            table.block_ids.append(bid)

        self.peak_blocks = max(self.peak_blocks, self.allocator.num_blocks - self.allocator.num_free)
        return cows

    def slot_mapping(self, seq: Sequence, start: int, count: int) -> list[int]:
        return [seq.table.slot(self.block_size, start + i) for i in range(count)]

    def commit(self, seq: Sequence, num_written: int) -> None:
        """Called after the forward pass actually wrote the KV."""
        table = seq.table
        before = table.num_filled
        table.num_filled += num_written
        self.live_tokens += num_written
        self.peak_live_tokens = max(self.peak_live_tokens, self.live_tokens)
        if not self.prefix_caching:
            return

        # hash and publish any block that just became full
        toks = seq.token_ids
        first = before // self.block_size
        last = table.num_filled // self.block_size
        prev_hash = self._hash_upto(seq, first)
        for i in range(first, last):
            chunk = tuple(toks[i * self.block_size : (i + 1) * self.block_size])
            if len(chunk) < self.block_size:
                break
            h = hash_block(prev_hash, chunk)
            bid = table.block_ids[i]
            self.allocator.mark_full(bid, h)
            prev_hash = h

    def _hash_upto(self, seq: Sequence, block_idx: int) -> str | None:
        if block_idx == 0:
            return None
        return self.allocator.blocks[seq.table.block_ids[block_idx - 1]].content_hash

    def free(self, seq: Sequence) -> None:
        for bid in seq.table.block_ids:
            self.allocator.free(bid)
        self.live_tokens -= seq.table.num_filled
        seq.table = BlockTable()

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Share the parent's blocks with a child sequence (n>1 sampling)."""
        child.table = BlockTable(list(parent.table.block_ids), parent.table.num_filled)
        for bid in child.table.block_ids:
            self.allocator.fork(bid)

    @property
    def usage(self) -> float:
        return 1.0 - self.allocator.num_free / self.allocator.num_blocks

    def stats(self) -> dict:
        d = self.allocator.stats()
        reserved = self.peak_blocks * self.block_size
        d.update(
            peak_blocks=self.peak_blocks,
            peak_live_tokens=self.peak_live_tokens,
            # how much of the memory we actually held onto was real KV rather
            # than padding. this is the number paging exists to move.
            slot_utilization=(self.peak_live_tokens / reserved) if reserved else 0.0,
        )
        return d
