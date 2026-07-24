"""Physical block bookkeeping for the paged KV cache.

The cache is one big tensor chopped into fixed-size blocks. A sequence never
owns contiguous memory, it owns a block table (list of physical block ids) and
the attention kernel follows that table. Same idea as virtual memory paging,
which is where the name comes from.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field


class OutOfBlocks(Exception):
    pass


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    # content hash of every token up to and including this block. only set once
    # the block is full, since a partial block can still change.
    content_hash: str | None = None
    num_tokens: int = 0

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


def hash_block(prev_hash: str | None, token_ids: tuple[int, ...]) -> str:
    h = hashlib.blake2b(digest_size=16)
    if prev_hash:
        h.update(prev_hash.encode())
    h.update(b"|")
    h.update(",".join(map(str, token_ids)).encode())
    return h.hexdigest()


class BlockAllocator:
    """Free list + refcounts + an LRU pool of cached-but-unused blocks.

    Prefix caching works by keeping full blocks around after their last
    sequence goes away. If a new request shares that prefix we hand the same
    physical block back instead of recomputing it.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_ids: list[int] = list(range(num_blocks))
        # hash -> block_id, ordered so the front is the coldest
        self.cached: OrderedDict[str, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def num_free(self) -> int:
        # a cached block still counts as free if nobody currently references it,
        # we can always evict it
        evictable = sum(1 for bid in self.cached.values() if self.blocks[bid].is_free)
        return len(self.free_ids) + evictable

    def _take_free_id(self) -> int:
        if self.free_ids:
            return self.free_ids.pop()
        # nothing truly free, evict the coldest cached block
        for h, bid in self.cached.items():
            if self.blocks[bid].is_free:
                self.cached.pop(h)
                self.blocks[bid].content_hash = None
                return bid
        raise OutOfBlocks("no blocks left")

    def allocate(self, content_hash: str | None = None) -> tuple[int, bool]:
        """Returns (block_id, was_cache_hit)."""
        if self.enable_prefix_caching and content_hash is not None:
            bid = self.cached.get(content_hash)
            if bid is not None:
                self.cached.move_to_end(content_hash)
                self.blocks[bid].ref_count += 1
                self.hits += 1
                return bid, True
            self.misses += 1

        bid = self._take_free_id()
        blk = self.blocks[bid]
        blk.ref_count = 1
        blk.num_tokens = 0
        blk.content_hash = None
        return bid, False

    def fork(self, block_id: int) -> None:
        self.blocks[block_id].ref_count += 1

    def free(self, block_id: int) -> None:
        blk = self.blocks[block_id]
        if blk.ref_count == 0:
            raise ValueError(f"double free on block {block_id}")
        blk.ref_count -= 1
        if blk.ref_count:
            return
        if self.enable_prefix_caching and blk.content_hash is not None:
            # keep it around, someone with the same prefix may want it
            self.cached[blk.content_hash] = block_id
            self.cached.move_to_end(blk.content_hash)
        else:
            self.free_ids.append(block_id)

    def mark_full(self, block_id: int, content_hash: str) -> None:
        """Publish a block the moment it fills up, not when it's freed.

        Waiting until free would mean two requests sent at the same time with
        the same system prompt can never share it, which is the case that
        matters most.
        """
        blk = self.blocks[block_id]
        blk.num_tokens = self.block_size
        if not self.enable_prefix_caching or blk.content_hash == content_hash:
            return
        blk.content_hash = content_hash
        self.cached[content_hash] = block_id
        self.cached.move_to_end(content_hash)

    def is_shared(self, block_id: int) -> bool:
        return self.blocks[block_id].ref_count > 1

    def stats(self) -> dict:
        used = self.num_blocks - self.num_free
        total = self.hits + self.misses
        return {
            "blocks_total": self.num_blocks,
            "blocks_used": used,
            "usage": used / self.num_blocks,
            "cached_blocks": len(self.cached),
            "prefix_hit_rate": self.hits / total if total else 0.0,
        }


@dataclass
class BlockTable:
    """Logical -> physical mapping for one sequence."""

    block_ids: list[int] = field(default_factory=list)
    # tokens actually written into the cache for this sequence
    num_filled: int = 0

    def slot(self, block_size: int, pos: int) -> int:
        """Flat index into the cache tensor for token position `pos`."""
        return self.block_ids[pos // block_size] * block_size + (pos % block_size)
