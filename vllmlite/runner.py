"""Turns a SchedulerOutput into an actual forward pass.

Owns the KV cache tensors. Everything above this line deals in block ids,
everything below deals in flat slot indices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .attention import gather_slots
from .config import CacheConfig, ModelConfig
from .model import TinyLlama
from .scheduler import SchedulerOutput


class ModelRunner(ABC):
    """Swap this out to put a different model behind the same scheduler."""

    @abstractmethod
    def execute(self, out: SchedulerOutput) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Returns (logits, rows).

        rows[i] is (first_row, count) for the i'th scheduled sequence. count is
        1 for ordinary decoding and 1 + num_drafts when speculating.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...


class NumpyRunner(ModelRunner):
    def __init__(self, model_cfg: ModelConfig, cache_cfg: CacheConfig):
        self.model_cfg = model_cfg
        self.block_size = cache_cfg.block_size
        self.model = TinyLlama(model_cfg, cache_cfg.block_size)

        slots = cache_cfg.num_blocks * cache_cfg.block_size
        shape = (model_cfg.n_layers, slots, model_cfg.n_kv_heads, model_cfg.head_dim)
        self.cache_k = np.zeros(shape, dtype=np.float32)
        self.cache_v = np.zeros(shape, dtype=np.float32)

    @property
    def vocab_size(self) -> int:
        return self.model_cfg.vocab_size

    @property
    def cache_bytes(self) -> int:
        return self.cache_k.nbytes + self.cache_v.nbytes

    def _do_cow(self, pairs: list[tuple[int, int]]) -> None:
        bs = self.block_size
        for src, dst in pairs:
            s = slice(src * bs, (src + 1) * bs)
            d = slice(dst * bs, (dst + 1) * bs)
            self.cache_k[:, d] = self.cache_k[:, s]
            self.cache_v[:, d] = self.cache_v[:, s]

    def execute(self, out: SchedulerOutput) -> np.ndarray:
        if out.cow:
            self._do_cow(out.cow)

        token_ids: list[int] = []
        positions: list[int] = []
        seq_meta = []
        want_logits: list[int] = []
        rows: list[tuple[int, int]] = []

        for item in out.scheduled:
            seq = item.seq
            start = item.start_pos
            n = item.num_tokens
            toks = seq.token_ids[start : start + n]

            offset = len(token_ids)
            token_ids += toks
            positions += range(start, start + n)

            # one gather per sequence per step, reused by every layer
            read_slots = gather_slots(seq.table.block_ids, self.block_size, start + n)
            seq_meta.append((offset, n, read_slots[start:], read_slots))

            # a speculating sequence needs logits at every drafted position so
            # it can check the guesses, everyone else only needs the last one
            count = min(1 + len(seq.draft_token_ids), n)
            rows.append((len(want_logits), count))
            want_logits += range(offset + n - count, offset + n)

        logits = self.model.forward(
            np.asarray(token_ids, dtype=np.int64),
            np.asarray(positions, dtype=np.int64),
            self.cache_k,
            self.cache_v,
            seq_meta,
            np.asarray(want_logits, dtype=np.int64),
        )
        return logits, rows
