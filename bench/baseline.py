"""The thing we are trying to beat.

Naive serving reserves a contiguous KV slot of max_model_len per sequence and
runs a fixed batch until the *longest* member finishes. Both of those are
exactly what paging plus continuous batching fix, so this is the honest
comparison: same weights, same math, different memory layout and scheduling.

A contiguous cache is just paging with one enormous block, so this reuses the
same attention path instead of a second implementation that might quietly
differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vllmlite.config import ModelConfig
from vllmlite.attention import gather_slots
from vllmlite.model import TinyLlama
from vllmlite.sampling import sample
from vllmlite.sequence import SamplingParams


@dataclass
class BaselineResult:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    submitted: float
    first_token: float | None = None
    finished: float | None = None
    output_ids: list[int] = field(default_factory=list)

    @property
    def ttft(self):
        return None if self.first_token is None else self.first_token - self.submitted

    @property
    def e2e(self):
        return None if self.finished is None else self.finished - self.submitted

    @property
    def tpot(self):
        if self.first_token is None or self.finished is None or self.output_tokens < 2:
            return None
        return (self.finished - self.first_token) / (self.output_tokens - 1)


class StaticBatchEngine:
    """batch_size=1 gives you the sequential HuggingFace generate() pattern."""

    def __init__(self, model_cfg: ModelConfig, batch_size: int, max_len: int, eos_id: int | None):
        self.cfg = model_cfg
        self.batch_size = batch_size
        self.max_len = max_len
        self.eos_id = eos_id
        # one block per slot, block_size == max_len. this is where the waste is:
        # every slot reserves room for the longest sequence we might ever see.
        self.model = TinyLlama(model_cfg, block_size=max_len)
        shape = (model_cfg.n_layers, batch_size * max_len, model_cfg.n_kv_heads, model_cfg.head_dim)
        self.cache_k = np.zeros(shape, dtype=np.float32)
        self.cache_v = np.zeros(shape, dtype=np.float32)
        self.busy = 0.0
        self.steps = 0
        self.peak_live_tokens = 0

    @property
    def cache_bytes(self) -> int:
        return self.cache_k.nbytes + self.cache_v.nbytes

    def run(self, requests: list[tuple[str, list[int], SamplingParams]]) -> list[BaselineResult]:
        # every request is submitted up front, same as the paged engine, so
        # time spent queued behind an earlier batch counts against latency
        submitted = time.perf_counter()
        results = []
        for i in range(0, len(requests), self.batch_size):
            results += self._run_batch(requests[i : i + self.batch_size], submitted)
        return results

    def _run_batch(self, batch, now: float) -> list[BaselineResult]:
        state = []
        for slot, (rid, prompt_ids, params) in enumerate(batch):
            state.append(
                {
                    "slot": slot,
                    "ids": list(prompt_ids),
                    "n_prompt": len(prompt_ids),
                    "params": params,
                    "computed": 0,
                    "done": False,
                    "rng": np.random.default_rng(params.seed),
                    "res": BaselineResult(rid, len(prompt_ids), 0, now),
                }
            )

        while any(not s["done"] for s in state):
            active = [s for s in state if not s["done"]]
            token_ids, positions, seq_meta, last_idx = [], [], [], []
            for s in active:
                start = s["computed"]
                n = len(s["ids"]) - start
                toks = s["ids"][start:]
                off = len(token_ids)
                token_ids += toks
                positions += range(start, start + n)
                read_slots = gather_slots([s["slot"]], self.max_len, start + n)
                seq_meta.append((off, n, read_slots[start:], read_slots))
                last_idx.append(off + n - 1)

            t0 = time.perf_counter()
            logits = self.model.forward(
                np.asarray(token_ids, dtype=np.int64),
                np.asarray(positions, dtype=np.int64),
                self.cache_k,
                self.cache_v,
                seq_meta,
                np.asarray(last_idx, dtype=np.int64),
            )
            self.busy += time.perf_counter() - t0
            self.steps += 1
            live = sum(len(s['ids']) for s in state if not s['done'])
            self.peak_live_tokens = max(self.peak_live_tokens, live)

            for row, s in enumerate(active):
                s["computed"] = len(s["ids"])
                p = s["params"]
                tok = sample(logits[row], p, s["ids"], s["rng"], len(s["ids"]) - s["n_prompt"], self.eos_id)
                s["ids"].append(tok)
                res = s["res"]
                res.output_ids.append(tok)
                res.output_tokens += 1
                if res.first_token is None:
                    res.first_token = time.perf_counter()
                hit_eos = not p.ignore_eos and tok == self.eos_id
                if hit_eos or res.output_tokens >= p.max_tokens or len(s["ids"]) >= self.max_len:
                    s["done"] = True
                    res.finished = time.perf_counter()

        # the batch slot stays reserved until every member is done, which is the
        # head of line blocking we're measuring
        return [s["res"] for s in state]
