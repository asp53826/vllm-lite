#!/usr/bin/env python3
"""Throughput and latency comparison.

Same weights and same sampling for every configuration, so the only variables
are the KV memory layout and the scheduling policy.

  python bench/bench.py --requests 64
  python bench/bench.py --requests 128 --json results/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.baseline import BaselineResult, StaticBatchEngine  # noqa: E402
from vllmlite.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig  # noqa: E402
from vllmlite.engine import LLMEngine  # noqa: E402
from vllmlite.sequence import SamplingParams  # noqa: E402
from vllmlite.tokenizer import ByteTokenizer  # noqa: E402


def pct(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else 0.0


def make_workload(n, rng, min_prompt, max_prompt, min_out, max_out, shared_prefix=0):
    """Lognormal-ish lengths, which is much closer to real traffic than fixed
    lengths and is the case static batching handles worst."""
    prefix = list(rng.integers(32, 126, size=shared_prefix)) if shared_prefix else []
    reqs = []
    for i in range(n):
        plen = int(np.clip(rng.lognormal(np.log(min_prompt * 3), 0.6), min_prompt, max_prompt))
        olen = int(np.clip(rng.lognormal(np.log(min_out * 3), 0.7), min_out, max_out))
        body = list(rng.integers(32, 126, size=plen))
        ids = [256] + prefix + body  # BOS then optional shared system prompt
        params = SamplingParams(max_tokens=olen, temperature=0.0, ignore_eos=True)
        reqs.append((f"r{i}", ids, params))
    return reqs


def summarize(name, results, wall, cache_bytes, extra=None):
    out_tokens = sum(r.output_tokens for r in results)
    prompt_tokens = sum(r.prompt_tokens for r in results)
    d = {
        "config": name,
        "requests": len(results),
        "wall_seconds": round(wall, 3),
        "prompt_tokens": prompt_tokens,
        "output_tokens": out_tokens,
        "output_tokens_per_sec": round(out_tokens / wall, 1),
        "total_tokens_per_sec": round((out_tokens + prompt_tokens) / wall, 1),
        "ttft_p50_ms": round(pct([r.ttft for r in results], 50) * 1000, 2),
        "ttft_p99_ms": round(pct([r.ttft for r in results], 99) * 1000, 2),
        "tpot_p50_ms": round(pct([r.tpot for r in results], 50) * 1000, 2),
        "tpot_p99_ms": round(pct([r.tpot for r in results], 99) * 1000, 2),
        "e2e_p50_ms": round(pct([r.e2e for r in results], 50) * 1000, 2),
        "e2e_p99_ms": round(pct([r.e2e for r in results], 99) * 1000, 2),
        "kv_cache_mb": round(cache_bytes / 1e6, 1),
    }
    if extra:
        d.update(extra)
    return d


def run_paged(model_cfg, reqs, max_seqs, num_blocks, block_size, max_batched, prefix_cache):
    cfg = EngineConfig(
        model=model_cfg,
        cache=CacheConfig(
            block_size=block_size, num_blocks=num_blocks, enable_prefix_caching=prefix_cache
        ),
        scheduler=SchedulerConfig(max_seqs=max_seqs, max_batched_tokens=max_batched),
    )
    engine = LLMEngine(cfg)
    t0 = time.perf_counter()
    for rid, ids, params in reqs:
        engine.add_request(params=params, request_id=rid, prompt_token_ids=ids)
    while engine.has_work:
        engine.step()
    wall = time.perf_counter() - t0

    results = []
    for seq in engine.scheduler.finished:
        results.append(
            BaselineResult(
                seq.request_id,
                seq.num_prompt_tokens,
                len(seq.output_token_ids),
                seq.arrival,
                seq.first_token_at,
                seq.finished_at,
            )
        )
    stats = engine.stats()
    extra = {
        "steps": stats["steps"],
        "preemptions": stats["preemptions"],
        "prefix_hit_rate": round(stats["prefix_hit_rate"], 3),
        "slot_utilization": round(stats["slot_utilization"], 3),
    }
    return results, wall, engine.runner.cache_bytes, extra


def run_static(model_cfg, reqs, batch_size, max_len):
    eng = StaticBatchEngine(model_cfg, batch_size, max_len, ByteTokenizer.eos_id)
    t0 = time.perf_counter()
    results = eng.run(reqs)
    wall = time.perf_counter() - t0
    reserved = batch_size * max_len
    return results, wall, eng.cache_bytes, {
        "steps": eng.steps,
        "slot_utilization": round(eng.peak_live_tokens / reserved, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=16)
    ap.add_argument("--max-batched-tokens", type=int, default=1024)
    ap.add_argument("--shared-prefix", type=int, default=0, help="tokens of common system prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model_cfg = ModelConfig(
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        vocab_size=ByteTokenizer.vocab_size,
        max_seq_len=args.max_len,
    )
    reqs = make_workload(args.requests, rng, 16, 256, 16, 192, args.shared_prefix)

    # give the paged engine the same KV budget the static engine reserves, so
    # nobody wins on raw memory
    kv_slots = args.max_seqs * args.max_len
    num_blocks = kv_slots // args.block_size

    configs = [
        ("sequential (batch=1)", lambda: run_static(model_cfg, reqs, 1, args.max_len)),
        ("static batch=4", lambda: run_static(model_cfg, reqs, 4, args.max_len)),
        (f"static batch={args.max_seqs}",
         lambda: run_static(model_cfg, reqs, args.max_seqs, args.max_len)),
        ("paged + continuous",
         lambda: run_paged(model_cfg, reqs, args.max_seqs, num_blocks, args.block_size,
                           args.max_batched_tokens, False)),
        ("paged + continuous + prefix cache",
         lambda: run_paged(model_cfg, reqs, args.max_seqs, num_blocks, args.block_size,
                           args.max_batched_tokens, True)),
    ]

    print(f"{args.requests} requests, "
          f"prompt {min(len(r[1]) for r in reqs)}-{max(len(r[1]) for r in reqs)} tok, "
          f"output {min(r[2].max_tokens for r in reqs)}-{max(r[2].max_tokens for r in reqs)} tok, "
          f"{args.repeat} rep(s), best of")
    print()

    # touch every code path once so BLAS threads and page faults don't land on
    # whichever config happens to run first
    warm = make_workload(4, np.random.default_rng(99), 16, 64, 8, 16)
    run_static(model_cfg, warm, 2, args.max_len)
    run_paged(model_cfg, warm, 4, 128, args.block_size, 256, True)

    rows = []
    for name, fn in configs:
        best = None
        for _ in range(args.repeat):
            r, w, cb, x = fn()
            if best is None or w < best[1]:
                best = (r, w, cb, x)
        rows.append(summarize(name, *best))

    cols = [
        ("config", 34, "s"),
        ("wall_seconds", 8, "f"),
        ("output_tokens_per_sec", 10, "f"),
        ("ttft_p50_ms", 11, "f"),
        ("ttft_p99_ms", 11, "f"),
        ("e2e_p99_ms", 11, "f"),
        ("steps", 7, "d"),
        ("kv_cache_mb", 8, "f"),
        ("slot_utilization", 8, "f"),
    ]
    header = "".join(name[:w].ljust(w + 1) if k == "s" else name[:w].rjust(w + 1)
                     for name, w, k in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for name, w, k in cols:
            v = row.get(name, "")
            line += str(v).ljust(w + 1) if k == "s" else str(v).rjust(w + 1)
        print(line)

    base = rows[0]["output_tokens_per_sec"]
    best = rows[-1]["output_tokens_per_sec"]
    print(f"\nthroughput vs sequential: {best / base:.2f}x")
    print(f"p99 e2e vs static batch={args.max_seqs}: "
          f"{rows[2]['e2e_p99_ms'] / max(rows[-1]['e2e_p99_ms'], 1e-9):.2f}x lower")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "rows": rows}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
