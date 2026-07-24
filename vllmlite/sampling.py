from __future__ import annotations

import numpy as np

from .sequence import SamplingParams


def _apply_repetition_penalty(logits: np.ndarray, seen: list[int], penalty: float) -> None:
    if penalty == 1.0 or not seen:
        return
    idx = np.unique(np.asarray(seen, dtype=np.int64))
    vals = logits[idx]
    logits[idx] = np.where(vals > 0, vals / penalty, vals * penalty)


def _top_k(logits: np.ndarray, k: int) -> None:
    if k <= 0 or k >= logits.size:
        return
    cut = np.partition(logits, -k)[-k]
    logits[logits < cut] = -np.inf


def _top_p(logits: np.ndarray, p: float) -> None:
    if p >= 1.0:
        return
    order = np.argsort(-logits)
    probs = _softmax(logits[order])
    cum = np.cumsum(probs)
    # keep everything up to and including the token that crosses p
    keep = cum - probs < p
    drop = order[~keep]
    logits[drop] = -np.inf


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(
    logits: np.ndarray,
    params: SamplingParams,
    seen_tokens: list[int],
    rng: np.random.Generator,
    num_generated: int = 0,
    eos_id: int | None = None,
) -> int:
    logits = logits.astype(np.float32, copy=True)

    if eos_id is not None and num_generated < params.min_tokens:
        logits[eos_id] = -np.inf

    _apply_repetition_penalty(logits, seen_tokens, params.repetition_penalty)

    if params.greedy:
        return int(np.argmax(logits))

    logits /= params.temperature
    _top_k(logits, params.top_k)
    _top_p(logits, params.top_p)

    probs = _softmax(logits)
    return int(rng.choice(probs.size, p=probs))
