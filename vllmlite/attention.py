"""Paged attention in numpy.

Keys and values never live in one contiguous run per sequence. They're spread
across fixed size blocks and the sequence's block table says where. So the
"kernel" here is a gather through the block table followed by ordinary
attention math. On a GPU the whole thing is one fused kernel, the reason to
write it out is that the memory layout is the interesting part, not the softmax.

Two things worth noting in here:
  - the slot list is built once per sequence per step, not once per layer,
    since the block table doesn't change between layers
  - grouped query attention never materialises the repeated KV heads, the
    query heads get folded into the batch dimension instead
"""

from __future__ import annotations

import numpy as np


def build_rope(head_dim: int, max_pos: int, theta: float = 10000.0):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(max_pos, dtype=np.float32)[:, None]
    ang = pos * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x: np.ndarray, positions: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x is [T, H, D]. Rotates each head's even/odd pairs by its position."""
    c = cos[positions][:, None, :]
    s = sin[positions][:, None, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def gather_slots(block_ids, block_size: int, count: int) -> np.ndarray:
    """Turn a block table into the flat cache slots for positions 0..count-1."""
    bids = np.asarray(block_ids, dtype=np.int64)
    offs = np.arange(block_size, dtype=np.int64)
    return (bids[:, None] * block_size + offs[None, :]).reshape(-1)[:count]


def write_kv(cache_k, cache_v, k, v, slots: np.ndarray) -> None:
    cache_k[slots] = k
    cache_v[slots] = v


def _softmax_(x: np.ndarray) -> np.ndarray:
    x -= x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def paged_attention(
    q: np.ndarray,
    cache_k: np.ndarray,
    cache_v: np.ndarray,
    read_slots: np.ndarray,
    n_kv_heads: int,
    scale: float,
) -> np.ndarray:
    """One sequence. q is [q_len, n_heads, D], returns the same shape.

    read_slots holds every cache slot this sequence owns, in position order,
    including the q_len tokens just written. Query j sits at absolute position
    ctx - q_len + j and may look at everything up to and including itself.
    """
    q_len, n_heads, head_dim = q.shape
    ctx = read_slots.shape[0]
    n_rep = n_heads // n_kv_heads

    k = cache_k[read_slots]  # [ctx, n_kv, D]
    v = cache_v[read_slots]

    # [q_len, H, D] -> [n_kv, n_rep * q_len, D]. head h reads kv head h // n_rep,
    # so folding the rep axis into the row axis lines the batch matmul up
    # without ever copying the kv heads n_rep times.
    qg = q.transpose(1, 0, 2).reshape(n_kv_heads, n_rep * q_len, head_dim)
    scores = np.matmul(qg, k.transpose(1, 2, 0))  # [n_kv, n_rep*q_len, ctx]
    scores *= scale

    if q_len > 1:
        start = ctx - q_len
        qpos = np.arange(start, ctx)[:, None]
        kpos = np.arange(ctx)[None, :]
        mask = kpos > qpos  # [q_len, ctx]
        scores = scores.reshape(n_kv_heads, n_rep, q_len, ctx)
        scores[:, :, mask] = -np.inf
        scores = scores.reshape(n_kv_heads, n_rep * q_len, ctx)

    _softmax_(scores)
    out = np.matmul(scores, v.transpose(1, 0, 2))  # [n_kv, n_rep*q_len, D]
    return out.reshape(n_heads, q_len, head_dim).transpose(1, 0, 2)
