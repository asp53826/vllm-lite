"""Paged attention has to match plain dense attention exactly.

If the block table gather or the GQA head folding is wrong this is where it
shows up, and it would otherwise look like "the model is just bad".
"""

import numpy as np
import pytest

from vllmlite.attention import apply_rope, build_rope, gather_slots, paged_attention


def dense_attention(q, k, v, n_kv_heads, scale):
    """Straightforward reference. q [T,H,D], k/v [ctx,n_kv,D]."""
    q_len, n_heads, d = q.shape
    ctx = k.shape[0]
    n_rep = n_heads // n_kv_heads
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    out = np.zeros_like(q)
    start = ctx - q_len
    for h in range(n_heads):
        s = (q[:, h] @ k[:, h].T) * scale
        for i in range(q_len):
            s[i, start + i + 1 :] = -np.inf
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=-1, keepdims=True)
        out[:, h] = p @ v[:, h]
    return out


@pytest.mark.parametrize("q_len", [1, 5, 17])
@pytest.mark.parametrize("n_heads,n_kv_heads", [(4, 4), (8, 2), (6, 1)])
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_paged_matches_dense(q_len, n_heads, n_kv_heads, block_size):
    rng = np.random.default_rng(0)
    d, ctx = 8, 37
    if q_len > ctx:
        pytest.skip("q_len must fit inside the context")

    n_blocks_needed = (ctx + block_size - 1) // block_size
    total_blocks = n_blocks_needed + 5
    cache_k = rng.standard_normal((total_blocks * block_size, n_kv_heads, d)).astype(np.float32)
    cache_v = rng.standard_normal((total_blocks * block_size, n_kv_heads, d)).astype(np.float32)

    # deliberately scattered and out of order, a real block table is not
    # contiguous and the kernel must not assume it is
    block_ids = list(rng.permutation(total_blocks)[:n_blocks_needed])
    slots = gather_slots(block_ids, block_size, ctx)

    q = rng.standard_normal((q_len, n_heads, d)).astype(np.float32)
    scale = 1.0 / np.sqrt(d)

    got = paged_attention(q, cache_k, cache_v, slots, n_kv_heads, scale)
    want = dense_attention(q, cache_k[slots], cache_v[slots], n_kv_heads, scale)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_gather_slots_follows_the_table():
    slots = gather_slots([3, 0], block_size=4, count=6)
    assert list(slots) == [12, 13, 14, 15, 0, 1]


def test_rope_is_position_dependent():
    cos, sin = build_rope(8, 16)
    x = np.ones((2, 1, 8), dtype=np.float32)
    a = apply_rope(x, np.array([0, 0]), cos, sin)
    b = apply_rope(x, np.array([0, 5]), cos, sin)
    np.testing.assert_allclose(a[0], b[0])
    assert not np.allclose(a[1], b[1])


def test_position_zero_is_identity():
    cos, sin = build_rope(8, 16)
    x = np.random.default_rng(0).standard_normal((1, 2, 8)).astype(np.float32)
    np.testing.assert_allclose(apply_rope(x, np.array([0]), cos, sin), x, atol=1e-6)
