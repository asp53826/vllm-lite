"""A small llama-shaped decoder in numpy: RMSNorm, RoPE, GQA, SwiGLU.

Weights are random by default. That sounds useless but it isn't for this
project, the output text is nonsense while the arithmetic, the memory traffic
and the cache behaviour are all the real thing, which is what the scheduler and
the paged cache are being measured on. Point it at a .npz to run real weights.
"""

from __future__ import annotations

import numpy as np

from .attention import apply_rope, build_rope, paged_attention, write_kv
from .config import ModelConfig


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class Layer:
    __slots__ = ("attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w_gate", "w_up", "w_down")

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator):
        d = cfg.d_model
        hd = cfg.head_dim
        s = 1.0 / np.sqrt(d)
        f = lambda *shape: (rng.standard_normal(shape) * s).astype(np.float32)

        self.attn_norm = np.ones(d, dtype=np.float32)
        self.wq = f(d, cfg.n_heads * hd)
        self.wk = f(d, cfg.n_kv_heads * hd)
        self.wv = f(d, cfg.n_kv_heads * hd)
        self.wo = f(cfg.n_heads * hd, d)
        self.ffn_norm = np.ones(d, dtype=np.float32)
        self.w_gate = f(d, cfg.d_ff)
        self.w_up = f(d, cfg.d_ff)
        self.w_down = (rng.standard_normal((cfg.d_ff, d)) / np.sqrt(cfg.d_ff)).astype(np.float32)


class TinyLlama:
    def __init__(self, cfg: ModelConfig, block_size: int = 16):
        self.cfg = cfg
        self.block_size = block_size
        rng = np.random.default_rng(cfg.seed)
        s = 1.0 / np.sqrt(cfg.d_model)
        self.tok_emb = (rng.standard_normal((cfg.vocab_size, cfg.d_model)) * s).astype(np.float32)
        self.layers = [Layer(cfg, rng) for _ in range(cfg.n_layers)]
        self.final_norm = np.ones(cfg.d_model, dtype=np.float32)
        self.lm_head = self.tok_emb.T.copy()  # tied

        self.cos, self.sin = build_rope(cfg.head_dim, cfg.max_seq_len)
        self.scale = 1.0 / np.sqrt(cfg.head_dim)
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        if cfg.weights_path:
            self.load(cfg.weights_path)

    def load(self, path: str) -> None:
        z = np.load(path)
        self.tok_emb = z["tok_emb"].astype(np.float32)
        self.lm_head = z["lm_head"].astype(np.float32) if "lm_head" in z else self.tok_emb.T.copy()
        self.final_norm = z["final_norm"].astype(np.float32)
        for i, layer in enumerate(self.layers):
            for name in Layer.__slots__:
                setattr(layer, name, z[f"layers.{i}.{name}"].astype(np.float32))

    @property
    def num_params(self) -> int:
        n = self.tok_emb.size + self.final_norm.size
        for layer in self.layers:
            n += sum(getattr(layer, name).size for name in Layer.__slots__)
        return n

    def forward(self, token_ids, positions, cache_k, cache_v, seq_meta, last_idx):
        """One batched forward pass.

        token_ids  flat [T] across every scheduled sequence
        positions  flat [T] absolute position of each token
        seq_meta   list of (start, q_len, write_slots, read_slots)
        last_idx   indices into T of each sequence's final token

        Only the final token of each sequence gets pushed through lm_head,
        everything before it is just there to fill the KV cache.
        """
        cfg = self.cfg
        hd = cfg.head_dim
        x = self.tok_emb[token_ids]

        for li, layer in enumerate(self.layers):
            h = rms_norm(x, layer.attn_norm)

            q = (h @ layer.wq).reshape(-1, cfg.n_heads, hd)
            k = (h @ layer.wk).reshape(-1, cfg.n_kv_heads, hd)
            v = (h @ layer.wv).reshape(-1, cfg.n_kv_heads, hd)

            q = apply_rope(q, positions, self.cos, self.sin)
            k = apply_rope(k, positions, self.cos, self.sin)

            ck, cv = cache_k[li], cache_v[li]
            attn = np.empty_like(q)
            for start, q_len, write_slots, read_slots in seq_meta:
                sl = slice(start, start + q_len)
                write_kv(ck, cv, k[sl], v[sl], write_slots)
                attn[sl] = paged_attention(
                    q[sl], ck, cv, read_slots, cfg.n_kv_heads, self.scale
                )

            x = x + attn.reshape(-1, cfg.n_heads * hd) @ layer.wo

            h = rms_norm(x, layer.ffn_norm)
            x = x + (silu(h @ layer.w_gate) * (h @ layer.w_up)) @ layer.w_down

        tail = rms_norm(x[last_idx], self.final_norm)
        return tail @ self.lm_head
