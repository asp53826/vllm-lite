from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 8  # set lower than n_heads for grouped-query attn
    d_model: int = 512
    d_ff: int = 2048
    vocab_size: int = 261
    max_seq_len: int = 2048
    weights_path: str | None = None
    seed: int = 1337

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly into n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads for GQA")


@dataclass
class CacheConfig:
    block_size: int = 16
    num_blocks: int = 1024
    enable_prefix_caching: bool = True

    @property
    def max_cached_tokens(self) -> int:
        return self.block_size * self.num_blocks


@dataclass
class SchedulerConfig:
    max_seqs: int = 32
    # token budget for one forward pass. prefill chunks get sliced to fit.
    max_batched_tokens: int = 2048
    max_waiting: int = 2048
    chunked_prefill: bool = True
    # what to do when we run out of blocks mid-decode
    preemption: str = "recompute"
    # stop admitting new work once the cache is this full
    watermark: float = 0.01

    def __post_init__(self):
        if self.preemption not in ("recompute", "swap"):
            raise ValueError("preemption must be recompute or swap")


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    backend: str = "numpy"  # numpy | hf
    hf_model: str | None = None
    device: str = "cpu"
