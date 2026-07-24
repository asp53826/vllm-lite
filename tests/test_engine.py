"""Invariance tests.

Paging, chunked prefill, prefix caching, preemption and batch composition are
all supposed to be invisible from outside. Greedy decoding gives one right
answer, so any of those knobs changing the tokens means a bug. These caught
more real problems than anything else while building this.
"""

import pytest

from vllmlite.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from vllmlite.engine import LLMEngine
from vllmlite.sequence import SamplingParams

PROMPTS = [
    "the quick brown fox jumps over",
    "a",
    "systems programming is mostly about memory and never about",
    "hello",
]


def build(block_size=16, num_blocks=512, max_seqs=8, max_batched=1024,
          prefix_cache=False, chunked=True, layers=2, d_model=64):
    cfg = EngineConfig(
        model=ModelConfig(
            n_layers=layers, n_heads=4, n_kv_heads=2, d_model=d_model, d_ff=128, max_seq_len=512
        ),
        cache=CacheConfig(
            block_size=block_size, num_blocks=num_blocks, enable_prefix_caching=prefix_cache
        ),
        scheduler=SchedulerConfig(
            max_seqs=max_seqs, max_batched_tokens=max_batched, chunked_prefill=chunked
        ),
    )
    return LLMEngine(cfg)


def run(engine, prompts, max_tokens=20):
    p = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    ids = [engine.add_request(x, p) for x in prompts]
    while engine.has_work:
        engine.step()
    return [engine.seqs[i].output_token_ids for i in ids]


REFERENCE = None


def reference():
    global REFERENCE
    if REFERENCE is None:
        REFERENCE = run(build(), PROMPTS)
    return REFERENCE


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
def test_block_size_does_not_change_output(block_size):
    assert run(build(block_size=block_size), PROMPTS) == reference()


def test_prefix_caching_does_not_change_output():
    assert run(build(prefix_cache=True), PROMPTS) == reference()


@pytest.mark.parametrize("max_batched", [8, 33, 64, 4096])
def test_chunked_prefill_does_not_change_output(max_batched):
    """A prompt sliced across several forward passes must land in the same
    place as one that went through in a single shot."""
    assert run(build(max_batched=max_batched), PROMPTS) == reference()


def test_no_chunked_prefill_matches():
    assert run(build(chunked=False, max_batched=4096), PROMPTS) == reference()


@pytest.mark.parametrize("max_seqs", [1, 2, 3, 8])
def test_batch_composition_does_not_change_output(max_seqs):
    """Who a request happens to share a step with must not leak into it."""
    assert run(build(max_seqs=max_seqs), PROMPTS) == reference()


def test_single_request_matches_batched():
    solo = [run(build(), [p])[0] for p in PROMPTS]
    assert solo == reference()


def test_preemption_preserves_output():
    """Squeeze the cache so the scheduler is forced to evict and recompute."""
    engine = build(block_size=4, num_blocks=24, max_seqs=8, prefix_cache=False)
    got = run(engine, PROMPTS)
    assert engine.scheduler.total_preemptions > 0, "test did not actually preempt"
    assert got == reference()


def test_preemption_with_prefix_cache_preserves_output():
    engine = build(block_size=4, num_blocks=24, max_seqs=8, prefix_cache=True)
    got = run(engine, PROMPTS)
    assert engine.scheduler.total_preemptions > 0
    assert got == reference()


def test_all_blocks_returned_when_idle():
    engine = build(prefix_cache=False)
    run(engine, PROMPTS)
    a = engine.scheduler.blocks.allocator
    assert a.num_free == a.num_blocks
    assert all(b.is_free for b in a.blocks)


def test_prefix_cache_hits_on_repeated_prompt():
    """Second request reuses the first one's blocks instead of prefilling them.

    Sent one after the other on purpose. Two requests admitted in the same
    forward pass can't share, the KV they'd share doesn't exist yet.
    """
    engine = build(block_size=4, prefix_cache=True)
    long = "the same long system prompt that two requests happen to share, word for word"
    run(engine, [long + " one"], max_tokens=4)
    run(engine, [long + " two"], max_tokens=4)
    stats = engine.scheduler.blocks.stats()
    assert stats["prefix_hit_rate"] > 0
    second = engine.seqs["req-1"]
    assert second.cached_prefix_tokens >= len(long) - 8


def test_prefix_cache_only_reuses_a_genuinely_matching_prefix():
    engine = build(block_size=4, prefix_cache=True)
    run(engine, ["aaaaaaaaaaaaaaaaaaaaaaaa"], max_tokens=4)
    run(engine, ["bbbbbbbbbbbbbbbbbbbbbbbb"], max_tokens=4)
    assert engine.seqs["req-1"].cached_prefix_tokens == 0


def test_max_tokens_is_respected():
    engine = build()
    p = SamplingParams(max_tokens=7, temperature=0.0, ignore_eos=True)
    rid = engine.add_request("hello there", p)
    while engine.has_work:
        engine.step()
    assert len(engine.seqs[rid].output_token_ids) == 7
    assert engine.seqs[rid].stop_reason == "length"


def test_stop_string_truncates_text():
    engine = build()
    p = SamplingParams(max_tokens=40, temperature=0.0, ignore_eos=True)
    rid = engine.add_request("hello", p)
    while engine.has_work:
        engine.step()
    full = engine.get_text(rid)
    assert len(full) > 3
    stop_at = full[2:4]

    engine2 = build()
    p2 = SamplingParams(max_tokens=40, temperature=0.0, ignore_eos=True, stop=[stop_at])
    rid2 = engine2.add_request("hello", p2)
    while engine2.has_work:
        engine2.step()
    assert stop_at not in engine2.get_text(rid2)
    assert engine2.seqs[rid2].stop_reason == "stop"


def test_abort_frees_the_blocks():
    engine = build()
    p = SamplingParams(max_tokens=50, temperature=0.0, ignore_eos=True)
    rid = engine.add_request("hello there friend", p)
    engine.step()
    engine.step()
    assert engine.abort(rid)
    assert not engine.has_work
    a = engine.scheduler.blocks.allocator
    assert a.num_free == a.num_blocks


def test_prompt_larger_than_cache_is_rejected_not_hung():
    engine = build(block_size=4, num_blocks=4, max_seqs=2)
    p = SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True)
    rid = engine.add_request("x" * 400, p)
    outs = []
    for _ in range(50):
        outs += engine.step()
        if not engine.has_work:
            break
    assert not engine.has_work, "engine spun forever on an impossible request"
    assert any(o.request_id == rid and o.finished for o in outs)


def test_seeded_sampling_is_reproducible():
    a = build()
    b = build()
    p = SamplingParams(max_tokens=16, temperature=0.9, top_p=0.95, seed=7, ignore_eos=True)
    ra = a.add_request("hello", p)
    rb = b.add_request("hello", p)
    while a.has_work:
        a.step()
    while b.has_work:
        b.step()
    assert a.seqs[ra].output_token_ids == b.seqs[rb].output_token_ids


def test_empty_token_list_rejected():
    engine = build()
    with pytest.raises(ValueError):
        engine.add_request(params=SamplingParams(), prompt_token_ids=[])
