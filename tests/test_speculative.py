import numpy as np
import pytest

from vllmlite.sequence import SamplingParams, Sequence
from vllmlite.speculative import NgramProposer, verify
from tests.test_engine import PROMPTS, build, reference, run

REPETITIVE = [
    "abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh abcdefgh",
    "one two three four one two three four one two three four one two three four",
]


def seq_with(tokens):
    return Sequence("t", list(tokens), SamplingParams())


def test_ngram_finds_the_earlier_continuation():
    p = NgramProposer(k=3, min_n=3, max_n=8)
    toks = [1, 2, 3, 9, 8, 7, 1, 2, 3]
    assert p.propose(seq_with(toks), 3) == [9, 8, 7]


def test_ngram_returns_nothing_without_a_repeat():
    p = NgramProposer(k=4, min_n=3)
    assert p.propose(seq_with([1, 2, 3, 4, 5, 6, 7, 8]), 4) == []


def test_ngram_prefers_the_longer_match():
    p = NgramProposer(k=1, min_n=2, max_n=6)
    # [5,6] appears twice, once followed by 7 and once (with more context) by 9
    toks = [5, 6, 7, 4, 4, 1, 5, 6, 9, 4, 4, 1, 5, 6]
    assert p.propose(seq_with(toks), 1) == [9]


def _logits_for(picks, vocab=10):
    out = np.full((len(picks), vocab), -5.0, dtype=np.float32)
    for i, t in enumerate(picks):
        out[i, t] = 5.0
    return out


def test_verify_accepts_everything_when_the_guess_was_right():
    drafts = [3, 4, 5]
    toks, accepted = verify(_logits_for([3, 4, 5, 6]), drafts)
    assert accepted == 3
    assert toks == [3, 4, 5, 6]  # three guesses plus the free bonus token


def test_verify_stops_at_the_first_mismatch():
    drafts = [3, 4, 5]
    toks, accepted = verify(_logits_for([3, 9, 5, 6]), drafts)
    assert accepted == 1
    assert toks == [3, 9]  # keeps the model's own token, drops the rest


def test_verify_always_emits_at_least_one_token():
    toks, accepted = verify(_logits_for([9, 1, 1]), [3, 4])
    assert accepted == 0
    assert toks == [9]


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_speculation_does_not_change_the_output(k):
    """The whole promise of speculative decoding. Same tokens, fewer passes."""
    spec = build()
    spec.spec_k = k
    spec.proposer = NgramProposer(k)
    assert run(spec, PROMPTS) == reference()


@pytest.mark.parametrize("k", [2, 4])
def test_speculation_matches_on_repetitive_text(k):
    plain = build()
    fast = build()
    fast.spec_k = k
    fast.proposer = NgramProposer(k)
    assert run(fast, REPETITIVE, max_tokens=40) == run(plain, REPETITIVE, max_tokens=40)


def test_speculation_cuts_the_number_of_forward_passes():
    plain = build()
    run(plain, REPETITIVE, max_tokens=48)

    fast = build()
    fast.spec_k = 6
    fast.proposer = NgramProposer(6)
    run(fast, REPETITIVE, max_tokens=48)

    assert fast.spec.accepted > 0
    assert fast.total_steps < plain.total_steps


def test_speculation_survives_preemption():
    plain = build(block_size=4, num_blocks=24)
    fast = build(block_size=4, num_blocks=24)
    fast.spec_k = 4
    fast.proposer = NgramProposer(4)
    got = run(fast, PROMPTS)
    assert fast.scheduler.total_preemptions > 0
    assert got == run(plain, PROMPTS)


def test_speculation_frees_every_block():
    fast = build()
    fast.spec_k = 4
    fast.proposer = NgramProposer(4)
    run(fast, REPETITIVE, max_tokens=32)
    a = fast.scheduler.blocks.allocator
    assert a.num_free == a.num_blocks


def test_speculation_respects_max_tokens():
    fast = build()
    fast.spec_k = 8
    fast.proposer = NgramProposer(8)
    p = SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True)
    rid = fast.add_request(REPETITIVE[0], p)
    while fast.has_work:
        fast.step()
    # a speculative round can emit several tokens at once, it must not run past
    # the limit just because the batch of guesses was accepted
    assert len(fast.seqs[rid].output_token_ids) == 5


def test_stats_report_acceptance():
    fast = build()
    fast.spec_k = 4
    fast.proposer = NgramProposer(4)
    run(fast, REPETITIVE, max_tokens=32)
    s = fast.stats()
    assert 0.0 < s["spec_acceptance_rate"] <= 1.0
    assert s["spec_tokens_per_step"] > 1.0
