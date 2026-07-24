"""Speculative decoding.

Decoding is memory bound: one forward pass moves the whole weight matrix
through cache to produce a single token. Verifying k guessed tokens costs
almost exactly the same pass, so if the guesses are any good you get several
tokens for the price of one.

The proposer here is n-gram lookup rather than a draft model. It searches the
context for the last few tokens and copies whatever followed last time. Costs
nothing, needs no second set of weights, and works well on the workloads people
actually run (long documents, code, RAG, anything where the answer quotes the
prompt). A draft model plugs into the same interface.

Acceptance is the greedy rule: keep a guess while it matches what the real
model would have picked, take the model's own token at the first mismatch. The
tokens are identical to what plain decoding produces, which the tests check.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .sequence import Sequence


class Proposer(Protocol):
    def propose(self, seq: Sequence, k: int) -> list[int]: ...


class NgramProposer:
    def __init__(self, k: int = 4, min_n: int = 3, max_n: int = 8):
        self.k = k
        self.min_n = min_n
        self.max_n = max_n

    def propose(self, seq: Sequence, k: int | None = None) -> list[int]:
        k = k or self.k
        toks = seq.prompt_token_ids + seq.output_token_ids
        # longest match first, a longer context match is a better bet
        for n in range(min(self.max_n, len(toks) - 1), self.min_n - 1, -1):
            tail = toks[-n:]
            hit = self._rfind(toks, tail)
            if hit is None:
                continue
            guess = toks[hit + n : hit + n + k]
            if guess:
                return guess
        return []

    @staticmethod
    def _rfind(toks: list[int], pat: list[int]) -> int | None:
        n = len(pat)
        # skip the trailing occurrence, that's the needle itself
        for i in range(len(toks) - n - 1, -1, -1):
            if toks[i : i + n] == pat:
                return i
        return None


class SpecStats:
    def __init__(self):
        self.proposed = 0
        self.accepted = 0
        self.rounds = 0
        self.rounds_with_proposal = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0

    def as_dict(self) -> dict:
        return {
            "spec_proposed": self.proposed,
            "spec_accepted": self.accepted,
            "spec_acceptance_rate": round(self.acceptance_rate, 3),
            "spec_tokens_per_step": round(self.tokens_per_round, 3),
        }


def verify(logits: np.ndarray, drafts: list[int]) -> tuple[list[int], int]:
    """logits has one row per drafted position plus one for the bonus token.

    Returns (tokens_to_keep, num_drafts_accepted). Always emits at least one
    token, so the loop can never stall no matter how bad the guesses are.
    """
    picked = [int(np.argmax(logits[i])) for i in range(len(drafts) + 1)]
    out = []
    for i, d in enumerate(drafts):
        out.append(picked[i])
        if picked[i] != d:
            return out, i  # first mismatch, the model's own token wins
    # every guess held up, so the extra row is a free token
    out.append(picked[len(drafts)])
    return out, len(drafts)
