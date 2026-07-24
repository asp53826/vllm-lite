from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from .config import EngineConfig
from .runner import NumpyRunner
from .sampling import sample
from .scheduler import Scheduler
from .sequence import RequestOutput, SamplingParams, Sequence, Status
from .speculative import NgramProposer, SpecStats, verify
from .tokenizer import get_tokenizer


@dataclass
class _Detok:
    text: str = ""
    consumed: int = 0


class LLMEngine:
    """step() runs exactly one forward pass and returns whatever came out.

    Callers drive the loop. The API server does it from an asyncio task, the
    benchmark does it in a plain while loop.
    """

    def __init__(
        self,
        cfg: EngineConfig | None = None,
        tokenizer: str | None = None,
        speculative_tokens: int = 0,
    ):
        self.cfg = cfg or EngineConfig()
        self.tokenizer = get_tokenizer(tokenizer)
        self.spec_k = speculative_tokens
        self.proposer = NgramProposer(speculative_tokens) if speculative_tokens else None
        self.spec = SpecStats()

        if self.cfg.model.vocab_size < self.tokenizer.vocab_size:
            self.cfg.model.vocab_size = self.tokenizer.vocab_size

        self.runner = NumpyRunner(self.cfg.model, self.cfg.cache)

        self.scheduler = Scheduler(self.cfg.scheduler, self.cfg.cache)
        self.seqs: dict[str, Sequence] = {}
        self.detok: dict[str, _Detok] = {}
        self.rngs: dict[str, np.random.Generator] = {}
        self._ids = itertools.count()

        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_steps = 0
        self.busy_time = 0.0

    def next_request_id(self) -> str:
        return f"req-{next(self._ids)}"

    def add_request(
        self,
        prompt: str | None = None,
        params: SamplingParams | None = None,
        request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> str:
        rid = request_id or self.next_request_id()
        params = params or SamplingParams()
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("need prompt or prompt_token_ids")
            prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("empty prompt")

        seq = Sequence(rid, list(prompt_token_ids), params)
        self.seqs[rid] = seq
        self.detok[rid] = _Detok()
        self.rngs[rid] = np.random.default_rng(params.seed)
        self.scheduler.add(seq)
        self.total_prompt_tokens += len(prompt_token_ids)
        return rid

    def abort(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id)

    def release(self, request_id: str) -> None:
        """Drop per request bookkeeping once nobody needs the text anymore."""
        self.seqs.pop(request_id, None)
        self.detok.pop(request_id, None)
        self.rngs.pop(request_id, None)

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        dropped = [self._retired_output(s) for s in self.scheduler.drain_reports()]
        if sched.is_empty:
            return dropped

        t0 = time.perf_counter()
        logits, rows = self.runner.execute(sched)
        self.busy_time += time.perf_counter() - t0
        self.total_steps += 1

        results: list[RequestOutput] = list(dropped)
        eos = self.tokenizer.eos_id

        for i, item in enumerate(sched.scheduled):
            seq = item.seq
            row0, count = rows[i]
            drafts = list(seq.draft_token_ids)

            if drafts:
                new_tokens, accepted = verify(logits[row0 : row0 + count], drafts)
                seq.draft_token_ids.clear()
                self.spec.proposed += len(drafts)
                self.spec.accepted += accepted
                # KV only survives for the tokens that held up, the rest of the
                # slots get overwritten on the next pass
                keep = accepted + 1
            else:
                new_tokens, keep = None, item.num_tokens

            self.scheduler.blocks.commit(seq, keep)
            seq.num_computed += keep

            # only sample once every token in the sequence has KV in the cache.
            # chunked prefill and post-preemption recompute both land here
            # partway through and have to wait for the rest.
            if seq.num_computed < seq.num_tokens:
                continue

            if new_tokens is None:
                new_tokens = [
                    sample(
                        logits[row0 + count - 1],
                        seq.params,
                        seq.token_ids,
                        self.rngs[seq.request_id],
                        len(seq.output_token_ids),
                        eos,
                    )
                ]

            emitted = []
            for tok in new_tokens:
                seq.append_token(tok)
                self.total_output_tokens += 1
                emitted.append(tok)
                self._check_stop(seq, tok, eos)
                if seq.status.finished:
                    break

            delta = self._advance_detok(seq)
            if seq.status.finished:
                self.scheduler.finish(seq)

            results.append(
                RequestOutput(
                    request_id=seq.request_id,
                    token_ids=list(seq.output_token_ids),
                    new_token_ids=emitted,
                    text_delta=delta,
                    finished=seq.status.finished,
                    finish_reason=self._reason(seq),
                    num_prompt_tokens=seq.num_prompt_tokens,
                    num_output_tokens=len(seq.output_token_ids),
                    cached_prefix_tokens=seq.cached_prefix_tokens,
                )
            )

        if self.proposer is not None:
            self.spec.rounds += 1
            self._make_proposals()

        return results

    def _make_proposals(self) -> None:
        """Guess ahead for anything that's past its prompt and just decoding."""
        budget = self.cfg.scheduler.max_batched_tokens
        cost = sum(1 for s in self.scheduler.running)
        for seq in self.scheduler.running:
            if seq.num_uncomputed != 1 or seq.status.finished:
                continue
            room = min(self.spec_k, max(0, budget - cost))
            if room <= 0:
                break
            guess = self.proposer.propose(seq, room)
            if guess:
                seq.draft_token_ids = list(guess)
                cost += len(guess)
                self.spec.rounds_with_proposal += 1

    def _retired_output(self, seq: Sequence) -> RequestOutput:
        return RequestOutput(
            request_id=seq.request_id,
            token_ids=list(seq.output_token_ids),
            new_token_ids=[],
            text_delta="",
            finished=True,
            finish_reason=seq.stop_reason or self._reason(seq),
            num_prompt_tokens=seq.num_prompt_tokens,
            num_output_tokens=len(seq.output_token_ids),
            cached_prefix_tokens=seq.cached_prefix_tokens,
        )

    def _advance_detok(self, seq: Sequence) -> str:
        st = self.detok[seq.request_id]
        delta, st.consumed = self.tokenizer.decode_stream(seq.output_token_ids, st.consumed)
        st.text += delta
        return delta

    def _check_stop(self, seq: Sequence, tok: int, eos: int | None) -> None:
        p = seq.params
        if not p.ignore_eos and (tok == eos or tok in p.stop_token_ids):
            seq.finish(Status.FINISHED_STOP, "stop")
            return
        if len(seq.output_token_ids) >= p.max_tokens:
            seq.finish(Status.FINISHED_LENGTH, "length")
            return
        if p.stop:
            text = self.detok[seq.request_id].text
            for s in p.stop:
                if s and s in text:
                    self.detok[seq.request_id].text = text[: text.index(s)]
                    seq.finish(Status.FINISHED_STOP, "stop")
                    return
        if seq.num_tokens >= self.cfg.model.max_seq_len:
            seq.finish(Status.FINISHED_LENGTH, "length")

    @staticmethod
    def _reason(seq: Sequence) -> str | None:
        if seq.status is Status.FINISHED_STOP:
            return "stop"
        if seq.status is Status.FINISHED_LENGTH:
            return "length"
        if seq.status is Status.ABORTED:
            return "abort"
        return None

    def get_text(self, request_id: str) -> str:
        return self.detok[request_id].text

    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[str]:
        """Offline path: submit everything, drain, return finished text in order."""
        ids = [self.add_request(p, params) for p in prompts]
        while self.has_work:
            self.step()
        return [self.detok[i].text for i in ids]

    def stats(self) -> dict:
        d = self.scheduler.stats()
        d.update(
            steps=self.total_steps,
            prompt_tokens=self.total_prompt_tokens,
            output_tokens=self.total_output_tokens,
            busy_seconds=round(self.busy_time, 4),
            output_tokens_per_second=(
                self.total_output_tokens / self.busy_time if self.busy_time else 0.0
            ),
        )
        if self.proposer is not None:
            d.update(self.spec.as_dict())
        return d
