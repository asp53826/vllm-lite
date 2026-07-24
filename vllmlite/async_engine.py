"""Async wrapper so many HTTP connections can share one engine loop.

There is exactly one background task calling step(). Each in flight request has
a queue that the loop pushes deltas onto, which is what makes streaming work
without giving every request its own model.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .engine import LLMEngine
from .sequence import RequestOutput, SamplingParams


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.queues: dict[str, asyncio.Queue] = {}
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.engine.has_work:
                self._wake.clear()
                await self._wake.wait()
                continue
            # step() is numpy heavy and drops the GIL inside BLAS, so a worker
            # thread keeps the event loop free to accept new connections
            outs = await loop.run_in_executor(None, self.engine.step)
            for out in outs:
                q = self.queues.get(out.request_id)
                if q is not None:
                    q.put_nowait(out)
            if not outs:
                await asyncio.sleep(0)

    async def generate(
        self, prompt: str, params: SamplingParams, request_id: str | None = None
    ) -> AsyncIterator[RequestOutput]:
        q: asyncio.Queue[RequestOutput] = asyncio.Queue()
        async with self._lock:
            rid = self.engine.add_request(prompt, params, request_id)
            self.queues[rid] = q
        self.start()
        self._wake.set()
        try:
            while True:
                out = await q.get()
                yield out
                if out.finished:
                    return
        finally:
            # covers the client-hung-up case too, abort is a no-op if the
            # request already finished on its own
            self.queues.pop(rid, None)
            self.engine.abort(rid)
            self.engine.release(rid)

    async def collect(self, prompt: str, params: SamplingParams, request_id: str | None = None):
        text = []
        last = None
        async for out in self.generate(prompt, params, request_id):
            text.append(out.text_delta)
            last = out
        return "".join(text), last
