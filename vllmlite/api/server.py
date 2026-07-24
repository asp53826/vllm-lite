from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..async_engine import AsyncLLMEngine
from ..config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from ..engine import LLMEngine
from .protocol import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatStreamChoice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    UsageInfo,
    _id,
)

MODEL_NAME = os.environ.get("VLLMLITE_MODEL", "vllm-lite")

state: dict = {}


def build_engine(args=None) -> AsyncLLMEngine:
    cfg = EngineConfig(
        model=ModelConfig(
            n_layers=int(os.environ.get("VLLMLITE_LAYERS", 6)),
            n_heads=int(os.environ.get("VLLMLITE_HEADS", 8)),
            n_kv_heads=int(os.environ.get("VLLMLITE_KV_HEADS", 4)),
            d_model=int(os.environ.get("VLLMLITE_DMODEL", 384)),
            d_ff=int(os.environ.get("VLLMLITE_DFF", 1024)),
            max_seq_len=int(os.environ.get("VLLMLITE_MAXLEN", 2048)),
            weights_path=os.environ.get("VLLMLITE_WEIGHTS") or None,
        ),
        cache=CacheConfig(
            block_size=int(os.environ.get("VLLMLITE_BLOCK_SIZE", 16)),
            num_blocks=int(os.environ.get("VLLMLITE_NUM_BLOCKS", 2048)),
            enable_prefix_caching=os.environ.get("VLLMLITE_PREFIX_CACHE", "1") == "1",
        ),
        scheduler=SchedulerConfig(
            max_seqs=int(os.environ.get("VLLMLITE_MAX_SEQS", 32)),
            max_batched_tokens=int(os.environ.get("VLLMLITE_MAX_BATCHED", 2048)),
        ),
        backend=os.environ.get("VLLMLITE_BACKEND", "numpy"),
        hf_model=os.environ.get("VLLMLITE_HF_MODEL") or None,
    )
    tok = os.environ.get("VLLMLITE_TOKENIZER") or None
    return AsyncLLMEngine(LLMEngine(cfg, tokenizer=tok))


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = build_engine()
    state["engine"].start()
    yield
    await state["engine"].stop()


app = FastAPI(title="vllm-lite", version="0.1.0", lifespan=lifespan)


def engine() -> AsyncLLMEngine:
    return state["engine"]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return engine().engine.stats()


@app.get("/v1/models")
async def models():
    return ModelList(data=[ModelCard(id=MODEL_NAME)])


def _prompt_from_messages(msgs) -> str:
    # plain chatml-ish rendering. a real deployment would use the tokenizer's
    # chat template, this keeps the byte tokenizer usable without one.
    parts = [f"<|{m.role}|>\n{m.content}" for m in msgs]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


@app.post("/v1/completions")
async def completions(req: CompletionRequest, raw: Request):
    prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
    if not prompt:
        raise HTTPException(400, "prompt is empty")
    params = req.to_params()
    eng = engine()

    if req.stream:
        cid = _id("cmpl")

        async def gen():
            async for out in eng.generate(prompt, params):
                if await raw.is_disconnected():
                    break
                chunk = CompletionResponse(
                    id=cid,
                    model=req.model,
                    choices=[
                        CompletionChoice(
                            text=out.text_delta,
                            finish_reason=out.finish_reason if out.finished else None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    text, last = await eng.collect(prompt, params)
    return CompletionResponse(
        model=req.model,
        choices=[CompletionChoice(text=text, finish_reason=last.finish_reason if last else None)],
        usage=_usage(last),
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw: Request):
    if not req.messages:
        raise HTTPException(400, "messages is empty")
    prompt = _prompt_from_messages(req.messages)
    params = req.to_params()
    eng = engine()

    if req.stream:
        cid = _id("chatcmpl")

        async def gen():
            first = ChatCompletionChunk(
                id=cid,
                model=req.model,
                choices=[ChatStreamChoice(delta=ChatDelta(role="assistant", content=""))],
            )
            yield f"data: {first.model_dump_json()}\n\n"
            async for out in eng.generate(prompt, params):
                if await raw.is_disconnected():
                    break
                chunk = ChatCompletionChunk(
                    id=cid,
                    model=req.model,
                    choices=[
                        ChatStreamChoice(
                            delta=ChatDelta(content=out.text_delta),
                            finish_reason=out.finish_reason if out.finished else None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    text, last = await eng.collect(prompt, params)
    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatChoice(
                message=ChatChoiceMessage(content=text),
                finish_reason=last.finish_reason if last else None,
            )
        ],
        usage=_usage(last),
    )


def _usage(last) -> UsageInfo:
    if last is None:
        return UsageInfo()
    return UsageInfo(
        prompt_tokens=last.num_prompt_tokens,
        completion_tokens=last.num_output_tokens,
        total_tokens=last.num_prompt_tokens + last.num_output_tokens,
        cached_prompt_tokens=last.cached_prefix_tokens,
    )


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})


def main():
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
