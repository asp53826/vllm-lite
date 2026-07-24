from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

from ..sequence import SamplingParams


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class SamplingMixin(BaseModel):
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False

    def to_params(self) -> SamplingParams:
        stop = self.stop
        if isinstance(stop, str):
            stop = [stop]
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=stop or [],
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            ignore_eos=self.ignore_eos,
        )


class CompletionRequest(SamplingMixin):
    model: str = "vllm-lite"
    prompt: str | list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(SamplingMixin):
    model: str = "vllm-lite"
    messages: list[ChatMessage]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None
    logprobs: None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "vllm-lite"
    choices: list[CompletionChoice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage = Field(default_factory=ChatChoiceMessage)
    finish_reason: str | None = None


class ChatDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatStreamChoice(BaseModel):
    index: int = 0
    delta: ChatDelta = Field(default_factory=ChatDelta)
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "vllm-lite"
    choices: list[ChatChoice] = []
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "vllm-lite"
    choices: list[ChatStreamChoice] = []


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm-lite"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = []
