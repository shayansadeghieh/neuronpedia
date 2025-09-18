from typing import Any, Literal
from pydantic import BaseModel, Field
from enum import Enum
from neuronpedia_inference.config import DEFAULT_MAX_TOKENS


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = True


class SteerMethod(str, Enum):
    SIMPLE_ADDITIVE = "SIMPLE_ADDITIVE"
    ORTHOGONAL_DECOMP = "ORTHOGONAL_DECOMP"


class SteerVector(BaseModel):
    vector: list[float]
    strength: float
    hook: str


class SteerOptions(BaseModel):
    steer_method: SteerMethod = SteerMethod.SIMPLE_ADDITIVE
    normalize_steering: bool = False
    vectors: list[SteerVector]
    steer_special_tokens: bool = True


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool | None = False
    stream_options: StreamOptions | None = None

    # adjustable params
    seed: int | None = None
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    max_tokens: int | None = Field(default=DEFAULT_MAX_TOKENS, ge=1)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)

    # unsupported
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    n: int | None = Field(default=1, ge=1, le=1)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = True
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    # we added this
    steer_options: SteerOptions | None = None


# Response Models
class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant", "system", "user", "tool", "function"]
    content: str | None = None
    # unsupported
    tool_calls: list[dict[str, Any]] | None = None
    refusal: str | None = None
    # we added this. if it exists, we do steering
    tokens: list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: dict[str, Any] | None = None
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | None
    )


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage | None = None

    # unsupported
    system_fingerprint: str | None = None


# Streaming Response Models
class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant", "system", "user", "tool", "function"] | None = None
    content: str | None = None
    # unsupported
    tool_calls: list[dict[str, Any]] | None = None
    refusal: str | None = None
    # we added this
    tokens: list[str] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    logprobs: dict[str, Any] | None = None
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | None
    ) = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: CompletionUsage | None = None
    # unsupported
    system_fingerprint: str | None = None
