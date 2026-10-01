"""OpenAI-compatible request/response models validated with Pydantic v2."""

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI /v1/chat/completions request body."""

    # Unknown OpenAI fields are accepted and ignored for client compatibility.
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    # Streaming is not supported by this gateway (responses are batched).
    stream: Literal[False] = False

    @property
    def prompt_text(self) -> str:
        """Text used for semantic matching: the content of the last user message."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return self.messages[-1].content


class Choice(BaseModel):
    """One completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    """Token accounting (approximate for the mock backend)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: Literal["ok"]
    version: str
