"""Tests for HTTP endpoints and request/response schemas."""

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from gateway import __version__
from gateway.api.schemas import ChatCompletionRequest, ChatMessage


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_prompt_text_uses_last_user_message() -> None:
    request = ChatCompletionRequest(
        model="m",
        messages=[
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="answer"),
            ChatMessage(role="user", content="second"),
        ],
    )
    assert request.prompt_text == "second"


def test_prompt_text_falls_back_to_last_message() -> None:
    request = ChatCompletionRequest(
        model="m", messages=[ChatMessage(role="system", content="only system")]
    )
    assert request.prompt_text == "only system"


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_streaming_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
