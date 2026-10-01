"""Tests for the simulated inference backend."""

from gateway.api.schemas import ChatCompletionRequest, ChatMessage
from gateway.domain.interfaces import LLMBackend
from gateway.infrastructure.mock_backend import MockLLMBackend


def make_request(text: str, model: str = "test-model") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content=text)])


def test_satisfies_llm_backend_protocol() -> None:
    assert isinstance(MockLLMBackend(), LLMBackend)


async def test_returns_one_response_per_request_in_order() -> None:
    backend = MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)
    responses = await backend.generate_batch([make_request("a"), make_request("b")])

    assert len(responses) == 2
    assert responses[0].choices[0].message.content == "Mock response to: a"
    assert responses[1].choices[0].message.content == "Mock response to: b"


async def test_response_keeps_requested_model_and_usage() -> None:
    backend = MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)
    [response] = await backend.generate_batch([make_request("hello world", model="llama-3")])

    assert response.model == "llama-3"
    assert response.usage.prompt_tokens == 2
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


async def test_empty_batch_is_noop() -> None:
    backend = MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)
    assert await backend.generate_batch([]) == []
    assert backend.call_count == 0


async def test_tracks_batch_sizes() -> None:
    backend = MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)
    await backend.generate_batch([make_request("a")])
    await backend.generate_batch([make_request("b"), make_request("c"), make_request("d")])

    assert backend.call_count == 2
    assert backend.batch_sizes == [1, 3]
