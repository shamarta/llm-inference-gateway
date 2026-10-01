"""Simulated vLLM-like backend: latency grows slowly with batch size."""

import asyncio
from collections.abc import Sequence

from gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)


class MockLLMBackend:
    """Fake inference engine.

    Models the key property of GPU batching: one forward pass over N prompts
    costs far less than N separate passes (base latency + small per-item cost).
    """

    def __init__(self, base_latency_ms: int = 150, per_item_latency_ms: int = 10) -> None:
        self._base_latency_ms = base_latency_ms
        self._per_item_latency_ms = per_item_latency_ms
        self.call_count = 0
        self.batch_sizes: list[int] = []

    async def generate_batch(
        self, requests: Sequence[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        """Simulate one batched inference call."""
        if not requests:
            return []

        latency_ms = self._base_latency_ms + self._per_item_latency_ms * len(requests)
        await asyncio.sleep(latency_ms / 1000)

        self.call_count += 1
        self.batch_sizes.append(len(requests))
        return [self._build_response(request) for request in requests]

    @staticmethod
    def _build_response(request: ChatCompletionRequest) -> ChatCompletionResponse:
        prompt = request.prompt_text
        answer = f"Mock response to: {prompt}"
        prompt_tokens = len(prompt.split())
        completion_tokens = len(answer.split())
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=ChatMessage(role="assistant", content=answer))],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )