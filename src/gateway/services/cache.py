"""Semantic cache service: embeds prompts and consults the vector store."""

import time
import uuid
from dataclasses import dataclass

from gateway.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from gateway.domain.interfaces import CacheHit, Embedder, VectorStore


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Lookup result; `vector` is reused by `put` so the prompt is embedded only once."""

    hit: CacheHit | None
    vector: list[float]


class SemanticCache:
    """Similarity-based response cache, isolated per (model, temperature)."""

    def __init__(self, embedder: Embedder, store: VectorStore, threshold: float = 0.90) -> None:
        self._embedder = embedder
        self._store = store
        self._threshold = threshold

    @staticmethod
    def namespace_for(request: ChatCompletionRequest) -> str:
        """Responses are only shared between requests with the same model and temperature."""
        return f"{request.model}|t={request.temperature}"

    async def lookup(self, request: ChatCompletionRequest) -> CacheLookup:
        """Embed the prompt and search for a sufficiently similar cached response."""
        vector = await self._embedder.embed(request.prompt_text)
        hit = await self._store.search(vector, self.namespace_for(request), self._threshold)
        if hit is None:
            return CacheLookup(hit=None, vector=vector)

        fresh = hit.response.model_copy(
            update={"id": f"chatcmpl-{uuid.uuid4().hex}", "created": int(time.time())}
        )
        return CacheLookup(hit=CacheHit(response=fresh, score=hit.score), vector=vector)

    async def put(
        self,
        request: ChatCompletionRequest,
        vector: list[float],
        response: ChatCompletionResponse,
    ) -> None:
        """Cache a freshly generated response under the prompt's vector."""
        await self._store.add(vector, self.namespace_for(request), response)
