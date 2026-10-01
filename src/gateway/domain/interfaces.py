"""Domain-level contracts. Services depend on these, never on concrete infrastructure."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from gateway.api.schemas import ChatCompletionRequest, ChatCompletionResponse


@dataclass(frozen=True, slots=True)
class CacheHit:
    """A semantic cache lookup result."""

    response: ChatCompletionResponse
    score: float


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a dense vector."""

    @property
    def dimension(self) -> int:
        """Size of the produced vectors."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores responses keyed by prompt embeddings and finds similar ones."""

    async def setup(self) -> None:
        """Create collections/indexes if needed."""
        ...

    async def search(
        self, vector: Sequence[float], namespace: str, threshold: float
    ) -> CacheHit | None:
        """Return the best match in `namespace` with cosine score >= threshold."""
        ...

    async def add(
        self, vector: Sequence[float], namespace: str, response: ChatCompletionResponse
    ) -> None:
        """Store a response under the given prompt vector."""
        ...

    async def close(self) -> None:
        """Release connections."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """An upstream inference engine that can process a whole batch at once."""

    async def generate_batch(
        self, requests: Sequence[ChatCompletionRequest]
    ) -> list[ChatCompletionResponse]:
        """Return one response per request, in the same order."""
        ...