"""Tests for the Qdrant vector store, using Qdrant's in-memory mode."""

from collections.abc import AsyncIterator

import pytest_asyncio

from gateway.api.schemas import ChatCompletionResponse, ChatMessage, Choice
from gateway.domain.interfaces import VectorStore
from gateway.infrastructure.qdrant_store import QdrantVectorStore


def make_response(text: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="m", choices=[Choice(message=ChatMessage(role="assistant", content=text))]
    )


@pytest_asyncio.fixture
async def store() -> AsyncIterator[QdrantVectorStore]:
    qdrant = QdrantVectorStore(url=":memory:", collection="test", dimension=3)
    await qdrant.setup()
    yield qdrant
    await qdrant.close()


async def test_satisfies_vector_store_protocol(store: QdrantVectorStore) -> None:
    assert isinstance(store, VectorStore)


async def test_search_on_empty_collection_returns_none(store: QdrantVectorStore) -> None:
    assert await store.search([1.0, 0.0, 0.0], "ns", 0.9) is None


async def test_exact_match_is_a_hit(store: QdrantVectorStore) -> None:
    await store.add([1.0, 0.0, 0.0], "ns", make_response("paris"))

    hit = await store.search([1.0, 0.0, 0.0], "ns", 0.9)

    assert hit is not None
    assert hit.response.choices[0].message.content == "paris"
    assert hit.score > 0.99


async def test_similar_vector_above_threshold_hits(store: QdrantVectorStore) -> None:
    await store.add([1.0, 0.0, 0.0], "ns", make_response("paris"))
    assert await store.search([1.0, 0.1, 0.0], "ns", 0.9) is not None


async def test_dissimilar_vector_below_threshold_misses(store: QdrantVectorStore) -> None:
    await store.add([1.0, 0.0, 0.0], "ns", make_response("paris"))
    assert await store.search([0.0, 1.0, 0.0], "ns", 0.9) is None


async def test_namespaces_are_isolated(store: QdrantVectorStore) -> None:
    await store.add([1.0, 0.0, 0.0], "model-a", make_response("from a"))
    assert await store.search([1.0, 0.0, 0.0], "model-b", 0.9) is None


async def test_setup_is_idempotent(store: QdrantVectorStore) -> None:
    await store.setup()
    await store.add([1.0, 0.0, 0.0], "ns", make_response("still works"))
    assert await store.search([1.0, 0.0, 0.0], "ns", 0.9) is not None
