"""Tests for the semantic cache service."""

from collections.abc import AsyncIterator

import pytest_asyncio

from gateway.api.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice
from gateway.infrastructure.qdrant_store import QdrantVectorStore
from gateway.services.cache import SemanticCache

VECTORS = {
    "what is the capital of france": [1.0, 0.0, 0.0],
    "capital of france?": [0.98, 0.2, 0.0],  # cosine ~0.98 with the first prompt
    "weather today": [0.0, 1.0, 0.0],
}


class FakeEmbedder:
    dimension = 3

    async def embed(self, text: str) -> list[float]:
        return list(VECTORS[text])


def make_request(text: str, model: str = "m", temperature: float = 1.0) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        temperature=temperature,
        messages=[ChatMessage(role="user", content=text)],
    )


def make_response(text: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="m", choices=[Choice(message=ChatMessage(role="assistant", content=text))]
    )


@pytest_asyncio.fixture
async def cache() -> AsyncIterator[SemanticCache]:
    store = QdrantVectorStore(url=":memory:", collection="cache-test", dimension=3)
    await store.setup()
    yield SemanticCache(FakeEmbedder(), store, threshold=0.90)
    await store.close()


async def test_miss_on_empty_cache_returns_vector_for_reuse(cache: SemanticCache) -> None:
    lookup = await cache.lookup(make_request("what is the capital of france"))
    assert lookup.hit is None
    assert lookup.vector == [1.0, 0.0, 0.0]


async def test_identical_prompt_hits_after_put(cache: SemanticCache) -> None:
    request = make_request("what is the capital of france")
    first = await cache.lookup(request)
    await cache.put(request, first.vector, make_response("Paris"))

    second = await cache.lookup(request)

    assert second.hit is not None
    assert second.hit.response.choices[0].message.content == "Paris"


async def test_semantically_similar_prompt_hits(cache: SemanticCache) -> None:
    original = make_request("what is the capital of france")
    lookup = await cache.lookup(original)
    await cache.put(original, lookup.vector, make_response("Paris"))

    similar = await cache.lookup(make_request("capital of france?"))

    assert similar.hit is not None
    assert similar.hit.score >= 0.90


async def test_unrelated_prompt_misses(cache: SemanticCache) -> None:
    original = make_request("what is the capital of france")
    lookup = await cache.lookup(original)
    await cache.put(original, lookup.vector, make_response("Paris"))

    assert (await cache.lookup(make_request("weather today"))).hit is None


async def test_different_model_does_not_share_entries(cache: SemanticCache) -> None:
    request = make_request("what is the capital of france", model="model-a")
    lookup = await cache.lookup(request)
    await cache.put(request, lookup.vector, make_response("Paris"))

    other = await cache.lookup(make_request("what is the capital of france", model="model-b"))
    assert other.hit is None


async def test_different_temperature_does_not_share_entries(cache: SemanticCache) -> None:
    request = make_request("what is the capital of france", temperature=0.0)
    lookup = await cache.lookup(request)
    await cache.put(request, lookup.vector, make_response("Paris"))

    other = await cache.lookup(make_request("what is the capital of france", temperature=1.0))
    assert other.hit is None


async def test_hit_gets_fresh_id_but_same_content(cache: SemanticCache) -> None:
    request = make_request("what is the capital of france")
    original = make_response("Paris")
    lookup = await cache.lookup(request)
    await cache.put(request, lookup.vector, original)

    hit = (await cache.lookup(request)).hit

    assert hit is not None
    assert hit.response.id != original.id
    assert hit.response.choices[0].message.content == "Paris"
