"""Manual check: real sentence-transformers embeddings + in-memory Qdrant."""

import asyncio

from gateway.api.schemas import ChatCompletionRequest, ChatMessage
from gateway.infrastructure.embedder import SentenceTransformerEmbedder
from gateway.infrastructure.mock_backend import MockLLMBackend
from gateway.infrastructure.qdrant_store import QdrantVectorStore
from gateway.services.cache import SemanticCache

THRESHOLD = 0.90
ORIGINAL = "What is the capital of France?"
QUERIES = [
    "What is the capital of France?",
    "capital of france?",
    "Which city is France's capital?",
    "Tell me the capital city of France",
    "How do I bake sourdough bread?",
]


def make_request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="demo", messages=[ChatMessage(role="user", content=text)])


async def main() -> None:
    print("Loading embedding model (first run downloads ~90 MB)...")
    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    store = QdrantVectorStore(url=":memory:", collection="check", dimension=embedder.dimension)
    await store.setup()
    cache = SemanticCache(embedder, store, threshold=THRESHOLD)
    backend = MockLLMBackend(base_latency_ms=0, per_item_latency_ms=0)

    original = make_request(ORIGINAL)
    lookup = await cache.lookup(original)
    print(f"\nembedding dimension: {embedder.dimension}")
    print(f"first lookup on empty cache: {'HIT' if lookup.hit else 'MISS'} (expected MISS)")

    [response] = await backend.generate_batch([original])
    await cache.put(original, lookup.vector, response)

    namespace = cache.namespace_for(original)
    print(f"\nSimilarity to cached prompt {ORIGINAL!r} (threshold {THRESHOLD}):")
    for text in QUERIES:
        vector = await embedder.embed(text)
        best = await store.search(vector, namespace, 0.0)
        score = best.score if best else 0.0
        verdict = "HIT " if score >= THRESHOLD else "MISS"
        print(f"  {verdict} {score:.3f}  {text}")

    other_model = make_request(ORIGINAL).model_copy(update={"model": "other-model"})
    isolated = await cache.lookup(other_model)
    print(f"\nsame prompt, different model: {'HIT' if isolated.hit else 'MISS'} (expected MISS)")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
