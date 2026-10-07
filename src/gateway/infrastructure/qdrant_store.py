"""Qdrant-backed vector store used by the semantic cache."""

import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from gateway.api.schemas import ChatCompletionResponse
from gateway.domain.interfaces import CacheHit


class QdrantVectorStore:
    """Stores responses as Qdrant points; the response JSON lives in the payload."""

    def __init__(
        self,
        url: str,
        collection: str,
        dimension: int,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        if client is None:
            client = (
                AsyncQdrantClient(location=":memory:")
                if url == ":memory:"
                else AsyncQdrantClient(url=url)
            )
        self._client = client
        self._collection = collection
        self._dimension = dimension

    async def setup(self) -> None:
        """Create the collection with cosine distance if it does not exist."""
        if await self._client.collection_exists(self._collection):
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )

    async def search(
        self, vector: Sequence[float], namespace: str, threshold: float
    ) -> CacheHit | None:
        """Return the closest entry in `namespace` whose cosine score >= threshold."""
        result = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            query_filter=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
            limit=1,
            score_threshold=threshold,
            with_payload=True,
        )
        if not result.points:
            return None
        point = result.points[0]
        payload = point.payload or {}
        response = ChatCompletionResponse.model_validate_json(payload["response"])
        return CacheHit(response=response, score=point.score)

    async def add(
        self, vector: Sequence[float], namespace: str, response: ChatCompletionResponse
    ) -> None:
        """Store a response under the given prompt vector."""
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=list(vector),
            payload={"namespace": namespace, "response": response.model_dump_json()},
        )
        await self._client.upsert(collection_name=self._collection, points=[point])

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.close()
