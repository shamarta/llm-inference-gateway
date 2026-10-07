"""Tests for the sentence-transformers embedder wrapper (with a fake model)."""

import pytest

from gateway.domain.interfaces import Embedder
from gateway.infrastructure.embedder import SentenceTransformerEmbedder


class FakeModel:
    def __init__(self, dimension: int | None = 3) -> None:
        self._dimension = dimension
        self.calls: list[tuple[str, bool]] = []

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(self, sentences: str, *, normalize_embeddings: bool) -> list[float]:
        self.calls.append((sentences, normalize_embeddings))
        return [1, 0, 0]  # ints on purpose: the embedder must convert to float


def test_exposes_model_dimension_and_satisfies_protocol() -> None:
    embedder = SentenceTransformerEmbedder("fake", model=FakeModel(dimension=3))
    assert embedder.dimension == 3
    assert isinstance(embedder, Embedder)


async def test_embed_returns_floats_and_requests_normalisation() -> None:
    model = FakeModel()
    embedder = SentenceTransformerEmbedder("fake", model=model)

    vector = await embedder.embed("hello")

    assert vector == [1.0, 0.0, 0.0]
    assert all(isinstance(v, float) for v in vector)
    assert model.calls == [("hello", True)]


def test_missing_dimension_rejected() -> None:
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEmbedder("fake", model=FakeModel(dimension=None))


class NewStyleModel:
    """A model exposing only the newer `get_embedding_dimension` name."""

    def get_embedding_dimension(self) -> int | None:
        return 5

    def encode(self, sentences: str, *, normalize_embeddings: bool) -> list[float]:
        return [0.0] * 5


def test_prefers_new_dimension_method_name() -> None:
    embedder = SentenceTransformerEmbedder("fake", model=NewStyleModel())  # type: ignore[arg-type]
    assert embedder.dimension == 5
