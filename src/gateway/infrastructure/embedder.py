"""sentence-transformers embedder; inference runs in a worker thread."""

import asyncio
from collections.abc import Sequence
from typing import Protocol, cast


class EncoderModel(Protocol):
    """The subset of the SentenceTransformer API used by the gateway."""

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: str, *, normalize_embeddings: bool) -> Sequence[float]: ...


class SentenceTransformerEmbedder:
    """Embeds text without blocking the event loop."""

    def __init__(self, model_name: str, model: EncoderModel | None = None) -> None:
        if model is None:  # pragma: no cover - loads real weights from disk/network
            from sentence_transformers import SentenceTransformer

            model = cast("EncoderModel", SentenceTransformer(model_name))

        dimension = self._read_dimension(model)
        if dimension is None:
            raise ValueError("embedding model does not report its dimension")
        self._model = model
        self._dimension = dimension

    @staticmethod
    def _read_dimension(model: EncoderModel) -> int | None:
        """Prefer the new method name; older sentence-transformers only have the old one."""
        newer = getattr(model, "get_embedding_dimension", None)
        if callable(newer):
            return cast("int | None", newer())
        return model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Size of the produced vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Return a unit-normalised embedding (cosine == dot product)."""
        vector = await asyncio.to_thread(self._model.encode, text, normalize_embeddings=True)
        return [float(value) for value in vector]
