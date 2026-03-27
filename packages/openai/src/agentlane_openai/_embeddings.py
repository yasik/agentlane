"""Async OpenAI embeddings client.

Thin wrapper around ``AsyncOpenAI().embeddings.create()`` for embedding
text inputs.
"""

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from an embeddings API call."""

    embeddings: list[list[float]]
    """One embedding vector per input text, in the same order as the input."""

    model: str
    """The model that produced the embeddings."""

    total_tokens: int
    """Total tokens consumed by the request."""


class EmbeddingsClient:
    """Async client for OpenAI text embeddings.

    Args:
        api_key: OpenAI API key.
        model: Embedding model identifier.
        dimensions: Optional reduced dimensionality for the embeddings.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of texts in a single API call.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            EmbeddingResult with one vector per input, preserving order.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response: CreateEmbeddingResponse = await self._client.embeddings.create(
            **kwargs
        )

        # API may return embeddings out of order — sort by index
        sorted_data = sorted(response.data, key=lambda d: d.index)

        return EmbeddingResult(
            embeddings=[d.embedding for d in sorted_data],
            model=response.model,
            total_tokens=response.usage.total_tokens,
        )
