"""Tests for the imported OpenAI embeddings client."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agentlane_openai import EmbeddingResult, EmbeddingsClient


def _make_embedding_data(index: int, vector: list[float]) -> MagicMock:
    """Create one mock embedding row."""
    item = MagicMock()
    item.index = index
    item.embedding = vector
    return item


def _make_response(
    data: list[MagicMock],
    *,
    model: str = "text-embedding-3-small",
    total_tokens: int = 10,
) -> MagicMock:
    """Create one mock embeddings response."""
    response = MagicMock()
    response.data = data
    response.model = model
    response.usage = MagicMock()
    response.usage.total_tokens = total_tokens
    return response


def test_embeddings_client_sorts_vectors_by_index() -> None:
    """Embeddings should be returned in input order even if the API reorders them."""
    client = EmbeddingsClient(api_key="test-key")
    mock_response = _make_response(
        [
            _make_embedding_data(1, [0.3, 0.4]),
            _make_embedding_data(0, [0.1, 0.2]),
        ]
    )
    openai_client = cast(Any, client)._client
    openai_client.embeddings.create = AsyncMock(return_value=mock_response)

    result = asyncio.run(client.embed(["a", "b"]))

    assert isinstance(result, EmbeddingResult)
    assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_embeddings_client_forwards_dimensions() -> None:
    """Configured embedding dimensions should be forwarded to the SDK call."""
    client = EmbeddingsClient(
        api_key="test-key",
        dimensions=256,
    )
    openai_client = cast(Any, client)._client
    openai_client.embeddings.create = AsyncMock(
        return_value=_make_response([_make_embedding_data(0, [0.1] * 256)]),
    )

    asyncio.run(client.embed(["hello"]))

    openai_client.embeddings.create.assert_awaited_once_with(
        input=["hello"],
        model="text-embedding-3-small",
        dimensions=256,
    )


def test_embeddings_client_rejects_empty_input() -> None:
    """Embedding requests should reject empty input lists."""
    client = EmbeddingsClient(api_key="test-key")

    with pytest.raises(ValueError, match="texts must be a non-empty list"):
        asyncio.run(client.embed([]))
