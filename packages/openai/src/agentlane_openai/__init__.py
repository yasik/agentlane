"""OpenAI native client package.

This package provides native OpenAI client wrappers that conform to the
AgentLane core model interfaces. Unlike LiteLLM, these clients use the
OpenAI SDK directly for better control over features like the Responses API.

Clients:
    ResponsesClient: Non-streaming client using OpenAI Responses API
    ResponsesFactory: Factory for creating ResponsesClient instances
    EmbeddingsClient: Async wrapper for OpenAI text embeddings

Types:
    Re-exports from openai.types.responses for Responses API types.
"""

from ._embeddings import EmbeddingResult, EmbeddingsClient
from ._responses_client import ResponsesClient, ResponsesFactory
from ._tool_output_adapter import ResponsesApiOutputAdapter

__all__ = [
    "EmbeddingResult",
    "EmbeddingsClient",
    "ResponsesClient",
    "ResponsesFactory",
    "ResponsesApiOutputAdapter",
]
