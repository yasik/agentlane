"""Utility functions for extracting data from ModelResponse."""

import re
import unicodedata
from typing import Any

from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ._json_repair import parse_json_dict
from ._types import ModelResponse

_ESCAPED_UNICODE_EXPLOSION = re.compile(r"(\\u[0-9a-fA-F]{4})\1{19,}")
"""Regex for escaped Unicode sequences repeated 20+ times."""

_COMBINING_CHAR_THRESHOLD = 20
"""Threshold for combining characters to detect escape sequence explosion."""


class ContentFilterResult(BaseModel):
    """Individual filter category result (hate, self_harm, sexual, violence, etc.)."""

    filtered: bool
    severity: str | None = None


class ContentFilter(BaseModel):
    """Azure content filter entry for prompt or completion."""

    blocked: bool
    source_type: str
    content_filter_results: dict[str, ContentFilterResult]


class ResponseWithContentFilters(BaseModel):
    """Response with content filters."""

    model_config = ConfigDict(extra="allow")

    content_filters: list[ContentFilter] = []


class ReasoningContent:
    """Reasoning content supporting both string and structured formats.

    This class provides backward compatibility for code expecting string reasoning
    while preserving structured ResponseReasoningItem data for consumers who need it.

    Backward compatible: use str(reasoning) or implicit string conversion.
    Structured access: use reasoning.item property for ResponseReasoningItem.

    Example:
        ```python
        reasoning = get_reasoning_content_or_none(response)
        if reasoning:
            # Backward compatible string usage
            print(f"Reasoning: {reasoning}")

            # Structured access (OpenAI Responses API)
            if reasoning.item:
                for summary in reasoning.item.summary:
                    print(f"Summary: {summary.text}")
        ```
    """

    __slots__ = ("_text", "_item")

    def __init__(self, content: str | ResponseReasoningItem) -> None:
        """Initialize ReasoningContent from string or ResponseReasoningItem.

        Args:
            content: Either a string or ResponseReasoningItem.
        """
        if isinstance(content, str):
            self._text = content
            self._item: ResponseReasoningItem | None = None
        else:
            self._item = content
            self._text = self._extract_text(content)

    @staticmethod
    def _extract_text(item: ResponseReasoningItem) -> str:
        """Extract text representation from ResponseReasoningItem as markdown.

        Formats summary and content sections with markdown headings.
        """
        sections: list[str] = []

        # Extract summary text
        if item.summary:
            summary_texts = [
                s.text for s in item.summary if hasattr(s, "text") and s.text
            ]
            if summary_texts:
                sections.append("## Summary\n\n" + "\n\n".join(summary_texts))

        # Extract content text
        if item.content:
            content_texts = [
                c.text for c in item.content if hasattr(c, "text") and c.text
            ]
            if content_texts:
                sections.append("## Reasoning\n\n" + "\n\n".join(content_texts))

        return "\n\n".join(sections)

    def __str__(self) -> str:
        """Return string representation for backward compatibility."""
        return self._text

    def __repr__(self) -> str:
        """Return detailed representation."""
        if self._item:
            return (
                f"ReasoningContent(item={self._item.id!r}, text={self._text[:50]!r}...)"
            )
        return f"ReasoningContent(text={self._text[:50]!r}...)"

    def __bool__(self) -> bool:
        """Return True if there is any reasoning content."""
        return bool(self._text)

    @property
    def text(self) -> str:
        """Get the text representation of the reasoning content."""
        return self._text

    @property
    def item(self) -> ResponseReasoningItem | None:
        """Get the original ResponseReasoningItem if available.

        Returns None if the reasoning was provided as a string.
        """
        return self._item

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Define Pydantic schema for validation and serialization.

        Accepts:
        - ReasoningContent instances (passed through)
        - str (wrapped in ReasoningContent)
        - ResponseReasoningItem (wrapped in ReasoningContent)

        Serializes to string for backward compatibility.
        """
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, info_arg=False, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def _validate(cls, value: Any) -> "ReasoningContent":
        """Validate and convert input to ReasoningContent."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, ResponseReasoningItem):
            return cls(value)
        raise ValueError(
            f"Expected str, ResponseReasoningItem, or ReasoningContent, got {type(value)}"
        )


def get_content_or_none(response: ModelResponse | None) -> str | None:
    """Get the content from the model response.

    Args:
        response: The model response to extract content from.

    Returns:
        The content string, or None if not available.
    """
    if response is None or not response.choices:
        return None
    message = response.choices[0].message
    return message.content if message else None


def get_json_dict_or_none(response: ModelResponse | None) -> dict[str, Any] | None:
    """Parse content as JSON dict.

    Args:
        response: The model response to extract JSON content from.

    Returns:
        The parsed JSON dict, or None if not available or invalid.
    """
    content = get_content_or_none(response)
    if content is None:
        return None
    return parse_json_dict(content)


def get_reasoning_content_or_none(
    response: ModelResponse | None,
) -> ReasoningContent | None:
    """Get reasoning content (for reasoning models like o1, o3, DeepSeek, Claude thinking).

    This extracts reasoning/thinking traces from model responses and wraps them
    in a ReasoningContent object for unified access.

    Args:
        response: The model response to extract reasoning content from.

    Returns:
        ReasoningContent wrapping the reasoning data, or None if not available.

    Example:
        ```python
        reasoning = get_reasoning_content_or_none(response)
        if reasoning:
            # Backward compatible string usage
            print(f"Reasoning: {reasoning}")

            # Structured access (OpenAI Responses API only)
            if reasoning.item:
                for summary in reasoning.item.summary:
                    print(f"Summary: {summary.text}")
        ```
    """
    if response is None:
        return None

    # Check response level first (OpenAI Responses API stores ResponseReasoningItem here)
    reasoning = getattr(response, "reasoning_content", None)
    if reasoning is not None:
        return ReasoningContent(reasoning)

    # Also check __dict__ directly (for dynamically added attributes)
    if hasattr(response, "__dict__") and "reasoning_content" in response.__dict__:
        raw = response.__dict__["reasoning_content"]
        if raw is not None:
            return ReasoningContent(raw)

    # Fall back to message level (some providers store string here)
    if not response.choices:
        return None

    message = response.choices[0].message

    raw = getattr(message, "reasoning_content", None)
    if raw is not None:
        return ReasoningContent(raw)

    return None


def get_search_results_or_none(
    response: ModelResponse | None,
) -> list[dict[str, Any]] | None:
    """Get search results (for Perplexity API responses).

    This field is a provider extension - not part of OpenAI's schema but preserved
    as an extra field during type adaptation.

    Args:
        response: The model response to extract search results from.

    Returns:
        List of search result dicts, or None if not available.
    """
    if response is None:
        return None
    return getattr(response, "search_results", None)


def parse_content_filter_block(response_dict: dict[str, Any]) -> str | None:
    """Check if response was blocked by Azure OpenAI content filters.

    Azure applies content filters to prompts and completions, checking for
    hate, self_harm, sexual, violence, and protected material.

    Args:
        response_dict: The response dictionary (from response.model_dump()).

    Returns:
        A description of the block if present, None otherwise.
        Format: "{source} blocked by content filter: {categories}"
        Example: "completion blocked by content filter: self_harm=high"
    """
    if response_dict.get("content_filters") is None:
        return None

    response = ResponseWithContentFilters.model_validate(response_dict)

    for filter_result in response.content_filters:
        if filter_result.blocked:
            triggered = [
                f"{category}={result.severity or 'filtered'}"
                for category, result in filter_result.content_filter_results.items()
                if result.filtered
            ]

            if triggered:
                categories = ", ".join(triggered)
                return f"{filter_result.source_type} blocked by content filter: {categories}"
            return f"{filter_result.source_type} blocked by content filter: unspecified"

    return None


def has_escape_sequence_explosion(response: ModelResponse) -> bool:
    r"""Detect repeated Unicode escape sequences in model response content.

    Gemini occasionally outputs hundreds of repeated combining diacritical marks
    (e.g. ``\\u0300`` x100s), corrupting structured output and crashing downstream
    parsing. This function catches two variants:

    1. Literal escaped ``\\uXXXX`` sequences repeated 20+ times in a row.
    2. Decoded Unicode combining characters (category ``M``) repeated 20+ times.

    Args:
        response: The model response to inspect.

    Returns:
        True if the response contains an escape-sequence explosion.
    """
    content = get_content_or_none(response)
    if not content:
        return False

    # Check for literal escaped unicode sequences (e.g. \\u0300\\u0300…)
    if _ESCAPED_UNICODE_EXPLOSION.search(content):
        return True

    # Check for decoded combining characters (Unicode category M)
    run_length = 0
    for char in content:
        if unicodedata.category(char).startswith("M"):
            run_length += 1
            if run_length >= _COMBINING_CHAR_THRESHOLD:
                return True
        else:
            run_length = 0

    return False
