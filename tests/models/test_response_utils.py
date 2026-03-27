"""Tests for response utility functions."""

from typing import Any

from agentlane.models import (
    Choice,
    Message,
    ModelResponse,
    has_escape_sequence_explosion,
    parse_content_filter_block,
)


def _make_response(content: str) -> ModelResponse:
    """Build a minimal ModelResponse containing the given assistant content."""
    return ModelResponse(
        id="resp_test",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
    )


def test_has_escape_sequence_explosion_returns_false_for_normal_text() -> None:
    """Normal text should not trigger escape-sequence detection."""
    response = _make_response("Hello, world! This is normal text.")
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_detects_escaped_unicode() -> None:
    r"""Repeated literal \uXXXX sequences should trigger detection."""
    content = "prefix " + "\\u0300" * 25 + " suffix"
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is True


def test_has_escape_sequence_explosion_below_threshold_escaped() -> None:
    r"""Repeated literal \uXXXX sequences below the threshold should not trigger."""
    content = "\\u0300" * 19
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_detects_decoded_combining_chars() -> None:
    """Repeated decoded combining characters should trigger detection."""
    content = "a" + "\u0300" * 25
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is True


def test_has_escape_sequence_explosion_below_threshold_combining() -> None:
    """Repeated decoded combining characters below the threshold should not trigger."""
    content = "a" + "\u0300" * 19
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_returns_false_for_empty_content() -> None:
    """Empty content should not trigger detection."""
    response = _make_response("")
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_returns_false_for_none_content() -> None:
    """Responses with no choices should not trigger detection."""
    response = ModelResponse(
        id="resp_test",
        choices=[],
        created=0,
        model="test",
        object="chat.completion",
    )
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_mixed_escaped_sequences() -> None:
    r"""Different \uXXXX sequences should not be counted as one repeated run."""
    content = ("\\u0300" * 10) + ("\\u0301" * 10)
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is False


def test_has_escape_sequence_explosion_combining_chars_reset_between_base() -> None:
    """Combining-character runs should reset when interrupted by base characters."""
    content = "a" + "\u0300" * 10 + "b" + "\u0301" * 10
    response = _make_response(content)
    assert has_escape_sequence_explosion(response) is False


def test_parse_content_filter_block_returns_none_without_content_filters() -> None:
    """Responses without content_filters should return None."""
    response_dict: dict[str, Any] = {"id": "resp_123", "output": []}

    result = parse_content_filter_block(response_dict)

    assert result is None


def test_parse_content_filter_block_returns_none_with_empty_content_filters() -> None:
    """Empty content_filters should return None."""
    response_dict: dict[str, Any] = {"content_filters": []}

    result = parse_content_filter_block(response_dict)

    assert result is None


def test_parse_content_filter_block_returns_none_with_content_filters_none() -> None:
    """A None content_filters value should return None."""
    response_dict: dict[str, Any] = {"content_filters": None}

    result = parse_content_filter_block(response_dict)

    assert result is None


def test_parse_content_filter_block_returns_none_when_not_blocked() -> None:
    """Unblocked content filters should return None."""
    response_dict: dict[str, Any] = {
        "content_filters": [
            {
                "blocked": False,
                "source_type": "completion",
                "content_filter_results": {
                    "self_harm": {"filtered": True, "severity": "low"}
                },
            }
        ]
    }

    result = parse_content_filter_block(response_dict)

    assert result is None


def test_parse_content_filter_block_returns_categories_when_filtered() -> None:
    """Blocked filters should report the triggered categories and severities."""
    response_dict: dict[str, Any] = {
        "content_filters": [
            {
                "blocked": True,
                "source_type": "completion",
                "content_filter_results": {
                    "self_harm": {"filtered": True, "severity": "high"},
                    "violence": {"filtered": True, "severity": "medium"},
                    "sexual": {"filtered": False, "severity": "low"},
                },
            }
        ]
    }

    result = parse_content_filter_block(response_dict)

    assert (
        result
        == "completion blocked by content filter: self_harm=high, violence=medium"
    )


def test_parse_content_filter_block_returns_unspecified_when_no_triggered_categories() -> (
    None
):
    """Blocked filters with no triggered categories should return unspecified."""
    response_dict: dict[str, Any] = {
        "content_filters": [
            {
                "blocked": True,
                "source_type": "prompt",
                "content_filter_results": {
                    "self_harm": {"filtered": False, "severity": "low"},
                    "violence": {"filtered": False, "severity": "medium"},
                },
            }
        ]
    }

    result = parse_content_filter_block(response_dict)

    assert result == "prompt blocked by content filter: unspecified"


def test_parse_content_filter_block_uses_filtered_fallback_for_missing_severity() -> (
    None
):
    """Missing severities should fall back to the word filtered."""
    response_dict: dict[str, Any] = {
        "content_filters": [
            {
                "blocked": True,
                "source_type": "completion",
                "content_filter_results": {"self_harm": {"filtered": True}},
            }
        ]
    }

    result = parse_content_filter_block(response_dict)

    assert result == "completion blocked by content filter: self_harm=filtered"


def test_parse_content_filter_block_returns_first_blocked_entry() -> None:
    """The first blocked filter entry should win when multiple are present."""
    response_dict: dict[str, Any] = {
        "content_filters": [
            {
                "blocked": True,
                "source_type": "prompt",
                "content_filter_results": {
                    "self_harm": {"filtered": True, "severity": "medium"}
                },
            },
            {
                "blocked": True,
                "source_type": "completion",
                "content_filter_results": {
                    "violence": {"filtered": True, "severity": "high"}
                },
            },
        ]
    }

    result = parse_content_filter_block(response_dict)

    assert result == "prompt blocked by content filter: self_harm=medium"
