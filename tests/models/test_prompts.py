"""Tests for PromptTemplate."""

from typing import Any

import pytest

from agentlane.models import (
    MultiPartPromptTemplate,
    PromptSpec,
    PromptTemplate,
    TextPart,
)


def test_prompt_template_renders_messages(mock_output_schema: Any) -> None:
    """PromptTemplate should render both system and user messages."""
    prompt_template = PromptTemplate[dict[str, object], list[str]](
        system_template="sys: {{ a }}",
        user_template="hi {{ b }}",
        output_schema=mock_output_schema,
    )

    messages = prompt_template.render_messages({"a": 1, "b": "you"})

    assert messages == [
        {"role": "system", "content": "sys: 1"},
        {"role": "user", "content": "hi you"},
    ]


def test_prompt_template_can_render_system_only_messages(
    mock_output_schema: Any,
) -> None:
    """PromptTemplate should allow instruction-only system prompts."""
    prompt_template = PromptTemplate[dict[str, object], list[str]](
        system_template="sys: {{ team }}",
        user_template=None,
        output_schema=mock_output_schema,
    )

    messages = prompt_template.render_messages({"team": "ops"})

    assert messages == [
        {"role": "system", "content": "sys: ops"},
    ]


def test_prompt_template_response_format_delegates(
    mock_output_schema: Any,
) -> None:
    """PromptTemplate should delegate response_format to the output schema."""
    prompt_template = PromptTemplate[dict[str, object], list[str]](
        system_template=None,
        user_template="hello",
        output_schema=mock_output_schema,
    )

    response_format = prompt_template.response_format()

    assert response_format == {"type": "mock", "ok": True}


def test_prompt_template_rejects_empty_template(mock_output_schema: Any) -> None:
    """PromptTemplate should require at least one rendered message."""
    with pytest.raises(
        ValueError,
        match="PromptTemplate requires at least one of `system_template` or `user_template`.",
    ):
        PromptTemplate[dict[str, object], list[str]](
            system_template=None,
            user_template=None,
            output_schema=mock_output_schema,
        )


def test_multipart_prompt_template_can_render_system_only_messages(
    mock_output_schema: Any,
) -> None:
    """MultiPartPromptTemplate should allow system-only prompt content."""
    prompt_template = MultiPartPromptTemplate[dict[str, object], list[str]](
        system_parts=[TextPart("policy for {{ team }}")],
        user_parts=None,
        output_schema=mock_output_schema,
    )

    messages = prompt_template.render_messages({"team": "ops"})

    assert messages == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "policy for ops"}],
        }
    ]


def test_prompt_spec_pairs_template_with_values(mock_output_schema: Any) -> None:
    """PromptSpec should preserve the typed values paired with a template."""
    prompt_template = PromptTemplate[dict[str, object], list[str]](
        system_template="sys: {{ team }}",
        user_template=None,
        output_schema=mock_output_schema,
    )

    prompt_spec = PromptSpec(
        template=prompt_template,
        values={"team": "ops"},
    )

    assert prompt_spec.template is prompt_template
    assert prompt_spec.values == {"team": "ops"}
