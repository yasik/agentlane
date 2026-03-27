"""Tests for PromptTemplate."""

from typing import Any

from agentlane.models import PromptTemplate


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
