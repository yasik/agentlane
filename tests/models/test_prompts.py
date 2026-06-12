"""Tests for PromptTemplate."""

from typing import Any

import pytest

from agentlane.models import (
    MultiPartPromptTemplate,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    TextPart,
    render_instruction_text,
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

    prompt_spec = PromptSpec[dict[str, object]](
        template=prompt_template,
        values={"team": "ops"},
    )

    assert prompt_spec.template is prompt_template
    assert prompt_spec.values == {"team": "ops"}


def test_prompt_template_defaults_to_plain_string_output() -> None:
    """Omitting `output_schema` should default to plain-text str output."""
    prompt_template = PromptTemplate[dict[str, object], str](
        system_template="sys: {{ team }}",
    )

    # Plain text output renders no structured response_format.
    assert prompt_template.response_format() is None


def test_prompt_template_explicit_schema_still_renders_response_format(
    mock_output_schema: Any,
) -> None:
    """Existing explicit `output_schema` usage stays unchanged."""
    prompt_template = PromptTemplate[dict[str, object], list[str]](
        system_template="sys: {{ team }}",
        output_schema=mock_output_schema,
    )

    assert prompt_template.response_format() == {"type": "mock", "ok": True}


def test_multipart_prompt_template_defaults_to_plain_string_output() -> None:
    """Omitting `output_schema` on the multipart template defaults to str."""
    prompt_template = MultiPartPromptTemplate[dict[str, object], str](
        system_parts=[TextPart(template="policy for {{ team }}")],
    )

    assert prompt_template.response_format() is None


def test_render_instruction_text_joins_system_messages() -> None:
    """render_instruction_text returns concatenated system-message text."""
    prompt_template = PromptTemplate[dict[str, object], str](
        system_template="You are {{ name }}.",
        user_template="hello {{ name }}",
    )
    spec = PromptSpec[dict[str, object]](
        template=prompt_template,
        values={"name": "Vera"},
    )

    # Only the system message contributes; the user message is filtered out.
    assert render_instruction_text(spec) == "You are Vera."


def test_render_instruction_text_joins_multiple_system_parts() -> None:
    """Multiple system messages join with blank lines."""

    class _TwoSystemTemplate(PromptTemplate[dict[str, object], str]):
        def render_messages(
            self, ctx: dict[str, object] | None = None
        ) -> list[dict[str, Any]]:
            del ctx
            return [
                {"role": "system", "content": "First."},
                {"role": "system", "content": "Second."},
            ]

    spec = PromptSpec[dict[str, object]](
        template=_TwoSystemTemplate(system_template="ignored"),
    )

    assert render_instruction_text(spec) == "First.\n\nSecond."


def test_render_instruction_text_without_system_message_raises() -> None:
    """A spec with no system message is a programming error, not a silent ''."""
    prompt_template = PromptTemplate[dict[str, object], str](
        user_template="just a user message",
    )
    spec = PromptSpec[dict[str, object]](template=prompt_template)

    with pytest.raises(ValueError, match="at least one system-role message"):
        render_instruction_text(spec)


def test_render_instruction_text_non_text_system_content_raises() -> None:
    """Multi-part (non-string) system content cannot render to instruction text."""
    prompt_template = MultiPartPromptTemplate[dict[str, object], str](
        system_parts=[TextPart(template="policy")],
    )
    spec = PromptSpec[dict[str, object]](template=prompt_template)

    with pytest.raises(ValueError, match="plain text"):
        render_instruction_text(spec)


def test_output_schema_str_is_plain_text() -> None:
    """The default schema behind an omitted output_schema is plain text."""
    schema = OutputSchema(str)

    assert schema.is_plain_text() is True
    assert schema.response_format() is None
