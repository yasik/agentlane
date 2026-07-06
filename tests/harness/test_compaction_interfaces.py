import importlib
from typing import Any

import pytest

from agentlane.harness.compaction import (
    DEFAULT_KEEP_RECENT_TOKENS,
    DEFAULT_SUMMARY_BRIDGE,
    DEFAULT_SUMMARY_ITEM_TEMPLATE,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TRIGGER_RATIO,
    NON_TEXT_PART_TOKENS,
    SUMMARY_CLOSE_TAG,
    SUMMARY_OPEN_TAG,
    CompactionError,
    CompactionResult,
    CompactionShimConfig,
    Compactor,
    ContextSignal,
    DefaultCompactorConfig,
    estimate_message_tokens,
    is_summary_item,
    render_request_messages,
    render_summary_item,
)
from agentlane.models import MessageDict


def test_context_signal_reports_remaining_tokens_and_used_fraction() -> None:
    signal = ContextSignal(
        estimated_tokens=450,
        reported_tokens=None,
        instructions_tokens=25,
        context_window=1_000,
        trigger_tokens=900,
        source="estimate",
        turn_count=3,
        history_item_count=12,
    )

    assert signal.remaining_tokens == 550
    assert signal.used_fraction == 0.45


def test_compaction_shim_config_resolves_ratio_trigger() -> None:
    config = CompactionShimConfig(context_window=10_000)

    assert config.trigger_ratio == DEFAULT_TRIGGER_RATIO
    assert config.resolved_trigger_tokens() == 9_000


def test_compaction_shim_config_prefers_explicit_trigger_tokens() -> None:
    config = CompactionShimConfig(
        context_window=10_000,
        trigger_ratio=0.5,
        trigger_tokens=6_500,
        on_failure="skip",
        name="primary-compaction",
    )

    assert config.resolved_trigger_tokens() == 6_500
    assert config.on_failure == "skip"
    assert config.name == "primary-compaction"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_window": 0},
        {"context_window": 10_000, "trigger_ratio": 0},
        {"context_window": 10_000, "trigger_ratio": 1.1},
        {"context_window": 10_000, "trigger_tokens": 0},
        {"context_window": 10_000, "trigger_tokens": 10_001},
        {"context_window": 10_000, "on_failure": "ignore"},
        {"context_window": 10_000, "name": ""},
    ],
)
def test_compaction_shim_config_rejects_invalid_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        CompactionShimConfig(**kwargs)


def test_default_compactor_config_uses_public_defaults() -> None:
    config = DefaultCompactorConfig()

    assert config.prompt
    assert config.summary_bridge == DEFAULT_SUMMARY_BRIDGE
    assert config.keep_recent_tokens == DEFAULT_KEEP_RECENT_TOKENS
    assert config.summary_placement == "before_tail"
    assert config.summary_max_tokens == DEFAULT_SUMMARY_MAX_TOKENS


def test_constants_are_defined_in_constants_module() -> None:
    constants = importlib.import_module("agentlane.harness.compaction._constants")

    assert constants.BYTES_PER_TOKEN == 4
    assert constants.SUMMARY_OPEN_TAG == SUMMARY_OPEN_TAG
    assert constants.SUMMARY_CLOSE_TAG == SUMMARY_CLOSE_TAG


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": ""},
        {"summary_bridge": ""},
        {"keep_recent_tokens": 0},
        {"summary_placement": "middle"},
        {"summary_max_tokens": 0},
    ],
)
def test_default_compactor_config_rejects_invalid_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        DefaultCompactorConfig(**kwargs)


def test_estimate_message_tokens_counts_text_and_non_text_parts() -> None:
    messages: list[MessageDict] = [
        {"role": "user", "content": "abcd"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "abcdefgh"},
                {"type": "image_url", "image_url": {"url": "file://image.png"}},
                "raw text",
            ],
        },
    ]

    assert estimate_message_tokens(messages) == 1 + 2 + NON_TEXT_PART_TOKENS + 2


def test_default_prompts_match_checkpoint_language() -> None:
    from agentlane.harness.compaction import DEFAULT_COMPACTION_PROMPT

    assert "CONTEXT CHECKPOINT COMPACTION" in DEFAULT_COMPACTION_PROMPT
    assert "Another language model started to solve this problem" in (
        DEFAULT_SUMMARY_BRIDGE
    )


def test_render_summary_item_marks_user_message_with_tags() -> None:
    item = render_summary_item(
        bridge="Continue from this handoff.",
        summary_text="Earlier turns discussed a refund.",
    )

    assert item["role"] == "user"
    assert item["content"] == (
        f"{SUMMARY_OPEN_TAG}\n"
        "Continue from this handoff.\n\n"
        f"Earlier turns discussed a refund.\n"
        f"{SUMMARY_CLOSE_TAG}"
    )
    assert "agentlane:" not in item["content"]
    assert is_summary_item(item)
    assert not is_summary_item({"role": "assistant", "content": item["content"]})
    assert not is_summary_item(
        {
            "role": "user",
            "content": (
                "Please preserve the literal text <compaction-summary> and "
                "</compaction-summary>."
            ),
        }
    )


def test_public_renderer_uses_runner_message_shape() -> None:
    messages = render_request_messages(
        "System instruction.",
        [
            "hello",
            {"role": "assistant", "content": "reply"},
        ],
    )

    assert messages == [
        {"role": "system", "content": "System instruction."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "reply"},
    ]


def test_summary_item_template_is_jinja2_parameterized() -> None:
    assert "{{ bridge }}" in DEFAULT_SUMMARY_ITEM_TEMPLATE
    assert "{{ summary_text }}" in DEFAULT_SUMMARY_ITEM_TEMPLATE


def test_compactor_protocol_is_runtime_checkable() -> None:
    class InlineCompactor:
        async def compact(self, request: object) -> CompactionResult:
            del request
            raise CompactionError("not implemented")

    assert isinstance(InlineCompactor(), Compactor)


def test_first_branch_does_not_export_runtime_implementations() -> None:
    import agentlane.harness.compaction as compaction

    assert not hasattr(compaction, "CompactionShim")
    assert not hasattr(compaction, "DefaultCompactor")
