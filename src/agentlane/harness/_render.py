"""Internal request rendering shared by the runner and harness helpers."""

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel

from agentlane.models import (
    MessageDict,
    ModelBehaviorError,
    ModelResponse,
    PromptSpec,
    RunErrorDetails,
)

from ._run import RunHistoryItem, RunInstructions


def render_request_messages(
    instructions: RunInstructions,
    history: list[RunHistoryItem] | tuple[RunHistoryItem, ...],
) -> list[MessageDict]:
    """Build the full model request from one persisted run state slice."""
    messages: list[MessageDict] = []

    messages.extend(_instruction_messages(instructions))
    for item in history:
        messages.extend(_history_item_to_messages(item))

    return messages


def _instruction_messages(instructions: RunInstructions) -> list[MessageDict]:
    """Render system instructions into canonical model messages."""
    if instructions is None:
        return []
    if isinstance(instructions, PromptSpec):
        return _prompt_messages("system", instructions)
    return [_normalize_message({"role": "system", "content": instructions})]


def _history_item_to_messages(item: RunHistoryItem) -> list[MessageDict]:
    """Render one persisted history item into canonical model messages."""
    if isinstance(item, ModelResponse):
        return [_assistant_message_from_response(item)]
    message = _as_message_dict(item)
    if message is not None:
        return [_normalize_message(message)]
    return _user_item_to_messages(item)


def _user_item_to_messages(item: RunHistoryItem) -> list[MessageDict]:
    """Render one user-side continuation item into canonical model messages."""
    if isinstance(item, PromptSpec):
        return _prompt_messages("user", item)
    return [_normalize_message({"role": "user", "content": item})]


def _prompt_messages(
    role: Literal["system", "user"],
    prompt_spec: PromptSpec[Any],
) -> list[MessageDict]:
    """Render a ``PromptSpec`` and keep only messages matching ``role``."""
    messages = [
        _normalize_message(message)
        for message in prompt_spec.template.render_messages(prompt_spec.values)
        if message.get("role") == role
    ]
    if messages:
        return messages
    raise ValueError(f"PromptSpec must render at least one {role}-role message.")


def _normalize_message(message: Mapping[str, object]) -> MessageDict:
    """Copy one message dict and normalize its ``content`` field."""
    normalized_message = dict(message)
    if "content" in normalized_message:
        normalized_message["content"] = normalize_message_content(
            normalized_message["content"]
        )
    return normalized_message


def normalize_message_content(content: object) -> object:
    """Normalize arbitrary content into a model-ready value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return cast(list[object], content)
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    if is_dataclass(content) and not isinstance(content, type):
        return json.dumps(asdict(content))
    try:
        return json.dumps(content)
    except TypeError:
        return str(content)


def _as_message_dict(item: object) -> dict[str, object] | None:
    """Return the item when it already looks like one canonical message dict."""
    if not isinstance(item, dict):
        return None

    message = cast(dict[str, object], item)
    role = message.get("role")
    if isinstance(role, str):
        return message
    return None


def _assistant_message_from_response(response: ModelResponse) -> MessageDict:
    """Extract the first assistant message for continuation history."""
    if not response.choices:
        error = ModelBehaviorError(
            "Runner expected the model response to contain at least one choice."
        )
        error.run_data = RunErrorDetails(raw_response=response)
        raise error
    return response.choices[0].message.model_dump(mode="json", exclude_none=True)
