"""Token estimation helpers for harness conversation compaction."""

import json
import math
from collections.abc import Sequence
from typing import cast

from agentlane.models import MessageDict

from ._constants import BYTES_PER_TOKEN, NON_TEXT_PART_TOKENS


def estimate_message_tokens(messages: Sequence[MessageDict]) -> int:
    """Estimate message tokens with the byte heuristic.

    It applies a conservative UTF-8 bytes per token approximation for local
    preflight decisions. Runtime shim logic should prefer provider-reported
    usage when available, and callers that need exact accounting can pass a
    custom ``TokenEstimator``.
    """
    return sum(_estimate_content(message.get("content")) for message in messages)


def _estimate_content(content: object) -> int:
    if isinstance(content, str):
        return _estimate_text(content)

    if isinstance(content, list):
        return sum(_estimate_content_part(part) for part in cast(list[object], content))

    return _estimate_text(_stringify(content))


def _estimate_content_part(part: object) -> int:
    if isinstance(part, str):
        return _estimate_text(part)

    if isinstance(part, dict):
        part_dict = cast(dict[str, object], part)
        text = part_dict.get("text")

        if isinstance(text, str) and part_dict.get("type") in {"text", "input_text"}:
            return _estimate_text(text)

    return NON_TEXT_PART_TOKENS


def _estimate_text(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)


def _stringify(value: object) -> str:
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
