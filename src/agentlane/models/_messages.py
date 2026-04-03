"""Canonical message construction helpers for LLM conversations."""

import json
from dataclasses import asdict, is_dataclass
from typing import cast

from pydantic import BaseModel

from ._interface import MessageDict


def create_system_message(content: object) -> MessageDict:
    """Create one canonical system-role message."""
    return {
        "role": "system",
        "content": _render_user_content(content),
    }


def create_user_message(content: object) -> MessageDict:
    """Create one canonical user-role message."""
    return {
        "role": "user",
        "content": _render_user_content(content),
    }


def _render_user_content(content: object) -> object:
    """Render arbitrary content into canonical message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multipart content is already in model-client message format.
        return cast(list[object], content)
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    if is_dataclass(content) and not isinstance(content, type):
        return json.dumps(asdict(content))
    try:
        return json.dumps(content)
    except TypeError:
        return str(content)
