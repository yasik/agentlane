"""Tests for ensure_strict_json_schema."""

from typing import Any

import pytest

from agentlane.models import ensure_strict_json_schema


def test_empty_schema_returns_strict_empty_object() -> None:
    """An empty schema should become a strict empty object schema."""
    schema: dict[str, object] = {}
    result = ensure_strict_json_schema(schema)
    assert result == {
        "additionalProperties": False,
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_object_with_additional_properties_true_raises() -> None:
    """Strict mode should reject object schemas with additionalProperties=True."""
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {},
    }

    with pytest.raises(ValueError):
        ensure_strict_json_schema(schema)


def test_default_none_is_stripped_when_present() -> None:
    """`default: None` should be removed during strict-schema conversion."""
    schema = {"type": "string", "default": None}
    result = ensure_strict_json_schema(schema)
    assert "default" not in result


def test_default_absent_does_not_error_and_is_unchanged() -> None:
    """Non-object schemas without defaults should remain unchanged."""
    schema = {"type": "string"}
    result = ensure_strict_json_schema(schema)
    assert result == {"type": "string"}


def test_default_non_none_is_preserved() -> None:
    """Non-None defaults should survive strict-schema conversion."""
    schema = {"type": "string", "default": "abc"}
    result = ensure_strict_json_schema(schema)
    assert result.get("default") == "abc"


def test_nested_property_default_none_is_stripped() -> None:
    """Nested properties should also have default None stripped."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "default": None},
        },
    }
    result = ensure_strict_json_schema(schema)
    assert result["additionalProperties"] is False
    assert set(result["required"]) == {"a"}
    assert "default" not in result["properties"]["a"]
