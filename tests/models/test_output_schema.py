"""Tests for OutputSchema."""

from typing import Any

import pytest
from pydantic import BaseModel

from agentlane.models import OutputSchema


class _User(BaseModel):
    """Simple model used for schema validation tests."""

    id: int
    name: str


def test_plain_text_schema_and_name_and_response_format_none() -> None:
    """Plain-text schemas should have no JSON schema or response format."""
    output_schema = OutputSchema(str)

    assert output_schema.is_plain_text() is True
    assert output_schema.name() == "str"
    assert output_schema.response_format() is None

    with pytest.raises(ValueError):
        output_schema.json_schema()


def test_basemodel_schema_validation_strict_success() -> None:
    """Strict validation should succeed for exact BaseModel JSON."""
    output_schema = OutputSchema(_User, strict_json_schema=False)

    result = output_schema.validate_json('{"id": 1, "name": "Ada"}', strict=True)

    assert isinstance(result, _User)
    assert result.id == 1
    assert result.name == "Ada"


def test_basemodel_validation_partial_nested_object() -> None:
    """Partial validation should locate a nested object matching the schema."""
    output_schema = OutputSchema(_User, strict_json_schema=False)

    result = output_schema.validate_json(
        '{"data": {"user": {"id": 2, "name": "Alan"}}}',
        strict=False,
    )

    assert isinstance(result, _User)
    assert result.id == 2
    assert result.name == "Alan"


def test_wrapped_type_list_validation_and_name() -> None:
    """Wrapped non-BaseModel types should validate under the response wrapper."""
    output_schema = OutputSchema(list[str], strict_json_schema=False)

    assert output_schema.name() == "list_str"

    wrapped_json = '{"response": ["a", "b"]}'
    assert output_schema.validate_json(wrapped_json, strict=True) == ["a", "b"]

    result = output_schema.validate_json(
        '{"data": {"response": ["x"]}}',
        strict=False,
    )

    assert result == ["x"]


def test_dict_with_strict_raises_and_non_strict_allows() -> None:
    """Dict schemas should reject strict mode and allow non-strict mode."""
    with pytest.raises(ValueError):
        OutputSchema(dict)

    output_schema = OutputSchema(dict[str, Any], strict_json_schema=False)
    response_format = output_schema.response_format()

    assert isinstance(response_format, dict)
    assert response_format["json_schema"]["strict"] is False


def test_wrapped_list_of_basemodel_validates_and_returns_models() -> None:
    """Wrapped lists of BaseModel items should validate in strict and partial modes."""
    output_schema = OutputSchema(list[_User], strict_json_schema=False)

    wrapped_json = '{"response": [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Alan"}]}'
    users = output_schema.validate_json(wrapped_json, strict=True)
    assert isinstance(users, list)
    assert all(isinstance(user, _User) for user in users)
    assert [(user.id, user.name) for user in users] == [(1, "Ada"), (2, "Alan")]

    nested_json = '{"data": {"payload": {"response": [{"id": 3, "name": "Grace"}]}}}'
    users_nested = output_schema.validate_json(nested_json, strict=False)
    assert isinstance(users_nested, list)
    assert all(isinstance(user, _User) for user in users_nested)
    assert [(user.id, user.name) for user in users_nested] == [(3, "Grace")]
