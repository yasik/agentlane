from dataclasses import dataclass

from pydantic import BaseModel

from agentlane.models import create_system_message, create_user_message


class _PayloadModel(BaseModel):
    value: str


@dataclass
class _PayloadDataclass:
    value: str


def test_create_system_message_preserves_string_content() -> None:
    assert create_system_message("hello") == {
        "role": "system",
        "content": "hello",
    }


def test_create_user_message_preserves_multipart_content() -> None:
    content = [{"type": "text", "text": "hello"}]

    assert create_user_message(content) == {
        "role": "user",
        "content": content,
    }


def test_create_user_message_serializes_pydantic_models() -> None:
    assert create_user_message(_PayloadModel(value="hello")) == {
        "role": "user",
        "content": '{"value":"hello"}',
    }


def test_create_user_message_serializes_dataclasses() -> None:
    assert create_user_message(_PayloadDataclass(value="hello")) == {
        "role": "user",
        "content": '{"value": "hello"}',
    }
