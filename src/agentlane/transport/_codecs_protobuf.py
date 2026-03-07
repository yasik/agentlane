"""Protobuf serializer adapter used by transport registry."""

from typing import TypeVar

from google.protobuf.message import Message as ProtobufMessage

from ._serializer import MessageSerializer
from ._types import PROTOBUF_CONTENT_TYPE, ContentType, SchemaId, WireEncoding
from ._utils import coerce_content_type, coerce_schema_id

ProtobufT = TypeVar("ProtobufT", bound=ProtobufMessage)
"""Type variable constrained to one concrete protobuf message class."""


class ProtobufSerializer(MessageSerializer):
    """Serialize and deserialize protobuf message instances."""

    def __init__(
        self,
        *,
        schema_id: SchemaId | str,
        message_type: type[ProtobufT],
        content_type: ContentType | str = PROTOBUF_CONTENT_TYPE,
    ) -> None:
        """Initialize serializer for a specific protobuf message type."""
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)
        self._message_type = message_type

    @property
    def schema_id(self) -> SchemaId:
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        return WireEncoding.PROTOBUF

    @property
    def python_type(self) -> type[object]:
        return self._message_type

    def serialize(self, value: object) -> bytes:
        if not isinstance(value, self._message_type):
            raise TypeError(
                f"Expected value type '{self._message_type.__name__}', "
                f"got '{type(value).__name__}'."
            )
        return value.SerializeToString()

    def deserialize(self, payload: bytes) -> object:
        # Construct a fresh message each call to avoid shared mutable state.
        message = self._message_type()
        message.ParseFromString(payload)
        return message
