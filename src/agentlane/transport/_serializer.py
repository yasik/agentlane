"""Serializer protocol contract for transport registry integration."""

from typing import Protocol, runtime_checkable

from ._types import ContentType, SchemaId, WireEncoding


@runtime_checkable
class MessageSerializer(Protocol):
    """Explicit serializer contract keyed by schema id and content type."""

    @property
    def schema_id(self) -> SchemaId:
        """Globally namespaced schema identifier handled by this serializer."""
        ...

    @property
    def content_type(self) -> ContentType:
        """MIME-like content type handled by this serializer."""
        ...

    @property
    def encoding(self) -> WireEncoding:
        """Wire encoding emitted and consumed by this serializer."""
        ...

    @property
    def python_type(self) -> type[object]:
        """Python type accepted during encode and returned during decode."""
        ...

    def serialize(self, value: object) -> bytes:
        """Serialize a typed value into wire bytes."""
        ...

    def deserialize(self, payload: bytes) -> object:
        """Deserialize wire bytes into a typed value."""
        ...
