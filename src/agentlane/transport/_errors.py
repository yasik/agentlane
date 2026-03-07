"""Transport serializer error types."""

from ._types import ContentType, SchemaId


class SerializationError(RuntimeError):
    """Base exception for serialization and deserialization failures."""


class SerializerConflictError(SerializationError):
    """Raised when registry registration collides with an existing key."""

    def __init__(self, *, schema_id: SchemaId, content_type: ContentType) -> None:
        super().__init__(
            "Serializer already registered for key "
            f"schema_id='{schema_id.value}', content_type='{content_type.value}'."
        )


class UnknownSerializerError(SerializationError):
    """Raised when no serializer exists for the requested key."""

    def __init__(self, *, schema_id: SchemaId, content_type: ContentType) -> None:
        super().__init__(
            "Unknown serializer for key "
            f"schema_id='{schema_id.value}', content_type='{content_type.value}'."
        )


class SerializerEncodeError(SerializationError):
    """Raised when serializer fails during object-to-bytes conversion."""

    def __init__(
        self,
        *,
        schema_id: SchemaId,
        content_type: ContentType,
        reason: str,
    ) -> None:
        super().__init__(
            "Failed to encode payload for key "
            f"schema_id='{schema_id.value}', content_type='{content_type.value}': "
            f"{reason}"
        )


class SerializerDecodeError(SerializationError):
    """Raised when serializer fails during bytes-to-object conversion."""

    def __init__(
        self,
        *,
        schema_id: SchemaId,
        content_type: ContentType,
        reason: str,
    ) -> None:
        super().__init__(
            "Failed to decode payload for key "
            f"schema_id='{schema_id.value}', content_type='{content_type.value}': "
            f"{reason}"
        )
