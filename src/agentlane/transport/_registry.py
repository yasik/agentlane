"""Serializer registry implementation and default inference helpers."""

from collections.abc import Sequence
from dataclasses import is_dataclass
from threading import RLock
from typing import TypeGuard

from google.protobuf.message import Message as ProtobufMessage
from pydantic import BaseModel

from ._codecs_json import (
    DataclassJsonSerializer,
    JsonValueSerializer,
    PydanticJsonSerializer,
)
from ._codecs_protobuf import ProtobufSerializer
from ._errors import (
    SerializerConflictError,
    SerializerDecodeError,
    SerializerEncodeError,
    UnknownSerializerError,
)
from ._serializer import MessageSerializer
from ._types import (
    JSON_CONTENT_TYPE,
    OCTET_STREAM_CONTENT_TYPE,
    PROTOBUF_CONTENT_TYPE,
    ContentType,
    SchemaId,
)
from ._utils import coerce_content_type, coerce_schema_id
from ._wire_payload import WirePayload

SerializerKey = tuple[SchemaId, ContentType]
"""Normalized registry key `(schema_id, content_type)` used for lookups."""

SerializerTable = dict[SerializerKey, MessageSerializer]
"""Mutable serializer map guarded by the registry lock."""


class SerializerRegistry:
    """Thread-safe serializer registry keyed by `(schema_id, content_type)`.

    Two operating modes:

    1. strict mode (`auto_register_defaults=False`): serializer key must exist,
    2. default mode (`auto_register_defaults=True`): registry can infer and cache
       serializers for common payload types when first used.
    """

    def __init__(self, *, auto_register_defaults: bool = True) -> None:
        """Initialize serializer mappings and runtime behavior flags.

        Args:
            auto_register_defaults: Enables default serializer inference on demand.
        """
        # All mapping mutations/read-modify-writes flow through this lock.
        self._lock = RLock()
        self._serializers: SerializerTable = {}
        self._auto_register_defaults = auto_register_defaults

    def register(
        self,
        serializer: MessageSerializer,
        *,
        replace: bool = False,
    ) -> None:
        """Register a serializer for its `(schema_id, content_type)` key.

        Args:
            serializer: Serializer instance to register.
            replace: Whether to replace an existing serializer at same key.

        Returns:
            None: Always returns after successful registration.

        Raises:
            SerializerConflictError: If key exists and `replace=False`.
        """
        key = (serializer.schema_id, serializer.content_type)
        with self._lock:
            existing = self._serializers.get(key)
            if existing is not None and not replace:
                raise SerializerConflictError(
                    schema_id=serializer.schema_id,
                    content_type=serializer.content_type,
                )
            self._serializers[key] = serializer

    def register_many(
        self,
        serializers: Sequence[MessageSerializer],
        *,
        replace: bool = False,
    ) -> None:
        """Register multiple serializers.

        Args:
            serializers: Sequence of serializers to register.
            replace: Whether to replace existing serializers on key conflict.

        Returns:
            None: Always returns after processing all serializers.
        """
        for serializer in serializers:
            self.register(serializer, replace=replace)

    def register_type(
        self,
        python_type: type[object],
        *,
        schema_id: SchemaId | str | None = None,
        content_type: ContentType | str | None = None,
        replace: bool = False,
    ) -> MessageSerializer:
        """Register one model type using default serializer selection rules.

        Args:
            python_type: Python runtime type to register.
            schema_id: Optional explicit schema id override.
            content_type: Optional explicit content type override.
            replace: Whether to replace existing serializer on conflict.

        Returns:
            MessageSerializer: Registered serializer instance.

        Raises:
            TypeError: If no default serializer exists for provided type.
            SerializerConflictError: If key exists and `replace=False`.
        """
        resolved_schema_id = (
            coerce_schema_id(schema_id)
            if schema_id is not None
            else infer_schema_id_for_type(python_type)
        )
        resolved_content_type = (
            coerce_content_type(content_type)
            if content_type is not None
            else infer_content_type_for_type(python_type)
        )
        serializer = _create_serializer_for_type(
            python_type=python_type,
            schema_id=resolved_schema_id,
            content_type=resolved_content_type,
        )
        if serializer is None:
            raise TypeError(
                "No default serializer available for type "
                f"'{python_type.__module__}.{python_type.__qualname__}'."
            )

        self.register(serializer, replace=replace)
        return serializer

    def unregister(
        self,
        *,
        schema_id: SchemaId | str,
        content_type: ContentType | str,
    ) -> None:
        """Remove one serializer entry by key.

        Args:
            schema_id: Schema id key to remove.
            content_type: Content type key to remove.

        Returns:
            None: Always returns after removal.

        Raises:
            UnknownSerializerError: If key does not exist.
        """
        normalized_schema_id = coerce_schema_id(schema_id)
        normalized_content_type = coerce_content_type(content_type)
        with self._lock:
            key = (normalized_schema_id, normalized_content_type)
            if key not in self._serializers:
                raise UnknownSerializerError(
                    schema_id=normalized_schema_id,
                    content_type=normalized_content_type,
                )
            del self._serializers[key]

    def has(
        self,
        *,
        schema_id: SchemaId | str,
        content_type: ContentType | str,
    ) -> bool:
        """Return whether a serializer exists for key.

        Args:
            schema_id: Schema id key to test.
            content_type: Content type key to test.

        Returns:
            bool: True when a serializer is registered for the key.
        """
        normalized_schema_id = coerce_schema_id(schema_id)
        normalized_content_type = coerce_content_type(content_type)
        with self._lock:
            return (normalized_schema_id, normalized_content_type) in self._serializers

    def encode(
        self,
        value: object,
        *,
        schema_id: SchemaId | str,
        content_type: ContentType | str,
    ) -> WirePayload:
        """Encode object value into a wire payload.

        Args:
            value: Runtime value to encode.
            schema_id: Serializer schema id key.
            content_type: Serializer content type key.

        Returns:
            WirePayload: Transport payload bytes and metadata.

        Raises:
            UnknownSerializerError: If serializer key cannot be resolved.
            SerializerEncodeError: If serializer raises during encode.
        """
        normalized_schema_id = coerce_schema_id(schema_id)
        normalized_content_type = coerce_content_type(content_type)
        serializer = self._resolve_serializer_for_encode(
            value=value,
            schema_id=normalized_schema_id,
            content_type=normalized_content_type,
        )
        if serializer is None:
            raise UnknownSerializerError(
                schema_id=normalized_schema_id,
                content_type=normalized_content_type,
            )

        try:
            body = serializer.serialize(value)
        except Exception as exc:  # noqa: BLE001
            raise SerializerEncodeError(
                schema_id=normalized_schema_id,
                content_type=normalized_content_type,
                reason=str(exc),
            ) from exc

        return WirePayload(
            schema_id=normalized_schema_id,
            content_type=normalized_content_type,
            encoding=serializer.encoding,
            body=body,
        )

    def decode(self, wire_payload: WirePayload) -> object:
        """Decode a wire payload.

        Args:
            wire_payload: Transport payload bytes and metadata.

        Returns:
            object: Decoded python value.

        Raises:
            UnknownSerializerError: If serializer key cannot be resolved.
            SerializerDecodeError: If serializer raises during decode.
        """
        serializer = self._resolve_serializer_for_decode(
            schema_id=wire_payload.schema_id,
            content_type=wire_payload.content_type,
        )
        if serializer is None:
            raise UnknownSerializerError(
                schema_id=wire_payload.schema_id,
                content_type=wire_payload.content_type,
            )

        try:
            return serializer.deserialize(wire_payload.body)
        except Exception as exc:  # noqa: BLE001
            raise SerializerDecodeError(
                schema_id=wire_payload.schema_id,
                content_type=wire_payload.content_type,
                reason=str(exc),
            ) from exc

    @property
    def serializers(self) -> tuple[MessageSerializer, ...]:
        """Return immutable snapshot of registered serializers.

        Returns:
            tuple[MessageSerializer, ...]: Registered serializer snapshot.
        """
        with self._lock:
            return tuple(self._serializers.values())

    def _lookup_serializer(
        self,
        *,
        schema_id: SchemaId,
        content_type: ContentType,
    ) -> MessageSerializer | None:
        with self._lock:
            return self._serializers.get((schema_id, content_type))

    def _resolve_serializer_for_encode(
        self,
        *,
        value: object,
        schema_id: SchemaId,
        content_type: ContentType,
    ) -> MessageSerializer | None:
        serializer = self._lookup_serializer(
            schema_id=schema_id,
            content_type=content_type,
        )
        if serializer is not None:
            return serializer
        if not self._auto_register_defaults:
            return None

        inferred_serializer = _create_serializer_for_type(
            python_type=type(value),
            schema_id=schema_id,
            content_type=content_type,
        )
        if inferred_serializer is None:
            return None

        # Auto-registration can race; treat conflict as "another thread won".
        self._register_if_missing(inferred_serializer)
        return self._lookup_serializer(
            schema_id=schema_id,
            content_type=content_type,
        )

    def _resolve_serializer_for_decode(
        self,
        *,
        schema_id: SchemaId,
        content_type: ContentType,
    ) -> MessageSerializer | None:
        serializer = self._lookup_serializer(
            schema_id=schema_id,
            content_type=content_type,
        )
        if serializer is not None:
            return serializer
        if not self._auto_register_defaults:
            return None

        # Decode-only fallback is intentionally conservative:
        # generic JSON payloads can still be recovered as dict/list primitives,
        # but typed protobuf/pydantic/dataclass decode requires explicit registration.
        if not _is_json_content_type(content_type):
            return None

        self._register_if_missing(
            JsonValueSerializer(
                schema_id=schema_id,
                content_type=content_type,
            )
        )
        return self._lookup_serializer(
            schema_id=schema_id,
            content_type=content_type,
        )

    def _register_if_missing(self, serializer: MessageSerializer) -> None:
        try:
            self.register(serializer)
        except SerializerConflictError:
            # Concurrent auto-registration can race; existing entry is valid.
            return


def create_default_serializer_registry() -> SerializerRegistry:
    """Create runtime default serializer registry instance.

    Returns:
        SerializerRegistry: Registry configured with default auto-inference.
    """
    return SerializerRegistry(auto_register_defaults=True)


def infer_schema_id_for_value(value: object) -> SchemaId:
    """Infer globally namespaced schema id from payload value.

    Args:
        value: Runtime payload value.

    Returns:
        SchemaId: Globally namespaced schema id.
    """
    return infer_schema_id_for_type(type(value))


def infer_schema_id_for_type(python_type: type[object]) -> SchemaId:
    """Infer globally namespaced schema id from Python type.

    Args:
        python_type: Python runtime type.

    Returns:
        SchemaId: Globally namespaced schema id.
    """
    raw_value = f"{python_type.__module__}.{python_type.__qualname__}"
    normalized_value = _normalize_schema_value(raw_value)
    return SchemaId(normalized_value)


def infer_content_type_for_value(value: object) -> ContentType:
    """Infer default content type from payload value.

    Args:
        value: Runtime payload value.

    Returns:
        ContentType: Default content type for payload.
    """
    return infer_content_type_for_type(type(value))


def infer_content_type_for_type(python_type: type[object]) -> ContentType:
    """Infer default content type from Python type.

    Args:
        python_type: Python runtime type.

    Returns:
        ContentType: Default content type for type.
    """
    if _is_protobuf_type(python_type):
        return PROTOBUF_CONTENT_TYPE
    if python_type in (bytes, bytearray, memoryview):
        return OCTET_STREAM_CONTENT_TYPE
    return JSON_CONTENT_TYPE


def _create_serializer_for_type(
    *,
    python_type: type[object],
    schema_id: SchemaId,
    content_type: ContentType,
) -> MessageSerializer | None:
    """Return a default serializer for `(python_type, content_type)` pair."""
    if _is_protobuf_type(python_type):
        if not _is_protobuf_content_type(content_type):
            return None
        return ProtobufSerializer(
            schema_id=schema_id,
            message_type=python_type,
            content_type=content_type,
        )

    if _is_pydantic_type(python_type):
        if not _is_json_content_type(content_type):
            return None
        return PydanticJsonSerializer(
            schema_id=schema_id,
            model_type=python_type,
            content_type=content_type,
        )

    if is_dataclass(python_type):
        if not _is_json_content_type(content_type):
            return None
        return DataclassJsonSerializer(
            schema_id=schema_id,
            model_type=python_type,
            content_type=content_type,
        )

    if _is_json_content_type(content_type):
        return JsonValueSerializer(
            schema_id=schema_id,
            content_type=content_type,
        )

    return None


def _is_json_content_type(content_type: ContentType) -> bool:
    """Return whether content type belongs to JSON family."""
    return content_type.value == JSON_CONTENT_TYPE.value or content_type.value.endswith(
        "+json"
    )


def _is_protobuf_content_type(content_type: ContentType) -> bool:
    """Return whether content type belongs to protobuf family."""
    return (
        content_type.value == PROTOBUF_CONTENT_TYPE.value
        or content_type.value.endswith("+protobuf")
    )


def _normalize_schema_value(raw_value: str) -> str:
    """Normalize Python type path into valid SchemaId value."""
    normalized_characters: list[str] = []
    for character in raw_value:
        if character.isalnum() or character in "_.:/-":
            normalized_characters.append(character)
            continue
        normalized_characters.append("_")
    return "".join(normalized_characters)


def _is_pydantic_type(python_type: type[object]) -> TypeGuard[type[BaseModel]]:
    return issubclass(python_type, BaseModel)


def _is_protobuf_type(
    python_type: type[object],
) -> TypeGuard[type[ProtobufMessage]]:
    return issubclass(python_type, ProtobufMessage)
