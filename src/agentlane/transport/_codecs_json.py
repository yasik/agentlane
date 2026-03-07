"""JSON serializer adapters used by transport default inference."""

import json
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from ._serializer import MessageSerializer
from ._types import JSON_CONTENT_TYPE, ContentType, SchemaId, WireEncoding
from ._utils import coerce_content_type, coerce_schema_id

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)
"""Type variable constrained to one concrete Pydantic model class."""

DataclassT = TypeVar("DataclassT")
"""Type variable constrained to one concrete dataclass model class."""


class PydanticJsonSerializer(MessageSerializer):
    """Serialize and deserialize Pydantic models using JSON bytes."""

    def __init__(
        self,
        *,
        schema_id: SchemaId | str,
        model_type: type[PydanticModelT],
        content_type: ContentType | str = JSON_CONTENT_TYPE,
    ) -> None:
        """Initialize serializer for a specific Pydantic model type.

        Args:
            schema_id: Globally namespaced schema id.
            model_type: Concrete pydantic model class.
            content_type: MIME-like content type for encoded payload.
        """
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)
        self._model_type = model_type

    @property
    def schema_id(self) -> SchemaId:
        """Return serializer schema id.

        Returns:
            SchemaId: Schema key handled by this serializer.
        """
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        """Return serializer content type key.

        Returns:
            ContentType: Content type handled by this serializer.
        """
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        """Return serializer wire encoding.

        Returns:
            WireEncoding: JSON wire encoding.
        """
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        """Return serializer python runtime type.

        Returns:
            type[object]: Configured pydantic model class.
        """
        return self._model_type

    def serialize(self, value: object) -> bytes:
        """Serialize a pydantic model instance to JSON bytes.

        Args:
            value: Runtime value expected to be the configured model class.

        Returns:
            bytes: UTF-8 JSON bytes.

        Raises:
            TypeError: If value is not instance of configured model class.
        """
        if not isinstance(value, self._model_type):
            raise TypeError(
                f"Expected value type '{self._model_type.__name__}', "
                f"got '{type(value).__name__}'."
            )
        # `model_dump_json` preserves Pydantic serialization semantics.
        return value.model_dump_json().encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
        """Deserialize JSON bytes into configured pydantic model type.

        Args:
            payload: UTF-8 JSON bytes payload.

        Returns:
            object: Reconstructed model instance.
        """
        return self._model_type.model_validate_json(payload)


class DataclassJsonSerializer(MessageSerializer):
    """Serialize and deserialize dataclass instances using JSON bytes."""

    def __init__(
        self,
        *,
        schema_id: SchemaId | str,
        model_type: type[DataclassT],
        content_type: ContentType | str = JSON_CONTENT_TYPE,
    ) -> None:
        """Initialize serializer for a specific dataclass type.

        Args:
            schema_id: Globally namespaced schema id.
            model_type: Concrete dataclass model class.
            content_type: MIME-like content type for encoded payload.

        Raises:
            TypeError: If `model_type` is not a dataclass type.
        """
        if not is_dataclass(model_type):
            raise TypeError(
                "DataclassJsonSerializer requires a dataclass type for `model_type`."
            )
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)
        self._model_type = model_type

    @property
    def schema_id(self) -> SchemaId:
        """Return serializer schema id.

        Returns:
            SchemaId: Schema key handled by this serializer.
        """
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        """Return serializer content type key.

        Returns:
            ContentType: Content type handled by this serializer.
        """
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        """Return serializer wire encoding.

        Returns:
            WireEncoding: JSON wire encoding.
        """
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        """Return serializer python runtime type.

        Returns:
            type[object]: Configured dataclass model class.
        """
        return self._model_type

    def serialize(self, value: object) -> bytes:
        """Serialize a dataclass instance to JSON bytes.

        Args:
            value: Runtime value expected to be configured dataclass instance.

        Returns:
            bytes: UTF-8 JSON bytes.

        Raises:
            TypeError: If value is wrong type or not a dataclass instance.
        """
        if not isinstance(value, self._model_type):
            raise TypeError(
                f"Expected value type '{self._model_type.__name__}', "
                f"got '{type(value).__name__}'."
            )
        if not _is_dataclass_instance(value):
            raise TypeError("Dataclass serializer expected a dataclass instance.")
        return json.dumps(asdict(cast(Any, value))).encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
        """Deserialize JSON bytes into configured dataclass model type.

        Args:
            payload: UTF-8 JSON bytes payload.

        Returns:
            object: Reconstructed dataclass instance.

        Raises:
            TypeError: If decoded JSON is not an object mapping.
        """
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise TypeError("Dataclass JSON payload must decode to an object.")
        # Dataclass reconstruction is strict to declared init fields.
        return self._model_type(**parsed)


class JsonValueSerializer(MessageSerializer):
    """Serialize and deserialize generic JSON-compatible values.

    This codec intentionally returns plain JSON structures (dict/list/scalars).
    It is used as the safe decode fallback when no typed model serializer exists.
    """

    def __init__(
        self,
        *,
        schema_id: SchemaId | str,
        content_type: ContentType | str = JSON_CONTENT_TYPE,
    ) -> None:
        """Initialize serializer for generic JSON values.

        Args:
            schema_id: Globally namespaced schema id.
            content_type: MIME-like content type for encoded payload.
        """
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)

    @property
    def schema_id(self) -> SchemaId:
        """Return serializer schema id.

        Returns:
            SchemaId: Schema key handled by this serializer.
        """
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        """Return serializer content type key.

        Returns:
            ContentType: Content type handled by this serializer.
        """
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        """Return serializer wire encoding.

        Returns:
            WireEncoding: JSON wire encoding.
        """
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        """Return serializer python runtime type.

        Returns:
            type[object]: Generic `object` marker for JSON-serializable values.
        """
        return object

    def serialize(self, value: object) -> bytes:
        """Serialize JSON-compatible value to UTF-8 bytes.

        Args:
            value: JSON-compatible python value.

        Returns:
            bytes: UTF-8 JSON bytes.
        """
        return json.dumps(value).encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
        """Deserialize UTF-8 JSON bytes into plain python structures.

        Args:
            payload: UTF-8 JSON bytes.

        Returns:
            object: Decoded JSON value (dict/list/scalar).
        """
        return json.loads(payload.decode("utf-8"))


def _is_dataclass_instance(value: object) -> bool:
    """Return whether a value is a dataclass instance (not a dataclass type)."""
    return not isinstance(value, type) and is_dataclass(value)
