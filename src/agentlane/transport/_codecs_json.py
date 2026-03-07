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
        """Initialize serializer for a specific Pydantic model type."""
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)
        self._model_type = model_type

    @property
    def schema_id(self) -> SchemaId:
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        return self._model_type

    def serialize(self, value: object) -> bytes:
        if not isinstance(value, self._model_type):
            raise TypeError(
                f"Expected value type '{self._model_type.__name__}', "
                f"got '{type(value).__name__}'."
            )
        # `model_dump_json` preserves Pydantic serialization semantics.
        return value.model_dump_json().encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
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
        """Initialize serializer for a specific dataclass type."""
        if not is_dataclass(model_type):
            raise TypeError(
                "DataclassJsonSerializer requires a dataclass type for `model_type`."
            )
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)
        self._model_type = model_type

    @property
    def schema_id(self) -> SchemaId:
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        return self._model_type

    def serialize(self, value: object) -> bytes:
        if not isinstance(value, self._model_type):
            raise TypeError(
                f"Expected value type '{self._model_type.__name__}', "
                f"got '{type(value).__name__}'."
            )
        if not _is_dataclass_instance(value):
            raise TypeError("Dataclass serializer expected a dataclass instance.")
        return json.dumps(asdict(cast(Any, value))).encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
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
        """Initialize serializer for generic JSON values."""
        self._schema_id = coerce_schema_id(schema_id)
        self._content_type = coerce_content_type(content_type)

    @property
    def schema_id(self) -> SchemaId:
        return self._schema_id

    @property
    def content_type(self) -> ContentType:
        return self._content_type

    @property
    def encoding(self) -> WireEncoding:
        return WireEncoding.JSON

    @property
    def python_type(self) -> type[object]:
        return object

    def serialize(self, value: object) -> bytes:
        return json.dumps(value).encode("utf-8")

    def deserialize(self, payload: bytes) -> object:
        return json.loads(payload.decode("utf-8"))


def _is_dataclass_instance(value: object) -> bool:
    """Return whether a value is a dataclass instance (not a dataclass type)."""
    return not isinstance(value, type) and is_dataclass(value)
