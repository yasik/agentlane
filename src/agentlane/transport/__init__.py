"""Transport serialization primitives and adapters."""

from ._codecs_json import (
    DataclassJsonSerializer,
    JsonValueSerializer,
    PydanticJsonSerializer,
)
from ._codecs_protobuf import ProtobufSerializer
from ._errors import (
    SerializationError,
    SerializerConflictError,
    SerializerDecodeError,
    SerializerEncodeError,
    UnknownSerializerError,
)
from ._registry import (
    SerializerRegistry,
    create_default_serializer_registry,
    infer_content_type_for_type,
    infer_content_type_for_value,
    infer_schema_id_for_type,
    infer_schema_id_for_value,
)
from ._serializer import MessageSerializer
from ._types import (
    JSON_CONTENT_TYPE,
    OCTET_STREAM_CONTENT_TYPE,
    PROTOBUF_CONTENT_TYPE,
    ContentType,
    SchemaId,
    WireEncoding,
)
from ._wire_payload import (
    WirePayload,
    payload_format_for_wire_encoding,
    payload_to_wire_payload,
    wire_encoding_for_payload_format,
    wire_payload_to_payload,
)

__all__ = [
    "ContentType",
    "DataclassJsonSerializer",
    "JsonValueSerializer",
    "JSON_CONTENT_TYPE",
    "MessageSerializer",
    "OCTET_STREAM_CONTENT_TYPE",
    "PydanticJsonSerializer",
    "ProtobufSerializer",
    "PROTOBUF_CONTENT_TYPE",
    "SchemaId",
    "SerializationError",
    "SerializerConflictError",
    "SerializerDecodeError",
    "SerializerEncodeError",
    "SerializerRegistry",
    "create_default_serializer_registry",
    "infer_content_type_for_type",
    "infer_content_type_for_value",
    "infer_schema_id_for_type",
    "infer_schema_id_for_value",
    "UnknownSerializerError",
    "WireEncoding",
    "WirePayload",
    "payload_format_for_wire_encoding",
    "payload_to_wire_payload",
    "wire_encoding_for_payload_format",
    "wire_payload_to_payload",
]
