"""Wire payload model and transport boundary conversion helpers."""

from dataclasses import dataclass
from typing import Protocol

from agentlane.messaging import Payload, PayloadFormat

from ._errors import SerializationError
from ._types import ContentType, SchemaId, WireEncoding


@dataclass(frozen=True, slots=True)
class WirePayload:
    """Transport-ready payload carrying bytes and explicit schema metadata."""

    schema_id: SchemaId
    """Globally namespaced schema identifier."""

    content_type: ContentType
    """MIME-like content type for the payload bytes."""

    encoding: WireEncoding
    """Physical wire encoding of `body`."""

    body: bytes
    """Serialized bytes payload."""


class PayloadCodecRegistry(Protocol):
    """Protocol for registry operations used by payload conversion helpers."""

    def encode(
        self,
        value: object,
        *,
        schema_id: SchemaId | str,
        content_type: ContentType | str,
    ) -> WirePayload:
        """Encode object into wire payload."""
        ...

    def decode(self, wire_payload: WirePayload) -> object:
        """Decode wire payload into object."""
        ...


def wire_encoding_for_payload_format(payload_format: PayloadFormat) -> WireEncoding:
    """Map messaging payload format to transport wire encoding."""
    if payload_format == PayloadFormat.JSON:
        return WireEncoding.JSON
    if payload_format == PayloadFormat.PROTOBUF:
        return WireEncoding.PROTOBUF
    return WireEncoding.BYTES


def payload_format_for_wire_encoding(encoding: WireEncoding) -> PayloadFormat:
    """Map transport wire encoding to messaging payload format."""
    if encoding == WireEncoding.JSON:
        return PayloadFormat.JSON
    if encoding == WireEncoding.PROTOBUF:
        return PayloadFormat.PROTOBUF
    return PayloadFormat.BYTES


def payload_to_wire_payload(
    payload: Payload,
    *,
    registry: PayloadCodecRegistry,
) -> WirePayload:
    """Convert in-memory messaging payload into bytes wire payload.

    BYTES payloads bypass registry codecs and are passed through directly.
    JSON/PROTOBUF payloads use the registry key `(schema_id, content_type)`.
    """
    schema_id = SchemaId(payload.schema_name)
    content_type = ContentType(payload.content_type)
    expected_encoding = wire_encoding_for_payload_format(payload.format)

    if payload.format == PayloadFormat.BYTES:
        if not isinstance(payload.data, bytes):
            raise SerializationError(
                "Payload with BYTES format must contain `bytes` in `data`."
            )
        return WirePayload(
            schema_id=schema_id,
            content_type=content_type,
            encoding=WireEncoding.BYTES,
            body=payload.data,
        )

    wire_payload = registry.encode(
        payload.data,
        schema_id=schema_id,
        content_type=content_type,
    )
    if wire_payload.encoding != expected_encoding:
        raise SerializationError(
            "Serializer encoding mismatch for payload conversion. "
            f"Expected '{expected_encoding.value}', got '{wire_payload.encoding.value}'."
        )
    return wire_payload


def wire_payload_to_payload(
    wire_payload: WirePayload,
    *,
    registry: PayloadCodecRegistry,
) -> Payload:
    """Convert transport wire bytes payload into messaging payload shape."""
    if wire_payload.encoding == WireEncoding.BYTES:
        decoded_data: object = wire_payload.body
    else:
        # For JSON/PROTOBUF encodings registry resolves typed or generic codecs.
        decoded_data = registry.decode(wire_payload)

    return Payload(
        schema_name=wire_payload.schema_id.value,
        content_type=wire_payload.content_type.value,
        format=payload_format_for_wire_encoding(wire_payload.encoding),
        data=decoded_data,
    )
