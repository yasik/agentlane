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
        """Encode object into wire payload.

        Args:
            value: Runtime value to encode.
            schema_id: Schema id key.
            content_type: Content type key.

        Returns:
            WirePayload: Encoded wire payload.
        """
        ...

    def decode(self, wire_payload: WirePayload) -> object:
        """Decode wire payload into object.

        Args:
            wire_payload: Wire payload bytes and metadata.

        Returns:
            object: Decoded runtime value.
        """
        ...


def wire_encoding_for_payload_format(payload_format: PayloadFormat) -> WireEncoding:
    """Map messaging payload format to transport wire encoding.

    Args:
        payload_format: Messaging payload format enum value.

    Returns:
        WireEncoding: Transport wire encoding equivalent.
    """
    if payload_format == PayloadFormat.JSON:
        return WireEncoding.JSON
    if payload_format == PayloadFormat.PROTOBUF:
        return WireEncoding.PROTOBUF
    return WireEncoding.BYTES


def payload_format_for_wire_encoding(encoding: WireEncoding) -> PayloadFormat:
    """Map transport wire encoding to messaging payload format.

    Args:
        encoding: Transport wire encoding enum value.

    Returns:
        PayloadFormat: Messaging payload format equivalent.
    """
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

    Args:
        payload: Canonical in-memory payload.
        registry: Serializer registry used for non-bytes payloads.

    Returns:
        WirePayload: Transport-ready payload representation.

    Raises:
        SerializationError: If BYTES payload data is invalid or encoding mismatches.
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
    """Convert transport wire bytes payload into messaging payload shape.

    Args:
        wire_payload: Transport payload bytes and metadata.
        registry: Serializer registry used for non-bytes payloads.

    Returns:
        Payload: Canonical in-memory payload representation.
    """
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
