"""Transport-level identity and encoding types."""

import enum
import re
from dataclasses import dataclass

_SCHEMA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_CONTENT_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")


@dataclass(frozen=True, slots=True)
class SchemaId:
    """Global schema identifier for wire payload contracts."""

    value: str
    """Globally namespaced schema identifier string."""

    def __post_init__(self) -> None:
        # Keep ids strict and deterministic so remote nodes can share contracts.
        if not self.value:
            raise ValueError("SchemaId value must be non-empty.")
        if _SCHEMA_ID_PATTERN.match(self.value) is None:
            raise ValueError(
                "SchemaId contains unsupported characters. "
                "Use letters, numbers, and `_.:/-` separators."
            )
        if not any(separator in self.value for separator in (".", ":", "/")):
            raise ValueError(
                "SchemaId must be globally namespaced. "
                "Expected at least one of `.`, `:`, or `/`."
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ContentType:
    """MIME-like payload content type identifier."""

    value: str
    """Normalized MIME-like content type string."""

    def __post_init__(self) -> None:
        # Transport contracts rely on MIME-like values for cross-runtime clarity.
        if not self.value:
            raise ValueError("ContentType value must be non-empty.")
        if _CONTENT_TYPE_PATTERN.match(self.value) is None:
            raise ValueError(
                "ContentType must be MIME-like, for example `application/json`."
            )

    def __str__(self) -> str:
        return self.value


class WireEncoding(enum.StrEnum):
    """Physical wire encoding for serialized payload bytes."""

    JSON = "json"
    """UTF-8 JSON bytes."""

    PROTOBUF = "protobuf"
    """Protobuf binary payload bytes."""

    BYTES = "bytes"
    """Opaque binary bytes."""


JSON_CONTENT_TYPE = ContentType("application/json")
"""Canonical JSON content type."""

PROTOBUF_CONTENT_TYPE = ContentType("application/x-protobuf")
"""Canonical protobuf content type."""

OCTET_STREAM_CONTENT_TYPE = ContentType("application/octet-stream")
"""Canonical raw bytes content type."""
