"""Bounded JSON records shared by the SPACE host and guest worker."""

import base64
import hashlib
from functools import cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import agentlane

INPUT_LIMIT = 48 * 1024
"""Maximum decoded request size in bytes, before base64 encoding."""

RESULT_LIMIT = 1024 * 1024
"""Maximum worker response size in bytes."""

FILE_TOOL_NAMES = ("read", "find", "grep", "patch", "write")
"""Guest file tools in the standard AgentLane selection order."""


class WorkerContext(BaseModel):
    """Serializable correlation fields for one guest tool call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None


class _Envelope(BaseModel):
    """Require an integer version instead of accepting JSON booleans as 1."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: Literal[1] = 1

    @field_validator("protocol", mode="before")
    @classmethod
    def validate_protocol(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("protocol must be an integer")

        return value


class WorkerRequest(_Envelope):
    """One file-tool call or guest readiness check."""

    tool: Literal["check", "read", "find", "grep", "patch", "write"]
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: WorkerContext = Field(default_factory=WorkerContext)


class WorkerError(BaseModel):
    """Sanitized failure metadata, including native tool failure kinds."""

    model_config = ConfigDict(extra="forbid", strict=True)

    message: str
    kind: str | None = None


class ToolMetadata(BaseModel):
    """Guest schema and prompt text for one native file tool."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    description: str
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = Field(default_factory=list)
    parameters_schema: dict[str, Any]


class WorkerInfo(BaseModel):
    """Guest build identity and path-dependent tool metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: str
    fingerprint: str
    cwd: str
    tools: list[ToolMetadata]


class WorkerResponse(_Envelope):
    """One complete worker response, without Python object serialization."""

    text: str = ""
    error: WorkerError | None = None
    info: WorkerInfo | None = None


def encode_request(request: WorkerRequest) -> str:
    """Encode one request as bounded JSON in a separate argv value.

    Raises:
        ValueError: If the JSON request exceeds the input limit.
    """
    payload = request.model_dump_json().encode("utf-8")
    if len(payload) > INPUT_LIMIT:
        raise ValueError("SPACE worker request exceeds the 48 KiB input limit")

    return base64.b64encode(payload).decode("ascii")


def decode_request(encoded_request: str) -> WorkerRequest:
    """Validate a bounded argv request before any guest tool executes."""
    # Reject oversized encoded input before allocating its decoded payload.
    # Check the decoded size too because base64 padding changes its exact length.
    if len(encoded_request) > 4 * ((INPUT_LIMIT + 2) // 3):
        raise ValueError("SPACE worker request exceeds the 48 KiB input limit")

    try:
        payload = base64.b64decode(encoded_request, validate=True)
    except ValueError:
        raise ValueError("invalid SPACE worker request encoding") from None

    if len(payload) > INPUT_LIMIT:
        raise ValueError("SPACE worker request exceeds the 48 KiB input limit")

    try:
        request = WorkerRequest.model_validate_json(payload)
    except ValidationError:
        raise ValueError("invalid SPACE worker request") from None

    # The model default helps local construction. It must not let unversioned
    # wire input silently claim compatibility with this worker.
    if "protocol" not in request.model_fields_set:
        raise ValueError("SPACE worker request has no protocol version")

    return request


def decode_response(payload: bytes) -> WorkerResponse:
    """Validate one complete JSON response after assembling its SSE frames."""
    if len(payload) > RESULT_LIMIT:
        raise ValueError("SPACE worker response exceeds the 1 MiB result limit")

    try:
        response = WorkerResponse.model_validate_json(payload)
    except ValidationError:
        raise ValueError("invalid SPACE worker response") from None

    # Require the peer to declare its protocol, even though local models default it.
    if "protocol" not in response.model_fields_set:
        raise ValueError("SPACE worker response has no protocol version")

    return response


@cache
def build_identity() -> tuple[str, str]:
    """Return the AgentLane version and a source fingerprint for both packages.

    Hash relative source paths and file contents so installed wheels and editable
    checkouts have the same identity. The cache assumes source files do not change
    while a sandbox handle is attached.
    """
    roots = (
        ("agentlane", Path(agentlane.__file__).parent),
        ("agentlane_space", Path(__file__).parent),
    )
    digest = hashlib.sha256()
    for package_name, root in roots:
        for source in sorted(root.rglob("*.py")):
            name = f"{package_name}/{source.relative_to(root).as_posix()}"
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(source.read_bytes()).digest())

    return agentlane.__version__, digest.hexdigest()
