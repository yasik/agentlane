"""Native SPACE command records."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded output and a confirmed command exit."""

    stdout: bytes
    stderr: bytes
    combined: bytes
    exit_code: int
    timed_out: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class CommandIdentity:
    """Identifiers retained for application recovery, without credentials."""

    idempotency_key: str
    process_id: str | None = None
    session_id: str | None = None


class AttachInfo(BaseModel):
    """The subset of attachment fields used by the native client."""

    model_config = ConfigDict(strict=True)
    events_url: str
    token: str = Field(min_length=1, repr=False)


class CreatedCommand(BaseModel):
    """A validated command creation response."""

    model_config = ConfigDict(strict=True)
    session_id: str = Field(pattern=r"^proc_[A-Za-z0-9_-]+$")
    process_id: str = Field(pattern=r"^proc_[A-Za-z0-9_-]+$")
    sandbox_id: str
    attach: AttachInfo


class ServerFrame(BaseModel):
    """Native SSE data parsed once at the transport boundary."""

    model_config = ConfigDict(strict=True)
    type: str
    seq: int | None = Field(default=None, ge=1)
    data_base64: str | None = None
    process_id: str | None = None
    exit_code: int | None = None
    status: str | None = None
