"""Stable errors for the SPACE extension."""

from strenum import LowercaseStrEnum


class SpaceErrorKind(LowercaseStrEnum):
    """Failure categories exposed by SPACE tools."""

    INPUT_LIMIT = "input_limit"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"
    RECOVERY_REQUIRED = "recovery_required"
    NOT_CHECKED = "not_checked"
    CLOSED = "closed"


class SpaceError(Exception):
    """A sanitized integration failure, safe to return as tool text."""

    def __init__(self, message: str, *, kind: SpaceErrorKind) -> None:
        super().__init__(message)
        self.kind = kind
