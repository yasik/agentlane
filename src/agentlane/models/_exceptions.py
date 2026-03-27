from dataclasses import dataclass
from typing import Any


@dataclass
class RunErrorDetails:
    """Run error details."""

    raw_response: Any | None


class ModelsException(Exception):
    """Base class for all agents exceptions."""

    run_data: RunErrorDetails | None

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.run_data = None


class ModelBehaviorError(ModelsException):
    """Exception raised when the model does something unexpected,
    e.g. calling a tool that doesn't exist, or providing malformed JSON.
    """

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
