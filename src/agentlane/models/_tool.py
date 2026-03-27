"""Native tool primitive shared by clients and the harness."""

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from ..runtime import CancellationToken
from ._strict_schema import ensure_strict_json_schema

type ToolHandler[ArgsT: BaseModel, ResultT] = Callable[
    [ArgsT, CancellationToken],
    ResultT | Awaitable[ResultT],
]
"""Callable used to execute one tool invocation."""

type ToolFormatter[ResultT] = Callable[[ResultT], str]
"""Callable used to render a tool result back into the conversation."""


class Tool[ArgsT: BaseModel, ResultT]:
    """Framework-native tool primitive used by model clients and the harness."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_model: type[ArgsT],
        handler: ToolHandler[ArgsT, ResultT],
        formatter: ToolFormatter[ResultT] | None = None,
        parameters_schema: dict[str, Any] | None = None,
    ) -> None:
        """Initialize one callable tool definition.

        Args:
            name: Stable tool name exposed to the model.
            description: Human-readable tool description for the model.
            args_model: Pydantic arguments model used for validation.
            handler: Sync or async callable that executes the tool.
            formatter: Optional formatter for converting return values to text.
            parameters_schema: Optional explicit parameters schema override.
        """
        self.name = name
        self.description = description
        self._args_model = args_model
        self._handler = handler
        self._formatter = formatter
        self._parameters_schema = parameters_schema or ensure_strict_json_schema(
            args_model.model_json_schema()
        )

    @property
    def schema(self) -> dict[str, Any]:
        """Return the model-facing function schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters_schema,
        }

    def args_type(self) -> type[ArgsT]:
        """Return the pydantic arguments model type for the tool."""
        return self._args_model

    async def run(
        self,
        args: ArgsT,
        cancellation_token: CancellationToken,
    ) -> ResultT:
        """Execute the tool handler.

        Args:
            args: Validated tool arguments.
            cancellation_token: Cooperative cancellation token.

        Returns:
            ResultT: Tool execution result.
        """
        result = self._handler(args, cancellation_token)
        if inspect.isawaitable(result):
            return await result
        return result

    def return_value_as_string(self, value: ResultT) -> str:
        """Render a tool result into the string returned to the model."""
        if self._formatter is not None:
            return self._formatter(value)
        return _default_tool_formatter(value)


def _default_tool_formatter(value: object) -> str:
    """Render common Python values into stable tool-output strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(asdict(value))
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)
