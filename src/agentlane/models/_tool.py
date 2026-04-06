"""Native tool primitive shared by clients and the harness."""

import functools
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model

from ..runtime import CancellationToken
from ._strict_schema import ensure_strict_json_schema

type ToolHandler[ArgsT: BaseModel, ResultT] = Callable[
    [ArgsT, CancellationToken],
    ResultT | Awaitable[ResultT],
]
"""Callable used to execute one tool invocation."""

type ToolFormatter[ResultT] = Callable[[ResultT], str]
"""Callable used to render a tool result back into the conversation."""

type ToolFunction = Callable[..., Any | Awaitable[Any]]
"""Developer-facing callable used to define a tool ergonomically."""


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

    @staticmethod
    def from_function(
        func: ToolFunction,
        *,
        name: str | None = None,
        description: str | None = None,
        formatter: ToolFormatter[Any] | None = None,
        parameters_schema: dict[str, Any] | None = None,
    ) -> "Tool[BaseModel, Any]":
        """Create a tool directly from a normal typed Python callable.

        The function signature becomes the tool schema. Visible parameters must
        be type annotated. A parameter named `cancellation_token` is treated as
        framework-injected and is excluded from the tool schema. Parameter
        descriptions may be supplied with `Annotated[..., "description"]`.

        Args:
            func: Sync or async Python callable to expose as a tool.
            name: Optional tool name override. Defaults to the function name.
            description: Optional tool description override. Defaults to the
                function docstring when present.
            formatter: Optional formatter for converting return values to text.
            parameters_schema: Optional explicit parameters schema override.
        """
        tool_name = name or _callable_name(func)
        tool_description = description or _callable_description(func, tool_name)
        signature = _typed_signature(func)
        args_model = _args_model_from_signature(tool_name, signature)
        parameter_names = tuple(
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.name != "cancellation_token"
        )
        has_cancellation_support = "cancellation_token" in signature.parameters

        async def inferred_handler(
            args: BaseModel,
            cancellation_token: CancellationToken,
        ) -> Any:
            kwargs = {name: getattr(args, name) for name in parameter_names}
            if has_cancellation_support:
                kwargs["cancellation_token"] = cancellation_token

            result = func(**kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        return Tool(
            name=tool_name,
            description=tool_description,
            args_model=args_model,
            handler=inferred_handler,
            formatter=formatter,
            parameters_schema=parameters_schema,
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


def _typed_signature(func: ToolFunction) -> inspect.Signature:
    """Return the callable signature with resolved type hints."""
    signature = inspect.signature(func)
    target = func.func if isinstance(func, functools.partial) else func
    globalns = getattr(target, "__globals__", {})
    type_hints = get_type_hints(target, globalns, include_extras=True)

    typed_parameters = [
        parameter.replace(
            annotation=type_hints.get(parameter.name, inspect.Signature.empty)
        )
        for parameter in signature.parameters.values()
    ]
    return signature.replace(parameters=typed_parameters)


def _callable_name(func: ToolFunction) -> str:
    """Return the default tool name for one callable."""
    target = func.func if isinstance(func, functools.partial) else func
    return getattr(target, "__name__", type(target).__name__)


def _callable_description(
    func: ToolFunction,
    tool_name: str,
) -> str:
    """Return the default tool description for one callable."""
    target = func.func if isinstance(func, functools.partial) else func
    docstring = inspect.getdoc(target)
    if docstring:
        return docstring.split("\n\n", maxsplit=1)[0].replace("\n", " ").strip()
    return f"Execute `{tool_name}`."


def _args_model_from_signature(
    tool_name: str,
    signature: inspect.Signature,
) -> type[BaseModel]:
    """Build a strict Pydantic args model from one callable signature."""
    field_definitions: dict[str, Any] = {}
    missing_annotations: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.name == "cancellation_token":
            continue

        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                "Tool.from_function does not support variadic parameters (`*args` or `**kwargs`)."
            )

        if parameter.annotation is inspect.Signature.empty:
            missing_annotations.append(parameter.name)
            continue

        field_definitions[parameter.name] = _field_definition(parameter)

    if missing_annotations:
        raise TypeError(
            "Tool.from_function requires type annotations for all exposed "
            f"parameters. Missing: {', '.join(missing_annotations)}."
        )

    model_name = _args_model_name(tool_name)
    return create_model(model_name, **field_definitions)


def _field_definition(parameter: inspect.Parameter) -> tuple[object, object]:
    """Build one Pydantic field definition from a function parameter."""
    description = _annotation_description(parameter.annotation)
    default_value: object
    if parameter.default is inspect.Signature.empty:
        default_value = ...
    else:
        default_value = parameter.default

    if description is None:
        return (parameter.annotation, default_value)

    return (parameter.annotation, Field(default_value, description=description))


def _annotation_description(annotation: object) -> str | None:
    """Extract a parameter description from `Annotated[..., '...']` metadata."""
    if get_origin(annotation) is not Annotated:
        return None

    for metadata in get_args(annotation)[1:]:
        if isinstance(metadata, str):
            return metadata
    return None


def _args_model_name(tool_name: str) -> str:
    """Return a stable generated args-model name from a tool name."""
    words = [word for word in re.split(r"[^0-9A-Za-z]+", tool_name) if word]
    if not words:
        return "ToolArgs"
    return "".join(word[:1].upper() + word[1:] for word in words) + "Args"
