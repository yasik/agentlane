"""Tests for the native Tool primitive and tool execution."""

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Annotated, cast

import pytest
from pydantic import BaseModel

from agentlane.models import (
    ModelBehaviorError,
    Tool,
    ToolCall,
    ToolExecutionContext,
    ToolExecutor,
    Tools,
    ToolSpec,
    as_tool,
)
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    """Arguments for the test echo tool."""

    text: str


class EchoResult(BaseModel):
    """Structured return value for the test echo tool."""

    echoed: str


async def _echo_handler(
    args: EchoArgs,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> EchoResult:
    """Return a structured echo result."""
    del context
    del cancellation_token
    return EchoResult(echoed=args.text)


def _make_tool_call(arguments: str) -> ToolCall:
    """Build one tool call payload for tests."""
    return ToolCall.model_validate(
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "echo",
                "arguments": arguments,
            },
        }
    )


def test_tool_runs_and_formats_pydantic_result() -> None:
    """Tool.run should execute the handler and stringify Pydantic output."""
    tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )

    result = asyncio.run(
        tool.run(
            EchoArgs(text="hello"),
            CancellationToken(),
        )
    )

    assert result == EchoResult(echoed="hello")
    assert json.loads(tool.return_value_as_string(result)) == {"echoed": "hello"}


def test_tool_run_passes_context() -> None:
    """Tool.run should pass explicit framework context to the handler."""
    seen_contexts: list[ToolExecutionContext] = []

    async def context_handler(
        args: EchoArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> EchoResult:
        del cancellation_token
        seen_contexts.append(context)
        return EchoResult(echoed=args.text)

    context = ToolExecutionContext(
        run_id="run_1",
        agent_name="Reviewer",
        tool_call_id="call_1",
        metadata={"surface": "cli"},
    )
    tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=context_handler,
    )

    result = asyncio.run(
        tool.run(
            EchoArgs(text="hello"),
            CancellationToken(),
            context=context,
        )
    )

    assert result == EchoResult(echoed="hello")
    assert seen_contexts == [context]


def test_tool_from_function_infers_name_description_and_schema() -> None:
    """Tool.from_function should infer the schema from a typed callable."""

    async def search_docs(query: str, limit: int = 3) -> str:
        """Search the docs for the requested topic."""
        return f"{query}:{limit}"

    tool = Tool.from_function(search_docs)

    args_model = tool.args_type()
    result = asyncio.run(
        tool.run(
            args_model(query="harness"),
            CancellationToken(),
        )
    )

    assert tool.name == "search_docs"
    assert tool.description == "Search the docs for the requested topic."
    assert result == "harness:3"
    assert args_model(query="harness").model_dump()["limit"] == 3
    assert tool.schema["parameters"]["required"] == ["query", "limit"]
    assert tool.schema["parameters"]["properties"]["limit"]["default"] == 3


def test_tool_from_function_supports_annotated_descriptions_and_cancellation() -> None:
    """`cancellation_token` should be injected and not exposed in the schema."""
    received_tokens: list[CancellationToken] = []

    async def lookup_city(
        city: Annotated[str, "City to search for"],
        cancellation_token: CancellationToken,
    ) -> str:
        """Look up one city."""
        received_tokens.append(cancellation_token)
        return city.upper()

    tool = Tool.from_function(lookup_city)
    token = CancellationToken()
    args_model = tool.args_type()
    result = asyncio.run(tool.run(args_model(city="berlin"), token))

    assert result == "BERLIN"
    assert received_tokens == [token]
    assert "cancellation_token" not in tool.schema["parameters"]["properties"]
    assert tool.schema["parameters"]["properties"]["city"]["description"] == (
        "City to search for"
    )


def test_tool_from_function_supports_context_injection() -> None:
    """`context` should be injected and not exposed in the schema."""
    received_contexts: list[ToolExecutionContext] = []

    async def lookup_order(
        order_id: str,
        context: ToolExecutionContext,
    ) -> str:
        """Look up one order."""
        received_contexts.append(context)
        return order_id.upper()

    tool = Tool.from_function(lookup_order)
    token = CancellationToken()
    context = ToolExecutionContext(
        run_id="run_1",
        agent_name="LookupAgent",
        tool_call_id="call_1",
    )
    args_model = tool.args_type()
    result = asyncio.run(
        tool.run(
            args_model(order_id="a-123"),
            token,
            context=context,
        )
    )

    assert result == "A-123"
    assert received_contexts == [context]
    assert "context" not in tool.schema["parameters"]["properties"]


def test_tool_from_function_raises_for_missing_parameter_annotations() -> None:
    """Visible callable parameters must be annotated for schema inference."""

    def invalid_tool(query: str, limit: int) -> str:
        return f"{query}:{limit}"

    invalid_tool.__annotations__.pop("query")

    with pytest.raises(TypeError, match="Missing: query"):
        Tool.from_function(cast(Callable[..., object], invalid_tool))


def test_as_tool_decorates_typed_callable() -> None:
    """as_tool should return a native Tool from a typed function declaration."""

    @as_tool
    async def lookup_order(
        order_id: str,
        cancellation_token: CancellationToken,
    ) -> str:
        """Look up one order."""
        del cancellation_token
        return f"order:{order_id}"

    args_model = lookup_order.args_type()
    result = asyncio.run(
        lookup_order.run(
            args_model(order_id="A-123"),
            CancellationToken(),
        )
    )

    assert isinstance(lookup_order, Tool)
    assert lookup_order.name == "lookup_order"
    assert lookup_order.description == "Look up one order."
    assert result == "order:A-123"


def test_as_tool_supports_configured_overrides() -> None:
    """as_tool(...) should support explicit native tool overrides."""

    @as_tool(name="help_search", description="Search the help center.")
    async def search_help_center(question: str) -> str:
        return f"result:{question}"

    args_model = search_help_center.args_type()
    result = asyncio.run(
        search_help_center.run(
            args_model(question="returns"),
            CancellationToken(),
        )
    )

    assert isinstance(search_help_center, Tool)
    assert search_help_center.name == "help_search"
    assert search_help_center.description == "Search the help center."
    assert result == "result:returns"


def test_tools_accept_plain_typed_callables() -> None:
    """Tools should normalize plain typed callables into native Tool values."""

    async def search_help_center(question: str) -> str:
        """Search the help center for one policy answer."""
        return f"result:{question}"

    tools = Tools(tools=[search_help_center])

    assert len(tools.normalized_tools) == 1
    normalized_tool = tools.normalized_tools[0]
    assert isinstance(normalized_tool, Tool)
    assert normalized_tool.name == "search_help_center"
    assert (
        normalized_tool.description == "Search the help center for one policy answer."
    )
    assert tools.as_args()["tools"] == [
        {
            "type": "function",
            "function": normalized_tool.schema,
        }
    ]


def test_tools_keep_declarative_tool_specs_while_filtering_executable_tools() -> None:
    """Tools should support schema-only tool definitions cleanly."""
    tool_spec = ToolSpec(
        name="delegate_policy",
        description="Delegate one policy lookup.",
        args_model=EchoArgs,
    )
    executable_tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )

    tools = Tools(tools=[tool_spec, executable_tool])

    assert [tool.name for tool in tools.normalized_tools] == [
        "delegate_policy",
        "echo",
    ]
    assert [tool.name for tool in tools.executable_tools] == ["echo"]


def test_tool_executor_returns_chat_completion_tool_message() -> None:
    """The default tool adapter should emit a chat-completions tool message."""
    tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
    )
    executor = ToolExecutor()

    messages = asyncio.run(
        executor.execute(
            tool_calls=[_make_tool_call('{"text": "hello"}')],
            tools=Tools(tools=[tool]),
        )
    )

    assert messages == [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "name": "echo",
            "content": '{"echoed":"hello"}',
        }
    ]


def test_tool_executor_passes_per_call_context() -> None:
    """ToolExecutor should pass explicit context keyed by tool call id."""
    seen_contexts: list[ToolExecutionContext] = []

    async def context_handler(
        args: EchoArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> EchoResult:
        del cancellation_token
        seen_contexts.append(context)
        return EchoResult(echoed=args.text)

    context = ToolExecutionContext(
        run_id="run_1",
        agent_name="Reviewer",
        tool_call_id="call_1",
    )
    tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=context_handler,
    )
    executor = ToolExecutor()

    messages = asyncio.run(
        executor.execute(
            tool_calls=[_make_tool_call('{"text": "hello"}')],
            tools=Tools(tools=[tool]),
            context={"call_1": context},
        )
    )

    assert messages[0]["content"] == '{"echoed":"hello"}'
    assert seen_contexts == [context]


def test_tool_executor_defaults_context_to_tool_call_id() -> None:
    """ToolExecutor should supply the call id even without caller context."""
    seen_contexts: list[ToolExecutionContext] = []

    async def context_handler(
        args: EchoArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> EchoResult:
        del cancellation_token
        seen_contexts.append(context)
        return EchoResult(echoed=args.text)

    tool: Tool[EchoArgs, EchoResult] = Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=context_handler,
    )
    executor = ToolExecutor()

    messages = asyncio.run(
        executor.execute(
            tool_calls=[_make_tool_call('{"text": "hello"}')],
            tools=Tools(tools=[tool]),
        )
    )

    assert messages[0]["content"] == '{"echoed":"hello"}'
    assert seen_contexts == [
        ToolExecutionContext(
            tool_call_id="call_1",
        )
    ]


def test_tool_executor_raises_for_unregistered_tool() -> None:
    """Unknown tool calls should fail fast with ModelBehaviorError."""
    executor = ToolExecutor()

    with pytest.raises(ModelBehaviorError, match="Tool 'echo' is not registered"):
        asyncio.run(
            executor.execute(
                tool_calls=[_make_tool_call('{"text": "hello"}')],
                tools=Tools(tools=[]),
            )
        )


def _explicit_schema_tool() -> Tool[EchoArgs, EchoResult]:
    """Build an echo tool with a custom formatter and explicit schema."""
    return Tool(
        name="echo",
        description="Echo text",
        args_model=EchoArgs,
        handler=_echo_handler,
        formatter=lambda result: f"<<{result.echoed}>>",
        parameters_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
            "title": "custom",
        },
    )


def test_tool_replace_preserves_unspecified_fields_returns_copy() -> None:
    """Tool.replace should copy every field not named in the overrides."""
    original = _explicit_schema_tool()

    replaced = original.replace(name="echo_v2")

    assert replaced is not original
    assert replaced.name == "echo_v2"
    # Description, handler, formatter, and the explicit schema all survive.
    assert replaced.description == original.description
    assert replaced.handler is original.handler
    assert replaced.return_value_as_string(EchoResult(echoed="hi")) == "<<hi>>"
    assert replaced.schema["parameters"] == original.schema["parameters"]
    assert replaced.schema["name"] == "echo_v2"


def test_tool_replace_does_not_drop_formatter_when_overriding_handler() -> None:
    """Overriding the handler must keep the formatter (the Vera regression)."""
    original = _explicit_schema_tool()

    async def new_handler(
        args: EchoArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> EchoResult:
        del cancellation_token
        del context
        return EchoResult(echoed=args.text.upper())

    replaced = original.replace(handler=new_handler)

    result = asyncio.run(replaced.run(EchoArgs(text="hi"), CancellationToken()))
    assert result == EchoResult(echoed="HI")
    # Formatter is preserved by construction, not silently dropped.
    assert replaced.return_value_as_string(result) == "<<HI>>"


def test_tool_replace_unknown_field_raises_type_error() -> None:
    """Tool.replace should reject overrides that are not constructor fields."""
    original = _explicit_schema_tool()

    with pytest.raises(TypeError, match="unknown field"):
        original.replace(parameters="oops")


def test_tool_with_handler_wraps_original_handler_preserves_fields() -> None:
    """Tool.with_handler should pass the original handler to the wrapper."""
    original = _explicit_schema_tool()
    seen_handlers: list[object] = []

    def wrapper(
        inner: Callable[
            [EchoArgs, CancellationToken, ToolExecutionContext],
            object,
        ],
    ) -> Callable[[EchoArgs, CancellationToken, ToolExecutionContext], object]:
        seen_handlers.append(inner)

        async def wrapped(
            args: EchoArgs,
            cancellation_token: CancellationToken,
            context: ToolExecutionContext,
        ) -> EchoResult:
            # Intercept arguments before calling the original handler.
            adjusted = EchoArgs(text=f"[{args.text}]")
            result = inner(adjusted, cancellation_token, context)
            if inspect.isawaitable(result):
                result = await result
            return cast(EchoResult, result)

        return wrapped

    wrapped_tool = original.with_handler(wrapper)

    assert seen_handlers == [original.handler]
    result = asyncio.run(wrapped_tool.run(EchoArgs(text="hi"), CancellationToken()))
    assert result == EchoResult(echoed="[hi]")
    # Every non-handler field is carried through unchanged.
    assert wrapped_tool.name == original.name
    assert wrapped_tool.schema["parameters"] == original.schema["parameters"]
    assert wrapped_tool.return_value_as_string(result) == "<<[hi]>>"


def test_tool_replace_round_trips_every_observable_field() -> None:
    """replace() with no overrides must reproduce every Tool field exactly.

    This is the framework-owned version of Vera's reflection drift test: a copy
    that silently drops a field (the ``formatter`` regression Vera hit) would
    fail here. If a new ``Tool`` field is added, this round-trip is the place
    that proves the copy carries it.
    """
    original = _explicit_schema_tool()

    copy = original.replace()

    assert copy is not original
    assert copy.name == original.name
    assert copy.description == original.description
    assert copy.args_type() is original.args_type()
    assert copy.handler is original.handler
    assert copy.formatter is original.formatter
    assert copy.schema["parameters"] == original.schema["parameters"]


def test_tool_constructor_signature_matches_known_copy_fields() -> None:
    """The Tool constructor keyword set must match the fields copies forward.

    ``replace``/``with_handler`` rebuild from a fixed field set. If a new
    keyword is added to ``Tool.__init__`` without being threaded through the
    copy path, this assertion flags it so the copy cannot silently drop it.
    """
    constructor_fields = {
        name
        for name, parameter in inspect.signature(Tool).parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }

    assert constructor_fields == {
        "name",
        "description",
        "args_model",
        "handler",
        "formatter",
        "parameters_schema",
    }
