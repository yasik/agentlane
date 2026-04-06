"""Tool execution for LLM clients.

This module provides reusable tool execution logic that can be shared
across different LLM clients (LiteLLM, OpenAI, Anthropic). It handles
parallel/sequential execution and optional tracing integration.
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from pydantic import ValidationError

from ..runtime import CancellationToken
from ..tracing import Span, function_span
from ._exceptions import ModelBehaviorError
from ._interface import ModelTracing, Tools
from ._tool_output_adapter import ChatCompletionsOutputAdapter, ToolOutputAdapter
from ._types import ToolCall

LOGGER = structlog.get_logger("agentlane.models.tool_executor")

type ToolStartCallback = Callable[[ToolCall], Awaitable[None] | None]
type ToolEndCallback = Callable[[ToolCall, object], Awaitable[None] | None]


class ToolExecutor:
    """Executes tool calls returned by LLM models.

    This class handles the execution of tool calls, supporting both parallel
    and sequential execution modes, with optional tracing integration.

    Example:
        ```python
        executor = ToolExecutor(tracing=ModelTracing.ENABLED)
        tool_messages = await executor.execute(
            tool_calls=tool_calls,
            tools=tools,
            parent_span=generation_span,
        )
        # tool_messages can be appended to the conversation
        ```
    """

    def __init__(
        self,
        tracing: ModelTracing = ModelTracing.DISABLED,
        adapter: ToolOutputAdapter | None = None,
    ) -> None:
        """Initialize the tool executor.

        Args:
            tracing: The tracing mode for tool execution spans.
            adapter: The output adapter for formatting tool results.
                Defaults to ChatCompletionsOutputAdapter if not provided.
        """
        self._tracing = tracing
        self._adapter = adapter or ChatCompletionsOutputAdapter()

    async def execute(
        self,
        *,
        tool_calls: list[ToolCall],
        tools: Tools,
        parent_span: Span[Any] | None = None,
        cancellation_token: CancellationToken | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return formatted messages.

        Args:
            tool_calls: List of tool calls from the model response.
            tools: Tools configuration containing available tools.
            parent_span: Optional parent tracing span for function calls.
            cancellation_token: Optional cancellation token for async operations.
            on_tool_start: Optional callback fired immediately before one tool
                handler invocation begins.
            on_tool_end: Optional callback fired after one tool invocation
                finishes or exhausts timeout retries.

        Returns:
            List of tool result messages formatted by the adapter.
            ```

        Raises:
            ModelBehaviorError: If a tool is not registered.
        """
        available_tools = {tool.name: tool for tool in tools.normalized_tools}
        token = cancellation_token or CancellationToken()
        timeout = tools.tool_call_timeout
        max_retries = tools.tool_call_max_retries

        async def _invoke(call: ToolCall) -> dict[str, Any]:
            """Invoke a single tool call."""
            function_name = call.function.name or ""
            tool = available_tools.get(function_name)
            if tool is None:
                raise ModelBehaviorError(f"Tool '{function_name}' is not registered.")

            raw_arguments = call.function.arguments or "{}"
            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                # Return error via adapter, allowing model to self-correct.
                return self._adapter.format_error(
                    call.id,
                    function_name,
                    f"Invalid arguments for {function_name}. "
                    f"Please call {function_name} with the required parameters.",
                )

            args_model_cls = tool.args_type()
            try:
                args_model = args_model_cls(**parsed_arguments)
            except ValidationError as exc:
                # Return validation error via adapter
                return self._adapter.format_error(
                    call.id,
                    function_name,
                    f"Invalid arguments for {function_name}: {exc}. "
                    f"Please retry with valid parameters.",
                )

            async def _run_tool() -> Any:
                if timeout is not None:
                    return await asyncio.wait_for(
                        tool.run(args_model, token),
                        timeout=timeout,
                    )
                return await tool.run(args_model, token)

            with function_span(
                name=function_name,
                inputs=raw_arguments,
                parent=parent_span,
                disabled=self._tracing.is_disabled(),
            ) as span_function:
                await _maybe_await(on_tool_start, call)
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        result = await _run_tool()
                        await _maybe_await(on_tool_end, call, result)
                        output_text = tool.return_value_as_string(result)
                        span_function.span_data.output = output_text
                        break
                    except TimeoutError:
                        if attempts <= max_retries:
                            LOGGER.warning(
                                "Tool call timed out, retrying",
                                tool=function_name,
                                attempt=attempts,
                                timeout=timeout,
                            )
                            continue
                        # All retries exhausted
                        LOGGER.error(
                            "Tool call timed out after all retries",
                            tool=function_name,
                            attempts=attempts,
                            timeout=timeout,
                        )
                        timeout_result = (
                            f"Error: Tool '{function_name}' timed out after {timeout}s "
                            f"({attempts} attempts)."
                        )
                        await _maybe_await(on_tool_end, call, timeout_result)
                        return self._adapter.format_error(
                            call.id,
                            function_name,
                            timeout_result,
                        )

            return self._adapter.format_success(call.id, function_name, output_text)

        # Invoke tool calls in parallel if enabled
        if tools.parallel_tool_calls and len(tool_calls) > 1:
            return await asyncio.gather(*[_invoke(call) for call in tool_calls])

        # Invoke tool calls sequentially
        responses: list[dict[str, Any]] = []
        for call in tool_calls:
            responses.append(await _invoke(call))

        return responses


async def _maybe_await(
    callback: Callable[..., Awaitable[None] | None] | None,
    *args: object,
) -> None:
    """Await an optional callback when it returns an awaitable."""
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result
