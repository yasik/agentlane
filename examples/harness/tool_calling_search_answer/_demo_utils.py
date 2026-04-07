"""Small utilities that keep the tool-calling demo focused on the harness API."""

import logging
from typing import Any, cast

import structlog
from rich.console import Console
from rich.panel import Panel

from agentlane.harness import RunResult, RunState
from agentlane.messaging import DeliveryOutcome, DeliveryStatus
from agentlane.models import MessageDict


def configure_demo_logging() -> None:
    """Mute provider debug logs so the transcript stays readable."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

CONSOLE = Console()


def print_intro() -> None:
    """Print the short intro panel for the live demo."""
    CONSOLE.rule("[bold cyan]AgentLane Harness Tool-Calling Demo[/bold cyan]")
    CONSOLE.print(
        Panel.fit(
            "This example uses a real OpenAI response plus one mocked search tool.\n"
            "The tool is declared with the native @as_tool decorator.\n"
            "The search result is fake on purpose so the harness tool loop is easy to follow.\n"
            "The assistant's answer is generated live from the model.",
            title="What This Proves",
        )
    )


def require_run_result(outcome: DeliveryOutcome) -> RunResult:
    """Return the delivered run result for one runtime send outcome."""
    if outcome.status != DeliveryStatus.DELIVERED:
        message = outcome.error.message if outcome.error else "unknown delivery failure"
        raise RuntimeError(
            f"Harness example delivery failed with status={outcome.status.value}: {message}"
        )
    payload = outcome.response_payload
    if isinstance(payload, RunResult):
        return payload
    raise TypeError("Expected the harness agent to return a `RunResult`.")


def print_turn(question: str, answer: object) -> None:
    """Print the user question and final assistant answer."""
    CONSOLE.print(f"[bold]User:[/bold] {question}")
    CONSOLE.print(f"[bold]Assistant:[/bold] {answer}")
    CONSOLE.print()


def print_summary(model_name: str, run_state: RunState) -> None:
    """Print the captured tool trace and run summary."""
    tool_name, tool_arguments = _first_tool_call_details(run_state)
    tool_output = _first_tool_output(run_state)

    CONSOLE.print(
        Panel.fit(
            f"Model: {model_name}\n"
            f"Turns: {run_state.turn_count}\n"
            f"Raw responses: {len(run_state.responses)}\n"
            f"Tool: {tool_name}\n"
            f"Arguments: {tool_arguments}\n"
            f"Mocked search result: {tool_output}",
            title="Tool Loop Summary",
            border_style="green",
        )
    )


def _first_tool_call_details(run_state: RunState) -> tuple[str, str]:
    """Return the first tool name and arguments observed in the run."""
    for response in run_state.responses:
        response_payload = cast(dict[str, object], response.model_dump(mode="python"))
        choices_object: object = response_payload.get("choices")
        if not isinstance(choices_object, list) or not choices_object:
            continue
        choices = cast(list[object], choices_object)
        first_choice: object = choices[0]
        if not isinstance(first_choice, dict):
            continue
        first_choice_dict = cast(dict[str, object], first_choice)
        message_object: object = first_choice_dict.get("message")
        if not isinstance(message_object, dict):
            continue
        message = cast(dict[str, object], message_object)
        tool_calls_object: object = message.get("tool_calls")
        if not isinstance(tool_calls_object, list) or not tool_calls_object:
            continue
        tool_calls = cast(list[object], tool_calls_object)
        first_call: object = tool_calls[0]
        if not isinstance(first_call, dict):
            continue
        first_call_dict = cast(dict[str, object], first_call)
        function_object: object = first_call_dict.get("function")
        if not isinstance(function_object, dict):
            continue
        function = cast(dict[str, object], function_object)

        name: object = function.get("name")
        arguments: object = function.get("arguments")
        if isinstance(name, str) and isinstance(arguments, str):
            return name, arguments
        if isinstance(name, str):
            return name, "not found"

    return "not found", "not found"


def _first_tool_output(run_state: RunState) -> str:
    """Return the first tool output captured in continuation history."""
    for item in run_state.continuation_history:
        if not isinstance(item, dict):
            continue
        message = cast(MessageDict, item)
        role = message.get("role")
        if role != "tool":
            continue
        content = message.get("content")
        return _stringify_tool_output(content)
    return "not found"


def _stringify_tool_output(content: object) -> str:
    """Render the tool output content shown in the demo summary."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return str(cast(list[object], content))
    if isinstance(content, dict):
        return str(cast(dict[str, Any], content))
    return str(content)
