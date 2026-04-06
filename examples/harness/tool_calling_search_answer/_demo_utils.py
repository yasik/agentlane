"""Small utilities that keep the tool-calling demo focused on the harness API."""

import logging
import os
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel

from agentlane.harness import RunResult, RunState
from agentlane.messaging import DeliveryOutcome, DeliveryStatus

CONSOLE = Console()


def configure_demo_logging() -> None:
    """Mute provider debug logs so the transcript stays readable."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )


def load_openai_api_key(env_file: Path) -> str:
    """Return `OPENAI_API_KEY`, loading it from `.env` when needed."""
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to the repository .env file or the environment."
    )


def print_intro() -> None:
    """Print the short intro panel for the live demo."""
    CONSOLE.rule("[bold cyan]AgentLane Harness Tool-Calling Demo[/bold cyan]")
    CONSOLE.print(
        Panel.fit(
            "This example uses a real OpenAI response plus one mocked search tool.\n"
            "The tool is declared as a plain typed Python function.\n"
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
    tool_name = "not found"
    tool_arguments = "not found"
    tool_output = "not found"

    for response in run_state.responses:
        if not response.choices:
            continue
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            continue
        first_call = tool_calls[0]
        tool_name = first_call.function.name or tool_name
        tool_arguments = first_call.function.arguments or tool_arguments
        break

    for item in run_state.continuation_history:
        if isinstance(item, dict) and item.get("role") == "tool":
            content = item.get("content")
            tool_output = str(content)
            break

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
