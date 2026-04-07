"""Small utilities that keep the harness demo focused on the core API."""

import logging

import structlog
from rich.console import Console
from rich.panel import Panel

from agentlane.harness import Agent, RunResult, RunState
from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus
from agentlane.runtime import SingleThreadedRuntimeEngine


def configure_demo_logging() -> None:
    """Mute provider debug logs so the demo transcript stays readable."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

CONSOLE = Console()


def print_intro() -> None:
    """Print the short intro panel for the live demo."""
    CONSOLE.rule("[bold cyan]AgentLane Harness Customer Support Demo[/bold cyan]")
    CONSOLE.print(
        Panel.fit(
            "This example uses real OpenAI responses through the Phase 4 harness.\n"
            "After two turns it simulates a restart and resumes from persisted RunState.\n"
            "The customer turns are injected by the demo itself. This is not yet a real user handoff flow.",
            title="What This Proves",
        )
    )


def print_restart(saved_state: RunState) -> None:
    """Print the restart checkpoint after persisting run state."""
    CONSOLE.print(
        Panel.fit(
            f"Saved RunState after {saved_state.turn_count} turns and "
            f"{len(saved_state.responses)} model responses.",
            title="Simulated Restart",
            border_style="yellow",
        )
    )


def print_summary(model_name: str, run_state: RunState) -> None:
    """Print the final run summary."""
    CONSOLE.print(
        Panel.fit(
            f"Model: {model_name}\n"
            f"Conversation resumed successfully: yes\n"
            f"Final turn count: {run_state.turn_count}\n"
            f"Accumulated raw responses: {len(run_state.responses)}",
            title="Run Summary",
            border_style="green",
        )
    )


def snapshot_run_state(agent: Agent) -> RunState:
    """Return the persisted run state from one agent instance."""
    run_state = agent.run_state
    if run_state is not None:
        return run_state
    raise RuntimeError("Expected the harness agent to expose persisted run state.")


def _require_run_result(outcome: DeliveryOutcome) -> RunResult:
    """Return the delivered run result for one runtime send outcome."""
    if outcome.status != DeliveryStatus.DELIVERED:
        message = outcome.error.message if outcome.error else "unknown delivery failure"
        raise RuntimeError(
            f"Harness example delivery failed with status={outcome.status.value}: {message}"
        )
    payload = outcome.response_payload
    if not isinstance(payload, RunResult):
        raise TypeError("Expected the harness agent to return a `RunResult`.")
    return payload


async def send_turn(
    *,
    runtime: SingleThreadedRuntimeEngine,
    agent_id: AgentId,
    customer_text: str,
    payload: str | list[object] | RunState,
) -> RunResult:
    """Send one customer turn and print the resulting assistant reply."""
    CONSOLE.print(f"[bold]Customer:[/bold] {customer_text}")
    outcome = await runtime.send_message(payload, recipient=agent_id)
    result = _require_run_result(outcome)
    CONSOLE.print(f"[bold]Assistant:[/bold] {result.final_output}")
    CONSOLE.print()
    return result
