"""Delegation helpers shared by handoffs and agent-as-tool execution."""

import json
import re
from dataclasses import asdict, is_dataclass
from uuid import uuid4

from pydantic import BaseModel, Field

from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus

from ._run import RunResult


class DelegatedTaskInput(BaseModel):
    """Structured input passed to delegated agents."""

    task: str | None = Field(
        default=None,
        description="Optional focused task or delegation preamble for the delegated agent.",
    )


class DefaultAgentToolInput(BaseModel):
    """Structured input for one generic spawned agent-as-tool call."""

    name: str = Field(
        min_length=1,
        pattern=r"^\S+$",
        description=(
            "Single-word name for logging and tracing the delegated helper "
            "agent. The name may be task-relevant or random."
        ),
    )
    task: str = Field(
        min_length=1,
        description=(
            "Complete delegated task instruction, including necessary context "
            "and expected output."
        ),
    )


def normalize_delegation_tool_name(value: str) -> str:
    """Return a tool-safe name derived from one human-readable label."""
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_").lower()
    if normalized:
        return normalized
    return "delegate"


def agent_tool_description(
    agent_name: str,
    description: str | None,
) -> str:
    """Return the default description for one agent-as-tool wrapper."""
    if description:
        return description
    return f"Delegate a focused task to {agent_name} and continue with the result."


def handoff_description(
    agent_name: str,
    description: str | None,
) -> str:
    """Return the default description for one first-class handoff tool."""
    if description:
        return description
    return f"Transfer the conversation to {agent_name}."


def default_handoff_task_message() -> str:
    """Return the fallback user-side delegation message for handoff transfer."""
    return (
        "You are being delegated this conversation. Continue from the "
        "transferred history and take over from here."
    )


def default_handoff_tool_result(tool_name: str) -> str:
    """Return the synthetic tool result recorded for an intercepted handoff."""
    return f"Handoff accepted by {tool_name}. Control transfers now."


def default_agent_tool_instructions() -> str:
    """Return the default prompt for one generic spawned agent-as-tool call."""
    return (
        "You are a newly spawned agent in a team of agents collaborating to "
        "complete a task. You can spawn sub-agents to handle subtasks, and "
        "those sub-agents can spawn their own sub-agents. Return the response "
        "to your assigned task directly; that response will be delivered back "
        "to your parent agent. Treat the next user message as your assigned "
        "task, and use any available prior history only as background context."
    )


def delegated_agent_type(
    owner_id: AgentId,
    tool_name: str,
    *,
    kind: str,
) -> str:
    """Return the runtime type shared by one delegated agent family."""
    return (
        f"{owner_id.type.value}.{owner_id.key.value}.{kind}."
        f"{normalize_delegation_tool_name(tool_name)}"
    )


def delegated_agent_id(
    owner_id: AgentId,
    tool_name: str,
    *,
    kind: str,
) -> AgentId:
    """Return one fresh delegated child id for the given tool invocation."""
    return AgentId.from_values(
        delegated_agent_type(owner_id, tool_name, kind=kind),
        str(uuid4()),
    )


def delegated_result_text(outcome: DeliveryOutcome) -> str:
    """Render one delegated delivery outcome into tool-result text.

    Agent-as-tool delegation behaves like a subroutine call, so failures are
    converted into text and fed back to the caller model as a normal tool
    result. That gives the model a chance to recover or choose a different
    strategy on the next turn.
    """
    if outcome.status != DeliveryStatus.DELIVERED:
        return "Error: delegated agent call failed."

    payload = outcome.response_payload
    if isinstance(payload, RunResult):
        return _delegated_value_as_text(payload.final_output)
    if payload is None:
        return ""
    return _delegated_value_as_text(payload)


def require_handoff_result(outcome: DeliveryOutcome) -> RunResult:
    """Return the delivered delegated run result or raise for transfer failure."""
    if outcome.status != DeliveryStatus.DELIVERED:
        message = outcome.error.message if outcome.error else "unknown delivery failure"
        raise RuntimeError(
            "Handoff delivery failed with " f"status={outcome.status.value}: {message}"
        )

    payload = outcome.response_payload
    if not isinstance(payload, RunResult):
        raise RuntimeError(
            "Handoff delivery returned an unexpected payload type: "
            f"{type(payload).__name__}."
        )
    return payload


def _delegated_value_as_text(value: object) -> str:
    """Render structured delegated results into stable tool-result text."""
    if isinstance(value, str):
        return value

    # Delegated sub-agents should preserve structured outputs instead of
    # collapsing immediately to repr-style text.
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(asdict(value))
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)
