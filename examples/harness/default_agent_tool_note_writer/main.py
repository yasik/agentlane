"""Real OpenAI-backed example of the generic `agent` tool."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    DefaultAgentTool,
    Runner,
    RunResult,
)
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"

logging.basicConfig(level=logging.WARNING)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)


async def run_demo() -> None:
    """Run the generic `agent` tool example."""
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("support-agent", "default-agent-tool-note-writer")

    Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=AgentDescriptor(
            name="Support Manager",
            description="Support agent that can spawn one focused helper agent.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You help Acme support agents. "
                "When the user asks for one focused artifact, call `agent` once. "
                "Pass a short helper name, an optional description, and a specific task. "
                "After the helper returns, present the final result clearly."
            ),
            tools=Tools(
                tools=[DefaultAgentTool(model=model, tools=None)],
                tool_choice="required",
                tool_call_limits={"agent": 1},
            ),
        ),
    )

    user_message = (
        "Write a short internal escalation note for order 88421. "
        "The phone arrived with a cracked screen, the shipping box was dented, "
        "and the customer can provide photos."
    )
    outcome = await runtime.send_message(user_message, recipient=agent_id)
    await runtime.stop_when_idle()

    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"Expected delivered outcome, got {outcome.status.value}.")
    if not isinstance(outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")

    print("Example: generic `agent` tool")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {outcome.response_payload.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
