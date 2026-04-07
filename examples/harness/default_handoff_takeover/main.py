"""Real OpenAI-backed example of the generic `handoff` transfer path."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import Agent, AgentDescriptor, DefaultHandoff, Runner, RunResult
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"


async def run_demo() -> None:
    """Run the default handoff example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("support-agent", "default-handoff-takeover")

    Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=AgentDescriptor(
            name="Support Triage",
            description="Frontline support triage that can handoff to a fresh helper.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are Acme support triage. "
                "For damaged-delivery cases, call `handoff` immediately and let the "
                "fresh specialist take over. Do not answer the case yourself."
            ),
            tools=Tools(tools=[], tool_choice="required"),
            default_handoff=DefaultHandoff(
                model=model,
                tools=None,
                instructions=(
                    "You are the specialist who takes over unresolved Acme support "
                    "cases. Read the transferred conversation and continue directly "
                    "with the customer. Give concrete next steps in under 80 words."
                ),
            ),
        ),
    )

    user_message = (
        "My Acme tablet arrived today with a cracked corner and I need to know "
        "the fastest next step."
    )
    outcome = await runtime.send_message(user_message, recipient=agent_id)
    await runtime.stop_when_idle()

    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"Expected delivered outcome, got {outcome.status.value}.")
    if not isinstance(outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")

    print("Example: default handoff")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {outcome.response_payload.final_output}")
    print()
    print(f"Turns completed: {outcome.response_payload.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
