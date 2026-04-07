"""Real OpenAI-backed example of a predefined first-class handoff."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import Agent, AgentDescriptor, Runner, RunResult
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"


async def run_demo() -> None:
    """Run the predefined handoff example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("support-agent", "handoff-returns-specialist")

    returns_specialist = AgentDescriptor(
        name="Returns Specialist",
        description="Takes over damaged-delivery and return conversations.",
        model=model,
        model_args={"reasoning_effort": "low"},
        # Handoff transfers the conversation, but the child specialist should
        # not inherit the triage agent's tool settings.
        tools=None,
        instructions=(
            "You are Acme's returns specialist. "
            "You take over damaged-delivery conversations and continue directly "
            "with the customer. Explain the next step clearly in under 80 words."
        ),
    )

    Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=AgentDescriptor(
            name="Support Triage",
            description="Frontline support triage that transfers damaged-order cases.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are Acme support triage. "
                "If the customer reports a damaged delivery or broken item, "
                "handoff to `returns_specialist` immediately. "
                "Do not answer the case yourself."
            ),
            # The base `Tools(...)` config controls tool-choice behavior.
            # The handoff target itself is added separately through
            # `handoffs=(...)`.
            tools=Tools(tools=[], tool_choice="required"),
            handoffs=(returns_specialist,),
        ),
    )

    user_message = (
        "My Acme phone arrived with a cracked screen and the shipping box was dented. "
        "What should I do next?"
    )
    outcome = await runtime.send_message(user_message, recipient=agent_id)
    await runtime.stop_when_idle()

    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"Expected delivered outcome, got {outcome.status.value}.")
    if not isinstance(outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")

    print("Example: predefined handoff")
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
