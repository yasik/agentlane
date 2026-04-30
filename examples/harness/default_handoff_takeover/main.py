"""Real OpenAI-backed example of the generic `handoff` transfer path."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor, DefaultHandoff, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Tools

MODEL_NAME = "gpt-5.4-mini"


async def run_demo() -> None:
    """Run the default handoff example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=AgentDescriptor(
            name="Patient Triage",
            description="Frontline patient triage that can handoff to a fresh helper.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are patient-message triage. "
                "For urgent symptom cases, call `handoff` immediately and let the "
                "fresh specialist take over. Do not answer the case yourself."
            ),
            tools=Tools(tools=[], tool_choice="required"),
            default_handoff=DefaultHandoff(
                model=model,
                tools=None,
                instructions=(
                    "You are the clinical escalation specialist who takes over urgent "
                    "patient messages. Read the transferred conversation and continue "
                    "directly with the patient. Do not diagnose. Give concrete next "
                    "steps in under 80 words."
                ),
            ),
        ),
    )

    user_message = (
        "I feel faint and my smartwatch says my heart rate is 145 while resting. "
        "I need to know the safest next step."
    )
    result = await agent.run(user_message)

    print("Example: default handoff")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {result.final_output}")
    print()
    print(f"Turns completed: {result.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
