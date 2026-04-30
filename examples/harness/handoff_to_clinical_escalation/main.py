"""Real OpenAI-backed example of a predefined clinical handoff."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Tools

MODEL_NAME = "gpt-5.4-mini"


async def run_demo() -> None:
    """Run the predefined handoff example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    nurse_triage_specialist = AgentDescriptor(
        name="Nurse Triage Specialist",
        description="Takes over patient messages with urgent clinical symptoms.",
        model=model,
        model_args={"reasoning_effort": "low"},
        # Handoff transfers the conversation, but the child specialist should
        # not inherit the triage agent's tool settings.
        tools=None,
        instructions=(
            "You are a nurse triage specialist. "
            "You take over patient messages with urgent symptoms and continue "
            "directly with the patient. Do not diagnose. Explain the next safe "
            "step clearly in under 80 words."
        ),
    )

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=AgentDescriptor(
            name="Patient Triage",
            description="Frontline patient triage that transfers urgent symptom cases.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are patient-message triage. "
                "If the patient reports chest pain, fainting, stroke symptoms, "
                "or severe shortness of breath, handoff to `nurse_triage_specialist` "
                "immediately. "
                "Do not answer the case yourself."
            ),
            # The base `Tools(...)` config controls tool-choice behavior.
            # The handoff target itself is added separately through
            # `handoffs=(...)`.
            tools=Tools(tools=[], tool_choice="required"),
            handoffs=(nurse_triage_specialist,),
        ),
    )

    user_message = (
        "I have chest pressure and shortness of breath that started 20 minutes ago. "
        "What should I do next?"
    )
    result = await agent.run(user_message)

    print("Example: predefined handoff")
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
