"""Real OpenAI-backed quickstart for the ergonomic default agent layer."""

import asyncio
import logging
import os
from typing import TypedDict

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, OutputSchema, PromptSpec, PromptTemplate

MODEL_NAME = "gpt-5.4-mini"


class InstructionValues(TypedDict):
    """Typed values used to render the patient intake instructions."""

    clinic_name: str
    tone: str


INSTRUCTIONS_TEMPLATE = PromptTemplate[InstructionValues, str](
    system_template="""
You are {{ clinic_name }}'s patient intake assistant.
Use a {{ tone }} tone.
Keep replies concise and practical. Do not diagnose. Identify urgent red flags
and tell the patient when to contact the care team or seek emergency care.
""".strip(),
    user_template=None,
    output_schema=OutputSchema(str),
)

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


class PatientIntakeAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Patient Intake",
        model=MODEL,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=INSTRUCTIONS_TEMPLATE,
            values={
                "clinic_name": "Harborview Cardiology",
                "tone": "clear and calm",
            },
        ),
    )


async def run_demo() -> None:
    """Run the minimal default-agent example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = PatientIntakeAgent()

    first_question = (
        "I started a new blood pressure medicine and felt lightheaded this morning. "
        "What should I do first?"
    )
    second_question = "Please summarize the next step in one sentence."
    first = await agent.run(first_question)
    second = await agent.run(second_question)

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print("Example: default agent quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print("The script creates one agent class and calls `run(...)` directly.")
    print("No runtime, runner, agent id, or send_message wiring is needed here.")
    print()
    print(f"User: {first_question}")
    print(f"Assistant: {first.final_output}")
    print()
    print(f"User: {second_question}")
    print(f"Assistant: {second.final_output}")
    print()
    print(
        "Run summary:"
        f" {run_state.turn_count} turns,"
        f" {len(run_state.responses)} raw model responses."
    )


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
