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
    """Typed values used to render the support instructions."""

    company_name: str
    tone: str


INSTRUCTIONS_TEMPLATE = PromptTemplate[InstructionValues, str](
    system_template="""
You are {{ company_name }}'s support assistant.
Use a {{ tone }} tone.
Keep replies concise and practical.
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


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Acme Support",
        model=MODEL,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=INSTRUCTIONS_TEMPLATE,
            values={
                "company_name": "Acme Devices",
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

    agent = SupportAgent()

    first = await agent.run("My order arrived damaged. What should I do first?")
    second = await agent.run("Please summarize the next step in one sentence.")

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print("Example: default agent quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print("The script creates one agent class and calls `run(...)` directly.")
    print("No runtime, runner, agent id, or send_message wiring is needed here.")
    print()
    print("User: My order arrived damaged. What should I do first?")
    print(f"Assistant: {first.final_output}")
    print()
    print("User: Please summarize the next step in one sentence.")
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
