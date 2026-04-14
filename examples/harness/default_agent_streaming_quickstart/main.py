"""Real OpenAI-backed quickstart for DefaultAgent.run_stream(...)."""

import asyncio
import logging
import os
from typing import TypedDict

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import (
    Config,
    ModelStreamEventKind,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    Tools,
    as_tool,
)

MODEL_NAME = "gpt-5.4-mini"


class InstructionValues(TypedDict):
    """Typed values used to render the support instructions."""

    company_name: str


INSTRUCTIONS_TEMPLATE = PromptTemplate[InstructionValues, str](
    system_template="""
You are {{ company_name }} support.
Before any tool call, first tell the user in one short two-step preamble what
you are about to check. Then call `search_returns_policy` exactly once.
After the tool result arrives, answer in at most two bullet points.
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


@as_tool
async def search_returns_policy(question: str) -> str:
    """Look up the current Acme returns policy."""
    del question
    return (
        "Acme Returns Policy: Opened laptops may be returned within 30 days only "
        "when they are in resellable condition. Accidental damage, including "
        "cracked screens, is not covered by the standard return window."
    )


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Acme Support",
        model=MODEL,
        model_args={"reasoning": {"effort": "medium", "summary": "detailed"}},
        instructions=PromptSpec(
            template=INSTRUCTIONS_TEMPLATE,
            values={"company_name": "Acme Devices"},
        ),
        tools=Tools(
            tools=[search_returns_policy],
            parallel_tool_calls=False,
        ),
    )


async def run_demo() -> None:
    """Run the streaming quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = SupportAgent()
    stream = await agent.run_stream(
        "Can I return an opened laptop if the screen is cracked?"
    )

    print("Example: default agent streaming quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print("The script creates one agent class and calls `run_stream(...)` directly.")
    print("No manual runtime, runner, agent id, or send_message wiring is needed.")
    print("The stream reuses `ModelStreamEvent` directly, so provider-native")
    print("reasoning summaries, when available, and preamble/phase details come")
    print("through unchanged.")
    print()
    print("User: Can I return an opened laptop if the screen is cracked?")
    print("Assistant stream:")

    last_openai_phase: str | None = None
    saw_reasoning = False
    async for event in stream:
        if event.kind == ModelStreamEventKind.REASONING and event.reasoning is not None:
            reasoning_value = event.reasoning
            if isinstance(reasoning_value, str):
                reasoning_text = reasoning_value
            else:
                reasoning_text = str(reasoning_value)

            if reasoning_text.strip():
                if not saw_reasoning:
                    print()
                    print("[reasoning]")
                    saw_reasoning = True
                print(reasoning_text, end="", flush=True)
            continue

        if (
            event.kind == ModelStreamEventKind.PROVIDER
            and event.provider_event_type is not None
        ):
            raw_item = getattr(event.raw, "item", None)
            raw_phase = getattr(raw_item, "phase", None)
            if isinstance(raw_phase, str) and raw_phase != last_openai_phase:
                last_openai_phase = raw_phase
                print()
                print(f"[openai phase] {raw_phase}")
            continue

        if event.kind == ModelStreamEventKind.TEXT_DELTA and event.text:
            print(event.text, end="", flush=True)
            continue
        if (
            event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA
            and event.arguments_delta
        ):
            print()
            print(f"[tool arguments] {event.arguments_delta}")

    result = await stream.result()
    print()
    print()
    print(f"Final output: {result.final_output}")

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

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
