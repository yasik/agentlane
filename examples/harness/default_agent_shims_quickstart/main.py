"""Real OpenAI-backed quickstart for custom harness shims."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.models import Config, ModelResponse

MODEL_NAME = "gpt-5.4-mini"

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


class ReplyPrefixShim(Shim):
    """Simple instruction-augmentation shim."""

    @property
    def name(self) -> str:
        return "reply-prefix"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if isinstance(turn.instructions, str):
            turn.instructions = (
                f"{turn.instructions}\n"
                "Always start every reply with `Support:`."
            )


class TurnCounterShim(Shim):
    """Persist how many model turns have completed for this conversation."""

    @property
    def name(self) -> str:
        return "turn-counter"

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        del response
        count = int(turn.run_state.shim_state.get("completed-turns", 0))
        turn.run_state.shim_state["completed-turns"] = count + 1


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Acme Support",
        model=MODEL,
        instructions=(
            "You are Acme support. Keep replies short and practical. "
            "Do not use bullet points."
        ),
        shims=(ReplyPrefixShim(), TurnCounterShim()),
    )


async def run_demo() -> None:
    """Run the shim quickstart example."""
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

    print("Example: default agent shims quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print("The script adds two custom shims through `AgentDescriptor.shims`:")
    print("1. one shim appends an extra instruction line before each turn")
    print("2. one shim persists its own counter in `RunState.shim_state`")
    print()
    print("User: My order arrived damaged. What should I do first?")
    print(f"Assistant: {first.final_output}")
    print()
    print("User: Please summarize the next step in one sentence.")
    print(f"Assistant: {second.final_output}")
    print()
    print(f"Shim state: {run_state.shim_state}")
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
