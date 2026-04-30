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
        # Bootstrap-time system-prompt augmentation is explicit. After the run
        # starts, the persisted system instruction remains stable unless a shim
        # deliberately changes it again.
        if turn.run_state.turn_count == 1:
            turn.append_system_instruction(
                "Always start every reply with `Care note:`.",
                separator="\n",
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
        await turn.run_state.shim_state.increment("completed-turns")


class ClinicianAssistant(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Clinician Assistant",
        model=MODEL,
        instructions=(
            "You help clinicians prepare concise follow-up guidance from patient "
            "messages. Keep replies short and practical. Do not diagnose, and do "
            "not use bullet points."
        ),
        shims=(ReplyPrefixShim(), TurnCounterShim()),
    )


async def run_demo() -> None:
    """Run the shim quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = ClinicianAssistant()

    first_question = (
        "A patient on warfarin reports two missed INR checks and new bruising. "
        "What follow-up guidance should we send?"
    )
    second_question = "Please summarize the next staff action in one sentence."
    first = await agent.run(first_question)
    second = await agent.run(second_question)

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print("Example: default agent shims quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print("The script adds two custom shims through `AgentDescriptor.shims`:")
    print("1. one shim appends an extra instruction line before each turn request")
    print(
        "2. one shim updates a persisted counter through the `ShimState` helper"
        " methods at `RunState.shim_state`"
    )
    print()
    print(f"User: {first_question}")
    print(f"Assistant: {first.final_output}")
    print()
    print(f"User: {second_question}")
    print(f"Assistant: {second.final_output}")
    print()
    print(f"Shim state: {dict(run_state.shim_state)}")
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
