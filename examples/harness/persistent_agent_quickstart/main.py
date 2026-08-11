"""Path-backed persistent DefaultAgent quickstart."""

import argparse
import asyncio
import logging
import os
from pathlib import Path

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.models import Config

MODEL_NAME = "gpt-5.4-mini"
DEFAULT_STATE_PATH = Path(".agentlane/persistent-assistant.json")


async def run_demo(*, prompt: str, state_path: Path) -> None:
    """Continue one persistent conversation and print its committed revision."""
    model = ResponsesClient(
        config=Config(
            api_key=os.environ["OPENAI_API_KEY"],
            model=MODEL_NAME,
        )
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Persistent Assistant",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are a persistent assistant. Remember relevant facts from "
                "earlier turns and answer concisely."
            ),
        ),
        agent_id=AgentId.from_values("persistent-assistant", "main"),
        state_path=state_path,
    )

    result = await agent.run(prompt)
    state = agent.run_state
    if state is None:
        raise RuntimeError("Expected a committed persistent state.")

    print(f"Assistant: {result.final_output}")
    print()
    print(f"Address: {agent.agent_id}")
    print(f"Revision: {state.revision}")
    print(f"State: {state_path}")


def main() -> None:
    """Parse arguments and run the persistent agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="New prompt for the persistent agent.")
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Snapshot path (default: {DEFAULT_STATE_PATH}).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    asyncio.run(run_demo(prompt=args.prompt, state_path=args.state))


if __name__ == "__main__":
    main()
