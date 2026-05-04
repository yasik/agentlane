"""Real OpenAI-backed example of the generic `agent` tool."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import (
    AgentDescriptor,
    Runner,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.tools import HarnessToolsShim, agent_tool
from agentlane.models import Config, Tools

MODEL_NAME = "gpt-5.4-mini"

logging.basicConfig(level=logging.WARNING)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)


async def run_demo() -> None:
    """Run the generic `agent` tool example."""
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=AgentDescriptor(
            name="Portfolio Operations Manager",
            description="Finance agent that can spawn one focused helper agent.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You help portfolio operations teams. "
                "When the user asks for one focused artifact, call `agent` once. "
                "Pass a one-word helper name and a specific task. "
                "After the helper returns, present the final result clearly."
            ),
            tools=Tools(
                tools=[],
                tool_choice="required",
            ),
            shims=(HarnessToolsShim((agent_tool(model=model),)),),
        ),
    )

    user_message = (
        "Write a short internal pre-trade review note for portfolio PM-17. "
        "The proposed rebalance adds semiconductor exposure, increases cash drag, "
        "and needs a compliance-ready summary."
    )
    result = await agent.run(user_message)

    print("Example: generic `agent` tool")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {result.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
