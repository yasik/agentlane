"""OpenAI-backed quickstart for the first-party plan tool."""

import asyncio
import json
import logging
import os

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.tools import HarnessToolsShim, plan_tool
from agentlane.models import Config, Tools

MODEL_NAME = "gpt-5.4-mini"


async def run_demo() -> None:
    """Run the plan tool quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Planner",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You are a concise planning assistant. Use the plan tool whenever "
                "the user asks you to create or update tracked work."
            ),
            tools=Tools(
                tools=[],
                tool_choice="required",
                tool_call_limits={"write_plan": 1},
            ),
            shims=(HarnessToolsShim((plan_tool(),)),),
        )
    )

    first_prompt = "Create a three-step plan for adding a small documentation page."
    first = await agent.run(first_prompt)

    second_prompt = (
        "Update the plan: mark the first step complete and make the second "
        "step in progress."
    )
    second = await agent.run(second_prompt)

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print("Example: harness base tools plan quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {first_prompt}")
    print(f"Assistant: {first.final_output}")
    print()
    print(f"User: {second_prompt}")
    print(f"Assistant: {second.final_output}")
    print()
    print("Persisted plan state:")
    print(json.dumps(dict(run_state.shim_state).get("harness-tools:plan"), indent=2))


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
