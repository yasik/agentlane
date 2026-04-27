"""Real OpenAI-backed quickstart for first-party base harness tools."""

import asyncio
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.tools import HarnessToolsShim, read_tool
from agentlane.models import Config, ToolCall, Tools

MODEL_NAME = "gpt-5.4-mini"
WORKSPACE_FILE = "inventory_note.txt"
WORKSPACE_TEXT = """\
Restock note:
- Product: Acme Field Charger
- On-hand units: 8
- Average daily orders: 6
- Supplier lead time: 5 days
"""


async def run_demo() -> None:
    """Run the base tools quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    user_prompt = (
        f"Read {WORKSPACE_FILE} and summarize the restock risk in one sentence."
    )

    with TemporaryDirectory() as workspace_dir:
        workspace = Path(workspace_dir)
        (workspace / WORKSPACE_FILE).write_text(WORKSPACE_TEXT, encoding="utf-8")

        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Workspace Reader",
                model=model,
                model_args={"reasoning_effort": "low"},
                instructions=(
                    "You inspect files in a local workspace. "
                    "Call `read` before answering and answer only from the file."
                ),
                tools=Tools(
                    tools=[],
                    tool_choice="required",
                    tool_call_limits={"read": 1},
                ),
                shims=(HarnessToolsShim((read_tool(cwd=workspace),)),),
            )
        )

        result = await agent.run(user_prompt)
        run_state = agent.run_state
        if run_state is None:
            raise RuntimeError("Expected the default agent to persist run state.")

    tool_name = "not found"
    tool_arguments = "not found"
    for response in run_state.responses:
        tool_calls = cast(list[ToolCall], response.choices[0].message.tool_calls or [])
        if not tool_calls:
            continue
        tool_name = tool_calls[0].function.name or "not found"
        tool_arguments = tool_calls[0].function.arguments or "not found"
        break

    tool_output = "not found"
    for item in run_state.history:
        if not isinstance(item, dict):
            continue
        message = cast(dict[str, object], item)
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        tool_output = content if isinstance(content, str) else str(content)
        break

    print("Example: base tools quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_prompt}")
    print()
    print(f"Assistant: {result.final_output}")
    print()
    print(f"Tool called: {tool_name}")
    print(f"Tool arguments: {tool_arguments}")
    print(f"Tool result: {tool_output}")
    print(f"Turns completed: {run_state.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
