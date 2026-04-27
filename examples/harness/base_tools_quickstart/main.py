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
from agentlane.harness.tools import HarnessToolsShim, read_tool, write_tool
from agentlane.models import Config, ToolCall, Tools

MODEL_NAME = "gpt-5.4-mini"
WORKSPACE_FILE = "restock_note.md"
WORKSPACE_TEXT = """\
# Restock Note

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
        f"Create {WORKSPACE_FILE} with exactly this content, then read it back "
        f"before summarizing the restock risk in one sentence:\n\n{WORKSPACE_TEXT}"
    )

    with TemporaryDirectory() as workspace_dir:
        workspace = Path(workspace_dir)

        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Workspace Agent",
                model=model,
                model_args={"reasoning_effort": "low"},
                instructions=(
                    "You create and inspect files in a local workspace. "
                    "Call `write` to create requested files, then call `read` "
                    "before answering from the file."
                ),
                tools=Tools(
                    tools=[],
                    tool_choice="required",
                    tool_call_limits={"write": 1, "read": 1},
                ),
                shims=(
                    HarnessToolsShim(
                        (
                            read_tool(cwd=workspace),
                            write_tool(cwd=workspace),
                        )
                    ),
                ),
            )
        )

        result = await agent.run(user_prompt)
        run_state = agent.run_state
        if run_state is None:
            raise RuntimeError("Expected the default agent to persist run state.")

    tool_calls_seen: list[str] = []
    for response in run_state.responses:
        tool_calls = cast(list[ToolCall], response.choices[0].message.tool_calls or [])
        for tool_call in tool_calls:
            tool_calls_seen.append(
                f"{tool_call.function.name or 'not found'}: "
                f"{tool_call.function.arguments or 'not found'}"
            )

    tool_outputs: list[str] = []
    for item in run_state.history:
        if not isinstance(item, dict):
            continue
        message = cast(dict[str, object], item)
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        tool_outputs.append(content if isinstance(content, str) else str(content))

    print("Example: base tools quickstart")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_prompt}")
    print()
    print(f"Assistant: {result.final_output}")
    print()
    print("Tool calls:")
    for tool_call in tool_calls_seen:
        print(f"- {tool_call}")
    print()
    print("Tool results:")
    for tool_output in tool_outputs:
        print(tool_output)
    print(f"Turns completed: {run_state.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
