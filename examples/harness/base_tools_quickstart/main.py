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
from agentlane.harness.tools import (
    HarnessToolsShim,
    bash_tool,
    find_tool,
    grep_tool,
    patch_tool,
    read_tool,
    write_tool,
)
from agentlane.models import Config, ToolCall, Tools

MODEL_NAME = "gpt-5.4-mini"
WORKSPACE_FILE = "portfolio_risk_note.md"
WORKSPACE_TEXT = """\
# Portfolio Risk Note

- Portfolio: PM-17 Growth
- Semiconductor exposure: 42%
- Leveraged ETF sleeve: 12%
- Cash position: 6%
TODO: confirm portfolio manager approval before adding more sector exposure.
"""
PATCHED_LINE = "Action: manager approval required before adding more sector exposure."


async def run_demo() -> None:
    """Run the base tools quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    user_prompt = (
        f"Create {WORKSPACE_FILE} with exactly this content, use patch to "
        f"replace the TODO line with `{PATCHED_LINE}`, locate the file with "
        "find, use grep to confirm the Action line, list the workspace with "
        "bash, read it back, and "
        "summarize the portfolio risk in one sentence:\n\n"
        f"{WORKSPACE_TEXT}"
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
                    "Call `write` to create requested files, `patch` for "
                    "precise edits to existing files, `find` to locate them, "
                    "`grep` to search inside text, `bash` only for simple "
                    "workspace shell inspection, and `read` before answering "
                    "from the file."
                ),
                tools=Tools(
                    tools=[],
                    tool_choice="required",
                    tool_call_limits={
                        "write": 1,
                        "patch": 1,
                        "find": 1,
                        "grep": 1,
                        "bash": 1,
                        "read": 1,
                    },
                ),
                shims=(
                    HarnessToolsShim(
                        (
                            write_tool(cwd=workspace),
                            patch_tool(cwd=workspace),
                            find_tool(cwd=workspace),
                            grep_tool(cwd=workspace),
                            bash_tool(cwd=workspace, default_timeout=5),
                            read_tool(cwd=workspace),
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
        tool_name = message.get("name")
        content = message.get("content")
        tool_output = content if isinstance(content, str) else str(content)
        tool_outputs.append(f"{tool_name}: {tool_output}")

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
    print("\n\n".join(tool_outputs) if tool_outputs else "not found")
    print(f"Turns completed: {run_state.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
