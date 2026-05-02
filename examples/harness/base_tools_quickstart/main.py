"""Real OpenAI-backed streaming quickstart for first-party base harness tools."""

import asyncio
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor, RunnerHooks, Task
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
from agentlane.models import (
    Config,
    ModelResponse,
    ModelStreamEventKind,
    ToolCall,
    Tools,
)

MODEL_NAME = "gpt-5.4-mini"
WORKSPACE_FILE = "portfolio_risk_note.md"
CHECKLIST_FILE = "risk_controls_checklist.md"
WORKSPACE_TEXT = """\
# Portfolio Risk Note

- Portfolio: PM-17 Growth
- Semiconductor exposure: 42%
- Leveraged ETF sleeve: 12%
- Cash position: 6%
TODO: confirm portfolio manager approval before adding more sector exposure.
"""
PATCHED_LINE = "Action: manager approval required before adding more sector exposure."
CHECKLIST_TEXT = """\
# Risk Controls Checklist

- Confirm sector exposure threshold.
- Confirm leveraged ETF sleeve threshold.
- Record approval owner before any new purchase.
- Archive final note in the review workspace.
"""
MAX_CONSOLE_RESULT_CHARS = 1400


class ToolTraceHooks(RunnerHooks):
    """Print tool execution while the streamed run is active."""

    async def on_tool_call_start(self, task: Task, tool_call: ToolCall) -> None:
        del task
        tool_name = tool_call.function.name or "unknown"
        print()
        print(f"[tool start] {tool_name}")
        print(_indent(_format_tool_arguments(tool_call.function.arguments)))

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        del task
        tool_name = tool_call.function.name or "unknown"
        print()
        print(f"[tool result] {tool_name}")
        print(_indent(_truncate_console_text(str(result))))


def _format_tool_arguments(arguments: str | None) -> str:
    """Pretty-print one tool call's JSON arguments."""
    if not arguments:
        return "{}"
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(parsed_arguments, indent=2, sort_keys=True)


def _truncate_console_text(text: str) -> str:
    """Keep streamed console output readable for large tool results."""
    if len(text) <= MAX_CONSOLE_RESULT_CHARS:
        return text
    trimmed = text[:MAX_CONSOLE_RESULT_CHARS].rstrip()
    return f"{trimmed}\n[console output truncated]"


def _indent(text: str) -> str:
    """Indent a multi-line block for console display."""
    lines = text.splitlines() or [""]
    return "\n".join(f"  {line}" for line in lines)


def _tool_path(responses: list[ModelResponse]) -> list[str]:
    """Extract the completed tool-call path from raw model responses."""
    tool_names: list[str] = []
    for response in responses:
        tool_calls = cast(list[ToolCall], response.choices[0].message.tool_calls or [])
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name:
                tool_names.append(tool_name)
    return tool_names


async def run_demo() -> None:
    """Run the base tools quickstart example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    user_prompt = (
        "Build a tiny portfolio risk review workspace. Follow this exact tool "
        "sequence: "
        f"1. Create `{WORKSPACE_FILE}` with the portfolio note below. "
        f"2. Create `{CHECKLIST_FILE}` with the checklist below. "
        f"3. Use patch to replace the TODO line in `{WORKSPACE_FILE}` with "
        f"`{PATCHED_LINE}`. "
        "4. Use find to locate the Markdown files. "
        "5. Use grep to confirm the Action line. "
        "6. Use bash for one non-interactive shell inspection that prints the "
        "workspace path, sorted Markdown filenames, and line counts. "
        "7. Read both Markdown files back. "
        "Then answer with three bullets: files created, required approval, and "
        "workspace verification.\n\n"
        f"Portfolio note content:\n{WORKSPACE_TEXT}\n"
        f"Checklist content:\n{CHECKLIST_TEXT}"
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
                    "Follow the user's requested base-tool sequence exactly. "
                    "Call `write` for new files, `patch` for precise edits, "
                    "`find` to locate files, `grep` to confirm exact text, "
                    "`bash` only for simple non-interactive workspace shell "
                    "inspection, and `read` before answering from file facts."
                ),
                tools=Tools(
                    tools=[],
                    tool_choice="required",
                    parallel_tool_calls=False,
                    tool_call_limits={
                        "write": 2,
                        "patch": 1,
                        "find": 1,
                        "grep": 1,
                        "bash": 1,
                        "read": 2,
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
            ),
            hooks=ToolTraceHooks(),
        )

        stream = await agent.run_stream(user_prompt)

        print("Example: base tools streaming quickstart")
        print(f"Model: {MODEL_NAME}")
        print(f"Workspace: {workspace}")
        print()
        print("This run streams model text and tool argument deltas, while hooks")
        print("print each base tool start/result as it executes.")
        print()
        print(f"User: {user_prompt}")
        print("Assistant/tool stream:")

        last_openai_phase: str | None = None
        saw_reasoning = False
        async for event in stream:
            if (
                event.kind == ModelStreamEventKind.REASONING
                and event.reasoning is not None
            ):
                reasoning_value = event.reasoning
                reasoning_text = (
                    reasoning_value
                    if isinstance(reasoning_value, str)
                    else str(reasoning_value)
                )
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
                print(f"[tool arguments delta] {event.arguments_delta}")

        result = await stream.result()
        run_state = agent.run_state
        if run_state is None:
            raise RuntimeError("Expected the default agent to persist run state.")

    print()
    print()
    print(f"Final output: {result.final_output}")
    tool_path = " -> ".join(_tool_path(run_state.responses)) or "not found"
    print(f"Tool path: {tool_path}")
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
