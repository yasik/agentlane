"""Real OpenAI-backed quickstart for clinical-case skills and hooks."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import (
    AgentDescriptor,
    RunnerHooks,
    RunResult,
    RunState,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.models import (
    Config,
    MessageDict,
    ModelResponse,
    PromptSpec,
    ToolCall,
    get_content_or_none,
)

MODEL_NAME = "gpt-5.4-mini"
SKILLS_ROOT = Path(__file__).resolve().parent / "skills"
SKILL_NAMES = (
    "acute-chest-pain",
    "drug-reaction-triage",
    "thyrotoxicosis-patterns",
)
LOGGER = structlog.get_logger("agentlane.examples.skills")

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


def _task_name(task: Task) -> str:
    """Return a readable task label for logging."""
    return getattr(task, "name", type(task).__name__)


def _requested_skill_name(tool_call: ToolCall) -> str:
    """Return the requested skill name from one activate-skill call."""
    raw_arguments = tool_call.function.arguments or ""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments or "<unknown>"

    if not isinstance(parsed, dict):
        return "<unknown>"

    parsed_dict = cast(dict[str, object], parsed)
    name = parsed_dict.get("name")
    if isinstance(name, str) and name:
        return name

    return "<unknown>"


def _skill_activation_status(result: object) -> str:
    """Return a compact status label for one skill-activation result."""
    if not isinstance(result, str):
        return type(result).__name__
    if "<skill_content" in result:
        return "loaded"
    if "already active in this run" in result:
        return "already_active"
    if "was not found" in result:
        return "not_found"
    return "text"


def _preview_text(text: str | None, *, limit: int = 96) -> str | None:
    """Return one compact preview for logging."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _active_skill_names(run_state: RunState) -> tuple[str, ...]:
    """Return the activated skill names persisted by the default shim name."""
    raw_value = run_state.shim_state.get("skills:active-skill-names")
    if not isinstance(raw_value, list):
        return ()
    raw_items = cast(list[object], raw_value)
    return tuple(value for value in raw_items if isinstance(value, str))


def _render_system_instruction(
    instructions: str | PromptSpec[Any] | None,
) -> str:
    """Render the persisted system instruction for demo output."""
    if instructions is None:
        return "<none>"

    if isinstance(instructions, str):
        return instructions

    system_parts: list[str] = []
    for message in instructions.template.render_messages(instructions.values):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            system_parts.append(content)

    if not system_parts:
        return "<non-text system instruction>"

    return "\n\n".join(system_parts)


class SkillLifecycleLoggingHooks(RunnerHooks):
    """Structured log hook for agent lifecycle and skill activation."""

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        LOGGER.info(
            "agent_started",
            agent=_task_name(task),
            next_turn=state.turn_count + 1,
        )

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        final_output = None if result is None else str(result.final_output)
        LOGGER.info(
            "agent_finished",
            agent=_task_name(task),
            final_output=_preview_text(final_output),
        )

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        LOGGER.info(
            "llm_request_started",
            agent=_task_name(task),
            message_count=len(messages),
        )

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        LOGGER.info(
            "llm_request_finished",
            agent=_task_name(task),
            output_preview=_preview_text(get_content_or_none(response)),
        )

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        if tool_call.function.name != "activate_skill":
            return

        LOGGER.info(
            "skill_activation_started",
            agent=_task_name(task),
            skill=_requested_skill_name(tool_call),
            tool_call_id=tool_call.id,
        )

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        if tool_call.function.name != "activate_skill":
            return

        LOGGER.info(
            "skill_activation_finished",
            agent=_task_name(task),
            skill=_requested_skill_name(tool_call),
            tool_call_id=tool_call.id,
            status=_skill_activation_status(result),
        )


class ClinicalReasoningAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Clinical Case Review",
        model=MODEL,
        model_args={"reasoning_effort": "low"},
        instructions=(
            "You are a physician-facing clinical reasoning assistant. Keep "
            "replies concise, surface the highest-risk causes that must be "
            "excluded first, and suggest focused next questions or tests. "
            "Do not imply certainty when the case data is incomplete."
        ),
        shims=(
            SkillsShim(
                loader=FilesystemSkillLoader(
                    roots=(SKILLS_ROOT,),
                    include_default_roots=False,
                )
            ),
        ),
    )


async def run_demo() -> None:
    """Run the skills quickstart example."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
    )

    agent = ClinicalReasoningAgent(hooks=SkillLifecycleLoggingHooks())

    prompts = (
        "A 64-year-old has crushing substernal chest pain radiating to the left arm with diaphoresis and nausea. What are the most urgent causes and immediate next steps?",
        "A 27-year-old developed diffuse hives, wheezing, and lip swelling 30 minutes after starting amoxicillin. What is the most likely cause and what should be done first?",
        "A 33-year-old has weight loss, tremor, heat intolerance, and palpitations over two months. What unifying diagnosis fits best and what focused tests would you order next?",
    )
    print("Example: clinical case skills quickstart")
    print(f"Model: {MODEL_NAME}")
    print(f"Skills: {', '.join(SKILL_NAMES)}")
    print()
    for prompt in prompts:
        print(f"Case: {prompt}")
        result = await agent.run(prompt)
        print(f"Assistant: {result.final_output}")
        print()

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print(f"Active skills: {_active_skill_names(run_state)}")
    print()
    print("Final system instruction:")
    print(_render_system_instruction(run_state.instructions))
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
