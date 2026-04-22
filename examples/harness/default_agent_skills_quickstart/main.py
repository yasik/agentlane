"""Real OpenAI-backed quickstart for harness skills."""

import asyncio
import logging
import os
from pathlib import Path

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.models import Config

MODEL_NAME = "gpt-5.4-mini"
SKILLS_ROOT = Path(__file__).resolve().parent / "skills"

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


class SupportAgent(DefaultAgent):
    descriptor = AgentDescriptor(
        name="Acme Support",
        model=MODEL,
        instructions=(
            "You are Acme support. Keep replies short, practical, and grounded "
            "in the available policy guidance."
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
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = SupportAgent()

    first = await agent.run("Can I still return shoes after 21 days if they are unused?")
    second = await agent.run("Please summarize the return window in one sentence.")

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the default agent to persist run state.")

    print("Example: default agent skills quickstart")
    print(f"Model: {MODEL_NAME}")
    print(f"Skills root: {SKILLS_ROOT}")
    print()
    print("The script attaches one `SkillsShim` through `AgentDescriptor.shims`.")
    print("The shim discovers local skills from an explicit filesystem loader,")
    print("merges one skills instruction block into the effective system prompt,")
    print("shows the skill catalog with name, description, and SKILL.md path,")
    print("adds `activate_skill(name: str)` to the visible tools, and keeps")
    print("activated skill state in `RunState.shim_state` for later turns.")
    print()
    print("User: Can I still return shoes after 21 days if they are unused?")
    print(f"Assistant: {first.final_output}")
    print()
    print("User: Please summarize the return window in one sentence.")
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
