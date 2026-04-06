"""Real OpenAI-backed harness demo showing the Phase 5 tool loop."""

import asyncio
from pathlib import Path
from typing import TypedDict

from _demo_utils import (
    configure_demo_logging,
    load_openai_api_key,
    print_intro,
    print_summary,
    print_turn,
    require_run_result,
)
from agentlane_openai import ResponsesClient
from pydantic import BaseModel

from agentlane.harness import Agent, AgentDescriptor, Runner
from agentlane.messaging import AgentId
from agentlane.models import (
    Config,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    Tool,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
MOCK_SEARCH_RESULT = (
    "Acme Help Center: Opened laptops may be returned within 30 days of delivery. "
    "Standard returns do not cover devices with accidental damage."
)


class AssistantInstructionValues(TypedDict):
    """Typed values used to render the policy assistant instructions."""

    company_name: str
    knowledge_source: str
    tone: str


class SearchHelpCenterArgs(BaseModel):
    """Arguments for the mocked search tool."""

    question: str


INSTRUCTIONS_TEMPLATE = PromptTemplate[AssistantInstructionValues, str](
    system_template="""
You are {{ company_name }}'s policy assistant.
Use a {{ tone }} tone.

When the user asks about policy, returns, warranties, or shipping, you must call
`search_help_center` before answering. Answer only from the returned {{ knowledge_source }}
result and keep the answer under 80 words.
""".strip(),
    user_template=None,
    output_schema=OutputSchema(str),
)


async def search_help_center(
    args: SearchHelpCenterArgs,
    cancellation_token: CancellationToken,
) -> str:
    """Return one mocked help-center result string."""
    _ = args
    _ = cancellation_token
    return MOCK_SEARCH_RESULT


def build_descriptor(api_key: str) -> AgentDescriptor:
    """Construct one policy-assistant descriptor for the live demo."""
    instruction_values: AssistantInstructionValues = {
        "company_name": "Acme",
        "knowledge_source": "help-center",
        "tone": "clear and practical",
    }
    search_tool = Tool(
        name="search_help_center",
        description="Search the Acme help center for the current policy answer.",
        args_model=SearchHelpCenterArgs,
        handler=search_help_center,
    )
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    return AgentDescriptor(
        name="Acme Policy Assistant",
        description="Answers customer policy questions with a search tool",
        model=model,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=INSTRUCTIONS_TEMPLATE,
            values=instruction_values,
        ),
        # Require one search call so the demo always exercises the tool path.
        # After that first call, the runner removes the tool for the answer turn.
        tools=Tools(
            tools=[search_tool],
            tool_choice="required",
            tool_call_limits={"search_help_center": 1},
        ),
    )


async def run_demo() -> None:
    """Run the real tool-calling demo."""
    configure_demo_logging()
    api_key = load_openai_api_key(ENV_FILE)
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("policy-agent", "tool-demo")
    descriptor = build_descriptor(api_key)

    print_intro()

    agent = Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=descriptor,
    )
    question = (
        "If I opened my Acme UltraBook yesterday, can I still return it next week?"
    )

    outcome = await runtime.send_message(question, recipient=agent_id)
    result = require_run_result(outcome)
    print_turn(question, result.final_output)

    run_state = agent.run_state
    await runtime.stop_when_idle()
    if run_state is None:
        raise RuntimeError("Expected the harness agent to expose persisted run state.")
    print_summary(MODEL_NAME, run_state)


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
