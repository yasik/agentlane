"""Real OpenAI-backed harness demo showing the tool loop."""

import asyncio
import logging
import os
from typing import TypedDict, cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import Agent, AgentDescriptor, Runner, RunResult
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import (
    Config,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    ToolCall,
    Tools,
    as_tool,
)
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"
MOCK_SEARCH_RESULT = (
    "Acme Help Center: Opened laptops may be returned within 30 days of delivery. "
    "Standard returns do not cover devices with accidental damage."
)


class AssistantInstructionValues(TypedDict):
    """Typed values used to render the policy assistant instructions."""

    company_name: str
    knowledge_source: str
    tone: str


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


@as_tool
async def search_help_center(question: str) -> str:
    """Search the Acme help center for the current policy answer."""
    del question
    return MOCK_SEARCH_RESULT


async def run_demo() -> None:
    """Run the real tool-calling demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("policy-agent", "tool-demo")
    instruction_values: AssistantInstructionValues = {
        "company_name": "Acme",
        "knowledge_source": "help-center",
        "tone": "clear and practical",
    }
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    agent = Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=AgentDescriptor(
            name="Acme Policy Assistant",
            description="Answers customer policy questions with a search tool",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=PromptSpec(
                template=INSTRUCTIONS_TEMPLATE,
                values=instruction_values,
            ),
            # Require one search call so the example always exercises the tool
            # loop before the assistant answers.
            tools=Tools(
                tools=[search_help_center],
                tool_choice="required",
                tool_call_limits={"search_help_center": 1},
            ),
        ),
    )
    question = (
        "If I opened my Acme UltraBook yesterday, can I still return it next week?"
    )

    outcome = await runtime.send_message(question, recipient=agent_id)
    await runtime.stop_when_idle()

    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"Expected delivered outcome, got {outcome.status.value}.")
    if not isinstance(outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")

    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the harness agent to expose persisted run state.")

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

    print("Example: tool-calling search answer")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {question}")
    print()
    print(f"Assistant: {outcome.response_payload.final_output}")
    print()
    print(f"Tool called: {tool_name}")
    print(f"Tool arguments: {tool_arguments}")
    print(f"Mocked tool result: {tool_output}")
    print(f"Turns completed: {run_state.turn_count}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
