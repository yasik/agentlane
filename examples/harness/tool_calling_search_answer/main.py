"""Real OpenAI-backed harness demo showing the tool loop."""

import asyncio
import logging
import os
from typing import TypedDict, cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.models import (
    Config,
    OutputSchema,
    PromptSpec,
    PromptTemplate,
    ToolCall,
    Tools,
    as_tool,
)

MODEL_NAME = "gpt-5.4-mini"
MOCK_SEARCH_RESULT = (
    "Harborview Clinical Protocol: A patient reporting new bruising while taking "
    "warfarin should be asked about bleeding symptoms, medication changes, and "
    "recent INR timing. Escalate same day for active bleeding, head injury, black "
    "stools, or severe headache."
)


class AssistantInstructionValues(TypedDict):
    """Typed values used to render the clinical protocol assistant instructions."""

    clinic_name: str
    knowledge_source: str
    tone: str


INSTRUCTIONS_TEMPLATE = PromptTemplate[AssistantInstructionValues, str](
    system_template="""
You are {{ clinic_name }}'s clinical protocol assistant.
Use a {{ tone }} tone.

When the user asks about a care protocol, you must call
`search_clinical_protocol` before answering. Answer only from the returned
{{ knowledge_source }} result and keep the answer under 80 words. Do not
diagnose or recommend medication changes.
""".strip(),
    user_template=None,
    output_schema=OutputSchema(str),
)


@as_tool
async def search_clinical_protocol(question: str) -> str:
    """Search the clinical protocol library for the current care guidance."""
    del question
    return MOCK_SEARCH_RESULT


async def run_demo() -> None:
    """Run the real tool-calling demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    instruction_values: AssistantInstructionValues = {
        "clinic_name": "Harborview Anticoagulation Clinic",
        "knowledge_source": "protocol-library",
        "tone": "clear and practical",
    }
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=AgentDescriptor(
            name="Clinical Protocol Assistant",
            description="Answers clinical workflow questions with a protocol search tool",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=PromptSpec(
                template=INSTRUCTIONS_TEMPLATE,
                values=instruction_values,
            ),
            # Require one search call so the example always exercises the tool
            # loop before the assistant answers.
            tools=Tools(
                tools=[search_clinical_protocol],
                tool_choice="required",
                tool_call_limits={"search_clinical_protocol": 1},
            ),
        ),
    )
    question = (
        "A patient on warfarin reports new bruising and missed their last INR check. "
        "What does our protocol say to ask and escalate?"
    )

    result = await agent.run(question)
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

    print("Example: tool-calling search answer")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {question}")
    print()
    print(f"Assistant: {result.final_output}")
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
