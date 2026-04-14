"""Real OpenAI-backed streaming example with tool call, agent-as-tool, and handoff."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient
from pydantic import BaseModel, Field

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, ModelResponse, ModelStreamEventKind, Tools, as_tool

MODEL_NAME = "gpt-5.4-mini"

MODEL = ResponsesClient(
    config=Config(
        api_key=os.environ["OPENAI_API_KEY"],
        model=MODEL_NAME,
    )
)


@as_tool
async def lookup_order_status(order_id: str) -> str:
    """Look up the current order facts for one customer order."""
    del order_id
    return (
        "Order 88421: Acme UltraBook Pro, delivered yesterday, reported with a "
        "cracked screen on arrival. The customer wants to know whether it can "
        "be returned or must be handled as a damage claim."
    )


class PolicySpecialistArgs(BaseModel):
    """Arguments exposed for the policy-specialist agent tool."""

    order_id: str = Field(description="The order id being reviewed.")
    product_name: str = Field(description="The product name from the order.")
    issue_summary: str = Field(
        description="A short summary of the return or damage issue."
    )


RETURNS_SPECIALIST = AgentDescriptor(
    name="Returns Specialist",
    description="Takes over damaged-order and return-claim conversations.",
    model=MODEL,
    model_args={"reasoning": {"effort": "low", "summary": "detailed"}},
    tools=None,
    instructions=(
        "You are Acme's returns specialist. "
        "The conversation has already been triaged and includes order facts plus "
        "policy findings. Take over directly with the customer. "
        "Explain the next step in at most three bullet points."
    ),
)

POLICY_SPECIALIST = AgentDescriptor(
    name="Policy Specialist",
    description="Interprets Acme's return policy for a specific order issue.",
    model=MODEL,
    model_args={"reasoning": {"effort": "low", "summary": "detailed"}},
    tools=None,
    instructions=(
        "You are Acme's policy specialist. "
        "Policy: opened laptops may be returned within 30 days only when they "
        "are in resellable condition. Cracked screens count as accidental damage "
        "and are not eligible for the standard return window. "
        "Return one short sentence with the final policy ruling."
    ),
)


class FrontlineSupportAgent(DefaultAgent):
    """Frontline support agent that gathers facts, consults policy, then transfers."""

    descriptor = AgentDescriptor(
        name="Acme Frontline Support",
        description="Frontline support that gathers facts and escalates damaged returns.",
        model=MODEL,
        model_args={"reasoning": {"effort": "medium", "summary": "detailed"}},
        instructions=(
            "You are Acme frontline support. "
            "For damaged-laptop return questions, follow this exact sequence: "
            "1. In one short two-step preamble, explain what you are about to verify. "
            "2. Call `lookup_order_status` exactly once. "
            "3. Call `policy_specialist` exactly once with the order facts and issue summary. "
            "4. If the case is damaged or not eligible for the standard return window, "
            "handoff to `returns_specialist`. "
            "Do not handoff before steps 2 and 3 are complete."
        ),
        tools=Tools(
            tools=[
                lookup_order_status,
                POLICY_SPECIALIST.as_tool(
                    name="policy_specialist",
                    description="Ask the policy specialist for a final policy ruling.",
                    args_model=PolicySpecialistArgs,
                ),
            ],
            tool_choice="required",
            parallel_tool_calls=False,
            tool_call_limits={
                "lookup_order_status": 1,
                "policy_specialist": 1,
            },
        ),
        handoffs=(RETURNS_SPECIALIST,),
    )


def _tool_path(run_state_responses: list[ModelResponse]) -> list[str]:
    """Extract the sequence of tool or handoff calls from raw responses."""
    names: list[str] = []
    for response in run_state_responses:
        tool_calls = response.choices[0].message.tool_calls or []
        for tool_call in tool_calls:
            name = tool_call.function.name or ""
            if name:
                names.append(name)
    return names


async def run_demo() -> None:
    """Run the orchestration streaming example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    agent = FrontlineSupportAgent()
    user_message = (
        "Order 88421 arrived with a cracked screen. Can I return it, or does this "
        "need a different process?"
    )
    stream = await agent.run_stream(user_message)

    print("Example: streaming escalation flow")
    print(f"Model: {MODEL_NAME}")
    print()
    print("One streamed run exercises:")
    print("1. a normal tool call")
    print("2. a predefined agent-as-tool delegation")
    print("3. a first-class handoff that keeps streaming through the specialist")
    print()
    print(f"User: {user_message}")
    print("Assistant stream:")

    last_openai_phase: str | None = None
    saw_reasoning = False
    async for event in stream:
        if event.kind == ModelStreamEventKind.REASONING and event.reasoning is not None:
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
            print(f"[tool arguments] {event.arguments_delta}")

    result = await stream.result()
    run_state = agent.run_state
    if run_state is None:
        raise RuntimeError("Expected the streamed run to persist run state.")

    print()
    print()
    print(f"Final output: {result.final_output}")
    print(f"Tool and handoff path: {' -> '.join(_tool_path(run_state.responses))}")
    print(
        "Run summary:"
        f" {run_state.turn_count} turns,"
        f" {len(run_state.responses)} raw model responses."
    )
    print()
    print(
        "Note: the outer stream shows the parent run plus the transferred "
        "specialist run after handoff. The internal policy-specialist "
        "agent-as-tool run stays internal in the current streaming contract."
    )


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
