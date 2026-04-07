"""Real OpenAI-backed harness demo with templated instructions and run resume."""

import asyncio
import logging
import os
from typing import TypedDict

import structlog
from agentlane_openai import ResponsesClient

from agentlane.harness import Agent, AgentDescriptor, Runner, RunResult
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, OutputSchema, PromptSpec, PromptTemplate
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"


class SupportInstructionValues(TypedDict):
    """Typed values used to render the support agent instructions."""

    company_name: str
    refund_window_days: int
    replacement_window_days: int
    escalation_channel: str
    brand_tone: str


class CustomerIssueValues(TypedDict):
    """Typed values used to render the initial customer message."""

    company_name: str
    order_id: str
    issue_summary: str
    delivery_timing: str


# The descriptor instructions are system-only in this example, so the prompt
# template does not need to manufacture a fake user message anymore.
SUPPORT_INSTRUCTIONS_TEMPLATE = PromptTemplate[SupportInstructionValues, str](
    system_template="""
You are {{ company_name }}'s frontline customer support assistant.
Use a {{ brand_tone }} tone and keep replies concise, practical, and honest.

Policy snapshot:
- Refund window: {{ refund_window_days }} days from delivery
- Replacement window: {{ replacement_window_days }} days from delivery
- Escalation channel: {{ escalation_channel }}

Rules:
- Never claim you checked internal systems or completed an action.
- Ask for missing details before suggesting a resolution.
- If enough details are available, explain the likely next step clearly.
- Keep replies under 120 words unless the customer explicitly asks for more detail.
""".strip(),
    output_schema=OutputSchema(str),
)

INITIAL_CUSTOMER_MESSAGE_TEMPLATE = PromptTemplate[CustomerIssueValues, str](
    system_template=None,
    user_template="""
Hi, I need help with a {{ company_name }} order.
Order number: {{ order_id }}.
Problem: {{ issue_summary }}.
Timing: {{ delivery_timing }}.
    """.strip(),
    output_schema=OutputSchema(str),
)


async def run_demo() -> None:
    """Run the real multi-turn support demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    runner = Runner(max_attempts=2)
    agent_id = AgentId.from_values("support-agent", "customer-88421")
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    instruction_values: SupportInstructionValues = {
        "company_name": "Acme Devices",
        "refund_window_days": 30,
        "replacement_window_days": 14,
        "escalation_channel": "priority-support@acme.test",
        "brand_tone": "warm and confident",
    }
    descriptor = AgentDescriptor(
        name="Acme Support",
        description="Frontline customer support assistant",
        model=model,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=SUPPORT_INSTRUCTIONS_TEMPLATE,
            values=instruction_values,
        ),
    )

    runtime = SingleThreadedRuntimeEngine()
    agent = Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=descriptor,
    )

    # This demo currently simulates user turns by sending them directly from the
    # script. That keeps the example focused on the harness surface:
    # instructions, multi-turn continuation, and `RunState` resume. It is not a
    # proper user handoff flow yet.
    initial_customer_values: CustomerIssueValues = {
        "company_name": "Acme Devices",
        "order_id": "88421",
        "issue_summary": "the phone arrived with a cracked screen",
        "delivery_timing": "it was delivered this morning",
    }
    initial_customer_text = (
        "Hi, I need help with an Acme Devices order. "
        "Order number: 88421. Problem: the phone arrived with a cracked screen. "
        "Timing: it was delivered this morning."
    )
    print("Example: customer support conversation with resume")
    print(f"Model: {MODEL_NAME}")
    print()
    print(
        "Note: this demo simulates customer follow-up turns by sending them from the script."
    )
    print("It does not implement proper user handoff yet.")
    print()

    first_outcome = await runtime.send_message(
        [
            PromptSpec(
                template=INITIAL_CUSTOMER_MESSAGE_TEMPLATE,
                values=initial_customer_values,
            )
        ],
        recipient=agent_id,
    )
    if first_outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(
            f"Expected delivered outcome, got {first_outcome.status.value}."
        )
    if not isinstance(first_outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")
    print(f"Customer: {initial_customer_text}")
    print(f"Assistant: {first_outcome.response_payload.final_output}")
    print()

    second_customer_text = (
        "Thanks. The order number is 88421, the shipping box was dented, "
        "and I can share photos if needed."
    )
    second_outcome = await runtime.send_message(
        second_customer_text,
        recipient=agent_id,
    )
    if second_outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(
            f"Expected delivered outcome, got {second_outcome.status.value}."
        )
    if not isinstance(second_outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")
    print(f"Customer: {second_customer_text}")
    print(f"Assistant: {second_outcome.response_payload.final_output}")
    print()

    saved_run_state = agent.run_state
    await runtime.stop_when_idle()
    if saved_run_state is None:
        raise RuntimeError("Expected the harness agent to expose persisted run state.")

    print(
        "Simulated restart:"
        f" saved RunState after {saved_run_state.turn_count} turns and"
        f" {len(saved_run_state.responses)} raw model responses."
    )
    print()

    resumed_runtime = SingleThreadedRuntimeEngine()
    Agent.bind(
        resumed_runtime,
        agent_id,
        runner=runner,
        descriptor=descriptor,
        run_state=saved_run_state,
    )

    third_customer_text = (
        "Please summarize the next steps in 3 bullet points so I know what to do next."
    )
    resumed_outcome = await resumed_runtime.send_message(
        third_customer_text,
        recipient=agent_id,
    )
    if resumed_outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(
            f"Expected delivered outcome, got {resumed_outcome.status.value}."
        )
    if not isinstance(resumed_outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")
    print(f"Customer: {third_customer_text}")
    print(f"Assistant: {resumed_outcome.response_payload.final_output}")
    print()

    resumed_agent_state = resumed_outcome.response_payload.run_state
    await resumed_runtime.stop_when_idle()
    if resumed_agent_state is None:
        raise RuntimeError("Expected the final RunResult to include run_state.")

    print(
        "Run summary:"
        f" resumed successfully, {resumed_agent_state.turn_count} total turns,"
        f" {len(resumed_agent_state.responses)} raw responses."
    )


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
