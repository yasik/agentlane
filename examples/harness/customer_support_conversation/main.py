"""Real OpenAI-backed harness demo with templated instructions and run resume."""

import asyncio
from pathlib import Path
from typing import TypedDict

from agentlane_openai import ResponsesClient

from agentlane.harness import Agent, AgentDescriptor, Runner
from agentlane.messaging import AgentId
from agentlane.models import Config, OutputSchema, PromptSpec, PromptTemplate
from agentlane.runtime import SingleThreadedRuntimeEngine
from _demo_utils import (
    configure_demo_logging,
    load_openai_api_key,
    print_intro,
    print_restart,
    print_summary,
    send_turn,
    snapshot_run_state,
)

MODEL_NAME = "gpt-5.4-mini"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


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


def _build_descriptor(api_key: str) -> AgentDescriptor:
    """Construct one support-agent descriptor for the live demo."""
    instruction_values: SupportInstructionValues = {
        "company_name": "Acme Devices",
        "refund_window_days": 30,
        "replacement_window_days": 14,
        "escalation_channel": "priority-support@acme.test",
        "brand_tone": "warm and confident",
    }
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    return AgentDescriptor(
        name="Acme Support",
        description="Frontline customer support assistant",
        model=model,
        model_args={"reasoning_effort": "low"},
        instructions=PromptSpec(
            template=SUPPORT_INSTRUCTIONS_TEMPLATE,
            values=instruction_values,
        ),
    )


async def run_demo() -> None:
    """Run the real multi-turn support demo."""
    configure_demo_logging()
    api_key = load_openai_api_key(ENV_FILE)
    runner = Runner(max_attempts=2)
    descriptor = _build_descriptor(api_key)
    agent_id = AgentId.from_values("support-agent", "customer-88421")

    print_intro()

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
    await send_turn(
        runtime=runtime,
        agent_id=agent_id,
        customer_text=initial_customer_text,
        payload=[
            PromptSpec(
                template=INITIAL_CUSTOMER_MESSAGE_TEMPLATE,
                values=initial_customer_values,
            )
        ],
    )

    await send_turn(
        runtime=runtime,
        agent_id=agent_id,
        customer_text=(
            "Thanks. The order number is 88421, the shipping box was dented, "
            "and I can share photos if needed."
        ),
        payload=(
            "Thanks. The order number is 88421, the shipping box was dented, "
            "and I can share photos if needed."
        ),
    )

    saved_run_state = snapshot_run_state(agent)
    await runtime.stop_when_idle()
    print_restart(saved_run_state)

    resumed_runtime = SingleThreadedRuntimeEngine()
    resumed_agent = Agent.bind(
        resumed_runtime,
        agent_id,
        runner=runner,
        descriptor=descriptor,
        run_state=saved_run_state,
    )

    await send_turn(
        runtime=resumed_runtime,
        agent_id=agent_id,
        customer_text=(
            "Please summarize the next steps in 3 bullet points so I know what to do next."
        ),
        payload="Please summarize the next steps in 3 bullet points so I know what to do next.",
    )

    resumed_state = snapshot_run_state(resumed_agent)
    await resumed_runtime.stop_when_idle()
    print_summary(MODEL_NAME, resumed_state)


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
