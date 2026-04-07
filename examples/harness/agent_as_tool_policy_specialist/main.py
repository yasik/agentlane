"""Real OpenAI-backed example of a predefined agent used as a tool."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient
from pydantic import BaseModel

from agentlane.harness import Agent, AgentDescriptor, Runner, RunResult
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"


class WarrantyLookupArgs(BaseModel):
    """Arguments exposed to the model for the policy specialist tool."""

    product_name: str
    issue_summary: str


async def run_demo() -> None:
    """Run the predefined agent-as-tool example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))
    runner = Runner(max_attempts=2)
    runtime = SingleThreadedRuntimeEngine()
    agent_id = AgentId.from_values("support-agent", "agent-tool-policy-specialist")

    policy_specialist = AgentDescriptor(
        name="Policy Specialist",
        description="Answers warranty policy questions for Acme devices.",
        model=model,
        model_args={"reasoning_effort": "low"},
        # Child agents inherit parent tools by default. This example turns that
        # off so the specialist stays focused on its own prompt only.
        tools=None,
        instructions=(
            "You are Acme's warranty policy specialist. "
            "Policy: accidental damage like cracked screens is not covered by the "
            "standard warranty. Manufacturing defects are covered for one year. "
            "Return one short sentence."
        ),
    )

    Agent.bind(
        runtime,
        agent_id,
        runner=runner,
        descriptor=AgentDescriptor(
            name="Support Manager",
            description="Frontline support agent that delegates policy questions.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You answer customer questions about Acme devices. "
                "Always call `policy_specialist` exactly once for warranty questions. "
                "After the tool returns, answer the customer in under 60 words."
            ),
            tools=Tools(
                tools=[
                    # `as_tool(...)` is declarative. The runner handles the
                    # actual runtime routing to the child agent.
                    policy_specialist.as_tool(
                        name="policy_specialist",
                        description="Ask the warranty policy specialist for a final policy ruling.",
                        args_model=WarrantyLookupArgs,
                    )
                ],
                # Require one tool call so the example always exercises the
                # agent-as-tool path before the manager answers.
                tool_choice="required",
                tool_call_limits={"policy_specialist": 1},
            ),
        ),
    )

    user_message = (
        "My Acme UltraBook screen cracked after one week when I dropped it. "
        "Does the standard warranty cover that?"
    )
    outcome = await runtime.send_message(user_message, recipient=agent_id)
    await runtime.stop_when_idle()

    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"Expected delivered outcome, got {outcome.status.value}.")
    if not isinstance(outcome.response_payload, RunResult):
        raise RuntimeError("Expected a RunResult response payload.")

    print("Example: predefined agent-as-tool")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {outcome.response_payload.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
