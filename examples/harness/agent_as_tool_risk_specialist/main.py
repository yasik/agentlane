"""Real OpenAI-backed example of a predefined finance agent used as a tool."""

import asyncio
import logging
import os

import structlog
from agentlane_openai import ResponsesClient
from pydantic import BaseModel

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Tools

MODEL_NAME = "gpt-5.4-mini"


class RiskReviewArgs(BaseModel):
    """Arguments exposed to the model for the risk specialist tool."""

    portfolio_name: str
    exposure_summary: str


async def run_demo() -> None:
    """Run the predefined agent-as-tool example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(config=Config(api_key=api_key, model=MODEL_NAME))

    risk_specialist = AgentDescriptor(
        name="Risk Specialist",
        description="Reviews portfolio exposure policy for Northstar Capital.",
        model=model,
        model_args={"reasoning_effort": "low"},
        # Child agents inherit parent tools by default. This example turns that
        # off so the specialist stays focused on its own prompt only.
        tools=None,
        instructions=(
            "You are Northstar Capital's portfolio risk specialist. "
            "Policy: any single sector above 35% exposure requires risk review "
            "before adding exposure; leveraged ETF exposure above 10% requires "
            "portfolio manager approval. Return one short sentence."
        ),
    )

    agent = DefaultAgent(
        runner=Runner(max_attempts=2),
        descriptor=AgentDescriptor(
            name="Portfolio Manager Assistant",
            description="Finance assistant that delegates risk policy questions.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "You help portfolio managers review exposure questions. "
                "Always call `risk_specialist` exactly once for risk policy questions. "
                "After the tool returns, answer in under 60 words and do not "
                "provide trade instructions."
            ),
            tools=Tools(
                tools=[
                    # `as_tool(...)` is declarative. The runner handles the
                    # actual runtime routing to the child agent.
                    risk_specialist.as_tool(
                        name="risk_specialist",
                        description="Ask the risk specialist for a final exposure ruling.",
                        args_model=RiskReviewArgs,
                    )
                ],
                # Require one tool call so the example always exercises the
                # agent-as-tool path before the manager answers.
                tool_choice="required",
                tool_call_limits={"risk_specialist": 1},
            ),
        ),
    )

    user_message = (
        "The growth portfolio is 42% semiconductors and 12% leveraged ETFs. "
        "Can we add more semiconductor exposure without a risk review?"
    )
    result = await agent.run(user_message)

    print("Example: predefined agent-as-tool")
    print(f"Model: {MODEL_NAME}")
    print()
    print(f"User: {user_message}")
    print()
    print(f"Assistant: {result.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
