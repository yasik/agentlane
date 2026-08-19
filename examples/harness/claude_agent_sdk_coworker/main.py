"""Real AgentLane to Claude Agent SDK coworker example."""

import asyncio
import logging
import os

import structlog
from agentlane_claude_agent_sdk import ClaudeAgent
from agentlane_openai import ResponsesClient
from claude_agent_sdk import ClaudeAgentOptions

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import Config, Tools, as_tool
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

MODEL_NAME = "gpt-5.4-mini"
NATIVE_ID = AgentId.from_values("agentlane-native", "researcher")
CLAUDE_ID = AgentId.from_values("claude-sdk", "analyst")


async def run_demo() -> None:
    """Run one native-to-Claude-to-native coworker flow."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    openai_api_key = os.environ["OPENAI_API_KEY"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    if not openai_api_key or not anthropic_api_key:
        raise ValueError("OPENAI_API_KEY and ANTHROPIC_API_KEY must be non-empty.")

    runtime = SingleThreadedRuntimeEngine(worker_count=2)
    ClaudeAgent.bind(
        runtime,
        CLAUDE_ID,
        options=ClaudeAgentOptions(
            tools=[],
            setting_sources=[],
            skills=[],
            mcp_servers={},
            strict_mcp_config=True,
            max_turns=1,
        ),
    )

    @as_tool
    async def ask_claude(
        task: str,
        cancellation_token: CancellationToken,
    ) -> str:
        """Ask the addressed Claude coworker to complete one text task."""
        print("1. Native AgentLane -> Claude")
        print(f"   Sender: {NATIVE_ID}")
        print(f"   Recipient: {CLAUDE_ID}")
        print(f"   Task: {task}")

        outcome = await runtime.send_message(
            task,
            sender=NATIVE_ID,
            recipient=CLAUDE_ID,
            cancellation_token=cancellation_token,
        )
        if outcome.status != DeliveryStatus.DELIVERED:
            message = (
                outcome.error.message
                if outcome.error is not None
                else f"Claude delivery failed with status {outcome.status.value}."
            )
            raise RuntimeError(message)
        if not isinstance(outcome.response_payload, str):
            raise RuntimeError("Claude delivery returned a non-string response.")

        print()
        print("2. Claude -> Native")
        print(f"   Delivery result: {outcome.response_payload}")
        return outcome.response_payload

    model = ResponsesClient(config=Config(api_key=openai_api_key, model=MODEL_NAME))
    native = DefaultAgent(
        runtime=runtime,
        runner=Runner(),
        agent_id=NATIVE_ID,
        descriptor=AgentDescriptor(
            name="AgentLane researcher",
            description="A native agent that delegates one subtask to Claude.",
            model=model,
            model_args={"reasoning_effort": "low"},
            instructions=(
                "Call `ask_claude` exactly once. Give Claude a focused version "
                "of the user's task. After the tool returns, use its result to "
                "write the final answer in no more than 80 words."
            ),
            tools=Tools(
                tools=[ask_claude],
                tool_choice="required",
                tool_call_limits={"ask_claude": 1},
            ),
        ),
    )

    user_message = (
        "Write one practical principle for a small team that uses AI coworkers."
    )
    result = await native.run(user_message)

    print()
    print("3. Final AgentLane answer")
    print(f"   {result.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
