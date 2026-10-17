"""Change an active OpenAI response through the harness steering channel."""

import asyncio
import json
import os
from contextlib import suppress

from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, MessageRole, SteeringReceipt, SteeringStatus

DEFAULT_MODEL = "gpt-6-astra"
"""Used when OPENAI_MODEL does not select another steering-capable model."""

STEERING_INPUT = (
    "Change of plan: visit Kyoto instead of Tokyo. Replace the earlier itinerary "
    "with three quiet Kyoto gardens. Start the revised answer with 'Kyoto'."
)
"""Submitted during generation and checked in the committed run history."""


async def run_demo() -> None:
    """Submit one live update and verify that the successor applies it."""
    model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Trip planner",
            model=ResponsesClient(
                Config(api_key=os.environ["OPENAI_API_KEY"], model=model_name),
                websocket=True,
            ),
            instructions="Follow the latest travel request. Keep the answer concise.",
        )
    )
    stream = await agent.run_stream(
        "Plan a five-day Tokyo trip with museums, food stops, and practical travel tips."
    )
    receipt: SteeringReceipt | None = None
    statuses: list[str] = []

    # Submit after the provider creates an active response, before it finishes.
    try:
        async with asyncio.timeout(90):
            async for event in stream:
                if event.provider_event_type == "response.created" and receipt is None:
                    receipt = await stream.steer(STEERING_INPUT)
                    statuses.append(receipt.status)
                    print(f"Steering: {receipt.status}", flush=True)

                for update in event.steering or ():
                    if not statuses or statuses[-1] != update.status:
                        statuses.append(update.status)
                        print(f"Steering: {update.status}", flush=True)

            result = await stream.result()
    finally:
        # Closing the public stream cancels an unfinished run and its SDK connection.
        await stream.aclose()
        with suppress(Exception, asyncio.CancelledError):
            await stream.result()

    # Acceptance alone is insufficient: require an applied successor and history.
    if receipt is None or receipt.status != SteeringStatus.APPLIED:
        raise RuntimeError(f"Steering was not applied: {receipt!r}")
    if result.run_state is None:
        raise RuntimeError("The run did not return committed history.")
    history_copies = result.run_state.history.count(
        {"role": MessageRole.USER.value, "content": STEERING_INPUT}
    )
    if history_copies != 1:
        raise RuntimeError("The steering input must appear once in committed history.")
    if not isinstance(result.final_output, str) or not result.final_output.startswith(
        "Kyoto"
    ):
        raise RuntimeError("The final answer did not follow the revised destination.")

    print(
        json.dumps(
            {
                "verified": True,
                "model": model_name,
                "steering_statuses": statuses,
                "previous_response_id": receipt.previous_response_id,
                "successor_response_id": receipt.response_id,
                "history_copies": history_copies,
                "answer": result.final_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
