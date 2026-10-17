"""Run native async OpenAI tool calls while the model continues its response."""

import asyncio
import json
import os
import time
from contextlib import suppress
from typing import Literal

from agentlane_openai import ResponsesClient

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, ModelStreamEventKind, Tools

DEFAULT_MODEL = "gpt-6-astra"
"""Used when OPENAI_MODEL does not select another async-tool-capable model."""


async def run_demo() -> None:
    """Verify tool overlap, continued model output, and the final source facts."""
    started_at = time.monotonic()
    started: dict[str, float] = {}
    finished: dict[str, float] = {}
    executions: list[str] = []

    async def lookup_order_detail(detail: Literal["stock", "delivery"]) -> str:
        """Fetch one synthetic order detail. Request both stock and delivery."""
        executions.append(detail)
        started[detail] = time.monotonic() - started_at
        print(f"\n[{started[detail]:.2f}s] {detail} started", flush=True)

        # Different service delays make overlapping work visible in the timeline.
        await asyncio.sleep(4 if detail == "stock" else 8)
        finished[detail] = time.monotonic() - started_at
        print(f"\n[{finished[detail]:.2f}s] {detail} finished", flush=True)

        return json.dumps(
            {"stock_units": 17}
            if detail == "stock"
            else {"delivery_reference": "THU-42"}
        )

    model_name = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Order assistant",
            model=ResponsesClient(
                Config(api_key=os.environ["OPENAI_API_KEY"], model=model_name),
                async_tools=[lookup_order_detail.__name__],
            ),
            model_args={"reasoning": {"effort": "low"}},
            instructions=(
                "Call lookup_order_detail twice immediately: stock and delivery. "
                "These are async tools. While they run, explain three general "
                "packing checks that do not depend on the results, then end your "
                "response without waiting. The harness will send the results in "
                "a later turn. After receiving both results, return only a JSON "
                "object with stock_units and "
                "delivery_reference copied from the tools."
            ),
            tools=Tools(
                [lookup_order_detail],
                parallel_tool_calls=True,
                tool_call_limits={lookup_order_detail.__name__: 2},
            ),
        )
    )
    stream = await agent.run_stream("Check stock and delivery for the sample order.")
    async_call_ids: set[str] = set()
    text_during_tools = 0
    first_response_finished: float | None = None

    # Observe the public stream while the harness owns dispatch and tool cleanup.
    try:
        async with asyncio.timeout(90):
            async for event in stream:
                if event.async_tool_call is not None:
                    async_call_ids.add(event.async_tool_call.id)
                if event.kind == ModelStreamEventKind.TEXT_DELTA and event.text:
                    if len(started) > len(finished):
                        text_during_tools += 1
                    print(event.text, end="", flush=True)
                if (
                    event.kind == ModelStreamEventKind.COMPLETED
                    and first_response_finished is None
                ):
                    first_response_finished = time.monotonic() - started_at

            result = await stream.result()
    finally:
        await stream.aclose()
        with suppress(Exception, asyncio.CancelledError):
            await stream.result()

    # A normal parallel tool batch is insufficient: require native async events
    # and model text while at least one tool is still running.
    if sorted(executions) != ["delivery", "stock"] or len(async_call_ids) != 2:
        raise RuntimeError("Expected two native async calls, each executed once.")
    if set(finished) != set(started) or max(started.values()) >= min(finished.values()):
        raise RuntimeError("The two tool executions did not overlap.")
    if (
        first_response_finished is None
        or min(started.values()) >= first_response_finished
        or text_during_tools == 0
    ):
        raise RuntimeError("The model did not produce text while tools were running.")
    if not isinstance(result.final_output, str):
        raise RuntimeError("The final answer must be a JSON object.")
    answer = json.loads(result.final_output)
    if answer != {
        "stock_units": 17,
        "delivery_reference": "THU-42",
    }:
        raise RuntimeError("The final answer did not contain both tool results.")

    print()
    print(
        json.dumps(
            {
                "verified": True,
                "model": model_name,
                "native_async_calls": len(async_call_ids),
                "executions": executions,
                "tool_started_seconds": started,
                "tool_finished_seconds": finished,
                "first_response_finished_seconds": first_response_finished,
                "text_deltas_during_tools": text_during_tools,
                "answer": answer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(run_demo())
