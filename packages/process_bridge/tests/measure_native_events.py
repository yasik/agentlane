"""Measure synthetic native record sizes and local bridge write costs.

Run with ``uv run python -m packages.process_bridge.tests.measure_native_events``.
Results describe this machine and an in-memory sink, not provider or pipe latency.
"""

import asyncio
import json
from io import StringIO
from statistics import median
from time import perf_counter

from agentlane_process_bridge import BridgeEventType, EventWriter

from agentlane.harness import (
    RunAgentEndEvent,
    RunEvent,
    RunModelStreamEvent,
    RunResult,
    RunState,
)
from agentlane.models import ModelStreamEvent, ModelStreamEventKind


async def measure(event: RunEvent, *, repeats: int = 30) -> dict[str, object]:
    """Return median conversion and conversion-plus-writer times in milliseconds."""
    conversion: list[float] = []
    delivery: list[float] = []
    wire_bytes = 0
    for _ in range(repeats):
        start = perf_counter()
        event.to_dict()
        conversion.append((perf_counter() - start) * 1000)
        sink = StringIO()
        writer = EventWriter(sink)
        start = perf_counter()
        await writer.emit(
            BridgeEventType.RUN_EVENT, verbatim_payload={"event": event.to_dict()}
        )
        await writer.drain()
        delivery.append((perf_counter() - start) * 1000)
        wire_bytes = len(sink.getvalue().encode("utf-8"))
        await writer.aclose()

    return {
        "repeats": repeats,
        "wire_bytes": wire_bytes,
        "conversion_ms_median": round(median(conversion), 3),
        "delivery_ms_median": round(median(delivery), 3),
    }


async def main() -> None:
    """Print reproducible synthetic workloads and measured costs as JSON."""
    results: dict[str, object] = {}
    results["text_delta"] = await measure(
        RunModelStreamEvent(
            event=ModelStreamEvent(kind=ModelStreamEventKind.TEXT_DELTA, text="界🧪")
        )
    )
    for count in (10, 100, 1000):
        state = RunState(
            instructions="Synthetic instructions.",
            history=[
                {"role": "user", "content": f"{index}: " + "界🧪 data\n" * 25}
                for index in range(count)
            ],
            responses=[],
        )
        results[f"agent_end_{count}_history_items"] = await measure(
            RunAgentEndEvent(
                task_name="Synthetic",
                task_id="root",
                result=RunResult(
                    final_output={"complete": True},
                    responses=[],
                    turn_count=1,
                    run_state=state,
                ),
            )
        )

    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    asyncio.run(main())
