import asyncio
import json
import threading
from io import StringIO
from typing import TextIO, cast

import pytest
from agentlane_process_bridge import (
    BridgeEventType,
    ContractPayloadError,
    EventWriter,
)
from pydantic import BaseModel

LONG_TEXT_CHARS = 5004
LONG_ITEM_COUNT = 52


def test_event_writer_preserves_complete_nested_json_values() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        await writer.emit(
            BridgeEventType.READY,
            text="x" * LONG_TEXT_CHARS,
            items=list(range(LONG_ITEM_COUNT)),
            mapping={
                str(index): {"text": "x" * LONG_TEXT_CHARS}
                for index in range(LONG_ITEM_COUNT)
            },
            nested=({"items": list(range(LONG_ITEM_COUNT))},),
            custom=object(),
        )

        line = output.getvalue().strip()
        event = json.loads(line)
        assert event["protocol_version"] == "2.0"
        assert event["type"] == "ready"
        assert isinstance(event["ts"], float)
        assert event["text"] == "x" * LONG_TEXT_CHARS
        assert event["items"] == list(range(LONG_ITEM_COUNT))
        assert event["mapping"] == {
            str(index): {"text": "x" * LONG_TEXT_CHARS}
            for index in range(LONG_ITEM_COUNT)
        }
        assert event["nested"] == [{"items": list(range(LONG_ITEM_COUNT))}]
        assert event["custom"].startswith("<object object at ")
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_preserves_complete_custom_object_strings() -> None:
    class LongText:
        def __str__(self) -> str:
            return "x" * LONG_TEXT_CHARS

    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        await writer.emit(BridgeEventType.READY, custom=LongText())

        [event] = [json.loads(line) for line in output.getvalue().splitlines()]
        assert event["custom"] == "x" * LONG_TEXT_CHARS
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_preserves_verbatim_contract_payload() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        config = {"items": list(range(LONG_ITEM_COUNT))}

        await writer.emit(
            BridgeEventType.READY,
            items=list(range(LONG_ITEM_COUNT)),
            verbatim_payload={"config": config},
        )

        [event] = [json.loads(line) for line in output.getvalue().splitlines()]
        assert event["items"] == list(range(LONG_ITEM_COUNT))
        assert event["config"] == config
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_rejects_non_json_contract_payload() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)

        with pytest.raises(ContractPayloadError):
            await writer.emit(
                BridgeEventType.READY,
                verbatim_payload={"config": {"bad": object()}},
            )

        assert output.getvalue() == ""
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_rejects_verbatim_payload_key_collisions() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)

        with pytest.raises(ContractPayloadError):
            await writer.emit(
                BridgeEventType.READY,
                config={"truncated": True},
                verbatim_payload={"config": {"authoritative": True}},
            )

        assert output.getvalue() == ""
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_rejects_reserved_envelope_payload_fields() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)

        for field in ("protocol_version", "type", "ts"):
            with pytest.raises(ContractPayloadError):
                await writer.emit_payload(
                    BridgeEventType.READY,
                    {field: "collision"},
                )

            with pytest.raises(ContractPayloadError):
                await writer.emit(
                    BridgeEventType.READY,
                    verbatim_payload={field: "collision"},
                )

        assert output.getvalue() == ""
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_preserves_large_contract_payload() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)

        config = {"text": "界🧪" * 40_000}
        await writer.emit(
            BridgeEventType.READY,
            verbatim_payload={"config": config},
        )

        [event] = [json.loads(line) for line in output.getvalue().splitlines()]
        assert event["config"] == config
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_preserves_structured_result_models() -> None:
    class Result(BaseModel):
        items: list[dict[str, object]]

    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        result = Result(items=[{"text": "界🧪" * 5001, "ok": True, "value": None}] * 51)
        await writer.emit(BridgeEventType.TOOL_END, result=result)

        [event] = [json.loads(line) for line in output.getvalue().splitlines()]
        assert event["result"] == result.model_dump()
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_sanitizes_non_finite_float_values() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        await writer.emit(BridgeEventType.READY, score=float("nan"), limit=float("inf"))

        [event] = [json.loads(line) for line in output.getvalue().splitlines()]
        assert event["score"] == "nan"
        assert event["limit"] == "inf"
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_times_out_blocked_stream_writes() -> None:
    class BlockingStream:
        def __init__(self) -> None:
            self.release = threading.Event()
            self.value = ""

        def write(self, value: str) -> int:
            self.release.wait(timeout=1)
            self.value += value
            return len(value)

        def flush(self) -> None:
            return None

    async def scenario() -> None:
        stream = BlockingStream()
        writer = EventWriter(cast(TextIO, stream), write_timeout_seconds=0.01)

        with pytest.raises(TimeoutError):
            await writer.emit(BridgeEventType.READY)

        with pytest.raises(BrokenPipeError):
            await writer.emit(BridgeEventType.READY)

        stream.release.set()
        with pytest.raises(BrokenPipeError):
            await writer.aclose()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_event_writer_preserves_concurrent_write_order() -> None:
    async def scenario() -> None:
        output = StringIO()
        writer = EventWriter(output)
        await asyncio.gather(
            *(
                writer.emit(BridgeEventType.RUN_EVENT, index=index)
                for index in range(10)
            )
        )

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [event["index"] for event in events] == list(range(10))
        await writer.aclose()

    asyncio.run(scenario())


def test_event_writer_batches_streaming_events_until_terminal_drain() -> None:
    class CountingStream(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1
            super().flush()

    async def scenario() -> None:
        output = CountingStream()
        writer = EventWriter(output)
        for index in range(100):
            await writer.emit(BridgeEventType.ASSISTANT_DELTA, text=str(index))
        await writer.emit(
            BridgeEventType.RUN_COMPLETE,
            final_output="done",
            turn_count=1,
            response_count=1,
            shim_state={},
        )

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        assert len(events) == 101
        assert events[-1]["type"] == "run_complete"
        assert output.flush_count < 100
        await writer.aclose()

    asyncio.run(scenario())
