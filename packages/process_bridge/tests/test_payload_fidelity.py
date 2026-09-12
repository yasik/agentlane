import asyncio
import json
from io import StringIO

import pytest
from agentlane_process_bridge import BridgeBackend, EventWriter, PromptCommand

from agentlane.harness import (
    RunAgentEndEvent,
    RunHandoffEndEvent,
    RunLLMEndEvent,
    RunModelStreamEvent,
    RunResult,
    RunToolEndEvent,
    RunToolStartEvent,
)
from agentlane.models import (
    Choice,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
)

from .helpers import FakeAgent, wait_for_event_count, wait_for_stream


@pytest.mark.parametrize(
    "length", [499, 500, 501, 1799, 1800, 1801, 4999, 5000, 5001, 40_000]
)
def test_backend_preserves_text_at_former_preview_limits(length: int) -> None:
    # Whitespace and non-ASCII text must survive as well as the visible body.
    suffix = "\n🧪 tail\t \n"
    text = "界" * (length - len(suffix)) + suffix
    _assert_backend_result(text)


@pytest.mark.parametrize(
    "result", [None, False, 42, [1, True, None], {"items": [{"text": "x" * 6000}] * 60}]
)
def test_backend_preserves_structured_results(result: object) -> None:
    _assert_backend_result(result)


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_backend_rejects_circular_tool_and_final_results(kind: str) -> None:
    value: object
    if kind == "list":
        items: list[object] = []
        items.append(items)
        value = items
    else:
        mapping: dict[str, object] = {}
        mapping["self"] = mapping
        value = mapping

    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        call = ToolCall.model_validate(
            {
                "id": "call",
                "type": "function",
                "function": {"name": "example", "arguments": "{}"},
            }
        )
        stream.emit(
            RunToolEndEvent(
                task_name="Root", task_id="root", tool_call=call, result=value, ok=True
            )
        )
        stream.finish(RunResult(final_output=value, responses=[], turn_count=1))

        events = await wait_for_event_count(output, 2)
        assert [event["type"] for event in events] == ["run_start", "error"]
        assert "Cyclic values" in events[-1]["message"]
        await backend.close()

    asyncio.run(scenario())


def test_backend_reports_final_output_serialization_failure() -> None:
    class InvalidResult:
        def __str__(self) -> str:
            raise ValueError("cannot serialize result")

    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        stream.finish(
            RunResult(final_output=InvalidResult(), responses=[], turn_count=1)
        )

        events = await wait_for_event_count(output, 2)
        assert [event["type"] for event in events] == ["run_start", "error"]
        assert events[-1]["scope"] == "run"
        assert "Unsupported event value" in events[-1]["message"]
        await backend.close()

    asyncio.run(scenario())


def _assert_backend_result(value: object) -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        call = ToolCall.model_validate(
            {
                "id": "call",
                "type": "function",
                "function": {"name": "read", "arguments": json.dumps({"value": value})},
            }
        )
        result = RunResult(final_output=value, responses=[], turn_count=1)
        stream.emit(RunToolStartEvent(task_name="Root", task_id="root", tool_call=call))
        tool_end = RunToolEndEvent(
            task_name="Root", task_id="root", tool_call=call, result=value, ok=True
        )
        stream.emit(tool_end)
        stream.emit(RunAgentEndEvent(task_name="Root", task_id="root", result=result))
        stream.emit(
            RunHandoffEndEvent(
                task_name="Root",
                task_id="root",
                tool_call=call,
                target_name="Child",
                result=result,
            )
        )
        stream.finish(result)

        events = await wait_for_event_count(output, 6)
        assert [event["type"] for event in events] == [
            "run_start",
            "run_event",
            "run_event",
            "run_event",
            "run_event",
            "run_complete",
        ]
        assert json.loads(
            events[1]["event"]["payload"]["tool_call"]["function"]["arguments"]
        ) == {"value": value}
        assert events[2]["event"]["payload"]["result"] == value
        assert events[3]["event"]["payload"]["result"]["final_output"] == value
        assert events[4]["event"]["payload"]["result"]["final_output"] == value
        assert events[5]["final_output"] == value
        assert tool_end.result is value
        await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("chunk_size", [2500, 50_000])
def test_backend_preserves_stream_content_independent_of_chunk_size(
    chunk_size: int,
) -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        text = "界🧪" * 4000 + "\n tail\t "
        arguments = json.dumps({"text": text}, ensure_ascii=False)
        await backend.handle_command(PromptCommand(text=text.strip()))
        stream = await wait_for_stream(agent)
        count = 1

        for kind, content in [
            (ModelStreamEventKind.TEXT_DELTA, text),
            (ModelStreamEventKind.REASONING, text),
            (ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA, arguments),
        ]:
            for start in range(0, len(content), chunk_size):
                chunk = content[start : start + chunk_size]
                stream.emit(
                    RunModelStreamEvent(
                        event=ModelStreamEvent(
                            kind=kind,
                            text=(
                                chunk
                                if kind == ModelStreamEventKind.TEXT_DELTA
                                else None
                            ),
                            reasoning=(
                                chunk
                                if kind == ModelStreamEventKind.REASONING
                                else None
                            ),
                            arguments_delta=(
                                chunk
                                if kind
                                == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA
                                else None
                            ),
                            tool_call_id="call",
                        )
                    )
                )
                count += 1

        stream.emit(
            RunLLMEndEvent(
                task_name="Root",
                task_id="root",
                response=ModelResponse(
                    id="response",
                    created=0,
                    model="test",
                    object="chat.completion",
                    choices=[
                        Choice(
                            index=0,
                            finish_reason="stop",
                            message=Message(role="assistant", content=text),
                        )
                    ],
                ),
            )
        )
        stream.finish(RunResult(final_output=text, responses=[], turn_count=1))
        events = await wait_for_event_count(output, count + 2)
        assert events[0]["prompt"] == text.strip()
        models = [
            event["event"]["payload"]["event"]
            for event in events
            if event["type"] == "run_event" and event["event"]["type"] == "model_stream"
        ]
        assert (
            "".join(event["text"] for event in models if event["kind"] == "text_delta")
            == text
        )
        assert (
            "".join(
                event["reasoning"] for event in models if event["kind"] == "reasoning"
            )
            == text
        )
        received_arguments = "".join(
            event["arguments_delta"]
            for event in models
            if event["kind"] == "tool_call_arguments_delta"
        )
        assert received_arguments == arguments
        assert json.loads(received_arguments) == {"text": text}
        assert (
            events[-2]["event"]["payload"]["response"]["choices"][0]["message"][
                "content"
            ]
            == text
        )
        assert events[-1]["final_output"] == text
        await backend.close()

    asyncio.run(scenario())
