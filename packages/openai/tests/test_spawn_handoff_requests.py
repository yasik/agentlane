"""Spawn assignments through Responses request translation, with mocked transport."""

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest
from agentlane_openai import ResponsesClient
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import BaseModel

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunState,
    SpawnModelConfig,
    SpawnModelContext,
    SpawnModelSelection,
)
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.models import Config, ModelTracing
from agentlane.runtime import SingleThreadedRuntimeEngine
from agentlane.tracing import (
    CustomSpanData,
    DefaultTraceProvider,
    GenerationSpanData,
    Span,
    TracingProcessor,
    get_trace_provider,
    set_trace_provider,
    trace,
)


class _Choice(BaseModel):
    work_class: Literal["quick", "reasoning"]


class _EffortShim(Shim):
    @property
    def name(self) -> str:
        return "effective-effort"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.model_args = {"reasoning": {"effort": "medium"}}


class _Recorder(TracingProcessor):
    def __init__(self) -> None:
        self.spans: list[Span[Any]] = []

    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class _Stream:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[ResponseCompletedEvent]:
        yield ResponseCompletedEvent.model_construct(
            type="response.completed", response=self.response
        )

    async def close(self) -> None:
        self.closed = True


def _response(output: list[Any], model: str) -> Response:
    return Response.model_construct(
        id="resp_test",
        created_at=1,
        model=model,
        output=output,
        status="completed",
        usage=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_spawn_handoff_preserves_native_history_and_effective_effort(
    streaming: bool,
) -> None:
    parent = ResponsesClient(
        Config(
            api_key="test-key",
            model="parent-model",
            tracing=ModelTracing.ENABLED_WITHOUT_DATA,
        )
    )
    child = ResponsesClient(
        Config(
            api_key="test-key",
            model="child-model",
            tracing=ModelTracing.ENABLED_WITHOUT_DATA,
        )
    )
    arguments = json.dumps(
        {
            "input": {"task": "Finish this task"},
            "spawn_parameters": {"work_class": "quick"},
        }
    )
    parent_response = _response(
        [
            ResponseFunctionToolCall(
                id="fc_test",
                type="function_call",
                call_id="call_spawn",
                name="child",
                arguments=arguments,
                status="completed",
            ),
        ],
        "parent-served",
    )
    child_response = _response(
        [
            ResponseOutputMessage(
                id="msg_test",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text", text="finished", annotations=[]
                    )
                ],
            ),
        ],
        "child-served",
    )
    parent_stream = _Stream(parent_response)
    child_stream = _Stream(child_response)
    parent_create = AsyncMock(
        return_value=parent_stream if streaming else parent_response
    )
    child_create = AsyncMock(return_value=child_stream if streaming else child_response)
    cast(Any, parent)._openai_client.responses.create = parent_create
    cast(Any, child)._openai_client.responses.create = child_create

    class Provider:
        parameters_model = _Choice
        calls = 0

        def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
            self.calls += 1
            assert context.task == "Finish this task"
            return SpawnModelSelection(
                SpawnModelConfig(child, {"reasoning": {"effort": "low"}}), label="quick"
            )

    provider = Provider()
    runner = Runner(spawn_model_provider=provider)
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        descriptor=AgentDescriptor(
            name="Parent",
            model=parent,
            handoffs=(
                AgentDescriptor(
                    name="Child",
                    instructions="Child instructions",
                    shims=[_EffortShim()],
                ),
            ),
        ),
    )
    state = RunState(
        instructions="Parent instructions", history=["Original question"], responses=[]
    )
    recorder = _Recorder()
    tracing = DefaultTraceProvider()
    tracing.register_processor(recorder)
    previous = get_trace_provider()
    set_trace_provider(tracing)
    try:
        with trace("native-handoff"):
            if streaming:
                stream = runner.run_stream(agent, state)
                try:
                    result = await stream.result()
                finally:
                    await stream.aclose()
            else:
                result = await runner.run(agent, state)
    finally:
        set_trace_provider(previous)

    assert result.final_output == "finished"
    assert provider.calls == 1
    child_create.assert_awaited_once()
    payload = child_create.call_args.kwargs
    assert payload["model"] == "child-model"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["input"] == [
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Child instructions"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Original question"}],
        },
        {
            "id": "fc_test",
            "type": "function_call",
            "call_id": "call_spawn",
            "name": "child",
            "arguments": arguments,
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_spawn",
            "output": "Handoff accepted by child. Control transfers now.",
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Finish this task"}],
        },
    ]
    decisions = [
        span
        for span in recorder.spans
        if isinstance(span.span_data, CustomSpanData)
        and span.span_data.name == "spawn_model_decision"
    ]
    assert len(decisions) == 1
    assert decisions[0].span_data.data["model"] == "child-model"
    assert decisions[0].span_data.data["requested_effort"] == "low"
    generations = [
        span
        for span in recorder.spans
        if isinstance(span.span_data, GenerationSpanData)
        and span.span_data.model == "child-model"
    ]
    assert len(generations) == 1
    assert generations[0].span_data.model_config["requested_effort"] == "medium"
    assert generations[0].span_data.model_config["response_model"] == "child-served"
    if streaming:
        assert parent_stream.closed and child_stream.closed
