"""Public run contracts backed by a fake native SDK connection."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from agentlane_openai import ResponsesClient
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemDoneEvent,
    ResponseSteerAcceptedEvent,
    ResponseSteerFailedEvent,
    ResponseSteerPendingEvent,
    ResponseTextDeltaEvent,
)

from agentlane.harness import AgentDescriptor, RunResult
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.models import Config, ModelBehaviorError, RateLimiter, Tool, Tools
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken


def make_response(text: str, response_id: str = "resp_1", **extra: Any) -> Response:
    return Response.model_validate(
        {
            "id": response_id,
            "created_at": 1,
            "object": "response",
            "model": "test",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            ],
            **extra,
        }
    )


class Connection:
    def __init__(self) -> None:
        self.events: asyncio.Queue[object] = asyncio.Queue()
        self.response = MagicMock(create=AsyncMock(), steer=AsyncMock())
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True
        self.events.put_nowait(None)

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        if isinstance(event, Exception):
            raise event

        return event


@pytest.mark.asyncio
async def test_native_connections_are_run_local_and_close_on_failure() -> None:
    client = ResponsesClient(Config(api_key="test", model="test"), websocket=True)
    connections: list[Connection] = []

    # Give concurrent runs separate connections, with one success and one failure.
    @asynccontextmanager
    async def connect(**kwargs: Any) -> AsyncGenerator[Connection, None]:
        assert kwargs == {"max_retries": 0}
        connection = Connection()
        connections.append(connection)
        index = len(connections)
        connection.events.put_nowait(
            ResponseCompletedEvent(
                type="response.completed",
                response=make_response(f"answer {index}"),
                sequence_number=0,
            )
            if index == 1
            else RuntimeError("connection lost")
        )
        try:
            yield connection
        finally:
            connection.closed = True

    cast(Any, client)._openai_client.responses.connect = connect

    async def run() -> object:
        agent = DefaultAgent(descriptor=AgentDescriptor(name="live", model=client))

        return (await agent.run("hello")).final_output

    # Run both agents against the same client and verify independent cleanup.
    results = await asyncio.gather(run(), run(), return_exceptions=True)

    assert results[0] == "answer 1"
    assert isinstance(results[1], RuntimeError)
    assert len(connections) == 2 and all(c.closed for c in connections)
    assert all(c.response.create.await_count == 1 for c in connections)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["completed", "steered", "disconnect", "rejected", "rejected_before_acceptance"],
)
async def test_steering_receipts_commit_only_on_successor(outcome: str) -> None:
    rejected = outcome.startswith("rejected")
    limiter = MagicMock(spec=RateLimiter)
    client = ResponsesClient(
        Config(api_key="test", model="test", rate_limiter=limiter), websocket=True
    )
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    connection = Connection()
    parent = make_response("parent", usage=usage)
    connection.events.put_nowait(
        ResponseCreatedEvent(
            type="response.created", response=parent, sequence_number=0
        )
    )

    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    agent = DefaultAgent(descriptor=AgentDescriptor(name="live", model=client))
    stream = await agent.run_stream("hello")
    async with asyncio.timeout(3):
        await anext(stream)
        receipt = await stream.steer("use the new direction")

        assert str(receipt.status) == "submitted"
        connection.response.steer.assert_awaited_once_with(
            input="use the new direction", previous_response_id="resp_1"
        )

        # Acceptance alone must leave the input pending until a successor starts.
        if outcome != "rejected_before_acceptance":
            connection.events.put_nowait(
                ResponseSteerAcceptedEvent.model_validate(
                    {
                        "type": "response.steer.accepted",
                        "sequence_number": 1,
                        "steer": {"id": "steer_1", "previous_response_id": "resp_1"},
                    }
                )
            )
            accepted = await anext(stream)

            assert accepted.steering == [receipt]
            assert str(receipt.status) == "accepted"

        # Exercise connection loss separately from a server rejection or successor.
        if outcome == "disconnect":
            connection.events.put_nowait(RuntimeError("disconnected"))
            with pytest.raises(RuntimeError, match="disconnected"):
                await stream.result()

            assert str(receipt.status) == "unknown"
        else:
            if rejected:
                connection.events.put_nowait(
                    ResponseSteerFailedEvent.model_validate(
                        {
                            "type": "response.steer.failed",
                            "sequence_number": 2,
                            "steer": {
                                "id": "steer_1",
                                "previous_response_id": "resp_1",
                                "input": "use the new direction",
                            },
                            "error": {
                                "code": "successor_creation_failed",
                                "message": "failed",
                                "type": "invalid_request_error",
                            },
                        }
                    )
                )

            # Finish the parent with either a normal answer or a steering interruption.
            if outcome == "steered":
                parent = make_response(
                    "parent",
                    status="incomplete",
                    incomplete_details={"reason": "steered"},
                    usage=usage,
                )
                connection.events.put_nowait(
                    ResponseIncompleteEvent(
                        type="response.incomplete", response=parent, sequence_number=3
                    )
                )
            else:
                connection.events.put_nowait(
                    ResponseCompletedEvent(
                        type="response.completed", response=parent, sequence_number=3
                    )
                )

            # Only successful steering creates a successor and commits the new input.
            if not rejected:
                child = make_response(
                    "child", "resp_2", previous_response_id="resp_1", usage=usage
                )
                connection.events.put_nowait(
                    ResponseCreatedEvent(
                        type="response.created", response=child, sequence_number=4
                    )
                )
                connection.events.put_nowait(
                    ResponseCompletedEvent(
                        type="response.completed", response=child, sequence_number=5
                    )
                )

            # Check the public result and durable history after the protocol settles.
            result = await stream.result()

            assert result.final_output == ("parent" if rejected else "child")
            assert str(receipt.status) == ("failed" if rejected else "applied")
            assert result.run_state is not None
            assert (
                {"role": "user", "content": "use the new direction"}
                in result.run_state.history
                if not rejected
                else len(result.responses) == 1
            )

        # Each provider terminal is charged once, even if steering delays completion.
        expected_responses = 0 if outcome == "disconnect" else (1 if rejected else 2)
        assert [call.args[0] for call in limiter.record_usage.call_args_list] == [
            15
        ] * expected_responses

        assert connection.closed
        assert connection.response.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_send", [False, True])
async def test_uncertain_steering_send_stops_the_run(cancel_send: bool) -> None:
    connection = Connection()
    client = ResponsesClient(Config(api_key="test", model="test"), websocket=True)
    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    parent = make_response("parent")
    connection.events.put_nowait(
        ResponseCreatedEvent(
            type="response.created", response=parent, sequence_number=0
        )
    )

    # Stop the send after it begins so failure cannot prove the server saw nothing.
    sent = asyncio.Event()

    async def send(**kwargs: Any) -> None:
        sent.set()
        if cancel_send:
            await asyncio.Event().wait()

        raise OSError("send outcome unknown")

    connection.response.steer.side_effect = send
    stream = await DefaultAgent(
        descriptor=AgentDescriptor(name="live", model=client)
    ).run_stream("hello")
    async with asyncio.timeout(3):
        await anext(stream)

        # Fail or cancel the in-flight send before supplying late server events.
        steering = asyncio.create_task(stream.steer("new direction"))
        await sent.wait()
        if cancel_send:
            steering.cancel()

        with pytest.raises(asyncio.CancelledError if cancel_send else OSError):
            await steering

        connection.events.put_nowait(
            ResponseSteerAcceptedEvent.model_validate(
                {
                    "type": "response.steer.accepted",
                    "sequence_number": 1,
                    "steer": {"id": "steer_1", "previous_response_id": parent.id},
                }
            )
        )
        connection.events.put_nowait(
            ResponseCompletedEvent(
                type="response.completed", response=parent, sequence_number=2
            )
        )

        # Late events must not turn an uncertain submission into a successful run.
        with pytest.raises(RuntimeError):
            await stream.result()

    assert connection.closed
    assert connection.response.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_steering", [False, True])
async def test_handoff_routes_controls_or_rejects_pending_steering(
    pending_steering: bool,
) -> None:
    client = ResponsesClient(Config(api_key="test", model="test"), websocket=True)
    parent, child = Connection(), Connection()
    connect = MagicMock(side_effect=[parent, child])
    cast(Any, client)._openai_client.responses.connect = connect
    parent_response = make_response("parent")

    async def create_parent(**kwargs: Any) -> None:
        nonlocal parent_response
        parent_response = make_response(
            "",
            output=[
                {
                    "type": "function_call",
                    "call_id": "handoff_1",
                    "name": kwargs["tools"][0]["name"],
                    "arguments": '{"task":"continue"}',
                }
            ],
        )
        parent.events.put_nowait(
            ResponseCreatedEvent(
                type="response.created",
                response=parent_response,
                sequence_number=0,
            )
        )

    # Prepare separate parent and child connections to observe control routing.
    parent.response.create.side_effect = create_parent
    child_response = make_response("child", "resp_child")
    child.events.put_nowait(
        ResponseCreatedEvent(
            type="response.created",
            response=child_response,
            sequence_number=0,
        )
    )
    stream = await DefaultAgent(
        descriptor=AgentDescriptor(
            name="parent",
            model=client,
            handoffs=(AgentDescriptor(name="child", model=client),),
        )
    ).run_stream("hello")
    async with asyncio.timeout(3):
        await anext(stream)

        # Optionally leave steering pending before the parent requests a handoff.
        receipt = None
        if pending_steering:
            receipt = await stream.steer("new direction")
            parent.events.put_nowait(
                ResponseSteerAcceptedEvent.model_validate(
                    {
                        "type": "response.steer.accepted",
                        "sequence_number": 1,
                        "steer": {
                            "id": "steer_1",
                            "previous_response_id": parent_response.id,
                        },
                    }
                )
            )

        parent.events.put_nowait(
            ResponseCompletedEvent(
                type="response.completed",
                response=parent_response,
                sequence_number=2,
            )
        )
        if pending_steering:
            parent.events.put_nowait(
                ResponseSteerPendingEvent.model_validate(
                    {
                        "type": "response.steer.pending",
                        "sequence_number": 3,
                        "steer": {
                            "id": "steer_1",
                            "previous_response_id": parent_response.id,
                        },
                        "reason": "waiting_for_required_input",
                        "required_input": [
                            {
                                "type": "function_call_output",
                                "call_id": "handoff_1",
                                "name": cast(
                                    ResponseFunctionToolCall, parent_response.output[0]
                                ).name,
                            }
                        ],
                    }
                )
            )

            # Let a broken handoff finish instead of masking it with a timeout.
            child.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed",
                    response=child_response,
                    sequence_number=1,
                )
            )
            with pytest.raises(ModelBehaviorError, match="steering"):
                await stream.result()

            assert receipt is not None and str(receipt.status) == "unknown"
            assert connect.call_count == 1
        else:
            # After transfer, wait for the child response before steering it.
            async for event in stream:
                if event.provider_event_type == "response.created":
                    break

            receipt = await stream.steer("new direction")
            child.response.steer.assert_awaited_once_with(
                input="new direction",
                previous_response_id=child_response.id,
            )
            child.events.put_nowait(
                ResponseSteerAcceptedEvent.model_validate(
                    {
                        "type": "response.steer.accepted",
                        "sequence_number": 1,
                        "steer": {
                            "id": "steer_2",
                            "previous_response_id": child_response.id,
                        },
                    }
                )
            )
            child.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed",
                    response=child_response,
                    sequence_number=2,
                )
            )
            successor = make_response(
                "new answer", "resp_successor", previous_response_id=child_response.id
            )
            child.events.put_nowait(
                ResponseCreatedEvent(
                    type="response.created",
                    response=successor,
                    sequence_number=3,
                )
            )
            child.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed",
                    response=successor,
                    sequence_number=4,
                )
            )

            assert (await stream.result()).final_output == "new answer"
            assert str(receipt.status) == "applied"
            assert child.closed

    assert parent.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_queued_steering_uses_one_tool_continuation(asynchronous: bool) -> None:
    executions: list[str] = []

    async def echo(text: str) -> str:
        executions.append(text)

        return text

    client = ResponsesClient(
        Config(api_key="test", model="test"),
        websocket=True,
        service_tier="priority",
        async_tools=["echo"] if asynchronous else [],
    )
    connection = Connection()
    call = ResponseFunctionToolCall.model_validate(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"ok"}',
            "async": asynchronous,
        }
    )
    parent = make_response("", output=[call.model_dump(by_alias=True)])
    child = make_response("finished", "resp_2", previous_response_id="resp_1")

    # Require exactly one tool-output request to resume the queued steering input.
    async def create(**kwargs: Any) -> None:
        if connection.response.create.await_count == 1:
            connection.events.put_nowait(
                ResponseCreatedEvent(
                    type="response.created", response=parent, sequence_number=0
                )
            )
        else:
            assert kwargs["previous_response_id"] == "resp_1"
            assert kwargs["input"] == [
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
            ]
            assert kwargs["service_tier"] == "priority"
            connection.events.put_nowait(
                ResponseCreatedEvent(
                    type="response.created", response=child, sequence_number=5
                )
            )
            connection.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed", response=child, sequence_number=6
                )
            )

    connection.response.create.side_effect = create

    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="live", model=client, tools=Tools([echo])),
    )
    stream = await agent.run_stream("hello")
    async with asyncio.timeout(3):
        await anext(stream)
        native_input = [{"type": "message", "role": "user", "content": "new direction"}]
        receipt = await stream.steer(native_input)
        connection.events.put_nowait(
            ResponseSteerAcceptedEvent.model_validate(
                {
                    "type": "response.steer.accepted",
                    "sequence_number": 1,
                    "steer": {"id": "steer_1", "previous_response_id": "resp_1"},
                }
            )
        )

        # Exercise the same continuation with work dispatched before or after
        # completion.
        if asynchronous:
            connection.events.put_nowait(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    output_index=0,
                    item=call,
                    sequence_number=2,
                )
            )

        connection.events.put_nowait(
            ResponseCompletedEvent(
                type="response.completed", response=parent, sequence_number=2
            )
        )
        connection.events.put_nowait(
            ResponseSteerPendingEvent.model_validate(
                {
                    "type": "response.steer.pending",
                    "sequence_number": 3,
                    "steer": {"id": "steer_1", "previous_response_id": "resp_1"},
                    "reason": "waiting_for_required_input",
                    "required_input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "name": "echo",
                        }
                    ],
                }
            )
        )

        # Check single execution, receipt application, and input order in run history.
        result = await stream.result()

        assert executions == ["ok"]
        assert connection.response.create.await_count == 2
        assert str(receipt.status) == "applied"
        assert result.final_output == "finished"
        assert result.run_state is not None
        assert result.run_state.history[2] == {
            "role": "user",
            "openai_responses_input": native_input,
        }


@pytest.mark.asyncio
async def test_async_tools_overlap_stream_and_deliver_each_call_once() -> None:
    # Control completion order without sleeps so the test exposes overlap reliably.
    started = {name: asyncio.Event() for name in ("slow", "fast")}
    release = {name: asyncio.Event() for name in started}
    executions: list[str] = []
    fast_submitted = asyncio.Event()

    async def lookup(name: str) -> str:
        executions.append(name)
        started[name].set()
        await release[name].wait()

        return name

    calls = [
        ResponseFunctionToolCall.model_validate(
            {
                "type": "function_call",
                "call_id": name,
                "name": "lookup",
                "arguments": json.dumps({"name": name}),
                "async": True,
            }
        )
        for name in started
    ]
    parent = make_response(
        "working", output=[call.model_dump(by_alias=True) for call in calls]
    )
    middle = make_response("fast received", "resp_2", previous_response_id="resp_1")
    final = make_response("all received", "resp_3", previous_response_id="resp_2")
    connection = Connection()
    client = ResponsesClient(
        Config(api_key="test", model="test"), websocket=True, async_tools=["lookup"]
    )

    async def create(**kwargs: Any) -> None:
        count = connection.response.create.await_count
        if count == 1:
            assert kwargs["tools"][0]["async"] is True
            connection.events.put_nowait(
                ResponseCreatedEvent(
                    type="response.created", response=parent, sequence_number=0
                )
            )

            # Replay one completed call event to check that it does not execute twice.
            for index, call in enumerate([*calls, calls[0]]):
                connection.events.put_nowait(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        output_index=index,
                        item=call,
                        sequence_number=index + 1,
                    )
                )
        else:
            name = "fast" if count == 2 else "slow"

            assert kwargs["previous_response_id"] == (
                "resp_1" if count == 2 else "resp_2"
            )
            assert kwargs["input"] == [
                {"type": "function_call_output", "call_id": name, "output": name}
            ]
            assert kwargs["tools"] == [] and kwargs["tool_choice"] == "none"
            response = middle if count == 2 else final
            connection.events.put_nowait(
                ResponseCreatedEvent(
                    type="response.created", response=response, sequence_number=6
                )
            )
            connection.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed", response=response, sequence_number=7
                )
            )
            if count == 2:
                fast_submitted.set()

    connection.response.create.side_effect = create

    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="live",
            model=client,
            tools=Tools([lookup], tool_call_limits={"lookup": 2}),
        ),
    )
    stream = await agent.run_stream("hello")
    async with asyncio.timeout(3):
        # Hold both tools open while the model continues to produce visible text.
        await asyncio.gather(*(event.wait() for event in started.values()))
        connection.events.put_nowait(
            ResponseTextDeltaEvent(
                type="response.output_text.delta",
                item_id="msg_1",
                output_index=2,
                content_index=0,
                delta="independent answer",
                logprobs=[],
                sequence_number=4,
            )
        )
        async for event in stream:
            if event.text == "independent answer":
                break

        assert executions == ["slow", "fast"] and not any(
            e.is_set() for e in release.values()
        )
        connection.events.put_nowait(
            ResponseCompletedEvent(
                type="response.completed", response=parent, sequence_number=5
            )
        )

        # Deliver the fast result first without waiting for the slow tool.
        release["fast"].set()
        await fast_submitted.wait()

        assert not release["slow"].is_set()

        release["slow"].set()
        result = await stream.result()

    assert result.final_output == "all received"
    assert connection.response.create.await_count == 3
    assert executions == ["slow", "fast"]
    assert connection.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_run_joins_tools_before_shim_cleanup(cancel: bool) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    cleanup: list[str] = []

    async def slow() -> str:
        started.set()
        try:
            await asyncio.Event().wait()

            return "unreachable"
        finally:
            stopped.set()

    # The end hook must observe tool cleanup already complete.
    class Cleanup(Shim):
        @property
        def name(self) -> str:
            return "cleanup"

        async def on_run_end(
            self, result: RunResult | None, transient_state: RunContext[Any]
        ) -> None:
            assert stopped.is_set()
            cleanup.append("end")

    connection = Connection()
    client = ResponsesClient(
        Config(api_key="test", model="test"), websocket=True, async_tools=["slow"]
    )
    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    parent = make_response("working")
    connection.events.put_nowait(
        ResponseCreatedEvent(
            type="response.created", response=parent, sequence_number=0
        )
    )
    call = ResponseFunctionToolCall.model_validate(
        {
            "type": "function_call",
            "call_id": "slow_call",
            "name": "slow",
            "arguments": "{}",
            "async": True,
        }
    )
    connection.events.put_nowait(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            item=call,
            output_index=0,
            sequence_number=1,
        )
    )
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="live", model=client, tools=Tools([slow]), shims=[Cleanup()]
        )
    )
    token = CancellationToken()
    stream = await agent.run_stream("hello", cancellation_token=token)
    async with asyncio.timeout(3):
        # Interrupt only after the tool starts, with a steering submission still
        # pending.
        await started.wait()
        receipt = await stream.steer("change direction")
        if cancel:
            token.cancel()
        else:
            connection.events.put_nowait(RuntimeError("connection lost"))

        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await stream.result()

    assert stopped.is_set() and cleanup == ["end"]
    assert str(receipt.status) == "unknown"
    assert connection.closed
    assert connection.response.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude", [False, True])
async def test_automatic_successor_keeps_handler_and_checks_permissions(
    exclude: bool,
) -> None:
    executions: list[str] = []

    async def original() -> str:
        executions.append("original")

        return "ok"

    async def replacement() -> str:
        raise AssertionError("A later catalog replaced the inherited handler.")

    # Change the next catalog to distinguish inherited handlers from current permission.
    class Catalog(Shim):
        @property
        def name(self) -> str:
            return "catalog"

        async def prepare_turn(self, turn: PreparedTurn) -> None:
            handler = original if turn.run_state.turn_count == 1 else replacement
            turn.tools = Tools([Tool.from_function(handler, name="lookup")])
            if exclude and turn.run_state.turn_count > 1:
                turn.exclude_tools(frozenset({"lookup"}))

    connection = Connection()
    client = ResponsesClient(
        Config(api_key="test", model="test"), websocket=True, async_tools=["lookup"]
    )
    cast(Any, client)._openai_client.responses.connect = MagicMock(
        return_value=connection
    )
    parent = make_response("parent")
    call = ResponseFunctionToolCall.model_validate(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "async": True,
        }
    )
    child = make_response(
        "child",
        "resp_2",
        previous_response_id=parent.id,
        output=[call.model_dump(by_alias=True)],
    )
    connection.events.put_nowait(
        ResponseCreatedEvent(
            type="response.created",
            response=parent,
            sequence_number=0,
        )
    )

    async def create(**kwargs: Any) -> None:
        if connection.response.create.await_count == 2:
            assert kwargs["previous_response_id"] == child.id
            assert kwargs["input"] == [
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
            ]
            connection.events.put_nowait(
                ResponseCompletedEvent(
                    type="response.completed",
                    response=make_response("done", "resp_3"),
                    sequence_number=6,
                )
            )

    connection.response.create.side_effect = create
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="live",
            model=client,
            shims=[Catalog()],
        )
    )
    stream = await agent.run_stream("hello")
    async with asyncio.timeout(3):
        await anext(stream)
        receipt = await stream.steer("new direction")

        # Start the successor automatically before its async call reaches the harness.
        for event in [
            ResponseSteerAcceptedEvent.model_validate(
                {
                    "type": "response.steer.accepted",
                    "sequence_number": 1,
                    "steer": {"id": "steer_1", "previous_response_id": parent.id},
                }
            ),
            ResponseCompletedEvent(
                type="response.completed", response=parent, sequence_number=2
            ),
            ResponseCreatedEvent(
                type="response.created", response=child, sequence_number=3
            ),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=call,
                output_index=0,
                sequence_number=4,
            ),
            ResponseCompletedEvent(
                type="response.completed", response=child, sequence_number=5
            ),
        ]:
            connection.events.put_nowait(event)

        # Keep the original handler only while the new turn still permits its tool.
        if exclude:
            with pytest.raises(ModelBehaviorError, match="permitted local handler"):
                await stream.result()

            assert executions == []
        else:
            assert (await stream.result()).final_output == "done"
            assert executions == ["original"]

        assert str(receipt.status) == "applied"

    assert connection.closed
