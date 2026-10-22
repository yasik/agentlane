import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    ExternalTool,
    JsonFileStateStore,
    Runner,
    RunnerHooks,
    RunPaused,
    RunResult,
    RunState,
    RunStream,
    Task,
    ToolResultInput,
)
from agentlane.harness._run import copy_run_state
from agentlane.harness.agents import AgentBase, DefaultAgent
from agentlane.harness.tools import HarnessToolDefinition, HarnessToolsShim
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.tracing import Span

from .tools_test_utils import EchoArgs, make_tool_call


def make_assistant_response(content: str) -> ModelResponse:
    """Build one canonical assistant response for default-agent tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_default_agent",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ],
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args
        del schema
        del tools
        del cancellation_token
        del kwargs

        self.calls.append([dict(message) for message in messages])
        if not self._outcomes:
            raise AssertionError("Expected one queued model response.")
        return self._outcomes.pop(0)


class _StreamingSequenceModel(Model[ModelResponse]):
    def __init__(
        self,
        outcomes: list[ModelResponse],
        *,
        wait_for_cancel: bool = False,
    ) -> None:
        self._outcomes = list(outcomes)
        self._wait_for_cancel = wait_for_cancel
        self.calls: list[list[MessageDict]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del messages, extra_call_args, schema, tools, cancellation_token, kwargs
        raise AssertionError("Streaming tests should call stream_response directly.")

    def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ):
        del extra_call_args, schema, tools, kwargs

        async def _stream():
            self.calls.append([dict(message) for message in messages])

            if self._wait_for_cancel:
                if cancellation_token is None:
                    raise AssertionError("Expected cancellation token for stream test.")
                await cancellation_token.wait_cancelled()
                raise asyncio.CancelledError()

            if not self._outcomes:
                raise AssertionError("Expected one queued model response.")

            response = self._outcomes.pop(0)
            content = response.choices[0].message.content or ""
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.TEXT_DELTA,
                text=content,
            )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response,
            )

        return _stream()


async def _collect_run_stream(stream: RunStream) -> list[ModelStreamEvent]:
    """Collect all streamed events from one harness run."""
    events: list[ModelStreamEvent] = []
    async for event in stream:
        events.append(event)
    return events


def _last_user_input(run_state: RunState) -> object:
    """Return the latest user-side item from one run state."""
    for item in reversed(run_state.history):
        if isinstance(item, ModelResponse):
            continue
        return item
    return None


class _RecordingRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RunState] = []

    async def run(
        self,
        agent: Task,
        state: RunState | ToolResultInput,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        del hooks
        del cancellation_token

        assert isinstance(state, RunState)
        copied = copy_run_state(state)
        self.calls.append(copied)

        if not isinstance(agent, Agent):
            raise AssertionError("Expected runtime-facing harness Agent.")

        state.turn_count += 1
        reply_text = f"{agent.name}:{_last_user_input(state)}"
        response = make_assistant_response(reply_text)
        state.responses.append(response)
        state.history.append(response)

        return RunResult(
            final_output=reply_text,
            responses=list(state.responses),
            turn_count=state.turn_count,
        )


class _NamedHooks(RunnerHooks):
    def __init__(self, name: str, events: list[tuple[str, str]]) -> None:
        self._name = name
        self._events = events

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        del task, state
        self._events.append((self._name, "agent_start"))

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        del task, result
        self._events.append((self._name, "agent_end"))


def test_default_agent_auto_manages_runtime_and_runner() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("Thanks, I can help.")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are a support agent.",
            )
        )

        result = await agent.run("My order arrived damaged.")

        assert result.final_output == "Thanks, I can help."
        assert result.turn_count == 1
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert model.calls == [
            [
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "My order arrived damaged."},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_accepts_hook_sequences() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("Thanks, I can help.")])
        events: list[tuple[str, str]] = []
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are a support agent.",
            ),
            hooks=(
                _NamedHooks("first", events),
                _NamedHooks("second", events),
            ),
        )

        result = await agent.run("My order arrived damaged.")

        assert result.final_output == "Thanks, I can help."
        assert events == [
            ("first", "agent_start"),
            ("second", "agent_start"),
            ("first", "agent_end"),
            ("second", "agent_end"),
        ]

    asyncio.run(scenario())


def test_default_agent_reuses_persisted_run_state_between_calls() -> None:
    async def scenario() -> None:
        model = _SequenceModel(
            [
                make_assistant_response("First answer."),
                make_assistant_response("Second answer."),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=model,
            )
        )

        first = await agent.run("first")
        second = await agent.run("second")

        assert first.turn_count == 1
        assert second.turn_count == 2
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 2
        assert model.calls == [
            [
                {"role": "user", "content": "first"},
            ],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "second"},
            ],
        ]

        agent.reset()
        assert agent.run_state is None

    asyncio.run(scenario())


def test_default_agent_supports_explicit_runtime_and_runner_injection() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Planner"),
            runtime=runtime,
            runner=runner,
        )

        result = await agent.run("draft a plan")

        assert result.final_output == "Planner:draft a plan"
        assert result.run_state is not None
        assert result.run_state.turn_count == 1
        assert len(runner.calls) == 1
        assert runner.calls[0] == RunState(
            instructions=None,
            history=["draft a plan"],
            responses=[],
            turn_count=0,
        )
        assert not runtime.is_running

    asyncio.run(scenario())


def test_default_agent_supports_subclass_descriptor_and_explicit_run_state_resume() -> (
    None
):
    async def scenario() -> None:
        runner = _RecordingRunner()

        class SupportAgent(DefaultAgent):
            descriptor = AgentDescriptor(name="Support")

        agent = SupportAgent(runner=runner)

        first = await agent.run("first")
        saved_state = first.run_state
        if saved_state is None:
            raise AssertionError("Expected saved run state from first run.")

        second = await agent.run(saved_state)

        assert second.final_output == "Support:first"
        assert second.run_state is not None
        assert second.run_state.turn_count == 2
        assert len(runner.calls) == 2
        assert runner.calls[0] == RunState(
            instructions=None,
            history=["first"],
            responses=[],
            turn_count=0,
        )
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1
        assert (
            second.run_state.responses[-1].choices[0].message.content == "Support:first"
        )

    asyncio.run(scenario())


def test_default_agent_implements_agent_base_contract() -> None:
    agent = DefaultAgent(descriptor=AgentDescriptor(name="Support"))

    assert isinstance(agent, AgentBase)


def test_default_agent_run_stream_emits_events_and_commits_state() -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel([make_assistant_response("streamed reply")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
                instructions="You are helpful.",
            )
        )

        stream = await agent.run_stream("hello")
        events = await _collect_run_stream(stream)
        result = await stream.result()

        assert [event.kind for event in events] == [
            ModelStreamEventKind.TEXT_DELTA,
            ModelStreamEventKind.COMPLETED,
        ]
        assert events[0].text == "streamed reply"
        assert result.final_output == "streamed reply"
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert model.calls == [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_run_stream_reuses_persisted_state() -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel(
            [
                make_assistant_response("first"),
                make_assistant_response("second"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
            )
        )

        first_stream = await agent.run_stream("one")
        await _collect_run_stream(first_stream)
        await first_stream.result()

        second_stream = await agent.run_stream("two")
        await _collect_run_stream(second_stream)
        second_result = await second_stream.result()

        assert second_result.turn_count == 2
        assert model.calls == [
            [{"role": "user", "content": "one"}],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "two"},
            ],
        ]

    asyncio.run(scenario())


def test_default_agent_stream_entry_points_persist_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        stream_model = _StreamingSequenceModel([make_assistant_response("first reply")])
        first = DefaultAgent(
            descriptor=AgentDescriptor(name="Support", model=stream_model),
            state_path=state_path,
        )

        model_stream = await first.run_stream("first question")
        await _collect_run_stream(model_stream)
        first_result = await model_stream.result()

        event_model = _SequenceModel([make_assistant_response("second reply")])
        second = DefaultAgent(
            descriptor=AgentDescriptor(name="Support", model=event_model),
            state_path=state_path,
        )
        event_stream = await second.run_events("second question")
        async for _ in event_stream:
            pass
        second_result = await event_stream.result()

        stored = JsonFileStateStore(state_path).load()
        assert first_result.run_state is not None
        assert first_result.run_state.revision == 1
        assert second_result.run_state is not None
        assert second_result.run_state.revision == 2
        assert stored is not None
        assert stored.revision == 2
        assert event_model.calls == [
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second question"},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_run_stream_close_cancels_without_committing_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        model = _StreamingSequenceModel([], wait_for_cancel=True)
        state_path = tmp_path / "assistant.json"
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Support",
                model=model,
            ),
            state_path=state_path,
        )

        stream = await agent.run_stream("hello")
        try:
            agent.reset()
        except RuntimeError as exc:
            assert "primary run is active" in str(exc)
        else:
            raise AssertionError("Expected reset to reject the scheduled stream.")
        await asyncio.sleep(0)
        await stream.aclose()

        try:
            await stream.result()
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected cancelled stream result.")

        assert agent.run_state is None
        assert not state_path.exists()

    asyncio.run(scenario())


def test_default_agent_fork_branches_without_mutating_persisted_state() -> None:
    async def scenario() -> None:
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Support"),
            runner=runner,
        )

        first = await agent.run("first")
        persisted_after_run = copy_run_state(agent.run_state)
        if persisted_after_run is None:
            raise AssertionError("Expected persisted state after first run.")

        branch = await agent.fork("branch")

        if branch.run_state is None:
            raise AssertionError("Expected fork result to expose branch run state.")
        assert first.run_state is not None
        assert branch.final_output == "Support:branch"
        assert branch.run_state.turn_count == 2
        assert agent.run_state == persisted_after_run
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert len(runner.calls) == 2
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1
        assert runner.calls[1].history[2] == "branch"

    asyncio.run(scenario())


def test_default_agent_fork_supports_explicit_run_state_without_persisting_it() -> None:
    async def scenario() -> None:
        runner = _RecordingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Support"),
            runner=runner,
        )

        first = await agent.run("first")
        saved_state = first.run_state
        if saved_state is None:
            raise AssertionError("Expected saved run state from first run.")

        branch = await agent.fork(saved_state)

        assert branch.final_output == "Support:first"
        assert branch.run_state is not None
        assert branch.run_state.turn_count == 2
        assert agent.run_state is not None
        assert agent.run_state.turn_count == 1
        assert len(runner.calls) == 2
        assert runner.calls[1].history[0] == "first"
        assert runner.calls[1].turn_count == 1

    asyncio.run(scenario())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "run_stream", "run_events"])
@pytest.mark.parametrize("same_instance", [False, True])
async def test_default_agent_pause_and_resume_preserve_commit_contract(
    mode: str,
    same_instance: bool,
    tmp_path: Path,
) -> None:
    root = AgentId.from_values("test", "external")
    pending = make_assistant_response("Choose a city.")
    pending.choices[0].message.tool_calls = [
        make_tool_call(
            tool_id="question", name="external", arguments='{"text":"City?"}'
        )
    ]

    tools = Tools(
        tools=[
            ExternalTool(
                name="external", description="Ask the host.", args_model=EchoArgs
            )
        ]
    )
    baseline = RunState(instructions=None, history=["prior"], responses=[], revision=7)
    model = _SequenceModel(
        [pending, make_assistant_response("done")] if same_instance else [pending]
    )
    agent = DefaultAgent(
        agent_id=root,
        run_state=baseline,
        descriptor=AgentDescriptor(name="test", model=model, tools=tools),
        runner=Runner(),
    )

    # Both stream consumption and result retrieval retain the same typed signal.
    if mode == "run":
        with pytest.raises(RunPaused) as paused:
            await agent.run("start")
    else:
        stream = await getattr(agent, mode)("start")
        try:
            with pytest.raises(RunPaused) as streamed:
                async for _ in stream:
                    pass
            with pytest.raises(RunPaused) as paused:
                await stream.result()
            assert streamed.value is paused.value
        finally:
            await stream.aclose()
    assert agent.run_state == baseline
    snapshot = agent.snapshot()
    assert snapshot is not None and snapshot.revision == 7
    continuation = paused.value.continuation
    assert continuation.checkpoint.revision == 7

    # The resumed execution commits once and adds no new user message.
    resumed_model = (
        model if same_instance else _SequenceModel([make_assistant_response("done")])
    )
    resumed = (
        agent
        if same_instance
        else DefaultAgent(
            agent_id=root,
            run_state=baseline,
            descriptor=AgentDescriptor(name="test", model=resumed_model, tools=tools),
            runner=Runner(),
        )
    )
    run_input = ToolResultInput(continuation=continuation, result="Prague")
    if mode == "run":
        result = await resumed.run(run_input)
    else:
        stream = await getattr(resumed, mode)(run_input)
        async for _ in stream:
            pass
        result = await stream.result()
    assert result.final_output == "done"
    assert resumed.run_state is not None and resumed.run_state.revision == 8
    assert resumed_model.calls[-1][-1] == {
        "role": "tool",
        "tool_call_id": "question",
        "name": "external",
        "content": "Prague",
    }

    # A completed run invalidates the old revision on this instance.
    completed_state = resumed.run_state
    model_calls = len(resumed_model.calls)
    if mode == "run":
        with pytest.raises(ValueError, match="Continuation revision"):
            await resumed.run(run_input)
    else:
        stream = await getattr(resumed, mode)(run_input)
        try:
            with pytest.raises(ValueError, match="Continuation revision"):
                await stream.result()
        finally:
            await stream.aclose()
    assert resumed.run_state == completed_state
    assert len(resumed_model.calls) == model_calls

    stored = DefaultAgent(
        agent_id=root,
        descriptor=AgentDescriptor(name="test"),
        state_path=tmp_path / "agent.json",
    )
    with pytest.raises(ValueError, match="host-managed continuation storage"):
        await stored.run(run_input)
    assert stored.run_state is None
    assert not (tmp_path / "agent.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "run_stream", "run_events", "fork"])
async def test_default_agent_rejects_dynamic_external_tools_with_store_or_fork(
    mode: str,
    tmp_path: Path,
) -> None:
    pending = make_assistant_response("Choose a city.")
    pending.choices[0].message.tool_calls = [
        make_tool_call(
            tool_id="question", name="external", arguments='{"text":"City?"}'
        )
    ]
    model = _SequenceModel([pending])
    source = HarnessToolsShim(
        definitions=(
            HarnessToolDefinition(
                tool=ExternalTool(
                    name="external", description="Ask the host.", args_model=EchoArgs
                )
            ),
        )
    )
    state_path = tmp_path / "agent.json"
    agent = DefaultAgent(
        descriptor=AgentDescriptor(name="test", model=model, shims=(source,)),
        state_path=None if mode == "fork" else state_path,
    )
    message = "forked runs" if mode == "fork" else "host-managed continuation storage"
    if mode in {"run", "fork"}:
        with pytest.raises(ValueError, match=message):
            await getattr(agent, mode)("start")
    else:
        stream = await getattr(agent, mode)("start")
        try:
            with pytest.raises(ValueError, match=message):
                async for _ in stream:
                    pass
            with pytest.raises(ValueError, match=message):
                await stream.result()
        finally:
            await stream.aclose()
    assert agent.run_state is None
    assert not state_path.exists()
