"""Compaction while a native asynchronous tool is still running."""

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any, cast

import pytest

from agentlane.harness import AgentDescriptor, RunState
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.compaction import (
    CompactionReport,
    CompactionShim,
    CompactionShimConfig,
    DefaultCompactor,
    DefaultCompactorConfig,
    is_summary_item,
    render_request_messages,
)
from agentlane.models import (
    MessageDict,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call


def _token_estimator(messages: Sequence[MessageDict]) -> int:
    return sum(
        int(match)
        for message in messages
        for match in re.findall(r"\[t=(\d+)]", str(message.get("content", "")))
    )


class _PendingToolModel(SequenceModel):
    def __init__(
        self, *, evidence_tokens: int, release_automatically: bool = True
    ) -> None:
        super().__init__([make_assistant_response("summary [t=1]")])
        self.evidence_tokens = evidence_tokens
        self.release_automatically = release_automatically
        self.successor_ready = asyncio.Event()
        self.requests: list[list[MessageDict]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()
        self.executions = 0
        self.completed = 0
        self.call = make_tool_call(tool_id="pending", name="lookup", arguments="{}")

    @property
    def requires_streaming(self) -> bool:
        return True

    async def lookup(self) -> str:
        """Return the lookup result after the model finishes its other work."""
        self.executions += 1
        self.started.set()
        try:
            await self.release.wait()
            self.completed += 1

            return "lookup result [t=1]"
        finally:
            self.closed.set()

    async def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ModelStreamEvent]:
        del extra_call_args, schema, tools, cancellation_token, parent_span, kwargs
        self.requests.append(deepcopy(messages))
        number = len(self.requests)
        assert number <= 6, "The tool result must produce one final model turn."

        # Dispatch the real tool before committing its opening assistant response.
        if number == 1:
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER, async_tool_call=self.call
            )
            await self.started.wait()

        # Automatic turns accumulate evidence while the tool remains blocked.
        if number <= 4:
            assert self.completed == 0
            response = make_assistant_response(
                f"evidence {number} [t={self.evidence_tokens}]",
                tool_calls=[self.call] if number == 1 else None,
            )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response,
                continuation="automatic",
            )
            return

        # Finish the automatic successor before a new request can be prepared.
        if number == 5:
            assert not any(is_summary_item(message) for message in messages)
            self.successor_ready.set()
            if self.release_automatically:
                self.release.set()
            await self.closed.wait()

        yield ModelStreamEvent(
            kind=ModelStreamEventKind.COMPLETED,
            response=make_assistant_response("waiting" if number == 5 else "done"),
        )


def _assert_tool_history(messages: Sequence[MessageDict], *, delivered: bool) -> None:
    calls = [
        call["id"]
        for message in messages
        for call in cast(list[dict[str, object]], message.get("tool_calls") or [])
    ]
    results = [
        message["tool_call_id"] for message in messages if message.get("role") == "tool"
    ]
    assert calls == ["pending"]
    assert results == (["pending"] if delivered else [])
    if delivered:
        opener = next(
            i for i, message in enumerate(messages) if message.get("tool_calls")
        )
        result = next(
            i for i, message in enumerate(messages) if message.get("role") == "tool"
        )
        assert opener < result


def _agent(model: _PendingToolModel, reports: list[CompactionReport]) -> DefaultAgent:
    compaction = CompactionShim(
        CompactionShimConfig(context_window=100, trigger_tokens=100, on_failure="skip"),
        DefaultCompactor(
            DefaultCompactorConfig(
                keep_recent_tokens=1, keep_recent_messages=1, summary_max_tokens=2
            )
        ),
        estimator=_token_estimator,
        on_compact=reports.append,
    )

    return DefaultAgent(
        descriptor=AgentDescriptor(
            name="pending-tool",
            model=model,
            tools=Tools([model.lookup]),
            shims=(compaction,),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_tool_span", [False, True])
async def test_compaction_preserves_pending_tool_until_result_delivery(
    oversized_tool_span: bool,
) -> None:
    model = _PendingToolModel(evidence_tokens=25 if oversized_tool_span else 15)
    reports: list[CompactionReport] = []
    agent = _agent(model, reports)
    request: MessageDict = {
        "role": "user",
        "content": [{"type": "text", "text": "Keep this exact request. [t=2]"}],
    }

    state = RunState(instructions="system", history=[request], responses=[])
    if not oversized_tool_span:
        state.history.insert(
            0, {"role": "assistant", "content": "Earlier evidence [t=40]"}
        )

    # Consume every automatic response and settle the tool before request validation.
    try:
        async with asyncio.timeout(5):
            if oversized_tool_span:
                with pytest.raises(
                    RuntimeError, match="Final request exceeds.*103.*100"
                ):
                    await agent.run(state)
                assert len(model.requests) == 5
                assert model.completed == 1
                assert len(reports) == 1 and not reports[0].compacted
                assert reports[0].items_after == reports[0].items_before
                assert reports[0].dropped_items == ()
                assert reports[0].error is not None
                assert request in model.requests[-1]
                _assert_tool_history(model.requests[-1], delivered=False)
                assert model.calls == []

                return

            result = await agent.run(state)
    finally:
        assert model.executions == 1
        assert model.closed.is_set()

    assert result.final_output == "done"
    assert model.completed == 1
    assert len(model.requests) == 6
    assert any(
        report.compacted and report.signal.estimated_tokens > 100 for report in reports
    )
    for number, messages in enumerate(model.requests, start=1):
        assert messages.count(request) == 1

        # Only the initial and final turns submit requests; others consume successors.
        if number in (1, 6):
            assert _token_estimator(messages) <= 100
        if number > 1:
            _assert_tool_history(messages, delivered=number == 6)

    assert any(is_summary_item(message) for message in model.requests[-1])

    assert result.run_state is not None
    history = render_request_messages(None, result.run_state.history)
    _assert_tool_history(history, delivered=True)
    assert result.run_state.shim_state["compaction:latest-user"] == request


@pytest.mark.asyncio
async def test_cancellation_settles_pending_tool_after_oversized_successor() -> None:
    model = _PendingToolModel(evidence_tokens=25, release_automatically=False)
    reports: list[CompactionReport] = []
    agent = _agent(model, reports)
    token = CancellationToken()
    stream = await agent.run_stream("Request [t=2]", cancellation_token=token)

    try:
        async with asyncio.timeout(5):
            # The successor is already active, even though its history exceeds the window.
            await model.successor_ready.wait()
            assert _token_estimator(model.requests[-1]) == 102
            token.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
    finally:
        await stream.aclose()

    assert model.executions == 1
    assert model.completed == 0
    assert model.closed.is_set()
    assert model.calls == []
    assert reports == []
    assert len(model.requests) == 5
    _assert_tool_history(model.requests[-1], delivered=False)
