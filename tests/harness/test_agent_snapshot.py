import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import AgentDescriptor, AgentSnapshot, RunState
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    ModelResponse,
    PromptSpec,
    PromptTemplate,
    Tools,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span

from .tools_test_utils import SequenceModel, make_assistant_response


class _StructuredContent(BaseModel):
    value: str


_AGENT_ID = AgentId.from_values("assistant", "main")


def _snapshot(run_state: RunState | None = None) -> AgentSnapshot:
    return AgentSnapshot.capture(
        agent_id=_AGENT_ID,
        run_state=(
            run_state
            if run_state is not None
            else RunState(instructions=None, history=[], responses=[])
        ),
    )


def test_agent_snapshot_round_trip_preserves_canonical_state() -> None:
    instructions = PromptSpec(
        template=PromptTemplate[dict[str, str], str](
            system_template="You support {{ team }}.",
        ),
        values={"team": "ops"},
    )
    user_prompt = PromptSpec(
        template=PromptTemplate[dict[str, str], str](
            user_template="Follow up with {{ team }}.",
        ),
        values={"team": "ops"},
    )
    response = make_assistant_response("First answer")
    state = RunState(
        instructions=instructions,
        history=["First question", response, user_prompt],
        responses=[response],
        turn_count=2,
    )
    state.shim_state["skills:active-skill-names"] = ["triage"]
    snapshot = _snapshot(state)

    encoded = snapshot.to_json()
    decoded = AgentSnapshot.from_json(json.loads(json.dumps(encoded)))

    assert decoded.agent_id == snapshot.agent_id
    restored = decoded.to_run_state()
    assert restored.instructions == "You support ops."
    assert restored.history == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow up with ops."},
    ]
    assert restored.responses == [response]
    assert restored.shim_state == {"skills:active-skill-names": ["triage"]}
    assert restored.turn_count == 2


def test_agent_snapshot_renders_structured_content_for_model_reuse() -> None:
    snapshot = _snapshot(
        RunState(
            instructions=None,
            history=[_StructuredContent(value="kept")],
            responses=[],
        )
    )

    restored = AgentSnapshot.from_json(snapshot.to_json()).to_run_state()

    assert restored.history == [{"role": "user", "content": '{"value":"kept"}'}]


def test_agent_snapshot_decode_tolerates_unknown_fields() -> None:
    snapshot = _snapshot()
    payload = snapshot.to_json()
    payload["future"] = {"ignored": True}
    state = payload["state"]
    assert isinstance(state, dict)
    state["future"] = "ignored"

    decoded = AgentSnapshot.from_json(payload)

    assert decoded.agent_id == snapshot.agent_id
    assert decoded.to_run_state() == snapshot.to_run_state()


def test_agent_snapshot_decode_rejects_unsupported_version() -> None:
    snapshot = _snapshot()
    payload = snapshot.to_json()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        AgentSnapshot.from_json(payload)


@pytest.mark.parametrize("field", ["schema_version", "created_at"])
def test_agent_snapshot_decode_requires_envelope_fields(field: str) -> None:
    payload = _snapshot().to_json()
    del payload[field]

    with pytest.raises(ValueError, match=field):
        AgentSnapshot.from_json(payload)


def test_agent_snapshot_decode_rejects_malformed_history() -> None:
    snapshot = _snapshot()
    payload = snapshot.to_json()
    state = payload["state"]
    assert isinstance(state, dict)
    state["history"] = [{"content": "missing role"}]

    with pytest.raises(ValueError, match=r"history\[0\]\.role"):
        AgentSnapshot.from_json(payload)


def test_agent_snapshot_isolates_captured_state() -> None:
    state = RunState(
        instructions=None,
        history=[{"role": "user", "content": "original"}],
        responses=[],
    )
    state.shim_state["nested"] = ["original"]
    snapshot = _snapshot(state)

    state.history.append({"role": "user", "content": "later"})
    nested = state.shim_state["nested"]
    assert isinstance(nested, list)
    cast(list[object], nested).append("later")

    restored = snapshot.to_run_state()
    assert restored.history == [{"role": "user", "content": "original"}]
    assert restored.shim_state == {"nested": ["original"]}

    restored.history.append({"role": "user", "content": "restored mutation"})
    assert snapshot.to_run_state().history == [{"role": "user", "content": "original"}]


def test_agent_snapshot_capture_rejects_non_json_shim_state_with_field_path() -> None:
    state = RunState(instructions=None, history=[], responses=[])
    state.shim_state["invalid"] = object()

    with pytest.raises(ValueError, match=r"shim_state\.invalid"):
        _snapshot(state)


def test_agent_snapshot_golden_fixture_v1_decodes() -> None:
    fixture = Path(__file__).with_name("fixtures") / "agent_snapshot_v1.json"

    snapshot = AgentSnapshot.from_json(json.loads(fixture.read_text(encoding="utf-8")))

    assert snapshot.agent_id == _AGENT_ID
    assert snapshot.to_run_state().history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_default_agent_snapshot_restores_and_continues_same_address() -> None:
    async def scenario() -> None:
        agent_id = AgentId.from_values("assistant", "main")
        first_model = SequenceModel([make_assistant_response("first reply")])
        first_agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=first_model),
            agent_id=agent_id,
        )
        await first_agent.run("first question")
        snapshot = first_agent.snapshot()
        if snapshot is None:
            raise AssertionError("Expected a snapshot after a completed run.")

        second_model = SequenceModel([make_assistant_response("second reply")])
        second_agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=second_model),
            snapshot=snapshot,
        )
        result = await second_agent.run("second question")

        assert second_agent.agent_id == agent_id
        assert result.final_output == "second reply"
        assert second_model.calls == [
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second question"},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_snapshot_rejects_mismatched_agent_id() -> None:
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="does not match"):
        DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            agent_id=AgentId.from_values("assistant", "other"),
            snapshot=snapshot,
        )


def test_default_agent_snapshot_rejects_run_state() -> None:
    state = RunState(instructions=None, history=[], responses=[])
    snapshot = _snapshot(state)

    with pytest.raises(ValueError, match="either run_state or snapshot"):
        DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            run_state=state,
            snapshot=snapshot,
        )


def test_default_agent_snapshot_before_first_run_returns_none() -> None:
    agent = DefaultAgent(descriptor=AgentDescriptor(name="Assistant"))

    assert agent.snapshot() is None


class _BlockingSequenceModel(SequenceModel):
    def __init__(self) -> None:
        super().__init__([make_assistant_response("second reply")])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        self.started.set()
        await self.release.wait()
        return await super().get_response(
            messages,
            extra_call_args=extra_call_args,
            tools=tools,
            schema=schema,
            cancellation_token=cancellation_token,
            parent_span=parent_span,
            **kwargs,
        )


def test_default_agent_snapshot_during_run_returns_committed_baseline() -> None:
    async def scenario() -> None:
        first_model = SequenceModel([make_assistant_response("first reply")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=first_model),
            agent_id=AgentId.from_values("assistant", "main"),
        )
        await agent.run("first question")
        before = agent.snapshot()
        if before is None:
            raise AssertionError("Expected a committed baseline.")

        blocking_model = _BlockingSequenceModel()
        agent.resolved_descriptor.model = blocking_model
        running = asyncio.create_task(agent.run("second question"))
        await blocking_model.started.wait()

        during = agent.snapshot()

        assert during is not None
        assert during.to_run_state() == before.to_run_state()
        blocking_model.release.set()
        await running

    asyncio.run(scenario())
