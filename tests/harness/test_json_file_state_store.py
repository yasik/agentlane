import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from agentlane.harness import (
    AgentDescriptor,
    AgentSnapshot,
    JsonFileStateStore,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    StateStore,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import AgentId
from agentlane.runtime import CancellationToken

from .tools_test_utils import SequenceModel, make_assistant_response

_AGENT_ID = AgentId.from_values("assistant", "main")


class _MemoryStateStore(StateStore):
    def __init__(self) -> None:
        self.snapshot: AgentSnapshot | None = None

    def load(self) -> AgentSnapshot | None:
        return self.snapshot

    def save(
        self,
        snapshot: AgentSnapshot,
        *,
        expected_revision: int | None,
    ) -> None:
        actual_revision = self.snapshot.revision if self.snapshot is not None else None
        if actual_revision != expected_revision:
            raise ValueError("State revision mismatch.")
        if expected_revision is not None and snapshot.revision <= expected_revision:
            raise ValueError("New state revision must increase.")
        self.snapshot = snapshot

    def delete(self, *, expected_revision: int | None) -> None:
        actual_revision = self.snapshot.revision if self.snapshot is not None else None
        if actual_revision != expected_revision:
            raise ValueError("State revision mismatch.")
        self.snapshot = None


class _ReplacingRunner(Runner):
    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        del agent, hooks, cancellation_token
        replacement = RunState(
            instructions=state.instructions,
            history=list(state.history),
            responses=list(state.responses),
            turn_count=state.turn_count + 1,
        )
        return RunResult(
            final_output="reply",
            responses=[],
            turn_count=replacement.turn_count,
            run_state=replacement,
        )


class _BlockingRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        del agent, hooks, cancellation_token
        self.started.set()
        await self.release.wait()
        state.turn_count += 1
        return RunResult(
            final_output="reply",
            responses=[],
            turn_count=state.turn_count,
            run_state=state,
        )


def _snapshot(*, revision: int, content: str) -> AgentSnapshot:
    return AgentSnapshot.capture(
        agent_id=_AGENT_ID,
        run_state=RunState(
            instructions="Keep the conversation continuous.",
            history=[{"role": "user", "content": content}],
            responses=[],
            revision=revision,
        ),
    )


def test_json_file_state_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "agent.json")

    assert isinstance(store, StateStore)
    assert store.load() is None


def test_json_file_state_store_load_propagates_file_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "agent.json"
    state_path.write_text("{}", encoding="utf-8")
    store = JsonFileStateStore(state_path)

    def fail_read(self: Path, *, encoding: str) -> str:
        del self, encoding
        raise FileNotFoundError("state file disappeared")

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(FileNotFoundError, match="state file disappeared"):
        store.load()


def test_json_file_state_store_save_then_load_round_trips_snapshot(
    tmp_path: Path,
) -> None:
    store = JsonFileStateStore(tmp_path / "nested" / "agent.json")
    snapshot = _snapshot(revision=0, content="first")

    store.save(snapshot, expected_revision=None)

    assert store.load() == snapshot


def test_json_file_state_store_syncs_directory_after_entry_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path = tmp_path / "agent.json"
    store = JsonFileStateStore(state_path)
    observations: list[tuple[Path, bool]] = []

    def record_sync(path: Path) -> None:
        observations.append((path, state_path.exists()))

    monkeypatch.setattr(
        "agentlane.harness._json_file_state_store._sync_directory",
        record_sync,
    )

    store.save(_snapshot(revision=0, content="first"), expected_revision=None)
    store.delete(expected_revision=0)

    assert observations == [(tmp_path, True), (tmp_path, False)]


def test_json_file_state_store_rejects_stale_revision(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "agent.json")
    store.save(_snapshot(revision=0, content="first"), expected_revision=None)

    with pytest.raises(ValueError, match=r"greater than 0, got 0"):
        store.save(_snapshot(revision=0, content="same"), expected_revision=0)

    store.save(_snapshot(revision=1, content="second"), expected_revision=0)

    with pytest.raises(ValueError, match=r"expected 0, found 1"):
        store.save(_snapshot(revision=1, content="stale"), expected_revision=0)

    current = store.load()
    assert current is not None
    assert current.revision == 1
    assert current.to_run_state().history == [{"role": "user", "content": "second"}]


def test_json_file_state_store_interrupted_save_preserves_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path / "agent.json")
    original = _snapshot(revision=0, content="first")
    store.save(original, expected_revision=None)

    def fail_replace(source: Path, destination: Path) -> None:
        del source
        del destination
        raise OSError("simulated interruption")

    monkeypatch.setattr(
        "agentlane.harness._json_file_state_store.os.replace",
        fail_replace,
    )

    with pytest.raises(OSError, match="simulated interruption"):
        store.save(_snapshot(revision=1, content="second"), expected_revision=0)

    assert store.load() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_default_agent_state_path_restores_continues_and_saves(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        first = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("first reply")]),
                instructions="Keep the conversation continuous.",
            ),
            agent_id=_AGENT_ID,
            state_path=state_path,
        )
        first_result = await first.run("first question")

        second_model = SequenceModel([make_assistant_response("second reply")])
        second = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=second_model),
            state_path=state_path,
        )
        second_result = await second.run("second question")

        stored = JsonFileStateStore(state_path).load()
        assert stored is not None
        assert first_result.run_state is not None
        assert first_result.run_state.revision == 1
        assert second_result.run_state is not None
        assert second_result.run_state.revision == 2
        assert second.agent_id == _AGENT_ID
        assert second.run_state is not None
        assert second.run_state.instructions == "Keep the conversation continuous."
        assert stored.revision == 2
        assert second_model.calls == [
            [
                {
                    "role": "system",
                    "content": "Keep the conversation continuous.",
                },
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second question"},
            ]
        ]

    asyncio.run(scenario())


def test_default_agent_accepts_custom_state_store() -> None:
    async def scenario() -> None:
        store = _MemoryStateStore()
        first = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("first reply")]),
                instructions="Keep the conversation continuous.",
            ),
            agent_id=_AGENT_ID,
            state_store=store,
        )
        await first.run("first question")

        second_model = SequenceModel([make_assistant_response("second reply")])
        second = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=second_model),
            state_store=store,
        )
        await second.run("second question")

        assert store.snapshot is not None
        assert store.snapshot.revision == 2
        assert second.agent_id == _AGENT_ID
        assert second_model.calls[0][-3:] == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second question"},
        ]

    asyncio.run(scenario())


def test_default_agent_replacement_runner_keeps_revision_sequence(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            runner=_ReplacingRunner(),
            state_path=state_path,
        )

        first = await agent.run("first")
        second = await agent.run("second")

        stored = JsonFileStateStore(state_path).load()
        assert first.run_state is not None
        assert first.run_state.revision == 1
        assert second.run_state is not None
        assert second.run_state.revision == 2
        assert stored is not None
        assert stored.revision == 2

    asyncio.run(scenario())


def test_default_agent_explicit_state_uses_persisted_revision(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        model = SequenceModel(
            [
                make_assistant_response("first reply"),
                make_assistant_response("replacement reply"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant", model=model),
            state_path=state_path,
        )
        await agent.run("first question")
        replacement = RunState(
            instructions="Use the replacement conversation.",
            history=[{"role": "user", "content": "replacement question"}],
            responses=[],
        )

        result = await agent.run(replacement)

        stored = JsonFileStateStore(state_path).load()
        assert replacement.revision == 0
        assert result.run_state is not None
        assert result.run_state.revision == 2
        assert stored is not None
        assert stored.revision == 2
        assert model.calls[1] == [
            {"role": "system", "content": "Use the replacement conversation."},
            {"role": "user", "content": "replacement question"},
        ]

    asyncio.run(scenario())


def test_default_agent_state_path_reset_removes_persisted_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("reply")]),
            ),
            state_path=state_path,
        )
        await agent.run("question")

        agent.reset()

        assert agent.run_state is None
        assert not state_path.exists()

    asyncio.run(scenario())


def test_default_agent_reset_rejects_active_primary_run(tmp_path: Path) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        runner = _BlockingRunner()
        agent = DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            runner=runner,
            state_path=state_path,
        )
        run_task = asyncio.create_task(agent.run("question"))
        await runner.started.wait()

        try:
            with pytest.raises(RuntimeError, match="primary run is active"):
                agent.reset()
        finally:
            runner.release.set()
            await run_task

        assert state_path.exists()

    asyncio.run(scenario())


def test_default_agent_state_path_save_failure_keeps_previous_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel(
                    [
                        make_assistant_response("first reply"),
                        make_assistant_response("second reply"),
                    ]
                ),
            ),
            state_path=state_path,
        )
        await agent.run("first question")
        baseline = agent.run_state

        def fail_save(
            self: JsonFileStateStore,
            snapshot: AgentSnapshot,
            *,
            expected_revision: int | None,
        ) -> None:
            del self
            del snapshot
            del expected_revision
            raise OSError("simulated save failure")

        monkeypatch.setattr(JsonFileStateStore, "save", fail_save)

        with pytest.raises(OSError, match="simulated save failure"):
            await agent.run("second question")

        assert agent.run_state == baseline
        stored = JsonFileStateStore(state_path).load()
        assert stored is not None
        assert stored.revision == 1

    asyncio.run(scenario())


def test_default_agent_state_path_rejects_stale_process_state(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state_path = tmp_path / "assistant.json"
        seed = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("seed reply")]),
            ),
            state_path=state_path,
        )
        await seed.run("seed question")

        first = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("first reply")]),
            ),
            state_path=state_path,
        )
        stale = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Assistant",
                model=SequenceModel([make_assistant_response("stale reply")]),
            ),
            state_path=state_path,
        )
        await first.run("first continuation")

        with pytest.raises(ValueError, match=r"expected 1, found 2"):
            await stale.run("stale continuation")

        assert stale.run_state is not None
        assert stale.run_state.revision == 1
        stored = JsonFileStateStore(state_path).load()
        assert stored is not None
        assert stored.revision == 2

    asyncio.run(scenario())


def test_default_agent_state_path_rejects_explicit_state_sources(
    tmp_path: Path,
) -> None:
    state = RunState(instructions=None, history=[], responses=[])

    with pytest.raises(
        ValueError,
        match="state_path or state_store without run_state or snapshot",
    ):
        DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            run_state=state,
            state_path=tmp_path / "assistant.json",
        )


def test_default_agent_rejects_state_path_with_state_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either state_path or state_store"):
        DefaultAgent(
            descriptor=AgentDescriptor(name="Assistant"),
            state_path=tmp_path / "assistant.json",
            state_store=_MemoryStateStore(),
        )
