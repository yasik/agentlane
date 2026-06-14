"""Tests for the live run-state read view exposed to tool handlers."""

from collections.abc import MutableMapping
from typing import cast

import pytest

from agentlane.harness import (
    ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX,
    LiveRunStateView,
    RunState,
)
from agentlane.messaging import AgentId
from agentlane.models import RunStateView


def _run_state() -> RunState:
    """Return one empty run state for view tests."""
    return RunState(instructions=None, history=[], responses=[])


def _task_id() -> AgentId:
    """Return one stable task identity for view tests."""
    return AgentId.from_values("agent", "task-1")


def test_live_run_state_view_satisfies_protocol() -> None:
    """LiveRunStateView should be a structural RunStateView."""
    view = LiveRunStateView(_run_state(), _task_id())

    assert isinstance(view, RunStateView)


def test_live_run_state_view_task_id_matches_stringified_identity() -> None:
    """The view task_id should equal str(task_id), the run_id convention."""
    task_id = _task_id()

    view = LiveRunStateView(_run_state(), task_id)

    assert view.task_id == str(task_id)


def test_live_run_state_view_shim_state_reads_through_to_live_state() -> None:
    """shim_state should reflect mutations made after the view is built."""
    state = _run_state()
    view = LiveRunStateView(state, _task_id())

    state.shim_state["skills:workspace"] = "/work/ws"

    assert view.shim_state.get("skills:workspace") == "/work/ws"


def test_live_run_state_view_shim_state_rejects_top_level_mutation() -> None:
    """shim_state should be live but not writable through the tool view."""
    state = _run_state()
    state.shim_state["skills:workspace"] = "/work/ws"
    view = LiveRunStateView(state, _task_id())

    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], view.shim_state)[
            "skills:workspace"
        ] = "/other"

    assert state.shim_state["skills:workspace"] == "/work/ws"


def test_live_run_state_view_active_skill_names_unions_suffix_keys() -> None:
    """active_skill_names should union every documented suffix key."""
    state = _run_state()
    state.shim_state["skills" + ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX] = ["alpha", "beta"]
    state.shim_state["workspace" + ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX] = [
        "beta",
        "gamma",
    ]
    state.shim_state["skills:workspace"] = "/work/ws"
    view = LiveRunStateView(state, _task_id())

    assert view.active_skill_names == ("alpha", "beta", "gamma")


def test_live_run_state_view_active_skill_names_empty_when_unset() -> None:
    """active_skill_names should be empty when no skills shim populated it."""
    view = LiveRunStateView(_run_state(), _task_id())

    assert view.active_skill_names == ()


def test_live_run_state_view_active_skill_names_ignores_non_list_and_non_string() -> (
    None
):
    """Malformed active-name state must not crash the accessor."""
    state = _run_state()
    state.shim_state["skills" + ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX] = "not-a-list"
    state.shim_state["workspace" + ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX] = [
        "valid",
        7,
        None,
    ]
    view = LiveRunStateView(state, _task_id())

    assert view.active_skill_names == ("valid",)
