import json
from pathlib import Path
from typing import get_args

from agentlane_process_bridge import (
    BRIDGE_EVENT_TYPES,
    COMMAND_TYPES,
    BridgeEventType,
    ProtocolError,
    UnknownCommand,
    parse_command_line,
)

from agentlane.harness import (
    HarnessEventType,
    RunEvent,
    RunEventKind,
    RunModelStreamEvent,
    RunToolApprovalEvent,
)
from agentlane.harness.tools import ToolApprovalStatus
from agentlane.models import ModelStreamEventKind

from .native_fixtures import native_events

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "protocol" / "events.json"


def test_command_parsers_cover_command_types() -> None:
    parsed_types: set[str] = set()
    for command_type in COMMAND_TYPES:
        command = parse_command_line(_command_fixture(command_type))
        assert not isinstance(command, ProtocolError)
        assert not isinstance(command, UnknownCommand)
        parsed_types.add(command.type)

    _assert_same_strings(
        expected=frozenset(COMMAND_TYPES),
        actual=parsed_types,
        expected_name="Python command types",
        actual_name="Python command parsers",
    )


def test_protocol_fixtures_are_versioned_and_cover_unique_event_types() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text())
    event_types = [event["type"] for event in fixtures]

    _assert_same_strings(
        expected={event_type.value for event_type in BRIDGE_EVENT_TYPES},
        actual=set(event_types),
        expected_name="Python bridge event types",
        actual_name="protocol fixtures",
    )
    assert all(event["protocol_version"] == "1.0" for event in fixtures)
    assert all(isinstance(event["ts"], int | float) for event in fixtures)


def test_bridge_event_type_uses_upstream_harness_run_event_values() -> None:
    assert BridgeEventType.RUN_START.value == HarnessEventType.RUN_START.value
    assert BridgeEventType.RUN_COMPLETE.value == HarnessEventType.RUN_COMPLETE.value
    assert BridgeEventType.RUN_CANCELLED.value == HarnessEventType.RUN_CANCELLED.value
    assert BridgeEventType.ERROR.value == HarnessEventType.ERROR.value
    assert not hasattr(BridgeEventType, "MODEL_STREAM")
    assert not hasattr(BridgeEventType, "TOOL_APPROVAL")


def test_every_native_kind_has_exact_generated_fixture() -> None:
    sources = native_events()
    fixtures = json.loads(FIXTURE_PATH.read_text())
    assert [event["event"] for event in fixtures if event["type"] == "run_event"] == [
        source.to_dict() for source in sources
    ]
    assert {type(source) for source in sources} == set(get_args(RunEvent.__value__))
    assert {source.kind for source in sources} == set(RunEventKind)
    assert {
        source.event.kind
        for source in sources
        if isinstance(source, RunModelStreamEvent)
    } == set(ModelStreamEventKind)
    assert {
        source.event.record.status
        for source in sources
        if isinstance(source, RunToolApprovalEvent)
    } == set(ToolApprovalStatus)


def _assert_same_strings(
    *,
    expected: set[str] | frozenset[str],
    actual: set[str] | frozenset[str],
    expected_name: str,
    actual_name: str,
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    assert not missing and not extra, (
        f"{actual_name} do not match {expected_name}. "
        f"Missing: {missing or 'none'}. Extra: {extra or 'none'}."
    )


def _command_fixture(command_type: str) -> str:
    match command_type:
        case "approve":
            return (
                '{"protocol_version":"1.0","type":"approve",'
                '"id":"request-1","allowed":true}\n'
            )
        case "cancel" | "reset" | "shutdown":
            return f'{{"protocol_version":"1.0","type":"{command_type}"}}\n'
        case "configure":
            return (
                '{"protocol_version":"1.0","type":"configure",'
                '"patch":{"model":"openai/gpt-5.5"}}\n'
            )
        case "prompt":
            return '{"protocol_version":"1.0","type":"prompt","text":"go"}\n'
        case _:
            raise AssertionError(f"Missing command fixture for {command_type}.")
