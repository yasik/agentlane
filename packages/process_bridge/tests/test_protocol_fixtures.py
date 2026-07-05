import json
from pathlib import Path

from agentlane_process_bridge import (
    BRIDGE_EVENT_TYPES,
    COMMAND_TYPES,
    RUN_EVENT_BRIDGE_HANDLERS,
    RUN_EVENT_KIND_BRIDGE_EVENT_TYPES,
    BridgeEventType,
    ProtocolError,
    UnknownCommand,
    parse_command_line,
)

from agentlane.harness import HarnessEventType, RunEventKind

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

    assert len(event_types) == len(set(event_types))
    _assert_same_strings(
        expected={event_type.value for event_type in BRIDGE_EVENT_TYPES},
        actual=set(event_types),
        expected_name="Python bridge event types",
        actual_name="protocol fixtures",
    )
    assert all(event["protocol_version"] == "1.0" for event in fixtures)
    assert all(isinstance(event["ts"], int | float) for event in fixtures)


def test_bridge_event_type_uses_upstream_harness_run_event_values() -> None:
    assert BridgeEventType.AGENT_START.value == RunEventKind.AGENT_START.value
    assert BridgeEventType.AGENT_END.value == RunEventKind.AGENT_END.value
    assert BridgeEventType.LLM_START.value == RunEventKind.LLM_START.value
    assert BridgeEventType.LLM_END.value == RunEventKind.LLM_END.value
    assert BridgeEventType.TOOL_START.value == RunEventKind.TOOL_START.value
    assert BridgeEventType.TOOL_END.value == RunEventKind.TOOL_END.value
    assert BridgeEventType.HANDOFF_START.value == RunEventKind.HANDOFF_START.value
    assert BridgeEventType.HANDOFF_END.value == RunEventKind.HANDOFF_END.value
    assert BridgeEventType.STATE_SNAPSHOT.value == RunEventKind.STATE_SNAPSHOT.value
    assert BridgeEventType.PLAN_UPDATED.value == RunEventKind.PLAN_UPDATED.value

    assert BridgeEventType.RUN_START.value == HarnessEventType.RUN_START.value
    assert BridgeEventType.RUN_COMPLETE.value == HarnessEventType.RUN_COMPLETE.value
    assert BridgeEventType.RUN_CANCELLED.value == HarnessEventType.RUN_CANCELLED.value
    assert BridgeEventType.ERROR.value == HarnessEventType.ERROR.value
    assert not hasattr(BridgeEventType, "MODEL_STREAM")
    assert not hasattr(BridgeEventType, "TOOL_APPROVAL")


def test_every_run_event_kind_has_bridge_coverage() -> None:
    handler_kinds = [handler.kind for handler in RUN_EVENT_BRIDGE_HANDLERS]

    assert len(handler_kinds) == len(set(handler_kinds))
    assert set(RUN_EVENT_KIND_BRIDGE_EVENT_TYPES) == set(RunEventKind)

    mapped_event_types: set[BridgeEventType] = set()
    for event_types in RUN_EVENT_KIND_BRIDGE_EVENT_TYPES.values():
        assert event_types
        mapped_event_types.update(event_types)

    assert mapped_event_types <= BRIDGE_EVENT_TYPES


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
