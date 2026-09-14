"""Verify that the bridge delivers complete native records in source order."""

import asyncio
from io import StringIO

from agentlane_process_bridge import BridgeBackend, EventWriter, PromptCommand

from agentlane.harness import RunResult

from .helpers import FakeAgent, wait_for_event_count, wait_for_stream
from .native_fixtures import native_events


def test_backend_forwards_every_native_record_without_projection() -> None:
    async def scenario() -> None:
        output = StringIO()
        agent = FakeAgent()
        backend = BridgeBackend(agent=agent, events=EventWriter(output))
        await backend.handle_command(PromptCommand(text="go"))
        stream = await wait_for_stream(agent)
        sources = native_events()
        for source in sources:
            stream.emit(source)
        stream.finish(
            RunResult(final_output={"done": True}, responses=[], turn_count=1)
        )
        events = await wait_for_event_count(output, len(sources) + 2)
        assert [event["type"] for event in events] == [
            "run_start",
            *["run_event" for _ in sources],
            "run_complete",
        ]
        assert [event["event"] for event in events[1:-1]] == [
            source.to_dict() for source in sources
        ]
        assert events[-1]["final_output"] == {"done": True}
        await backend.close()

    asyncio.run(scenario())
