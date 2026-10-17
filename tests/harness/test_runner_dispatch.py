"""Runner stream settlement without model or tool execution."""

from unittest.mock import create_autospec, patch

import pytest

from agentlane.harness import Agent, Runner, RunResult, RunState, Task


@pytest.mark.asyncio
@pytest.mark.parametrize("event_stream", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_runner_settles_stream_from_loop_outcome(
    event_stream: bool, fails: bool
) -> None:
    state = RunState(instructions=None, history=[], responses=[])
    result = RunResult(final_output="done", responses=[], turn_count=0, run_state=state)
    failure = RuntimeError("loop failed")
    agent = create_autospec(Agent, instance=True)

    # Isolate stream settlement from model calls and turn execution.
    with patch("agentlane.harness._runner.RunLoop", autospec=True) as loop_type:
        loop_type.return_value.run.return_value = result
        if fails:
            loop_type.return_value.run.side_effect = failure

        runner = Runner()
        stream = (
            runner.run_events(agent, state)
            if event_stream
            else runner.run_stream(agent, state)
        )
        try:
            if fails:
                with pytest.raises(RuntimeError) as caught:
                    await stream.result()

                assert caught.value is failure
            else:
                assert await stream.result() is result
        finally:
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("has_task_contract", [False, True])
async def test_runner_rejects_invalid_dependencies_before_starting_loop(
    mode: str, has_task_contract: bool
) -> None:
    agent = create_autospec(Agent if has_task_contract else Task, instance=True)
    if has_task_contract:
        agent.model = None

    expected = "configured 'model'" if has_task_contract else "RunnerTask interface"
    state = RunState(instructions=None, history=["hello"], responses=[])

    # Check the same dependency boundary through all three runner entry points.
    with patch("agentlane.harness._runner.RunLoop", autospec=True):
        runner = Runner()
        if mode == "run":
            with pytest.raises(RuntimeError, match=expected):
                await runner.run(agent, state)
        else:
            stream = (
                runner.run_stream(agent, state)
                if mode == "stream"
                else runner.run_events(agent, state)
            )
            try:
                with pytest.raises(RuntimeError, match=expected):
                    await stream.result()
            finally:
                await stream.aclose()

    # Invalid dependencies must fail before the run state changes.
    assert state.turn_count == 0
    assert state.history == ["hello"]
    assert state.responses == []
