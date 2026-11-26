"""Optional spawn diagnostics must not prevent a child from running."""

from unittest.mock import Mock, patch

import pytest

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.models import ModelTracing
from agentlane.runtime import SingleThreadedRuntimeEngine
from agentlane.tracing import custom_span

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call


class _TracedSequenceModel(SequenceModel):
    @property
    def tracing(self) -> ModelTracing:
        return ModelTracing.ENABLED_WITHOUT_DATA


@pytest.mark.asyncio
@pytest.mark.parametrize("tracing_enabled", [False, True])
async def test_spawn_metadata_failure_does_not_change_baseline_assignment(
    tracing_enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = SequenceModel([make_assistant_response("finished")])
    metadata = Mock(side_effect=RuntimeError("private diagnostic error"))
    monkeypatch.setattr(child, "request_metadata", metadata)
    parent_type = _TracedSequenceModel if tracing_enabled else SequenceModel
    parent = parent_type(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(tool_id="spawn_1", name="child", arguments="{}")
                ],
            )
        ]
    )
    runner = Runner()
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        descriptor=AgentDescriptor(
            name="Parent",
            model=parent,
            handoffs=(AgentDescriptor(name="Child", model=child),),
        ),
    )

    with patch("agentlane.harness._run_loop.custom_span", wraps=custom_span) as span:
        result = await runner.run(
            agent, RunState(instructions=None, history=["Help me"], responses=[])
        )

    assert result.final_output == "finished"
    assert len(child.calls) == 1
    if tracing_enabled:
        metadata.assert_called_once_with(None)
    else:
        metadata.assert_not_called()
    span.assert_called_once()
    decision = span.call_args.kwargs["data"]
    assert decision["status"] == "baseline"
    assert "model" not in decision
    assert "private diagnostic error" not in str(decision)
