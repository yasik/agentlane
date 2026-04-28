import asyncio
from typing import cast

from tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_state,
    run_tool,
)

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import HarnessToolsShim, plan_tool
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken


def test_plan_tool_updates_plan_with_codex_success_message() -> None:
    output = run_tool(
        plan_tool(),
        explanation="Track the implementation.",
        plan=[
            {"step": "Inspect implementation", "status": "completed"},
            {"step": "Add tests", "status": "in_progress"},
            {"step": "Update docs", "status": "pending"},
        ],
    )

    assert output == "Plan updated"


def test_plan_tool_accepts_pending_and_completed_plans() -> None:
    assert (
        run_tool(
            plan_tool(),
            plan=[{"step": "Start", "status": "pending"}],
        )
        == "Plan updated"
    )
    assert (
        run_tool(
            plan_tool(),
            plan=[{"step": "Finish", "status": "completed"}],
        )
        == "Plan updated"
    )


def test_plan_tool_rejects_empty_or_invalid_plans() -> None:
    definition = plan_tool()

    assert run_tool(definition, plan=[]) == "plan must contain at least one item"
    assert (
        run_tool(definition, plan=[{"step": "   ", "status": "pending"}])
        == "plan steps must not be empty"
    )
    assert (
        run_tool(
            definition,
            plan=[
                {"step": "Start", "status": "in_progress"},
                {"step": "Continue", "status": "in_progress"},
            ],
        )
        == "at most one plan step can be in_progress"
    )


def test_plan_tool_sanitizes_unexpected_error_text() -> None:
    def raise_unexpected_error(snapshot: dict[str, object]) -> None:
        del snapshot
        raise RuntimeError("Traceback (most recent call last): private details")

    output = run_tool(
        plan_tool(persist_to=raise_unexpected_error),
        plan=[{"step": "Start", "status": "pending"}],
    )

    assert output == "failed to update plan"


def test_plan_tool_persists_latest_plan_through_harness_tools_shim() -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((plan_tool(),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        await bound.on_run_start(state, DefaultRunContext())

        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)
        assert turn.tools is not None
        bound_plan = turn.tools.executable_tools[0]
        args_model = bound_plan.args_type()

        first = args_model(
            plan=[{"step": "Start", "status": "in_progress"}],
        )
        second = args_model(
            plan=[{"step": "Finish", "status": "completed"}],
        )

        await bound_plan.run(first, CancellationToken())
        assert state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Start", "status": "in_progress"}],
        }

        await bound_plan.run(second, CancellationToken())
        assert state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Finish", "status": "completed"}],
        }

    asyncio.run(scenario())


def test_plan_tool_runs_through_normal_runner_execution() -> None:
    async def scenario() -> None:
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="write_plan",
                            arguments=(
                                '{"plan":['
                                '{"step":"Create plan","status":"completed"}]}'
                            ),
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Planner",
                model=model,
                instructions="You create plans.",
                shims=(HarnessToolsShim((plan_tool(),)),),
            )
        )

        result = await agent.run("Plan this task.")

        assert result.final_output == "done"
        run_state = result.run_state
        if run_state is None:
            raise AssertionError("Expected DefaultAgent to return run state.")
        assert run_state.shim_state["harness-tools:plan"] == {
            "explanation": None,
            "plan": [{"step": "Create plan", "status": "completed"}],
        }
        assert any(
            isinstance(message, dict)
            and message["role"] == "tool"
            and message["name"] == "write_plan"
            and message["content"] == "Plan updated"
            for message in run_state.history
        )

    asyncio.run(scenario())


def test_plan_tool_prompt_metadata_renders_through_shim() -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((plan_tool(),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert isinstance(state.instructions, str)
        assert "- write_plan: Write or update the task plan" in state.instructions
        assert (
            "Use `write_plan` to maintain a visible, step-by-step plan"
            in state.instructions
        )
        assert state.instructions.endswith("</default_tools>")

    asyncio.run(scenario())
