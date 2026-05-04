import asyncio

import pytest
from pydantic import ValidationError
from tools_test_utils import (
    SequenceModel,
    echo_tool,
    make_assistant_response,
    make_tool_call,
)

from agentlane.harness import AgentDescriptor, Runner
from agentlane.harness._handoff import (
    default_agent_tool_instructions,
    delegated_result_text,
)
from agentlane.harness._lifecycle import DefaultAgentTool
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.tools import HarnessToolsShim, agent_tool, base_harness_tools
from agentlane.harness.tools._shim import render_harness_tools_prompt
from agentlane.messaging import DeliveryOutcome, DeliveryStatus, MessageId
from agentlane.models import MessageDict, ModelResponse, Tools
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def _message(role: str, content: object) -> MessageDict:
    return {
        "role": role,
        "content": content,
    }


def _expected_default_child_system_prompt() -> str:
    prompt = render_harness_tools_prompt(definitions=base_harness_tools())
    if prompt is None:
        raise AssertionError("Base harness tools unexpectedly rendered no prompt.")
    return f"{default_agent_tool_instructions()}\n\n{prompt}"


def test_agent_tool_exposes_name_and_required_task_schema() -> None:
    definition = agent_tool()
    args_model = definition.tool.args_type()

    args = args_model(name="Researcher", task="Find the refund policy.")

    assert definition.tool.name == "agent"
    assert args.name == "Researcher"
    assert args.task == "Find the refund policy."
    assert "description" not in args_model.model_fields

    with pytest.raises(ValidationError):
        args_model(name="two words", task="Find the refund policy.")
    with pytest.raises(ValidationError):
        args_model(name="Researcher")


def test_default_agent_tool_custom_instructions_do_not_inject_name() -> None:
    tool = DefaultAgentTool(instructions="Use the delegated task exactly.")

    assert tool.resolved_instructions() == "Use the delegated task exactly."


def test_runner_rejects_invalid_agent_limits() -> None:
    with pytest.raises(ValueError, match="agent_max_depth"):
        Runner(agent_max_depth=0)
    with pytest.raises(ValueError, match="agent_max_threads"):
        Runner(agent_max_threads=0)


def test_agent_tool_executes_through_harness_tools_shim_with_inherited_parent_tools() -> (
    None
):
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        child_model = SequenceModel(
            [make_assistant_response(content="researched answer")]
        )
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="agent",
                            arguments=(
                                '{"name":"Researcher",'
                                '"task":"Research the refund exception."}'
                            ),
                        )
                    ],
                ),
                make_assistant_response(content="Here is the researched answer."),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                instructions="Parent system prompt must stay with the parent.",
                tools=Tools(tools=[echo_tool("custom")]),
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        result = await agent.run("Need a researched answer")

        assert result.final_output == "Here is the researched answer."
        assert child_model.calls == [
            [
                _message(
                    "system",
                    _expected_default_child_system_prompt(),
                ),
                _message("user", "Research the refund exception."),
            ]
        ]
        child_tools = child_model.call_tools[0]
        assert child_tools is not None
        child_tool_names = [tool.name for tool in child_tools.normalized_tools]
        assert "agent" in child_tool_names
        assert "read" in child_tool_names
        assert "custom" in child_tool_names

    asyncio.run(scenario())


def test_agent_tool_can_override_child_tools_to_none() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        child_model = SequenceModel([make_assistant_response(content="child answer")])
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="agent",
                            arguments='{"name":"Blank","task":"Work without tools."}',
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                tools=Tools(
                    tools=[
                        DefaultAgentTool(model=child_model, tools=None),
                        echo_tool("custom"),
                    ]
                ),
            ),
        )

        result = await agent.run("Delegate without child tools")

        assert result.final_output == "done"
        assert child_model.call_tools == [None]

    asyncio.run(scenario())


def test_agent_tool_supports_parallel_spawned_agents() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        release = asyncio.Event()
        child_model = _BlockingChildModel(expected_calls=2, release=release)
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="agent",
                            arguments='{"name":"Alpha","task":"Handle alpha."}',
                        ),
                        make_tool_call(
                            tool_id="call_2",
                            name="agent",
                            arguments='{"name":"Beta","task":"Handle beta."}',
                        ),
                    ],
                ),
                make_assistant_response(content="Both helpers finished."),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                tools=Tools(tools=[], parallel_tool_calls=True),
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        run_task = asyncio.create_task(agent.run("Run both helpers"))
        await asyncio.wait_for(child_model.all_started.wait(), timeout=1.0)
        release.set()

        result = await run_task

        assert result.final_output == "Both helpers finished."
        assert set(child_model.tasks) == {"Handle alpha.", "Handle beta."}

    asyncio.run(scenario())


def test_agent_tool_rejects_at_depth_limit() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner(agent_max_depth=1)
        child_model = SequenceModel([])
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="agent",
                            arguments='{"name":"Researcher","task":"Research."}',
                        )
                    ],
                ),
                make_assistant_response(content="Solved locally."),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        result = await agent.run("Research")

        assert result.final_output == "Solved locally."
        assert child_model.calls == []
        assert parent_model.calls[1][-1]["content"] == (
            "Agent depth limit reached. Solve the task yourself."
        )

    asyncio.run(scenario())


def test_agent_tool_tracks_recursive_depth_in_runner() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner(agent_max_depth=2)
        child_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_child",
                            name="agent",
                            arguments='{"name":"Nested","task":"Handle nested."}',
                        )
                    ],
                ),
                make_assistant_response(content="Child handled locally."),
            ]
        )
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_parent",
                            name="agent",
                            arguments='{"name":"Worker","task":"Handle child."}',
                        )
                    ],
                ),
                make_assistant_response(content="Parent done."),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        result = await agent.run("Start")

        assert result.final_output == "Parent done."
        assert len(child_model.calls) == 2
        nested_tool_results = [
            message["content"]
            for message in child_model.calls[1]
            if message.get("role") == "tool"
        ]
        assert nested_tool_results == [
            "Agent depth limit reached. Solve the task yourself."
        ]

    asyncio.run(scenario())


def test_agent_tool_rejects_at_thread_limit() -> None:
    async def scenario() -> None:
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner(agent_max_threads=1)
        release = asyncio.Event()
        child_model = _BlockingChildModel(expected_calls=1, release=release)
        parent_model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="agent",
                            arguments='{"name":"Alpha","task":"Handle alpha."}',
                        ),
                        make_tool_call(
                            tool_id="call_2",
                            name="agent",
                            arguments='{"name":"Beta","task":"Handle beta."}',
                        ),
                    ],
                ),
                make_assistant_response(content="Handled with limit."),
            ]
        )
        agent = DefaultAgent(
            runtime=runtime,
            runner=runner,
            descriptor=AgentDescriptor(
                name="Manager",
                model=parent_model,
                tools=Tools(tools=[], parallel_tool_calls=True),
                shims=(HarnessToolsShim((agent_tool(model=child_model),)),),
            ),
        )

        run_task = asyncio.create_task(agent.run("Run helpers"))
        await asyncio.wait_for(child_model.all_started.wait(), timeout=1.0)
        release.set()

        result = await run_task

        assert result.final_output == "Handled with limit."
        tool_result_contents = [
            message["content"]
            for message in parent_model.calls[1]
            if message.get("role") == "tool"
        ]
        assert "Agent thread limit reached. Solve the task yourself." in (
            tool_result_contents
        )
        assert child_model.tasks == ["Handle alpha."]

    asyncio.run(scenario())


def test_delegated_failure_text_is_sanitized() -> None:
    outcome = DeliveryOutcome.failed(
        status=DeliveryStatus.HANDLER_ERROR,
        message_id=MessageId("msg-1"),
        correlation_id=None,
        message="Traceback: secret internal path /tmp/private.py",
        retryable=False,
    )

    assert delegated_result_text(outcome) == "Error: delegated agent call failed."


class _BlockingChildModel(SequenceModel):
    def __init__(self, *, expected_calls: int, release: asyncio.Event) -> None:
        super().__init__([])
        self._expected_calls = expected_calls
        self._release = release
        self.all_started = asyncio.Event()
        self.tasks: list[str] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args
        del schema
        del cancellation_token
        del kwargs

        self.calls.append([dict(message) for message in messages])
        self.call_tools.append(tools)
        task = str(messages[-1]["content"])
        self.tasks.append(task)
        if len(self.tasks) >= self._expected_calls:
            self.all_started.set()
        await self._release.wait()
        return make_assistant_response(content=f"done: {task}")
