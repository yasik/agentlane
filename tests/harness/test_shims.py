import asyncio
from collections.abc import Mapping

from pydantic import BaseModel

from agentlane.harness import Agent, AgentDescriptor, Runner, ShimState
from agentlane.harness._tooling import merge_tools
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.shims import (
    BoundShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
)
from agentlane.messaging import AgentId
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    Tool,
    ToolCall,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def make_assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    """Build one canonical assistant response for shim tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_shims",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def _make_tool_call(
    *,
    tool_id: str,
    arguments: str,
    name: str,
) -> ToolCall:
    """Build one canonical tool call payload."""
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


class _SequenceModel(Model[ModelResponse]):
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[list[MessageDict]] = []
        self.call_options: list[dict[str, object]] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: object | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del cancellation_token
        self.calls.append([dict(message) for message in messages])
        self.call_options.append(
            {
                "extra_call_args": extra_call_args,
                "schema": schema,
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        if not self._outcomes:
            raise AssertionError("Expected one queued model response.")
        return self._outcomes.pop(0)


def _append_instruction(
    current: object,
    extra: str,
) -> str:
    """Append one line to string instructions."""
    if current is None:
        return extra
    if not isinstance(current, str):
        raise AssertionError("Test shim expects string instructions.")
    return f"{current}\n{extra}"


def _shim_counter_value(shim_state: Mapping[str, object], key: str) -> int:
    """Read one integer counter from shim state for tests."""
    value = shim_state.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    raise AssertionError(f"Expected integer shim state for `{key}`.")


class _AppendInstructionShim(Shim):
    def __init__(self, name: str, line: str) -> None:
        self._name = name
        self._line = line

    @property
    def name(self) -> str:
        return self._name

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.instructions = _append_instruction(turn.instructions, self._line)


class _RewriteLastUserMessageShim(Shim):
    @property
    def name(self) -> str:
        return "rewrite-last-user"

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        del turn
        rewritten = [dict(message) for message in messages]
        for message in reversed(rewritten):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                raise AssertionError("Expected string user content in the test.")
            message["content"] = f"{content} [rewritten]"
            return rewritten
        return rewritten


class _EchoArgs(BaseModel):
    text: str


async def _echo_from_shim(
    args: _EchoArgs, cancellation_token: CancellationToken
) -> str:
    del cancellation_token
    return f"echo:{args.text}"


class _AddEchoToolShim(Shim):
    def __init__(self) -> None:
        self._tool = Tool(
            name="echo_from_shim",
            description="Echo the provided text.",
            args_model=_EchoArgs,
            handler=_echo_from_shim,
        )

    @property
    def name(self) -> str:
        return "add-echo-tool"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.tools = merge_tools(turn.tools, (self._tool,))


class _PersistCounterShim(Shim):
    @property
    def name(self) -> str:
        return "persist-counter"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        count = _shim_counter_value(turn.run_state.shim_state, "persist-counter")
        turn.instructions = _append_instruction(
            turn.instructions,
            f"persist-counter={count}",
        )

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        del response
        count = _shim_counter_value(turn.run_state.shim_state, "persist-counter")
        turn.run_state.shim_state["persist-counter"] = count + 1


class _PrivateCounterBoundShim(BoundShim):
    def __init__(self) -> None:
        self._count = 0

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self._count += 1
        turn.instructions = _append_instruction(
            turn.instructions,
            f"private-count={self._count}",
        )


class _PrivateCounterShim(Shim):
    @property
    def name(self) -> str:
        return "private-counter"

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        _ = context
        return _PrivateCounterBoundShim()


class _FailAfterMutationShim(Shim):
    @property
    def name(self) -> str:
        return "fail-after-mutation"

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if turn.run_state.turn_count >= 1:
            turn.run_state.shim_state["marker"] = "before-failure"
            raise RuntimeError("shim failure")


def test_default_agent_applies_shims_in_descriptor_order() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("done")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Ordered",
                model=model,
                instructions="Base",
                shims=(
                    _AppendInstructionShim("first", "First line"),
                    _AppendInstructionShim("second", "Second line"),
                ),
            )
        )

        result = await agent.run("hello")

        assert result.final_output == "done"
        assert model.calls[0][0] == {
            "role": "system",
            "content": "Base\nFirst line\nSecond line",
        }

    asyncio.run(scenario())


def test_default_agent_shim_can_transform_messages() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("done")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Rewrite",
                model=model,
                shims=(_RewriteLastUserMessageShim(),),
            )
        )

        await agent.run("hello")

        assert model.calls[0] == [
            {"role": "user", "content": "hello [rewritten]"},
        ]

    asyncio.run(scenario())


def test_default_agent_shim_can_add_tools() -> None:
    async def scenario() -> None:
        model = _SequenceModel(
            [
                make_assistant_response(
                    None,
                    tool_calls=[
                        _make_tool_call(
                            tool_id="call_1",
                            name="echo_from_shim",
                            arguments='{"text":"hello"}',
                        )
                    ],
                ),
                make_assistant_response("tool complete"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="ToolShim",
                model=model,
                shims=(_AddEchoToolShim(),),
            )
        )

        result = await agent.run("run the shim tool")

        assert result.final_output == "tool complete"
        first_tools = model.call_options[0]["tools"]
        if not isinstance(first_tools, Tools):
            raise AssertionError("Expected shim-added tools to be visible.")
        tool_names = [tool.name for tool in first_tools.normalized_tools]
        assert tool_names == ["echo_from_shim"]
        assert model.calls[1][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "echo_from_shim",
            "content": "echo:hello",
        }

    asyncio.run(scenario())


def test_default_agent_shim_state_persists_across_runs() -> None:
    async def scenario() -> None:
        model = _SequenceModel(
            [
                make_assistant_response("first"),
                make_assistant_response("second"),
            ]
        )
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Persist",
                model=model,
                instructions="Base",
                shims=(_PersistCounterShim(),),
            )
        )

        await agent.run("one")
        await agent.run("two")

        assert model.calls[0][0]["content"] == "Base\npersist-counter=0"
        assert model.calls[1][0]["content"] == "Base\npersist-counter=1"
        if agent.run_state is None:
            raise AssertionError("Expected persisted run state.")
        assert isinstance(agent.run_state.shim_state, ShimState)
        assert agent.run_state.shim_state == {"persist-counter": 2}

    asyncio.run(scenario())


def test_default_agent_bound_shims_are_isolated_per_agent_instance() -> None:
    async def scenario() -> None:
        shared_shim = _PrivateCounterShim()
        runtime = SingleThreadedRuntimeEngine()
        model_a = _SequenceModel(
            [
                make_assistant_response("a1"),
                make_assistant_response("a2"),
            ]
        )
        model_b = _SequenceModel([make_assistant_response("b1")])
        agent_a = Agent.bind(
            runtime,
            AgentId.from_values("shim-agent", "a"),
            runner=Runner(),
            descriptor=AgentDescriptor(
                name="AgentA",
                model=model_a,
                instructions="Base",
                shims=(shared_shim,),
            ),
        )
        agent_b = Agent.bind(
            runtime,
            AgentId.from_values("shim-agent", "b"),
            runner=Runner(),
            descriptor=AgentDescriptor(
                name="AgentB",
                model=model_b,
                instructions="Base",
                shims=(shared_shim,),
            ),
        )

        await agent_a.send_message("first", recipient=agent_a.id)
        await agent_b.send_message("first", recipient=agent_b.id)
        await agent_a.send_message("second", recipient=agent_a.id)

        assert model_a.calls[0][0]["content"] == "Base\nprivate-count=1"
        assert model_b.calls[0][0]["content"] == "Base\nprivate-count=1"
        assert model_a.calls[1][0]["content"] == "Base\nprivate-count=2"

    asyncio.run(scenario())


def test_default_agent_failed_shim_work_does_not_commit_state() -> None:
    async def scenario() -> None:
        model = _SequenceModel([make_assistant_response("unused")])
        agent = DefaultAgent(
            descriptor=AgentDescriptor(
                name="Failing",
                model=model,
                shims=(_FailAfterMutationShim(),),
            )
        )

        with_raised = False
        try:
            await agent.run("fail")
        except RuntimeError as exc:
            with_raised = True
            assert str(exc) == (
                "DefaultAgent run failed with delivery status "
                "`handler_error`: shim failure"
            )

        assert with_raised
        assert agent.run_state is None

    asyncio.run(scenario())
