"""Spawn policy contracts through public runner calls and captured requests."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, Literal, Self

import pytest
from pydantic import BaseModel, Field

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    DefaultAgentTool,
    DefaultHandoff,
    DelegationKind,
    Runner,
    RunnerHooks,
    RunResult,
    RunState,
    SpawnModelConfig,
    SpawnModelContext,
    SpawnModelRejected,
    SpawnModelSelection,
    Task,
)
from agentlane.harness._handoff import default_handoff_task_message
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.messaging import AgentId, DeliveryStatus
from agentlane.models import (
    MessageDict,
    Model,
    ModelBehaviorError,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ModelTracing,
    OutputSchema,
    ToolCall,
    ToolFailure,
    Tools,
)
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine
from agentlane.runtime._types import DeliveryTask
from agentlane.tracing import (
    CustomSpanData,
    DefaultTraceProvider,
    Span,
    TracingProcessor,
    get_trace_provider,
    set_trace_provider,
    trace,
)

from .tools_test_utils import make_assistant_response, make_tool_call


class _Choice(BaseModel):
    work_class: Literal["quick", "reasoning"] | None = Field(
        default=None, description="Suggested work class; application policy decides."
    )


class _TaskInput(BaseModel):
    text: str


class _TaskWithRoutingField(BaseModel):
    spawn_parameters: str


class _CaptureModel(Model[ModelResponse]):
    """Require requests to run inside this exact client's resource scope."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[MessageDict]] = []
        self.args: list[dict[str, Any] | None] = []
        self.catalogs: list[Tools | None] = []
        self.opens = 0
        self.closes = 0
        self.active = False
        self.before_request: Callable[[], None] | None = None

    @asynccontextmanager
    async def open_run(self) -> AsyncGenerator[Self, None]:
        self.opens += 1
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self.closes += 1

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        assert self.active, "The selected client must own the child's run resources."
        if self.before_request is not None:
            self.before_request()
        self.calls.append(deepcopy(messages))
        self.args.append(deepcopy(extra_call_args))
        self.catalogs.append(tools)
        return self.responses.pop(0)


class _Provider:
    def __init__(
        self,
        selection: SpawnModelSelection | None,
        *,
        parameters_model: type[BaseModel] | None = _Choice,
    ) -> None:
        self.parameters_model = parameters_model
        self.selection = selection
        self.contexts: list[SpawnModelContext] = []
        self.error: Exception | None = None

    def resolve(self, context: SpawnModelContext) -> SpawnModelSelection | None:
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return self.selection


class _ToolHooks(RunnerHooks):
    def __init__(self) -> None:
        self.results: list[object] = []

    async def on_tool_call_end(
        self, task: Task, tool_call: ToolCall, result: object
    ) -> None:
        self.results.append(result)


def _agent(
    runner: Runner,
    parent: _CaptureModel,
    baseline: _CaptureModel,
    kind: str,
    *,
    args_model: type[BaseModel] = _TaskInput,
) -> Agent:
    child = AgentDescriptor(name="Specialist", model=baseline)
    descriptor = AgentDescriptor(
        name="Parent", model=parent, model_args={"reasoning_effort": "high"}
    )
    if kind == "agent_tool":
        descriptor.tools = Tools(tools=[child.as_tool(args_model=args_model)])
    elif kind == "default_agent_tool":
        descriptor.tools = Tools(tools=[DefaultAgentTool(model=baseline)])
    elif kind == "handoff":
        descriptor.handoffs = (child,)
    else:
        assert kind == "default_handoff"
        descriptor.default_handoff = DefaultHandoff(model=baseline)
    return Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        bind_id=AgentId.from_values("parent", "one"),
        descriptor=descriptor,
    )


def _call(kind: str, *, wrapped: bool = True, parameters: object = None) -> ToolCall:
    if kind == "agent_tool":
        name, payload = "specialist", {"text": "Find the answer"}
    elif kind == "default_agent_tool":
        name, payload = "agent", {"name": "Helper", "task": "Find the answer"}
    else:
        name = "specialist" if kind == "handoff" else "handoff"
        payload = {}
    arguments: object = payload
    if wrapped:
        arguments = {
            "input": payload,
            "spawn_parameters": parameters or {"work_class": "reasoning"},
        }
    return make_tool_call(tool_id="spawn_1", name=name, arguments=json.dumps(arguments))


def _parent(call: ToolCall) -> _CaptureModel:
    return _CaptureModel(
        [
            make_assistant_response(None, tool_calls=[call]),
            make_assistant_response("parent finished"),
        ]
    )


def _state() -> RunState:
    return RunState(instructions=None, history=["Help me"], responses=[])


async def _run(runner: Runner, agent: Agent, mode: str) -> RunResult:
    if mode == "run":
        return await runner.run(agent, _state())
    stream = (
        runner.run_stream(agent, _state())
        if mode == "stream"
        else runner.run_events(agent, _state())
    )
    try:
        return await stream.result()
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["agent_tool", "default_agent_tool", "handoff", "default_handoff"]
)
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
async def test_spawn_selection_binds_each_path_before_open_run(
    kind: str, mode: str
) -> None:
    selected = _CaptureModel([make_assistant_response("child finished")])
    baseline = _CaptureModel([])
    provider = _Provider(
        SpawnModelSelection(
            SpawnModelConfig(selected, {"reasoning_effort": "low"}), label="quick"
        )
    )
    runner = Runner(spawn_model_provider=provider)
    call = _call(kind)
    parent = _parent(call)
    agent = _agent(runner, parent, baseline, kind)

    result = await _run(runner, agent, mode)

    assert result.final_output == (
        "child finished" if "handoff" in kind else "parent finished"
    )
    assert selected.args == [{"reasoning_effort": "low"}]
    assert selected.opens == selected.closes == 1
    assert baseline.opens == 0
    assert len(provider.contexts) == 1
    context = provider.contexts[0]
    assert context.kind is (
        DelegationKind.HANDOFF if "handoff" in kind else DelegationKind.TOOL
    )
    assert context.parent_id == agent.task_id
    assert context.child_id != context.parent_id
    assert context.parent_name == "Parent"
    assert context.tool_name == call.function.name
    assert context.tool_call_id == call.id
    assert context.baseline.model is baseline
    assert context.baseline.model_args == {"reasoning_effort": "high"}
    assert isinstance(context.spawn_parameters, _Choice)
    assert context.spawn_parameters.work_class == "reasoning"
    if kind == "agent_tool":
        assert context.task == _TaskInput(text="Find the answer")
    elif kind == "default_agent_tool":
        assert context.task == "Find the answer"
    else:
        assert context.task == default_handoff_task_message()

    # Routing stays out of the task. Handoffs retain the original tool call.
    child_messages = selected.calls[0]
    if "handoff" in kind:
        assert child_messages[-1]["content"] == default_handoff_task_message()
        assert child_messages[-2]["tool_call_id"] == call.id
        assert child_messages[-3]["tool_calls"][0]["function"]["arguments"] == (
            call.function.arguments
        )
    else:
        assert "spawn_parameters" not in json.dumps(child_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["run", "stream", "events"])
@pytest.mark.parametrize("select_model", [False, True])
async def test_provider_task_edits_do_not_change_child_input(
    mode: str, select_model: bool
) -> None:
    class NestedInput(BaseModel):
        text: str
        labels: list[str]

    child = _CaptureModel([make_assistant_response("child finished")])

    class Provider:
        parameters_model = None

        def resolve(self, context: SpawnModelContext) -> SpawnModelSelection | None:
            assert isinstance(context.task, NestedInput)
            context.task.text = "Policy changed the task"
            context.task.labels.append("policy-only")
            if select_model:
                return SpawnModelSelection(SpawnModelConfig(child, None))
            return None

    payload = {"text": "Original task", "labels": ["original"]}
    parent = _parent(
        make_tool_call(
            tool_id="spawn_1", name="specialist", arguments=json.dumps(payload)
        )
    )
    runner = Runner(spawn_model_provider=Provider())
    agent = _agent(runner, parent, child, "agent_tool", args_model=NestedInput)

    await _run(runner, agent, mode)

    assert json.loads(str(child.calls[0][-1]["content"])) == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("with_provider", [False, True])
@pytest.mark.parametrize(
    "kind", ["agent_tool", "default_agent_tool", "handoff", "default_handoff"]
)
async def test_spawn_baseline_without_model_preferences(
    kind: str, with_provider: bool
) -> None:
    baseline = _CaptureModel([make_assistant_response("baseline finished")])
    provider = _Provider(None, parameters_model=None)
    runner = Runner(spawn_model_provider=provider if with_provider else None)
    parent = _parent(_call(kind, wrapped=False))
    agent = _agent(runner, parent, baseline, kind)

    await runner.run(agent, _state())

    assert baseline.args == [{"reasoning_effort": "high"}]
    assert len(provider.contexts) == int(with_provider)
    if with_provider:
        assert provider.contexts[0].spawn_parameters is None
    catalog = parent.catalogs[0]
    assert catalog is not None
    assert "input" not in catalog.normalized_tools[0].schema["parameters"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_args", [None, {}])
@pytest.mark.parametrize("fallback", [False, True])
async def test_selection_replaces_all_request_args(
    model_args: Mapping[str, object] | None, fallback: bool
) -> None:
    baseline = _CaptureModel([make_assistant_response("selected baseline model")])
    provider = _Provider(
        SpawnModelSelection(
            SpawnModelConfig(baseline, model_args),
            fallback_reason="route_unavailable" if fallback else None,
        )
    )
    runner = Runner(spawn_model_provider=provider)
    parent = _parent(_call("agent_tool"))
    agent = _agent(runner, parent, baseline, "agent_tool")

    await runner.run(agent, _state())

    assert baseline.args == [model_args]
    assert len(provider.contexts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("with_provider", [False, True])
async def test_disabled_preferences_preserve_task_field_named_spawn_parameters(
    with_provider: bool,
) -> None:
    baseline = _CaptureModel([make_assistant_response("finished")])
    provider = _Provider(None, parameters_model=None)
    runner = Runner(spawn_model_provider=provider if with_provider else None)
    parent = _parent(
        make_tool_call(
            tool_id="spawn_1",
            name="specialist",
            arguments='{"spawn_parameters":"ordinary task data"}',
        )
    )
    agent = _agent(
        runner, parent, baseline, "agent_tool", args_model=_TaskWithRoutingField
    )

    await runner.run(agent, _state())

    assert "ordinary task data" in json.dumps(baseline.calls)
    if with_provider:
        assert provider.contexts[0].spawn_parameters is None
        assert provider.contexts[0].task == _TaskWithRoutingField(
            spawn_parameters="ordinary task data"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        '{"input":{"text":"task"},"spawn_parameters":{"work_class":"invalid"}}',
        '{"input":{"text":"task"},"spawn_parameters":{"work_class":"quick","secret":1}}',
        '{"input":{"text":"task"},"spawn_parameters":{},"extra":1}',
        '{"input":{"text":"task"},"spawn_parameters":"quick"}',
    ],
)
async def test_invalid_routing_arguments_fail_before_provider(arguments: str) -> None:
    provider = _Provider(None)
    runner = Runner(spawn_model_provider=provider)
    baseline = _CaptureModel([])
    parent = _parent(
        make_tool_call(tool_id="spawn_1", name="specialist", arguments=arguments)
    )
    agent = _agent(runner, parent, baseline, "agent_tool")

    with pytest.raises(ModelBehaviorError):
        await runner.run(agent, _state())

    assert provider.contexts == []
    assert baseline.opens == 0


@pytest.mark.asyncio
async def test_schema_and_validation_share_nullable_default_and_reject_extra_fields() -> (
    None
):
    baseline = _CaptureModel([make_assistant_response("finished")])
    provider = _Provider(None)
    runner = Runner(spawn_model_provider=provider)
    parent = _parent(
        make_tool_call(
            tool_id="spawn_1",
            name="specialist",
            arguments='{"input":{"text":"task"},"spawn_parameters":{}}',
        )
    )
    agent = _agent(runner, parent, baseline, "agent_tool")

    await runner.run(agent, _state())

    parameters = provider.contexts[0].spawn_parameters
    assert isinstance(parameters, _Choice)
    assert parameters.work_class is None
    catalog = parent.catalogs[0]
    assert catalog is not None
    schema = catalog.normalized_tools[0].schema["parameters"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"input", "spawn_parameters"}
    routing = schema["properties"]["spawn_parameters"]
    if "$ref" in routing:
        routing = schema["$defs"][routing["$ref"].split("/")[-1]]
    assert routing["additionalProperties"] is False
    assert routing["required"] == ["work_class"]
    assert {"type": "null"} in routing["properties"]["work_class"]["anyOf"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["agent_tool", "default_agent_tool"])
async def test_expected_refusal_is_a_typed_tool_failure(kind: str) -> None:
    provider = _Provider(None)
    provider.error = SpawnModelRejected("route_denied")
    runner = Runner(spawn_model_provider=provider)
    baseline = _CaptureModel([])
    parent = _parent(_call(kind))
    agent = _agent(runner, parent, baseline, kind)
    hooks = _ToolHooks()

    result = await runner.run(agent, _state(), hooks=hooks)

    assert result.final_output == "parent finished"
    assert len(hooks.results) == 1
    failure = hooks.results[0]
    assert isinstance(failure, ToolFailure)
    assert "route_denied" in failure.error.message
    assert baseline.opens == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["handoff", "default_handoff"])
@pytest.mark.parametrize("mode", ["run", "stream"])
async def test_handoff_refusal_prevents_child_activation(kind: str, mode: str) -> None:
    provider = _Provider(None)
    provider.error = SpawnModelRejected("route_denied")
    runner = Runner(spawn_model_provider=provider)
    baseline = _CaptureModel([])
    parent = _parent(_call(kind))
    agent = _agent(runner, parent, baseline, kind)

    with pytest.raises(SpawnModelRejected, match="route_denied"):
        await _run(runner, agent, mode)

    assert baseline.opens == 0
    assert len(provider.contexts) == 1


@pytest.mark.asyncio
async def test_unexpected_policy_error_is_not_fallback() -> None:
    provider = _Provider(None)
    provider.error = RuntimeError("policy bug")
    runner = Runner(spawn_model_provider=provider)
    baseline = _CaptureModel([])
    parent = _parent(_call("agent_tool"))
    agent = _agent(runner, parent, baseline, "agent_tool")

    with pytest.raises(RuntimeError, match="policy bug"):
        await runner.run(agent, _state())

    assert baseline.opens == 0


@pytest.mark.asyncio
async def test_selected_nested_args_are_owned_before_child_request() -> None:
    nested = {"budget": [1, 2]}
    model_args: dict[str, object] = {"provider": nested}
    selected = _CaptureModel([make_assistant_response("finished")])
    selected.before_request = lambda: nested["budget"].append(3)
    provider = _Provider(SpawnModelSelection(SpawnModelConfig(selected, model_args)))
    runner = Runner(spawn_model_provider=provider)
    baseline = _CaptureModel([])
    parent = _parent(_call("agent_tool"))
    agent = _agent(runner, parent, baseline, "agent_tool")

    await runner.run(agent, _state())

    assert selected.args == [{"provider": {"budget": [1, 2]}}]
    assert nested == {"budget": [1, 2, 3]}


def test_descriptor_owns_nested_arguments() -> None:
    nested = {"budget": [1, 2]}
    descriptor = AgentDescriptor(model_args={"provider": nested})
    nested["budget"].append(3)
    assert descriptor.model_args == {"provider": {"budget": [1, 2]}}


@pytest.mark.asyncio
async def test_cancelled_policy_releases_generic_child_slot() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingProvider:
        parameters_model = None
        calls = 0

        async def resolve(
            self, context: SpawnModelContext
        ) -> SpawnModelSelection | None:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return None

    provider = BlockingProvider()
    runner = Runner(spawn_model_provider=provider, agent_max_threads=1)
    baseline = _CaptureModel([make_assistant_response("after cancellation")])
    first_parent = _parent(_call("default_agent_tool", wrapped=False))
    first = _agent(runner, first_parent, baseline, "default_agent_tool")
    token = CancellationToken()
    pending = asyncio.create_task(runner.run(first, _state(), cancellation_token=token))
    await asyncio.wait_for(entered.wait(), timeout=2)
    token.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=2)
    assert cancelled.is_set()
    assert baseline.opens == 0

    # A second public run proves that the one available slot was released.
    second_parent = _parent(_call("default_agent_tool", wrapped=False))
    second = _agent(runner, second_parent, baseline, "default_agent_tool")
    result = await runner.run(second, _state())
    assert result.final_output == "parent finished"
    assert provider.calls == 2
    assert baseline.opens == baseline.closes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [None, False, True])
async def test_envelope_keeps_shared_tool_schema_and_strict_mode(
    strict: bool | None,
) -> None:
    baseline = _CaptureModel([make_assistant_response("finished")])
    provider = _Provider(None)
    runner = Runner(spawn_model_provider=provider)
    parent = _parent(_call("agent_tool"))
    child = AgentDescriptor(name="Specialist", model=baseline)
    tool = child.as_tool(args_model=_TaskInput)
    tool.strict = strict
    original_schema = deepcopy(tool.schema)
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        descriptor=AgentDescriptor(
            name="Parent", model=parent, tools=Tools(tools=[tool])
        ),
    )

    await runner.run(agent, _state())

    assert tool.schema == original_schema
    assert tool.args_type() is _TaskInput
    catalog = parent.catalogs[0]
    assert catalog is not None
    shown_tool = catalog.normalized_tools[0]
    assert shown_tool.name == tool.name
    assert shown_tool.description == tool.description
    assert shown_tool.strict is strict
    assert set(shown_tool.schema["parameters"]["properties"]) == {
        "input",
        "spawn_parameters",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["not a selection", 42])
async def test_invalid_provider_result_fails_without_model_retry(
    invalid: object,
) -> None:
    class InvalidProvider:
        parameters_model = None

        def resolve(self, context: SpawnModelContext) -> Any:
            return invalid

    runner = Runner(spawn_model_provider=InvalidProvider())
    baseline = _CaptureModel([])
    parent = _parent(_call("agent_tool", wrapped=False))
    agent = _agent(runner, parent, baseline, "agent_tool")

    with pytest.raises(TypeError):
        await runner.run(agent, _state())

    assert baseline.opens == 0
    assert len(parent.calls) == 1


class _TraceRecorder(TracingProcessor):
    def __init__(self) -> None:
        self.spans: list[Span[Any]] = []

    def on_trace_start(self, trace: Any) -> None:
        pass

    def on_trace_end(self, trace: Any) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


class _TracedModel(_CaptureModel):
    @property
    def tracing(self) -> ModelTracing:
        return ModelTracing.ENABLED_WITHOUT_DATA

    def request_metadata(
        self, extra_call_args: dict[str, Any] | None = None
    ) -> dict[str, str]:
        return {"model": "configured-test-model", "requested_effort": "low"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["agent_tool", "default_agent_tool", "handoff", "default_handoff"]
)
@pytest.mark.parametrize(
    "status", ["baseline", "selected", "fallback", "rejected", "error"]
)
async def test_spawn_trace_is_correlated_and_omits_private_payload(
    kind: str, status: str
) -> None:
    recorder = _TraceRecorder()
    tracing = DefaultTraceProvider()
    tracing.register_processor(recorder)
    previous = get_trace_provider()
    baseline = _TracedModel([make_assistant_response("finished")])
    selection = (
        None
        if status == "baseline"
        else SpawnModelSelection(
            SpawnModelConfig(baseline, {"secret": "ARGS_CANARY"}),
            label="quick",
            fallback_reason="router_unavailable" if status == "fallback" else None,
        )
    )
    provider = _Provider(selection)
    if status == "rejected":
        provider.error = SpawnModelRejected("route_denied")
    elif status == "error":
        provider.error = RuntimeError("EXCEPTION_CANARY")
    runner = Runner(spawn_model_provider=provider)
    call = _call(kind)
    parent = _TracedModel(
        [
            make_assistant_response(None, tool_calls=[call]),
            make_assistant_response("parent finished"),
        ]
    )
    agent = _agent(runner, parent, baseline, kind)
    set_trace_provider(tracing)
    try:
        with trace("spawn-test"):
            if status == "error" or (status == "rejected" and "handoff" in kind):
                with pytest.raises((RuntimeError, SpawnModelRejected)):
                    await runner.run(agent, _state())
            else:
                await runner.run(agent, _state())
    finally:
        set_trace_provider(previous)

    decisions = [
        span
        for span in recorder.spans
        if isinstance(span.span_data, CustomSpanData)
        and span.span_data.name == "spawn_model_decision"
    ]
    assert len(decisions) == 1
    decision = decisions[0]
    data = decision.span_data.data
    assert data is not None
    assert data["kind"] == (
        DelegationKind.HANDOFF.value if "handoff" in kind else DelegationKind.TOOL.value
    )
    assert data["status"] == status
    assert data["tool_call_id"] == call.id
    assert data["child_id"] == str(provider.contexts[0].child_id)
    assert decision.parent_id in {
        span.span_id for span in recorder.spans if span.span_data.type == "generation"
    }
    if status in {"baseline", "selected", "fallback"}:
        assert data["model"] == "configured-test-model"
        assert data["requested_effort"] == "low"
    if status == "fallback":
        assert data["fallback_reason"] == "router_unavailable"
    if status == "rejected":
        assert data["refusal_reason"] == "route_denied"
    serialized = json.dumps(data)
    assert "ARGS_CANARY" not in serialized
    assert "EXCEPTION_CANARY" not in serialized
    assert "Find the answer" not in serialized
    assert "baseline" not in data
    assert "config" not in data


@pytest.mark.asyncio
async def test_parallel_spawns_keep_model_assignment_for_later_messages() -> None:
    """Two pending instances of one tool type must retain separate clients."""
    policy_ready = asyncio.Event()
    contexts: dict[str, SpawnModelContext] = {}
    provider_calls: list[SpawnModelContext] = []
    left = _CaptureModel(
        [
            make_assistant_response("left first"),
            make_assistant_response("left later"),
        ]
    )
    right = _CaptureModel([make_assistant_response("right first")])
    clients = {"left": left, "right": right}

    class OverlapRuntime(SingleThreadedRuntimeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.pending = 0
            self.ready = asyncio.Event()

        async def _submit(self, task: DeliveryTask) -> None:
            # Keep both initial child registrations pending before dispatch.
            self.pending += 1
            if self.pending == 2:
                self.ready.set()
            await self.ready.wait()
            await super()._submit(task)

    class Provider:
        parameters_model = None

        async def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
            assert isinstance(context.task, _TaskInput)
            name = context.task.text
            contexts[name] = context
            provider_calls.append(context)
            if len(contexts) == 2:
                policy_ready.set()
            await policy_ready.wait()
            return SpawnModelSelection(
                SpawnModelConfig(clients[name], {"route": name}),
                label=name,
            )

    runner = Runner(spawn_model_provider=Provider())
    runtime = OverlapRuntime()
    calls = [
        make_tool_call(
            tool_id=name, name="specialist", arguments=json.dumps({"text": name})
        )
        for name in ("left", "right")
    ]
    parent = _CaptureModel(
        [
            make_assistant_response(None, tool_calls=calls),
            make_assistant_response("parent finished"),
        ]
    )
    agent = Agent(
        runtime,
        runner,
        descriptor=AgentDescriptor(
            name="Parent",
            model=parent,
            tools=Tools(
                tools=[
                    AgentDescriptor(name="Specialist").as_tool(args_model=_TaskInput)
                ],
                parallel_tool_calls=True,
            ),
        ),
    )

    try:
        result = await asyncio.wait_for(runner.run(agent, _state()), timeout=3)
        assert result.final_output == "parent finished"
        assert len(contexts) == 2
        assert contexts["left"].child_id != contexts["right"].child_id
        assert contexts["left"].child_id.type == contexts["right"].child_id.type
        assert left.args == [{"route": "left"}]
        assert right.args == [{"route": "right"}]
        assert left.opens == right.opens == 1

        # Delivery to the live identity continues its assigned model and history.
        outcome = await runtime.send_message("Follow up", contexts["left"].child_id)
        assert outcome.status is DeliveryStatus.DELIVERED
        assert isinstance(outcome.response_payload, RunResult)
        assert outcome.response_payload.final_output == "left later"
        assert len(provider_calls) == 2
        assert left.args == [{"route": "left"}, {"route": "left"}]
        assert right.args == [{"route": "right"}]
        assert left.opens == left.closes == 2
        assert left.calls[-1][-1]["content"] == "Follow up"
        assert any(message.get("content") == "left first" for message in left.calls[-1])
    finally:
        await runtime.stop_when_idle()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["agent_tool", "default_agent_tool", "handoff", "default_handoff"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_sync_provider_cancellation_prevents_child_registration(
    kind: str, streaming: bool
) -> None:
    token = CancellationToken()
    contexts: list[SpawnModelContext] = []
    selected = _CaptureModel([])

    class Provider:
        parameters_model = None

        def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
            contexts.append(context)
            token.cancel()
            return SpawnModelSelection(SpawnModelConfig(selected, None))

    runner = Runner(spawn_model_provider=Provider())
    parent = _parent(_call(kind, wrapped=False))
    agent = _agent(runner, parent, selected, kind)
    if streaming:
        stream = runner.run_stream(agent, _state(), cancellation_token=token)
        try:
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
        finally:
            await stream.aclose()
    else:
        with pytest.raises(asyncio.CancelledError):
            await runner.run(agent, _state(), cancellation_token=token)

    assert len(contexts) == 1
    assert selected.opens == 0
    # No delivery was queued, so no runtime workers were started for this child.
    outcome = await agent.send_message("Follow up", recipient=contexts[0].child_id)
    assert outcome.status is DeliveryStatus.UNDELIVERABLE


@pytest.mark.asyncio
async def test_automatic_successor_keeps_spawn_envelope_when_catalog_changes() -> None:
    original = _CaptureModel([make_assistant_response("original child")])
    replacement = _CaptureModel([])
    rebuilt_args: list[type[BaseModel]] = []

    class ReplacementInput(BaseModel):
        replacement_task: int

    class Catalog(Shim):
        @property
        def name(self) -> str:
            return "changing-spawn-catalog"

        async def prepare_turn(self, turn: PreparedTurn) -> None:
            # The successor was already started with the original task schema.
            first = turn.run_state.turn_count == 1
            args_model = _TaskInput if first else ReplacementInput
            rebuilt_args.append(args_model)
            descriptor = AgentDescriptor(
                name="Specialist",
                model=original if first else replacement,
            )
            turn.tools = Tools(tools=[descriptor.as_tool(args_model=args_model)])

    class AutomaticModel(_CaptureModel):
        async def stream_response(
            self,
            messages: list[MessageDict],
            extra_call_args: dict[str, Any] | None = None,
            schema: type[BaseModel] | OutputSchema[Any] | None = None,
            tools: Tools | None = None,
            cancellation_token: CancellationToken | None = None,
            parent_span: Span[Any] | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ModelStreamEvent]:
            response = await self.get_response(
                messages,
                extra_call_args,
                schema,
                tools,
                cancellation_token,
                parent_span,
                **kwargs,
            )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response,
                continuation="automatic" if len(self.calls) == 1 else None,
            )

    provider = _Provider(None)
    runner = Runner(spawn_model_provider=provider)
    parent = AutomaticModel(
        [
            make_assistant_response("The provider starts a successor"),
            make_assistant_response(None, tool_calls=[_call("agent_tool")]),
            make_assistant_response("parent finished"),
        ]
    )
    agent = Agent(
        SingleThreadedRuntimeEngine(),
        runner,
        descriptor=AgentDescriptor(
            name="Parent",
            model=parent,
            shims=[Catalog()],
        ),
    )
    # The lifecycle binds the dynamic catalog before it invokes the runner.
    stream = await agent.enqueue_input_stream(_state())
    try:
        result = await stream.result()
    finally:
        await stream.aclose()

    assert result.final_output == "parent finished"
    assert len(provider.contexts) == 1
    assert provider.contexts[0].task == _TaskInput(text="Find the answer")
    assert isinstance(provider.contexts[0].spawn_parameters, _Choice)
    assert provider.contexts[0].baseline.model is original
    assert original.opens == 1
    assert replacement.opens == 0
    assert ReplacementInput in rebuilt_args

    # Both the delivered schema and its parser stay bound to the accepted catalog.
    first_catalog, successor_catalog = parent.catalogs[:2]
    assert first_catalog is not None and successor_catalog is not None
    first_tool = first_catalog.normalized_tools[0]
    successor_tool = successor_catalog.normalized_tools[0]
    assert successor_tool.schema == first_tool.schema
    assert successor_tool.args_type() is first_tool.args_type()
