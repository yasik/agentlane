"""Tests for the delegate-by-default shim bases.

These cover `DelegatingBoundShim` and `DelegatingShim`: the reflection drift
guards that prove the bases forward every public callback, plus behavioral
tests that prove forwarding actually reaches the inner shim and that overrides
compose with the delegated default.
"""

import asyncio
import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agentlane.harness import RunResult, RunState
from agentlane.harness._task import Task
from agentlane.harness.shims import (
    BoundShim,
    DelegatingBoundShim,
    DelegatingShim,
    PreparedTurn,
    Shim,
    ShimBindingContext,
    ToolSourceBinding,
)
from agentlane.harness.shims._manager import BoundShimManager
from agentlane.messaging import AgentId
from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import DefaultRunContext, RunContext
from agentlane.runtime import SingleThreadedRuntimeEngine

from .test_shims import make_assistant_response
from .tools_test_utils import named_tool


def _public_callbacks(cls: type) -> set[str]:
    """Return the public callbacks/properties declared across `cls`'s MRO.

    Walks every class in the MRO except `object` so a callback added on a base
    class still counts. Includes properties (e.g. `name`) because the
    delegating bases must forward those too.
    """
    names: set[str] = set()
    for base in cls.__mro__:
        if base is object:
            continue
        for name, member in vars(base).items():
            if name.startswith("_"):
                continue
            if callable(member) or isinstance(member, property):
                names.add(name)
    return names


def _leaf_overrides(cls: type) -> set[str]:
    """Return public callbacks/properties declared on `cls` itself only.

    Reads only `vars(cls)` — the leaf class body — so inherited members are
    excluded. A delegating base that fails to forward an upstream callback is
    absent from this set and fails the guard.
    """
    names: set[str] = set()
    for name, member in vars(cls).items():
        if name.startswith("_"):
            continue
        if callable(member) or isinstance(member, property):
            names.add(name)
    return names


def test_delegating_bound_shim_covers_every_callback() -> None:
    """Every public `BoundShim` callback is forwarded by `DelegatingBoundShim`.

    Compares the framework callback surface (MRO-walked over `BoundShim`)
    against the delegating base's own leaf-declared overrides. A new callback
    added to `BoundShim` that the base does not explicitly forward would fall
    through to the inert no-op default and silently bypass delegation; this
    test fails first when that happens.
    """
    framework_callbacks = _public_callbacks(BoundShim)
    delegated = _leaf_overrides(DelegatingBoundShim)
    # `inner` is the wrapper's own accessor, not a forwarded BoundShim callback.
    delegated.discard("inner")
    missing = sorted(framework_callbacks - delegated)
    assert missing == [], f"DelegatingBoundShim does not forward: {missing}"


def test_delegating_shim_covers_every_callback() -> None:
    """Every public `Shim` callback is forwarded by `DelegatingShim`."""
    framework_callbacks = _public_callbacks(Shim)
    delegated = _leaf_overrides(DelegatingShim)
    delegated.discard("inner")
    missing = sorted(framework_callbacks - delegated)
    assert missing == [], f"DelegatingShim does not forward: {missing}"


def test_delegating_bound_shim_forwarded_signatures_match_boundshim() -> None:
    """Forwarded callbacks keep the same signature as the `BoundShim` base.

    Forwarding the wrong arguments is as silent a failure as not forwarding at
    all, so the drift guard also pins parameter shape.
    """
    for name in _public_callbacks(BoundShim):
        base_member = inspect.getattr_static(BoundShim, name)
        delegating_member = inspect.getattr_static(DelegatingBoundShim, name)
        if isinstance(base_member, property):
            continue
        assert inspect.signature(base_member) == inspect.signature(
            delegating_member
        ), f"DelegatingBoundShim.{name} signature drifted from BoundShim"


class _RecordingBoundShim(BoundShim):
    """Inner bound shim that records which callbacks the delegate forwarded."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.transformed: list[MessageDict] | None = None

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        del state, transient_state
        self.calls.append("on_run_start")

    async def on_input_messages(
        self, state: RunState, messages: Sequence[MessageDict]
    ) -> None:
        del state, messages
        self.calls.append("on_input_messages")

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        del turn
        self.calls.append("prepare_turn")

    async def prepare_request(self, turn: PreparedTurn) -> None:
        del turn
        self.calls.append("prepare_request")

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        del turn
        self.calls.append("transform_messages")
        self.transformed = [{"role": "user", "content": "from-inner"}]
        return self.transformed

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        del turn, response
        self.calls.append("on_model_response")

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        del result, transient_state
        self.calls.append("on_run_end")

    def runner_hooks(self) -> tuple[Any, ...]:
        self.calls.append("runner_hooks")
        return ()


def _make_prepared_turn() -> PreparedTurn:
    """Build one real prepared turn for forwarding tests."""
    run_state = RunState(instructions="Base", history=[], responses=[])
    return PreparedTurn(run_state=run_state, tools=None, model_args=None)


def test_delegating_bound_shim_forwards_unoverridden_callbacks() -> None:
    """An un-overridden delegate forwards every callback to its inner shim."""

    async def scenario() -> None:
        inner = _RecordingBoundShim()
        delegate = DelegatingBoundShim(inner)
        transient: RunContext[Any] = DefaultRunContext()
        turn = _make_prepared_turn()
        response = make_assistant_response("done")

        await delegate.on_run_start(turn.run_state, transient)
        await delegate.on_input_messages(turn.run_state, [])
        await delegate.prepare_turn(turn)
        await delegate.prepare_request(turn)
        result = await delegate.transform_messages(turn, [])
        await delegate.on_model_response(turn, response)
        await delegate.on_run_end(None, transient)
        hooks = delegate.runner_hooks()

        assert inner.calls == [
            "on_run_start",
            "on_input_messages",
            "prepare_turn",
            "prepare_request",
            "transform_messages",
            "on_model_response",
            "on_run_end",
            "runner_hooks",
        ]
        assert result == inner.transformed
        assert hooks == ()
        assert delegate.inner is inner

    asyncio.run(scenario())


def test_delegating_bound_shim_override_composes_with_default_forwarding() -> None:
    """Overriding one callback still forwards the rest to the inner shim."""

    class _OnlyOverridesPrepareTurn(DelegatingBoundShim):
        def __init__(self, inner: BoundShim) -> None:
            super().__init__(inner)
            self.prepared = 0

        async def prepare_turn(self, turn: PreparedTurn) -> None:
            self.prepared += 1
            await super().prepare_turn(turn)

    async def scenario() -> None:
        inner = _RecordingBoundShim()
        wrapper = _OnlyOverridesPrepareTurn(inner)
        transient: RunContext[Any] = DefaultRunContext()
        turn = _make_prepared_turn()

        await wrapper.prepare_turn(turn)
        await wrapper.on_run_end(None, transient)

        assert wrapper.prepared == 1
        # Overridden callback still reaches the inner via super(); the
        # un-overridden one is forwarded by the base.
        assert inner.calls == ["prepare_turn", "on_run_end"]

    asyncio.run(scenario())


class _RecordingShim(Shim):
    """Inner shim definition that records forwarded definition-level calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.bound = _RecordingBoundShim()

    @property
    def name(self) -> str:
        return "recording-inner"

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        del context
        self.calls.append("bind")
        return self.bound

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        del turn
        self.calls.append("prepare_turn")

    async def prepare_request(self, turn: PreparedTurn) -> None:
        del turn
        self.calls.append("prepare_request")

    async def on_input_messages(
        self, state: RunState, messages: Sequence[MessageDict]
    ) -> None:
        del state, messages
        self.calls.append("on_input_messages")


def _make_binding_context() -> ShimBindingContext:
    """Build one real shim binding context backed by a runtime task."""
    engine = SingleThreadedRuntimeEngine()
    task = Task(engine, bind_id=AgentId.from_values("delegating-shim", "x"))
    return ShimBindingContext(task=task)


def test_delegating_shim_forwards_name_and_callbacks() -> None:
    """`DelegatingShim` forwards `name`, `bind`, and lifecycle callbacks."""

    async def scenario() -> None:
        inner = _RecordingShim()
        delegate = DelegatingShim(inner)

        assert delegate.name == "recording-inner"
        assert delegate.inner is inner

        bound = await delegate.bind(_make_binding_context())
        assert bound is inner.bound
        assert inner.calls == ["bind"]

        turn = _make_prepared_turn()
        await delegate.prepare_turn(turn)
        await delegate.prepare_request(turn)
        await delegate.on_input_messages(turn.run_state, [])
        assert inner.calls == [
            "bind",
            "prepare_turn",
            "prepare_request",
            "on_input_messages",
        ]

        # The base binding adapter must forward the new callback as well.
        adapter = await Shim.bind(inner, _make_binding_context())
        await adapter.prepare_request(turn)
        assert inner.calls[-1] == "prepare_request"
        await adapter.on_input_messages(turn.run_state, [])
        assert inner.calls[-1] == "on_input_messages"
        assert inner.calls.count("on_input_messages") == 2

    asyncio.run(scenario())


def test_delegating_shim_bind_can_wrap_inner_bound_session() -> None:
    """A `DelegatingShim` subclass can wrap the inner bound shim on bind.

    This is the pattern that retires Vera's manual delegate: override `bind`
    only, wrap the inner bound session in a `DelegatingBoundShim` subclass, and
    let every other callback forward by default.
    """

    class _InjectingBoundShim(DelegatingBoundShim):
        def __init__(self, inner: BoundShim) -> None:
            super().__init__(inner)
            self.injected = 0

        async def prepare_turn(self, turn: PreparedTurn) -> None:
            self.injected += 1
            await super().prepare_turn(turn)

    class _WrappingShim(DelegatingShim):
        async def bind(self, context: ShimBindingContext) -> BoundShim:
            return _InjectingBoundShim(await self.inner.bind(context))

    async def scenario() -> None:
        inner = _RecordingShim()
        wrapper = _WrappingShim(inner)

        bound = await wrapper.bind(_make_binding_context())
        assert isinstance(bound, _InjectingBoundShim)

        turn = _make_prepared_turn()
        await bound.prepare_turn(turn)

        assert bound.injected == 1
        # The wrapped bound session still forwarded prepare_turn to the inner.
        assert inner.bound.calls == ["prepare_turn"]

    asyncio.run(scenario())


def test_delegating_shims_forward_dynamic_tool_inheritance() -> None:
    """Source replacement results and name restrictions cross both wrappers."""
    names = frozenset({"remote__read"})
    seen: list[frozenset[str]] = []

    class SourceSession(BoundShim):
        def inherit_tools(
            self, names: frozenset[str]
        ) -> tuple[ToolSourceBinding, ...] | None:
            seen.append(names)
            return ()

    class SourceDefinition(Shim):
        @property
        def name(self) -> str:
            return "source"

        def inherit_tools(
            self, names: frozenset[str]
        ) -> tuple[ToolSourceBinding, ...] | None:
            seen.append(names)
            return ()

    assert DelegatingBoundShim(SourceSession()).inherit_tools(names) == ()
    assert DelegatingShim(SourceDefinition()).inherit_tools(names) == ()
    assert seen == [names, names]


@dataclass
class _BoundToolSource(BoundShim):
    source: object
    names: frozenset[str]

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        turn.add_tools(tuple(named_tool(name) for name in sorted(self.names)))

    def inherit_tools(self, names: frozenset[str]) -> tuple[ToolSourceBinding, ...]:
        allowed = self.names & names

        async def bind_restricted(context: ShimBindingContext) -> BoundShim:
            del context
            return _BoundToolSource(self.source, allowed)

        return (ToolSourceBinding(source=self.source, bind=bind_restricted),)


@dataclass
class _EqualToolSource(Shim):
    """Equal, unhashable definitions still denote separate source identities."""

    group: str
    names: frozenset[str] = field(compare=False)

    @property
    def name(self) -> str:
        return self.group

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        for binding in context.tool_source_bindings:
            if binding.source is self:
                return await binding.bind(context)

        if context.tool_source_bindings:
            return BoundShim()

        return _BoundToolSource(self, self.names)


@dataclass
class _CombinedSourcesBound(BoundShim):
    sessions: tuple[BoundShim, ...]

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        for session in self.sessions:
            await session.prepare_turn(turn)

    def inherit_tools(self, names: frozenset[str]) -> tuple[ToolSourceBinding, ...]:
        return tuple(
            binding
            for session in self.sessions
            for binding in session.inherit_tools(names) or ()
        )


@dataclass
class _CombinedSources(Shim):
    sources: tuple[_EqualToolSource, ...]
    contexts: list[ShimBindingContext] = field(default_factory=list[ShimBindingContext])

    @property
    def name(self) -> str:
        return "combined-sources"

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        self.contexts.append(context)
        return _CombinedSourcesBound(
            tuple([await source.bind(context) for source in self.sources])
        )


class _CountingWrapper(DelegatingShim):
    def __init__(self, inner: Shim) -> None:
        super().__init__(inner)
        self.sessions: list[BoundShim] = []

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        session = DelegatingBoundShim(await self.inner.bind(context))
        self.sessions.append(session)
        return session


async def _visible_source_names(manager: BoundShimManager) -> set[str]:
    turn = _make_prepared_turn()
    await manager.prepare_turn(turn)
    return (
        {tool.name for tool in turn.tools.normalized_tools}
        if turn.tools is not None
        else set()
    )


def test_tool_source_unmatched_binding_exposes_no_tools() -> None:
    async def scenario() -> None:
        included = _EqualToolSource("same", frozenset({"read"}))
        excluded = _EqualToolSource("same", frozenset({"write"}))
        assert included == excluded
        assert type(included).__hash__ is None
        parent = await included.bind(_make_binding_context())
        bindings = parent.inherit_tools(frozenset({"read"}))
        assert bindings

        context = _make_binding_context()
        context.tool_source_bindings = bindings
        child = await BoundShimManager.bind(shims=(excluded,), context=context)

        assert await _visible_source_names(child) == set()

    asyncio.run(scenario())


def test_multiple_source_bindings_keep_identity_and_bind_wrapper_once() -> None:
    async def scenario() -> None:
        first = _EqualToolSource("same", frozenset({"alpha", "alpha_extra"}))
        second = _EqualToolSource("same", frozenset({"beta", "beta_extra"}))
        assert first == second
        assert type(first).__hash__ is None
        combined = _CombinedSources((first, second))
        wrapper = _CountingWrapper(combined)
        parent = await BoundShimManager.bind(
            shims=(wrapper,), context=_make_binding_context()
        )
        child_definitions = parent.child_shims(
            frozenset({"alpha", "beta"}), inherit_definitions=True
        )
        assert len(child_definitions) == 1
        child = await BoundShimManager.bind(
            shims=child_definitions, context=_make_binding_context()
        )

        assert await _visible_source_names(child) == {"alpha", "beta"}
        assert await _visible_source_names(parent) == {
            "alpha",
            "alpha_extra",
            "beta",
            "beta_extra",
        }
        assert len(wrapper.sessions) == len(combined.contexts) == 2
        assert wrapper.sessions[0] is not wrapper.sessions[1]
        bindings = combined.contexts[1].tool_source_bindings
        assert len(bindings) == 2
        assert bindings[0].source is first
        assert bindings[1].source is second

    asyncio.run(scenario())


def test_source_binding_none_and_empty_keep_distinct_inheritance_semantics() -> None:
    class SuppressedShim(_RecordingShim):
        def inherit_tools(self, names: frozenset[str]) -> tuple[ToolSourceBinding, ...]:
            del names
            return ()

        async def bind(self, context: ShimBindingContext) -> BoundShim:
            return await Shim.bind(self, context)

    async def scenario() -> None:
        normal = _RecordingShim()
        suppressed = SuppressedShim()
        source = _EqualToolSource("dynamic", frozenset({"read"}))
        parent = await BoundShimManager.bind(
            shims=(normal, suppressed, source), context=_make_binding_context()
        )

        inherited = parent.child_shims(frozenset({"read"}), inherit_definitions=True)
        assert len(inherited) == 2
        assert inherited[0] is normal
        assert all(shim is not suppressed for shim in inherited)
        restricted = parent.child_shims(frozenset({"read"}), inherit_definitions=False)
        assert len(restricted) == 1
        assert restricted[0] is not normal
        child = await BoundShimManager.bind(
            shims=restricted, context=_make_binding_context()
        )
        assert await _visible_source_names(child) == {"read"}

    asyncio.run(scenario())


def test_grandchild_source_bindings_replace_older_caps_without_unwrapping() -> None:
    async def scenario() -> None:
        first = _EqualToolSource("same", frozenset({"alpha", "alpha_extra"}))
        second = _EqualToolSource("same", frozenset({"beta", "beta_extra"}))
        combined = _CombinedSources((first, second))
        wrapper = _CountingWrapper(combined)
        parent = await BoundShimManager.bind(
            shims=(wrapper,), context=_make_binding_context()
        )
        child = await BoundShimManager.bind(
            shims=parent.child_shims(
                frozenset({"alpha", "beta"}), inherit_definitions=True
            ),
            context=_make_binding_context(),
        )
        grandchild = await BoundShimManager.bind(
            shims=child.child_shims(frozenset({"alpha"}), inherit_definitions=True),
            context=_make_binding_context(),
        )

        assert await _visible_source_names(grandchild) == {"alpha"}
        assert await _visible_source_names(child) == {"alpha", "beta"}
        assert len(wrapper.sessions) == len(combined.contexts) == 3
        assert len({id(session) for session in wrapper.sessions}) == 3
        assert len(combined.contexts[2].tool_source_bindings) == 2

    asyncio.run(scenario())
