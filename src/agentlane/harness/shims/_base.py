"""Base shim contracts for the harness."""

import abc
from typing import Any

from agentlane.models import MessageDict, ModelResponse
from agentlane.models.run import RunContext

from .._hooks import RunnerHooks
from .._run import RunResult, RunState
from ._types import PreparedTurn, ShimBindingContext


class BoundShim:
    """Per-agent bound shim session.

    Concrete shims may override only the callbacks they need. The default
    implementations are no-ops so simple shims can stay compact.
    """

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle one run start and optionally mutate the working state."""
        _ = state
        _ = transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Mutate the prepared turn before one model request is built."""
        _ = turn

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        """Optionally replace the canonical message list for one turn."""
        _ = turn
        _ = messages
        return None

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        """Handle one completed model response and update shim state."""
        _ = turn
        _ = response

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle the end of one run."""
        _ = result
        _ = transient_state

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        """Return additional hooks for this bound shim session."""
        return ()


class DelegatingBoundShim(BoundShim):
    """Bound shim that delegates every callback to an inner bound shim.

    Wrapping a framework bound shim — to inject run state, redact arguments, or
    observe a lifecycle — otherwise forces a subclass that forwards every
    callback by hand. That boilerplate is silently incomplete the moment the
    framework adds a callback: the new callback falls through to the no-op
    ``BoundShim`` default and never reaches the wrapped shim.

    Subclass this base and override only the callbacks that change. Every other
    callback, including callbacks added in future releases, forwards to
    ``inner`` automatically. ``test_delegating_bound_shim_covers_every_callback``
    pins this guarantee so a newly added ``BoundShim`` callback cannot silently
    bypass delegation.

    Typical usage example:

      class RunStateInjectingShim(DelegatingBoundShim):
          async def prepare_turn(self, turn: PreparedTurn) -> None:
              self._consumer.set_run_state(turn.run_state)
              await super().prepare_turn(turn)
    """

    def __init__(self, inner: BoundShim) -> None:
        """Wrap one inner bound shim for delegate-by-default forwarding.

        Args:
            inner: The bound shim every un-overridden callback forwards to.
        """
        self._inner = inner

    @property
    def inner(self) -> BoundShim:
        """Return the wrapped bound shim that receives forwarded callbacks."""
        return self._inner

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        await self._inner.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        await self._inner.prepare_turn(turn)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        return await self._inner.transform_messages(turn, messages)

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        await self._inner.on_model_response(turn, response)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        await self._inner.on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return self._inner.runner_hooks()


class _ForwardingBoundShim(BoundShim):
    """Default bound adapter that forwards callbacks to one shim definition."""

    def __init__(self, shim: "Shim") -> None:
        self._shim = shim

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        await self._shim.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        await self._shim.prepare_turn(turn)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        return await self._shim.transform_messages(turn, messages)

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        await self._shim.on_model_response(turn, response)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        await self._shim.on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return self._shim.runner_hooks()


class Shim(abc.ABC):
    """Definition-time contract for one harness shim.

    Most shims should subclass this one type only and override whichever
    lifecycle callbacks they need. The default `bind(...)` implementation
    creates a simple forwarding bound session automatically.

    Override `bind(...)` only when the shim needs private per-agent in-memory
    state or custom bind-time setup.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the stable shim name used for persisted state keys."""
        raise NotImplementedError

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        """Bind the shim to one concrete agent instance."""
        _ = context
        return _ForwardingBoundShim(self)

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle one run start and optionally mutate the working state."""
        _ = state
        _ = transient_state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        """Mutate the prepared turn before one model request is built."""
        _ = turn

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        """Optionally replace the canonical message list for one turn."""
        _ = turn
        _ = messages
        return None

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        """Handle one completed model response and update shim state."""
        _ = turn
        _ = response

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        """Handle the end of one run."""
        _ = result
        _ = transient_state

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        """Return additional runner hooks for this shim definition."""
        return ()


class DelegatingShim(Shim):
    """Definition-time shim that delegates every callback to an inner shim.

    This is the bind-time analog of `DelegatingBoundShim`. Wrapping a framework
    shim at the definition level — to add bind-time setup, a paired catalog, or
    a wrapped bound session — otherwise forces a subclass that re-forwards
    `name`, `bind`, and every lifecycle callback by hand, with the same silent
    drift problem when the framework adds a callback.

    Subclass this base and override only what changes. `bind` forwards to the
    inner shim by default; override it to wrap the inner bound session, usually
    by returning a `DelegatingBoundShim` subclass around
    ``await self.inner.bind(context)``. ``name`` and every lifecycle callback
    forward to ``inner`` automatically, including callbacks added in future
    releases. ``test_delegating_shim_covers_every_callback`` pins this
    guarantee.

    Typical usage example:

      class WorkspaceToolsShim(DelegatingShim):
          async def bind(self, context: ShimBindingContext) -> BoundShim:
              return _RunStateInjectingBoundShim(await self.inner.bind(context))
    """

    def __init__(self, inner: Shim) -> None:
        """Wrap one inner shim definition for delegate-by-default forwarding.

        Args:
            inner: The shim every un-overridden callback forwards to.
        """
        self._inner = inner

    @property
    def inner(self) -> Shim:
        """Return the wrapped shim definition that receives forwarded calls."""
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        return await self._inner.bind(context)

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        await self._inner.on_run_start(state, transient_state)

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        await self._inner.prepare_turn(turn)

    async def transform_messages(
        self,
        turn: PreparedTurn,
        messages: list[MessageDict],
    ) -> list[MessageDict] | None:
        return await self._inner.transform_messages(turn, messages)

    async def on_model_response(
        self,
        turn: PreparedTurn,
        response: ModelResponse,
    ) -> None:
        await self._inner.on_model_response(turn, response)

    async def on_run_end(
        self,
        result: RunResult | None,
        transient_state: RunContext[Any],
    ) -> None:
        await self._inner.on_run_end(result, transient_state)

    def runner_hooks(self) -> tuple[RunnerHooks, ...]:
        return self._inner.runner_hooks()
