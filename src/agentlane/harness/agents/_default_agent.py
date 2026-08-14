"""Default high-level agent for stateful harness execution.

This module provides the standard agent implementation that
layers persisted run state, optional runtime and runner provisioning, and
branch execution on top of the runtime-facing harness ``Agent``.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Self
from uuid import uuid4

from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus
from agentlane.models import Model, ModelResponse
from agentlane.runtime import (
    CancellationToken,
    RuntimeEngine,
    runtime_scope,
    single_threaded_runtime,
)

from .._agent import Agent as RuntimeAgent
from .._cancellation import cancel_task_callback, cancellation_relay_task
from .._events import RunEventStream
from .._hooks import RunnerHooks
from .._json_file_state_store import JsonFileStateStore
from .._lifecycle import AgentDescriptor
from .._run import RunInput, RunResult, RunState, copy_run_state
from .._runner import Runner
from .._snapshot import AgentSnapshot
from .._state_store import StateStore
from .._stream import RunStream
from .._stream_base import close_stream_callback
from ._base import AgentBase
from .definitions import (
    AgentFileError,
    ModelResolver,
    SubagentLink,
    descriptor_from_markdown,
    with_subagents,
)

if TYPE_CHECKING:
    from ..tools import ToolApprovalEvent


class DefaultAgent(AgentBase):
    """Default high-level stateful agent interface.

    This agent owns higher-level orchestration concerns:

    1. descriptor resolution,
    2. optional automatic runtime provisioning,
    3. optional automatic runner provisioning, and
    4. persisted ``RunState`` across repeated ``run(...)`` calls.

    It does not replace the runtime-facing harness ``Agent``. Each execution
    still binds and routes through the existing runtime model so the lower-level
    behavior stays canonical. The implementation adds a stable primary conversation
    line plus explicit forked branch runs on top of that lower-level contract.
    """

    descriptor: AgentDescriptor | None = None

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor | None = None,
        subagents: Sequence[AgentDescriptor] = (),
        subagent_link: SubagentLink = SubagentLink.AS_TOOL,
        runtime: RuntimeEngine | None = None,
        runner: Runner | None = None,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        agent_id: AgentId | None = None,
        run_state: RunState | None = None,
        snapshot: AgentSnapshot | None = None,
        state_path: str | Path | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        """Initialize one stateful default agent.

        Args:
            descriptor: Optional instance-level descriptor override. When
                omitted, the agent uses ``type(self).descriptor``.
            subagents: Agent descriptors to attach as sub-agents. ``DefaultAgent``
                wires each one in (no manual ``as_tool()`` needed) per
                ``subagent_link``. Pass a ``DefaultAgent``'s ``resolved_descriptor``
                to reuse a built agent; use ``from_markdown(subagents=...)`` to
                load sub-agents from markdown files. ``tools=[child.as_tool()]``
                remains an alternative for fine-grained control.
            subagent_link: How sub-agents attach — ``AS_TOOL`` (subroutine,
                default) or ``HANDOFF`` (control transfer).
            runtime: Optional runtime to reuse across runs.
            runner: Optional runner to reuse across runs.
            hooks: Optional runner hook or ordered hook list forwarded to the
                low-level agent lifecycle callbacks.
            agent_id: Optional stable runtime id override.
            run_state: Optional initial resumable state.
            snapshot: Optional durable snapshot to restore. This cannot be
                combined with `run_state`; an explicit `agent_id` must match
                the snapshot address.
            state_path: Optional JSON file that is loaded at construction and
                atomically updated after each successful primary run. This
                cannot be combined with `run_state` or `snapshot`.
            state_store: Optional custom snapshot store. This cannot be
                combined with `state_path`, `run_state`, or `snapshot`.
        """
        if run_state is not None and snapshot is not None:
            raise ValueError("Pass either run_state or snapshot, not both.")
        if state_path is not None and state_store is not None:
            raise ValueError("Pass either state_path or state_store, not both.")
        if (state_path is not None or state_store is not None) and (
            run_state is not None or snapshot is not None
        ):
            raise ValueError(
                "Pass state_path or state_store without run_state or snapshot."
            )

        self._state_store = (
            JsonFileStateStore(state_path) if state_path is not None else state_store
        )
        restored_snapshot = (
            self._state_store.load() if self._state_store is not None else snapshot
        )
        if restored_snapshot is not None and agent_id not in (
            None,
            restored_snapshot.agent_id,
        ):
            raise ValueError(
                f"Agent id {agent_id} does not match snapshot id "
                f"{restored_snapshot.agent_id}."
            )

        self._descriptor = with_subagents(
            _resolve_descriptor(
                descriptor=descriptor,
                class_descriptor=type(self).descriptor,
            ),
            _coerce_subagents(subagents),
            link=subagent_link,
        )
        self._runtime = runtime
        self._runner = runner
        self._hooks = hooks
        self._agent_id = (
            restored_snapshot.agent_id
            if restored_snapshot is not None
            else agent_id or _default_agent_id(self._descriptor)
        )
        self._run_state = (
            restored_snapshot.to_run_state()
            if restored_snapshot is not None
            else copy_run_state(run_state)
        )

        # The agent persists one resumable state value and one stable runtime
        # identity locally. Concurrent ``run(...)`` calls on the same agent
        # therefore cannot safely overlap:
        #
        # 1. both calls would otherwise fork from the same baseline
        #    ``RunState`` and race the final commit, losing one branch, and
        # 2. both calls would bind the same runtime ``AgentId`` while one
        #    logical conversation is meant to continue in order.
        #
        # The full-run lock is intentional for one stateful agent instance.
        self._run_lock = asyncio.Lock()
        self._primary_stream_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    def from_markdown(
        cls,
        path: str | Path,
        *,
        model_resolver: ModelResolver | None = None,
        model: Model[ModelResponse] | None = None,
        subagent_link: SubagentLink = SubagentLink.AS_TOOL,
        subagents: Sequence[AgentDescriptor | str | Path] = (),
        runtime: RuntimeEngine | None = None,
        runner: Runner | None = None,
        hooks: RunnerHooks | Sequence[RunnerHooks] | None = None,
        agent_id: AgentId | None = None,
        run_state: RunState | None = None,
        state_path: str | Path | None = None,
        state_store: StateStore | None = None,
    ) -> Self:
        """Build a runnable agent from a Claude-Code-style markdown file.

        Parses `path` into a descriptor (attaching any `subagents`) and
        constructs a `DefaultAgent`. The root agent must resolve to a model: it
        comes from the explicit `model` argument, or from the frontmatter `model`
        spec via `model_resolver`. If neither yields a model, this raises — an
        agent cannot run without one. Sub-agents may omit a model to inherit the
        parent's at runtime.

        Args:
            path: Path to the `AGENT.md` file.
            model_resolver: Optional resolver for the frontmatter `model` spec.
            model: Optional pre-built client; supplies or overrides the root model.
            subagent_link: How resolved sub-agents attach; defaults to as-tool.
            subagents: Child descriptors or paths to attach as sub-agents.
            runtime: Optional runtime to reuse across runs.
            runner: Optional runner to reuse across runs.
            hooks: Optional runner hook or ordered hook list.
            agent_id: Optional stable runtime id override.
            run_state: Optional initial resumable state.
            state_path: Optional JSON file loaded at construction and updated
                after each successful primary run.
            state_store: Optional custom snapshot store.

        Returns:
            DefaultAgent: A runnable agent bound to the parsed descriptor.

        Raises:
            FileNotFoundError: When `path` or a sub-agent path does not exist.
            AgentFileError: When a file is unparseable, sub-agent nesting is too
                deep or cyclic, or the root agent resolves to no model.
        """
        descriptor = descriptor_from_markdown(
            path,
            model_resolver=model_resolver,
            subagent_link=subagent_link,
            subagents=subagents,
        )

        if model is not None:
            descriptor = replace(descriptor, model=model)

        if descriptor.model is None:
            raise AgentFileError(
                f"agent file `{path}` does not resolve to a model; a top-level "
                "agent cannot run without one. Declare a `model:` and pass "
                "`model_resolver=`, or pass `model=`."
            )
        return cls(
            descriptor=descriptor,
            runtime=runtime,
            runner=runner,
            hooks=hooks,
            agent_id=agent_id,
            run_state=run_state,
            state_path=state_path,
            state_store=state_store,
        )

    @property
    def resolved_descriptor(self) -> AgentDescriptor:
        """Return the resolved static descriptor for this agent."""
        return self._descriptor

    @property
    def agent_id(self) -> AgentId:
        """Return the stable runtime id used by this agent instance."""
        return self._agent_id

    @property
    def run_state(self) -> RunState | None:
        """Return a defensive copy of the latest persisted run state."""
        return copy_run_state(self._run_state)

    def snapshot(self) -> AgentSnapshot | None:
        """Return the latest committed state as a portable snapshot.

        An agent with no completed run has no state to snapshot. While a run is
        active, this returns the prior committed baseline rather than the
        private working state.
        """
        if self._run_state is None:
            return None
        return AgentSnapshot.capture(
            agent_id=self._agent_id,
            run_state=self._run_state,
        )

    async def run(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute one primary-line run and persist the resulting state.

        Args:
            input: Raw run input or an explicit ``RunState`` resume payload.
                When a ``RunState`` is provided directly, it takes precedence
                over the agent's stored baseline for that call.
            cancellation_token: Optional shared cancellation token.

        Returns:
            RunResult: Final result from the low-level harness run.
        """
        async with self._run_lock:
            effective_runner = self._resolved_runner()
            run_input, initial_state = self._prepare_primary_run(input)

            if self._runtime is None:
                async with single_threaded_runtime() as runtime:
                    result = await self._run_once(
                        runtime=runtime,
                        runner=effective_runner,
                        input=run_input,
                        initial_state=initial_state,
                        agent_id=self._agent_id,
                        cancellation_token=cancellation_token,
                    )
            else:
                async with runtime_scope(self._runtime) as runtime:
                    result = await self._run_once(
                        runtime=runtime,
                        runner=effective_runner,
                        input=run_input,
                        initial_state=initial_state,
                        agent_id=self._agent_id,
                        cancellation_token=cancellation_token,
                    )

            self._commit_run_state(result.run_state)
            return result

    async def fork(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run one branch without mutating the agent's persisted main state.

        This method snapshots the current persisted baseline, if any, runs the
        branch under a fresh runtime agent id, and returns the branch result
        without storing it back onto internal run state.

        Args:
            input: Raw run input or an explicit ``RunState`` resume payload.
                When a ``RunState`` is provided directly, it takes precedence
                over the agent's stored baseline for that call.
            cancellation_token: Optional shared cancellation token.

        Returns:
            RunResult: Final result for the forked branch run.
        """
        async with self._run_lock:
            # Wait for any active primary run to commit its latest baseline,
            # then capture one coherent snapshot for this branch. The lock is
            # released before the branch executes because forked runs do not
            # write back into the agent's primary conversation line.
            effective_runner = self._resolved_runner()
            initial_state = (
                None if isinstance(input, RunState) else copy_run_state(self._run_state)
            )

        fork_agent_id = _fork_agent_id(self._agent_id)

        if self._runtime is None:
            async with single_threaded_runtime() as runtime:
                return await self._run_once(
                    runtime=runtime,
                    runner=effective_runner,
                    input=input,
                    initial_state=initial_state,
                    agent_id=fork_agent_id,
                    cancellation_token=cancellation_token,
                )

        async with runtime_scope(self._runtime) as runtime:
            return await self._run_once(
                runtime=runtime,
                runner=effective_runner,
                input=input,
                initial_state=initial_state,
                agent_id=fork_agent_id,
                cancellation_token=cancellation_token,
            )

    def reset(self) -> None:
        """Clear the stored primary-line run state for future runs.

        When a state store is configured, this also removes its snapshot. It
        does not replace the resolved descriptor, stable ``agent_id``, configured
        runtime, runner, or hooks.

        Raises:
            RuntimeError: If a primary run is active.
        """
        if self._run_lock.locked() or self._primary_stream_tasks:
            raise RuntimeError("Cannot reset while a primary run is active.")

        if self._state_store is not None:
            expected_revision = (
                self._run_state.revision if self._run_state is not None else None
            )
            self._state_store.delete(expected_revision=expected_revision)
        self._run_state = None

    async def run_stream(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunStream:
        """Execute one primary-line run with live model streaming."""
        stream_token = CancellationToken()
        relay_task = cancellation_relay_task(
            source=cancellation_token,
            target=stream_token,
        )
        stream = RunStream(on_close=stream_token.cancel)
        if relay_task is not None:
            stream.add_cleanup(cancel_task_callback(relay_task))

        stream_task = asyncio.create_task(
            self._run_stream_task(
                input=input,
                stream=stream,
                cancellation_token=stream_token,
            )
        )
        self._primary_stream_tasks.add(stream_task)
        stream_task.add_done_callback(self._primary_stream_tasks.discard)
        stream.add_cleanup(cancel_task_callback(stream_task))
        return stream

    async def run_events(
        self,
        input: RunInput,
        *,
        approval_events: AsyncIterator["ToolApprovalEvent"] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunEventStream:
        """Execute one primary-line run with high-level run events."""
        stream_token = CancellationToken()
        relay_task = cancellation_relay_task(
            source=cancellation_token,
            target=stream_token,
        )
        stream = RunEventStream(on_close=stream_token.cancel)
        if relay_task is not None:
            stream.add_cleanup(cancel_task_callback(relay_task))

        stream_task = asyncio.create_task(
            self._run_events_task(
                input=input,
                stream=stream,
                approval_events=approval_events,
                cancellation_token=stream_token,
            )
        )
        self._primary_stream_tasks.add(stream_task)
        stream_task.add_done_callback(self._primary_stream_tasks.discard)
        stream.add_cleanup(cancel_task_callback(stream_task))
        return stream

    def _resolved_runner(self) -> Runner:
        """Return the configured runner, provisioning one lazily if needed."""
        if self._runner is None:
            self._runner = Runner()
        return self._runner

    async def _run_once(
        self,
        *,
        runtime: RuntimeEngine,
        runner: Runner,
        input: RunInput,
        initial_state: RunState | None,
        agent_id: AgentId,
        cancellation_token: CancellationToken | None,
    ) -> RunResult:
        """Bind the low-level harness agent, route one input, and unwrap result."""
        runtime_agent = RuntimeAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=self._descriptor,
            run_state=initial_state,
            hooks=self._hooks,
        )
        outcome = await runtime.send_message(
            input,
            recipient=agent_id,
            cancellation_token=cancellation_token,
        )
        result = _require_run_result(outcome)
        if result.run_state is not None:
            return result

        # Custom runners may omit ``run_state`` on the returned result even
        # though the low-level lifecycle still persisted the completed state.
        return RunResult(
            final_output=result.final_output,
            responses=list(result.responses),
            turn_count=result.turn_count,
            run_state=runtime_agent.run_state,
        )

    async def _run_stream_task(
        self,
        *,
        input: RunInput,
        stream: RunStream,
        cancellation_token: CancellationToken,
    ) -> None:
        """Drive one high-level streamed run and commit final state on success."""
        try:
            async with self._run_lock:
                effective_runner = self._resolved_runner()
                run_input, initial_state = self._prepare_primary_run(input)

                if self._runtime is None:
                    async with single_threaded_runtime() as runtime:
                        result = await self._run_stream_once(
                            runtime=runtime,
                            runner=effective_runner,
                            input=run_input,
                            initial_state=initial_state,
                            agent_id=self._agent_id,
                            stream=stream,
                            cancellation_token=cancellation_token,
                        )
                else:
                    async with runtime_scope(self._runtime) as runtime:
                        result = await self._run_stream_once(
                            runtime=runtime,
                            runner=effective_runner,
                            input=run_input,
                            initial_state=initial_state,
                            agent_id=self._agent_id,
                            stream=stream,
                            cancellation_token=cancellation_token,
                        )

                self._commit_run_state(result.run_state)
        except Exception as exc:
            stream.fail(exc)
        except BaseException as exc:
            stream.fail(exc)
            raise
        else:
            stream.finish(result)

    async def _run_events_task(
        self,
        *,
        input: RunInput,
        stream: RunEventStream,
        approval_events: AsyncIterator["ToolApprovalEvent"] | None,
        cancellation_token: CancellationToken,
    ) -> None:
        """Drive one high-level event run and commit final state on success."""
        try:
            async with self._run_lock:
                effective_runner = self._resolved_runner()
                run_input, initial_state = self._prepare_primary_run(input)

                if self._runtime is None:
                    async with single_threaded_runtime() as runtime:
                        result = await self._run_events_once(
                            runtime=runtime,
                            runner=effective_runner,
                            input=run_input,
                            initial_state=initial_state,
                            agent_id=self._agent_id,
                            stream=stream,
                            approval_events=approval_events,
                            cancellation_token=cancellation_token,
                        )
                else:
                    async with runtime_scope(self._runtime) as runtime:
                        result = await self._run_events_once(
                            runtime=runtime,
                            runner=effective_runner,
                            input=run_input,
                            initial_state=initial_state,
                            agent_id=self._agent_id,
                            stream=stream,
                            approval_events=approval_events,
                            cancellation_token=cancellation_token,
                        )

                self._commit_run_state(result.run_state)
        except Exception as exc:
            stream.fail(exc)
        except BaseException as exc:
            stream.fail(exc)
            raise
        else:
            stream.finish(result)

    async def _run_stream_once(
        self,
        *,
        runtime: RuntimeEngine,
        runner: Runner,
        input: RunInput,
        initial_state: RunState | None,
        agent_id: AgentId,
        stream: RunStream,
        cancellation_token: CancellationToken,
    ) -> RunResult:
        """Bind the low-level harness agent and stream one input locally."""
        runtime_agent = RuntimeAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=self._descriptor,
            run_state=initial_state,
            hooks=self._hooks,
        )
        low_level_stream = await runtime_agent.enqueue_input_stream(
            input,
            cancellation_token=cancellation_token,
        )
        stream.add_cleanup(close_stream_callback(low_level_stream))

        try:
            async for event in low_level_stream:
                stream.emit(event)

            result = await low_level_stream.result()
        finally:
            await low_level_stream.aclose()

        if result.run_state is not None:
            return result

        return RunResult(
            final_output=result.final_output,
            responses=list(result.responses),
            turn_count=result.turn_count,
            run_state=runtime_agent.run_state,
        )

    def _commit_run_state(self, run_state: RunState | None) -> None:
        """Keep and, when configured, durably save one committed state."""
        committed_state = copy_run_state(run_state)
        if self._state_store is None or committed_state is None:
            self._run_state = committed_state
            return

        snapshot = AgentSnapshot.capture(
            agent_id=self._agent_id,
            run_state=committed_state,
        )
        expected_revision = (
            self._run_state.revision if self._run_state is not None else None
        )
        self._state_store.save(
            snapshot,
            expected_revision=expected_revision,
        )
        self._run_state = committed_state

    def _prepare_primary_run(
        self,
        run_input: RunInput,
    ) -> tuple[RunInput, RunState | None]:
        """Prepare one primary input against the durable revision baseline."""
        if not isinstance(run_input, RunState):
            return run_input, self._run_state
        if self._state_store is None or self._run_state is None:
            return run_input, None

        resumed_state = copy_run_state(run_input)
        if resumed_state is None:
            raise AssertionError("RunState copy unexpectedly returned None.")
        resumed_state.revision = self._run_state.revision
        return resumed_state, None

    async def _run_events_once(
        self,
        *,
        runtime: RuntimeEngine,
        runner: Runner,
        input: RunInput,
        initial_state: RunState | None,
        agent_id: AgentId,
        stream: RunEventStream,
        approval_events: AsyncIterator["ToolApprovalEvent"] | None,
        cancellation_token: CancellationToken,
    ) -> RunResult:
        """Bind the low-level harness agent and stream high-level run events."""
        runtime_agent = RuntimeAgent.bind(
            runtime,
            agent_id,
            runner=runner,
            descriptor=self._descriptor,
            run_state=initial_state,
            hooks=self._hooks,
        )
        low_level_stream = await runtime_agent.enqueue_input_events(
            input,
            approval_events=approval_events,
            cancellation_token=cancellation_token,
        )
        stream.add_cleanup(close_stream_callback(low_level_stream))

        try:
            async for event in low_level_stream:
                stream.emit(event)

            result = await low_level_stream.result()
        finally:
            await low_level_stream.aclose()

        if result.run_state is not None:
            return result

        return RunResult(
            final_output=result.final_output,
            responses=list(result.responses),
            turn_count=result.turn_count,
            run_state=runtime_agent.run_state,
        )


def _coerce_subagents(
    subagents: Sequence[object],
) -> tuple[AgentDescriptor, ...]:
    """Validate that programmatic sub-agents are agent descriptors.

    Typed against `object` because this is the runtime guard for the public
    `subagents` argument: it gives callers without a type checker a clear error
    instead of an opaque failure deeper in attachment.
    """
    children: list[AgentDescriptor] = []
    for item in subagents:
        if not isinstance(item, AgentDescriptor):
            raise TypeError(
                "DefaultAgent(subagents=...) accepts AgentDescriptor values. "
                "Pass a DefaultAgent's `resolved_descriptor`, or load sub-agents "
                "from markdown files with DefaultAgent.from_markdown(subagents=[...])."
            )

        children.append(item)

    return tuple(children)


def _resolve_descriptor(
    *,
    descriptor: AgentDescriptor | None,
    class_descriptor: AgentDescriptor | None,
) -> AgentDescriptor:
    """Resolve the agent descriptor from instance or class configuration."""
    if descriptor is not None:
        return descriptor

    if class_descriptor is not None:
        return class_descriptor

    raise ValueError(
        "DefaultAgent requires an `AgentDescriptor`, either via `descriptor=` "
        "or a class-level `descriptor` attribute."
    )


def _default_agent_id(descriptor: AgentDescriptor) -> AgentId:
    """Create one stable local runtime id for an agent instance."""
    return AgentId.from_values(descriptor.name, uuid4().hex)


def _fork_agent_id(agent_id: AgentId) -> AgentId:
    """Create a fresh runtime id for one forked branch run."""
    return AgentId.from_values(
        agent_id.type.value,
        f"{agent_id.key.value}-fork-{uuid4().hex}",
    )


def _require_run_result(outcome: DeliveryOutcome) -> RunResult:
    """Return the delivered run result or raise a useful runtime error."""
    if outcome.status != DeliveryStatus.DELIVERED:
        if outcome.error is None:
            detail = "missing runtime error details"
        else:
            detail = outcome.error.message
        raise RuntimeError(
            "DefaultAgent run failed with delivery status "
            f"`{outcome.status.value}`: {detail}"
        )

    if not isinstance(outcome.response_payload, RunResult):
        raise TypeError(
            "Expected the harness runtime delivery to return a `RunResult` "
            "response payload."
        )
    return outcome.response_payload
