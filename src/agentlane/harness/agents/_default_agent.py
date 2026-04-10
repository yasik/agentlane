"""Default high-level wrapper for stateful harness agent execution.

This module provides the standard wrapper that layers persisted run state,
optional runtime and runner provisioning, and branch execution on top of the
runtime-facing harness ``Agent``.
"""

import asyncio
from uuid import uuid4

from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus
from agentlane.runtime import (
    CancellationToken,
    RuntimeEngine,
    runtime_scope,
    single_threaded_runtime,
)

from .._agent import Agent as RuntimeAgent
from .._hooks import RunnerHooks
from .._lifecycle import AgentDescriptor
from .._run import RunInput, RunResult, RunState, copy_run_state
from .._runner import Runner
from ._base import AgentBase


class DefaultAgent(AgentBase):
    """High-level stateful wrapper around the runtime-facing harness agent.

    This wrapper owns higher-level orchestration concerns:

    1. descriptor resolution,
    2. optional automatic runtime provisioning,
    3. optional automatic runner provisioning, and
    4. persisted ``RunState`` across repeated ``run(...)`` calls.

    It does not replace the runtime-facing harness ``Agent``. Each execution
    still binds and routes through the existing runtime model so the lower-level
    behavior stays canonical. The wrapper adds a stable primary conversation
    line plus explicit forked branch runs on top of that lower-level contract.
    """

    descriptor: AgentDescriptor | None = None

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor | None = None,
        runtime: RuntimeEngine | None = None,
        runner: Runner | None = None,
        hooks: RunnerHooks | None = None,
        agent_id: AgentId | None = None,
        run_state: RunState | None = None,
    ) -> None:
        """Initialize one stateful default agent wrapper.

        Args:
            descriptor: Optional instance-level descriptor override. When
                omitted, the wrapper uses ``type(self).descriptor``.
            runtime: Optional runtime to reuse across runs.
            runner: Optional runner to reuse across runs.
            hooks: Optional runner hooks forwarded to the low-level agent.
            agent_id: Optional stable runtime id override.
            run_state: Optional initial resumable state.
        """
        self._descriptor = _resolve_descriptor(
            descriptor=descriptor,
            class_descriptor=type(self).descriptor,
        )
        self._runtime = runtime
        self._runner = runner
        self._hooks = hooks
        self._agent_id = agent_id or _default_agent_id(self._descriptor)
        self._run_state = copy_run_state(run_state)

        # The wrapper persists one resumable state value and one stable runtime
        # identity locally. Concurrent ``run(...)`` calls on the same wrapper
        # therefore cannot safely overlap:
        #
        # 1. both calls would otherwise fork from the same baseline
        #    ``RunState`` and race the final commit, losing one branch, and
        # 2. both calls would bind the same runtime ``AgentId`` while one
        #    logical conversation is meant to continue in order.
        #
        # The full-run lock is intentional for one stateful wrapper instance.
        self._run_lock = asyncio.Lock()

    @property
    def resolved_descriptor(self) -> AgentDescriptor:
        """Return the resolved static descriptor for this wrapper."""
        return self._descriptor

    @property
    def agent_id(self) -> AgentId:
        """Return the stable runtime id used by this wrapper instance."""
        return self._agent_id

    @property
    def run_state(self) -> RunState | None:
        """Return a defensive copy of the latest persisted run state."""
        return copy_run_state(self._run_state)

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
                over the wrapper's stored baseline for that call.
            cancellation_token: Optional shared cancellation token.

        Returns:
            RunResult: Final result from the low-level harness run.
        """
        async with self._run_lock:
            effective_runner = self._resolved_runner()
            initial_state = None if isinstance(input, RunState) else self._run_state

            if self._runtime is None:
                async with single_threaded_runtime() as runtime:
                    result = await self._run_once(
                        runtime=runtime,
                        runner=effective_runner,
                        input=input,
                        initial_state=initial_state,
                        agent_id=self._agent_id,
                        cancellation_token=cancellation_token,
                    )
            else:
                async with runtime_scope(self._runtime) as runtime:
                    result = await self._run_once(
                        runtime=runtime,
                        runner=effective_runner,
                        input=input,
                        initial_state=initial_state,
                        agent_id=self._agent_id,
                        cancellation_token=cancellation_token,
                    )

            self._run_state = copy_run_state(result.run_state)
            return result

    async def fork(
        self,
        input: RunInput,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Run one branch without mutating the wrapper's persisted state.

        This method snapshots the current persisted baseline, if any, runs the
        branch under a fresh runtime agent id, and returns the branch result
        without storing it back onto internal run state.

        Args:
            input: Raw run input or an explicit ``RunState`` resume payload.
                When a ``RunState`` is provided directly, it takes precedence
                over the wrapper's stored baseline for that call.
            cancellation_token: Optional shared cancellation token.

        Returns:
            RunResult: Final result for the forked branch run.
        """
        async with self._run_lock:
            # Wait for any active primary run to commit its latest baseline,
            # then capture one coherent snapshot for this branch. The lock is
            # released before the branch executes because forked runs do not
            # write back into the wrapper's primary conversation line.
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

        This resets only the wrapper's persisted ``RunState`` baseline. It does
        not replace the resolved descriptor, stable ``agent_id``, configured
        runtime, runner, or hooks.
        """
        self._run_state = None

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


def _resolve_descriptor(
    *,
    descriptor: AgentDescriptor | None,
    class_descriptor: AgentDescriptor | None,
) -> AgentDescriptor:
    """Resolve the wrapper descriptor from instance or class configuration."""
    if descriptor is not None:
        return descriptor
    if class_descriptor is not None:
        return class_descriptor
    raise ValueError(
        "DefaultAgent requires an `AgentDescriptor`, either via `descriptor=` "
        "or a class-level `descriptor` attribute."
    )


def _default_agent_id(descriptor: AgentDescriptor) -> AgentId:
    """Create one stable local runtime id for a wrapper instance."""
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
