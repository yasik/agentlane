"""Lifecycle helpers for harness agent execution."""

import asyncio
from asyncio import Future, get_running_loop
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentlane.models import MessageDict
from agentlane.models import Tool as HarnessTool
from agentlane.models import create_system_message, create_user_message
from agentlane.runtime import CancellationToken

from ._hooks import RunnerHooks
from ._runner import Runner
from ._task import Task


@dataclass(slots=True)
class AgentDescriptor:
    """Static descriptive properties for one harness agent instance."""

    name: str
    """Human-readable agent name."""

    description: str | None
    """Short description of the agent responsibility."""

    system_prompt: str | None
    """Optional system prompt used to seed new conversations."""

    tools: tuple[HarnessTool[Any, Any], ...] | None
    """Tools visible to the agent in later phases."""

    skills: tuple[object, ...] | None
    """Skills associated with the agent in later phases."""

    context: object | None
    """Opaque context reference reserved for later phases."""

    memory: object | None
    """Opaque memory reference reserved for later phases."""


@dataclass(slots=True)
class _QueuedUserTurn:
    """One pending inbound user turn waiting for runner execution."""

    message: MessageDict
    """Canonical user-role message to append to the conversation."""

    future: Future[object]
    """Future resolved with the packaged runner result for this turn."""

    cancellation_token: CancellationToken | None
    """Optional cancellation token for the runner turn."""


class AgentLifecycle:
    """Owns per-agent conversation history and next-turn message queueing."""

    def __init__(
        self,
        *,
        descriptor: AgentDescriptor,
    ) -> None:
        """Initialize empty lifecycle state for one agent instance."""
        self._descriptor = descriptor
        self._history: list[MessageDict] = []
        self._pending_turns: deque[_QueuedUserTurn] = deque()
        self._is_running = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the lifecycle is currently executing a runner turn."""
        return self._is_running

    @property
    def pending_message_count(self) -> int:
        """Return the number of queued user turns not yet started."""
        return len(self._pending_turns)

    def history_snapshot(self) -> list[MessageDict]:
        """Return a shallow copy of the current conversation history."""
        return [dict(message) for message in self._history]

    async def enqueue_user_message(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
        content: object,
        cancellation_token: CancellationToken | None = None,
    ) -> object:
        """Queue one user message and return the packaged turn result."""
        queued_turn = _QueuedUserTurn(
            message=create_user_message(content),
            future=get_running_loop().create_future(),
            cancellation_token=cancellation_token,
        )
        should_drain = False

        async with self._lock:
            _initialize_conversation_if_needed(
                history=self._history,
                system_prompt=self._descriptor.system_prompt,
            )
            self._pending_turns.append(queued_turn)
            if not self._is_running:
                # Only the first enqueuer becomes the active drainer. Later
                # arrivals simply append turns and wait for their own futures.
                self._is_running = True
                should_drain = True

        if should_drain:
            await self._drain_pending_turns(
                agent=agent,
                runner=runner,
                hooks=hooks,
            )

        return await queued_turn.future

    async def _drain_pending_turns(
        self,
        *,
        agent: Task,
        runner: Runner,
        hooks: RunnerHooks | None,
    ) -> None:
        """Drain queued turns sequentially under the current runtime guarantee.

        The lifecycle intentionally processes one queued user turn per runner
        invocation. It does not batch-append all outstanding queued messages
        into one larger turn before calling the runner again.
        """
        active_turn: _QueuedUserTurn | None = None
        try:
            while True:
                async with self._lock:
                    if not self._pending_turns:
                        self._is_running = False
                        return
                    # Intentionally drain one queued user turn per runner call.
                    active_turn = self._pending_turns.popleft()
                    self._history.append(active_turn.message)

                try:
                    # The runner receives the shared mutable history so later
                    # phases can append assistant/tool outputs in place.
                    results = await runner.run(
                        agent=agent,
                        messages=self._history,
                        hooks=hooks,
                        cancellation_token=active_turn.cancellation_token,
                    )
                except Exception as exc:  # noqa: BLE001
                    _set_future_exception(active_turn.future, exc)
                else:
                    _set_future_result(
                        active_turn.future,
                        package_runner_results(results),
                    )
                finally:
                    active_turn = None
        except BaseException as exc:
            if active_turn is not None:
                _set_future_exception(active_turn.future, exc)
            async with self._lock:
                pending_turns = list(self._pending_turns)
                self._pending_turns.clear()
                self._is_running = False
            for queued_turn in pending_turns:
                _set_future_exception(queued_turn.future, exc)
            raise


def package_runner_results(results: Sequence[object]) -> object:
    """Package runner results into the agent response shape for this phase."""
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    return list(results)


def _initialize_conversation_if_needed(
    *,
    history: list[MessageDict],
    system_prompt: str | None,
) -> None:
    """Seed a new conversation with the system prompt exactly once."""
    if history or system_prompt is None:
        return
    # System prompt seeding happens lazily on the first inbound user turn.
    history.append(create_system_message(system_prompt))


def _set_future_result(future: Future[object], value: object) -> None:
    """Resolve a future once when it is still pending."""
    if not future.done():
        future.set_result(value)


def _set_future_exception(future: Future[object], exc: BaseException) -> None:
    """Fail a future once when it is still pending."""
    if not future.done():
        future.set_exception(exc)
