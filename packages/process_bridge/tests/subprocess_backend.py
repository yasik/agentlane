"""Scripted runtime for process-boundary tests without provider credentials.

The stream, approval broker, backend, command loop, and pipes are real. Only the
model and tool work are scripted so tests can select exact lifecycle failures.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

from agentlane_process_bridge import AgentBackend

from agentlane.harness import (
    RunAgentEndEvent,
    RunAgentStartEvent,
    RunEventStream,
    RunModelStreamEvent,
    RunResult,
    RunState,
    RunToolApprovalEvent,
    RunToolEndEvent,
    RunToolStartEvent,
)
from agentlane.harness.tools import (
    ToolApprovalBroker,
    ToolApprovalEvent,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionOutcome,
    ToolPermissionRequest,
)
from agentlane.models import ModelStreamEvent, ModelStreamEventKind, ToolCall
from agentlane.runtime import CancellationToken


class SubprocessAgent:
    """Produce native events and await real host approval commands."""

    def __init__(self, approvals: ToolApprovalBroker) -> None:
        self.approvals = approvals
        self.run_state: RunState | None = None

    def reset(self) -> None:
        self.run_state = None

    async def run_events(
        self,
        input: str,
        /,
        *,
        approval_events: AsyncIterator[ToolApprovalEvent],
        cancellation_token: CancellationToken | None = None,
    ) -> RunEventStream:
        stream = RunEventStream(
            on_close=None if cancellation_token is None else cancellation_token.cancel
        )

        async def forward_approvals() -> None:
            async for event in approval_events:
                stream.emit(RunToolApprovalEvent(event=event))

        forwarding = asyncio.create_task(forward_approvals())

        def stop_forwarding() -> None:
            forwarding.cancel()

        stream.add_cleanup(stop_forwarding)
        # Register the broker subscriber before the producer requests approval.
        await asyncio.sleep(0)
        producer = asyncio.create_task(self._produce(input, stream))

        def stop_producer() -> None:
            producer.cancel()

        stream.add_cleanup(stop_producer)
        return stream

    async def _produce(self, prompt: str, stream: RunEventStream) -> None:
        try:
            if prompt == "disconnect":
                os._exit(7)

            stream.emit(RunAgentStartEvent(task_name="Scripted", task_id="root"))
            for event in (
                ModelStreamEvent(kind=ModelStreamEventKind.TEXT_DELTA, text=""),
                ModelStreamEvent(
                    kind=ModelStreamEventKind.PROVIDER,
                    raw={"provider_extra": {"unicode": "界🧪\n"}},
                ),
                ModelStreamEvent(kind=ModelStreamEventKind.COMPLETED),
                ModelStreamEvent(
                    kind=ModelStreamEventKind.ERROR,
                    error=RuntimeError("recoverable model attempt"),
                ),
            ):
                stream.emit(RunModelStreamEvent(event=event))

            if prompt == "bad-native":
                stream.emit(
                    RunModelStreamEvent(
                        event=ModelStreamEvent(
                            kind=ModelStreamEventKind.PROVIDER, raw=object()
                        )
                    )
                )
                await asyncio.Event().wait()

            if prompt == "wait":
                stream.emit(
                    RunModelStreamEvent(
                        event=ModelStreamEvent(
                            kind=ModelStreamEventKind.TEXT_DELTA, text="waiting"
                        )
                    )
                )
                await asyncio.Event().wait()

            call = ToolCall.model_validate(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "write", "arguments": '{"text":"界🧪"}'},
                }
            )
            stream.emit(
                RunToolStartEvent(task_name="Scripted", task_id="root", tool_call=call)
            )
            decision = await self.approvals.callback(
                ToolPermissionRequest(
                    tool_name="write",
                    operation=ToolOperation.CREATE_FILE,
                    cwd=Path("/synthetic"),
                    path=Path("/synthetic/output.txt"),
                    tool_call_id=call.id,
                    metadata={"synthetic": True},
                ),
                ToolPermissionDecision.require_approval("Review the synthetic write."),
            )
            # Forward the resolved record before stream termination stops its task.
            await asyncio.sleep(0)
            allowed = decision.outcome == ToolPermissionOutcome.ALLOW
            output = {"allowed": allowed, "text": "界🧪\n" * 2000, "empty": None}
            stream.emit(
                RunToolEndEvent(
                    task_name="Scripted",
                    task_id="root",
                    tool_call=call,
                    result=output,
                    ok=allowed,
                )
            )
            stream.emit(
                RunModelStreamEvent(
                    event=ModelStreamEvent(
                        kind=ModelStreamEventKind.TEXT_DELTA, text="continued"
                    )
                )
            )
            result = RunResult(final_output=output, responses=[], turn_count=2)
            stream.emit(
                RunAgentEndEvent(task_name="Scripted", task_id="root", result=result)
            )
            if prompt == "late-error":
                stream.fail(RuntimeError("authoritative result failed after agent_end"))
                return

            if prompt == "bad-result":
                result = RunResult(final_output=object(), responses=[], turn_count=2)

            stream.finish(result)
        except asyncio.CancelledError:
            await stream.aclose()
            raise
        except Exception as exc:
            stream.fail(exc)


def create_backend() -> AgentBackend:
    """Share one approval broker between the runtime and bridge commands."""
    approvals = ToolApprovalBroker()
    return AgentBackend(agent=SubprocessAgent(approvals), approvals=approvals)
