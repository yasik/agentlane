import asyncio
from collections.abc import AsyncIterator

from agentlane.harness import (
    RunAgentEndEvent,
    RunAgentStartEvent,
    RunEventStream,
    RunModelStreamEvent,
    RunResult,
    RunState,
)
from agentlane.harness.tools import ToolApprovalEvent
from agentlane.models import ModelStreamEvent, ModelStreamEventKind
from agentlane.runtime import CancellationToken
from agentlane_process_bridge import run_stdio


class ScriptedAgent:
    """Minimal AgentRuntime used by the stdio bridge example."""

    def __init__(self) -> None:
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
        del approval_events
        stream = RunEventStream(
            on_close=None if cancellation_token is None else cancellation_token.cancel
        )
        task = asyncio.create_task(self._produce(input, stream))

        def cancel_task() -> None:
            task.cancel()

        stream.add_cleanup(cancel_task)
        return stream

    async def _produce(self, prompt: str, stream: RunEventStream) -> None:
        try:
            task_id = "scripted-task"
            stream.emit(RunAgentStartEvent(task_name="Scripted", task_id=task_id))
            await asyncio.sleep(0)
            stream.emit(
                RunModelStreamEvent(
                    event=ModelStreamEvent(
                        kind=ModelStreamEventKind.TEXT_DELTA,
                        text=f"Echo: {prompt}",
                    )
                )
            )
            result = RunResult(
                final_output=f"Echo: {prompt}",
                responses=[],
                turn_count=1,
            )
            stream.emit(
                RunAgentEndEvent(
                    task_name="Scripted",
                    task_id=task_id,
                    result=result,
                )
            )
            stream.finish(result)
        except asyncio.CancelledError:
            await stream.aclose()
            raise
        except Exception as exc:
            stream.fail(exc)


async def main() -> None:
    await run_stdio(
        agent=ScriptedAgent(),
        ready_metadata=lambda: {"example": "process_bridge_stdio"},
    )


if __name__ == "__main__":
    asyncio.run(main())
