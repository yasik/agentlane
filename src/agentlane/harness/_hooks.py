"""Runner hook contracts for harness lifecycle events."""

from agentlane.models import MessageDict, ModelResponse, ToolCall

from ._run import RunResult, RunState
from ._task import Task


class RunnerHooks:
    """Default no-op hook surface for harness runner events."""

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        """Observe the start of one agent run.

        Args:
            task: Harness task or agent being executed.
            state: Current run state before the loop executes.
        """
        _ = task
        _ = state

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        """Observe the end of one agent run.

        Args:
            task: Harness task or agent being executed.
            result: Final run result, if the run completed successfully.
        """
        _ = task
        _ = result

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        """Observe the start of one LLM request.

        Args:
            task: Harness task or agent being executed.
            messages: Canonical conversation input passed to the model.
        """
        _ = task
        _ = messages

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        """Observe the end of one LLM request.

        Args:
            task: Harness task or agent being executed.
            response: Canonical OpenAI-shaped model response.
        """
        _ = task
        _ = response

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        """Observe the start of one tool invocation.

        Args:
            task: Harness task or agent being executed.
            tool_call: Canonical OpenAI-shaped tool call.
        """
        _ = task
        _ = tool_call

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        """Observe the end of one tool invocation.

        Args:
            task: Harness task or agent being executed.
            tool_call: Canonical OpenAI-shaped tool call.
            result: Tool result before any later packaging.
        """
        _ = task
        _ = tool_call
        _ = result
