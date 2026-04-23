"""Runner hook contracts for harness lifecycle callbacks."""

from collections.abc import Sequence
from typing import cast

from agentlane.models import MessageDict, ModelResponse, ToolCall

from ._run import RunResult, RunState
from ._task import Task


class RunnerHooks:
    """Default no-op hook surface for harness runner lifecycle callbacks."""

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        """Handle the start of one agent run.

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
        """Handle the end of one agent run.

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
        """Handle the start of one LLM request.

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
        """Handle the end of one LLM request.

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
        """Handle the start of one tool invocation.

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
        """Handle the end of one tool invocation.

        Args:
            task: Harness task or agent being executed.
            tool_call: Canonical OpenAI-shaped tool call.
            result: Tool result before any later packaging.
        """
        _ = task
        _ = tool_call
        _ = result


class _MergedRunnerHooks(RunnerHooks):
    """Private ordered fan-out wrapper around multiple runner hook instances."""

    def __init__(self, hooks: Sequence[RunnerHooks]) -> None:
        self._hooks = tuple(hooks)

    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        for hook in self._hooks:
            await hook.on_agent_start(task, state)

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        for hook in self._hooks:
            await hook.on_agent_end(task, result)

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_start(task, messages)

    async def on_llm_end(
        self,
        task: Task,
        response: ModelResponse,
    ) -> None:
        for hook in self._hooks:
            await hook.on_llm_end(task, response)

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        for hook in self._hooks:
            await hook.on_tool_call_start(task, tool_call)

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        for hook in self._hooks:
            await hook.on_tool_call_end(task, tool_call, result)


def coerce_runner_hooks(
    *hooks: RunnerHooks | Sequence[RunnerHooks] | None,
) -> RunnerHooks:
    """Return one normalized hook implementation for the provided inputs."""
    flattened_hooks = _flatten_runner_hooks(*hooks)
    if len(flattened_hooks) == 0:
        return RunnerHooks()
    if len(flattened_hooks) == 1:
        return flattened_hooks[0]
    return _MergedRunnerHooks(flattened_hooks)


def _flatten_runner_hooks(
    *hooks: RunnerHooks | Sequence[RunnerHooks] | None,
) -> tuple[RunnerHooks, ...]:
    """Return one flattened ordered tuple of hook instances."""
    flattened_hooks: list[RunnerHooks] = []
    for hook_input in hooks:
        _extend_flattened_hooks(flattened_hooks, hook_input)
    return tuple(flattened_hooks)


def _extend_flattened_hooks(
    target: list[RunnerHooks],
    hook_input: RunnerHooks | Sequence[RunnerHooks] | None,
) -> None:
    """Append one hook input onto the flattened ordered hook list."""
    if hook_input is None:
        return

    if isinstance(hook_input, RunnerHooks):
        target.append(hook_input)
        return

    raw_hooks = cast(Sequence[object], hook_input)
    for hook in raw_hooks:
        if isinstance(hook, RunnerHooks):
            target.append(hook)
            continue

        raise TypeError(
            "Runner hook sequences must contain only `RunnerHooks` instances; "
            f"got {type(hook).__name__}."
        )
