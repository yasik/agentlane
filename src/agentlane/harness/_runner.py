"""Default stateless runner for the harness agent loop.

The runner is the inner execution engine of one agent turn. It is
intentionally stateless — all mutable conversation state lives in
``RunState``, which the lifecycle owns and copies before each invocation.

For the current phase the runner handles a single terminal model turn:

1. Build the ``list[MessageDict]`` request from instructions + run state.
2. Call the model (with optional retry).
3. Validate the response as a direct assistant answer.
4. Return a ``RunResult`` with the extracted output.

Tool execution and handoff loops are deferred to later phases and will
fail fast with ``ModelBehaviorError`` if the model returns tool calls.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from agentlane.models import (
    MessageDict,
    Model,
    ModelBehaviorError,
    ModelResponse,
    OutputSchema,
    PromptSpec,
    RunErrorDetails,
    Tools,
    retry_on_errors,
)
from agentlane.runtime import CancellationToken

from ._hooks import RunnerHooks
from ._run import RunResult, RunState
from ._task import Task


@runtime_checkable
class RunnerTask(Protocol):
    """Structural protocol for tasks compatible with the default runner.

    The runner accesses these properties during request building and model
    calls. Any task that exposes them is compatible — no subclassing needed.
    """

    @property
    def model(self) -> Model[ModelResponse] | None: ...

    @property
    def model_args(self) -> dict[str, Any] | None: ...

    @property
    def schema(self) -> type[BaseModel] | OutputSchema[Any] | None: ...

    @property
    def tools(self) -> Tools | None: ...

    @property
    def instructions(self) -> str | PromptSpec[Any] | None: ...


class Runner:
    """Stateless default runner for harness agents.

    A single ``Runner`` instance can be shared across multiple agents
    safely — it holds only configuration (limits, retry policy), never
    per-conversation state.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        max_turns: int = 128,
        is_retryable: Callable[[BaseException], bool] | None = None,
    ) -> None:
        """Initialize retry and loop limits for one reusable runner.

        Args:
            max_attempts: Total attempts per model call (1 = no retry).
            max_turns: Safety cap on conversation turns per run.
            is_retryable: Optional predicate for retry eligibility. Defaults
                to the standard HTTP-status-code check in ``retry_on_errors``.
        """
        if max_attempts < 1:
            raise ValueError("Runner.max_attempts must be at least 1.")
        if max_turns < 1:
            raise ValueError("Runner.max_turns must be at least 1.")

        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._is_retryable = is_retryable

        # Cache the retry-wrapped model call once at init so we don't
        # recreate the decorator closure on every turn.
        self._retryable_run_once = retry_on_errors(
            max_retries=self._max_attempts,
            is_retryable=self._is_retryable,
        )(self._run_once)

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute the generic harness loop for the provided run state.

        The lifecycle passes a private working copy of ``state``, so the
        runner mutates it freely (incrementing ``turn_count``, appending
        responses) without risking the persisted baseline.
        """
        resolved_hooks = hooks or RunnerHooks()
        result: RunResult | None = None

        # Hook receives the same working copy — safe because the lifecycle
        # already isolated it before calling us.
        await resolved_hooks.on_agent_start(agent, state)
        try:
            while True:
                state.turn_count += 1
                _check_turn_limit(state.turn_count, self._max_turns)

                # 1. Build the full message list from instructions + history
                messages = _build_request(agent, state)

                # 2. Call the model (with retry if configured)
                response = await self._run_with_retry(
                    agent=agent,
                    messages=messages,
                    hooks=resolved_hooks,
                    cancellation_token=cancellation_token,
                )

                # 3. Record the raw response on the working state
                state.responses.append(response)

                # 4. Validate this is a terminal answer (no tool calls)
                _validate_terminal_response(response)

                # 5. Append the response as an assistant turn for future
                #    continuation when the agent receives more input
                state.continuation_history.append(response)

                # 6. Extract and return the final output
                result = RunResult(
                    final_output=_extract_direct_answer(response),
                    responses=list(state.responses),
                    turn_count=state.turn_count,
                )
                return result
        finally:
            # Always fire the end hook — result is None if the loop raised.
            await resolved_hooks.on_agent_end(agent, result)

    async def _run_with_retry(
        self,
        *,
        agent: Task,
        messages: list[MessageDict],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one model turn under the configured retry policy.

        Uses the pre-cached ``self._retryable_run_once`` wrapper so the
        retry decorator is not re-created per turn.
        """
        retry_result = await self._retryable_run_once(
            agent=agent,
            messages=messages,
            hooks=hooks,
            cancellation_token=cancellation_token,
        )
        return retry_result.result

    async def _run_once(
        self,
        *,
        agent: Task,
        messages: list[MessageDict],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one single model attempt (no retry logic here)."""
        model = _require_model(agent)

        await hooks.on_llm_start(agent, messages)
        response = await model(
            messages,
            extra_call_args=_model_args(agent),
            schema=_schema(agent),
            tools=_tools(agent),
            cancellation_token=cancellation_token,
        )
        await hooks.on_llm_end(agent, response)

        return response


def _build_request(agent: Task, state: RunState) -> list[MessageDict]:
    """Build the full model request from agent config and run state."""
    runner_task = _as_runner_task(agent)

    messages: list[MessageDict] = []

    # System instructions (if any) always come first
    messages.extend(
        _instruction_messages(runner_task.instructions if runner_task else None)
    )

    # Original input — the first user message(s) that started this run
    messages.extend(_input_to_messages(state.original_input))

    # Continuation history — alternating assistant/user turns
    for item in state.continuation_history:
        messages.extend(_history_item_to_messages(item))

    return messages


def _instruction_messages(
    instructions: str | PromptSpec[Any] | None,
) -> list[MessageDict]:
    """Render system instructions into canonical model messages."""
    if instructions is None:
        return []
    if isinstance(instructions, PromptSpec):
        return _prompt_messages("system", instructions)
    return [_normalize_message({"role": "system", "content": instructions})]


def _input_to_messages(run_input: str | list[object]) -> list[MessageDict]:
    """Render the original run input into canonical model messages."""
    if isinstance(run_input, str):
        return [_normalize_message({"role": "user", "content": run_input})]

    # List inputs may contain mixed types (strings, PromptSpecs,
    # ModelResponses) — delegate each item through the same rendering
    # pipeline used for continuation history.
    messages: list[MessageDict] = []
    for item in run_input:
        messages.extend(_history_item_to_messages(item))
    return messages


def _history_item_to_messages(item: object) -> list[MessageDict]:
    """Render one continuation item into canonical model messages.

    The run-state boundary is intentionally generic (``list[object]``).
    The runner applies a simple convention: ``ModelResponse`` objects are
    assistant turns; everything else is user-side input.
    """
    if isinstance(item, ModelResponse):
        return [_assistant_message_from_response(item)]
    return _user_item_to_messages(item)


def _user_item_to_messages(item: object) -> list[MessageDict]:
    """Render one user-side continuation item into canonical model messages."""
    if isinstance(item, PromptSpec):
        prompt_spec = cast(PromptSpec[Any], item)
        return _prompt_messages("user", prompt_spec)
    return [_normalize_message({"role": "user", "content": item})]


def _prompt_messages(
    role: Literal["system", "user"],
    prompt_spec: PromptSpec[Any],
) -> list[MessageDict]:
    """Render a ``PromptSpec`` and keep only messages matching ``role``.

    A single template may render both system and user messages. The runner
    calls this once per role so that instruction-only templates produce
    system messages and user-only templates produce user messages.
    """
    messages = [
        _normalize_message(message)
        for message in prompt_spec.template.render_messages(prompt_spec.values)
        if message.get("role") == role
    ]
    if messages:
        return messages
    raise ValueError(f"PromptSpec must render at least one {role}-role message.")


def _normalize_message(message: Mapping[str, object]) -> MessageDict:
    """Copy one message dict and normalize its ``content`` field."""
    normalized_message = dict(message)
    if "content" in normalized_message:
        normalized_message["content"] = _normalize_content(
            normalized_message["content"]
        )
    return normalized_message


def _normalize_content(content: object) -> object:
    """Normalize arbitrary content into a model-ready value.

    Fast-path checks are ordered by expected frequency:
      1. ``str`` — most common, returned as-is.
      2. ``list`` — multi-part content arrays, passed through.
      3. ``BaseModel`` — Pydantic models serialized to JSON.
      4. dataclass — stdlib dataclasses serialized to JSON.
      5. Anything else — attempt ``json.dumps``, fall back to ``str()``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return cast(list[object], content)
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    if is_dataclass(content) and not isinstance(content, type):
        return json.dumps(asdict(content))
    try:
        return json.dumps(content)
    except TypeError:
        return str(content)


def _as_runner_task(agent: Task) -> RunnerTask | None:
    """Narrow a ``Task`` to ``RunnerTask`` if it exposes runner properties."""
    if isinstance(agent, RunnerTask):
        return agent
    return None


def _require_model(agent: Task) -> Model[ModelResponse]:
    """Return the agent's model client, or raise if unconfigured."""
    runner_task = _as_runner_task(agent)
    if runner_task is not None and runner_task.model is not None:
        return runner_task.model

    raise RuntimeError(
        "Runner requires the task to expose a configured 'model' client."
    )


def _model_args(agent: Task) -> dict[str, object] | None:
    """Return a defensive copy of the agent's model-call arguments."""
    runner_task = _as_runner_task(agent)
    if runner_task is None:
        return None

    model_args = runner_task.model_args
    if model_args is None:
        return None
    # Copy so per-call mutations (e.g. provider-side updates) never leak
    # back into the agent descriptor.
    return dict(model_args)


def _schema(agent: Task) -> type[BaseModel] | OutputSchema[Any] | None:
    """Return the structured-output schema exposed by the agent, if any."""
    runner_task = _as_runner_task(agent)
    return runner_task.schema if runner_task else None


def _tools(agent: Task) -> Tools | None:
    """Return the tool configuration exposed by the agent, if any."""
    runner_task = _as_runner_task(agent)
    return runner_task.tools if runner_task else None


def _check_turn_limit(turn_count: int, max_turns: int) -> None:
    """Fail fast when the run exceeds the configured turn limit."""
    if turn_count > max_turns:
        raise RuntimeError(f"Runner exceeded the configured turn limit of {max_turns}.")


def _validate_terminal_response(response: ModelResponse) -> None:
    """Validate the response is a direct assistant answer.

    In the current phase the runner only handles terminal responses.
    Tool calls are not yet supported and will raise ``ModelBehaviorError``
    so the caller gets a clear signal rather than silently dropping them.
    """
    _validate_response_choice(response)

    message = response.choices[0].message

    if message.role != "assistant":
        raise _model_behavior_error(
            "Runner expected the first model choice to be an assistant message.",
            raw_response=response,
        )

    if message.tool_calls:
        raise _model_behavior_error(
            "Runner does not execute tool calls yet. Phase 5 adds tool-loop support.",
            raw_response=response,
        )


def _validate_response_choice(response: ModelResponse) -> None:
    """Ensure the response contains at least one choice."""
    if response.choices:
        return

    raise _model_behavior_error(
        "Runner expected the model response to contain at least one choice.",
        raw_response=response,
    )


def _assistant_message_from_response(response: ModelResponse) -> MessageDict:
    """Extract the first assistant message for continuation history."""
    _validate_response_choice(response)
    return response.choices[0].message.model_dump(mode="json", exclude_none=True)


def _extract_direct_answer(response: ModelResponse) -> object:
    """Extract the text content from a validated terminal response."""
    _validate_response_choice(response)
    return response.choices[0].message.content


def _model_behavior_error(
    message: str,
    *,
    raw_response: ModelResponse | None,
) -> ModelBehaviorError:
    """Create a ``ModelBehaviorError`` annotated with raw response details."""
    error = ModelBehaviorError(message)
    error.run_data = RunErrorDetails(raw_response=raw_response)
    return error
