"""Default stateless runner for the harness agent loop."""

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, cast

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


class Runner:
    """Stateless default runner for harness agents.

    The runner owns the generic loop mechanics for the current phase:

    1. build the next model request from instructions and run state,
    2. call the model once,
    3. accumulate the raw `ModelResponse`,
    4. interpret the response as a direct answer, and
    5. return a minimal `RunResult`.

    Tool execution, handoffs, and richer interruption semantics land in later
    phases. For now they still fail explicitly from the runner layer.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 1,
        max_turns: int = 128,
        is_retryable: Callable[[BaseException], bool] | None = None,
    ) -> None:
        """Initialize retry and loop limits for one reusable runner."""
        if max_attempts < 1:
            raise ValueError("Runner.max_attempts must be at least 1.")
        if max_turns < 1:
            raise ValueError("Runner.max_turns must be at least 1.")
        self._max_attempts = max_attempts
        self._max_turns = max_turns
        self._is_retryable = is_retryable

    async def run(
        self,
        agent: Task,
        state: RunState,
        *,
        hooks: RunnerHooks | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        """Execute the generic harness loop for the provided run state."""
        resolved_hooks = hooks or RunnerHooks()
        result: RunResult | None = None

        await resolved_hooks.on_agent_start(agent, _copy_run_state(state))
        try:
            while True:
                state.turn_count += 1
                _check_turn_limit(state.turn_count, self._max_turns)

                messages = _build_request(agent, state)
                response = await self._run_with_retry(
                    agent=agent,
                    messages=messages,
                    hooks=resolved_hooks,
                    cancellation_token=cancellation_token,
                )

                state.responses.append(response)
                _validate_terminal_response(response)
                state.continuation_history.append(response)

                result = RunResult(
                    final_output=_extract_direct_answer(response),
                    responses=list(state.responses),
                    turn_count=state.turn_count,
                )
                return result
        finally:
            await resolved_hooks.on_agent_end(agent, result)

    async def _run_with_retry(
        self,
        *,
        agent: Task,
        messages: list[MessageDict],
        hooks: RunnerHooks,
        cancellation_token: CancellationToken | None,
    ) -> ModelResponse:
        """Execute one model turn under the configured retry policy."""
        retryable_call = retry_on_errors(
            max_retries=self._max_attempts,
            is_retryable=self._is_retryable,
        )(self._run_once)
        retry_result = await retryable_call(
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
        """Execute one single model attempt without outer retry orchestration."""
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
    """Build one concrete model request from the current run state."""
    messages: list[MessageDict] = []
    messages.extend(_instruction_messages(getattr(agent, "instructions", None)))
    messages.extend(_input_to_messages(state.original_input))
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

    messages: list[MessageDict] = []
    for item in run_input:
        messages.extend(_history_item_to_messages(item))
    return messages


def _history_item_to_messages(item: object) -> list[MessageDict]:
    """Render one continuation item into canonical model messages."""
    # The public run-state boundary stays generic. For the current phase the
    # runner uses one simple convention internally: prior `ModelResponse`
    # objects represent assistant turns, while any other history item is
    # treated as new user-side input that must be normalized for the model.
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
    """Render full role-specific prompt messages without dropping metadata."""
    messages = [
        _normalize_message(message)
        for message in prompt_spec.template.render_messages(prompt_spec.values)
        if message.get("role") == role
    ]
    if messages:
        return messages
    raise ValueError(f"PromptSpec must render at least one {role}-role message.")


def _normalize_message(message: Mapping[str, object]) -> MessageDict:
    """Copy one message and normalize its content field when present."""
    normalized_message = dict(message)
    if "content" in normalized_message:
        normalized_message["content"] = _normalize_content(
            normalized_message["content"]
        )
    return normalized_message


def _normalize_content(content: object) -> object:
    """Normalize arbitrary content into model-ready message content."""
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


def _copy_run_state(state: RunState) -> RunState:
    """Copy run state for hook observation without exposing live mutation."""
    original_input = state.original_input
    copied_input = (
        original_input if isinstance(original_input, str) else list(original_input)
    )
    return RunState(
        original_input=copied_input,
        continuation_history=list(state.continuation_history),
        responses=list(state.responses),
        turn_count=state.turn_count,
    )


def _check_turn_limit(turn_count: int, max_turns: int) -> None:
    """Fail fast when the run exceeds the configured turn limit."""
    if turn_count > max_turns:
        raise RuntimeError(f"Runner exceeded the configured turn limit of {max_turns}.")


def _require_model(agent: Task) -> Model[ModelResponse]:
    """Return the configured model client for a harness task or raise."""
    model = getattr(agent, "model", None)
    if model is None:
        raise RuntimeError(
            "Runner requires the task to expose a configured 'model' client."
        )
    return model


def _assistant_message_from_response(response: ModelResponse) -> MessageDict:
    """Extract the first assistant message from an already validated response."""
    _validate_response_choice(response)
    return response.choices[0].message.model_dump(mode="json", exclude_none=True)


def _extract_direct_answer(response: ModelResponse) -> object:
    """Extract the direct assistant answer from a validated terminal response."""
    _validate_response_choice(response)
    return response.choices[0].message.content


def _model_args(agent: Task) -> dict[str, object] | None:
    """Return model request args exposed by the task, if any."""
    model_args = getattr(agent, "model_args", None)
    if model_args is None:
        return None
    return dict(model_args)


def _schema(agent: Task) -> type[BaseModel] | OutputSchema[Any] | None:
    """Return the structured-output schema exposed by the task, if any."""
    return getattr(agent, "schema", None)


def _tools(agent: Task) -> Tools | None:
    """Return model-call tool config exposed by the task, if any."""
    return getattr(agent, "tools", None)


def _validate_terminal_response(response: ModelResponse) -> None:
    """Fail fast when the response is not a direct final answer."""
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
    """Fail fast when the response is missing the first model choice."""
    if response.choices:
        return
    raise _model_behavior_error(
        "Runner expected the model response to contain at least one choice.",
        raw_response=response,
    )


def _model_behavior_error(
    message: str,
    *,
    raw_response: ModelResponse | None,
) -> ModelBehaviorError:
    """Create a model-behavior error annotated with raw response details."""
    error = ModelBehaviorError(message)
    error.run_data = RunErrorDetails(raw_response=raw_response)
    return error
