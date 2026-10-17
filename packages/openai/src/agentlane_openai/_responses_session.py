"""Keep native connection state and steering transitions local to one harness run."""

import contextlib
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

from openai.resources.responses.responses import AsyncResponsesConnection
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseSteerAcceptedEvent,
    ResponseSteerFailedEvent,
    ResponseSteerPendingEvent,
)
from openai.types.responses.response_steer_input_param import ResponseSteerInputParam
from strenum import LowercaseStrEnum

from agentlane.models import (
    MessageDict,
    MessageRole,
    ModelContinuation,
    ModelStreamEvent,
    ModelStreamEventKind,
    SteeringInput,
    SteeringReceipt,
    SteeringStatus,
)
from agentlane.tracing import Span

from ._constants import NO_ACTIVE_RESPONSE
from ._event_adapter import to_stream_event
from ._request_adapter import ResponseInputItem, messages_to_input
from ._response_adapter import ResponseError, response_to_model_response


class _NextAction(LowercaseStrEnum):
    CREATE_RESPONSE = "create_response"
    """Send a request when the harness starts its next turn."""

    CONSUME_SUCCESSOR = "consume_successor"
    """Consume a successor already started by the server without sending a request."""


class ResponsesSession:
    """Adapt one SDK connection; the harness owns tools and committed run history."""

    def __init__(
        self,
        connection: AsyncResponsesConnection,
        model: str,
        async_tools: frozenset[str],
        record_response: Callable[[Span[Any], OpenAIResponse], None],
    ) -> None:
        self._connection = connection
        self._model = model
        self._async_tools = async_tools
        self._record_response = record_response

        # These values describe only this connection's accepted native history.
        self._active_response_id: str | None = None
        self._previous_response_id: str | None = None
        self._request_input: list[ResponseInputItem] = []
        self._next_action = _NextAction.CREATE_RESPONSE
        self._steering: list[SteeringReceipt] = []

    def settle(self) -> None:
        """Leave unresolved receipts unknown when the connection scope ends."""
        self._active_response_id = None
        for receipt in self._steering:
            if receipt.status in {
                SteeringStatus.SUBMITTED.value,
                SteeringStatus.ACCEPTED.value,
            }:
                receipt.status = SteeringStatus.UNKNOWN.value

    async def stream(
        self,
        input_data: list[ResponseInputItem],
        call_args: dict[str, Any],
        span: Span[Any],
    ) -> AsyncIterator[ModelStreamEvent]:
        """Create or resume one logical response, then resolve its continuation."""
        await self._start_connection_response(input_data, call_args)

        async for event in self._connection_events(span):
            yield event

    async def steer(self, input: SteeringInput) -> SteeringReceipt:
        """Send once through the SDK; the event stream updates the returned receipt."""
        if self._active_response_id is None:
            raise RuntimeError(NO_ACTIVE_RESPONSE)

        # Track the submission before sending; only later events can confirm
        # application.
        receipt = SteeringReceipt(deepcopy(input), self._active_response_id)
        self._steering.append(receipt)
        try:
            await self._connection.response.steer(
                input=cast(ResponseSteerInputParam, receipt.input),
                previous_response_id=receipt.previous_response_id,
            )
        except BaseException:
            receipt.status = SteeringStatus.UNKNOWN.value
            self._active_response_id = None

            # The SDK send may have reached the server. End this connection
            # rather than return an answer that may omit accepted input.
            with contextlib.suppress(Exception):
                await self._connection.close()

            raise

        return receipt

    def prepare_input(
        self,
        messages: list[MessageDict],
        call_args: dict[str, Any],
    ) -> list[ResponseInputItem]:
        """Validate history ownership before the next response starts."""
        # Reject options that would give the SDK separate ownership of run history.
        for key in (
            "conversation",
            "context_management",
            "background",
            "stream",
            "stream_id",
            "previous_response_id",
        ):
            if call_args.get(key) is not None:
                raise ValueError(
                    f"Live harness runs own {key}; start a separate SDK request for this option."
                )

        # An automatic successor already uses the accepted history; it cannot be
        # replaced.
        input_data = messages_to_input(messages)
        if (
            self._next_action == _NextAction.CONSUME_SUCCESSOR
            and input_data != self._request_input
        ):
            raise ValueError("Cannot replace history during an automatic successor.")

        return input_data

    async def _start_connection_response(
        self,
        input_data: list[ResponseInputItem],
        call_args: dict[str, Any],
    ) -> None:
        # Consume an automatic successor without submitting the same turn again.
        if self._next_action == _NextAction.CONSUME_SUCCESSOR:
            self._next_action = _NextAction.CREATE_RESPONSE
            return

        # Continue from the prior response only when the history prefix is unchanged.
        if (
            self._previous_response_id is not None
            and input_data[: len(self._request_input)] == self._request_input
        ):
            call_args["previous_response_id"] = self._previous_response_id
            delta = input_data[len(self._request_input) :]
        else:
            if any(r.status == SteeringStatus.ACCEPTED.value for r in self._steering):
                raise ValueError(
                    "Cannot replace history while accepted steering is pending."
                )

            delta = input_data

        # Retain the full history locally while sending only the required input.
        self._request_input = list(input_data)
        await self._connection.response.create(
            model=self._model,
            input=cast(ResponseInputParam, delta),
            **call_args,
        )

    async def _connection_events(
        self,
        span: Span[Any],
    ) -> AsyncIterator[ModelStreamEvent]:
        terminal: OpenAIResponse | None = None
        async for raw in self._connection:
            if any(r.status == SteeringStatus.UNKNOWN.value for r in self._steering):
                raise RuntimeError("Native steering send outcome is unknown.")

            # Keep a terminal response until pending steering settles.
            events: Iterable[ModelStreamEvent]
            if isinstance(raw, ResponseCreatedEvent):
                events = self._response_created_events(raw, terminal)
            elif isinstance(raw, ResponseSteerAcceptedEvent):
                events = (self._steering_accepted_event(raw),)
            elif isinstance(raw, ResponseSteerFailedEvent):
                events = self._steering_failed_events(raw, terminal)
            elif isinstance(raw, ResponseSteerPendingEvent):
                events = self._steering_pending_events(raw, terminal)
            elif isinstance(raw, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                terminal = raw.response
                events = (self._response_finished_event(raw, span),)
            else:
                events = (to_stream_event(raw, self._async_tools),)

            # Settle the harness turn only after the adapter resolves its continuation.
            for event in events:
                yield event
                if event.kind == ModelStreamEventKind.ERROR:
                    raise event.error or ResponseError(raw)
                if event.kind == ModelStreamEventKind.COMPLETED:
                    return

        raise RuntimeError(
            "Native connection ended before the logical response completed."
        )

    def _apply_steering(
        self, response: OpenAIResponse
    ) -> tuple[list[SteeringReceipt], list[MessageDict]]:
        # Acceptance becomes application only when a matching successor starts.
        applied = [
            r
            for r in self._steering
            if r.status == SteeringStatus.ACCEPTED.value
            and r.previous_response_id == response.previous_response_id
        ]

        # Commit the applied input in the same order as its receipts.
        inputs: list[MessageDict] = []
        for receipt in applied:
            receipt.status = SteeringStatus.APPLIED.value
            receipt.response_id = response.id
            inputs.append(
                {"role": MessageRole.USER.value, "content": receipt.input}
                if isinstance(receipt.input, str)
                else {
                    "role": MessageRole.USER.value,
                    "openai_responses_input": deepcopy(receipt.input),
                }
            )

        return applied, inputs

    def _response_created_events(
        self,
        raw: ResponseCreatedEvent,
        terminal: OpenAIResponse | None,
    ) -> Iterator[ModelStreamEvent]:
        # Reject an unrelated successor before committing steering input.
        if terminal is not None and raw.response.previous_response_id != terminal.id:
            raise ResponseError(raw, "Unexpected steering successor.")

        self._active_response_id = raw.response.id
        applied, inputs = self._apply_steering(raw.response)

        # An automatic successor completes the prior harness turn without another
        # create.
        if terminal is not None:
            self._request_input.extend(messages_to_input(inputs))
            self._next_action = _NextAction.CONSUME_SUCCESSOR
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                raw=raw,
                provider_event_type=raw.type,
                steering=[replace(r) for r in applied],
            )
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response_to_model_response(terminal, allow_steered=True),
                continuation=ModelContinuation.AUTOMATIC.value,
                input_messages=inputs,
            )
        elif inputs:
            # The server prepends queued user input to the explicit tool outputs.
            offset = len(self._request_input)
            while (
                offset
                and self._request_input[offset - 1].get("type")
                == "function_call_output"
            ):
                offset -= 1

            self._request_input[offset:offset] = messages_to_input(inputs)
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                raw=raw,
                provider_event_type=raw.type,
                steering=[replace(r) for r in applied],
                input_messages=inputs,
            )
        else:
            yield to_stream_event(raw, self._async_tools)

    def _steering_accepted_event(
        self, raw: ResponseSteerAcceptedEvent
    ) -> ModelStreamEvent:
        receipt = next(
            (
                r
                for r in self._steering
                if r.status == SteeringStatus.SUBMITTED.value
                and r.previous_response_id == raw.steer.previous_response_id
            ),
            None,
        )

        # Record acceptance without adding input to history until a successor starts.
        if receipt is not None:
            receipt.id = raw.steer.id
            receipt.status = SteeringStatus.ACCEPTED.value

        return ModelStreamEvent(
            kind=ModelStreamEventKind.PROVIDER,
            raw=raw,
            provider_event_type=raw.type,
            steering=[replace(receipt)] if receipt else None,
        )

    def _steering_failed_events(
        self,
        raw: ResponseSteerFailedEvent,
        terminal: OpenAIResponse | None,
    ) -> Iterator[ModelStreamEvent]:
        receipt = next(
            (
                r
                for r in self._steering
                if (raw.steer.id is not None and r.id == raw.steer.id)
                or (
                    r.status == SteeringStatus.SUBMITTED.value
                    and r.previous_response_id == raw.steer.previous_response_id
                )
            ),
            None,
        )
        if receipt is not None:
            receipt.id = raw.steer.id or receipt.id
            receipt.status = SteeringStatus.FAILED.value
            receipt.error = raw.error.model_dump(mode="json")

        yield ModelStreamEvent(
            kind=ModelStreamEventKind.PROVIDER,
            raw=raw,
            provider_event_type=raw.type,
            steering=[replace(receipt)] if receipt else None,
        )

        # Release the saved terminal response when no steering submission remains.
        if terminal is not None and not self._has_pending_steering():
            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=response_to_model_response(terminal),
            )

    def _steering_pending_events(
        self,
        raw: ResponseSteerPendingEvent,
        terminal: OpenAIResponse | None,
    ) -> Iterator[ModelStreamEvent]:
        if (
            terminal is None
            and raw.steer.previous_response_id == self._previous_response_id
        ):
            # Several submissions can name the same required results.
            yield to_stream_event(raw, self._async_tools)
            return
        if terminal is None or any(
            item.type != "function_call_output" for item in raw.required_input
        ):
            raise ResponseError(raw, "Unsupported steering continuation input.")

        yield to_stream_event(raw, self._async_tools)
        yield ModelStreamEvent(
            kind=ModelStreamEventKind.COMPLETED,
            response=response_to_model_response(terminal),
            continuation=ModelContinuation.EXPLICIT.value,
            required_tool_call_ids=tuple(
                item.call_id
                for item in raw.required_input
                if item.type == "function_call_output"
            ),
        )

    def _has_pending_steering(self) -> bool:
        return any(
            r.status in {SteeringStatus.SUBMITTED.value, SteeringStatus.ACCEPTED.value}
            for r in self._steering
        )

    def _response_finished_event(
        self,
        raw: ResponseCompletedEvent | ResponseIncompleteEvent,
        span: Span[Any],
    ) -> ModelStreamEvent:
        response = raw.response
        self._active_response_id = None
        self._previous_response_id = response.id

        # Save native output so the next request can send only new input.
        self._request_input.extend(
            item.model_dump(mode="json", exclude_unset=True, by_alias=True)
            for item in response.output
        )

        # Account for the finished provider response even if steering delays the turn.
        self._record_response(span, response)

        # Wait for successor or steering failure events before completing the turn.
        if self._has_pending_steering():
            return ModelStreamEvent(
                kind=ModelStreamEventKind.PROVIDER,
                raw=raw,
                provider_event_type=raw.type,
            )

        return to_stream_event(raw, self._async_tools)
