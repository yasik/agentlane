"""Project native stream events without owning transport or harness execution."""

from typing import cast

from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningTextDeltaEvent,
    ResponsesServerEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from agentlane.models import ModelStreamEvent, ModelStreamEventKind

from ._response_adapter import (
    ResponseError,
    convert_function_call,
    response_to_model_response,
)


def to_stream_event(
    raw_event: ResponseStreamEvent | ResponsesServerEvent, async_tools: frozenset[str]
) -> ModelStreamEvent:
    """Map one OpenAI streaming event into the shared streaming envelope."""
    provider_event_type = raw_event.type
    item_index = getattr(raw_event, "output_index", None)

    if (
        isinstance(raw_event, ResponseOutputItemDoneEvent)
        and isinstance(raw_event.item, ResponseFunctionToolCall)
        and raw_event.item.async_
    ):
        # Validate complete async calls before exposing them to the harness
        # executor.
        item = raw_event.item
        if (
            item.name not in async_tools
            or item.namespace is not None
            or (item.caller is not None and item.caller.type != "direct")
        ):
            raise ResponseError(
                raw_event, "Unsupported or undeclared async function call."
            )

        return ModelStreamEvent(
            kind=ModelStreamEventKind.PROVIDER,
            raw=raw_event,
            provider_event_type=provider_event_type,
            async_tool_call=convert_function_call(item),
        )

    if isinstance(raw_event, ResponseTextDeltaEvent):
        return ModelStreamEvent(
            kind=ModelStreamEventKind.TEXT_DELTA,
            raw=raw_event,
            provider_event_type=provider_event_type,
            item_index=cast(int | None, item_index),
            text=raw_event.delta,
        )

    if isinstance(raw_event, ResponseFunctionCallArgumentsDeltaEvent):
        return ModelStreamEvent(
            kind=ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA,
            raw=raw_event,
            provider_event_type=provider_event_type,
            item_index=cast(int | None, item_index),
            tool_call_id=raw_event.item_id,
            arguments_delta=raw_event.delta,
        )

    if isinstance(
        raw_event,
        (ResponseReasoningTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent),
    ):
        return ModelStreamEvent(
            kind=ModelStreamEventKind.REASONING,
            raw=raw_event,
            provider_event_type=provider_event_type,
            item_index=cast(int | None, item_index),
            reasoning=raw_event.delta,
        )

    if isinstance(raw_event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
        # Reject a missing provider payload before it reaches output conversion.
        response = cast(OpenAIResponse | None, getattr(raw_event, "response", None))
        if response is None:
            raise RuntimeError(
                "OpenAI completed stream event did not include a response."
            )

        return ModelStreamEvent(
            kind=ModelStreamEventKind.COMPLETED,
            raw=raw_event,
            provider_event_type=provider_event_type,
            response=response_to_model_response(response),
        )

    if provider_event_type in {"error", "response.failed"}:
        return ModelStreamEvent(
            kind=ModelStreamEventKind.ERROR,
            raw=raw_event,
            provider_event_type=provider_event_type,
            error=ResponseError(getattr(raw_event, "response", raw_event)),
        )

    # Retain other SDK events without inventing a harness operation for each type.
    raw_item = getattr(raw_event, "item", None)

    return ModelStreamEvent(
        kind=ModelStreamEventKind.PROVIDER,
        raw=raw_event,
        provider_event_type=provider_event_type,
        item_index=cast(int | None, item_index),
        item_type=cast(str | None, getattr(raw_item, "type", None)),
    )
