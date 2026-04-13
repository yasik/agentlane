"""Tests for the shared model streaming contract."""

import asyncio
from typing import Any

from pydantic import BaseModel

from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    OutputSchema,
    Tools,
)


class _FallbackModel(Model[ModelResponse]):
    """Minimal model implementation for default stream fallback tests."""

    def __init__(self, response: ModelResponse | None = None) -> None:
        self._response = response

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        del messages, extra_call_args, schema, tools, kwargs
        if self._response is None:
            raise RuntimeError("boom")
        return self._response


async def _collect_stream(model: Model[ModelResponse]) -> list[ModelStreamEvent]:
    """Collect one stream into a list of events."""
    events: list[ModelStreamEvent] = []
    async for event in model.stream_response(
        messages=[{"role": "user", "content": "hello"}],
    ):
        events.append(event)
    return events


def _make_response(content: str) -> ModelResponse:
    """Create one canonical model response for fallback tests."""
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
        }
    )


def test_model_stream_response_fallback_emits_completed_event() -> None:
    """Default streaming should emit one completed event from get_response."""
    model = _FallbackModel(response=_make_response("hello"))

    events = asyncio.run(_collect_stream(model))

    assert len(events) == 1
    assert events[0].kind == ModelStreamEventKind.COMPLETED
    assert events[0].response is not None
    assert events[0].response.choices[0].message.content == "hello"


def test_model_stream_response_fallback_emits_error_then_raises() -> None:
    """Default streaming should surface errors as both an event and an exception."""
    model = _FallbackModel()
    collected_events: list[ModelStreamEvent] = []

    async def _run() -> None:
        async for event in model.stream_response(
            messages=[{"role": "user", "content": "hello"}],
        ):
            collected_events.append(event)

    try:
        asyncio.run(_run())
    except RuntimeError as error:
        assert str(error) == "boom"
    else:
        raise AssertionError("expected RuntimeError from fallback stream")

    assert len(collected_events) == 1
    assert collected_events[0].kind == ModelStreamEventKind.ERROR
    assert str(collected_events[0].error) == "boom"
