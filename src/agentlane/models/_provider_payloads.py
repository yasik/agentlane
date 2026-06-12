"""Typed accessors for provider-native stream and usage payload shapes.

The framework already normalizes most provider detail onto `ModelStreamEvent`
and `ModelResponse`, but a few common provider-native shapes are still only
reachable through the preserved raw payload. These helpers give those shapes a
typed boundary so consumers do not duck-type `raw.item.phase` with `getattr` or
read usage totals off an `object`-typed attribute.
"""

from enum import StrEnum

from openai.types.responses.response_output_item_added_event import (
    ResponseOutputItemAddedEvent,
)
from openai.types.responses.response_output_item_done_event import (
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_message import ResponseOutputMessage

from ._streaming import ModelStreamEvent
from ._types import ModelResponse, Usage


class ReasoningPhase(StrEnum):
    """Provider-native output phase for a Responses output message.

    The OpenAI Responses API tags some output messages with a phase separating
    intermediate commentary from the final answer. Values mirror
    `openai.types.responses.ResponseOutputMessage.phase`.
    """

    COMMENTARY = "commentary"
    FINAL_ANSWER = "final_answer"


def get_reasoning_phase(event: ModelStreamEvent) -> ReasoningPhase | None:
    """Return the output-message phase carried by one stream event, if any.

    The phase lives on the provider-native `output_item` events preserved in
    `ModelStreamEvent.raw`. `None` means "no usable phase reported" — the event
    is not an output-item event, the item type carries no phase, the provider
    sent no phase, or the provider sent a phase string this enum does not model
    — and never signals a failure.

    The preserved raw payload is not re-validated, so a provider may surface a
    phase value outside `ReasoningPhase`. Such values degrade to `None` rather
    than raising, because an unmodeled phase is missing data to a consumer, not
    an error.

    Args:
        event: One normalized model stream event.

    Returns:
        The `ReasoningPhase` reported by the provider, or `None`.
    """
    raw = event.raw
    if not isinstance(raw, (ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent)):
        return None

    item = raw.item
    if not isinstance(item, ResponseOutputMessage):
        return None

    phase = item.phase
    if phase is None:
        return None
    try:
        return ReasoningPhase(phase)
    except ValueError:
        return None


class UsageTotals:
    """Typed view over the token totals on one model response.

    Wraps `ModelResponse.usage` so consumers read `prompt_tokens`,
    `completion_tokens`, and `total_tokens` through a typed surface instead of
    duck-typing the provider usage object.
    """

    __slots__ = ("_usage",)

    def __init__(self, usage: Usage) -> None:
        """Wrap one provider usage payload.

        Args:
            usage: The provider usage object from `ModelResponse.usage`.
        """
        self._usage = usage

    @property
    def prompt_tokens(self) -> int:
        """Return the prompt (input) token count."""
        return self._usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Return the completion (output) token count."""
        return self._usage.completion_tokens

    @property
    def total_tokens(self) -> int:
        """Return the combined prompt-plus-completion token count."""
        return self._usage.total_tokens


def get_usage_totals(response: ModelResponse | None) -> UsageTotals | None:
    """Return typed token totals for one response, or `None` when absent.

    Providers may omit usage on some responses; `None` means "no usage
    reported", which callers should treat as missing data rather than zero.

    Args:
        response: The model response to read usage from.

    Returns:
        A `UsageTotals` view, or `None` when no usage payload is present.
    """
    if response is None:
        return None
    usage = response.usage
    if usage is None:
        return None
    return UsageTotals(usage)
