"""This module defines the base class for prompt templates and a concrete implementation
for single and multi-part prompt templates.
"""

import abc
from dataclasses import dataclass
from typing import Any, TypeVar

from jinja2 import Template

from ._output_schema import OutputSchema

CtxT = TypeVar("CtxT")
OutT = TypeVar("OutT")


class PromptTemplateBase[CtxT, OutT](abc.ABC):
    """A base class for prompt templates."""

    @abc.abstractmethod
    def render_messages(self, ctx: CtxT | None = None) -> list[dict[str, Any]]:
        """Render the prompt template into a list of messages."""

    @abc.abstractmethod
    def response_format(self) -> dict[str, Any] | None:
        """Returns the response format for the prompt template."""


@dataclass(slots=True)
class PromptSpec[CtxT]:
    """A prompt template paired with the concrete values used to render it.

    This is the developer-facing templating surface used by the harness.
    Callers provide a strongly typed values object, often a `TypedDict`, and
    the harness later resolves the template into the system or user message
    content needed for the current role.
    """

    template: PromptTemplateBase[CtxT, Any]
    """Prompt template responsible for rendering the instructions."""

    values: CtxT | None = None
    """Concrete values supplied to the prompt template at render time."""


@dataclass(init=False)
class PromptTemplate(PromptTemplateBase[CtxT, OutT]):
    """A prompt template that can be used to generate a prompt for an LLM."""

    def __init__(
        self,
        *,
        system_template: str | None,
        user_template: str,
        output_schema: OutputSchema[OutT],
    ) -> None:
        self._system_template = (
            Template(system_template, trim_blocks=True) if system_template else None
        )
        self._user_template = Template(user_template, trim_blocks=True)
        self._output_schema = output_schema

    def render_messages(self, ctx: CtxT | None = None) -> list[dict[str, Any]]:
        """Render the prompt template into a list of messages."""
        msgs: list[dict[str, Any]] = []
        if self._system_template:
            msgs.append(
                {
                    "role": "system",
                    "content": (
                        self._system_template.render()
                        if ctx is None
                        else self._system_template.render(ctx)
                    ),
                }
            )
        msgs.append(
            {
                "role": "user",
                "content": (
                    self._user_template.render()
                    if ctx is None
                    else self._user_template.render(ctx)
                ),
            }
        )
        return msgs

    def response_format(self) -> dict[str, Any] | None:
        """Returns the response format for the prompt template."""
        return self._output_schema.response_format()


class PartTemplate[CtxT](abc.ABC):
    """A single message content part template that renders to a dict."""

    @abc.abstractmethod
    def render(self, ctx: CtxT | None = None) -> dict[str, Any]:
        """Render the content part using the provided context."""


@dataclass
class TextPart(PartTemplate[CtxT]):
    """A single message content part template that renders to a text."""

    template: str
    """Template string."""

    def __post_init__(self) -> None:
        self._tpl = Template(self.template, trim_blocks=True)

    def render(self, ctx: CtxT | None = None) -> dict[str, Any]:
        text = self._tpl.render() if ctx is None else self._tpl.render(ctx)
        return {"type": "text", "text": text}


@dataclass
class FilePart(PartTemplate[CtxT]):
    """A single message content part template that renders to a file."""

    base64_data: str
    """Base64-encoded file data."""

    media_type: str
    """File MIME type (e.g., 'application/pdf', 'image/png')."""

    def render(self, ctx: CtxT | None = None) -> dict[str, Any]:
        return {
            "type": "file",
            "file": {
                "file_data": f"data:{self.media_type};base64,{self.base64_data}",
            },
        }


@dataclass
class ImagePart(PartTemplate[CtxT]):
    """A single message content part template for images.

    Uses the OpenAI Responses API input_image format.
    """

    base64_data: str
    """Base64-encoded image data."""

    media_type: str
    """Image MIME type (e.g., 'image/png', 'image/jpeg')."""

    detail: str = "high"
    """Image detail level: 'low', 'high', or 'auto'. Default is 'high'."""

    def render(self, ctx: CtxT | None = None) -> dict[str, Any]:
        return {
            "type": "input_image",
            "image_url": f"data:{self.media_type};base64,{self.base64_data}",
            "detail": self.detail,
        }


@dataclass(init=False)
class MultiPartPromptTemplate(PromptTemplateBase[CtxT, OutT]):
    """A prompt template that renders message content as a list of parts.

    - system_parts: optional list of parts for the system message
    - user_parts: list of parts for the user message
    """

    def __init__(
        self,
        *,
        system_parts: list[PartTemplate[CtxT]] | None,
        user_parts: list[PartTemplate[CtxT]],
        output_schema: OutputSchema[OutT],
    ) -> None:
        """Initialize the multi-part prompt template."""
        self._system_parts = system_parts or []
        self._user_parts = user_parts
        self._output_schema = output_schema

    def render_messages(self, ctx: CtxT | None = None) -> list[dict[str, Any]]:
        """Render the multi-part prompt template into a list of messages."""
        msgs: list[dict[str, Any]] = []
        if self._system_parts:
            msgs.append(
                {
                    "role": "system",
                    "content": [part.render(ctx) for part in self._system_parts],
                }
            )
        msgs.append(
            {
                "role": "user",
                "content": [part.render(ctx) for part in self._user_parts],
            }
        )
        return msgs

    def response_format(self) -> dict[str, Any] | None:
        """Returns the response format for the multi-part prompt template."""
        return self._output_schema.response_format()
