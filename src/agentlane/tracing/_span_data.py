"""Span data types for different kinds of operations."""

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class SpanData(abc.ABC):
    """Abstract base class for span data."""

    @property
    @abc.abstractmethod
    def type(self) -> str:
        """Return the type identifier for this span data.

        Returns:
            The span type as a string.
        """

    @abc.abstractmethod
    def export(self) -> dict[str, Any]:
        """Export the span data as a dictionary.

        Returns:
            Dictionary representation of the span data.
        """


@dataclass
class AgentSpanData(SpanData):
    """Data for agent spans."""

    name: str
    handoffs: list[str] | None = None
    tools: list[str] | None = None
    output_type: str | None = None

    @property
    def type(self) -> str:
        return "agent"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "handoffs": self.handoffs,
            "tools": self.tools,
            "output_type": self.output_type,
        }


@dataclass
class FunctionSpanData(SpanData):
    """Data for function/operation spans."""

    name: str
    input: str | None = None
    output: str | None = None

    @property
    def type(self) -> str:
        return "function"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": str(self.output) if self.output else None,
        }


@dataclass
class GenerationSpanData(SpanData):
    """Data for LLM generation spans."""

    input: Sequence[Mapping[str, Any]] | None = None
    output: Sequence[Mapping[str, Any]] | None = None
    events: Sequence[Mapping[str, Any]] | None = None
    model: str | None = None
    model_config: Mapping[str, Any] | None = None
    usage: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "generation"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": self.output,
            "events": self.events,
            "model": self.model,
            "model_config": self.model_config,
            "usage": self.usage,
        }


@dataclass
class CustomSpanData(SpanData):
    """Data for custom spans with arbitrary attributes."""

    name: str
    data: dict[str, Any] | None = None

    @property
    def type(self) -> str:
        return "custom"

    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
        }
