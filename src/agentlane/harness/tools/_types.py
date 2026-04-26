"""Shared types for first-party harness tools."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentlane.models import Tool


@dataclass(frozen=True, slots=True)
class HarnessToolDefinition:
    """Executable tool plus optional prompt metadata for harness integration."""

    tool: Tool[Any, str]
    """Executable native tool exposed to the model."""

    prompt_snippet: str | None = None
    """Optional one-line summary rendered under `Available tools:`."""

    prompt_guidelines: Sequence[str] = ()
    """Optional usage guidance rendered under `Guidelines:`."""

    def __post_init__(self) -> None:
        """Normalize developer-provided sequences to immutable tuples."""
        object.__setattr__(
            self,
            "prompt_guidelines",
            tuple(self.prompt_guidelines),
        )
