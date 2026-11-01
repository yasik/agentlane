"""Shared state contracts for shim preparation and compaction."""

from dataclasses import dataclass

from agentlane.models import MessageDict


@dataclass(frozen=True, kw_only=True, slots=True)
class PreservedContext:
    """Keep application state available to the model after history compaction.

    State can survive in storage while its conversation messages are removed.
    For example, a plan remains in `RunState.shim_state` after compaction drops
    its `write_plan` call, but the generated summary can omit plan details.
    The owning shim supplies a complete text projection through this contract
    so compaction can preserve it without interpreting plan-specific storage.

    The default and ranked selectors keep these projections. A custom selector
    can change that choice. Selected content is injected as internal context
    only when no retained `represented_by` message already contains the complete
    current state. Its cost is included in the preservation token budget.

    The owner remains responsible for storing and updating the state. This
    value carries its text and message references for one preservation decision.
    """

    origin: str
    """Unique owner key within the combined preserved-context message."""

    content: str
    """Complete text to inject as internal context when selected."""

    represented_by: tuple[MessageDict, ...] = ()
    """Original messages that each already contain this complete state.

    Each message must still establish current state after later updates;
    an earlier equal value is insufficient.
    """
