"""Persistent snapshot storage contract for harness agents."""

from abc import ABC, abstractmethod

from ._snapshot import AgentSnapshot


class StateStore(ABC):
    """Store committed snapshots for one persistent agent."""

    @abstractmethod
    def load(self) -> AgentSnapshot | None:
        """Load the current snapshot, or return ``None`` when absent."""

    @abstractmethod
    def save(
        self,
        snapshot: AgentSnapshot,
        *,
        expected_revision: int | None,
    ) -> None:
        """Save a snapshot when the current revision matches.

        ``expected_revision=None`` requires the store to be empty.
        Otherwise, ``snapshot.revision`` must be greater than
        ``expected_revision``.
        """

    @abstractmethod
    def delete(self, *, expected_revision: int | None) -> None:
        """Delete the current snapshot when its revision matches.

        ``expected_revision=None`` requires the store to be empty.
        """
