"""Budgeted preservation from a client-owned block ranking."""

from dataclasses import replace

from ._errors import CompactionError
from ._types import PreservationRanker, PreservationRequest, PreservationSelection


class RankedPreservationSelector:
    """Keep required blocks and owner state, then fit ranked optional blocks.

    The ranker owns relevance and any model calls. This selector replaces the
    default recency preference; it keeps the default current-state projections.
    """

    def __init__(self, ranker: PreservationRanker) -> None:
        self._ranker = ranker

    async def __call__(self, request: PreservationRequest, /) -> PreservationSelection:
        """Select ranked blocks within the canonical preservation budget."""
        selection = PreservationSelection(
            block_ids=tuple(block.id for block in request.blocks if block.required),
            context=request.default.context,
        )
        retainable_ids = {block.id for block in request.blocks if block.retainable}

        # The hard window alone can retain all history at the compaction trigger.
        # Share the default soft allowance so ranked selection leaves summary room.
        budget = min(request.token_budget, request.retention_budget)

        # Required blocks are already selected. Ignore already-selected IDs;
        # invalid model output fails instead of silently losing chosen evidence.
        for block_id in await self._ranker.rank(request):
            if type(block_id) is not int or block_id not in retainable_ids:
                raise CompactionError(
                    "Ranker returned an invalid or unretainable block ID."
                )

            if block_id in selection.block_ids:
                continue

            # A large block can be skipped while a later smaller block fits.
            # Price the full candidate because retained state can avoid injection.
            candidate = replace(selection, block_ids=(*selection.block_ids, block_id))
            if request.estimate_tokens(candidate) <= budget:
                selection = candidate

        return selection
