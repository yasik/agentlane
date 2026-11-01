"""Model-backed relevance ranking for conversation preservation."""

import json

from pydantic import BaseModel, StrictInt, ValidationError

from agentlane.models import Model, ModelResponse, get_content_or_none

from ._errors import CompactionError
from ._prompt import DEFAULT_RANKING_PROMPT, RANKING_INSTRUCTIONS
from ._types import PreservationRequest, RankingPayload, RankingPayloadBlock


class _Ranking(BaseModel):
    block_ids: list[StrictInt]


class ModelPreservationRanker:
    """Rank history with a model; use its configuration for limits and retries."""

    def __init__(
        self,
        model: Model[ModelResponse],
        *,
        prompt: str = DEFAULT_RANKING_PROMPT,
    ) -> None:
        """Use a relevance policy prompt while keeping the fixed output contract."""
        self._model = model
        self._prompt = prompt

    async def rank(self, request: PreservationRequest, /) -> list[int]:
        """Return ranked block IDs for the selector to validate and budget."""
        if not any(block.retainable and not block.required for block in request.blocks):
            return []

        # Old summaries still explain the task even though they cannot be kept.
        # Include all blocks once and mark which IDs the selector can retain.
        blocks: list[RankingPayloadBlock] = [
            {
                "id": block.id,
                "required": block.required,
                "retainable": block.retainable,
                "messages": block.messages,
            }
            for block in request.blocks
        ]
        messages = [message for block in blocks for message in block["messages"]]
        required_messages = tuple(
            message for message in request.required_messages if message not in messages
        )
        messages.extend(required_messages)

        # Owner-provided representations establish current state. Deduplication
        # here does not remove state from the selector's later retention decision.
        payload: RankingPayload = {
            "instructions": request.instructions,
            "required_messages": required_messages,
            "current_state": [
                item.content
                for item in request.default.context
                if not any(message in messages for message in item.represented_by)
            ],
            "blocks": blocks,
        }
        response = await self._model(
            [
                {
                    "role": "system",
                    "content": f"{RANKING_INSTRUCTIONS}\n\nRanking policy:\n{self._prompt}",
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            schema=_Ranking,
        )

        # An interrupted response can contain valid JSON but an incomplete
        # ranking. Reject it before the compactor summarizes or replaces history.
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise CompactionError("Model ranker returned an incomplete response.")

        try:
            ranking = _Ranking.model_validate_json(get_content_or_none(response) or "")
        except ValidationError as error:
            raise CompactionError(
                "Model ranker returned an invalid ranking."
            ) from error

        return ranking.block_ids
