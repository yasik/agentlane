from typing import cast

from agentlane.agents import Agent, is_on_message_handler, on_message
from agentlane.messaging import MessageContext


class _EchoAgent:
    @on_message
    async def process(self, payload: object, context: MessageContext) -> object:
        _ = context
        return payload


def test_agent_protocol_shape() -> None:
    _ = cast(Agent, _EchoAgent())
    assert is_on_message_handler(_EchoAgent.process)
