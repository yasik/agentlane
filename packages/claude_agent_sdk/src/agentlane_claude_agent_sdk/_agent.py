"""AgentLane task backed by one-shot Claude Agent SDK queries."""

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from agentlane.harness import Task
from agentlane.runtime import Engine, MessageContext, on_message


class ClaudeAgent(Task):
    """Run one fresh Claude Agent SDK query for each addressed text message."""

    def __init__(
        self,
        engine: Engine,
        options: ClaudeAgentOptions | None = None,
    ) -> None:
        """Create a Claude task with explicit or isolated default options."""
        super().__init__(engine)
        self._options = options if options is not None else _default_options()

    @on_message
    async def handle(self, payload: str, context: MessageContext) -> str:
        """Return the final successful string from one fresh Claude query."""
        del context
        _reject_continuity_options(self._options)

        result_message: ResultMessage | None = None
        async for message in query(prompt=payload, options=self._options):
            if isinstance(message, ResultMessage):
                result_message = message

        if (
            result_message is None
            or result_message.subtype != "success"
            or result_message.is_error
            or not isinstance(result_message.result, str)
        ):
            raise RuntimeError("Claude query did not return a successful string.")

        return result_message.result


def _default_options() -> ClaudeAgentOptions:
    """Return options that do not load tools or local configuration."""
    return ClaudeAgentOptions(
        tools=[],
        setting_sources=[],
        skills=[],
        mcp_servers={},
        strict_mcp_config=True,
        max_turns=1,
    )


def _reject_continuity_options(options: ClaudeAgentOptions) -> None:
    """Reject options that can load or continue an earlier Claude session."""
    if (
        options.continue_conversation
        or options.resume is not None
        or options.fork_session
        or options.resume_session_at is not None
        or options.resume_drops_turn is not None
    ):
        raise ValueError("ClaudeAgent options must start a fresh session.")
