# pylint: disable=W2301
"""Tool output adapter protocol and default implementation.

This module provides the protocol that adapters must implement to format
tool execution output for different LLM API formats. The default
ChatCompletionsOutputAdapter is included here to avoid circular imports.
Other implementations (like ResponsesApiOutputAdapter) live in their
respective client packages.
"""

from typing import Any, Protocol


class ToolOutputAdapter(Protocol):
    """Protocol for adapting tool execution output to client-specific formats."""

    def format_success(
        self, call_id: str, tool_name: str, content: str
    ) -> dict[str, Any]:
        """Format successful tool execution output.

        Args:
            call_id: The unique identifier for the tool call.
            tool_name: The name of the tool that was executed.
            content: The string content returned by the tool.

        Returns:
            A dictionary formatted for the specific LLM API.
        """
        ...

    def format_error(
        self, call_id: str, tool_name: str, error_message: str
    ) -> dict[str, Any]:
        """Format tool execution error output.

        Args:
            call_id: The unique identifier for the tool call.
            tool_name: The name of the tool that failed.
            error_message: A description of the error.

        Returns:
            A dictionary formatted for the specific LLM API.
        """
        ...


class ChatCompletionsOutputAdapter:
    """Adapter for Chat Completions API format (LiteLLM client).

    Formats tool outputs in the standard Chat Completions format used by
    OpenAI's Chat Completions API and LiteLLM.
    """

    def format_success(
        self, call_id: str, tool_name: str, content: str
    ) -> dict[str, Any]:
        """Format successful tool execution for Chat Completions API."""
        return {
            "tool_call_id": call_id,
            "role": "tool",
            "name": tool_name,
            "content": content,
        }

    def format_error(
        self, call_id: str, tool_name: str, error_message: str
    ) -> dict[str, Any]:
        """Format error output for Chat Completions API."""
        return {
            "tool_call_id": call_id,
            "role": "tool",
            "name": tool_name,
            "content": error_message,
        }
