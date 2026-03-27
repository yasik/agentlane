# pylint: disable=W0613
"""Tool output adapter for OpenAI Responses API format."""

from typing import Any

from agentlane.models import ToolOutputAdapter


class ResponsesApiOutputAdapter(ToolOutputAdapter):
    """Adapter for OpenAI Responses API format.

    Formats tool outputs in the format expected by OpenAI's Responses API,
    which uses a different structure than Chat Completions.

    Note: tool_name is accepted but unused as the Responses API format
    doesn't include the tool name in the output structure.
    """

    def format_success(
        self, call_id: str, tool_name: str, content: str
    ) -> dict[str, Any]:
        """Format successful tool execution for Responses API."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": content,
        }

    def format_error(
        self, call_id: str, tool_name: str, error_message: str
    ) -> dict[str, Any]:
        """Format error output for Responses API."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": error_message,
        }
