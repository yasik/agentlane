"""Type aliases for OpenAI Responses API.

We re-export OpenAI types following the same pattern as
diadiax.agents.models._types to maintain consistency.
"""

# Responses API types (non-streaming)
from openai.types.responses import Response as ResponsesAPIResponse
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)

# Function tool call output (for tool results)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_input_param import ResponseInputParam

# Output item types
from openai.types.responses.response_output_item import ResponseOutputItem
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.response_usage import ResponseUsage

# Tool parameter types
from openai.types.responses.tool_param import ToolParam

__all__ = [
    # Response types
    "ResponsesAPIResponse",
    "ResponsesToolChoice",
    "ResponseCreateParamsNonStreaming",
    "ResponseInputParam",
    "ResponseUsage",
    # Output items
    "ResponseOutputItem",
    "ResponseOutputMessage",
    "ResponseOutputText",
    "ResponseOutputRefusal",
    "ResponseFunctionToolCall",
    "ResponseReasoningItem",
    # Tool types
    "ToolParam",
    "FunctionToolParam",
    "ResponseFunctionToolCallOutputItem",
]
