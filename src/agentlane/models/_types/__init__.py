"""Type aliases for LLM responses.

We use OpenAI Chat Completions types as our baseline. This is the
industry standard format supported by virtually all providers.
By defining aliases, we leave room for future forking if needed.
"""

from openai.types.chat import ChatCompletion as ModelResponse
from openai.types.chat import ChatCompletionMessage as Message
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as ToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage as Usage

__all__ = [
    # Main response type
    "ModelResponse",  # ChatCompletion - main response with id, choices, model, usage
    # Response components
    "Choice",  # Choice - finish_reason, index, message, logprobs
    "ChoiceLogprobs",  # Logprobs information for a choice
    "Message",  # ChatCompletionMessage - content, role, tool_calls
    # Tool call types
    "ToolCall",  # ChatCompletionMessageToolCall - id, function, type
    "ToolCallFunction",  # Function - name, arguments
    # Usage
    "Usage",  # CompletionUsage - prompt_tokens, completion_tokens, total_tokens
]
