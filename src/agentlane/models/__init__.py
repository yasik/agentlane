"""Core LLM-facing primitives for AgentLane."""

from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from ._exceptions import ModelBehaviorError, ModelsException, RunErrorDetails
from ._interface import Config, Factory, MessageDict, Model, ModelTracing, Tools
from ._json_repair import parse_json_dict
from ._output_schema import OutputSchema, SchemaValidationResult, resolve_output_schema
from ._prompts import (
    FilePart,
    ImagePart,
    MultiPartPromptTemplate,
    PromptSpec,
    PromptTemplate,
    PromptTemplateBase,
    TextPart,
)
from ._rate_limiter import (
    CompositeRateLimiter,
    ConcurrentRequestLimiter,
    RateLimiter,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
)
from ._response_utils import (
    ReasoningContent,
    get_content_or_none,
    get_json_dict_or_none,
    get_reasoning_content_or_none,
    get_search_results_or_none,
    has_escape_sequence_explosion,
    parse_content_filter_block,
)
from ._retry import (
    DEFAULT_RETRY_STATUS_CODES,
    RetryMetrics,
    RetryResult,
    extract_retry_after,
    is_retryable_by_status_code,
    retry_on_errors,
    wait_with_retry_after,
)
from ._strict_schema import ensure_strict_json_schema
from ._tool import Tool, as_tool
from ._tool_executor import ToolExecutor
from ._tool_output_adapter import ChatCompletionsOutputAdapter, ToolOutputAdapter
from ._types import (
    Choice,
    ChoiceLogprobs,
    Message,
    ModelResponse,
    ToolCall,
    ToolCallFunction,
    Usage,
)

__all__ = [
    "ModelsException",
    "ModelBehaviorError",
    "RunErrorDetails",
    "parse_json_dict",
    "OutputSchema",
    "SchemaValidationResult",
    "resolve_output_schema",
    "get_content_or_none",
    "get_json_dict_or_none",
    "get_reasoning_content_or_none",
    "get_search_results_or_none",
    "has_escape_sequence_explosion",
    "parse_content_filter_block",
    "ReasoningContent",
    "ResponseReasoningItem",
    "PromptTemplate",
    "PromptTemplateBase",
    "MultiPartPromptTemplate",
    "PromptSpec",
    "TextPart",
    "FilePart",
    "ImagePart",
    "ensure_strict_json_schema",
    "Model",
    "Config",
    "Factory",
    "MessageDict",
    "ModelTracing",
    "Tool",
    "as_tool",
    "Tools",
    "RateLimiter",
    "SlidingWindowRateLimiter",
    "ConcurrentRequestLimiter",
    "CompositeRateLimiter",
    "TokenBucketRateLimiter",
    "RetryMetrics",
    "RetryResult",
    "retry_on_errors",
    "extract_retry_after",
    "wait_with_retry_after",
    "is_retryable_by_status_code",
    "DEFAULT_RETRY_STATUS_CODES",
    "ToolExecutor",
    "ToolOutputAdapter",
    "ChatCompletionsOutputAdapter",
    "ModelResponse",
    "Message",
    "Choice",
    "ChoiceLogprobs",
    "ToolCall",
    "ToolCallFunction",
    "Usage",
]
