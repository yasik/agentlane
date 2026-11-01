"""Core LLM-facing primitives for AgentLane."""

from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from ._constants import ESCAPE_SEQUENCE_RETRY_MESSAGE, SCHEMA_NORMALIZATION_ERROR
from ._exceptions import ModelBehaviorError, ModelsException, RunErrorDetails
from ._interface import (
    Config,
    Factory,
    MessageDict,
    Model,
    ModelTracing,
    ToolChoice,
    Tools,
)
from ._json_repair import parse_json_dict
from ._message_metadata import (
    internal_message,
    is_ephemeral_message,
    is_internal_message,
    provider_messages,
)
from ._message_role import MessageRole
from ._output_schema import OutputSchema, SchemaValidationResult, resolve_output_schema
from ._prompts import (
    FilePart,
    ImagePart,
    MultiPartPromptTemplate,
    PromptSpec,
    PromptTemplate,
    PromptTemplateBase,
    TextPart,
    render_instruction_text,
)
from ._provider_payloads import (
    ReasoningPhase,
    UsageTotals,
    get_reasoning_phase,
    get_usage_totals,
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
from ._streaming import (
    ModelContinuation,
    ModelStreamEvent,
    ModelStreamEventKind,
    SteeringInput,
    SteeringReceipt,
    SteeringStatus,
)
from ._strict_schema import ensure_strict_json_schema
from ._tool import (
    RunStateView,
    Tool,
    ToolExecutionContext,
    ToolHandler,
    ToolSpec,
    as_tool,
)
from ._tool_executor import ToolExecutor
from ._tool_output_adapter import ChatCompletionsOutputAdapter, ToolOutputAdapter
from ._tool_result import (
    PlanStepState,
    PlanStepStatus,
    ToolError,
    ToolErrorKind,
    ToolFailure,
)
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
    "internal_message",
    "is_ephemeral_message",
    "is_internal_message",
    "provider_messages",
    "ESCAPE_SEQUENCE_RETRY_MESSAGE",
    "SCHEMA_NORMALIZATION_ERROR",
    "ToolChoice",
    "SteeringStatus",
    "ModelContinuation",
    "ToolErrorKind",
    "PlanStepState",
    "SteeringInput",
    "SteeringReceipt",
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
    "ModelStreamEvent",
    "ModelStreamEventKind",
    "PromptTemplate",
    "PromptTemplateBase",
    "MultiPartPromptTemplate",
    "PromptSpec",
    "render_instruction_text",
    "TextPart",
    "FilePart",
    "ImagePart",
    "ReasoningPhase",
    "UsageTotals",
    "get_reasoning_phase",
    "get_usage_totals",
    "ensure_strict_json_schema",
    "Model",
    "Config",
    "Factory",
    "MessageDict",
    "MessageRole",
    "ModelTracing",
    "ToolSpec",
    "Tool",
    "ToolExecutionContext",
    "ToolHandler",
    "RunStateView",
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
    "ToolError",
    "ToolFailure",
    "PlanStepStatus",
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
