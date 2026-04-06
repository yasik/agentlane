# pylint: disable=R0917, C0301

"""OpenAI Responses API client.

This module provides a native OpenAI client using the Responses API that
conforms to the agentlane.models interfaces. The Responses API offers
several advantages over Chat Completions:

1. Unified agentic loop: Model can call multiple tools in one request
2. Built-in tools: web_search, file_search, code_interpreter
3. First-class reasoning: ResponseReasoningItem with summary and full content
4. Better cache utilization: 40% to 80% better cache hit rates
"""

import asyncio
from dataclasses import fields, replace
from typing import Any, Literal, cast

import structlog
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response as OpenAIResponse
from pydantic import BaseModel

from agentlane.models import (
    Config,
)
from agentlane.models import Factory as BaseFactory
from agentlane.models import (
    MessageDict,
    Model,
    ModelResponse,
    ModelTracing,
    OutputSchema,
    RetryMetrics,
    ToolCall,
    ToolCallFunction,
    Tools,
    has_escape_sequence_explosion,
    is_retryable_by_status_code,
    parse_content_filter_block,
    resolve_output_schema,
    retry_on_errors,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span, generation_span

from .types import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)

LOGGER = structlog.get_logger(log_tag="agentlane.openai.responses_client")
_MAX_REFUSAL_RETRIES = 3


def _is_openai_retryable(exception: BaseException) -> bool:
    """Determine if an exception from OpenAI should trigger a retry.

    Checks for OpenAI-specific exception types and falls back to
    HTTP status code checking.

    Args:
        exception: The exception to check.

    Returns:
        True if the exception should trigger a retry.
    """
    if isinstance(
        exception,
        (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ),
    ):
        return True

    return is_retryable_by_status_code(exception)


def _parse_model_provider(model: str) -> tuple[str, str]:
    """Parse provider prefix from model name.

    Supports LiteLLM-style model naming:
    - "openai/gpt-5.1" -> ("openai", "gpt-5.1")
    - "azure/gpt-5.1" -> ("azure", "gpt-5.1")
    - "gpt-5.1" -> ("openai", "gpt-5.1")  # default to openai

    Args:
        model: The model name, optionally with provider prefix.

    Returns:
        Tuple of (provider, model_name).
    """
    if "/" in model:
        provider, model_name = model.split("/", 1)
        provider = provider.lower()
        if provider in ("openai", "azure"):
            return provider, model_name
    return "openai", model


type TResponseType = ModelResponse
"""Type alias for the return type of the client.

Uses OpenAI's ChatCompletion type as our standard to maintain
interchangeability with the LiteLLM client.
"""

type ResponseInputItem = dict[str, Any]
type ResponseContentPart = dict[str, Any]


def _extract_message_content(item: ResponseOutputMessage) -> str | None:
    """Extract text content from a ResponseOutputMessage."""
    texts = [
        content_item.text
        for content_item in item.content
        if content_item.type == "output_text"
    ]
    return "".join(texts) if texts else None


def _convert_tool_call(item: ResponseFunctionToolCall) -> ChatCompletionMessageToolCall:
    """Convert ResponseFunctionToolCall to ChatCompletionMessageToolCall."""
    return ChatCompletionMessageToolCall(
        id=item.call_id,
        type="function",
        function=Function(
            name=item.name,
            arguments=item.arguments,
        ),
    )


def _append_refusal_context(
    input_data: list[dict[str, Any]],
    refusal_context: str,
) -> list[dict[str, Any]]:
    """Return a copy of input_data with a refusal-context user message appended.

    Used to retry Azure content-filter blocks by providing additional context
    that may help the model produce an acceptable response.

    Args:
        input_data: The conversation input messages (after _messages_to_input conversion).
        refusal_context: The context message to append.

    Returns:
        A copy with a new user message containing the refusal context.
    """
    messages_with_context = input_data.copy()
    messages_with_context.append(
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": refusal_context,
                }
            ],
        }
    )
    return messages_with_context


def response_to_model_response(response: OpenAIResponse) -> ModelResponse:
    """Convert OpenAI Responses API Response to Chat Completions format.

    This enables interchangeability between ResponsesClient and LiteLLM client
    by converting to our canonical ModelResponse (ChatCompletion) format.

    Args:
        response: The OpenAI Responses API Response object.

    Returns:
        ModelResponse in Chat Completions format.
    """
    content: str | None = None
    tool_calls: list[ChatCompletionMessageToolCall] = []
    reasoning_item: ResponseReasoningItem | None = None

    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            content = _extract_message_content(item) or content
        elif isinstance(item, ResponseFunctionToolCall):
            tool_calls.append(_convert_tool_call(item))
        elif isinstance(item, ResponseReasoningItem):
            reasoning_item = item

    # Determine finish reason
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
    elif response.status == "incomplete":
        finish_reason = "length"

    # Build the message - cast to satisfy type checker
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls if tool_calls else None,  # type: ignore[arg-type]
    )

    # Build usage
    usage = None
    if response.usage:
        usage = CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

    # Build the choice
    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
    )

    # Create the ChatCompletion response
    model_response = ChatCompletion(
        id=response.id,
        choices=[choice],
        created=int(response.created_at),
        model=response.model,
        object="chat.completion",
        usage=usage,
    )

    # Store reasoning item as an extension field if present (raw ResponseReasoningItem)
    if reasoning_item:
        cast(Any, model_response).reasoning_content = reasoning_item

    return model_response


class ResponsesFactory(BaseFactory[TResponseType]):
    """Factory for creating OpenAI Responses API clients.

    All clients created using this factory will share the same common
    configuration, following the same pattern as the LiteLLM factory.
    """

    def get_model_client(
        self, tracing: ModelTracing = ModelTracing.DISABLED, **kwargs: Any
    ) -> Model[TResponseType]:
        """Get a client for the OpenAI Responses API.

        Args:
            tracing: The tracing mode to use for the client.
            kwargs: Additional keyword arguments. Keys matching Config fields
                override the default config; remaining keys are forwarded as
                default model-call args.
        """

        config = replace(self._default_config, tracing=tracing)

        if kwargs:
            config_field_names = {f.name for f in fields(Config)}
            config_overrides = {
                k: v for k, v in kwargs.items() if k in config_field_names
            }
            if config_overrides:
                config = replace(config, **config_overrides)
                kwargs = {
                    k: v for k, v in kwargs.items() if k not in config_field_names
                }

        return ResponsesClient(config, **kwargs)


class ResponsesClient(Model[TResponseType]):
    """OpenAI Responses API client.

    A native OpenAI client using the Responses API that conforms to the
    Model[ModelResponse] interface for interchangeability with LiteLLM.

    Example:
        ```python
        config = Config(
            api_key="sk-...",
            model="gpt-5.1",
        )
        factory = ResponsesFactory(config)
        client = factory.get_model_client(tracing=ModelTracing.ENABLED)

        messages = [{"role": "user", "content": "Hello!"}]
        response = await client.get_response(messages)
        ```
    """

    def __init__(
        self,
        config: Config,
        refusal_context: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            config: The configuration for the client.
            refusal_context: Optional message appended to retry requests when
                Azure content filters block a response.  When ``None``, no
                refusal-retry context is added.
            kwargs: Default model-call arguments forwarded to the Responses API.
        """
        if "temperature" in kwargs and "reasoning_effort" in kwargs:
            raise ValueError(
                "Either temperature or reasoning_effort must be provided, not both."
            )

        # Parse provider prefix from model name
        # (e.g., "azure/gpt-5.1" -> ("azure", "gpt-5.1"))
        provider, model_name = _parse_model_provider(config.model)
        self._provider = provider
        self._model = model_name
        self._enforce_structured_output = config.enforce_structured_output
        self._tracing = config.tracing
        self._trace_settings = config.to_trace_settings()
        self._rate_limiter = config.rate_limiter
        self._max_retries = config.max_retries
        self._schema_validation_retries = config.schema_validation_retries
        self._refusal_context = refusal_context

        # Build common client parameters
        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": 0,  # We handle retries ourselves
        }
        if config.organization:
            client_kwargs["organization"] = config.organization
        if config.default_headers:
            client_kwargs["default_headers"] = config.default_headers

        # Create the appropriate OpenAI client based on provider
        if provider == "azure":
            if not config.base_url:
                raise ValueError(
                    "base_url is required for Azure OpenAI (e.g., "
                    "'https://your-resource.openai.azure.com/')"
                )
            client_kwargs["azure_endpoint"] = config.base_url
            client_kwargs["api_version"] = (
                "2025-03-01-preview"  # Responses API requires preview
            )
            self._openai_client: AsyncOpenAI = AsyncAzureOpenAI(**client_kwargs)
        else:
            if config.base_url:
                client_kwargs["base_url"] = config.base_url
            self._openai_client = AsyncOpenAI(**client_kwargs)

        # Common parameters for API calls
        self._common_params: dict[str, Any] = dict(kwargs)

    async def get_response(  # pylint: disable=R0912
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        **_kwargs: Any,
    ) -> TResponseType:
        """Get a response from the OpenAI Responses API.

        Args:
            messages: List of messages in OpenAI Chat Completions format.
                These are converted to Responses API input format.
            extra_call_args: Additional arguments to pass to the API call.
            schema: Optional Pydantic model or OutputSchema for structured output.
            tools: Optional tools configuration for function calling.
            cancellation_token: Optional cancellation token.

        Returns:
            ModelResponse in Chat Completions format for interchangeability.
        """
        # Convert messages to Responses API input format
        conversation_input = self._messages_to_input(messages)
        schema_retry_count = 0

        with generation_span(
            model=self._model,
            model_config=self._trace_settings,
            disabled=self._tracing.is_disabled(),
        ) as span_generation:
            if self._tracing.include_data():
                span_generation.span_data.input = messages

            while True:
                call_args = self._build_call_args(
                    extra_call_args=extra_call_args,
                    schema=schema,
                    tools=tools,
                )

                LOGGER.debug(
                    "Responses API call started",
                    input=conversation_input,
                    call_args=call_args,
                )

                if self._provider == "azure":
                    (
                        result,
                        retry_metrics,
                    ) = await self._execute_with_azure_refusal_retry(
                        input_data=conversation_input,
                        call_args=call_args,
                        cancellation_token=cancellation_token,
                    )
                else:
                    result, retry_metrics = await self._execute_with_retry(
                        input_data=conversation_input,
                        call_args=call_args,
                        cancellation_token=cancellation_token,
                    )

                LOGGER.debug(
                    "Responses API call finished",
                    result=result,
                    retry_metrics=retry_metrics,
                )

                model_response = response_to_model_response(result)
                tool_calls = self._extract_tool_calls(result)

                if tool_calls:
                    # Tool calls are returned as raw model output.
                    self._record_tracing_output(span_generation, result)
                    self._record_usage(span_generation, result, retry_metrics)
                    return model_response

                # Check for escape sequence explosion (e.g. Gemini combining char runs)
                if has_escape_sequence_explosion(model_response):
                    schema_retry_count += 1
                    if schema_retry_count <= self._schema_validation_retries:
                        conversation_input.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": (
                                            "Your previous response contained repeated "
                                            "Unicode escape sequences and could not be "
                                            "processed. Please regenerate your response "
                                            "without using repeated combining diacritical "
                                            "marks or \\uXXXX sequences."
                                        ),
                                    }
                                ],
                            }
                        )
                        continue

                # Validate structured output if schema is provided
                if schema is not None:
                    output_schema = resolve_output_schema(schema)
                    if output_schema is None:
                        raise ValueError(
                            "schema normalization unexpectedly returned None"
                        )

                    validation_result = output_schema.validate_response_with_retry(
                        response=model_response,
                        retry_count=schema_retry_count,
                        max_retries=self._schema_validation_retries,
                        enforce_structured_output=self._enforce_structured_output,
                    )

                    if validation_result.should_retry:
                        schema_retry_count += 1
                        # Add error feedback message in Responses API format
                        conversation_input.append(
                            {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": validation_result.error_message,
                                    }
                                ],
                            }
                        )
                        continue  # Retry with augmented conversation
                    # Fall through if valid or no more retries

                # Record tracing and return
                self._record_tracing_output(span_generation, result)
                self._record_usage(span_generation, result, retry_metrics)

                return model_response

    def get_model(self) -> str:
        """Get the model name."""
        return self._model

    def get_is_enforce_structured_output(self) -> bool:
        """Get the enforce_structured_output flag."""
        return self._enforce_structured_output

    def _messages_to_input(
        self, messages: list[MessageDict]
    ) -> list[ResponseInputItem]:
        """Convert Chat Completions messages to Responses API input format.

        The Responses API accepts a more flexible input format. We convert
        standard messages to ResponseInputItem format.

        Args:
            messages: Messages in Chat Completions format.

        Returns:
            List of input items for Responses API.
        """
        result: list[ResponseInputItem] = []

        for msg in messages:
            role_value = msg.get("role", "user")
            role = role_value if isinstance(role_value, str) else "user"
            content = msg.get("content", "")

            if role == "system":
                # System messages become instructions in Responses API
                # We'll handle this separately in the API call
                result.append(
                    {
                        "type": "message",
                        "role": "developer",  # developer role for system messages
                        "content": self._convert_content(content),
                    }
                )
            elif role == "user":
                result.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": self._convert_content(content),
                    }
                )
            elif role == "assistant":
                assistant_content: list[ResponseContentPart] = []
                assistant_item: ResponseInputItem = {
                    "type": "message",
                    "role": "assistant",
                    "content": assistant_content,
                }
                if isinstance(content, str) and content:
                    assistant_content.append(
                        {
                            "type": "output_text",
                            "text": content,
                        }
                    )
                # Append assistant message first (before function calls)
                if assistant_item["content"]:
                    result.append(assistant_item)
                # Handle tool calls in assistant messages (after message)
                tool_calls_raw = msg.get("tool_calls", [])
                if isinstance(tool_calls_raw, list):
                    for tc in cast(list[object], tool_calls_raw):
                        if not isinstance(tc, dict):
                            continue
                        tc_dict = cast(dict[str, Any], tc)
                        function_payload = tc_dict.get("function", {})
                        function_dict = (
                            cast(dict[str, Any], function_payload)
                            if isinstance(function_payload, dict)
                            else cast(dict[str, Any], {})
                        )
                        result.append(
                            {
                                "type": "function_call",
                                "call_id": tc_dict.get("id", ""),
                                "name": function_dict.get("name", ""),
                                "arguments": function_dict.get("arguments", "{}"),
                            }
                        )
            elif role == "tool":
                # Tool results
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": msg.get("content", ""),
                    }
                )

        return result

    def _convert_content(self, content: object) -> list[ResponseContentPart]:
        """Convert message content to Responses API format.

        Handles both string content and multi-part content (list of parts).

        Args:
            content: Either a string or a list of content parts.

        Returns:
            List of content parts in Responses API format.
        """
        if isinstance(content, str):
            return [{"type": "input_text", "text": content}]

        if not isinstance(content, list):
            return [{"type": "input_text", "text": str(content)}]

        # Multi-part content - convert each part
        result: list[ResponseContentPart] = []
        for part in cast(list[object], content):
            if not isinstance(part, dict):
                continue
            part_dict = cast(dict[str, Any], part)
            part_type = part_dict.get("type", "")
            if part_type == "text":
                result.append({"type": "input_text", "text": part_dict.get("text", "")})
            elif part_type == "input_text":
                # Already in correct format
                result.append(part_dict)
            elif part_type == "input_image":
                # Already in correct format
                result.append(part_dict)
            elif part_type == "file":
                # FilePart format - pass through for PDF support
                result.append(part_dict)
        return result

    def _build_call_args(
        self,
        extra_call_args: dict[str, Any] | None,
        schema: type[BaseModel] | OutputSchema[Any] | None,
        tools: Tools | None,
    ) -> dict[str, Any]:
        """Build the arguments for the Responses API call."""
        call_args: dict[str, Any] = {**self._common_params}

        if extra_call_args:
            call_args.update(extra_call_args)

        # Handle structured output via response_format / text parameter
        if schema is not None and self._enforce_structured_output:
            output_schema = resolve_output_schema(schema)
            if output_schema is not None:
                response_format = output_schema.response_format()
                # Convert Chat Completions response_format to Responses API text format
                if isinstance(response_format, dict):
                    json_schema = response_format.get("json_schema", {})
                    call_args["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": json_schema.get("name", "response"),
                            "schema": json_schema.get("schema", {}),
                            "strict": json_schema.get("strict", True),
                        }
                    }

        # Handle tools
        if tools is not None:
            tools_payload: list[dict[str, Any]] = []
            for tool in tools.tools:
                tools_payload.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema.get("parameters", {}),
                        "strict": True,
                    }
                )
            call_args["tools"] = tools_payload

            # Map tool_choice (same values for Responses API)
            call_args["tool_choice"] = tools.tool_choice
            call_args["parallel_tool_calls"] = tools.parallel_tool_calls

        # Convert max_tokens to max_output_tokens (Responses API naming)
        if "max_tokens" in call_args:
            call_args["max_output_tokens"] = call_args.pop("max_tokens")

        # Convert reasoning_effort to reasoning dict (Responses API format)
        if "reasoning_effort" in call_args:
            reasoning = call_args.get("reasoning", {})
            reasoning["effort"] = call_args.pop("reasoning_effort")
            call_args["reasoning"] = reasoning

        # Convert verbosity to text config (Responses API format)
        # verbosity -> text.verbosity (low, medium, high)
        if "verbosity" in call_args:
            text = call_args.get("text", {})
            text["verbosity"] = call_args.pop("verbosity")
            call_args["text"] = text

        return call_args

    def _extract_tool_calls(self, result: OpenAIResponse) -> list[ToolCall]:
        """Extract tool calls from Responses API response.

        Converts ResponseFunctionToolCall to our standard ToolCall format.
        """
        tool_calls: list[ToolCall] = []

        for item in result.output:
            if isinstance(item, ResponseFunctionToolCall):
                tool_call = ToolCall(
                    id=item.call_id,
                    type="function",
                    function=ToolCallFunction(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
                tool_calls.append(tool_call)

        return tool_calls

    async def _execute_with_retry(
        self,
        *,
        input_data: list[dict[str, Any]],
        call_args: dict[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> tuple[OpenAIResponse, RetryMetrics]:
        """Execute Responses API call with retry logic.

        Returns:
            A tuple of (response, retry_metrics).
        """

        @retry_on_errors(
            max_retries=self._max_retries, is_retryable=_is_openai_retryable
        )
        async def _api_call() -> OpenAIResponse:
            if self._rate_limiter is not None:
                async with self._rate_limiter:
                    result = cast(  # type: ignore[redundant-cast]
                        OpenAIResponse,
                        await self._openai_client.responses.create(
                            model=self._model,
                            input=input_data,  # type: ignore[arg-type]
                            **call_args,
                        ),
                    )
                    # Record token usage for TPM limiters
                    if result.usage is not None:
                        self._rate_limiter.record_usage(result.usage.total_tokens)
                    return result

            return cast(  # type: ignore[redundant-cast]
                OpenAIResponse,
                await self._openai_client.responses.create(
                    model=self._model,
                    input=input_data,  # type: ignore[arg-type]
                    **call_args,
                ),
            )

        if cancellation_token is None:
            retry_result = await _api_call()
            return retry_result.result, retry_result.metrics

        future = asyncio.ensure_future(_api_call())
        cancellation_token.link_future(future)
        retry_result = await future
        return retry_result.result, retry_result.metrics

    async def _execute_with_azure_refusal_retry(
        self,
        *,
        input_data: list[dict[str, Any]],
        call_args: dict[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> tuple[OpenAIResponse, RetryMetrics]:
        """Execute API call with retry on Azure content filter blocks.

        When a ``refusal_context`` was provided at construction, checks
        responses for Azure content filters and retries with the context
        message appended up to ``_MAX_REFUSAL_RETRIES`` times.

        Returns:
            A tuple of (response, retry_metrics).
        """
        response, metrics = await self._execute_with_retry(
            input_data=input_data,
            call_args=call_args,
            cancellation_token=cancellation_token,
        )

        if self._refusal_context is None:
            return response, metrics

        for attempt in range(_MAX_REFUSAL_RETRIES):
            # Check for Azure content filters in response
            response_dict = response.model_dump()
            block_msg = parse_content_filter_block(response_dict)

            if block_msg is None:
                return response, metrics

            LOGGER.warning(
                "model output blocked by content filter, retrying with refusal context",
                block_reason=block_msg[:100],
                attempt=attempt + 1,
                max_attempts=_MAX_REFUSAL_RETRIES,
            )
            response, metrics = await self._execute_with_retry(
                input_data=_append_refusal_context(input_data, self._refusal_context),
                call_args=call_args,
                cancellation_token=cancellation_token,
            )

        return response, metrics

    def _record_tracing_output(
        self,
        span_generation: Span[Any],
        result: OpenAIResponse,
    ) -> None:
        """Persist the model output in tracing when enabled."""
        if not self._tracing.include_data():
            return

        output_data = [result.model_dump(mode="json")]
        span_generation.span_data.output = output_data

    def _record_usage(
        self,
        span_generation: Span[Any],
        result: OpenAIResponse,
        retry_metrics: RetryMetrics | None = None,
    ) -> None:
        """Record token usage and retry information when available."""
        usage = getattr(result, "usage", None)
        if usage is None:
            span_generation.span_data.usage = {}
        else:
            span_generation.span_data.usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }

            # Add cache metrics for monitoring cache effectiveness
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
                span_generation.span_data.usage["cached_tokens"] = cached_tokens
                # Calculate cache hit rate as a percentage
                if usage.input_tokens > 0:
                    cache_hit_rate = (cached_tokens / usage.input_tokens) * 100
                    span_generation.span_data.usage["cache_hit_rate_pct"] = round(
                        cache_hit_rate, 2
                    )

        # Add retry metrics
        if retry_metrics is not None:
            span_generation.span_data.usage["attempts"] = retry_metrics.attempts
            span_generation.span_data.usage["retry_wait"] = retry_metrics.retry_wait
            span_generation.span_data.usage["backend_latency"] = (
                retry_metrics.backend_latency
            )
