# pylint: disable=R0917, C0301

import asyncio
from dataclasses import fields, replace
from typing import Any, cast

import litellm
import structlog
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
    Tools,
    has_escape_sequence_explosion,
    is_retryable_by_status_code,
    resolve_output_schema,
    retry_on_errors,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span, generation_span

from .types import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

LOGGER = structlog.get_logger(log_tag="agentlane.litellm.client")

_litellm_transport_configured = False


def _is_litellm_retryable(exception: BaseException) -> bool:
    """Determine if an exception from LiteLLM should trigger a retry.

    Checks for LiteLLM-specific exception types and falls back to
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
            ServiceUnavailableError,
            InternalServerError,
            RateLimitError,
            Timeout,
        ),
    ):
        return True

    return is_retryable_by_status_code(exception)


type TResponseType = ModelResponse
"""Type alias for the return type of the LLM client.

This uses OpenAI's ChatCompletion type as our standard. LiteLLM returns
a structurally compatible response that can be used interchangeably.
"""

type ResponseFormat = dict[str, Any] | type[BaseModel]


async def _litellm_acompletion(**kwargs: Any) -> ModelResponse:
    """Call LiteLLM's async completion API with a typed return value."""
    acompletion: Any = getattr(litellm, "acompletion")  # noqa: B009
    return cast(ModelResponse, await acompletion(**kwargs))


class Factory(BaseFactory[TResponseType]):
    """Factory for creating LLM clients of the same model class. All
    clients created using this factory will share the same common
    configuration."""

    def __init__(self, default_config: Config):
        """Initialize the factory."""
        self._default_config = default_config

    def get_model_client(
        self, tracing: ModelTracing = ModelTracing.DISABLED, **kwargs: Any
    ) -> Model[TResponseType]:
        """Get a client for the LLM.

        Args:
            tracing: The tracing mode to use for the client.
            kwargs: Additional keyword arguments to pass to the client. Any keys that
                match fields of ``Config`` will override the default config via dataclass
                replace; remaining keys are forwarded as default model-call args.
        """
        config = replace(self._default_config, tracing=tracing)

        if kwargs:
            config_field_names = {f.name for f in fields(Config)}
            config_overrides = {
                k: v for k, v in kwargs.items() if k in config_field_names
            }
            if config_overrides:
                config = replace(config, **config_overrides)
                # Remove config keys so only non-config extras are passed to the client
                kwargs = {
                    k: v for k, v in kwargs.items() if k not in config_field_names
                }

        return Client(config, **kwargs)


class Client(Model[TResponseType]):
    """
    A thread-safe reusable client wrapper for LiteLLM that allows setting the
    model and other key parameters once at initialization, instead of passing
    them on each call.

    The same parameters passed at the call time will take precedence over the
    parameters set at initialization.
    """

    def __init__(
        self,
        config: Config,
        **kwargs: Any,
    ) -> None:
        """Initialize the client with a specific model.

        Args:
            config: The configuration to use for the client
            kwargs: Default model-call arguments forwarded to LiteLLM.
        """
        # Disable aiohttp transport which leads to sessions being created for
        # each call and never closed properly.  Forces litellm to use httpx.
        global _litellm_transport_configured  # noqa: PLW0603
        if not _litellm_transport_configured:
            litellm.use_aiohttp_transport = False
            litellm.disable_aiohttp_transport = True
            _litellm_transport_configured = True

        if "temperature" in kwargs and "reasoning_effort" in kwargs:
            raise ValueError(
                "Either temperature or reasoning_effort must be provided, not both."
            )

        self._model = config.model
        self._enforce_structured_output = config.enforce_structured_output
        self._tracing = config.tracing
        self._trace_settings = config.to_trace_settings()
        self._rate_limiter = config.rate_limiter
        self._max_retries = config.max_retries
        self._schema_validation_retries = config.schema_validation_retries
        # Common parameters for both clients
        # Note: max_retries/num_retries=0 disables litellm's internal retry.
        # We handle retries ourselves via the retry_on_errors decorator.
        self._common_params = {
            "api_key": config.api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": 0,
            "num_retries": 0,
            "default_headers": config.default_headers,
            "api_base": config.base_url,
            **kwargs,
        }
        if config.vertex_project_id and config.vertex_location:
            self._common_params["vertex_project"] = config.vertex_project_id
            self._common_params["vertex_location"] = config.vertex_location

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        **_kwargs: Any,
    ) -> TResponseType:
        """Asynchronously call the LLM with cancellation, tooling, and schema support."""
        conversation: list[MessageDict] = [*messages]
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
                    "LLM call started",
                    messages=conversation,
                    call_args=call_args,
                )

                result, retry_metrics = await self._execute_with_retry(
                    messages=conversation,
                    call_args=call_args,
                    cancellation_token=cancellation_token,
                )

                # Normalize provider-specific usage cost payloads immediately
                # so any downstream serialization path (logging/tracing/validation)
                # sees a scalar cost value.
                self._normalize_usage_cost(result)

                LOGGER.debug(
                    "LLM call finished",
                    result=result,
                    retry_metrics=retry_metrics,
                )

                # If the result is a ModelResponse, we extract the assistant message
                # and tool calls and extend the conversation
                assistant_message = self._extract_assistant_message(result)
                if assistant_message is not None:
                    conversation.append(assistant_message)
                model_response = self._to_model_response(result)

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
                        # Remove the corrupted assistant message that was
                        # appended to avoid re-sending the escape
                        # sequence explosion to the model.
                        if conversation and conversation[-1].get("role") == "assistant":
                            conversation.pop()
                        conversation.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous response contained repeated Unicode "
                                    "escape sequences and could not be processed. Please "
                                    "regenerate your response without using repeated "
                                    "combining diacritical marks or \\uXXXX sequences."
                                ),
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
                        conversation.append(
                            {
                                "role": "user",
                                "content": validation_result.error_message,
                            }
                        )
                        continue  # Retry with augmented conversation
                    # Fall through if valid or no more retries

                # We record the tracing output and usage and return the result
                self._record_tracing_output(span_generation, result)
                self._record_usage(span_generation, result, retry_metrics)

                return model_response

    def get_model(self) -> str:
        """Get the model name."""
        return self._model

    def get_is_enforce_structured_output(self) -> bool:
        """Get the enforce_structured_output flag."""
        return self._enforce_structured_output

    def _build_call_args(
        self,
        extra_call_args: dict[str, Any] | None,
        schema: type[BaseModel] | OutputSchema[Any] | None,
        tools: Tools | None,
    ) -> dict[str, Any]:
        """Compose the kwargs for the LiteLLM call."""

        call_args: dict[str, Any] = {**self._common_params}

        if extra_call_args:
            call_args.update(extra_call_args)

        # If max_retries is greater than 0, set it to 0
        # This is to disable retries for the LiteLLM call and
        # instead use the retry_on_errors decorator to handle retries.
        if "max_retries" in call_args and call_args["max_retries"] > 0:
            call_args["max_retries"] = 0
            call_args["num_retries"] = 0

        response_format = self._schema_response_format(schema)
        if response_format is not None:
            self._validate_response_format(response_format)
            call_args["response_format"] = response_format

        if tools is not None:
            call_args.update(tools.as_args())

        call_args["drop_params"] = True
        return call_args

    def _extract_assistant_message(self, result: ModelResponse) -> MessageDict | None:
        """Return the assistant message payload from the LLM response."""

        if not result.choices:
            return None

        choice0 = result.choices[0]
        message = getattr(choice0, "message", None)
        if not message:
            return None

        msg_dict: dict[str, Any] = message.model_dump(mode="json")
        return msg_dict

    def _extract_tool_calls(self, result: ModelResponse) -> list[ToolCall]:
        """Extract tool call objects from an LLM response choice."""

        if not result.choices:
            return []

        choice0 = result.choices[0]
        message = getattr(choice0, "message", None)
        if not message or not getattr(message, "tool_calls", None):
            return []

        # Adapt LiteLLM tool calls to our ToolCall type
        return [
            ToolCall.model_validate(tc.model_dump(mode="json"))
            for tc in message.tool_calls
        ]

    def _schema_response_format(
        self,
        schema: type[BaseModel] | OutputSchema[Any] | None,
    ) -> ResponseFormat | None:
        """Prepare the response format payload when JSON schema enforcement is on."""

        if schema is None or not self.get_is_enforce_structured_output():
            return None

        output_schema = resolve_output_schema(schema)

        if output_schema is None:
            return None
        return output_schema.response_format()

    async def _execute_with_retry(
        self,
        *,
        messages: list[MessageDict],
        call_args: dict[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> tuple[TResponseType, RetryMetrics]:
        """Invoke LiteLLM with retry and optional cancellation support.

        Returns:
            A tuple of (response, retry_metrics).
        """

        @retry_on_errors(
            max_retries=self._max_retries, is_retryable=_is_litellm_retryable
        )
        async def _llm_call() -> TResponseType:
            # Use context manager to acquire and automatically release rate limiter
            if self._rate_limiter is not None:
                async with self._rate_limiter:
                    result = await _litellm_acompletion(
                        model=self._model,
                        messages=messages,
                        **call_args,
                    )
                    # Record token usage for TPM limiters
                    if hasattr(self._rate_limiter, "record_usage"):
                        usage = getattr(result, "usage", None)
                        if usage:
                            self._rate_limiter.record_usage(usage.total_tokens)
                    return result

            return await _litellm_acompletion(
                model=self._model,
                messages=messages,
                **call_args,
            )

        if cancellation_token is None:
            retry_result = await _llm_call()
            return retry_result.result, retry_result.metrics

        future = asyncio.ensure_future(_llm_call())
        cancellation_token.link_future(future)
        retry_result = await future
        return retry_result.result, retry_result.metrics

    @staticmethod
    def _normalize_usage_cost(resp: ModelResponse) -> None:
        """Normalize ``Usage.cost`` from dict to float.

        Some providers (e.g. Perplexity) populate ``Usage.cost`` with a dict
        instead of a float.  Normalising before Pydantic serialization prevents
        ``PydanticSerializationUnexpectedValue`` warnings during ``model_dump()``.
        """
        usage = getattr(resp, "usage", None)
        if usage is not None:
            cost = getattr(usage, "cost", None)
            if isinstance(cost, dict):
                cost_dict = cast(dict[str, Any], cost)
                total_cost = cost_dict.get("total_cost")
                usage.cost = (
                    total_cost if isinstance(total_cost, (int, float)) else None
                )

    def _record_tracing_output(
        self,
        span_generation: Span[Any],
        result: ModelResponse,
    ) -> None:
        """Persist the model output in tracing when enabled."""

        if not self._tracing.include_data():
            return

        output_data = [cast(dict[str, Any], cast(Any, result).model_dump(mode="json"))]
        span_generation.span_data.output = output_data

    def _record_usage(
        self,
        span_generation: Span[Any],
        result: ModelResponse,
        retry_metrics: RetryMetrics | None = None,
    ) -> None:
        """Record token usage, cost, and retry information when available."""

        usage = getattr(result, "usage", None)
        if usage is None:
            span_generation.span_data.usage = {}
        else:
            span_generation.span_data.usage = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }

        # Add retry metrics
        if retry_metrics is not None:
            span_generation.span_data.usage["attempts"] = retry_metrics.attempts
            span_generation.span_data.usage["retry_wait"] = retry_metrics.retry_wait
            span_generation.span_data.usage["backend_latency"] = (
                retry_metrics.backend_latency
            )

        if not hasattr(result, "_hidden_params"):
            return

        hidden_params = getattr(result, "_hidden_params", {})
        response_cost = (
            cast(dict[str, Any], hidden_params).get("response_cost")
            if isinstance(hidden_params, dict)
            else None
        )

        if isinstance(response_cost, dict):
            response_cost_dict = cast(dict[str, Any], response_cost)
            total_cost = response_cost_dict.get("total_cost")
            if isinstance(total_cost, (int, float)):
                span_generation.span_data.usage["total_cost"] = total_cost
                return

        if response_cost is not None and isinstance(response_cost, (int, float)):
            span_generation.span_data.usage["total_cost"] = response_cost
            return

        usage_cost = getattr(usage, "cost", None) if usage is not None else None
        if isinstance(usage_cost, (int, float)):
            span_generation.span_data.usage["total_cost"] = usage_cost

    def _to_model_response(self, resp: ModelResponse) -> TResponseType:
        """Adapt LiteLLM response to our ModelResponse (OpenAI ChatCompletion).

        This ensures strict type conformance and catches any schema mismatches.
        """
        data = cast(dict[str, Any], cast(Any, resp).model_dump(mode="json"))
        return ModelResponse.model_validate(data)

    def _validate_response_format(self, response_format: ResponseFormat | None) -> None:
        """Validate the response format."""

        if response_format is None:
            raise TypeError(
                "response_format must be a dict or a Pydantic model when enforce_structured_output is True."
            )

        if not isinstance(response_format, dict):
            return

        # If the response format is a dict, validate its structure
        required_keys = {"type", "json_schema"}
        if not required_keys.issubset(response_format.keys()):
            raise ValueError(f"response_format must contain the keys: {required_keys}.")
        if response_format["type"] != "json_schema":
            raise ValueError(
                "The 'type' key in response_format must have the value 'json_schema'."
            )

        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise TypeError("The 'json_schema' key in response_format must be a dict.")
        json_schema_dict = cast(dict[str, Any], json_schema)
        schema_keys = {"schema", "name", "strict"}
        if not schema_keys.issubset(json_schema_dict.keys()):
            raise ValueError(
                f"The 'json_schema' dict must contain the keys: {schema_keys}."
            )
