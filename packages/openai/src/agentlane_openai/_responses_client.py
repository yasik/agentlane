# pylint: disable=R0917, C0301

"""Adapt the native Responses SDK to the harness model and streaming contracts."""

import asyncio
import contextlib
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Collection,
    Generator,
)
from copy import copy
from dataclasses import fields, replace
from typing import Any, Self, cast

import structlog
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AsyncStream,
    InternalServerError,
)
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseInputParam,
    ResponseStreamEvent,
)
from pydantic import BaseModel

from agentlane.models import (
    ESCAPE_SEQUENCE_RETRY_MESSAGE,
    Config,
)
from agentlane.models import Factory as BaseFactory
from agentlane.models import (
    MessageDict,
    MessageRole,
    Model,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ModelTracing,
    OutputSchema,
    RetryMetrics,
    SteeringInput,
    SteeringReceipt,
    Tools,
    has_escape_sequence_explosion,
    is_retryable_by_status_code,
    parse_content_filter_block,
    resolve_output_schema,
    retry_on_errors,
)
from agentlane.runtime import CancellationToken
from agentlane.tracing import Span, generation_span

from ._constants import NO_ACTIVE_RESPONSE
from ._event_adapter import to_stream_event
from ._request_adapter import RequestAdapter, messages_to_input
from ._response_adapter import (
    response_to_model_response,
)
from ._responses_session import ResponsesSession

LOGGER = structlog.get_logger(log_tag="agentlane.openai.responses_client")
"""Used to report native request attempts and refusal retries."""

_MAX_REFUSAL_RETRIES = 3
"""Limit extra Azure attempts after a content-filter refusal."""


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
        websocket: bool = False,
        async_tools: Collection[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            config: The configuration for the client.
            refusal_context: Optional message appended to retry requests when
                Azure content filters block a response.  When ``None``, no
                refusal-retry context is added.
            websocket: Use a native connection within a harness run or open_run context.
            async_tools: Local function names that can run while model output continues.
            kwargs: Default model-call arguments forwarded to the Responses API.
        """
        self._async_tools = frozenset(async_tools)
        self._websocket = websocket

        # The configured client has no connection until open_run binds a session.
        self._session: ResponsesSession | None = None

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
        self._request_adapter = RequestAdapter(
            kwargs, self._enforce_structured_output, self._async_tools
        )

    @contextlib.asynccontextmanager
    async def open_run(self) -> AsyncGenerator[Self, None]:
        """Use one native connection per run, with no automatic request replay."""
        if not self._websocket:
            yield self
            return

        # Share SDK configuration, but isolate connection and steering state per run.
        async with self._openai_client.responses.connect(max_retries=0) as connection:
            bound = copy(self)
            session = ResponsesSession(
                connection, self._model, self._async_tools, self._record_stream_response
            )
            bound._session = session
            try:
                yield bound
            finally:
                session.settle()
                bound._session = None

    async def steer(self, input: SteeringInput) -> SteeringReceipt:
        """Submit steering through this run's native session."""
        session = self._session
        if session is None:
            raise RuntimeError(NO_ACTIVE_RESPONSE)

        return await session.steer(input)

    @property
    def requires_streaming(self) -> bool:
        """Keep native connection events available to the harness."""
        return self._websocket or bool(self._async_tools)

    @property
    def handles_retries(self) -> bool:
        """Keep HTTP retries in this adapter, including server delay handling."""
        return True

    @property
    def tracing(self) -> ModelTracing:
        """Return the tracing mode configured for this client."""
        return self._tracing

    async def get_response(  # pylint: disable=R0912
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
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
            parent_span: Pre-opened generation span owned by the caller. When
                provided, model metadata and usage are recorded onto it instead
                of opening a new span — so subsequent tool spans can nest under
                the same generation.

        Returns:
            ModelResponse in Chat Completions format for interchangeability.
        """
        if self._websocket:
            raise ValueError(
                "WebSocket mode requires a harness run or stream_response inside open_run."
            )

        # Convert history once; repair attempts append only their feedback.
        conversation_input = messages_to_input(messages)
        schema_retry_count = 0

        with self._generation_span(parent_span, messages) as span_generation:
            output_schema = resolve_output_schema(schema)
            call_args = self._request_adapter.build_call_args(
                extra_call_args=extra_call_args,
                schema=output_schema,
                tools=tools,
            )

            while True:
                LOGGER.debug(
                    "Responses API call started",
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
                    retry_metrics=retry_metrics,
                )

                model_response = response_to_model_response(result)

                # Return tool calls to the harness before validating final-answer text.
                if model_response.choices[0].message.tool_calls:
                    break

                # Retry malformed text within the same output-repair budget.
                if has_escape_sequence_explosion(model_response):
                    schema_retry_count += 1
                    if schema_retry_count <= self._schema_validation_retries:
                        conversation_input.extend(
                            messages_to_input(
                                [
                                    {
                                        "role": MessageRole.USER.value,
                                        "content": ESCAPE_SEQUENCE_RETRY_MESSAGE,
                                    }
                                ]
                            )
                        )
                        continue

                # Preserve the shared schema validator's repair and enforcement policy.
                if output_schema is not None:
                    validation_result = output_schema.validate_response_with_retry(
                        response=model_response,
                        retry_count=schema_retry_count,
                        max_retries=self._schema_validation_retries,
                        enforce_structured_output=self._enforce_structured_output,
                    )
                    if validation_result.should_retry:
                        schema_retry_count += 1
                        conversation_input.extend(
                            messages_to_input(
                                [
                                    {
                                        "role": MessageRole.USER.value,
                                        "content": validation_result.error_message,
                                    }
                                ]
                            )
                        )
                        continue

                break

            # Both tool calls and final answers use the same completion bookkeeping.
            self._record_response(span_generation, result, retry_metrics)

            return model_response

    def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        parent_span: Span[Any] | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream semantic OpenAI Responses API events as normalized model events.

        Retry behavior is scoped to stream setup only: transient errors that
        occur before the first chunk trigger the shared retry policy, while
        mid-stream failures surface as an ``ERROR`` event and a raised
        exception without retry.

        See :meth:`get_response` for ``parent_span`` semantics.
        """
        if self._websocket:
            return self._stream_connection(
                messages,
                extra_call_args,
                schema,
                tools,
                parent_span,
            )

        # Convert history once for the HTTP stream setup and its retries.
        conversation_input = messages_to_input(messages)

        async def _stream() -> AsyncIterator[ModelStreamEvent]:
            trace_events: list[dict[str, Any]] = []
            stream: AsyncStream[ResponseStreamEvent] | None = None

            with self._generation_span(parent_span, messages) as span_generation:
                call_args = self._request_adapter.build_call_args(
                    extra_call_args=extra_call_args,
                    schema=schema,
                    tools=tools,
                )

                LOGGER.debug(
                    "Responses API stream started",
                )

                retry_metrics: RetryMetrics | None = None
                emitted_error = False

                # Hold the limiter for the lifetime of the stream so one active
                # streamed generation counts as one in-flight model call.
                limiter_cm = (
                    self._rate_limiter
                    if self._rate_limiter is not None
                    else contextlib.nullcontext()
                )

                try:
                    async with limiter_cm:
                        stream, retry_metrics = await self._create_stream_with_retry(
                            input_data=conversation_input,
                            call_args=call_args,
                            cancellation_token=cancellation_token,
                        )
                        async for event in self._iterate_stream(
                            stream=stream,
                            trace_events=trace_events,
                            span_generation=span_generation,
                            retry_metrics=retry_metrics,
                        ):
                            if event.kind == ModelStreamEventKind.ERROR:
                                emitted_error = True

                            yield event
                except Exception as error:
                    if not emitted_error:
                        error_event = ModelStreamEvent(
                            kind=ModelStreamEventKind.ERROR,
                            error=error,
                        )
                        self._append_trace_event(trace_events, error_event)
                        yield error_event

                    raise
                finally:
                    if self._tracing.include_data() and trace_events:
                        span_generation.span_data.events = trace_events
                    if stream is not None:
                        await stream.close()

        return _stream()

    async def _stream_connection(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None,
        schema: type[BaseModel] | OutputSchema[Any] | None,
        tools: Tools | None,
        parent_span: Span[Any] | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Prepare a native turn and map SDK events under the existing trace scope."""
        session = self._session
        if session is None:
            raise RuntimeError("WebSocket calls require an active open_run context.")

        # Validate turn options and history before sending through the native
        # connection.
        call_args = self._request_adapter.build_call_args(
            extra_call_args,
            schema,
            tools,
        )
        input_data = session.prepare_input(messages, call_args)

        # Keep tracing and rate limits scoped to this logical response.
        with self._generation_span(parent_span, messages) as span:
            async with contextlib.AsyncExitStack() as scope:
                if self._rate_limiter is not None:
                    await scope.enter_async_context(self._rate_limiter)

                async for event in session.stream(input_data, call_args, span):
                    yield event

    def get_model(self) -> str:
        """Get the model name."""
        return self._model

    def get_is_enforce_structured_output(self) -> bool:
        """Get the enforce_structured_output flag."""
        return self._enforce_structured_output

    async def _iterate_stream(
        self,
        *,
        stream: AsyncStream[ResponseStreamEvent],
        trace_events: list[dict[str, Any]],
        span_generation: Span[Any],
        retry_metrics: RetryMetrics | None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Yield normalized events from one open OpenAI response stream."""
        async for raw_event in stream:
            event = to_stream_event(raw_event, self._async_tools)
            self._append_trace_event(trace_events, event)

            if isinstance(raw_event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                # Event conversion validates the terminal output before accounting.
                self._record_stream_response(
                    span_generation, raw_event.response, retry_metrics
                )
                yield event
                return

            yield event

            if event.kind == ModelStreamEventKind.ERROR:
                error = event.error or RuntimeError("OpenAI streaming error.")
                raise error

        raise RuntimeError("OpenAI stream ended without a completed event.")

    def _append_trace_event(
        self,
        trace_events: list[dict[str, Any]],
        event: ModelStreamEvent,
    ) -> None:
        """Persist one serialized stream event when tracing includes payload data."""
        if not self._tracing.include_data():
            return
        trace_events.append(event.to_trace_dict())

    async def _create_stream_with_retry(
        self,
        *,
        input_data: list[dict[str, Any]],
        call_args: dict[str, Any],
        cancellation_token: CancellationToken | None,
    ) -> tuple[AsyncStream[ResponseStreamEvent], RetryMetrics]:
        """Create one OpenAI response stream with retry on stream setup errors."""

        @retry_on_errors(
            max_retries=self._max_retries, is_retryable=_is_openai_retryable
        )
        async def _api_call() -> AsyncStream[ResponseStreamEvent]:
            stream: AsyncStream[ResponseStreamEvent] = (
                await self._openai_client.responses.create(
                    model=self._model,
                    input=cast(ResponseInputParam, input_data),
                    stream=True,
                    **call_args,
                )
            )

            return stream

        if cancellation_token is None:
            retry_result = await _api_call()
            return retry_result.result, retry_result.metrics

        future = asyncio.ensure_future(_api_call())
        cancellation_token.link_future(future)
        retry_result = await future
        return retry_result.result, retry_result.metrics

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
            async with (
                self._rate_limiter
                if self._rate_limiter is not None
                else contextlib.nullcontext()
            ):
                result = cast(
                    OpenAIResponse,
                    await self._openai_client.responses.create(
                        model=self._model,
                        input=cast(ResponseInputParam, input_data),
                        **call_args,
                    ),
                )
                if self._rate_limiter is not None and result.usage is not None:
                    self._rate_limiter.record_usage(result.usage.total_tokens)

                return result

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
                input_data=[
                    *input_data,
                    *messages_to_input(
                        [
                            {
                                "role": MessageRole.USER.value,
                                "content": self._refusal_context,
                            }
                        ]
                    ),
                ],
                call_args=call_args,
                cancellation_token=cancellation_token,
            )

        return response, metrics

    @contextlib.contextmanager
    def _generation_span(
        self,
        parent_span: Span[Any] | None,
        messages: list[MessageDict],
    ) -> Generator[Span[Any], None, None]:
        if parent_span is not None:
            parent_span.span_data.model = self._model
            parent_span.span_data.model_config = self._trace_settings

        span_context = (
            contextlib.nullcontext(parent_span)
            if parent_span is not None
            else generation_span(
                model=self._model,
                model_config=self._trace_settings,
                disabled=self._tracing.is_disabled(),
            )
        )
        with span_context as span:
            if self._tracing.include_data():
                span.span_data.input = messages

            yield span

    def _record_stream_response(
        self,
        span: Span[Any],
        response: OpenAIResponse,
        retry_metrics: RetryMetrics | None = None,
    ) -> None:
        """Record a provider terminal even when steering delays harness completion."""
        self._record_response(span, response, retry_metrics)

        if self._rate_limiter is not None and response.usage is not None:
            self._rate_limiter.record_usage(response.usage.total_tokens)

    def _record_response(
        self,
        span_generation: Span[Any],
        result: OpenAIResponse,
        retry_metrics: RetryMetrics | None = None,
    ) -> None:
        """Record token usage and retry information when available.

        Accumulates into ``span_generation.span_data.usage`` so a generation
        span shared across multiple model calls in one run captures every
        call's tokens rather than only the last call.
        """
        # Trace payloads are optional, but token and retry accounting always applies.
        if self._tracing.include_data():
            span_generation.span_data.output = [result.model_dump(mode="json")]

        existing_usage = getattr(span_generation.span_data, "usage", None)
        if not isinstance(existing_usage, dict):
            usage_dict: dict[str, Any] = {}
            span_generation.span_data.usage = usage_dict
        else:
            usage_dict = cast(dict[str, Any], existing_usage)

        usage = getattr(result, "usage", None)
        if usage is not None:
            usage_dict["input_tokens"] = (
                usage_dict.get("input_tokens", 0) + usage.input_tokens
            )
            usage_dict["output_tokens"] = (
                usage_dict.get("output_tokens", 0) + usage.output_tokens
            )
            usage_dict["total_tokens"] = (
                usage_dict.get("total_tokens", 0) + usage.total_tokens
            )

            # Accumulate cache metrics for monitoring cache effectiveness across
            # the full run.
            input_details = getattr(usage, "input_tokens_details", None)
            if input_details is not None:
                cached_tokens = getattr(input_details, "cached_tokens", 0) or 0
                usage_dict["cached_tokens"] = (
                    usage_dict.get("cached_tokens", 0) + cached_tokens
                )
                # Recompute cache hit rate over the accumulated totals.
                total_input = usage_dict.get("input_tokens", 0)
                if total_input > 0:
                    usage_dict["cache_hit_rate_pct"] = round(
                        (usage_dict["cached_tokens"] / total_input) * 100, 2
                    )

        # Add retry metrics
        if retry_metrics is not None:
            usage_dict["attempts"] = (
                usage_dict.get("attempts", 0) + retry_metrics.attempts
            )
            usage_dict["retry_wait"] = (
                usage_dict.get("retry_wait", 0) + retry_metrics.retry_wait
            )
            usage_dict["backend_latency"] = (
                usage_dict.get("backend_latency", 0) + retry_metrics.backend_latency
            )
