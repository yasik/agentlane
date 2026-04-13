import abc
import enum
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel

from ._output_schema import OutputSchema
from ._rate_limiter import RateLimiter
from ._streaming import ModelStreamEvent, ModelStreamEventKind
from ._tool import Tool, ToolFunction, ToolSpec

type MessageDict = dict[str, Any]
"""Conversation message payload shared across client interfaces."""


class ModelTracing(enum.Enum):
    """The tracing mode for the LLM client."""

    DISABLED = 0
    """Tracing is disabled entirely."""

    ENABLED = 1
    """Tracing is enabled, and all data is included."""

    ENABLED_WITHOUT_DATA = 2
    """Tracing is enabled, but inputs/outputs are not included."""

    def is_disabled(self) -> bool:
        """Whether tracing is disabled."""
        return self == ModelTracing.DISABLED

    def include_data(self) -> bool:
        """Whether tracing includes data."""
        return self == ModelTracing.ENABLED


@dataclass(frozen=True)
class Config:
    """Common configuration for the LLM clients.

    This config is intentionally limited to control-plane and networking concerns
    shared across provider clients. Model-specific request parameters such as
    temperature, reasoning effort, cache retention, and provider-specific extras
    should be passed as client kwargs or per-call model args instead of being
    normalized into this shared config.

    Note on ``enforce_structured_output`` behavior:
    1. If ``True``, the schema should not be included in the prompt. The client
       will enforce schema-constrained output via the model API (e.g., response
       format / structured output settings).
    2. ``True`` may not be compatible with certain model features (e.g., Anthropic
       extended thinking mode). In such cases, disable those features or set
       ``enforce_structured_output=False``.
    3. If ``False``, the schema must be passed in the prompt manually (e.g., as part
       of the system/user instructions) to guide the model toward the desired
       structure.
    """

    api_key: str
    """API key for the LLM client."""

    model: str
    """Model for the LLM client."""

    max_retries: int = 3
    """Maximum number of retries for the LLM client."""

    timeout: float | httpx.Timeout = 600
    """Timeout for the LLM client.

    Can be a float (seconds) or httpx.Timeout for granular control:
    - httpx.Timeout(600.0, connect=30.0) for 10min read, 30sec connect timeout
    """

    organization: str | None = None
    """Organization for the LLM client."""

    base_url: str | None = None
    """Base URL for the LLM client.

    For standard OpenAI: overrides the default API endpoint.
    For Azure: the Azure resource endpoint (e.g., ``https://your-resource.openai.azure.com/``).
    For LiteLLM: forwarded as both ``base_url`` and ``api_base``.
    """

    default_headers: dict[str, str] | None = None
    """Default headers for the LLM client."""

    vertex_project_id: str | None = None
    """Vertex AI project ID."""

    vertex_location: str | None = None
    """Vertex AI project location."""

    enforce_structured_output: bool = False
    """Whether to enforce JSON schema on the LLM response. If this is True,
    the response format should be explicitly set to expect structured deterministic
    output from the LLM."""

    tracing: ModelTracing = ModelTracing.DISABLED
    """The tracing mode for the LLM client."""

    schema_validation_retries: int = 3
    """Number of retries for JSON schema validation failures.

    This is separate from HTTP retries (429, 5xx) and controls how many times
    the client will retry when the model returns invalid JSON that doesn't
    match the expected schema. On retry, the client adds guidance to help
    the model self-correct.
    """

    rate_limiter: RateLimiter | None = None
    """Optional rate limiter to enforce usage limits across client instances.

    When provided, the rate limiter is shared across all client instances
    created from the same factory, enabling global rate limiting per model.
    """

    def to_trace_settings(self) -> dict[str, Any]:
        """Convert the configuration to trace settings."""
        trace_settings: dict[str, Any] = {}

        if self.base_url is not None:
            trace_settings["base_url"] = self.base_url
        trace_settings["timeout"] = self.timeout
        trace_settings["max_retries"] = self.max_retries
        if self.vertex_project_id is not None:
            trace_settings["vertex_project_id"] = self.vertex_project_id
        if self.vertex_location is not None:
            trace_settings["vertex_location"] = self.vertex_location
        if self.enforce_structured_output:
            trace_settings["enforce_structured_output"] = self.enforce_structured_output
        trace_settings["schema_validation_retries"] = self.schema_validation_retries
        return trace_settings


type ToolSource = ToolSpec[Any] | ToolFunction
"""Accepted tool input at the developer boundary.

`Tools(...)` accepts either a declarative `ToolSpec` (including executable
`Tool` values) or a normal typed Python callable. Plain callables are
normalized into native `Tool` instances once so the rest of the framework can
work with one canonical schema surface.
"""


def _normalize_tool(tool: ToolSource) -> ToolSpec[Any]:
    """Return one canonical tool schema from the accepted developer input."""
    if isinstance(tool, ToolSpec):
        return cast(ToolSpec[Any], tool)
    return Tool.from_function(tool)


@dataclass(frozen=True, slots=True)
class Tools:
    """Configuration for enabling tool calling on LLM requests."""

    tools: Sequence[ToolSource]
    """Collection of available tools that can be invoked by the model.

    Each item may be either a declarative `ToolSpec` or a typed Python
    callable. Plain callables are converted into `Tool.from_function(...)`
    during construction so application code can stay lightweight without
    widening the internal execution contract.
    """

    tool_choice: Literal["auto", "required", "none"] = "auto"
    """Strategy that controls if and how the model may call tools."""

    parallel_tool_calls: bool = False
    """Whether the model is allowed to call tools in parallel."""

    tool_call_timeout: float | None = None
    """Timeout in seconds for individual tool calls.

    When set, each tool call will be wrapped in asyncio.wait_for with this timeout.
    On timeout, retries up to tool_call_max_retries times, then returns an error
    message to the LLM allowing it to self-correct or proceed without the tool result.
    """

    tool_call_max_retries: int = 3
    """Number of retries after timeout (default: 3 retries = 4 total attempts)."""

    tool_call_limits: Mapping[str, int] | None = None
    """Per-tool call limits.  Maps tool name to max allowed calls.

    When a tool's call count reaches its limit, the harness runner removes it
    from the next model request so the model cannot invoke it again. When all
    tools are exhausted, tools are stripped entirely from later requests,
    forcing the model to produce a text response.
    """

    max_tool_round_trips: int = 10
    """Global safety limit on LLM → tool → LLM cycles.

    Acts as a last resort to prevent infinite loops when
    ``tool_call_limits`` is not configured. When reached, the harness runner
    strips all tools from the next model request.
    """

    def __post_init__(self) -> None:
        """Normalize plain callables into native ``Tool`` values once."""
        normalized_tools = tuple(_normalize_tool(tool) for tool in self.tools)
        object.__setattr__(self, "tools", normalized_tools)

    @property
    def normalized_tools(self) -> tuple[ToolSpec[Any], ...]:
        """Return the canonical tool schemas after construction-time normalization."""
        return cast(tuple[ToolSpec[Any], ...], self.tools)

    @property
    def executable_tools(self) -> tuple[Tool[Any, Any], ...]:
        """Return only executable native tools from the normalized tool set."""
        executable_tools: list[Tool[Any, Any]] = []
        for tool in self.normalized_tools:
            if isinstance(tool, Tool):
                executable_tools.append(cast(Tool[Any, Any], tool))
        return tuple(executable_tools)

    def as_args(self) -> dict[str, Any]:
        """Render the configuration into the kwargs expected by LiteLLM."""

        tools_payload: list[dict[str, Any]] = []
        for tool in self.normalized_tools:
            output_schema = {
                "type": "function",
                "function": tool.schema,
            }
            tools_payload.append(output_schema)

        return {
            "tools": tools_payload,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
        }


class Model[TResponse](abc.ABC):
    """The base interface for calling an LLM."""

    async def __call__(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        **kwargs: Any,
    ) -> TResponse:
        """Get the response from the LLM."""
        return await self.get_response(
            messages,
            extra_call_args=extra_call_args,
            schema=schema,
            tools=tools,
            **kwargs,
        )

    @abc.abstractmethod
    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        **kwargs: Any,
    ) -> TResponse:
        """Get the response from the LLM."""
        raise NotImplementedError("get_response method must be implemented")

    def stream_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, Any] | None = None,
        schema: type[BaseModel] | OutputSchema[Any] | None = None,
        tools: Tools | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Return a stream of model events.

        The default implementation preserves backward compatibility for
        non-streaming model implementations by calling ``get_response(...)`` and
        emitting one terminal ``COMPLETED`` event. Concrete provider clients can
        override this with real streaming support.
        """

        async def _fallback() -> AsyncIterator[ModelStreamEvent]:
            try:
                response = await self.get_response(
                    messages,
                    extra_call_args=extra_call_args,
                    schema=schema,
                    tools=tools,
                    **kwargs,
                )
            except Exception as error:
                yield ModelStreamEvent(
                    kind=ModelStreamEventKind.ERROR,
                    error=error,
                )
                raise

            yield ModelStreamEvent(
                kind=ModelStreamEventKind.COMPLETED,
                response=cast(Any, response),
            )

        return _fallback()


class Factory[TResponse](abc.ABC):
    """The base interface for creating an LLM client."""

    def __init__(self, default_config: Config):
        """Initialize the factory."""
        self._default_config = default_config

    @abc.abstractmethod
    def get_model_client(
        self, tracing: ModelTracing = ModelTracing.DISABLED, **kwargs: Any
    ) -> Model[TResponse]:
        """Get a client for the LLM.

        Args:
            tracing: The tracing mode to use for the client.
            kwargs: Additional keyword arguments to pass to the client. These will override
                the default configuration.
        """
        raise NotImplementedError("get_model_client method must be implemented")
