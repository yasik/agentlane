"""Project native Responses output onto the harness result and replay contracts."""

from dataclasses import dataclass
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
    ResponseApplyPatchToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseFunctionShellToolCall,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseToolSearchCall,
)
from openai.types.responses.response_output_item import (
    LocalShellCall,
    McpApprovalRequest,
    ResponseOutputItem,
)

from agentlane.models import MessageRole, ModelResponse


class ResponseError(RuntimeError):
    """A terminal provider failure with its native response or event attached."""

    def __init__(self, response: object, message: str | None = None) -> None:
        self.response = response
        self.error = getattr(response, "error", None)
        super().__init__(
            message
            or f"OpenAI response ended with {getattr(response, 'status', 'error')}."
        )


def convert_function_call(
    item: ResponseFunctionToolCall,
) -> ChatCompletionMessageToolCall:
    """Project the call identity and arguments used by the harness executor."""
    return ChatCompletionMessageToolCall(
        id=item.call_id,
        type="function",
        function=Function(name=item.name, arguments=item.arguments),
    )


@dataclass
class ResponseAdapter:
    """Adapt one response while retaining the native payload for lossless replay."""

    response: Response

    def adapt(self, *, allow_steered: bool = False) -> ModelResponse:
        if self.response.status in {"failed", "cancelled", "queued", "in_progress"}:
            raise ResponseError(self.response)

        # Project output for the harness while preserving native items for replay.
        message, reasoning = self._project_output()
        finish_reason = self._finish_reason(bool(message.tool_calls), allow_steered)
        Response.model_rebuild()
        native_response = self.response.model_dump(
            mode="json", exclude_unset=True, by_alias=True
        )
        if "output" in native_response:
            message = message.model_copy(
                update={"openai_responses_input": native_response["output"]}
            )

        # Build the common result without losing the original response payload.
        result = ChatCompletion(
            id=self.response.id,
            choices=[
                Choice(
                    index=0, message=message, finish_reason=finish_reason, logprobs=None
                )
            ],
            created=int(self.response.created_at),
            model=self.response.model,
            object="chat.completion",
            usage=self._usage(),
        ).model_copy(update={"openai_response": native_response})
        if reasoning is not None:
            # SDK parsing can leave the typed extra's schema deferred.
            ResponseReasoningItem.model_rebuild()
            cast(Any, result).reasoning_content = reasoning

        return result

    def _project_output(
        self,
    ) -> tuple[ChatCompletionMessage, ResponseReasoningItem | None]:
        content: str | None = None
        tool_calls: list[ChatCompletionMessageToolCall] = []
        reasoning: ResponseReasoningItem | None = None

        # Reject unsupported local execution before projecting text and function calls.
        for item in self.response.output:
            self._validate_execution(item)
            if isinstance(item, ResponseOutputMessage):
                text = "".join(
                    part.text for part in item.content if part.type == "output_text"
                )
                content = text or content
            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(convert_function_call(item))
            elif isinstance(item, ResponseReasoningItem):
                reasoning = item

        return (
            ChatCompletionMessage(
                role=MessageRole.ASSISTANT.value,
                content=content,
                tool_calls=tool_calls or None,  # type: ignore[arg-type]
            ),
            reasoning,
        )

    def _validate_execution(self, item: ResponseOutputItem) -> None:
        unsupported = isinstance(
            item,
            (
                ResponseCustomToolCall,
                ResponseComputerToolCall,
                ResponseApplyPatchToolCall,
                LocalShellCall,
                McpApprovalRequest,
            ),
        )

        # Hosted operations can remain native; local work needs a harness handler.
        if isinstance(item, ResponseFunctionShellToolCall):
            unsupported = item.environment is None or item.environment.type == "local"
        if isinstance(item, ResponseToolSearchCall):
            unsupported = item.execution == "client"

        # The harness can dispatch only direct, unqualified function calls.
        if isinstance(item, ResponseFunctionToolCall):
            unsupported = item.namespace is not None or (
                item.caller is not None and item.caller.type != "direct"
            )
        if unsupported:
            raise ResponseError(
                self.response,
                f"Unsupported AgentLane tool execution: {item.type}. Use a supported function handler or the native SDK.",
            )

    def _finish_reason(
        self,
        has_tool_calls: bool,
        allow_steered: bool,
    ) -> Literal["stop", "length", "tool_calls", "content_filter"]:
        if self.response.status == "incomplete":
            reason = getattr(self.response.incomplete_details, "reason", None)
            if reason == "max_output_tokens":
                return "length"
            if reason == "content_filter":
                return "content_filter"

            # A steered response is usable only when the caller handles its
            # continuation.
            if not (allow_steered and reason == "steered"):
                raise ResponseError(self.response)

        return "tool_calls" if has_tool_calls else "stop"

    def _usage(self) -> CompletionUsage | None:
        usage = self.response.usage
        if usage is None:
            return None

        return CompletionUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )


def response_to_model_response(
    response: Response,
    *,
    allow_steered: bool = False,
) -> ModelResponse:
    """Convert a native response to the canonical harness result with replay data."""
    return ResponseAdapter(response).adapt(allow_steered=allow_steered)
