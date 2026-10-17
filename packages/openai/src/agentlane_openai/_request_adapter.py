"""Convert harness messages and options into native Responses request payloads."""

from typing import Any, cast

from pydantic import BaseModel

from agentlane.models import (
    MessageDict,
    MessageRole,
    OutputSchema,
    ToolChoice,
    Tools,
    resolve_output_schema,
)

type ResponseInputItem = dict[str, Any]
type ResponseContentPart = dict[str, Any]


def messages_to_input(messages: list[MessageDict]) -> list[ResponseInputItem]:
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
        # Replay native items directly so they are not also converted from chat
        # fields.
        if "openai_responses_input" in msg:
            result.extend(msg["openai_responses_input"])
            continue
        if "type" in msg and "role" not in msg:
            result.append(dict(msg))
            continue

        role_value = msg.get("role", MessageRole.USER.value)
        role = role_value if isinstance(role_value, str) else MessageRole.USER.value
        content = msg.get("content", "")

        if role in {MessageRole.SYSTEM.value, MessageRole.DEVELOPER.value}:
            # Keep the existing system-to-developer mapping. Explicit
            # developer instructions must reach the same input path.
            result.append(
                {
                    "type": "message",
                    "role": MessageRole.DEVELOPER.value,
                    "content": _convert_content(content),
                }
            )

        elif role == MessageRole.USER.value:
            result.append(
                {
                    "type": "message",
                    "role": MessageRole.USER.value,
                    "content": _convert_content(content),
                }
            )

        elif role == MessageRole.ASSISTANT.value:
            assistant_content = _convert_assistant_content(content)

            # Chat Completions can store a refusal outside the content list.
            # Replay it as content so a refusal-only turn is not lost.
            refusal = msg.get("refusal")
            if isinstance(refusal, str) and refusal:
                assistant_content.append({"type": "refusal", "refusal": refusal})

            # Converted Chat Completions history is complete by default.
            # Retain an explicit provider status or ID when supplied; never
            # manufacture an ID for a message from another API.
            assistant_item: ResponseInputItem = {
                "type": "message",
                "role": MessageRole.ASSISTANT.value,
                "content": assistant_content,
                "status": msg.get("status", "completed"),
            }

            if "id" in msg:
                assistant_item["id"] = msg["id"]

            # Preserve the distinction between commentary and a final answer
            # when the caller supplies a phase for the prior assistant turn.
            if "phase" in msg:
                assistant_item["phase"] = msg["phase"]

            # A tool-only turn needs function calls, not an empty message.
            if assistant_item["content"]:
                result.append(assistant_item)

            # Responses uses separate function-call items. Place them after
            # the assistant text to preserve the original conversation order.
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

        elif role == MessageRole.TOOL.value:
            # The call ID connects this result to its earlier function call.
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                }
            )

    return result


def _convert_assistant_content(content: object) -> list[ResponseContentPart]:
    """Convert assistant text and refusal parts without changing their order."""
    if isinstance(content, str):
        return (
            [{"type": "output_text", "text": content, "annotations": []}]
            if content
            else []
        )

    if not isinstance(content, list):
        return []

    # Keep part boundaries and order when replaying a prior assistant turn.
    result: list[ResponseContentPart] = []
    for part in cast(list[object], content):
        if not isinstance(part, dict):
            continue

        part_dict = cast(dict[str, Any], part)
        part_type = part_dict.get("type")
        text = part_dict.get("text", "")

        # Empty text parts carry no assistant text. Treat them like an empty
        # string so a tool-only turn does not gain a blank message item.
        if part_type in {"text", "input_text", "output_text"} and not text:
            continue

        if part_type in {"text", "input_text"}:
            # Converted text has no citation metadata; Responses output text
            # still needs an annotations list.
            result.append(
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }
            )

        elif part_type == "output_text":
            # Retain existing annotations and other output-text metadata.
            result.append({"annotations": [], **part_dict})

        elif part_type == "refusal":
            result.append(dict(part_dict))

    return result


def _convert_content(content: object) -> list[ResponseContentPart]:
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
        elif part_type in {"input_text", "input_image", "input_file"}:
            result.append(part_dict)
        elif part_type == "image_url":
            image = part_dict["image_url"]
            result.append(
                {
                    "type": "input_image",
                    "image_url": image["url"],
                    **({"detail": image["detail"]} if "detail" in image else {}),
                }
            )
        elif part_type == "file":
            result.append({"type": "input_file", **part_dict["file"]})

    return result


class RequestAdapter:
    """Apply configured defaults when converting each harness request."""

    def __init__(
        self,
        common_params: dict[str, Any],
        enforce_structured_output: bool,
        async_tools: frozenset[str],
    ) -> None:
        self._common_params = dict(common_params)
        self._enforce_structured_output = enforce_structured_output
        self._async_tools = async_tools

    def build_call_args(
        self,
        extra_call_args: dict[str, Any] | None,
        schema: type[BaseModel] | OutputSchema[Any] | None,
        tools: Tools | None,
    ) -> dict[str, Any]:
        """Build the arguments for the Responses API call."""
        call_args: dict[str, Any] = {**self._common_params}

        if extra_call_args:
            call_args.update(extra_call_args)

        # Preserve native text options while applying the harness output schema.
        output_schema = (
            resolve_output_schema(schema) if self._enforce_structured_output else None
        )
        if output_schema is not None:
            response_format = output_schema.response_format()
            if isinstance(response_format, dict):
                json_schema = response_format.get("json_schema", {})
                call_args["text"] = {
                    **(call_args.get("text") or {}),
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", "response"),
                        "schema": json_schema.get("schema", {}),
                        "strict": json_schema.get("strict", True),
                    },
                }

        # Handle tools
        if tools is not None:
            tools_payload: list[dict[str, Any]] = list(call_args.get("tools") or [])
            local_names = {tool.name for tool in tools.normalized_tools}

            # Native declarations must not bypass the prepared local tool catalog.
            for native_tool in tools_payload:
                if native_tool.get("name") in local_names:
                    raise ValueError(
                        f"Duplicate native tool name: {native_tool['name']}"
                    )
                if native_tool.get("type") in {"function", "custom"}:
                    raise ValueError(
                        "A native tool declaration cannot add a local handler to prepared Tools. "
                        "Register function handlers through Tools instead."
                    )

            # Expose only the local handlers that the harness prepared for this turn.
            for tool in tools.normalized_tools:
                tools_payload.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema.get("parameters", {}),
                        "strict": tool.strict if tool.strict is not None else True,
                    }
                )

            # Opt selected function declarations into native asynchronous execution.
            for declaration in tools_payload:
                if (
                    declaration.get("type") == "function"
                    and declaration.get("name") in self._async_tools
                ):
                    declaration["async"] = True

            call_args["tools"] = tools_payload

            # Map tool_choice (same values for Responses API)
            if (
                tools.tool_choice != ToolChoice.AUTO.value
                or "tool_choice" not in call_args
            ):
                call_args["tool_choice"] = tools.tool_choice

            call_args["parallel_tool_calls"] = tools.parallel_tool_calls

        # Convert max_tokens to max_output_tokens (Responses API naming)
        if "max_tokens" in call_args:
            call_args["max_output_tokens"] = call_args.pop("max_tokens")

        # Convert reasoning_effort to reasoning dict (Responses API format)
        if "reasoning_effort" in call_args:
            reasoning = dict(call_args.get("reasoning") or {})
            reasoning["effort"] = call_args.pop("reasoning_effort")
            call_args["reasoning"] = reasoning

        # Convert verbosity to text config (Responses API format)
        # verbosity -> text.verbosity (low, medium, high)
        if "verbosity" in call_args:
            text = dict(call_args.get("text") or {})
            text["verbosity"] = call_args.pop("verbosity")
            call_args["text"] = text

        return call_args
