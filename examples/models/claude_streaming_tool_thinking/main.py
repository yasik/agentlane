"""Claude Sonnet 4.6 streaming example with thinking blocks and a tool call."""

import asyncio
import json
import logging
import os
from typing import Any, cast

import structlog
from agentlane_litellm import Client

from agentlane.models import (
    Config,
    ModelStreamEvent,
    ModelStreamEventKind,
    Tools,
    as_tool,
)
from agentlane.runtime import CancellationToken

MODEL_NAME = "anthropic/claude-sonnet-4-6"
QUESTION = (
    "A patient reports dizziness and two glucose readings in the 60s after starting "
    "a new diabetes medication. Look up the triage protocol first."
)
TOOL_RESULT = (
    "Harborview Triage Protocol: Recurrent glucose readings below 70 mg/dL after a "
    "new diabetes medication require same-day clinician review. Confusion, fainting, "
    "chest pain, or glucose below 54 mg/dL require urgent escalation."
)


@as_tool
async def search_triage_protocol(question: str) -> str:
    """Look up the current clinical triage protocol."""
    del question
    return TOOL_RESULT


async def stream_turn(
    client: Client,
    *,
    messages: list[dict[str, Any]],
    tools: Tools | None,
    extra_call_args: dict[str, Any],
) -> ModelStreamEvent:
    """Run one streamed turn and print the interesting pieces."""
    text_parts: list[str] = []
    reasoning_text_parts: list[str] = []
    thinking_blocks: list[object] = []
    tool_argument_parts: list[str] = []
    provider_events: list[str] = []
    completed_event: ModelStreamEvent | None = None

    async for event in client.stream_response(
        messages=messages,
        tools=tools,
        extra_call_args=extra_call_args,
    ):
        if event.kind == ModelStreamEventKind.TEXT_DELTA:
            if event.text:
                text_parts.append(event.text)
            continue

        if event.kind == ModelStreamEventKind.REASONING:
            reasoning_value: Any = event.reasoning
            if isinstance(reasoning_value, str):
                reasoning_text_parts.append(reasoning_value)
            elif isinstance(reasoning_value, list):
                thinking_blocks.extend(cast(list[Any], reasoning_value))
            elif reasoning_value is not None:
                reasoning_text_parts.append(str(reasoning_value))
            continue

        if event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA:
            if event.arguments_delta:
                tool_argument_parts.append(event.arguments_delta)
            continue

        if event.kind == ModelStreamEventKind.PROVIDER and event.provider_event_type:
            provider_events.append(event.provider_event_type)
            continue

        if event.kind == ModelStreamEventKind.COMPLETED:
            completed_event = event

    if completed_event is None:
        raise RuntimeError("Expected one completed stream event.")

    response = completed_event.response
    if response is None:
        raise RuntimeError("Expected a completed response payload.")

    tool_calls = response.choices[0].message.tool_calls or []
    has_tool_calls = bool(tool_calls)

    if reasoning_text_parts:
        print("Reasoning text stream:")
        print("".join(reasoning_text_parts).strip())
        print()
    if thinking_blocks:
        print("Thinking blocks:")
        print(json.dumps(thinking_blocks, indent=2))
        print()
    if text_parts:
        if has_tool_calls:
            print("Assistant preamble:")
        else:
            print("Assistant:")
        print("".join(text_parts).strip())
        print()
    if tool_argument_parts:
        print("Tool argument stream:")
        print("".join(tool_argument_parts).strip())
        print()
    if provider_events:
        print("Other provider events:")
        print(", ".join(provider_events))
        print()

    return completed_event


def _function_tool_payload(tool_call: object) -> tuple[str, str, str]:
    """Extract the function tool payload from a tool call object."""
    tool_call_any: Any = tool_call
    tool_call_data: dict[str, Any] = tool_call_any.model_dump(mode="json")
    function_data = cast(dict[str, Any] | None, tool_call_data.get("function"))
    if not isinstance(function_data, dict):
        raise RuntimeError("Expected a function tool call.")

    tool_call_id = cast(str | None, tool_call_data.get("id"))
    tool_name = cast(str | None, function_data.get("name"))
    tool_arguments = cast(str | None, function_data.get("arguments"))
    if not isinstance(tool_call_id, str):
        raise RuntimeError("Expected a string tool call id.")
    if not isinstance(tool_name, str):
        raise RuntimeError("Expected a string tool name.")
    if not isinstance(tool_arguments, str):
        raise RuntimeError("Expected stringified JSON tool arguments.")

    return tool_call_id, tool_name, tool_arguments


def _prompt_user() -> str | None:
    """Prompt the user for the next chat message."""
    try:
        user_input = input("You: ").strip()
    except EOFError:
        return None

    if not user_input:
        return ""

    if user_input.lower() in {"exit", "quit"}:
        return None

    return user_input


def _assistant_message_with_thinking(
    completed_event: ModelStreamEvent,
) -> dict[str, Any]:
    """Build the assistant message to resend with the tool result."""
    response = completed_event.response
    if response is None:
        raise RuntimeError("Expected a completed response payload.")

    assistant_message = response.choices[0].message.model_dump(mode="json")

    raw_response: Any = completed_event.raw
    raw_message = raw_response.choices[0].message
    thinking_blocks = getattr(raw_message, "thinking_blocks", None)
    if thinking_blocks is not None:
        assistant_message["thinking_blocks"] = thinking_blocks

    reasoning_content = getattr(raw_message, "reasoning_content", None)
    if reasoning_content is not None:
        assistant_message["reasoning_content"] = reasoning_content

    return assistant_message


async def _run_assistant_turn(
    client: Client,
    *,
    conversation: list[dict[str, Any]],
    tools: Tools,
    extra_call_args: dict[str, Any],
) -> None:
    """Run one full assistant turn, including internal tool continuations."""
    for _ in range(8):
        completed = await stream_turn(
            client,
            messages=conversation,
            tools=tools,
            extra_call_args=extra_call_args,
        )

        response = completed.response
        if response is None:
            raise RuntimeError("Expected a completed response payload.")

        conversation.append(_assistant_message_with_thinking(completed))

        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return

        for tool_call in tool_calls:
            tool_call_id, tool_name, tool_arguments = _function_tool_payload(tool_call)
            tool_args = search_triage_protocol.args_type().model_validate_json(
                tool_arguments
            )
            tool_output = search_triage_protocol.return_value_as_string(
                await search_triage_protocol.run(tool_args, CancellationToken())
            )

            print("Tool execution")
            print()
            print(f"Tool name: {tool_name}")
            print(f"Tool arguments: {tool_arguments}")
            print(f"Tool result: {tool_output}")
            print()

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_output,
                }
            )

    raise RuntimeError("Exceeded the maximum number of internal tool continuations.")


async def run_demo() -> None:
    """Run the Claude streaming chat example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    client = Client(
        Config(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=MODEL_NAME,
        )
    )
    tools = Tools(
        tools=[search_triage_protocol],
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    call_args: dict[str, Any] = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 1024,
        },
        "max_tokens": 1400,
    }

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a clinical triage assistant. Think carefully and you must call "
                "`search_triage_protocol` before you answer. After the tool result "
                "arrives, answer in at most two bullet points. Do not diagnose or "
                "recommend medication changes."
            ),
        },
    ]

    print(f"Claude streaming chat demo using {MODEL_NAME}")
    print("Type a question and press enter. Type 'exit' to quit.")
    print(f"Try: {QUESTION}")
    print()

    while True:
        user_input = _prompt_user()
        if user_input is None:
            return
        if not user_input:
            continue

        conversation.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        print()
        await _run_assistant_turn(
            client,
            conversation=conversation,
            tools=tools,
            extra_call_args=call_args,
        )
        print()


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
