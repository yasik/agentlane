"""OpenAI streaming example with reasoning, preamble, and one tool call."""

import asyncio
import logging
import os
from typing import Any, cast

import structlog
from agentlane_openai import ResponsesClient

from agentlane.models import Config, ModelStreamEvent, ModelStreamEventKind, Tools, as_tool
from agentlane.runtime import CancellationToken

MODEL_NAME = "gpt-5.4-mini"
QUESTION = "Can I return an opened laptop if the screen is cracked? Please look it up."
TOOL_RESULT = (
    "Acme Returns Policy: Opened laptops may be returned within 30 days only when they "
    "are in resellable condition. Accidental damage, including cracked screens, is not "
    "covered by the standard return window."
)


@as_tool
async def search_returns_policy(question: str) -> str:
    """Look up the current Acme returns policy."""
    del question
    return TOOL_RESULT


async def stream_turn(
    client: ResponsesClient,
    *,
    messages: list[dict[str, Any]],
    tools: Tools | None,
    extra_call_args: dict[str, Any] | None,
) -> ModelStreamEvent:
    """Run one streamed turn and print the interesting pieces."""
    reasoning_parts: list[str] = []
    assistant_text_parts: list[str] = []
    tool_argument_parts: list[str] = []
    provider_events: list[str] = []
    provider_message_phases: list[str] = []
    completed_event: ModelStreamEvent | None = None

    async for event in client.stream_response(
        messages=messages,
        tools=tools,
        extra_call_args=extra_call_args,
    ):
        if event.kind == ModelStreamEventKind.REASONING:
            reasoning_value: Any = event.reasoning
            if isinstance(reasoning_value, str):
                reasoning_parts.append(reasoning_value)
            elif reasoning_value is not None:
                reasoning_parts.append(str(reasoning_value))
            continue

        if event.kind == ModelStreamEventKind.TEXT_DELTA:
            if event.text:
                assistant_text_parts.append(event.text)
            continue

        if event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA:
            if event.arguments_delta:
                tool_argument_parts.append(event.arguments_delta)
            continue

        if event.kind == ModelStreamEventKind.PROVIDER and event.provider_event_type:
            provider_events.append(event.provider_event_type)
            raw_item = getattr(event.raw, "item", None)
            raw_phase = getattr(raw_item, "phase", None)
            if isinstance(raw_phase, str):
                provider_message_phases.append(raw_phase)
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

    if reasoning_parts:
        print("Reasoning stream:")
        print("".join(reasoning_parts).strip())
        print()
    if assistant_text_parts:
        if has_tool_calls:
            print("Assistant preamble / thinking steps:")
        else:
            print("Assistant:")
        print("".join(assistant_text_parts).strip())
        print()
    if provider_message_phases:
        print("OpenAI message phases:")
        print(", ".join(dict.fromkeys(provider_message_phases)))
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


async def _run_assistant_turn(
    client: ResponsesClient,
    *,
    conversation: list[dict[str, Any]],
    tools: Tools,
    extra_call_args: dict[str, Any],
) -> None:
    """Run one full assistant turn, including internal tool continuations."""
    used_tools = False

    for _ in range(8):
        completed = await stream_turn(
            client,
            messages=conversation,
            tools=tools if not used_tools else None,
            extra_call_args=extra_call_args,
        )

        response = completed.response
        if response is None:
            raise RuntimeError("Expected a completed response payload.")

        assistant_message = response.choices[0].message.model_dump(mode="json")
        conversation.append(assistant_message)

        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            return

        used_tools = True
        for tool_call in tool_calls:
            tool_call_id, tool_name, tool_arguments = _function_tool_payload(tool_call)
            tool_args = search_returns_policy.args_type().model_validate_json(
                tool_arguments
            )
            tool_output = search_returns_policy.return_value_as_string(
                await search_returns_policy.run(tool_args, CancellationToken())
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
    """Run the OpenAI streaming chat example."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )

    client = ResponsesClient(
        Config(
            api_key=os.environ["OPENAI_API_KEY"],
            model=MODEL_NAME,
        )
    )
    tools = Tools(
        tools=[search_returns_policy],
        tool_choice="required",
        parallel_tool_calls=False,
    )

    conversation: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are Acme support. Before any tool call, first tell the user in one "
                "short two-step preamble what you are about to check. Then call the "
                "tool. After you get the tool result, answer in at most two bullet "
                "points."
            ),
        },
    ]
    extra_call_args = {
        "reasoning": {
            "effort": "low",
            "summary": "detailed",
        },
    }

    print(f"OpenAI streaming chat demo using {MODEL_NAME}")
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
            extra_call_args=extra_call_args,
        )
        print()


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
