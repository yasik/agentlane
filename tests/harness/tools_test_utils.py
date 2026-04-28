import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

from agentlane.harness import RunState
from agentlane.harness.tools import HarnessToolDefinition
from agentlane.models import MessageDict, Model, ModelResponse, Tool, ToolCall, Tools
from agentlane.runtime import CancellationToken


class EchoArgs(BaseModel):
    text: str


async def echo(args: EchoArgs, cancellation_token: CancellationToken) -> str:
    del cancellation_token
    return args.text


def echo_tool(name: str) -> Tool[EchoArgs, str]:
    return Tool(
        name=name,
        description=f"{name} tool.",
        args_model=EchoArgs,
        handler=echo,
    )


def run_state(*, turn_count: int = 1) -> RunState:
    return RunState(
        instructions="Base",
        history=[],
        responses=[],
        turn_count=turn_count,
    )


def run_tool(definition: HarnessToolDefinition, **arguments: object) -> str:
    args_model = definition.tool.args_type()
    return asyncio.run(
        definition.tool.run(
            args_model(**arguments),
            CancellationToken(),
        )
    )


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


class FakeRipgrepResult:
    def __init__(self, output: str) -> None:
        self._output = output

    @property
    def as_string(self) -> str:
        return self._output


def make_assistant_response(
    content: str | None,
    *,
    tool_calls: list[ToolCall] | None = None,
) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                }
            ],
        }
    )


def make_tool_call(*, tool_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall.model_validate(
        {
            "id": tool_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }
    )


class SequenceModel(Model[ModelResponse]):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[MessageDict]] = []
        self.call_tools: list[Tools | None] = []

    async def get_response(
        self,
        messages: list[MessageDict],
        extra_call_args: dict[str, object] | None = None,
        schema: object | None = None,
        tools: Tools | None = None,
        cancellation_token: CancellationToken | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        del extra_call_args
        del schema
        del cancellation_token
        del kwargs

        self.calls.append([dict(message) for message in messages])
        self.call_tools.append(tools)
        return self._responses.pop(0)
