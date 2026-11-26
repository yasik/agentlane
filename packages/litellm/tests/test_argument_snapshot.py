"""Harness argument snapshots retain native clients and own request data."""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Self, cast

import httpx2
import openai
import pytest
from agentlane_litellm import Client

from agentlane.harness import (
    Agent,
    AgentDescriptor,
    Runner,
    RunState,
    SpawnModelConfig,
    SpawnModelContext,
    SpawnModelSelection,
)
from agentlane.harness.shims import PreparedTurn, Shim
from agentlane.models import Config, Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


class _TrackedClient(Client):
    opens = 0

    @asynccontextmanager
    async def open_run(self) -> AsyncGenerator[Self, None]:
        self.opens += 1
        yield self


@pytest.mark.asyncio
async def test_runner_snapshots_data_and_retains_native_sdk_client() -> None:
    requests: list[dict[str, Any]] = []
    contexts: list[SpawnModelContext] = []
    prepared: list[dict[str, Any]] = []
    parent_requests = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal parent_requests
        body = json.loads(request.content)
        requests.append(body)
        child_request = body["model"] == "gpt-4o-mini"
        if not child_request:
            parent_requests += 1
        tool_calls = (
            [
                {
                    "id": "spawn_child",
                    "type": "function",
                    "function": {"name": "child", "arguments": "{}"},
                }
            ]
            if not child_request and parent_requests == 1
            else None
        )
        return httpx2.Response(
            200,
            json={
                "id": f"response_{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                        "message": {
                            "role": "assistant",
                            "content": None if tool_calls else "done",
                            "tool_calls": tool_calls,
                        },
                    }
                ],
            },
        )

    async with openai.AsyncOpenAI(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    ) as native:
        parent_model = _TrackedClient(
            Config(api_key="test", model="openai/gpt-4o", max_retries=0)
        )
        child_model = _TrackedClient(
            Config(api_key="test", model="openai/gpt-4o-mini", max_retries=0)
        )
        timeout = openai.Timeout(2.0)
        extra_body = {"snapshot_data": {"items": [1]}}

        class RequestShim(Shim):
            @property
            def name(self) -> str:
                return "request-snapshot"

            async def prepare_turn(self, turn: PreparedTurn) -> None:
                assert turn.model_args is not None
                prepared.append(turn.model_args)
                assert turn.model_args["client"] is native
                current_timeout = cast(openai.Timeout, turn.model_args["timeout"])
                assert current_timeout.read == 2.0
                assert current_timeout is not timeout
                body = cast(dict[str, Any], turn.model_args["extra_body"])
                assert body == {"snapshot_data": {"items": [1]}}
                body["snapshot_data"]["items"].append(2)

        # The descriptor has no model yet; copying cannot depend on an adapter.
        child_descriptor = AgentDescriptor(
            name="child",
            model_args={"client": native, "timeout": timeout, "extra_body": extra_body},
            shims=[RequestShim()],
            tools=None,
        )
        assert child_descriptor.model is None
        assert child_descriptor.model_args is not None
        assert child_descriptor.model_args["client"] is native
        timeout.read = 9.0
        extra_body["snapshot_data"]["items"].append(9)

        class Provider:
            parameters_model = None

            def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
                contexts.append(context)
                assert context.baseline.model is parent_model
                assert context.baseline.model_args is not None
                assert context.baseline.model_args["client"] is native
                return SpawnModelSelection(
                    SpawnModelConfig(child_model, context.baseline.model_args)
                )

        runner = Runner(spawn_model_provider=Provider())
        runtime = SingleThreadedRuntimeEngine()
        parent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="parent",
                model=parent_model,
                model_args={"client": native},
                tools=Tools(tools=[child_descriptor.as_tool()]),
            ),
        )
        try:
            result = await runner.run(
                parent, RunState(instructions=None, history=["start"], responses=[])
            )
            assert result.final_output == "done"
        finally:
            await runtime.stop_when_idle()

        # Calls used the configured objects and the isolated shim request data.
        assert parent_model.opens == child_model.opens == 1
        assert len(contexts) == len(prepared) == 1
        assert requests[1]["snapshot_data"] == {"items": [1, 2]}
        assert contexts[0].baseline.model_args is not None
        assert contexts[0].baseline.model_args["extra_body"] == {
            "snapshot_data": {"items": [1]}
        }
        assert child_descriptor.model_args["extra_body"] == {
            "snapshot_data": {"items": [1]}
        }
        assert not native.is_closed()
