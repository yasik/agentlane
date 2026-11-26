"""Exercise external spawn routing through the example's public agent setup."""

import asyncio
import json
import runpy
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Self
from unittest.mock import AsyncMock

import httpx
import pytest

from agentlane.harness import SpawnModelConfig

from .tools_test_utils import SequenceModel, make_assistant_response, make_tool_call

TASK = "Multiply 17 by 19."


class _ChildModel(SequenceModel):
    def __init__(self) -> None:
        super().__init__([make_assistant_response("323")])
        self.started = False

    @asynccontextmanager
    async def open_run(self) -> AsyncGenerator[Self, None]:
        self.started = True
        yield self


@pytest.fixture(name="example")
def fixture_example() -> dict[str, Any]:
    path = (
        Path(__file__).parents[2] / "examples/harness/spawn_model_opendecisions/main.py"
    )

    return runpy.run_path(str(path))


@pytest.fixture(name="routes")
def fixture_routes() -> dict[str, SpawnModelConfig]:
    return {
        name: SpawnModelConfig(
            _ChildModel(), {"reasoning_effort": effort, "max_tokens": 300}
        )
        for name, effort in (("quick", "low"), ("reasoning", "medium"))
    }


def _started(routes: dict[str, SpawnModelConfig]) -> list[str]:
    started: list[str] = []
    for name, config in routes.items():
        assert isinstance(config.model, _ChildModel)
        if config.model.started:
            started.append(name)

    return started


def _parent() -> SequenceModel:
    return SequenceModel(
        [
            make_assistant_response(
                None,
                tool_calls=[
                    make_tool_call(
                        tool_id="delegate",
                        name="agent",
                        arguments=json.dumps({"name": "Calculator", "task": TASK}),
                    )
                ],
            ),
            make_assistant_response("323"),
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["quick", "reasoning"])
async def test_decision_selects_client_and_args_before_child_starts(
    example: dict[str, Any],
    routes: dict[str, SpawnModelConfig],
    route: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    selected = routes[route].model
    get_response = AsyncMock(wraps=selected.get_response)
    monkeypatch.setattr(selected, "get_response", get_response)

    def decide(request: httpx.Request) -> httpx.Response:
        assert _started(routes) == []
        requests.append(request)

        return httpx.Response(200, json={"results": [{"id": "route", "answer": route}]})

    parent = _parent()
    async with httpx.AsyncClient(
        base_url="https://decisions.test",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(decide),
    ) as client:
        provider = example["RoutingProvider"](client, routes, model="router-model")
        result = await example["create_agent"](parent, provider).run(TASK)

    assert result.final_output == "323"
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://decisions.test/v1/decisions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "router-model"
    assert body["context"] == TASK
    assert len(body["questions"]) == 1
    assert body["questions"][0]["id"] == "route"
    assert set(body["questions"][0]["options"]) == set(routes)
    assert _started(routes) == [route]
    get_response.assert_awaited_once()
    assert get_response.call_args.kwargs["extra_call_args"] == routes[route].model_args
    tools = parent.call_tools[0]
    assert tools is not None
    schema = tools.normalized_tools[0].args_type().model_json_schema()
    assert "spawn_parameters" not in schema["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, results, error",
    [
        (200, [{"id": "route", "answer": "unknown"}], "results.0.answer"),
        (200, [{"id": "wrong", "answer": "quick"}], "results.0.id"),
        (200, [], "Field required"),
        (200, [{"id": "route", "answer": "quick"}] * 2, "at most 1 item"),
        (503, [], "503 Service Unavailable"),
    ],
)
async def test_invalid_decision_does_not_start_child(
    example: dict[str, Any],
    routes: dict[str, SpawnModelConfig],
    status: int,
    results: list[dict[str, str]],
    error: str,
) -> None:
    requests: list[httpx.Request] = []

    def decide(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(status, json={"results": results})

    async with httpx.AsyncClient(
        base_url="https://decisions.test", transport=httpx.MockTransport(decide)
    ) as client:
        provider = example["RoutingProvider"](client, routes)
        # DefaultAgent reports failed runtime delivery as a RuntimeError.
        with pytest.raises(RuntimeError, match=error):
            await example["create_agent"](_parent(), provider).run(TASK)

    assert len(requests) == 1
    assert _started(routes) == []


@pytest.mark.asyncio
async def test_cancelled_decision_does_not_start_child(
    example: dict[str, Any], routes: dict[str, SpawnModelConfig]
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def decide(request: httpx.Request) -> httpx.Response:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

        raise AssertionError("The cancelled request must not return a decision.")

    async with httpx.AsyncClient(
        base_url="https://decisions.test", transport=httpx.MockTransport(decide)
    ) as client:
        provider = example["RoutingProvider"](client, routes)
        run = asyncio.create_task(
            example["create_agent"](_parent(), provider).run(TASK)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        finally:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert cancelled.is_set()
    assert _started(routes) == []
