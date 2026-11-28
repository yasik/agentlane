"""Let OpenDecisions select a configured model before a helper starts."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Literal

import httpx
import structlog
from agentlane_litellm import Client
from pydantic import BaseModel

from agentlane.harness import (
    AgentDescriptor,
    DefaultAgentTool,
    Runner,
    SpawnModelConfig,
    SpawnModelContext,
    SpawnModelProvider,
    SpawnModelSelection,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Model, ModelResponse, Tools


class RouteResult(BaseModel):
    """Validate the named choice received from the decision service."""

    id: Literal["route"]
    answer: Literal["quick", "reasoning"]


class DecisionResponse(BaseModel):
    """Require exactly one answer to the routing question."""

    results: tuple[RouteResult]


@dataclass
class RoutingProvider:
    """Ask OpenDecisions once for each new child."""

    client: httpx.AsyncClient
    routes: dict[str, SpawnModelConfig]
    model: str = "auto"
    parameters_model: None = None

    async def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
        task = context.task
        if isinstance(task, BaseModel):
            task = task.model_dump(mode="json")

        # The service chooses between application-configured clients.
        response = await self.client.post(
            "/v1/decisions",
            json={
                "model": self.model,
                "context": task,
                "questions": [
                    {
                        "id": "route",
                        "question": "Which model should handle this task?",
                        "options": {
                            "quick": "Simple arithmetic, extraction, or short factual tasks.",
                            "reasoning": "Tasks that require complex reasoning or planning.",
                        },
                    }
                ],
            },
        )
        response.raise_for_status()
        route = DecisionResponse.model_validate(response.json()).results[0].answer

        return SpawnModelSelection(self.routes[route], label=route)


def create_agent(
    model: Model[ModelResponse], provider: SpawnModelProvider
) -> DefaultAgent:
    """Connect the external decision provider to the generic agent tool."""
    return DefaultAgent(
        runner=Runner(spawn_model_provider=provider, max_turns=4),
        descriptor=AgentDescriptor(
            name="Manager",
            model=model,
            model_args={"max_tokens": 1024},
            instructions=(
                "Delegate the calculation to agent exactly once. "
                "After the helper returns, reply with only the integer result."
            ),
            tools=Tools(
                tools=[DefaultAgentTool(tools=None)],
                tool_choice="required",
                parallel_tool_calls=False,
                tool_call_limits={"agent": 1},
            ),
        ),
    )


def openrouter_model(name: str) -> Model[ModelResponse]:
    """Build a real model client with explicit connection settings."""
    return Client(
        Config(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=f"openrouter/{name.removeprefix('openrouter/')}",
            timeout=60,
        )
    )


async def run_demo() -> None:
    """Run a delegation with an external decision and real model clients."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    parent_name = os.environ["OPENROUTER_MODEL"]
    routes = {
        "quick": SpawnModelConfig(
            openrouter_model(
                os.environ.get("SPAWN_QUICK_MODEL", "openai/gpt-5.4-mini")
            ),
            {"reasoning_effort": "low", "max_tokens": 1024},
        ),
        "reasoning": SpawnModelConfig(
            openrouter_model(os.environ.get("SPAWN_REASONING_MODEL", parent_name)),
            {"reasoning_effort": "medium", "max_tokens": 1024},
        ),
    }
    headers: dict[str, str] = {}
    if api_key := os.environ.get("OPENDECISIONS_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"

    # Keep the decision connection open until the agent run completes.
    async with httpx.AsyncClient(
        base_url=os.environ.get("OPENDECISIONS_BASE_URL", "http://127.0.0.1:8000"),
        headers=headers,
        timeout=130,
    ) as client:
        provider = RoutingProvider(
            client, routes, model=os.environ.get("OPENDECISIONS_MODEL", "auto")
        )
        agent = create_agent(openrouter_model(parent_name), provider)
        user_message = "Compute 17 multiplied by 19. Return only the integer result."
        async with asyncio.timeout(300):
            result = await agent.run(user_message)

    print("Example: spawn model selection through OpenDecisions")
    print(f"User: {user_message}")
    print(f"Assistant: {result.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
