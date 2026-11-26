"""Choose a spawned helper's model with explicit application settings."""

import asyncio
import logging
import os
from typing import Literal

import structlog
from agentlane_litellm import Client
from pydantic import BaseModel, Field

from agentlane.harness import (
    AgentDescriptor,
    DefaultAgentTool,
    Runner,
    SpawnModelConfig,
    SpawnModelContext,
    SpawnModelSelection,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Model, ModelResponse, Tools

type WorkClass = Literal["quick", "reasoning"]


class SpawnChoice(BaseModel):
    """Optional parent preference exposed by the spawn tool."""

    work_class: WorkClass | None = Field(
        default=None, description="Suggested work class. Application policy decides."
    )


class RoutingProvider:
    """Apply a mandatory route, a parent suggestion, or the default route."""

    def __init__(
        self,
        routes: dict[WorkClass, SpawnModelConfig],
        *,
        model_can_choose: bool,
        mandatory_route: WorkClass | None = None,
        default_route: WorkClass = "quick",
    ) -> None:
        self.routes = routes
        self.parameters_model = SpawnChoice if model_can_choose else None
        self.mandatory_route: WorkClass | None = mandatory_route
        self.default_route: WorkClass = default_route

    def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
        choice = context.spawn_parameters
        suggestion = choice.work_class if isinstance(choice, SpawnChoice) else None

        # Application requirements take priority over the parent's preference.
        route = self.mandatory_route or suggestion or self.default_route

        return SpawnModelSelection(self.routes[route], label=route)


def create_agent(
    model: Model[ModelResponse], provider: RoutingProvider
) -> DefaultAgent:
    """Connect application policy to the generic agent tool."""
    return DefaultAgent(
        runner=Runner(spawn_model_provider=provider, max_turns=4),
        descriptor=AgentDescriptor(
            name="Manager",
            model=model,
            model_args={"max_tokens": 1024},
            instructions=(
                "Delegate the calculation to agent exactly once. "
                "If spawn_parameters is available, suggest the reasoning work class. "
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


async def run_demo() -> None:
    """Run one delegation with real OpenRouter clients."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENROUTER_API_KEY"]
    parent_name = os.environ["OPENROUTER_MODEL"].removeprefix("openrouter/")
    quick_name = os.environ.get(
        "SPAWN_QUICK_MODEL", "openai/gpt-5.4-mini"
    ).removeprefix("openrouter/")
    reasoning_name = os.environ.get("SPAWN_REASONING_MODEL", parent_name).removeprefix(
        "openrouter/"
    )

    parent = Client(
        Config(api_key=api_key, model=f"openrouter/{parent_name}", timeout=60)
    )
    quick = Client(
        Config(api_key=api_key, model=f"openrouter/{quick_name}", timeout=60)
    )
    reasoning = Client(
        Config(api_key=api_key, model=f"openrouter/{reasoning_name}", timeout=60)
    )

    # These settings belong to the application. The parent can only suggest.
    provider = RoutingProvider(
        {
            "quick": SpawnModelConfig(
                quick, {"reasoning_effort": "low", "max_tokens": 1024}
            ),
            "reasoning": SpawnModelConfig(
                reasoning, {"reasoning_effort": "medium", "max_tokens": 1024}
            ),
        },
        model_can_choose=True,
        mandatory_route=None,
        default_route="quick",
    )
    agent = create_agent(parent, provider)
    user_message = "Compute 17 multiplied by 19. Return only the integer result."
    async with asyncio.timeout(240):
        result = await agent.run(user_message)

    print("Example: spawn model routing")
    print(f"User: {user_message}")
    print(f"Assistant: {result.final_output}")


def main() -> None:
    """Run the example from the command line."""
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
