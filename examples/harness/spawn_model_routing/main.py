"""Choose a spawned helper's model with explicit application settings."""

import asyncio
import logging
import os
from dataclasses import dataclass
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
    SpawnModelProvider,
    SpawnModelSelection,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.models import Config, Model, ModelResponse, Tools


class SpawnChoice(BaseModel):
    """Optional parent preference exposed by the spawn tool."""

    work_class: Literal["quick", "reasoning"] | None = Field(
        default=None, description="Suggested work class. Application policy decides."
    )


@dataclass
class ExampleProvider:
    """Apply a mandatory route, a parent suggestion, or the default route."""

    routes: dict[str, SpawnModelConfig]
    parameters_model: type[BaseModel] | None = SpawnChoice
    mandatory_route: str | None = None
    default_route: str = "quick"

    def resolve(self, context: SpawnModelContext) -> SpawnModelSelection:
        choice = context.spawn_parameters
        suggestion = choice.work_class if isinstance(choice, SpawnChoice) else None

        # Application requirements take priority over the parent's preference.
        route = self.mandatory_route or suggestion or self.default_route

        return SpawnModelSelection(self.routes[route], label=route)


def create_agent(
    model: Model[ModelResponse], provider: SpawnModelProvider
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


def openrouter_model(name: str) -> Model[ModelResponse]:
    """Build each real client with the same connection settings."""
    return Client(
        Config(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=f"openrouter/{name.removeprefix('openrouter/')}",
            timeout=60,
        )
    )


async def run_demo() -> None:
    """Run one delegation with real OpenRouter clients."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    parent_name = os.environ["OPENROUTER_MODEL"]
    parent = openrouter_model(parent_name)
    quick = openrouter_model(os.environ.get("SPAWN_QUICK_MODEL", "openai/gpt-5.4-mini"))
    reasoning = openrouter_model(os.environ.get("SPAWN_REASONING_MODEL", parent_name))

    # These settings belong to the application. The parent can only suggest.
    provider = ExampleProvider(
        {
            "quick": SpawnModelConfig(
                quick, {"reasoning_effort": "low", "max_tokens": 1024}
            ),
            "reasoning": SpawnModelConfig(
                reasoning, {"reasoning_effort": "medium", "max_tokens": 1024}
            ),
        },
        parameters_model=SpawnChoice,
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
