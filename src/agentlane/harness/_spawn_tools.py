"""Typed spawn envelopes built from the final tool catalog."""

from copy import copy
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from agentlane.models import Tools, ToolSpec

from ._lifecycle import AgentTool, DefaultAgentTool, DefaultHandoffTool, HandoffTool


class SpawnToolInput[InputT: BaseModel, ParametersT: BaseModel](BaseModel):
    """Explicit envelope marker; application payload fields cannot mimic it."""

    model_config = ConfigDict(extra="forbid")

    input: InputT = Field(description="Input for the child task.")
    spawn_parameters: ParametersT = Field(
        description="Application routing preferences."
    )


class SpawnToolSchemas:
    """Keep each source tool's schema and parser stable for one run loop."""

    def __init__(self, parameters_model: type[BaseModel]) -> None:
        # A subclass keeps app validators and isinstance checks while rejecting
        # undeclared routing keys, also on Pydantic 2.8.
        self.parameters_model = type(
            parameters_model.__name__,
            (parameters_model,),
            {"model_config": {**parameters_model.model_config, "extra": "forbid"}},
        )
        self._tools: dict[ToolSpec[Any], ToolSpec[Any]] = {}

    def apply(self, tools: Tools) -> Tools:
        adapted: list[ToolSpec[Any]] = []
        for tool in tools.normalized_tools:
            if isinstance(
                tool, (AgentTool, DefaultAgentTool, HandoffTool, DefaultHandoffTool)
            ):
                if not issubclass(tool.args_type(), SpawnToolInput):
                    if tool not in self._tools:
                        envelope = SpawnToolInput.__class_getitem__(
                            (tool.args_type(), self.parameters_model)
                        )

                        # Only declarative spawn tools can change their parser
                        # without changing an executable handler's arguments.
                        derived = copy(tool)
                        spec_type = cast(type[ToolSpec[Any]], ToolSpec)
                        spec_type.__init__(
                            derived,
                            name=tool.name,
                            description=tool.description,
                            args_model=cast(type[BaseModel], envelope),
                            strict=tool.strict,
                        )
                        self._tools[tool] = derived
                    tool = self._tools[tool]
            adapted.append(tool)

        return replace(tools, tools=tuple(adapted))


def unwrap_spawn_input(value: BaseModel) -> tuple[Any, BaseModel | None]:
    """Separate only an explicitly constructed spawn envelope."""
    if isinstance(value, SpawnToolInput):
        envelope = cast(SpawnToolInput[BaseModel, BaseModel], value)
        return envelope.input, envelope.spawn_parameters

    return value, None
