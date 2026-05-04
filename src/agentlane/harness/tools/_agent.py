"""Agent delegation tool for first-party harness base tools."""

from typing import Any

from pydantic import BaseModel

from agentlane.models import Model, ModelResponse, OutputSchema

from .._lifecycle import DefaultAgentTool
from ._types import HarnessToolDefinition

_TOOL_NAME = "agent"
_TOOL_DESCRIPTION = (
    "Spawn a fresh helper agent for one focused task, then continue with the result."
)
_TOOL_PROMPT_SNIPPET = "Delegate a focused task to a fresh helper agent"
_TOOL_PROMPT_GUIDELINE = (
    "Use `agent` for independent subtasks that can run in parallel. Choose a "
    "single-word name and include all context the helper needs in `task`."
)


def agent_tool(
    *,
    model: Model[ModelResponse] | None = None,
    model_args: dict[str, Any] | None = None,
    output_schema: type[BaseModel] | OutputSchema[Any] | None = None,
) -> HarnessToolDefinition:
    """Build the first-party generic spawned-agent harness tool.

    Args:
        model: Optional model override for spawned helper agents.
        model_args: Optional model arguments for spawned helper agents.
        output_schema: Optional structured-output schema for spawned helpers.

    Returns:
        HarnessToolDefinition: Declarative agent tool with prompt metadata.
    """
    return HarnessToolDefinition(
        tool=DefaultAgentTool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            model=model,
            model_args=model_args,
            output_schema=output_schema,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )
