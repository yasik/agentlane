"""Agent delegation tool for first-party harness base tools."""

from typing import Any

from pydantic import BaseModel

from agentlane.models import Model, ModelResponse, OutputSchema, Tools

from .._lifecycle import (
    AgentToolChildToolsFactory,
    AgentToolThreadState,
    DefaultAgentTool,
)
from .._tooling import ToolConfig
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
_DEFAULT_AGENT_MAX_DEPTH = 4
_DEFAULT_AGENT_MAX_THREADS = 16


def agent_tool(
    *,
    model: Model[ModelResponse] | None = None,
    model_args: dict[str, Any] | None = None,
    output_schema: type[BaseModel] | OutputSchema[Any] | None = None,
    agent_max_depth: int = _DEFAULT_AGENT_MAX_DEPTH,
    agent_max_threads: int = _DEFAULT_AGENT_MAX_THREADS,
    _agent_depth: int = 0,
    _agent_thread_state: AgentToolThreadState | None = None,
    _child_tools_factory: AgentToolChildToolsFactory | None = None,
) -> HarnessToolDefinition:
    """Build the first-party generic spawned-agent harness tool.

    Args:
        model: Optional model override for spawned helper agents.
        model_args: Optional model arguments for spawned helper agents.
        output_schema: Optional structured-output schema for spawned helpers.
        agent_max_depth: Maximum recursive spawned-agent depth. A child whose
            computed depth is greater than or equal to this value is rejected.
        agent_max_threads: Maximum number of live spawned agents sharing this
            tool state.

    Returns:
        HarnessToolDefinition: Declarative agent tool with prompt metadata.
    """
    thread_state = _agent_thread_state or AgentToolThreadState()
    child_tools_factory = _child_tools_factory or _default_child_tools_factory(
        agent_max_depth=agent_max_depth,
        agent_max_threads=agent_max_threads,
        agent_thread_state=thread_state,
    )

    return HarnessToolDefinition(
        tool=DefaultAgentTool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            model=model,
            model_args=model_args,
            output_schema=output_schema,
            agent_depth=_agent_depth,
            agent_max_depth=agent_max_depth,
            agent_max_threads=agent_max_threads,
            agent_thread_state=thread_state,
            child_tools_factory=child_tools_factory,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


def _default_child_tools_factory(
    *,
    agent_max_depth: int,
    agent_max_threads: int,
    agent_thread_state: AgentToolThreadState,
) -> AgentToolChildToolsFactory:
    """Return a recursive base-tools factory for spawned helper agents."""

    def build_child_tools(child_depth: int) -> ToolConfig:
        from ._shim import base_harness_tools

        definitions = base_harness_tools(
            agent_max_depth=agent_max_depth,
            agent_max_threads=agent_max_threads,
            _agent_depth=child_depth,
            _agent_thread_state=agent_thread_state,
        )
        return Tools(tools=tuple(definition.tool for definition in definitions))

    return build_child_tools
