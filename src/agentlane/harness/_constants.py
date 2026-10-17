"""Built-in harness names, descriptions, and shared messages."""

DEFAULT_SKILLS_SHIM_NAME = "skills"
"""Used as the default identity and state-key prefix for the skills shim."""

DEFAULT_EXCLUDE_TOOLS_SHIM_NAME = "exclude-tools"
"""Used as the default identity of the tool-exclusion shim."""

DEFAULT_AGENT_NAME_WORD = "agent"
"""Used when a generated agent-name word becomes empty after normalization."""

DEFAULT_DELEGATION_TOOL_NAME = "delegate"
"""Used when a delegation label contains no characters valid in a tool name."""

DEFAULT_AGENT_TOOL_NAME = "agent"
"""Used to register and select the generic helper-agent tool."""

DEFAULT_AGENT_TOOL_DESCRIPTION = (
    "Spawn a focused helper agent by name and task, then continue with the result."
)
"""Shown to the model when the generic helper-agent tool has no custom description."""

DEFAULT_HANDOFF_TOOL_NAME = "handoff"
"""Used as the default model-facing name of the fresh-agent handoff tool."""

DEFAULT_HANDOFF_TOOL_DESCRIPTION = (
    "Transfer the conversation to a fresh helper agent and continue there."
)
"""Shown to the model to describe the default fresh-agent handoff."""

DEFAULT_HANDOFF_AGENT_NAME = "Delegated Agent"
"""Used to identify the fresh child created by a default handoff."""

DEFAULT_HANDOFF_AGENT_DESCRIPTION = "Fresh delegated handoff agent."
"""Used in the descriptor of the fresh child created by a default handoff."""

AGENT_TOOL_DESCRIPTION_TEMPLATE = (
    "Delegate a focused task to {agent_name} and continue with the result."
)
"""Used to describe a named agent tool when no custom description is supplied."""

HANDOFF_DESCRIPTION_TEMPLATE = "Transfer the conversation to {agent_name}."
"""Used to describe a named handoff target when no custom description is supplied."""

HANDOFF_TOOL_RESULT_TEMPLATE = "Handoff accepted by {tool_name}. Control transfers now."
"""Recorded as the tool result in transferred history when a handoff is accepted."""

DEFAULT_HANDOFF_TASK_MESSAGE = (
    "You are being delegated this conversation. Continue from the "
    "transferred history and take over from here."
)
"""Added to transferred history when the handoff supplies no task message."""

DEFAULT_AGENT_TOOL_INSTRUCTIONS = (
    "You are a newly spawned agent in a team of agents collaborating to "
    "complete a task. You can spawn sub-agents to handle subtasks, and "
    "those sub-agents can spawn their own sub-agents. Return the response "
    "to your assigned task directly; that response will be delivered back "
    "to your parent agent. Treat the next user message as your assigned "
    "task, and use any available prior history only as background context."
)
"""Used to instruct a generic helper agent when no custom instructions are supplied."""

AGENT_DEPTH_LIMIT_REACHED = "Agent depth limit reached. Solve the task yourself."
"""Returned to the model when another delegated child would exceed the depth limit."""

AGENT_THREAD_LIMIT_REACHED = "Agent thread limit reached. Solve the task yourself."
"""Returned to the model when no concurrent delegated-agent slot is available."""

MISSING_MODEL_CHOICE = (
    "Runner expected the model response to contain at least one choice."
)
"""Used in model-behavior errors when a response contains no choice to process."""

CHANGED_TOOL_CALL_ARGUMENTS = "Repeated call ID changed its arguments."
"""Used in model-behavior errors when a repeated call ID changes its arguments."""

INVALID_DIRECTORY_ENTRY = "Directory entries must contain one child name"
"""Used in errors when a directory entry is not a single normalized child name."""

EMPTY_PATH_ERROR = "path must not be empty"
"""Returned or raised when a file operation receives an empty path."""
