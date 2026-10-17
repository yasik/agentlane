"""Stable names shared by built-in tool definitions and selection."""

DEFAULT_HARNESS_TOOLS_SHIM_NAME = "harness-tools"
"""Used as the default identity and state-key prefix for the built-in tools shim."""

DEFAULT_WORKSPACE_TOOLS_SHIM_NAME = "workspace-tools"
"""Used as the default identity of the workspace tools shim."""

READ_TOOL_NAME = "read"
"""Used to register and select the built-in file-reading tool."""

FIND_TOOL_NAME = "find"
"""Used to register and select the built-in file-search tool."""

GREP_TOOL_NAME = "grep"
"""Used to register and select the built-in content-search tool."""

PATCH_TOOL_NAME = "patch"
"""Used to register and select the built-in file-editing tool."""

WRITE_TOOL_NAME = "write"
"""Used to register and select the built-in file-writing tool."""

BASH_TOOL_NAME = "bash"
"""Used to register and select the built-in shell command tool."""

PLAN_TOOL_NAME = "write_plan"
"""Public, stable name of the first-party plan tool."""

EMPTY_PATTERN_ERROR = "pattern must not be empty"
"""Returned when a search tool receives an empty pattern."""

INVALID_LIMIT_ERROR = "limit must be greater than zero"
"""Returned when a read or find request has a non-positive result limit."""

NULL_BYTE_PATH_ERROR = "path contains a null byte"
"""Returned when a write or patch request contains a null byte in its path."""
