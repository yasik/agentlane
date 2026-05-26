"""First-party harness tool definitions and integration helpers."""

from ._agent import agent_tool
from ._bash import bash_tool
from ._bash_executor import (
    BashExecutionRequest,
    BashExecutionResult,
    BashExecutor,
    BashShellConfig,
    LocalBashExecutor,
    resolve_bash_shell,
)
from ._find import find_tool
from ._gitignore import GitignoreMatcher
from ._grep import grep_tool
from ._output import (
    BASH_MAX_BYTES,
    BASH_MAX_LINES,
    FIND_DEFAULT_LIMIT,
    GREP_DEFAULT_LIMIT,
    GREP_MAX_LINE_LENGTH,
    LS_DEFAULT_LIMIT,
    TEXT_MAX_BYTES,
    TEXT_MAX_LINES,
    TruncatedOutput,
    truncate_output,
)
from ._patch import patch_tool
from ._paths import ToolPathResolver
from ._permissions import (
    AllOfToolPermissionPolicy,
    AllowAllToolPermissionPolicy,
    PathScopeToolPermissionPolicy,
    SideEffectApprovalToolPermissionPolicy,
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionDecision,
    ToolPermissionGrant,
    ToolPermissionGrantPolicy,
    ToolPermissionOutcome,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
    evaluate_tool_permission,
    format_tool_permission_result,
    parse_tool_permission_grants,
    workspace_tool_policy,
)
from ._plan import plan_tool
from ._read import read_tool
from ._shim import HarnessToolsShim, base_harness_tools
from ._types import HarnessToolDefinition
from ._write import write_tool

__all__ = [
    "BASH_MAX_BYTES",
    "BASH_MAX_LINES",
    "BashExecutionRequest",
    "BashExecutionResult",
    "BashExecutor",
    "BashShellConfig",
    "FIND_DEFAULT_LIMIT",
    "GREP_DEFAULT_LIMIT",
    "GREP_MAX_LINE_LENGTH",
    "GitignoreMatcher",
    "HarnessToolDefinition",
    "HarnessToolsShim",
    "LS_DEFAULT_LIMIT",
    "LocalBashExecutor",
    "TEXT_MAX_BYTES",
    "TEXT_MAX_LINES",
    "AllOfToolPermissionPolicy",
    "AllowAllToolPermissionPolicy",
    "PathScopeToolPermissionPolicy",
    "SideEffectApprovalToolPermissionPolicy",
    "ToolPathResolver",
    "ToolApprovalCallback",
    "ToolOperation",
    "ToolPermissionDecision",
    "ToolPermissionGrant",
    "ToolPermissionGrantPolicy",
    "ToolPermissionOutcome",
    "ToolPermissionPolicy",
    "ToolPermissionRequest",
    "TruncatedOutput",
    "WorkspaceToolPermissionPolicy",
    "agent_tool",
    "base_harness_tools",
    "bash_tool",
    "evaluate_tool_permission",
    "find_tool",
    "format_tool_permission_result",
    "grep_tool",
    "patch_tool",
    "plan_tool",
    "parse_tool_permission_grants",
    "read_tool",
    "resolve_bash_shell",
    "truncate_output",
    "write_tool",
    "workspace_tool_policy",
]
