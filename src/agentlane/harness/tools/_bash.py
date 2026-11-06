"""Bash tool implementation for first-party harness base tools."""

from pathlib import Path

from pydantic import BaseModel, Field

from agentlane.models import (
    Tool,
    ToolError,
    ToolErrorKind,
    ToolExecutionContext,
    ToolFailure,
)
from agentlane.runtime import CancellationToken

from ._bash_executor import (
    BashExecutionRequest,
    BashExecutionResult,
    BashExecutor,
    LocalBashExecutor,
)
from ._command_output import format_command_output
from ._constants import BASH_TOOL_NAME
from ._output import BASH_MAX_BYTES, BASH_MAX_LINES, TruncatedOutput
from ._paths import ToolPathResolver
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
from ._types import HarnessToolDefinition

_TOOL_DESCRIPTION = (
    "Execute a non-interactive command with `bash -lc` in the current working "
    "directory. Returns combined stdout and stderr. Output is tail-truncated "
    f"to the last {BASH_MAX_LINES} lines or {BASH_MAX_BYTES} bytes."
)
_TOOL_PROMPT_SNIPPET = "Execute non-interactive bash commands"
_TOOL_PROMPT_GUIDELINES = (
    "Use dedicated file tools for direct file reads, writes, searches, and "
    "patches when they fit.",
    "Use bash for shell workflows, short inspection commands, and commands "
    "that need existing CLIs.",
    "Prefer `rg` over `grep` or `find` when searching from bash.",
    "Avoid interactive commands; bash does not accept follow-up stdin.",
    "Set `timeout` for commands that may hang or run for a long time.",
    "The default local bash executor is not filesystem-confined; pass a "
    "permission policy or sandboxed executor when an application needs a "
    "stricter boundary.",
)
_GENERIC_BASH_ERROR = "failed to execute bash command"


class _ToolArgs(BaseModel):
    """Model-visible arguments for the bash tool."""

    command: str = Field(
        description="The non-interactive bash command to execute with `bash -lc`."
    )
    timeout: float | None = Field(
        default=None,
        description="Optional timeout in seconds for this command.",
    )


def bash_tool(
    *,
    cwd: str | Path | None = None,
    default_timeout: float | None = None,
    executor: BashExecutor | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Build the first-party bash harness tool.

    Args:
        cwd: Optional working directory used to resolve relative command
            execution. When omitted, the current working directory is captured
            at construction time.
        default_timeout: Optional default timeout in seconds for calls that do
            not provide their own timeout.
        executor: Optional executor implementation for tests or host
            applications. An explicit cwd is passed to custom executors without
            host resolution or tilde expansion. Relative paths are executor-owned.
        permissions: Optional shared policy for command-execution decisions.
        approval_callback: Optional callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable bash tool with prompt metadata.
    """
    if default_timeout is not None and default_timeout <= 0:
        raise ValueError("default_timeout must be greater than zero.")

    # Custom executors own path resolution. A sandbox can interpret the same
    # path differently from the host's symlink and home-directory rules.
    is_local = executor is None or isinstance(executor, LocalBashExecutor)
    if is_local:
        workspace = ToolPathResolver.for_optional(cwd).cwd
    elif cwd is not None:
        workspace = Path(cwd)
    else:
        workspace = Path.cwd()

    bash_executor = executor or LocalBashExecutor(default_timeout=default_timeout)

    async def run_bash(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str | ToolFailure:
        try:
            return await _run_bash(
                args,
                cwd=workspace,
                validate_local_cwd=is_local,
                executor=bash_executor,
                permissions=permissions,
                approval_callback=approval_callback,
                cancellation_token=cancellation_token,
                context=context,
            )
        except Exception:
            # A crashed handler must still mark the call as failed, so wrap the
            # unchanged model-facing text in a ``ToolFailure`` rather than
            # returning a plain string the runner would read as success.
            return ToolFailure(
                text=_GENERIC_BASH_ERROR,
                error=ToolError(
                    message=_GENERIC_BASH_ERROR, kind=ToolErrorKind.ERROR.value
                ),
            )

    return HarnessToolDefinition(
        # ``ToolFailure`` is a ``str`` subclass, so the default formatter renders
        # it as its model-facing text unchanged while the runner reads the
        # structured outcome off the same result.
        tool=Tool[_ToolArgs, str | ToolFailure](
            name=BASH_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_bash,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=_TOOL_PROMPT_GUIDELINES,
    )


async def _run_bash(
    args: _ToolArgs,
    *,
    cwd: Path,
    validate_local_cwd: bool,
    executor: BashExecutor,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> str | ToolFailure:
    """Validate model arguments, execute the command, and render the result."""
    if args.command.strip() == "":
        return "command must not be empty"
    if args.timeout is not None and args.timeout <= 0:
        return "timeout must be greater than zero"

    if cancellation_token.is_cancelled:
        return format_command_output(
            BashExecutionResult(
                command=args.command,
                cwd=cwd,
                exit_code=None,
                timed_out=False,
                cancelled=True,
                timeout_seconds=None,
                output=TruncatedOutput(text="", truncated=False),
                full_output_path=None,
            )
        )

    request = BashExecutionRequest(
        command=args.command,
        cwd=cwd,
        timeout_seconds=args.timeout,
    )
    permission_error = await evaluate_tool_permission(
        ToolPermissionRequest(
            tool_name=BASH_TOOL_NAME,
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=cwd,
            command=args.command,
        ),
        policy=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    # Remote workspaces may not exist on the host; their executor validates them.
    if validate_local_cwd:
        if not cwd.exists():
            return f"working directory not found: `{cwd}`"
        if not cwd.is_dir():
            return f"working directory is not a directory: `{cwd}`"

    result = await executor.run(request, cancellation_token)
    return format_command_output(result)
