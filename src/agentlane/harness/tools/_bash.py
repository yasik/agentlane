"""Bash tool implementation for first-party harness base tools."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._bash_executor import (
    BashExecutionRequest,
    BashExecutionResult,
    BashExecutor,
    LocalBashExecutor,
)
from ._output import TruncatedOutput
from ._paths import ToolPathResolver
from ._types import HarnessToolDefinition

_TOOL_NAME = "bash"
_TOOL_DESCRIPTION = (
    "Execute a non-interactive command with `bash -lc` in the current working "
    "directory. Returns combined stdout and stderr. Output is tail-truncated "
    "to the last 2000 lines or 51200 bytes."
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
    "The bash tool does not provide sandboxing, approvals, or permission "
    "enforcement.",
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


@dataclass(frozen=True, slots=True)
class BashPolicyDecision:
    """Decision returned by a bash policy check."""

    allowed: bool
    reason: str | None = None


class BashPolicy(Protocol):
    """Pre-execution policy hook for bash commands."""

    def check(self, request: BashExecutionRequest) -> BashPolicyDecision:
        """Return whether the command should execute."""
        ...


class _AllowBashPolicy:
    """Default permissive policy for local bash execution."""

    def check(self, request: BashExecutionRequest) -> BashPolicyDecision:
        del request
        return BashPolicyDecision(allowed=True)


def bash_tool(
    *,
    cwd: str | Path | None = None,
    default_timeout: float | None = None,
    executor: BashExecutor | None = None,
    policy: BashPolicy | None = None,
) -> HarnessToolDefinition:
    """Build the first-party bash harness tool.

    Args:
        cwd: Optional working directory used to resolve relative command
            execution. When omitted, the current working directory is captured
            at construction time.
        default_timeout: Optional default timeout in seconds for calls that do
            not provide their own timeout.
        executor: Optional executor implementation for tests or host
            applications.
        policy: Optional pre-execution policy hook.

    Returns:
        HarnessToolDefinition: Executable bash tool with prompt metadata.
    """
    if default_timeout is not None and default_timeout <= 0:
        raise ValueError("default_timeout must be greater than zero.")

    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))
    bash_executor = executor or LocalBashExecutor(default_timeout=default_timeout)
    bash_policy = policy or _AllowBashPolicy()

    async def run_bash(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        try:
            return await _run_bash(
                args,
                cwd=resolver.cwd,
                executor=bash_executor,
                policy=bash_policy,
                cancellation_token=cancellation_token,
            )
        except Exception:
            return _GENERIC_BASH_ERROR

    return HarnessToolDefinition(
        tool=Tool(
            name=_TOOL_NAME,
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
    executor: BashExecutor,
    policy: BashPolicy,
    cancellation_token: CancellationToken,
) -> str:
    """Validate model arguments, execute the command, and render the result."""
    if args.command.strip() == "":
        return "command must not be empty"
    if args.timeout is not None and args.timeout <= 0:
        return "timeout must be greater than zero"

    if cancellation_token.is_cancelled:
        return _format_bash_output(
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

    if not cwd.exists():
        return f"working directory not found: `{cwd}`"
    if not cwd.is_dir():
        return f"working directory is not a directory: `{cwd}`"

    request = BashExecutionRequest(
        command=args.command,
        cwd=cwd,
        timeout_seconds=args.timeout,
    )
    decision = policy.check(request)
    if not decision.allowed:
        return decision.reason or "bash command denied"

    result = await executor.run(request, cancellation_token)
    return _format_bash_output(result)


def _format_bash_output(result: BashExecutionResult) -> str:
    """Render the final model-facing tool result."""
    output = result.output.text.rstrip("\n") or "(no output)"
    notices: list[str] = []

    if result.output_truncated and result.full_output_path is not None:
        notices.append(
            "Showing last "
            f"2000 lines or 51200 bytes. Full output: {result.full_output_path}"
        )

    if result.timed_out:
        if result.timeout_seconds is None:
            notices.append("Command timed out")
        else:
            notices.append(
                f"Command timed out after {_format_seconds(result.timeout_seconds)} seconds"
            )
    elif result.cancelled:
        notices.append("Command cancelled")
    elif result.exit_code is not None and result.exit_code != 0:
        notices.append(f"Command exited with code {result.exit_code}")

    if not notices:
        return output

    return output + "\n\n" + "\n".join(f"[{notice}]" for notice in notices)


def _format_seconds(seconds: float) -> str:
    numeric_seconds = float(seconds)
    if numeric_seconds.is_integer():
        return str(int(numeric_seconds))
    return str(seconds)
