"""Query local or mounted JSON files with the jq command-line processor."""

import asyncio
import shlex
from math import isfinite
from pathlib import Path, PurePath

from pydantic import BaseModel, Field

from agentlane.harness.filesystem import FileReader, LocalFileSystem
from agentlane.io import buffered_reader
from agentlane.models import (
    Tool,
    ToolError,
    ToolErrorKind,
    ToolExecutionContext,
    ToolFailure,
)
from agentlane.runtime import CancellationToken

from .._constants import EMPTY_PATH_ERROR
from ._bash_executor import (
    BashExecutionRequest,
    BashExecutor,
    LocalBashExecutor,
    StdinBashExecutor,
)
from ._command_output import format_command_output
from ._constants import JQ_TOOL_NAME
from ._output import BASH_MAX_BYTES, BASH_MAX_LINES
from ._paths import StorageToolPathResolver, ToolPathResolver, tool_path_guideline
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
from ._types import HarnessToolDefinition

_TOOL_DESCRIPTION = (
    "Query a JSON file with a jq filter. Supports JSON streams, raw string output, "
    "slurp, and compact output. Returns combined stdout and stderr, tail-truncated "
    f"to {BASH_MAX_LINES} lines or {BASH_MAX_BYTES} bytes."
)
"""Describe jq input and the shared command-output limits to the model."""

_TOOL_PROMPT_SNIPPET = "Select and transform JSON file contents with jq"
"""List the tool in the harness prompt."""

_GENERIC_JQ_ERROR = "failed to run jq"
"""Hide internal details when file access or execution fails unexpectedly."""


class _ToolArgs(BaseModel):
    """Model-visible jq arguments; input bytes and execution policy are host-owned."""

    path: str = Field(description="One literal JSON file path; no wildcard expansion.")
    filter: str = Field(description="jq filter, such as '.items[] | {id, name}'.")
    raw_output: bool = Field(
        default=False,
        description="Write string results without JSON quotes, like jq --raw-output.",
    )
    slurp: bool = Field(
        default=False,
        description="Collect all input JSON values into an array, like jq --slurp.",
    )
    compact_output: bool = Field(
        default=False,
        description="Write compact JSON instead of pretty JSON, like jq --compact-output.",
    )
    timeout: float | None = Field(
        default=None,
        description="Optional positive execution timeout in seconds; uses the host default.",
    )


def jq_tool(
    *,
    cwd: str | Path | None = None,
    reader: FileReader | None = None,
    process_cwd: str | Path | None = None,
    executor: BashExecutor | None = None,
    default_timeout: float | None = None,
    max_input_bytes: int = 8 * 1024 * 1024,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Build the first-party jq tool with native jq parsing and filter semantics.

    Args:
        cwd: File working directory, with the same path rules as `read_tool`.
        reader: Optional file reader. Defaults to the local filesystem.
        process_cwd: Executor working directory for jq imports. Defaults to cwd
            for local files, or the host cwd when a reader is injected. Explicit
            paths for custom executors are passed through without host resolution.
        executor: Command executor with the `StdinBashExecutor` capability enabled.
            The local default requires jq on PATH. Execution is not a filesystem
            sandbox.
        default_timeout: Optional default timeout in seconds for the locally created
            executor. Supplied executors own their default timeout. A model call
            can override the executor's default with a positive finite timeout.
        max_input_bytes: Maximum bytes buffered from one input file; defaults to
            8 MiB. Backend read timeouts remain the reader's responsibility.
        permissions: Shared policy checked for both file reads and command execution.
        approval_callback: Host callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable jq tool and its prompt metadata.
    """
    if default_timeout is not None and (
        not isfinite(default_timeout) or default_timeout <= 0
    ):
        raise ValueError("default_timeout must be finite and greater than zero.")
    if max_input_bytes < 1:
        raise ValueError("max_input_bytes must be greater than zero.")

    # File paths keep read's namespace. Process paths belong to the executor.
    resolver = (
        ToolPathResolver.for_optional(cwd)
        if reader is None
        else StorageToolPathResolver.for_optional(cwd, filesystem=reader)
    )
    file_reader = reader if reader is not None else LocalFileSystem()
    command_cwd = process_cwd
    if command_cwd is None:
        command_cwd = str(resolver.cwd) if reader is None else Path.cwd()
    is_local = executor is None or isinstance(executor, LocalBashExecutor)
    workspace = (
        ToolPathResolver.for_optional(command_cwd).cwd
        if is_local
        else Path(command_cwd)
    )
    command_executor = (
        executor
        if executor is not None
        else LocalBashExecutor(default_timeout=default_timeout)
    )

    async def run_jq(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str | ToolFailure:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        try:
            return await _query_file(
                args,
                resolver=resolver,
                reader=file_reader,
                cwd=workspace,
                validate_local_cwd=is_local,
                executor=command_executor,
                max_input_bytes=max_input_bytes,
                permissions=permissions,
                approval_callback=approval_callback,
                cancellation_token=cancellation_token,
                context=context,
            )
        except Exception:
            return _failure(_GENERIC_JQ_ERROR)

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str | ToolFailure](
            name=JQ_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_jq,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(
            "Use jq to select JSON fields, filter records, and compute summaries. "
            "Pass only a jq filter, without shell quoting or a jq command prefix.",
            "Use narrower filters or array slices when output is large. "
            "Truncated output is a text preview, not a complete JSON document.",
            f"jq reads at most {max_input_bytes} bytes from one file. "
            "Set timeout for filters that may run for a long time; otherwise "
            "the executor's default timeout applies.",
            "jq receives file bytes on stdin. Imports and environment access use "
            f"its executor, with process working directory `{workspace}`. "
            "The default local executor is not filesystem-confined.",
            tool_path_guideline(resolver),
        ),
    )


async def _query_file(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    reader: FileReader,
    cwd: Path,
    validate_local_cwd: bool,
    executor: BashExecutor,
    max_input_bytes: int,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> str | ToolFailure:
    """Validate and authorize a file query, then execute it with bounded input."""
    if not args.path.strip():
        return _failure(EMPTY_PATH_ERROR)
    if not args.filter.strip():
        return _failure("filter must not be empty")
    if args.timeout is not None and (not isfinite(args.timeout) or args.timeout <= 0):
        return _failure("timeout must be finite and greater than zero")

    path = await asyncio.to_thread(resolver.resolve, args.path)
    command = _jq_command(args)

    # jq expressions can import files and read the process environment.
    # A selected-file grant alone must not authorize that execution.
    for request in (
        ToolPermissionRequest(
            tool_name=JQ_TOOL_NAME,
            operation=ToolOperation.READ_FILE,
            cwd=resolver.cwd,
            path=path,
        ),
        ToolPermissionRequest(
            tool_name=JQ_TOOL_NAME,
            operation=ToolOperation.EXECUTE_COMMAND,
            cwd=cwd,
            command=command,
        ),
    ):
        error = await evaluate_tool_permission(
            request,
            policy=permissions,
            approval_callback=approval_callback,
            context=context,
        )
        if error is not None:
            return _failure(error)

    if not isinstance(executor, StdinBashExecutor) or not executor.supports_stdin:
        return _failure("jq requires an executor that supports stdin")

    # A custom executor owns its directory; local configuration errors use the
    # same actionable diagnostics as bash, before the selected input is opened.
    if validate_local_cwd:
        if not cwd.exists():
            return _failure(f"working directory not found: `{cwd}`")
        if not cwd.is_dir():
            return _failure(f"working directory is not a directory: `{cwd}`")

    # Read bounded, unchanged bytes. jq owns JSON parsing, including
    # multiple input values and diagnostics for malformed input.
    pending = asyncio.create_task(
        asyncio.to_thread(_read_input, reader, path, max_input_bytes)
    )
    cancellation_token.link_future(pending)
    content = await pending
    if isinstance(content, str):
        return _failure(content)
    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    result = await executor.run(
        BashExecutionRequest(
            command=command,
            cwd=cwd,
            timeout_seconds=args.timeout,
            stdin=content,
        ),
        cancellation_token,
    )

    return format_command_output(result)


def _jq_command(args: _ToolArgs) -> str:
    """Quote each argument so a jq program cannot become shell syntax."""
    argv = ["jq", "--monochrome-output"]
    if args.raw_output:
        argv.append("--raw-output")
    if args.slurp:
        argv.append("--slurp")
    if args.compact_output:
        argv.append("--compact-output")
    argv.extend(("--", args.filter))

    return shlex.join(argv)


def _read_input(reader: FileReader, path: PurePath, max_bytes: int) -> bytes | str:
    """Read at most the configured input budget plus one overflow sentinel byte."""
    try:
        with reader.open_read(str(path)) as source, buffered_reader(source) as stream:
            content = stream.read(max_bytes + 1)
    except IsADirectoryError:
        return f"path is a directory: `{path}`"
    except FileNotFoundError:
        return f"file not found: `{path}`"
    except PermissionError:
        return f"permission denied: `{path}`"
    except OSError:
        return f"failed to read file: `{path}`"

    if len(content) > max_bytes:
        return f"file exceeds {max_bytes} byte input limit: `{path}`"

    return content


def _failure(message: str) -> ToolFailure:
    """Return stable error text and mark the tool call as failed in the runner."""
    return ToolFailure(
        text=message,
        error=ToolError(message=message, kind=ToolErrorKind.ERROR.value),
    )
