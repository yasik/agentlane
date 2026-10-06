"""Write tool implementation for first-party harness base tools."""

import asyncio
from pathlib import Path, PurePath

from pydantic import BaseModel, Field

from agentlane.harness.filesystem import FileInfo, LocalFileSystem, WritableFileSystem
from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ._paths import StorageToolPathResolver, ToolPathResolver, tool_path_guideline
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
from ._storage_write import write_content
from ._types import HarnessToolDefinition

_TOOL_NAME = "write"
_TOOL_DESCRIPTION = (
    "Write content to a file. Creates the file if it does not exist, overwrites "
    "it if it does, and automatically creates parent directories."
)
_TOOL_PROMPT_SNIPPET = "Create or overwrite files"
_TOOL_PROMPT_GUIDELINE = "Use write only for new files or complete rewrites."
_GENERIC_WRITE_ERROR = "failed to write file"


class _ToolArgs(BaseModel):
    """Model-visible arguments for the write tool."""

    path: str = Field(description="File path to create or overwrite.")
    content: str = Field(description="Complete UTF-8 text content to write.")


def write_tool(
    *,
    cwd: str | Path | None = None,
    writer: WritableFileSystem | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Return the first-party harness write tool definition.

    Args:
        cwd: Optional working directory for resolving relative tool paths.
            Injected writers use their storage namespace; mounted writers default
            to the virtual root. Other injected writers default to `.`.
        writer: Optional file writer. Defaults to the local filesystem.
        permissions: Optional policy for create/overwrite permission decisions.
        approval_callback: Optional callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable tool plus prompt metadata.
    """
    resolver = (
        ToolPathResolver.for_optional(cwd)
        if writer is None
        else StorageToolPathResolver.for_optional(cwd, filesystem=writer)
    )
    file_writer = writer if writer is not None else LocalFileSystem()

    async def run_write(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        try:
            return await _write_file(
                args,
                resolver=resolver,
                writer=file_writer,
                cancellation_token=cancellation_token,
                permissions=permissions,
                approval_callback=approval_callback,
                context=context,
            )
        except Exception:
            return _GENERIC_WRITE_ERROR

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str](
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_write,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE, tool_path_guideline(resolver)),
    )


async def _write_file(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    writer: WritableFileSystem,
    cancellation_token: CancellationToken,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
) -> str:
    """Write text content and return a model-facing status message."""
    if args.path.strip() == "":
        return "path must not be empty"
    if "\x00" in args.path:
        return "path contains a null byte"

    try:
        encoded_content = args.content.encode("utf-8")
    except UnicodeEncodeError:
        return "content is not valid UTF-8"

    resolved_path = await asyncio.to_thread(resolver.resolve, args.path)

    # Metadata determines create versus overwrite permissions. Do not open a
    # writer yet: entering its context may create directories or stage a file.
    parent_info: FileInfo | None = None
    target_info: FileInfo | None = None
    invalid_parent = False
    try:
        parent_info = await asyncio.to_thread(writer.stat, str(resolved_path.parent))
        if parent_info is None or parent_info.is_directory:
            target_info = await asyncio.to_thread(writer.stat, str(resolved_path))
    except NotADirectoryError:
        # Match missing-path classification until permissions allow a diagnostic.
        invalid_parent = True
    except OSError:
        return _GENERIC_WRITE_ERROR

    permission_error = await _check_write_permissions(
        resolved_path,
        resolver=resolver,
        parent_info=parent_info,
        target_info=target_info,
        permissions=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    # Report path details only after the corresponding permission checks pass.
    if target_info is not None and target_info.is_directory:
        return f"path is a directory: `{resolved_path}`"
    if invalid_parent or (parent_info is not None and not parent_info.is_directory):
        return f"parent path is not a directory: `{resolved_path.parent}`"
    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    try:
        await write_content(writer, str(resolved_path), encoded_content)
    except (FileExistsError, NotADirectoryError):
        return f"parent path is not a directory: `{resolved_path.parent}`"
    except IsADirectoryError:
        return f"path is a directory: `{resolved_path}`"
    except PermissionError:
        return f"permission denied: `{resolved_path}`"
    except OSError:
        return f"failed to write file: `{resolved_path}`"

    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    return f"Wrote {len(encoded_content)} bytes to {resolved_path}."


async def _check_write_permissions(
    path: PurePath,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    parent_info: FileInfo | None,
    target_info: FileInfo | None,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
) -> str | None:
    """Return a model-facing permission result before any write side effect."""
    requests: list[ToolPermissionRequest] = []
    if parent_info is None:
        requests.append(
            ToolPermissionRequest(
                tool_name=_TOOL_NAME,
                operation=ToolOperation.CREATE_DIRECTORY,
                cwd=resolver.cwd,
                path=path.parent,
            )
        )

    operation = (
        ToolOperation.OVERWRITE_FILE
        if target_info is not None
        else ToolOperation.CREATE_FILE
    )
    requests.append(
        ToolPermissionRequest(
            tool_name=_TOOL_NAME,
            operation=operation,
            cwd=resolver.cwd,
            path=path,
        )
    )

    # Obtain every required grant before the backend creates parents or writes.
    for request in requests:
        permission_error = await evaluate_tool_permission(
            request,
            policy=permissions,
            approval_callback=approval_callback,
            context=context,
        )
        if permission_error is not None:
            return permission_error

    return None
