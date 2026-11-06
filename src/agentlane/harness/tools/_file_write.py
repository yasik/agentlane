"""Authorize file replacement before a tool opens its storage writer."""

import asyncio
from pathlib import PurePath

from agentlane.harness.filesystem import FileInfo, WritableFileSystem
from agentlane.models import ToolExecutionContext

from ._paths import StorageToolPathResolver, ToolPathResolver
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)

GENERIC_WRITE_ERROR = "failed to write file"
"""Hide provider details when write preparation or execution fails."""


async def prepare_file_write(
    path: PurePath,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    writer: WritableFileSystem,
    tool_name: str,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
) -> str | None:
    """Inspect and authorize a write target without creating directories or files."""
    # Metadata determines create versus overwrite permissions. Do not open a
    # writer yet: entering its context may create directories or stage a file.
    parent_info: FileInfo | None = None
    target_info: FileInfo | None = None
    invalid_parent = False
    try:
        parent_info = await asyncio.to_thread(writer.stat, str(path.parent))
        if parent_info is None or parent_info.is_directory:
            target_info = await asyncio.to_thread(writer.stat, str(path))
    except NotADirectoryError:
        # Match missing-path classification until permissions allow a diagnostic.
        invalid_parent = True
    except OSError:
        return GENERIC_WRITE_ERROR

    permission_error = await _check_write_permissions(
        path,
        resolver=resolver,
        tool_name=tool_name,
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
        return f"path is a directory: `{path}`"

    if invalid_parent or (parent_info is not None and not parent_info.is_directory):
        return f"parent path is not a directory: `{path.parent}`"

    return None


async def _check_write_permissions(
    path: PurePath,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    tool_name: str,
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
                tool_name=tool_name,
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
            tool_name=tool_name,
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
