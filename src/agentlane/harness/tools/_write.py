"""Write tool implementation for first-party harness base tools."""

import contextlib
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ._paths import ToolPathResolver
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
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
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Return the first-party harness write tool definition.

    Args:
        cwd: Optional working directory for resolving relative tool paths.
        permissions: Optional policy for create/overwrite permission decisions.
        approval_callback: Optional callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable tool plus prompt metadata.
    """
    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))

    async def run_write(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del cancellation_token
        try:
            return await _write_file(
                args,
                resolver=resolver,
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
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


async def _write_file(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver,
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

    resolved_path = resolver.resolve(args.path)
    permission_error = await _check_write_permissions(
        resolved_path,
        resolver=resolver,
        permissions=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    if resolved_path.is_dir():
        return f"path is a directory: `{resolved_path}`"

    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        return f"parent path is not a directory: `{resolved_path.parent}`"
    except PermissionError:
        return f"permission denied: `{resolved_path}`"
    except OSError:
        return f"failed to write file: `{resolved_path}`"

    if resolved_path.exists():
        write_result = _replace_text_atomically(
            target=resolved_path,
            content=args.content,
        )
    else:
        write_result = _write_new_file(target=resolved_path, content=args.content)

    if write_result is not None:
        return write_result

    return f"Wrote {len(encoded_content)} bytes to {resolved_path}."


async def _check_write_permissions(
    path: Path,
    *,
    resolver: ToolPathResolver,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
) -> str | None:
    """Return a model-facing permission result before any write side effect."""
    requests: list[ToolPermissionRequest] = []
    if not path.parent.exists():
        requests.append(
            ToolPermissionRequest(
                tool_name=_TOOL_NAME,
                operation=ToolOperation.CREATE_DIRECTORY,
                cwd=resolver.cwd,
                path=path.parent,
            )
        )

    operation = (
        ToolOperation.OVERWRITE_FILE if path.exists() else ToolOperation.CREATE_FILE
    )
    requests.append(
        ToolPermissionRequest(
            tool_name=_TOOL_NAME,
            operation=operation,
            cwd=resolver.cwd,
            path=path,
        )
    )

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


def _write_new_file(*, target: Path, content: str) -> str | None:
    """Write a new text file and return a model-facing error if it fails."""
    try:
        target.write_text(content, encoding="utf-8", newline="")
    except PermissionError:
        return f"permission denied: `{target}`"
    except OSError:
        return f"failed to write file: `{target}`"
    return None


def _replace_text_atomically(*, target: Path, content: str) -> str | None:
    """Replace an existing file and return a model-facing error if it fails."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=target.parent,
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)

        temp_path.replace(target)
    except PermissionError:
        if temp_path is not None:
            _unlink_temp_file(temp_path)
        return f"permission denied: `{target}`"
    except OSError:
        if temp_path is not None:
            _unlink_temp_file(temp_path)
        return f"failed to write file: `{target}`"
    return None


def _unlink_temp_file(path: Path) -> None:
    """Best-effort cleanup for failed atomic replacements."""
    with contextlib.suppress(OSError):
        path.unlink()
