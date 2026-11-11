"""Patch tool implementation for first-party harness base tools."""

import asyncio
from pathlib import Path, PurePath

import patch_tool as patch_engine
from pydantic import BaseModel, Field

from agentlane.harness.filesystem import (
    FileReader,
    FileRevisionConflictError,
    FileRevisions,
    FileWriter,
)
from agentlane.io import read_all
from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from .._constants import EMPTY_PATH_ERROR
from .._storage_write import write_content
from ._constants import NULL_BYTE_PATH_ERROR, PATCH_TOOL_NAME
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
    "Apply precise SEARCH/REPLACE edits to an existing text file. "
    "The path is provided as a structured argument; edits must contain one or "
    "more bare SEARCH/REPLACE blocks. Edits are all-or-nothing."
)
_TOOL_PROMPT_SNIPPET = "Apply search/replace edits to existing files"
_TOOL_PROMPT_GUIDELINES = (
    "Use patch for precise edits to existing files after reading them; use write for new files or complete rewrites.",
    "Each patch SEARCH block must match exactly one location; include enough surrounding lines to make it unique.",
)
_GENERIC_PATCH_ERROR = "failed to patch file"

# Another writer can change a revision-checked file between this tool's read and
# write; each retry reads the new content and applies the same edits again.
_DEFAULT_REVISION_ATTEMPTS = 3


class _ToolArgs(BaseModel):
    """Model-visible arguments for the patch tool."""

    path: str = Field(description="File path to edit.")
    edits: str = Field(
        description=(
            "One or more bare SEARCH/REPLACE blocks. Use exactly: "
            "<<<<<<< SEARCH, then old text, then =======, then new text, "
            "then >>>>>>> REPLACE. Do not include file path lines in this value."
        )
    )


def patch_tool(
    *,
    cwd: str | Path | None = None,
    reader: FileReader | None = None,
    writer: FileWriter | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
    max_attempts: int = _DEFAULT_REVISION_ATTEMPTS,
) -> HarnessToolDefinition:
    """Build the first-party patch harness tool.

    Args:
        cwd: Optional working directory used to resolve relative paths. When
            omitted, the current working directory is captured at construction
            time. Mounted backends default to the virtual root; other injected
            backends default to `.`.
        reader: Optional storage reader. Supply both reader and writer.
        writer: Writer for the same namespace as reader. Mounted paths are rooted
            virtual paths; other injected paths remain relative POSIX paths.
            Separate reader and writer objects must use the same namespace and
            the same revision contract. Calls through this tool instance
            serialize read-edit-write. When the backend implements
            `FileRevisions`, each write is also checked against the revision
            that was read, which protects against separate tool instances and
            external writers that commit in between. Without `FileRevisions`,
            callers must coordinate separate tool instances and external
            writers. Whole-file writes are never checked, so a later whole-file
            write can still replace a patch result.
        permissions: Optional policy for modify permission decisions.
        approval_callback: Optional callback for approval-required decisions.
        max_attempts: Attempts per patch on a revision conflict. Each attempt
            reads the file again and applies the same edits. `1` makes one
            checked write and returns the conflict without a replay.

    Returns:
        HarnessToolDefinition: Executable patch tool with prompt metadata.

    Raises:
        ValueError: If only one of reader and writer is supplied, if the reader
            reports revisions but the writer cannot check them, or if
            `max_attempts` is less than 1.
    """
    # A partial override could read one namespace and replace a different file.
    if (reader is None) != (writer is None):
        raise ValueError("patch requires both reader and writer for the same namespace")

    # A revision from the reader is useless to a writer that cannot check it.
    if isinstance(reader, FileRevisions) and not isinstance(writer, FileRevisions):
        raise ValueError(
            "patch reader reports revisions, so its writer must implement "
            "FileRevisions for the same storage"
        )

    if max_attempts < 1:
        raise ValueError("patch max_attempts must be at least 1")

    resolver = (
        ToolPathResolver.for_optional(cwd)
        if reader is None
        else StorageToolPathResolver.for_optional(cwd, filesystem=reader)
    )

    # One lock keeps tool-local edits ordered without a shared backend registry.
    storage_lock = asyncio.Lock()

    async def run_patch(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        try:
            return await _patch_file(
                args,
                resolver=resolver,
                reader=reader,
                writer=writer,
                storage_lock=storage_lock,
                cancellation_token=cancellation_token,
                permissions=permissions,
                approval_callback=approval_callback,
                context=context,
                max_attempts=max_attempts,
            )
        except Exception:
            return _GENERIC_PATCH_ERROR

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str](
            name=PATCH_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_patch,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(*_TOOL_PROMPT_GUIDELINES, tool_path_guideline(resolver)),
    )


async def _patch_file(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    reader: FileReader | None,
    writer: FileWriter | None,
    storage_lock: asyncio.Lock,
    cancellation_token: CancellationToken,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
    max_attempts: int,
) -> str:
    """Parse, apply, and render one model-facing patch result."""
    validation_error = _validate_args(args)
    if validation_error is not None:
        return validation_error

    resolved_path = resolver.resolve(args.path)
    permission_error = await evaluate_tool_permission(
        ToolPermissionRequest(
            tool_name=PATCH_TOOL_NAME,
            operation=ToolOperation.MODIFY_FILE,
            cwd=resolver.cwd,
            path=resolved_path,
        ),
        policy=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    # Logical paths must be inspected by their provider, never by host stat.
    if isinstance(resolved_path, Path):
        if resolved_path.is_dir():
            return f"Path is a directory: {resolved_path}"
        if not resolved_path.exists():
            return f"File not found: {resolved_path}"

    try:
        edits = patch_engine.parse_blocks(args.edits)
    except patch_engine.ParseError as exc:
        return _format_parse_error(exc)

    if not edits:
        return (
            "Patch tool input is invalid. edits must contain at least one replacement."
        )

    try:
        if reader is None:
            # The local engine retains its own locking and atomic replacement.
            assert isinstance(resolved_path, Path)
            result = patch_engine.apply_edits(resolved_path, edits)
            applied = result.edits_applied
        else:
            assert writer is not None

            # Keep the snapshot and replacement in one per-instance operation.
            # A cancelled commit settles before the next patch can read.
            async with storage_lock:
                applied = await _patch_storage_file(
                    reader,
                    writer,
                    resolved_path,
                    edits,
                    cancellation_token,
                    max_attempts,
                )
    except patch_engine.EmptyOldTextError as exc:
        return _format_empty_search_error(resolved_path, exc.edit_index, len(edits))
    except patch_engine.TextNotFoundError as exc:
        return _format_not_found_error(resolved_path, exc.edit_index, len(edits))
    except patch_engine.AmbiguousMatchError as exc:
        return _format_ambiguous_match_error(
            resolved_path,
            exc.edit_index,
            len(edits),
            exc.occurrences,
        )
    except patch_engine.OverlappingEditsError as exc:
        return _format_overlap_error(
            resolved_path, exc.edit_index, exc.other_edit_index
        )
    except patch_engine.NoChangesError:
        return _format_no_changes_error(resolved_path, len(edits))
    except FileRevisionConflictError:
        return (
            f"File changed while it was patched: {resolved_path}. "
            "Read it again and repeat the patch."
        )
    except IsADirectoryError:
        return f"Path is a directory: {resolved_path}"
    except FileNotFoundError:
        return f"File not found: {resolved_path}"
    except PermissionError:
        return f"Permission denied: {resolved_path}"
    except UnicodeDecodeError:
        return f"File is not valid UTF-8: {resolved_path}"
    except OSError:
        return f"Failed to patch file: {resolved_path}"

    return _format_success(resolved_path, applied)


async def _patch_storage_file(
    reader: FileReader,
    writer: FileWriter,
    path: PurePath,
    edits: list[patch_engine.Edit],
    token: CancellationToken,
    max_attempts: int,
) -> int:
    """Apply edits to injected storage and return how many were applied.

    Raises:
        FileRevisionConflictError: If the file changes between the read and the
            write on every one of `max_attempts` attempts.
    """
    conflicts = 0
    while True:
        if token.is_cancelled:
            raise asyncio.CancelledError

        text_result, revision = await asyncio.to_thread(
            _prepare_storage_patch, reader, path, edits, token
        )

        # A worker read can finish after cancellation; do not start a
        # commit merely because it returned a valid edit result.
        if token.is_cancelled:
            raise asyncio.CancelledError

        # A conflict means another writer committed after this read, so read
        # its content and apply the same edits again until attempts run out.
        try:
            await write_content(
                writer,
                str(path),
                text_result.content.encode("utf-8"),
                revision=revision,
            )
        except FileRevisionConflictError:
            conflicts += 1
            if conflicts == max_attempts:
                raise

            continue

        return text_result.edits_applied


def _prepare_storage_patch(
    reader: FileReader,
    path: PurePath,
    edits: list[patch_engine.Edit],
    token: CancellationToken,
) -> tuple[patch_engine.TextEditResult, str | None]:
    """Apply all edits privately before any write to the injected backend.

    Returns the edited text and the revision of the content that was read, or
    None when the backend does not report revisions.
    """
    if isinstance(reader, FileRevisions):
        with reader.open_read_revision(str(path)) as (source, revision):
            content = read_all(source).decode("utf-8")
    else:
        revision = None
        with reader.open_read(str(path)) as source:
            content = read_all(source).decode("utf-8")

    if token.is_cancelled:
        raise asyncio.CancelledError

    # Finish all matching in memory before opening a writer. A failed edit must
    # leave the original content intact, including its BOM and line endings.
    result = patch_engine.apply_edits_to_text(content, edits, path_hint=str(path))

    return result, revision


def _validate_args(args: _ToolArgs) -> str | None:
    """Return a model-facing validation error when arguments are invalid."""
    if args.path.strip() == "":
        return EMPTY_PATH_ERROR
    if "\x00" in args.path:
        return NULL_BYTE_PATH_ERROR
    try:
        args.edits.encode("utf-8")
    except UnicodeEncodeError:
        return "patch is not valid UTF-8"
    return None


def _format_success(path: PurePath, edits_applied: int) -> str:
    """Render a successful patch result as deterministic text."""
    edit_noun = "edit" if edits_applied == 1 else "edits"
    return f"Applied {edits_applied} {edit_noun} to {path}."


def _format_parse_error(exc: patch_engine.ParseError) -> str:
    """Render a parser error without exposing stack traces."""
    return f"Invalid patch format: {exc}"


def _edit_label(index: int | None) -> str:
    """Return a 1-indexed edit label for model-facing errors."""
    if index is None:
        return "edit"
    return f"edit {index + 1}"


def _format_not_found_error(
    path: PurePath,
    edit_index: int | None,
    total_edits: int,
) -> str:
    """Render a recoverable missing SEARCH text error."""
    if total_edits == 1:
        return (
            f"Could not find the SEARCH text in {path}. The SEARCH text must "
            "match exactly including all whitespace and newlines."
        )
    return (
        f"Could not find {_edit_label(edit_index)} in {path}. The SEARCH text "
        "must match exactly including all whitespace and newlines."
    )


def _format_ambiguous_match_error(
    path: PurePath,
    edit_index: int | None,
    total_edits: int,
    occurrences: int,
) -> str:
    """Render a recoverable duplicate SEARCH text error."""
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the SEARCH text in {path}. "
            "The SEARCH text must be unique. Please provide more context to "
            "make it unique."
        )
    return (
        f"Found {occurrences} occurrences of {_edit_label(edit_index)} in {path}. "
        "Each SEARCH text must be unique. Please provide more context to make "
        "it unique."
    )


def _format_empty_search_error(
    path: PurePath,
    edit_index: int | None,
    total_edits: int,
) -> str:
    """Render a recoverable empty SEARCH text error."""
    if total_edits == 1:
        return f"SEARCH text must not be empty in {path}."
    return f"{_edit_label(edit_index)} SEARCH text must not be empty in {path}."


def _format_overlap_error(
    path: PurePath,
    edit_index: int | None,
    other_edit_index: int | None,
) -> str:
    """Render a recoverable overlapping edits error."""
    return (
        f"{_edit_label(edit_index)} and {_edit_label(other_edit_index)} overlap "
        f"in {path}. Merge them into one edit or target disjoint regions."
    )


def _format_no_changes_error(path: PurePath, total_edits: int) -> str:
    """Render a recoverable no-op replacement error."""
    if total_edits == 1:
        return f"No changes made to {path}. The replacement produced identical content."
    return f"No changes made to {path}. The replacements produced identical content."
