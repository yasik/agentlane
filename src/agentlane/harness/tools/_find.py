"""Find tool implementation for first-party harness base tools."""

import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePath

from pydantic import BaseModel, Field
from wcmatch import glob as wcmatch_glob
from wcmatch.glob import WcMatcher

from agentlane.harness.filesystem import (
    LocalFileSystem,
    ReadableFileSystem,
    normalize_relative_path,
)
from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ._gitignore import GitignoreMatcher
from ._output import FIND_DEFAULT_LIMIT, TEXT_MAX_BYTES
from ._paths import RelativeToolPathResolver, ToolPathResolver
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
from ._types import HarnessToolDefinition

_TOOL_NAME = "find"
_TOOL_DESCRIPTION = (
    "Searches files by glob pattern. Returns paths relative to the "
    "search directory, sorted by modification time (newest first), with ties "
    "broken alphabetically. Use `**/` for recursive matches and `{a,b}` for "
    "brace expansion. Matching is case-insensitive. Symlinked directories are "
    "not followed. Respects .gitignore. Output is capped at 1000 results or "
    "51200 bytes."
)
_TOOL_PROMPT_SNIPPET = "Find files by glob pattern (use `**/` for recursion)"
_TOOL_PROMPT_GUIDELINE = "Use find to locate files in its storage namespace. Use shell commands only for process files outside that namespace."
_GENERIC_FIND_ERROR = "failed to find files"

# IGNORECASE makes matching consistent across case-sensitive (Linux) and
# case-insensitive (macOS APFS, Windows NTFS) filesystems. BRACE supports the
# common `*.{ext1,ext2}` idiom. FORCEUNIX keeps path matching POSIX-style on
# Windows.
_GLOB_FLAGS = (
    wcmatch_glob.GLOBSTAR
    | wcmatch_glob.DOTMATCH
    | wcmatch_glob.BRACE
    | wcmatch_glob.IGNORECASE
    | wcmatch_glob.FORCEUNIX
)


class _ToolArgs(BaseModel):
    """Model-visible arguments for the find tool."""

    pattern: str = Field(
        description=(
            "Glob pattern to match against file paths. Use `**/` for "
            "recursive matches (e.g. `**/*.py`) and `{a,b}` for brace "
            "expansion (e.g. `**/*.{ts,tsx}`). Matching is case-insensitive."
        )
    )
    path: str | None = Field(
        default=None,
        description=(
            "Directory path to search. Defaults to the configured working directory."
        ),
    )
    limit: int = Field(
        default=FIND_DEFAULT_LIMIT,
        description=(
            "Maximum number of matching file paths to return. Hard maximum "
            f"is {FIND_DEFAULT_LIMIT}; larger values are capped."
        ),
    )


@dataclass(frozen=True, slots=True)
class _FindContent:
    """Formatted relative paths plus optional continuation state."""

    paths: tuple[str, ...]
    continuation_message: str | None = None


def find_tool(
    *,
    cwd: str | Path | None = None,
    reader: ReadableFileSystem | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Build the first-party file find harness tool.

    Args:
        cwd: Optional working directory used to resolve relative search paths.
            When omitted, the current working directory is captured at
            construction time.
        reader: Optional filesystem with read, list, and metadata capabilities.
            Injected paths are relative POSIX paths. Missing modification times
            sort as zero, with the same alphabetical tie-break as local files.
        permissions: Optional policy for search permission decisions.
        approval_callback: Optional callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable find tool with prompt metadata.
    """
    resolver = (
        ToolPathResolver.for_optional(cwd)
        if reader is None
        else RelativeToolPathResolver.for_optional(cwd)
    )
    filesystem = reader if reader is not None else LocalFileSystem()

    async def run_find(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        try:
            return await _find_files(
                args,
                resolver=resolver,
                filesystem=filesystem,
                permissions=permissions,
                approval_callback=approval_callback,
                cancellation_token=cancellation_token,
                context=context,
            )
        except Exception:
            return _GENERIC_FIND_ERROR

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str](
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_find,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


async def _find_files(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver | RelativeToolPathResolver,
    filesystem: ReadableFileSystem,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    cancellation_token: CancellationToken,
    context: ToolExecutionContext,
) -> str:
    """Find files and render a plain-text tool result."""
    pattern = args.pattern.strip()
    if pattern == "":
        return "pattern must not be empty"
    if args.path is not None and args.path.strip() == "":
        return "path must not be empty"
    if args.limit < 1:
        return "limit must be greater than zero"

    search_dir = resolver.cwd if args.path is None else resolver.resolve(args.path)
    permission_error = await evaluate_tool_permission(
        ToolPermissionRequest(
            tool_name=_TOOL_NAME,
            operation=ToolOperation.SEARCH_FILES,
            cwd=resolver.cwd,
            path=search_dir,
        ),
        policy=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    info = await asyncio.to_thread(filesystem.stat, str(search_dir))
    if info is None or not info.is_directory:
        return f"path is not a directory: `{search_dir}`"

    content = await asyncio.to_thread(
        _collect_find_content,
        filesystem=filesystem,
        search_dir=search_dir,
        pattern=_normalize_pattern(pattern),
        requested_limit=args.limit,
        cancellation_token=cancellation_token,
    )

    return _format_find_output(search_dir, content)


def _collect_find_content(
    *,
    search_dir: PurePath,
    filesystem: ReadableFileSystem,
    pattern: str,
    requested_limit: int,
    cancellation_token: CancellationToken,
) -> _FindContent:
    """Collect matching relative paths sorted newest-first by mtime."""
    glob_matcher = wcmatch_glob.compile(pattern, flags=_GLOB_FLAGS)
    matcher = GitignoreMatcher.from_path(search_dir, filesystem=filesystem)
    raw_matches = _matching_paths(
        search_dir=search_dir,
        filesystem=filesystem,
        glob_matcher=glob_matcher,
        matcher=matcher,
        cancellation_token=cancellation_token,
    )

    # Apply limits after sorting: a later directory may contain the newest match.
    sorted_paths = _sort_by_mtime_desc(raw_matches)

    effective_limit = min(requested_limit, FIND_DEFAULT_LIMIT)
    limited_paths = sorted_paths[:effective_limit]
    paths, byte_truncated = _limit_paths_by_bytes(
        limited_paths,
        max_bytes=TEXT_MAX_BYTES,
    )

    continuation_message = _build_continuation_message(
        requested_limit=requested_limit,
        effective_limit=effective_limit,
        total_matches=len(sorted_paths),
        byte_truncated=byte_truncated,
    )

    return _FindContent(
        paths=tuple(paths),
        continuation_message=continuation_message,
    )


def _matching_paths(
    *,
    search_dir: PurePath,
    filesystem: ReadableFileSystem,
    glob_matcher: WcMatcher[str],
    matcher: GitignoreMatcher,
    cancellation_token: CancellationToken,
) -> list[tuple[str, float]]:
    """Walk the filesystem interface and collect matching path metadata."""
    matches: list[tuple[str, float]] = []
    pending = [search_dir]

    while pending:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        root_path = pending.pop()
        try:
            entries = filesystem.list_directory(str(root_path))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            # Removed or inaccessible directories are skippable. Other failures
            # must reach the tool boundary instead of looking like empty results.
            continue

        directories: list[PurePath] = []
        for entry in sorted(entries, key=lambda entry: entry.name):
            if not isinstance(root_path, Path):
                # A provider entry must not redirect traversal to another path.
                name = normalize_relative_path(entry.name)
                if len(name.parts) != 1 or name.as_posix() != entry.name:
                    raise ValueError("Directory entries must contain one child name")

            path = root_path / entry.name
            if matcher.is_ignored(path, is_dir=entry.is_directory):
                continue

            if entry.is_directory:
                # Directory links can lead back to an ancestor and create a cycle.
                if not entry.is_symlink:
                    directories.append(path)
                continue

            relative = path.relative_to(search_dir).as_posix()
            if glob_matcher.match(relative):
                # Listing metadata avoids a backend request for every match.
                matches.append((relative, entry.modified_time or 0.0))

        # The stack visits the first sorted directory next.
        pending.extend(reversed(directories))

    return matches


def _sort_by_mtime_desc(matches: list[tuple[str, float]]) -> list[str]:
    """Sort paths newest-first by mtime, breaking ties alphabetically."""
    return [path for path, _ in sorted(matches, key=lambda item: (-item[1], item[0]))]


def _normalize_pattern(pattern: str) -> str:
    """Normalize user-provided glob syntax for stable matching.

    Strips leading `./` and `/` since match candidates are always relative
    POSIX paths (never absolute, never `./`-prefixed). Without this, an
    absolute-style pattern like `/src/*.py` would silently match nothing.
    """
    normalized = pattern.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _limit_paths_by_bytes(
    paths: list[str],
    *,
    max_bytes: int,
) -> tuple[list[str], bool]:
    """Return whole path lines that fit within one UTF-8 byte budget."""
    selected: list[str] = []
    used_bytes = 0
    for path in paths:
        separator_bytes = 1 if selected else 0
        path_bytes = len(path.encode("utf-8"))
        candidate_bytes = used_bytes + separator_bytes + path_bytes
        if candidate_bytes > max_bytes:
            return selected, True

        selected.append(path)
        used_bytes = candidate_bytes

    return selected, False


def _build_continuation_message(
    *,
    requested_limit: int,
    effective_limit: int,
    total_matches: int,
    byte_truncated: bool,
) -> str | None:
    """Build an actionable continuation message, or None when nothing to say."""
    if byte_truncated:
        return (
            f"Output truncated at {TEXT_MAX_BYTES} bytes; "
            "refine the pattern or narrow `path`."
        )
    if total_matches <= effective_limit:
        return None
    if requested_limit < FIND_DEFAULT_LIMIT:
        return (
            f"{total_matches} files matched; returned first {effective_limit}. "
            f"Refine the pattern or raise `limit` (max {FIND_DEFAULT_LIMIT})."
        )
    return (
        f"{total_matches} files matched; returned first {effective_limit} "
        "(maximum). Refine the pattern or narrow `path`."
    )


def _format_find_output(search_dir: PurePath, content: _FindContent) -> str:
    """Render the final model-facing tool result."""
    output = [f"Search directory: {search_dir}"]
    if content.paths:
        output.extend(content.paths)
    else:
        output.append("No files matched.")

    if content.continuation_message is not None:
        output.append(content.continuation_message)

    return "\n".join(output)
