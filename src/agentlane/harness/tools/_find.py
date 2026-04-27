"""Find tool implementation for first-party harness base tools."""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from wcmatch import glob as wcmatch_glob
from wcmatch.glob import WcMatcher

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._gitignore import GitignoreMatcher
from ._output import FIND_DEFAULT_LIMIT, TEXT_MAX_BYTES
from ._paths import ToolPathResolver
from ._types import HarnessToolDefinition

_TOOL_NAME = "find"
_TOOL_DESCRIPTION = (
    "Searches local files by glob pattern. Returns paths relative to the "
    "search directory, sorted by modification time (newest first), with ties "
    "broken alphabetically. Use `**/` for recursive matches and `{a,b}` for "
    "brace expansion. Matching is case-insensitive. Symlinked directories are "
    "not followed. Respects .gitignore. Output is capped at 1000 results or "
    "51200 bytes."
)
_TOOL_PROMPT_SNIPPET = "Find files by glob pattern (use `**/` for recursion)"
_TOOL_PROMPT_GUIDELINE = (
    "Use find to locate files instead of shelling out to find or ls."
)
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


def find_tool(*, cwd: str | Path | None = None) -> HarnessToolDefinition:
    """Build the first-party file find harness tool.

    Args:
        cwd: Optional working directory used to resolve relative search paths.
            When omitted, the current working directory is captured at
            construction time.

    Returns:
        HarnessToolDefinition: Executable find tool with prompt metadata.
    """
    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))

    async def run_find(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        try:
            return _find_files(
                args,
                resolver=resolver,
                cancellation_token=cancellation_token,
            )
        except Exception:
            return _GENERIC_FIND_ERROR

    return HarnessToolDefinition(
        tool=Tool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_find,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


def _find_files(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver,
    cancellation_token: CancellationToken,
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
    if not search_dir.is_dir():
        return f"path is not a directory: `{search_dir}`"

    content = _collect_find_content(
        search_dir=search_dir,
        pattern=_normalize_pattern(pattern),
        requested_limit=args.limit,
        cancellation_token=cancellation_token,
    )

    return _format_find_output(search_dir, content)


def _collect_find_content(
    *,
    search_dir: Path,
    pattern: str,
    requested_limit: int,
    cancellation_token: CancellationToken,
) -> _FindContent:
    """Collect matching relative paths sorted newest-first by mtime."""
    glob_matcher = wcmatch_glob.compile(pattern, flags=_GLOB_FLAGS)
    matcher = GitignoreMatcher.from_path(search_dir)
    raw_matches = _matching_paths(
        search_dir=search_dir,
        glob_matcher=glob_matcher,
        matcher=matcher,
        cancellation_token=cancellation_token,
    )
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
    search_dir: Path,
    glob_matcher: WcMatcher[str],
    matcher: GitignoreMatcher,
    cancellation_token: CancellationToken,
) -> list[tuple[str, float]]:
    """Return (relative_path, mtime) pairs for files matching the glob.

    `os.walk` is used with the default `followlinks=False` so symlinked
    directories are not traversed. This avoids cycles and prevents pattern
    matching from escaping the search directory through symlinks.
    """
    matches: list[tuple[str, float]] = []
    for current_root, directory_names, file_names in os.walk(search_dir, topdown=True):
        if cancellation_token.is_cancelled:
            break

        root_path = Path(current_root)
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if not matcher.is_ignored(root_path / directory_name, is_dir=True)
        )

        for file_name in file_names:
            file_path = root_path / file_name
            if matcher.is_ignored(file_path, is_dir=False):
                continue

            relative_path = file_path.relative_to(search_dir).as_posix()
            if not glob_matcher.match(relative_path):
                continue

            matches.append((relative_path, _safe_mtime(file_path)))

    return matches


def _safe_mtime(path: Path) -> float:
    """Return mtime, or 0.0 when the file disappeared mid-walk."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


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


def _format_find_output(search_dir: Path, content: _FindContent) -> str:
    """Render the final model-facing tool result."""
    output = [f"Search directory: {search_dir}"]
    if content.paths:
        output.extend(content.paths)
    else:
        output.append("No files matched.")

    if content.continuation_message is not None:
        output.append(content.continuation_message)

    return "\n".join(output)
