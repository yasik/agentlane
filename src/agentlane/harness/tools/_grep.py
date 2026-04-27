"""Grep tool implementation for first-party harness base tools."""

import base64
import binascii
import fnmatch
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from ripgrepy import RipGrepNotFound, Ripgrepy

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._gitignore import GitignoreMatcher
from ._output import (
    GREP_DEFAULT_LIMIT,
    GREP_MAX_LINE_LENGTH,
    TEXT_MAX_BYTES,
    is_likely_binary_file,
)
from ._paths import ToolPathResolver
from ._types import HarnessToolDefinition

_TOOL_NAME = "grep"
_TOOL_DESCRIPTION = (
    "Searches local text files for a pattern using ripgrep. Supports regex or "
    "literal matching, multiline patterns, file-type and glob filtering, and "
    "three output modes (content, files_with_matches, count). Respects "
    f".gitignore. Output is truncated to {GREP_DEFAULT_LIMIT} entries or "
    f"{TEXT_MAX_BYTES} bytes, whichever is hit first."
)
_TOOL_PROMPT_SNIPPET = "Search file contents for patterns"
_TOOL_PROMPT_GUIDELINE = (
    'Use grep with output_mode="files_with_matches" or "count" to discover '
    "where a symbol appears; switch to content mode when you need surrounding "
    "lines."
)
_GENERIC_GREP_ERROR = "failed to search files"
_RIPGREP_NOT_FOUND_ERROR = "ripgrep executable not found"
_INVALID_GLOB_ERROR = "invalid glob pattern"
_INVALID_REGEX_ERROR = "invalid regex pattern"
_NO_MATCHES = "No matches."

JsonObject = dict[str, object]
_OutputMode = Literal["content", "files_with_matches", "count"]


class _ToolArgs(BaseModel):
    """Model-visible arguments for the grep tool."""

    model_config = ConfigDict(populate_by_name=True)

    pattern: str = Field(
        description="Ripgrep regex pattern, or literal text when literal is true."
    )
    path: str | None = Field(
        default=None,
        description="File or directory to search. Defaults to the tool working directory.",
    )
    output_mode: _OutputMode = Field(
        default="content",
        alias="outputMode",
        description=(
            "Result shape. 'content' returns matching lines (default). "
            "'files_with_matches' returns one path per matching file. "
            "'count' returns 'path:N' rows. Prefer the latter two for cheap "
            "discovery."
        ),
    )
    glob: str | None = Field(
        default=None,
        description=(
            "Optional glob filter. Directory searches use ripgrep glob syntax "
            "(supports ** and !-negation); explicit file paths fall back to "
            "fnmatch."
        ),
    )
    file_type: str | None = Field(
        default=None,
        alias="type",
        description=(
            "Optional ripgrep file-type name (e.g. 'py', 'js'). More compact "
            "than glob when one of ripgrep's predefined types fits."
        ),
    )
    ignore_case: bool = Field(
        default=False,
        alias="ignoreCase",
        description="Whether matching should ignore letter case.",
    )
    literal: bool = Field(
        default=False,
        description="Treat pattern as literal text instead of a regular expression.",
    )
    multiline: bool = Field(
        default=False,
        description=(
            "Allow patterns to match across newlines. '.' also matches " "newlines."
        ),
    )
    context: int = Field(
        default=0,
        ge=0,
        description=(
            "Lines of context before and after each match. Only applies in "
            "content mode."
        ),
    )
    limit: int = Field(
        default=GREP_DEFAULT_LIMIT,
        ge=1,
        description=(
            "Maximum entries to return. In content mode this caps matching "
            "lines; in files_with_matches and count modes it caps rendered "
            "rows."
        ),
    )


@dataclass(frozen=True, slots=True)
class _GrepContent:
    """Formatted search results plus continuation or error state."""

    lines: tuple[str, ...]
    continuation_messages: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _RipgrepRow:
    """One ripgrep JSON line relevant to final rendering."""

    path: Path
    line_number: int
    line: str
    is_match: bool


@dataclass(frozen=True, slots=True)
class _ParsedRipgrepOutput:
    """Parsed ripgrep rows plus match-count metadata."""

    rows: tuple[_RipgrepRow, ...]
    match_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _RenderedMatches:
    """Rendered grep rows plus output truncation metadata."""

    lines: tuple[str, ...]
    byte_truncated: bool = False
    line_truncated: bool = False


def grep_tool(*, cwd: str | Path | None = None) -> HarnessToolDefinition:
    """Build the first-party grep harness tool.

    Args:
        cwd: Optional working directory used to resolve relative paths. When
            omitted, the current working directory is captured at construction
            time.

    Returns:
        HarnessToolDefinition: Executable grep tool with prompt metadata.
    """
    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))

    async def run_grep(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        del cancellation_token
        try:
            return _search_files(args, resolver=resolver)
        except Exception:
            return _GENERIC_GREP_ERROR

    return HarnessToolDefinition(
        tool=Tool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_grep,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


def _search_files(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver,
) -> str:
    """Search files with ripgrep and render a plain-text tool result."""
    validation_error = _validate_args(args)
    if validation_error is not None:
        return validation_error

    search_path = resolver.resolve(args.path or ".")
    content = _collect_grep_content(search_path, args=args, resolver=resolver)
    if content.error is not None:
        return content.error

    return _format_grep_output(search_path, content, resolver=resolver)


def _validate_args(args: _ToolArgs) -> str | None:
    """Return one model-facing validation error when arguments are invalid."""
    if args.pattern == "":
        return "pattern must not be empty"
    if args.path is not None and args.path.strip() == "":
        return "path must not be empty"
    if args.glob is not None and args.glob.strip() == "":
        return "glob must not be empty"
    if args.file_type is not None and args.file_type.strip() == "":
        return "type must not be empty"
    return None


def _collect_grep_content(
    search_path: Path,
    *,
    args: _ToolArgs,
    resolver: ToolPathResolver,
) -> _GrepContent:
    """Validate inputs and dispatch to a mode-specific search."""
    path_error = _validate_search_path(search_path)
    if path_error is not None:
        return _GrepContent(lines=(), error=path_error)

    binary_error = _validate_explicit_binary_file(search_path)
    if binary_error is not None:
        return _GrepContent(lines=(), error=binary_error)

    if _is_ignored_explicit_search_path(search_path):
        return _GrepContent(lines=(_NO_MATCHES,))

    if not _matches_explicit_file_glob(
        search_path,
        cwd=resolver.cwd,
        glob=args.glob,
    ):
        return _GrepContent(lines=(_NO_MATCHES,))

    if args.output_mode == "content":
        return _content_search(search_path, args=args, resolver=resolver)

    return _listing_search(search_path, args=args, resolver=resolver)


def _validate_search_path(search_path: Path) -> str | None:
    """Return an operational path error before invoking ripgrep."""
    if not search_path.exists():
        return f"path not found: `{search_path}`"
    if not search_path.is_file() and not search_path.is_dir():
        return f"path is not a file or directory: `{search_path}`"
    return None


def _validate_explicit_binary_file(search_path: Path) -> str | None:
    """Return a binary-file error for explicit file searches."""
    if not search_path.is_file():
        return None
    try:
        is_binary = is_likely_binary_file(search_path)
    except PermissionError:
        return f"permission denied: `{search_path}`"
    except OSError:
        return f"failed to read file: `{search_path}`"

    if not is_binary:
        return None

    return f"file appears to be binary and cannot be searched as text: `{search_path}`"


def _is_ignored_explicit_search_path(search_path: Path) -> bool:
    """Return whether ripgrep would skip an explicitly requested path."""
    if not search_path.is_file() and not search_path.is_dir():
        return False

    matcher = GitignoreMatcher.from_path(search_path)
    return matcher.is_ignored(search_path, is_dir=search_path.is_dir())


def _matches_explicit_file_glob(
    search_path: Path,
    *,
    cwd: Path,
    glob: str | None,
) -> bool:
    """Apply the model-visible glob to explicit file path searches."""
    if glob is None or not search_path.is_file():
        return True

    display_path = _display_path(search_path, cwd)
    parent_relative = _display_path(search_path, search_path.parent)
    return (
        fnmatch.fnmatchcase(search_path.name, glob)
        or fnmatch.fnmatchcase(display_path, glob)
        or fnmatch.fnmatchcase(parent_relative, glob)
    )


def _content_search(
    search_path: Path,
    *,
    args: _ToolArgs,
    resolver: ToolPathResolver,
) -> _GrepContent:
    """Run ripgrep in JSON content mode and render matching lines."""
    try:
        ripgrep = _build_ripgrep_content(search_path, args=args)
    except RipGrepNotFound:
        return _GrepContent(lines=(), error=_RIPGREP_NOT_FOUND_ERROR)

    output = ripgrep.run().as_string
    parsed = _parse_ripgrep_output(output)
    if parsed.error is not None:
        return _GrepContent(lines=(), error=parsed.error)

    if parsed.match_count == 0:
        return _GrepContent(lines=(_NO_MATCHES,))

    rendered = _render_rows(
        rows=parsed.rows,
        context=args.context,
        limit=args.limit,
        cwd=resolver.cwd,
    )

    continuation_messages: list[str] = []
    if parsed.match_count > args.limit:
        continuation_messages.append(
            f"Showing first {args.limit} matches; more remain."
        )
    if rendered.byte_truncated:
        continuation_messages.append(f"Output truncated after {TEXT_MAX_BYTES} bytes")
    if rendered.line_truncated:
        continuation_messages.append(
            f"One or more matching lines were truncated after {GREP_MAX_LINE_LENGTH} characters"
        )

    return _GrepContent(
        lines=rendered.lines,
        continuation_messages=tuple(continuation_messages),
    )


def _listing_search(
    search_path: Path,
    *,
    args: _ToolArgs,
    resolver: ToolPathResolver,
) -> _GrepContent:
    """Run a non-JSON ripgrep mode (files_with_matches/count) and render rows."""
    try:
        ripgrep = _build_ripgrep_listing(search_path, args=args)
    except RipGrepNotFound:
        return _GrepContent(lines=(), error=_RIPGREP_NOT_FOUND_ERROR)

    output = ripgrep.run().as_string
    text_error = _detect_text_error(output)
    if text_error is not None:
        return _GrepContent(lines=(), error=text_error)

    raw_rows = [
        line
        for line in output.splitlines()
        if line and not _is_ripgrep_warning_line(line)
    ]
    if not raw_rows:
        return _GrepContent(lines=(_NO_MATCHES,))

    output_lines: list[str] = []
    output_bytes = 0
    overflow = False
    byte_truncated = False
    for row in raw_rows:
        if len(output_lines) >= args.limit:
            overflow = True
            break

        rendered_row = _render_listing_row(row, args=args, cwd=resolver.cwd)
        appended = _append_with_byte_limit(output_lines, rendered_row, output_bytes)
        if appended is None:
            byte_truncated = True
            break

        output_bytes = appended

    continuation_messages: list[str] = []
    if overflow:
        continuation_messages.append(f"Showing first {args.limit} files; more remain.")
    if byte_truncated:
        continuation_messages.append(f"Output truncated after {TEXT_MAX_BYTES} bytes")

    return _GrepContent(
        lines=tuple(output_lines),
        continuation_messages=tuple(continuation_messages),
    )


def _render_listing_row(row: str, *, args: _ToolArgs, cwd: Path) -> str:
    """Render one ripgrep listing row relative to cwd."""
    if args.output_mode == "files_with_matches":
        return _display_path(Path(row), cwd)
    path_str, separator, count_str = row.rpartition(":")
    if separator == "":
        return row
    return f"{_display_path(Path(path_str), cwd)}:{count_str}"


def _build_ripgrep_base(search_path: Path, *, args: _ToolArgs) -> Ripgrepy:
    """Build the shared ripgrep options used by every output mode."""
    ripgrep = (
        Ripgrepy(args.pattern, search_path.as_posix())
        .no_messages()
        .hidden()
        .glob("!.git")
        .glob("!**/.git")
        .sort("path")
    )

    if args.ignore_case:
        ripgrep.ignore_case()
    if args.literal:
        ripgrep.fixed_strings()
    if args.multiline:
        ripgrep.multiline().multiline_dotall()
    if args.glob is not None:
        ripgrep.glob(args.glob)
    if args.file_type is not None:
        ripgrep.type_(args.file_type)

    return ripgrep


def _build_ripgrep_content(search_path: Path, *, args: _ToolArgs) -> Ripgrepy:
    """Add content-mode flags on top of the shared ripgrep options."""
    ripgrep = _build_ripgrep_base(search_path, args=args)
    (
        ripgrep.with_filename()
        .line_number()
        .no_heading()
        .max_count(args.limit + 1)
        .json()
    )
    if args.context > 0:
        ripgrep.context(args.context)
    return ripgrep


def _build_ripgrep_listing(search_path: Path, *, args: _ToolArgs) -> Ripgrepy:
    """Add files_with_matches/count flags on top of the shared options."""
    ripgrep = _build_ripgrep_base(search_path, args=args)
    if args.output_mode == "files_with_matches":
        ripgrep.files_with_matches()
    else:
        ripgrep.with_filename().count_matches()
    return ripgrep


def _detect_text_error(output: str) -> str | None:
    """Return a model-facing error when ripgrep printed an arg-parse failure."""
    if output.startswith("rg: regex parse error:"):
        return _INVALID_REGEX_ERROR
    if output.startswith("rg: error parsing glob "):
        return _INVALID_GLOB_ERROR
    return None


def _is_ripgrep_warning_line(line: str) -> bool:
    """Return whether one ripgrep output line is a non-fatal per-file warning.

    Ripgrep emits these for unreadable, denied, or binary files alongside real
    matches; they should not poison results from the rest of the search.
    """
    return line.startswith("rg: ") or "binary file matches" in line.lower()


def _parse_ripgrep_output(output: str) -> _ParsedRipgrepOutput:
    """Parse ripgrep JSON lines without exposing raw ripgrep errors."""
    if output == "":
        return _ParsedRipgrepOutput(rows=(), match_count=0)
    text_error = _detect_text_error(output)
    if text_error is not None:
        return _ParsedRipgrepOutput(rows=(), match_count=0, error=text_error)

    rows: list[_RipgrepRow] = []
    match_count = 0
    for line in output.splitlines():
        try:
            raw_message = json.loads(line)
        except json.JSONDecodeError:
            if _is_ripgrep_warning_line(line):
                continue

            return _ParsedRipgrepOutput(
                rows=(),
                match_count=0,
                error=_GENERIC_GREP_ERROR,
            )

        message = _as_mapping(raw_message)
        if message is None:
            return _ParsedRipgrepOutput(
                rows=(),
                match_count=0,
                error=_GENERIC_GREP_ERROR,
            )

        row = _parse_ripgrep_row(message)
        if row is None:
            continue
        rows.append(row)
        if row.is_match:
            match_count += 1

    return _ParsedRipgrepOutput(rows=tuple(rows), match_count=match_count)


def _parse_ripgrep_row(message: JsonObject) -> _RipgrepRow | None:
    """Parse one ripgrep JSON message into a renderable row."""
    message_type = message.get("type")
    if message_type not in {"match", "context"}:
        return None

    data = _as_mapping(message.get("data"))
    if data is None:
        return None

    path_text = _json_data_text(data.get("path"))
    line_text = _json_data_text(data.get("lines"))
    line_number = _json_line_number(data.get("line_number"))
    if path_text is None or line_text is None or line_number is None:
        return None

    return _RipgrepRow(
        path=Path(path_text),
        line_number=line_number,
        line=line_text.removesuffix("\n").removesuffix("\r"),
        is_match=message_type == "match",
    )


def _as_mapping(value: object) -> JsonObject | None:
    """Return a JSON mapping when a parsed value has object shape."""
    if isinstance(value, dict):
        return cast(JsonObject, value)
    return None


def _json_data_text(value: object) -> str | None:
    """Decode ripgrep JSON text-or-bytes data fields."""
    data = _as_mapping(value)
    if data is None:
        return None

    text = data.get("text")
    if isinstance(text, str):
        return text

    encoded_bytes = data.get("bytes")
    if not isinstance(encoded_bytes, str):
        return None

    try:
        raw_bytes = base64.b64decode(encoded_bytes)
    except (binascii.Error, ValueError):
        return None
    return raw_bytes.decode("utf-8", errors="replace")


def _json_line_number(value: object) -> int | None:
    """Return one 1-based line number from ripgrep JSON data."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _render_rows(
    *,
    rows: tuple[_RipgrepRow, ...],
    context: int,
    limit: int,
    cwd: Path,
) -> _RenderedMatches:
    """Render ripgrep rows using grep-style path and line prefixes."""
    selected_rows = _select_rows(rows=rows, context=context, limit=limit)
    output: list[str] = []
    output_bytes = 0
    line_truncated = False
    previous_row: _RipgrepRow | None = None

    for row in selected_rows:
        if (
            context > 0
            and previous_row is not None
            and _has_context_gap(
                previous=previous_row,
                current=row,
            )
        ):
            append_result = _append_with_byte_limit(output, "--", output_bytes)
            if append_result is None:
                return _RenderedMatches(
                    lines=tuple(output),
                    byte_truncated=True,
                    line_truncated=line_truncated,
                )
            output_bytes = append_result

        line, was_truncated = _truncate_result_line(row.line)
        line_truncated = line_truncated or was_truncated
        rendered = _result_line(
            path=row.path,
            line_number=row.line_number,
            line=line,
            cwd=cwd,
        )
        append_result = _append_with_byte_limit(output, rendered, output_bytes)
        if append_result is None:
            return _RenderedMatches(
                lines=tuple(output),
                byte_truncated=True,
                line_truncated=line_truncated,
            )
        output_bytes = append_result
        previous_row = row

    return _RenderedMatches(lines=tuple(output), line_truncated=line_truncated)


def _select_rows(
    *,
    rows: tuple[_RipgrepRow, ...],
    context: int,
    limit: int,
) -> tuple[_RipgrepRow, ...]:
    """Select rows for the first global match limit."""
    selected_matches = _first_match_rows(rows, limit=limit)
    if context == 0:
        return selected_matches

    selected_line_numbers = {
        (row.path, line_number)
        for row in selected_matches
        for line_number in range(
            row.line_number - context, row.line_number + context + 1
        )
        if line_number >= 1
    }
    return tuple(
        row for row in rows if (row.path, row.line_number) in selected_line_numbers
    )


def _first_match_rows(
    rows: Iterable[_RipgrepRow],
    *,
    limit: int,
) -> tuple[_RipgrepRow, ...]:
    """Return the first matching rows up to the caller limit."""
    matches: list[_RipgrepRow] = []
    for row in rows:
        if not row.is_match:
            continue
        if len(matches) >= limit:
            break
        matches.append(row)
    return tuple(matches)


def _has_context_gap(*, previous: _RipgrepRow, current: _RipgrepRow) -> bool:
    """Return whether a rendered context separator is needed."""
    return (
        previous.path != current.path or current.line_number > previous.line_number + 1
    )


def _truncate_result_line(line: str) -> tuple[str, bool]:
    """Apply grep's per-line character cap."""
    if len(line) <= GREP_MAX_LINE_LENGTH:
        return line, False
    return line[:GREP_MAX_LINE_LENGTH], True


def _result_line(*, path: Path, line_number: int, line: str, cwd: Path) -> str:
    """Render one grep result line."""
    return f"{_display_path(path, cwd)}:{line_number}:{line}"


def _append_with_byte_limit(
    lines: list[str],
    line: str,
    current_bytes: int,
) -> int | None:
    """Append a line and return the new byte count when it fits."""
    separator_bytes = 1 if lines else 0
    line_bytes = len(line.encode("utf-8"))
    next_bytes = current_bytes + separator_bytes + line_bytes

    if next_bytes > TEXT_MAX_BYTES:
        return None

    lines.append(line)
    return next_bytes


def _format_grep_output(
    path: Path,
    content: _GrepContent,
    *,
    resolver: ToolPathResolver,
) -> str:
    """Render the final model-facing grep result."""
    output = [f"Search path: {_display_path(path, resolver.cwd)}"]
    output.extend(content.lines)
    output.extend(content.continuation_messages)
    return "\n".join(output)


def _display_path(path: Path, base: Path) -> str:
    """Render a stable relative path when possible."""
    if path == base:
        return "."
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
