"""Patch tool implementation for first-party harness base tools."""

from pathlib import Path

import patch_tool as patch_engine
from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._paths import ToolPathResolver
from ._types import HarnessToolDefinition

_TOOL_NAME = "patch"
_TOOL_DESCRIPTION = (
    "Apply precise SEARCH/REPLACE edits to an existing local text file. "
    "The path is provided as a structured argument; edits must contain one or "
    "more bare SEARCH/REPLACE blocks. Edits are all-or-nothing."
)
_TOOL_PROMPT_SNIPPET = "Apply search/replace edits to existing files"
_TOOL_PROMPT_GUIDELINES = (
    "Use patch for precise edits to existing files after reading them; use write for new files or complete rewrites.",
    "Each patch SEARCH block must match exactly one location; include enough surrounding lines to make it unique.",
)
_GENERIC_PATCH_ERROR = "failed to patch file"


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


def patch_tool(*, cwd: str | Path | None = None) -> HarnessToolDefinition:
    """Build the first-party patch harness tool.

    Args:
        cwd: Optional working directory used to resolve relative paths. When
            omitted, the current working directory is captured at construction
            time.

    Returns:
        HarnessToolDefinition: Executable patch tool with prompt metadata.
    """
    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))

    async def run_patch(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        del cancellation_token
        try:
            return _patch_file(args, resolver=resolver)
        except Exception:
            return _GENERIC_PATCH_ERROR

    return HarnessToolDefinition(
        tool=Tool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_patch,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=_TOOL_PROMPT_GUIDELINES,
    )


def _patch_file(args: _ToolArgs, *, resolver: ToolPathResolver) -> str:
    """Parse, apply, and render one model-facing patch result."""
    validation_error = _validate_args(args)
    if validation_error is not None:
        return validation_error

    resolved_path = resolver.resolve(args.path)
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
        result = patch_engine.apply_edits(resolved_path, edits)
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
    except FileNotFoundError:
        return f"File not found: {resolved_path}"
    except PermissionError:
        return f"Permission denied: {resolved_path}"
    except UnicodeDecodeError:
        return f"File is not valid UTF-8: {resolved_path}"
    except OSError:
        return f"Failed to patch file: {resolved_path}"

    return _format_success(result)


def _validate_args(args: _ToolArgs) -> str | None:
    """Return a model-facing validation error when arguments are invalid."""
    if args.path.strip() == "":
        return "path must not be empty"
    if "\x00" in args.path:
        return "path contains a null byte"
    try:
        args.edits.encode("utf-8")
    except UnicodeEncodeError:
        return "patch is not valid UTF-8"
    return None


def _format_success(result: patch_engine.EditResult) -> str:
    """Render a successful patch result as deterministic text."""
    edit_noun = "edit" if result.edits_applied == 1 else "edits"
    return f"Applied {result.edits_applied} {edit_noun} to {result.path}."


def _format_parse_error(exc: patch_engine.ParseError) -> str:
    """Render a parser error without exposing stack traces."""
    return f"Invalid patch format: {exc}"


def _edit_label(index: int | None) -> str:
    """Return a 1-indexed edit label for model-facing errors."""
    if index is None:
        return "edit"
    return f"edit {index + 1}"


def _format_not_found_error(
    path: Path,
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
    path: Path,
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
    path: Path,
    edit_index: int | None,
    total_edits: int,
) -> str:
    """Render a recoverable empty SEARCH text error."""
    if total_edits == 1:
        return f"SEARCH text must not be empty in {path}."
    return f"{_edit_label(edit_index)} SEARCH text must not be empty in {path}."


def _format_overlap_error(
    path: Path,
    edit_index: int | None,
    other_edit_index: int | None,
) -> str:
    """Render a recoverable overlapping edits error."""
    return (
        f"{_edit_label(edit_index)} and {_edit_label(other_edit_index)} overlap "
        f"in {path}. Merge them into one edit or target disjoint regions."
    )


def _format_no_changes_error(path: Path, total_edits: int) -> str:
    """Render a recoverable no-op replacement error."""
    if total_edits == 1:
        return f"No changes made to {path}. The replacement produced identical content."
    return f"No changes made to {path}. The replacements produced identical content."
