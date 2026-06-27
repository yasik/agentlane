"""Read tool for skill resources stored on a `SkillFilesystem`.

The first-party `read` tool reads the local disk. When skills live on a custom
`SkillFilesystem` (a remote repository, an object store), their bundled
resources are not on disk, so this tool reads them through the same filesystem
the loader uses. The model joins the `Skill directory: ...` line from an
activated skill with a listed resource path and passes the result here. The
loader records a skill's directory as ``<root>/<skill dir>``, so a configured
root prefixes that path; the tool strips the matching root and reads the
remainder from it, rejecting paths under no configured root.
"""

from collections.abc import Sequence
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ..tools import TEXT_MAX_LINES, HarnessToolDefinition
from ._filesystem import SkillFilesystem

_BINARY_SAMPLE_BYTES = 4096
_TOOL_NAME = "read"
_TOOL_DESCRIPTION = (
    "Reads a skill resource file by its root-relative path. Supports offset and "
    f"limit for large files. Output is truncated to {TEXT_MAX_LINES} lines."
)
_TOOL_PROMPT_SNIPPET = "Read file contents"
_TOOL_PROMPT_GUIDELINE = (
    "Use read to open a skill's bundled resources. Resource paths are listed "
    "relative to the skill directory; join them with the `Skill directory: ...` "
    "line shown when the skill activates."
)


class _ToolArgs(BaseModel):
    """Model-visible arguments for the skill-resource read tool."""

    path: str = Field(
        description=(
            "Path of a skill resource: the skill directory shown when the skill "
            "activated, joined with a listed resource path."
        )
    )
    offset: int | None = Field(
        default=None,
        description=(
            "The line number to start reading from. Must be 1 or greater. "
            "Defaults to 1."
        ),
    )
    limit: int | None = Field(
        default=None,
        description="The maximum number of lines to return.",
    )


def filesystem_read_tool(
    filesystem: SkillFilesystem,
    roots: Sequence[str],
) -> HarnessToolDefinition:
    """Build a read tool that resolves skill-resource paths through a filesystem.

    Args:
        filesystem: Storage the skill resources are read from.
        roots: Allowed roots, matched against a path's leading prefix. Match the
            roots a `FilesystemSkillLoader` discovers from so the model can read
            any resource it was shown, and no path outside them.

    Returns:
        HarnessToolDefinition: Executable read tool with prompt metadata.
    """
    allowed_roots = tuple(roots)

    async def run_read(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del cancellation_token, context
        if args.offset is not None and args.offset < 1:
            return "offset must be a 1-indexed line number"
        if args.limit is not None and args.limit < 1:
            return "limit must be greater than zero"
        if args.path.strip() == "":
            return "path must not be empty"

        normalized = _normalize_resource_path(args.path)
        if normalized is None:
            return f"path is not a valid skill-relative path: `{args.path}`"

        return await _read_resource(
            filesystem,
            roots=allowed_roots,
            path=normalized,
            offset=args.offset or 1,
            limit=args.limit,
        )

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str](
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_read,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


async def _read_resource(
    filesystem: SkillFilesystem,
    *,
    roots: Sequence[str],
    path: str,
    offset: int,
    limit: int | None,
) -> str:
    """Read one resource, selecting its root by the configured root that prefixes it."""
    selected = _select_root(path, roots)
    if selected is None:
        return f"file not found: `{path}`"
    root, in_root_path = selected
    if not in_root_path:
        return f"path is a directory: `{path}`"

    try:
        data = await filesystem.read_bytes(root, in_root_path)
    except FileNotFoundError:
        return f"file not found: `{path}`"
    except IsADirectoryError:
        return f"path is a directory: `{path}`"

    if b"\x00" in data[:_BINARY_SAMPLE_BYTES]:
        return f"file appears to be binary and cannot be read as text: `{path}`"

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return f"file appears to be binary and cannot be read as text: `{path}`"

    return _format_line_slice(text, offset=offset, limit=limit)


def _format_line_slice(text: str, *, offset: int, limit: int | None) -> str:
    """Render a bounded line window with a continuation note when truncated."""
    lines = text.splitlines()
    if offset > len(lines):
        return "offset exceeds file length"

    max_returned_lines = TEXT_MAX_LINES if limit is None else min(limit, TEXT_MAX_LINES)
    start = offset - 1
    window = lines[start : start + max_returned_lines]

    end_line = start + len(window)
    output = list(window)
    if end_line < len(lines):
        if output:
            output.append("")
        output.append(
            f"[Showing lines {offset}-{end_line}. "
            f"Use offset={end_line + 1} to continue.]"
        )
    return "\n".join(output)


def _select_root(path: str, roots: Sequence[str]) -> tuple[str, str] | None:
    """Return the (root, in-root path) for the configured root that prefixes `path`.

    The loader records a skill directory as ``<root>/<skill dir>`` and a root may
    itself contain ``/`` (e.g. ``org/repo``), so roots are matched longest-first and
    the matched root is stripped. A path under no configured root has no match.
    """
    for root in sorted(roots, key=len, reverse=True):
        if path == root:
            return root, ""
        prefix = f"{root}/"
        if path.startswith(prefix):
            return root, path[len(prefix) :]
    return None


def _normalize_resource_path(path: str) -> str | None:
    """Return a clean root-relative path, rejecting absolute paths and traversal."""
    pure = PurePosixPath(path)
    if pure.is_absolute():
        return None
    parts = [part for part in pure.parts if part != "."]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)
