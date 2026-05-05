"""Read tool implementation for first-party harness base tools."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.runtime import CancellationToken

from ._output import TEXT_MAX_BYTES, TEXT_MAX_LINES
from ._paths import ToolPathResolver
from ._types import HarnessToolDefinition

_BINARY_SAMPLE_BYTES = 4096
_TOOL_NAME = "read"
_TOOL_DESCRIPTION = (
    "Reads raw local text file contents. Supports offset and limit for large "
    f"files. Output is truncated to {TEXT_MAX_LINES} lines or "
    f"{TEXT_MAX_BYTES} bytes, whichever is hit first."
)
_TOOL_PROMPT_SNIPPET = "Read file contents"
_TOOL_PROMPT_GUIDELINE = "Use read to examine files instead of cat or sed."
_GENERIC_READ_ERROR = "failed to read file"


class _ToolArgs(BaseModel):
    """Model-visible arguments for the read tool."""

    path: str = Field(description="File path to read.")
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


@dataclass(frozen=True, slots=True)
class _ReadContent:
    """Raw file slice and any continuation note."""

    lines: tuple[str, ...]
    continuation_message: str | None = None
    error: str | None = None


def read_tool(*, cwd: str | Path | None = None) -> HarnessToolDefinition:
    """Build the first-party text-file read harness tool.

    Args:
        cwd: Optional working directory used to resolve relative paths. When
            omitted, the current working directory is captured at construction
            time.

    Returns:
        HarnessToolDefinition: Executable read tool with prompt metadata.
    """
    resolver = ToolPathResolver() if cwd is None else ToolPathResolver(cwd=Path(cwd))

    async def run_read(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
    ) -> str:
        del cancellation_token
        try:
            return _read_file(args, resolver=resolver)
        except Exception:
            return _GENERIC_READ_ERROR

    return HarnessToolDefinition(
        tool=Tool(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_read,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE,),
    )


def _read_file(args: _ToolArgs, *, resolver: ToolPathResolver) -> str:
    """Read one file and render a plain-text tool result."""
    if args.offset is not None and args.offset < 1:
        return "offset must be a 1-indexed line number"
    if args.limit is not None and args.limit < 1:
        return "limit must be greater than zero"
    if args.path.strip() == "":
        return "path must not be empty"

    resolved_path = resolver.resolve(args.path)
    if resolved_path.is_dir():
        return f"path is a directory: `{resolved_path}`"

    content = _read_text_slice(
        resolved_path,
        offset=args.offset or 1,
        limit=args.limit,
    )
    if content.error is not None:
        return content.error

    return _format_read_output(content)


def _read_text_slice(
    path: Path,
    *,
    offset: int,
    limit: int | None,
) -> _ReadContent:
    """Read a text slice from disk without loading the full file."""
    try:
        binary_file = path.open("rb")
    except FileNotFoundError:
        return _read_error(f"file not found: `{path}`")
    except PermissionError:
        return _read_error(f"permission denied: `{path}`")
    except OSError:
        return _read_error(f"failed to read file: `{path}`")

    with binary_file:
        sample = binary_file.read(_BINARY_SAMPLE_BYTES)
        if b"\x00" in sample:
            return _read_error(
                f"file appears to be binary and cannot be read as text: `{path}`"
            )
        binary_file.seek(0)
        return _collect_text_slice(binary_file, offset=offset, limit=limit)


def _read_error(message: str) -> _ReadContent:
    """Build one model-facing read error result."""
    return _ReadContent(lines=(), error=message)


def _collect_text_slice(
    lines: Iterable[bytes],
    *,
    offset: int,
    limit: int | None,
) -> _ReadContent:
    """Collect a bounded line slice from an iterable byte stream."""
    output_lines: list[str] = []
    output_bytes = 0
    total_lines = 0
    continuation_message: str | None = None
    max_returned_lines = TEXT_MAX_LINES if limit is None else min(limit, TEXT_MAX_LINES)

    for line_number, raw_line in enumerate(lines, start=1):
        total_lines = line_number
        if line_number < offset:
            continue

        if len(output_lines) >= max_returned_lines:
            continuation_message = _line_continuation_message(
                offset=offset,
                returned_line_count=len(output_lines),
            )
            break

        decoded_line = _decode_line(raw_line)
        is_continuation = bool(output_lines)
        line_byte_count = _joined_line_byte_count(
            decoded_line, has_previous=is_continuation
        )

        if output_bytes + line_byte_count > TEXT_MAX_BYTES:
            if output_lines:
                continuation_message = _byte_continuation_message(
                    offset=offset,
                    returned_line_count=len(output_lines),
                    next_offset=line_number,
                )
            else:
                continuation_message = _oversized_line_message(
                    line_number=line_number,
                    line_bytes=line_byte_count,
                )
            break

        output_lines.append(decoded_line)
        output_bytes += line_byte_count

    if (
        not output_lines
        and continuation_message is None
        and (
            (total_lines == 0 and offset > 1)
            or (total_lines > 0 and offset > total_lines)
        )
    ):
        return _read_error("offset exceeds file length")

    return _ReadContent(
        lines=tuple(output_lines),
        continuation_message=continuation_message,
    )


def _decode_line(line: bytes) -> str:
    """Decode one line and trim one line ending."""
    return line.decode("utf-8", errors="replace").removesuffix("\n").removesuffix("\r")


def _joined_line_byte_count(line: str, *, has_previous: bool) -> int:
    """Return byte count for one line in a newline-joined output."""
    separator_bytes = 1 if has_previous else 0
    return separator_bytes + len(line.encode("utf-8"))


def _line_continuation_message(
    *,
    offset: int,
    returned_line_count: int,
) -> str:
    """Build the model-facing continuation note for line-bounded output."""
    end_line = offset + returned_line_count - 1
    next_offset = end_line + 1
    return f"[Showing lines {offset}-{end_line}. Use offset={next_offset} to continue.]"


def _byte_continuation_message(
    *,
    offset: int,
    returned_line_count: int,
    next_offset: int,
) -> str:
    """Build the model-facing continuation note for byte-bounded output."""
    end_line = offset + returned_line_count - 1
    return (
        f"[Showing lines {offset}-{end_line} ({TEXT_MAX_BYTES} byte limit). "
        f"Use offset={next_offset} to continue.]"
    )


def _oversized_line_message(*, line_number: int, line_bytes: int) -> str:
    """Build the model-facing note for a single line beyond the byte limit."""
    return (
        f"[Line {line_number} is {line_bytes} bytes, exceeds "
        f"{TEXT_MAX_BYTES} byte limit. Use bash to inspect it.]"
    )


def _format_read_output(content: _ReadContent) -> str:
    """Render the final model-facing tool result."""
    output = list(content.lines)
    if content.continuation_message is not None:
        if output:
            output.append("")
        output.append(content.continuation_message)
    return "\n".join(output)
