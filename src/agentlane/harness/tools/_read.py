"""Read tool implementation for first-party harness base tools."""

import asyncio
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import BufferedReader, BytesIO
from pathlib import Path, PurePath

from pydantic import BaseModel, Field

from agentlane.harness.filesystem import FileReader, LocalFileSystem
from agentlane.io import buffered_reader
from agentlane.models import Tool, ToolExecutionContext
from agentlane.runtime import CancellationToken

from ._output import TEXT_MAX_BYTES, TEXT_MAX_LINES
from ._paths import StorageToolPathResolver, ToolPathResolver, tool_path_guideline
from ._permissions import (
    ToolApprovalCallback,
    ToolOperation,
    ToolPermissionPolicy,
    ToolPermissionRequest,
    evaluate_tool_permission,
)
from ._types import HarnessToolDefinition

_BINARY_SAMPLE_BYTES = 4096
_TOOL_NAME = "read"
_TOOL_DESCRIPTION = (
    "Reads raw text file contents. Supports offset and limit for large "
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
    """Decoded lines selected for the tool output."""

    continuation_message: str | None = None
    """Instructions to retrieve content omitted by an output limit."""

    error: str | None = None
    """Sanitized failure message when no file slice can be returned."""


def read_tool(
    *,
    cwd: str | Path | None = None,
    reader: FileReader | None = None,
    permissions: ToolPermissionPolicy | None = None,
    approval_callback: ToolApprovalCallback | None = None,
) -> HarnessToolDefinition:
    """Build the first-party text-file read harness tool.

    Args:
        cwd: Optional working directory used to resolve relative paths. When
            omitted, the current working directory is captured at construction
            time. Mounted readers default to the virtual root; other injected
            readers default to `.` within their storage namespace.
        reader: Optional binary file reader. Defaults to the local filesystem.
        permissions: Optional policy for read-file permission decisions.
        approval_callback: Optional callback for approval-required decisions.

    Returns:
        HarnessToolDefinition: Executable read tool with prompt metadata.
    """
    resolver = (
        ToolPathResolver.for_optional(cwd)
        if reader is None
        else StorageToolPathResolver.for_optional(cwd, filesystem=reader)
    )
    file_reader = reader if reader is not None else LocalFileSystem()

    async def run_read(
        args: _ToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        if cancellation_token.is_cancelled:
            raise asyncio.CancelledError

        try:
            return await _read_file(
                args,
                resolver=resolver,
                reader=file_reader,
                cancellation_token=cancellation_token,
                permissions=permissions,
                approval_callback=approval_callback,
                context=context,
            )
        except Exception:
            return _GENERIC_READ_ERROR

    return HarnessToolDefinition(
        tool=Tool[_ToolArgs, str](
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            args_model=_ToolArgs,
            handler=run_read,
        ),
        prompt_snippet=_TOOL_PROMPT_SNIPPET,
        prompt_guidelines=(_TOOL_PROMPT_GUIDELINE, tool_path_guideline(resolver)),
    )


async def _read_file(
    args: _ToolArgs,
    *,
    resolver: ToolPathResolver | StorageToolPathResolver,
    reader: FileReader,
    cancellation_token: CancellationToken,
    permissions: ToolPermissionPolicy | None,
    approval_callback: ToolApprovalCallback | None,
    context: ToolExecutionContext,
) -> str:
    """Read one file and render a plain-text tool result."""
    if args.offset is not None and args.offset < 1:
        return "offset must be a 1-indexed line number"
    if args.limit is not None and args.limit < 1:
        return "limit must be greater than zero"
    if args.path.strip() == "":
        return "path must not be empty"

    resolved_path = await asyncio.to_thread(resolver.resolve, args.path)
    permission_error = await evaluate_tool_permission(
        ToolPermissionRequest(
            tool_name=_TOOL_NAME,
            operation=ToolOperation.READ_FILE,
            cwd=resolver.cwd,
            path=resolved_path,
        ),
        policy=permissions,
        approval_callback=approval_callback,
        context=context,
    )
    if permission_error is not None:
        return permission_error

    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    content = await asyncio.to_thread(
        _read_text_slice,
        resolved_path,
        reader=reader,
        offset=args.offset or 1,
        limit=args.limit,
    )
    if cancellation_token.is_cancelled:
        raise asyncio.CancelledError

    if content.error is not None:
        return content.error

    return _format_read_output(content)


def _read_text_slice(
    path: PurePath,
    *,
    reader: FileReader,
    offset: int,
    limit: int | None,
) -> _ReadContent:
    """Read and close a bounded text slice in the worker that opens the stream."""
    try:
        # Providers need only byte reads. The wrapper supplies line buffering,
        # and the outer context remains responsible for closing the source.
        with (
            reader.open_read(str(path)) as source,
            buffered_reader(source) as binary_file,
        ):
            sample = bytearray()
            while len(sample) < _BINARY_SAMPLE_BYTES:
                chunk = binary_file.read(_BINARY_SAMPLE_BYTES - len(sample))
                if not chunk:
                    break
                sample.extend(chunk)

            if b"\x00" in sample:
                return _read_error(
                    f"file appears to be binary and cannot be read as text: `{path}`"
                )

            # Reuse the sampled bytes; remote streams need not support seeking.
            return _collect_text_slice(
                _replay_lines(bytes(sample), binary_file), offset=offset, limit=limit
            )
    except IsADirectoryError:
        return _read_error(f"path is a directory: `{path}`")
    except FileNotFoundError:
        return _read_error(f"file not found: `{path}`")
    except PermissionError:
        return _read_error(f"permission denied: `{path}`")
    except OSError:
        return _read_error(f"failed to read file: `{path}`")


def _replay_lines(sample: bytes, stream: BufferedReader) -> Iterator[bytes]:
    """Replay sampled bytes and complete the last line without seeking the stream."""
    buffered = BytesIO(sample)
    while line := buffered.readline():
        if not line.endswith(b"\n"):
            # Sampling can stop inside a line or UTF-8 character. Complete the
            # line before decoding so that sampling does not change its text.
            line += stream.readline()

        yield line

    while line := stream.readline():
        yield line


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

    # No output can mean either an invalid offset or a valid but oversized line.
    # Preserve the continuation diagnostic in the latter case.
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
