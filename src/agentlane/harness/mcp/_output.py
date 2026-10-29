"""Save oversized MCP results for later reads."""

import json
from collections.abc import Collection
from pathlib import PurePosixPath
from uuid import uuid4

from agentlane.models import ToolFailure

from .._storage_write import write_content
from ..filesystem import FileWriter
from ._redaction import redact_known_secrets, redact_secret_pairs
from ._result import (
    read_result,
    render_mcp_storage_text,
    result_response,
)
from ._types import MCPResultPolicy

_UNAVAILABLE = "\n\n[Full MCP result unavailable.]"
"""Returned when the full result cannot be saved."""

_RESULT_REFERENCE = "\n\n[Full MCP result: {}]"

_FILE_LINE_CHARS = 4096
"""Keep saved lines within the native read tool byte limit, including UTF-8."""


def validate_result_storage(
    writer: FileWriter | None,
    directory: str | None,
    max_text_chars: int,
) -> None:
    if (writer is None) != (directory is None):
        raise ValueError(
            "result_writer and result_directory must be configured together."
        )
    if directory is None:
        return

    # UUID filenames have a fixed length; include JSON escaping of the directory.
    path = str(PurePosixPath(directory) / "00000000-0000-0000-0000-000000000000.txt")
    reference = _RESULT_REFERENCE.format(json.dumps(path, ensure_ascii=False))
    if len(reference) > max_text_chars:
        raise ValueError(
            "MCP result reference exceeds max_text_chars. "
            "Use a shorter result_directory or increase max_text_chars."
        )


async def render_mcp_result_with_storage(
    result: object,
    *,
    policy: MCPResultPolicy,
    secrets: Collection[str],
    writer: FileWriter,
    directory: str | None,
) -> str | ToolFailure:
    source = read_result(result)
    is_error = source.is_error
    try:
        text, preview, complete = render_mcp_storage_text(
            source, policy, secrets=secrets
        )
    except Exception:
        return result_response(
            "[MCP result unavailable: could not prepare readable text.]", is_error
        )

    if (
        complete
        and max(len(text), len(preview)) <= policy.max_text_chars
        and len(source.content) <= policy.max_content_blocks
    ):
        return result_response(preview, is_error)

    if not complete:
        notice = "\n\n[MCP processing limit exceeded; full result not saved.]"
    else:
        try:
            notice = await _save_result(
                text, writer=writer, directory=directory, secrets=secrets
            )
        except Exception:
            notice = _UNAVAILABLE

    # Truncate the preview, never the file path the model needs to read it.
    available = policy.max_text_chars - len(notice)
    preview_text = preview[:available]
    if len(preview) > available and available:
        preview_text = preview_text[:-1] + "…"

    return result_response(preview_text + notice, is_error)


async def _save_result(
    text: str,
    *,
    writer: FileWriter,
    directory: str | None,
    secrets: Collection[str],
) -> str:
    if not directory:
        raise ValueError("MCP result storage requires a directory.")
    read_path = str(PurePosixPath(directory) / f"{uuid4()}.txt")
    if (
        redact_known_secrets(read_path, secrets) != read_path
        or redact_secret_pairs(read_path) != read_path
    ):
        raise ValueError("MCP output path contains credentials.")

    # read cannot return a single line larger than its byte limit.
    file_text = "\n".join(
        text[i : i + _FILE_LINE_CHARS] for i in range(0, len(text), _FILE_LINE_CHARS)
    )
    await write_content(writer, read_path, file_text.encode("utf-8"))

    return _RESULT_REFERENCE.format(json.dumps(read_path, ensure_ascii=False))
