"""Test MCP result storage and previews."""

import json
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID

import pytest
from mcp import types

from agentlane.harness.filesystem import FileWriter, LocalFileSystem
from agentlane.harness.mcp import MCPResultPolicy
from agentlane.harness.mcp._output import render_mcp_result_with_storage
from agentlane.harness.mcp._result import render_mcp_result
from agentlane.models import ToolFailure


async def _prepare(
    result: object,
    writer: FileWriter | None,
    *,
    limit: int = 512,
    directory: str = "results",
    secrets: tuple[str, ...] = (),
) -> str | ToolFailure:
    return await render_mcp_result_with_storage(
        result,
        policy=MCPResultPolicy(max_text_chars=limit, include_structured_content=False),
        secrets=secrets,
        writer=writer,
        directory=directory if writer is not None else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [10, 100_000])
async def test_no_writer_keeps_existing_response(size: int) -> None:
    result = types.CallToolResult(content=[types.TextContent(text="x" * size)])
    expected = render_mcp_result(
        result, MCPResultPolicy(max_text_chars=512, include_structured_content=False)
    )
    assert await _prepare(result, None) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [200, 10_000])
async def test_exact_fit_is_inline_and_overflow_saves_the_same_text(
    tmp_path: Path,
    size: int,
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    result = types.CallToolResult(content=[types.TextContent(text="x" * size)])
    inline = render_mcp_result(
        result, MCPResultPolicy(max_text_chars=size + 1000), complete=True
    )
    assert await _prepare(result, filesystem, limit=size + 1000) == inline
    assert await _prepare(result, filesystem, limit=len(inline)) == inline
    assert not tuple(tmp_path.iterdir())

    response = await _prepare(result, filesystem, limit=len(inline) - 1)
    (path,) = tmp_path.rglob("*.txt")
    notice = f"\n\n[Full MCP result: {json.dumps(str(path.relative_to(tmp_path)))}]"
    assert "".join(path.read_text().splitlines()) == inline
    assert len(response) <= len(inline) - 1
    assert response.endswith(notice)
    assert inline.startswith(response.removesuffix(notice).removesuffix("…"))
    assert path.stem == str(UUID(path.stem))


@pytest.mark.asyncio
async def test_small_result_keeps_text_and_binary_metadata_without_saving(
    tmp_path: Path,
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    result = types.CallToolResult(
        content=[
            types.TextContent(text="Transcript"),
            types.EmbeddedResource(
                resource=types.TextResourceContents(
                    uri="resource://notes/1", text="Embedded text"
                )
            ),
            types.ImageContent(data="binary-canary", mime_type="image/png"),
        ],
        structured_content={"meeting_id": "meeting-123"},
    )
    inline = await _prepare(result, filesystem, limit=1000)
    assert not tuple(tmp_path.iterdir())
    assert all(
        text in inline for text in ("Transcript", "Embedded text", "meeting-123")
    )
    assert "image/png" in inline and "binary-canary" not in inline


@pytest.mark.asyncio
async def test_redaction_runs_before_budgeting_and_preserves_full_text(
    tmp_path: Path,
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    secret = "credential-canary-" + "s" * 5000
    inline = await _prepare(
        types.CallToolResult(content=[types.TextContent(text=secret)]),
        filesystem,
        secrets=(secret,),
    )
    assert "[redacted]" in inline and secret not in inline
    assert not tuple(tmp_path.iterdir())

    meeting_id = "meeting-" + "x" * 20_000
    result = types.CallToolResult(
        content=[
            types.TextContent(
                text=json.dumps(
                    {
                        "meeting_id": meeting_id,
                        "transcript": "a" * 4090 + secret + "final words",
                        "escaped": '{"pass\\u0077ord": "unknown-secret-canary"}',
                    }
                )
            )
        ],
        structured_content={"authorization": secret, "extra": "structured-only"},
    )
    response = await _prepare(result, filesystem, secrets=(secret,))
    (path,) = tmp_path.rglob("*.txt")
    saved = "".join(path.read_text().splitlines())
    assert "credential-canary" not in saved + response
    assert "unknown-secret-canary" not in saved + response
    assert all(text in saved for text in (meeting_id, "final words", "structured-only"))


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
@pytest.mark.parametrize("outcome", ["inline", "saved", "failed"])
async def test_storage_outcomes_keep_server_error_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_error: bool, outcome: str
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    if outcome == "failed":
        monkeypatch.setattr(
            filesystem, "open_write", Mock(side_effect=TimeoutError("private-detail"))
        )
    text = "Not found" + ("x" * 1000 if outcome != "inline" else "")
    result = types.CallToolResult(
        content=[types.TextContent(text=text)], is_error=is_error
    )
    response = await _prepare(result, filesystem)
    assert "Not found" in response and len(response) <= 512
    assert isinstance(response, ToolFailure) is is_error
    if isinstance(response, ToolFailure):
        assert response.error.kind == "mcp_server"
    if outcome == "saved":
        (path,) = tmp_path.rglob("*.txt")
        assert "Not found" in path.read_text()
        assert "[Full MCP result: " in response
    else:
        assert not tuple(tmp_path.iterdir())
    if outcome == "failed":
        assert response.endswith("\n\n[Full MCP result unavailable.]")
        assert "private-detail" not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_chars", [-1, 0, 1])
async def test_quoted_path_notice_at_the_response_budget_boundary(
    extra_chars: int, tmp_path: Path
) -> None:
    limit = 128
    prefix = 'results/notes\n"quoted" '
    sample_notice = (
        f"\n\n[Full MCP result: {json.dumps(f'{prefix}/' + 'x' * 36 + '.txt')}]"
    )
    directory = prefix + "x" * (limit + extra_chars - len(sample_notice))
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text="x" * 5000)]),
        LocalFileSystem(tmp_path),
        directory=directory,
        limit=limit,
    )
    (path,) = tmp_path.rglob("*.txt")
    relative = str(path.relative_to(tmp_path))
    notice = f"\n\n[Full MCP result: {json.dumps(relative)}]"
    assert len(notice) == limit + extra_chars
    assert len(response) <= limit
    if extra_chars > 0:
        assert response.endswith("\n\n[Full MCP result unavailable.]")
    else:
        assert response.endswith(notice)
        assert json.loads(response.split("[Full MCP result: ", 1)[1][:-1]) == relative
        if extra_chars == 0:
            assert response == notice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "directory", ["results/credential-canary", 'results/"token":"hidden"']
)
async def test_credential_paths_are_rejected_before_writing(
    directory: str, tmp_path: Path
) -> None:
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text="x" * 5000)]),
        LocalFileSystem(tmp_path),
        directory=directory,
        secrets=("credential-canary",),
    )
    assert not tuple(tmp_path.iterdir())
    assert response.endswith("\n\n[Full MCP result unavailable.]")
    assert directory not in response


@pytest.mark.asyncio
async def test_more_than_eight_mib_is_saved_with_readable_lines(tmp_path: Path) -> None:
    transcript = "x" * (8 * 1024 * 1024) + "final words beyond eight MiB"
    response = await _prepare(
        types.CallToolResult(content=[types.TextContent(text=transcript)]),
        LocalFileSystem(tmp_path),
    )
    (path,) = tmp_path.rglob("*.txt")
    lines = path.read_text().splitlines()
    assert transcript in "".join(lines)
    assert all(len(line) <= 4096 for line in lines)
    assert len(response) <= 512
    assert response.endswith(
        f"\n\n[Full MCP result: {json.dumps(str(path.relative_to(tmp_path)))}]"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nodes", "depth", "blocks", "integer"])
async def test_preview_processing_limits_do_not_drop_saved_text(
    kind: str, tmp_path: Path
) -> None:
    content: list[object] = []
    structured: object = None
    if kind == "nodes":
        structured = [f"node-{index}" for index in range(5000)]
        final_value = "node-4999"
    elif kind == "depth":
        final_value = "deep text"
        structured = final_value
        for _ in range(40):
            structured = {"nested": structured}
    elif kind == "blocks":
        content = [{"type": "text", "text": f"block-{index}"} for index in range(4100)]
        final_value = "block-4099"
    else:
        structured = 1 << 5000
        final_value = str(structured)

    response = await _prepare(
        {"content": content, "structuredContent": structured},
        LocalFileSystem(tmp_path),
        limit=128,
    )
    (path,) = tmp_path.rglob("*.txt")
    assert final_value in "".join(path.read_text().splitlines())
    assert len(response) <= 512
