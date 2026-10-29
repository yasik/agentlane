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
from agentlane.harness.mcp._result import (
    read_result,
    render_mcp_result,
    render_mcp_storage_text,
)
from agentlane.models import ToolFailure


async def _prepare(
    result: object,
    writer: FileWriter,
    *,
    limit: int = 512,
    directory: str = "results",
    secrets: tuple[str, ...] = (),
    include_structured_content: bool = False,
    max_content_blocks: int = 32,
) -> str | ToolFailure:
    return await render_mcp_result_with_storage(
        result,
        policy=MCPResultPolicy(
            max_text_chars=limit,
            include_structured_content=include_structured_content,
            max_content_blocks=max_content_blocks,
        ),
        secrets=secrets,
        writer=writer,
        directory=directory,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [200, 10_000])
async def test_exact_fit_is_inline_and_overflow_saves_the_same_text(
    tmp_path: Path,
    size: int,
) -> None:
    filesystem = LocalFileSystem(tmp_path)
    result = types.CallToolResult(content=[types.TextContent(text="x" * size)])
    inline = render_mcp_result(result, MCPResultPolicy(max_text_chars=size + 1000))
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
async def test_small_result_keeps_existing_resource_and_media_preview(
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
    assert "Transcript" in inline
    assert "Embedded text" not in inline and "meeting-123" not in inline
    assert "textCharsOmitted" in inline
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
    response = await _prepare(
        result, filesystem, secrets=(secret,), include_structured_content=True
    )
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
@pytest.mark.parametrize("extra_chars", [-1, 0])
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
async def test_block_preview_limit_saves_all_supported_blocks(tmp_path: Path) -> None:
    content = [{"type": "text", "text": f"block-{index}"} for index in range(40)]
    response = await _prepare(
        {"content": content},
        LocalFileSystem(tmp_path),
        limit=10_000,
        max_content_blocks=1,
    )
    (path,) = tmp_path.rglob("*.txt")
    payload = json.loads("".join(path.read_text().splitlines()))
    assert payload["content"] == content
    assert "block-0" in response and "block-1" not in response
    assert "[Full MCP result: " in response
    assert len(response) <= 10_000


@pytest.mark.asyncio
async def test_saved_text_resource_keeps_its_body_and_readable_lines(
    tmp_path: Path,
) -> None:
    transcript = "x" * 100_000 + "final words"
    response = await _prepare(
        types.CallToolResult(
            content=[
                types.EmbeddedResource(
                    resource=types.TextResourceContents(
                        uri="resource://notes/1", text=transcript
                    )
                )
            ]
        ),
        LocalFileSystem(tmp_path),
    )
    (path,) = tmp_path.rglob("*.txt")
    lines = path.read_text().splitlines()
    payload = json.loads("".join(lines))
    assert payload["content"][0]["resource"]["text"] == transcript
    assert all(len(line) <= 4096 for line in lines)
    assert len(response) <= 512
    assert "[Full MCP result: " in response


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["nodes", "depth", "input", "output"])
async def test_processing_limits_do_not_write_partial_results(
    kind: str, tmp_path: Path
) -> None:
    if kind == "nodes":
        value: object = [f"node-{index}" for index in range(5000)]
    elif kind == "depth":
        value = "deep-value"
        for _ in range(40):
            value = {"nested": value}
    elif kind == "input":
        value = "x" * (8 * 1024 * 1024 + 1)
    else:
        value = "\x00" * (2 * 1024 * 1024)

    response = await _prepare(
        {
            "content": [{"type": "text", "text": "credential-canary"}],
            "structuredContent": value,
        },
        LocalFileSystem(tmp_path),
        include_structured_content=True,
        secrets=("credential-canary",),
    )
    assert not tuple(tmp_path.iterdir())
    assert "processing limit" in response.lower()
    assert "[Full MCP result: " not in response
    assert "credential-canary" not in response
    assert len(response) <= 512


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", ["type", "text"])
async def test_redacted_block_keys_remain_readable(tmp_path: Path, secret: str) -> None:
    response = await _prepare(
        {"content": [{"type": "text", "text": "hello"}]},
        LocalFileSystem(tmp_path),
        secrets=(secret,),
    )
    assert "hello" in response
    assert "[redacted]" in response
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_saved_text_can_use_more_than_the_inline_content_share(
    tmp_path: Path,
) -> None:
    transcript = "x" * (5 * 1024 * 1024)
    response = await _prepare(
        {
            "content": [{"type": "text", "text": transcript}],
            "structuredContent": {"id": "meeting"},
        },
        LocalFileSystem(tmp_path),
        include_structured_content=True,
    )
    (path,) = tmp_path.rglob("*.txt")
    payload = json.loads("".join(path.read_text().splitlines()))
    assert payload["content"][0]["text"] == transcript
    assert payload["structuredContent"] == {"id": "meeting"}
    assert "[Full MCP result: " in response
    assert len(response) <= 512


@pytest.mark.asyncio
async def test_redacted_key_collision_does_not_save_incomplete_content(
    tmp_path: Path,
) -> None:
    response = await _prepare(
        {
            "content": [],
            "structuredContent": {"first-key": "first", "second-key": "second"},
        },
        LocalFileSystem(tmp_path),
        include_structured_content=True,
        secrets=("first-key", "second-key"),
    )
    assert not tuple(tmp_path.iterdir())
    assert "full result not saved" in response
    assert "[Full MCP result: " not in response


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", [False, True])
@pytest.mark.parametrize(
    "kind", ["depth", "duplicate", "escaped-duplicate", "nested-duplicate"]
)
async def test_unverifiable_json_is_omitted_without_saving(
    tmp_path: Path, storage: bool, kind: str
) -> None:
    if kind == "depth":
        text = "[" * 15_000 + r'{"pass\u0077ord":"credential-canary"}' + "]" * 15_000
    else:
        key = r"rec\u006frd" if kind == "escaped-duplicate" else "record"
        text = (
            '{"record":"first-canary","' + key + '":"second-canary",'
            '"padding":"' + "x" * 1000 + '"}'
        )
        if kind == "nested-duplicate":
            text = json.dumps({"nested": text})
    limit = 64 * 1024 if kind == "depth" and not storage else 512
    result = types.CallToolResult(content=[types.TextContent(text=text)])
    response = (
        await _prepare(result, LocalFileSystem(tmp_path))
        if storage
        else render_mcp_result(result, MCPResultPolicy(max_text_chars=limit))
    )
    assert len(response) <= limit
    assert "canary" not in response
    assert "omitted" in response
    assert not tuple(tmp_path.iterdir())
    assert "[Full MCP result:" not in response
    if storage:
        assert "full result not saved" in response
    else:
        assert json.loads(response)["truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", [False, True])
async def test_oversized_text_is_rejected_before_stripping(
    tmp_path: Path, storage: bool
) -> None:
    stripped: list[int] = []

    class ObservedText(str):
        def lstrip(self, chars: str | None = None) -> str:
            stripped.append(len(self))
            return super().lstrip(chars)

    size = 8 * 1024 * 1024 if storage else 64 * 1024
    text = ObservedText(" " + "x" * size)
    result = {"content": [{"type": "text", "text": text}]}
    response = (
        await _prepare(result, LocalFileSystem(tmp_path))
        if storage
        else render_mcp_result(result, MCPResultPolicy(max_text_chars=512))
    )
    assert stripped == []
    assert "omitted" in response
    assert len(response) <= 512
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["text", "structured", "resource"])
@pytest.mark.parametrize("is_error", [False, True])
def test_overflow_preview_is_bounded_before_writing(kind: str, is_error: bool) -> None:
    transcript = "visit \n🙂" * 20_000 + "final words"
    result = types.CallToolResult(
        content=[types.TextContent(text=transcript)], is_error=is_error
    )
    if kind == "structured":
        result.structured_content = {"transcript": transcript}
    elif kind == "resource":
        result.content = [
            types.EmbeddedResource(
                resource=types.TextResourceContents(
                    uri="resource://notes/1", text=transcript
                )
            )
        ]
    text, preview, complete = render_mcp_storage_text(
        read_result(result), MCPResultPolicy(max_text_chars=512), secrets=()
    )
    assert complete
    assert len(text) > 512
    assert len(preview) <= 512
    saved = json.loads(text)
    inline = json.loads(preview)
    assert saved["isError"] is inline["isError"] is is_error
    if kind == "structured":
        assert saved["structuredContent"]["transcript"] == transcript
        assert saved["content"][0]["text"] == transcript
        assert bool(inline["content"]) is is_error
        assert "structuredContent" in inline
    elif kind == "resource":
        assert saved["content"][0]["resource"]["text"] == transcript
        assert "text" not in inline["content"][0]["resource"]
        assert inline["content"][0]["resource"]["textCharsOmitted"] == len(transcript)
    else:
        assert saved["content"][0]["text"] == transcript
        assert inline["truncated"] is True
