"""MCP result limits bound work before omitted values are processed."""

import asyncio
import json
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Any, overload

import pytest
from mcp import types

from agentlane.harness.mcp import MCPResultPolicy
from agentlane.harness.mcp import _result as mcp_result
from agentlane.harness.mcp._redaction import redact_known_secrets
from agentlane.harness.mcp._result import render_mcp_result


class _CountingBlocks(Sequence[object]):
    def __init__(self, count: int) -> None:
        self.size = count
        self.reads = 0

    def __len__(self) -> int:
        return self.size

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            raise AssertionError("Do not copy a content slice.")
        if index >= self.size:
            raise IndexError(index)

        self.reads += 1
        return {"type": "text", "text": "a useful result"}


class _CountingMetadata(Mapping[str, object]):
    def __init__(self, count: int) -> None:
        self.count = count
        self.reads = 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[str]:
        return (f"item-{index}" for index in range(self.count))

    def __getitem__(self, key: str) -> object:
        self.reads += 1
        return key


def test_sdk_result_and_content_do_not_require_full_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = types.CallToolResult(content=[types.TextContent(text="useful")])

    def reject_dump(*args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        raise AssertionError("Do not dump the full SDK result or content block.")

    monkeypatch.setattr(types.CallToolResult, "model_dump", reject_dump)
    monkeypatch.setattr(types.TextContent, "model_dump", reject_dump)
    payload = json.loads(render_mcp_result(result, MCPResultPolicy()))
    assert payload["content"][0]["text"] == "useful"


@pytest.mark.parametrize("count", [1000, 100_000])
def test_block_limit_does_not_visit_omitted_content(count: int) -> None:
    content = _CountingBlocks(count)
    rendered = render_mcp_result(
        {"content": content}, MCPResultPolicy(max_content_blocks=1)
    )
    assert content.reads == 1
    assert json.loads(rendered)["omittedBlocks"] == count - 1
    assert "omittedChars" not in rendered


@pytest.mark.parametrize("count", [1000, 100_000])
def test_output_limit_stops_visiting_structured_content(count: int) -> None:
    metadata = _CountingMetadata(count)
    rendered = render_mcp_result(
        {"content": [], "structuredContent": metadata},
        MCPResultPolicy(max_text_chars=128),
    )
    assert len(rendered) <= 128
    assert metadata.reads <= 4
    assert json.loads(rendered)["truncated"] is True


def test_node_limit_stops_a_wide_result() -> None:
    metadata = _CountingMetadata(10_000)
    rendered = render_mcp_result(
        {"content": [], "structuredContent": metadata},
        MCPResultPolicy(max_text_chars=1_000_000),
    )
    assert metadata.reads <= 4096
    assert json.loads(rendered)["truncated"] is True


@pytest.mark.parametrize(
    "text",
    [
        "visible-prefix " + "x" * (10 * 1024 * 1024),
        json.dumps({"password": "credential-canary", "padding": "x" * 70_000}),
        "https://user:credential-canary@example.test/" + "x" * 70_000,
        "x" * (64 * 1024 - 8) + "credential-canary",
    ],
    ids=["large-text", "large-json", "large-url", "boundary-secret"],
)
def test_oversized_strings_are_omitted_before_redaction_or_parsing(text: str) -> None:
    result = types.CallToolResult(content=[types.TextContent(text=text)])
    rendered = render_mcp_result(
        result,
        MCPResultPolicy(max_text_chars=128),
        secrets={"credential-canary"},
    )
    assert len(rendered) <= 128
    assert "visible-prefix" not in rendered
    assert "credential" not in rendered
    assert "omitted" in json.loads(rendered)["content"][0]["text"]
    assert json.loads(rendered)["truncated"] is True


def test_oversized_mapping_key_is_not_partially_exposed() -> None:
    result: dict[str, object] = {
        "content": [],
        "structuredContent": {"credential-canary" + "x" * 70_000: "not visited"},
    }
    rendered = render_mcp_result(result, MCPResultPolicy(max_text_chars=128))
    assert "credential" not in rendered
    assert json.loads(rendered)["truncated"] is True


def test_sensitive_mapping_value_is_not_visited() -> None:
    class SecretMapping(Mapping[str, object]):
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[str]:
            return iter(("password",))

        def __getitem__(self, key: str) -> object:
            raise AssertionError(f"Do not read the value for {key}.")

    result: dict[str, object] = {"content": [], "structuredContent": SecretMapping()}
    rendered = render_mcp_result(result, MCPResultPolicy())
    assert json.loads(rendered)["structuredContent"] == {"password": "[redacted]"}


@pytest.mark.parametrize("as_json_text", [False, True])
def test_depth_limit_stops_before_deep_values(as_json_text: bool) -> None:
    value: object = "deep-value-canary"
    for _ in range(100):
        value = {"nested": value}

    result: dict[str, object] = (
        {"content": [{"type": "text", "text": json.dumps(value)}]}
        if as_json_text
        else {"content": [], "structuredContent": value}
    )
    rendered = render_mcp_result(result, MCPResultPolicy())
    assert "deep-value-canary" not in rendered
    assert "omitted" in rendered
    assert json.loads(rendered)["truncated"] is True


def test_oversized_text_does_not_enter_redaction_or_json_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = types.CallToolResult(
        content=[types.TextContent(text='{"password":"' + "x" * 10_000_000 + '"}')]
    )
    encode = json.dumps
    decode = json.loads

    def bounded_redact(value: object, secrets: Collection[str]) -> object:
        assert not isinstance(value, str) or len(value) <= 64 * 1024
        return redact_known_secrets(value, secrets)

    def bounded_encode(value: object, **kwargs: Any) -> str:
        assert not isinstance(value, str) or len(value) <= 128
        return encode(value, **kwargs)

    def bounded_decode(value: str, **kwargs: Any) -> Any:
        assert len(value) <= 64 * 1024
        return decode(value, **kwargs)

    monkeypatch.setattr(mcp_result, "redact_known_secrets", bounded_redact)
    monkeypatch.setattr(json, "dumps", bounded_encode)
    monkeypatch.setattr(json, "loads", bounded_decode)
    rendered = render_mcp_result(result, MCPResultPolicy(max_text_chars=128))
    assert len(rendered) <= 128
    assert "omitted" in rendered


def test_input_character_budget_is_shared_across_blocks() -> None:
    secret = "test-only-secret-" * 3000
    result = types.CallToolResult(
        content=[types.TextContent(text=secret), types.TextContent(text=secret)]
    )
    rendered = render_mcp_result(result, MCPResultPolicy(), secrets={secret})
    content = json.loads(rendered)["content"]
    assert content[0]["text"] == "[redacted]"
    assert "omitted" in content[1]["text"]
    assert "test-only-secret" not in rendered


@pytest.mark.parametrize("text", ['{"password":"credential-canary"', "[notes]"])
def test_unparseable_json_text_is_omitted(text: str) -> None:
    rendered = render_mcp_result(
        types.CallToolResult(content=[types.TextContent(text=text)]), MCPResultPolicy()
    )
    payload = json.loads(rendered)
    assert "omitted" in payload["content"][0]["text"]
    assert "credential-canary" not in rendered
    assert payload["truncated"] is True


def test_ordinary_prose_remains_unchanged() -> None:
    text = "Read the [notes](https://example.test/notes), then continue."
    rendered = render_mcp_result(
        types.CallToolResult(content=[types.TextContent(text=text)]), MCPResultPolicy()
    )
    payload = json.loads(rendered)
    assert payload["content"][0]["text"] == text
    assert "truncated" not in payload


def test_sdk_resource_fields_remain_shallow_and_safe() -> None:
    result = types.CallToolResult(
        content=[
            types.ImageContent(data="binary-canary", mime_type="image/png"),
            types.AudioContent(data="binary-canary", mime_type="audio/wav"),
            types.EmbeddedResource(
                resource=types.TextResourceContents(
                    uri="resource://notes/1", text="resource-body-canary"
                )
            ),
            types.ResourceLink(
                name="Notes", uri="https://example.test/notes", mime_type="text/plain"
            ),
        ]
    )
    payload = json.loads(render_mcp_result(result, MCPResultPolicy()))
    assert payload["content"][0]["encodedBytesOmitted"] == len("binary-canary")
    assert payload["content"][1]["encodedBytesOmitted"] == len("binary-canary")
    assert payload["content"][2]["resource"]["textCharsOmitted"] == len(
        "resource-body-canary"
    )
    assert payload["content"][3]["type"] == "resource_link"
    assert payload["content"][3]["mimeType"] == "text/plain"
    assert "canary" not in json.dumps(payload)


def test_depth_omissions_still_consume_the_shared_node_budget() -> None:
    content = _CountingBlocks(100_000)
    value: object = content
    for _ in range(31):
        value = [value]

    rendered = render_mcp_result(
        {"content": [], "structuredContent": value},
        MCPResultPolicy(max_text_chars=1_000_000),
    )
    assert content.reads <= 4096
    assert json.loads(rendered)["truncated"] is True


@pytest.mark.asyncio
async def test_large_result_does_not_stall_the_event_loop() -> None:
    result = types.CallToolResult(content=[types.TextContent(text="x" * 10_000_000)])
    loop = asyncio.get_running_loop()
    tick: asyncio.Future[float] = loop.create_future()
    started = time.monotonic()
    loop.call_soon(lambda: tick.set_result(time.monotonic() - started))

    rendered = render_mcp_result(result, MCPResultPolicy(max_text_chars=128))
    delay = await asyncio.wait_for(tick, timeout=1)

    # Counting fixtures and helper guards above are the primary cost checks.
    # This loose ceiling also detects an accidental event-loop stall in CI.
    assert delay < 0.5
    assert len(rendered) <= 128


def test_json_text_over_half_the_budget_renders_complete() -> None:
    transcript = "Speaker 1: a long meeting. " * 1500
    text = json.dumps({"id": "meeting-1", "text": transcript})
    rendered = render_mcp_result(
        types.CallToolResult(content=[types.TextContent(text=text)]), MCPResultPolicy()
    )
    payload = json.loads(rendered)
    assert json.loads(payload["content"][0]["text"])["text"] == transcript


def test_structured_result_with_repeated_text_renders_structured_only() -> None:
    structured = {"id": "meeting-1", "text": "Speaker 1: a long meeting. " * 1500}
    rendered = render_mcp_result(
        types.CallToolResult(
            content=[types.TextContent(text=json.dumps(structured))],
            structured_content=structured,
        ),
        MCPResultPolicy(),
    )
    payload = json.loads(rendered)
    assert payload["structuredContent"] == structured
    assert "content" not in payload


def test_nested_json_text_parses_within_the_input_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "[1." + "0" * 30_000 + "]"
    while len(json.dumps([text])) <= 60_000:
        text = json.dumps([text])

    parsed: list[int] = []
    decode = json.loads

    def counting_decode(value: str, **kwargs: Any) -> Any:
        parsed.append(len(value))
        return decode(value, **kwargs)

    monkeypatch.setattr(json, "loads", counting_decode)
    render_mcp_result(
        types.CallToolResult(content=[types.TextContent(text=text)]), MCPResultPolicy()
    )
    assert sum(parsed) <= 64 * 1024


def test_structured_result_with_non_text_blocks_renders_content() -> None:
    rendered = render_mcp_result(
        types.CallToolResult(
            content=[
                types.TextContent(text='{"chart": "weekly"}'),
                types.ImageContent(data="chart-bytes", mime_type="image/png"),
            ],
            structured_content={"chart": "weekly"},
        ),
        MCPResultPolicy(),
    )
    payload = json.loads(rendered)
    assert payload["content"][1]["mimeType"] == "image/png"
    assert "structuredContent" not in payload
