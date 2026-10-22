"""Convert MCP tool results to bounded, safe model-facing text."""

import json
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, cast

from pydantic import AnyUrl, BaseModel

from agentlane.models import ToolError, ToolFailure

from ._errors import MCP_SERVER_TOOL_ERROR_KIND
from ._redaction import is_secret_key, redact_known_secrets, redact_secret_pairs
from ._types import MCPResultPolicy

_OMISSION = "[omitted: exceeds processing limit]"
_MAX_NODES = 4096
_MAX_DEPTH = 32
_NOT_JSON = object()
"""Returned by `_parse_json` when text is not valid JSON."""


class _SDKResult(Protocol):
    content: Sequence[object]
    structured_content: object
    is_error: bool


@dataclass(frozen=True, slots=True)
class _Result:
    content: Sequence[object]
    structured_content: object
    is_error: bool


def render_mcp_result(
    result: object,
    policy: MCPResultPolicy,
    *,
    secrets: Collection[str] = (),
    complete: bool = False,
) -> str | ToolFailure:
    """Render sanitized text; complete=True disables preview limits."""
    source = read_result(result)
    renderer = _Renderer(policy.max_text_chars, secrets, complete=complete)

    # An empty structured result carries no data, so it does not replace text.
    structured = (
        source.structured_content
        if complete or (policy.include_structured_content and source.structured_content)
        else None
    )
    reserved: dict[str, object] = {
        "content": [],
        "isError": source.is_error,
        "truncated": True,
        "omittedBlocks": len(source.content),
    }
    if structured is not None:
        reserved["structuredContent"] = {"omitted": True}

    available = (
        sys.maxsize if complete else policy.max_text_chars - len(_json(reserved))
    )
    content_budget = 2 + (int(available * 0.6) if structured is not None else available)

    # Servers repeat structured content in text blocks for clients that cannot
    # read it, so skip that copy. Keep the text blocks of a failure, because
    # the server puts the failure details there.
    content, kept = renderer.content(
        source.content,
        max(2, content_budget),
        len(source.content) if complete else policy.max_content_blocks,
        skip_text=not complete and structured is not None and not source.is_error,
    )
    fields = [f'"content":{content}', f'"isError":{_json(source.is_error)}']
    if structured is not None:
        structured_budget = len(_json(reserved["structuredContent"])) + (
            available - (len(content) - 2)
        )
        preview = renderer.value(structured, structured_budget) or '{"omitted":true}'
        fields.append(f'"structuredContent":{preview}')

    omitted_blocks = len(source.content) - kept
    if renderer.truncated or omitted_blocks:
        fields.extend(['"truncated":true', f'"omittedBlocks":{omitted_blocks}'])

    text = "{" + ",".join(fields) + "}"
    return result_response(text, source.is_error)


def result_response(text: str, is_error: bool) -> str | ToolFailure:
    if is_error:
        return ToolFailure(
            text=text,
            error=ToolError(
                message="MCP server reported a tool failure.",
                kind=MCP_SERVER_TOOL_ERROR_KIND,
            ),
        )

    return text


def read_result(value: object) -> _Result:
    if isinstance(value, Mapping):
        raw = cast(Mapping[str, object], value)
        content = raw.get("content", [])
        return _Result(
            content=_sequence(content),
            structured_content=raw.get(
                "structuredContent", raw.get("structured_content")
            ),
            is_error=raw.get("isError", raw.get("is_error", False)) is True,
        )

    source = cast(_SDKResult, value)
    return _Result(source.content, source.structured_content, source.is_error)


def _parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _NOT_JSON


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return cast(Sequence[object], value)

    return ()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _fields(value: Mapping[str, object] | BaseModel) -> Iterator[tuple[str, str]]:
    if isinstance(value, BaseModel):
        for name, field in type(value).model_fields.items():
            yield name, field.serialization_alias or field.alias or name

        for name in value.model_extra or {}:
            yield name, name

        return

    for name in value:
        yield name, name


def _field(value: Mapping[str, object] | BaseModel, name: str) -> object:
    if isinstance(value, BaseModel):
        return getattr(value, name)

    return value[name]


def _block_type(block: object) -> object | None:
    if isinstance(block, (Mapping, BaseModel)):
        return _field(cast(Mapping[str, object] | BaseModel, block), "type")

    return None


class _Renderer:
    """Own the input and output budgets for one result, including JSON text."""

    def __init__(
        self, output_limit: int, secrets: Collection[str], *, complete: bool = False
    ) -> None:
        self.complete = complete
        self.characters = sys.maxsize if complete else max(64 * 1024, output_limit)
        self.nodes = sys.maxsize if complete else _MAX_NODES
        self.secrets = tuple(secrets)
        self.truncated = False

    def _take_text(self, value: str) -> bool:
        if len(value) > self.characters:
            self.truncated = True
            return False

        self.characters -= len(value)
        return True

    def _omission(self, available: int) -> str | None:
        self.truncated = True
        return self._text(_OMISSION, available)

    def _text(self, value: str, available: int) -> str | None:
        if available < 2:
            self.truncated = True
            return None

        # Only serialize a prefix of already sanitized text. Raw strings must
        # pass the complete-string processing budget before this method runs.
        prefix = value[:available]
        encoded = _json(prefix)
        if len(prefix) == len(value) and len(encoded) <= available:
            return encoded

        self.truncated = True
        if available < 3:
            return '""'

        low, high = 0, len(prefix)
        while low < high:
            midpoint = (low + high + 1) // 2
            if len(_json(prefix[:midpoint] + "…")) <= available:
                low = midpoint
            else:
                high = midpoint - 1

        return _json(prefix[:low] + "…")

    def value(
        self,
        value: object,
        available: int,
        depth: int = 0,
        *,
        resource: bool = False,
        block: bool = False,
        charged: bool = False,
    ) -> str | None:
        if available < 2:
            self.truncated = True
            return None

        if self.nodes <= 0:
            return self._omission(available)

        self.nodes -= 1
        if not self.complete and depth >= _MAX_DEPTH:
            return self._omission(available)

        if isinstance(value, (Mapping, BaseModel)):
            return self._object(
                cast(Mapping[str, object] | BaseModel, value),
                available,
                depth,
                resource=resource,
                block=block,
                charged=charged,
            )

        if self.complete and isinstance(value, AnyUrl):
            value = str(value)

        if isinstance(value, str):
            # Charge JSON text before each parse. A string parsed from charged
            # JSON text is part of that text, so do not charge it again.
            is_json = value.lstrip().startswith(("{", "["))
            if (is_json or not charged) and not self._take_text(value):
                return self._omission(available)

            # Render JSON text as structure, so credential keys are redacted.
            # Text that does not parse is text, so redact it as text.
            parsed = _parse_json(value) if is_json else _NOT_JSON
            if parsed is not _NOT_JSON:
                safe = (
                    self.value(parsed, max(2, available - 2), depth + 1, charged=True)
                    or "null"
                )
            else:
                text = cast(str, redact_known_secrets(value, self.secrets))
                safe = redact_secret_pairs(text)

            return self._text(safe, available)

        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return self._array(
                cast(Sequence[object], value), available, depth, charged=charged
            )

        if value is None or isinstance(value, (bool, int, float)):
            if isinstance(value, int) and value.bit_length() > available * 3:
                return self._omission(available)

            if isinstance(value, float) and not isfinite(value):
                return self._omission(available)

            encoded = _json(value)
            if len(encoded) <= available:
                return encoded

        return self._omission(available)

    def _object(
        self,
        value: Mapping[str, object] | BaseModel,
        available: int,
        depth: int,
        *,
        resource: bool,
        block: bool,
        charged: bool,
    ) -> str | None:
        kind = _field(value, "type") if block else None
        fields = _fields(value)
        if block:
            fields = self._content_fields(fields, text=kind == "text")

        parts: list[str] = []
        used = 2
        kept: set[str] = set()
        for name, alias in fields:
            if self.nodes <= 0 or available - used < 5:
                self.truncated = True
                break

            self.nodes -= 1
            if not charged and not self._take_text(alias):
                continue

            if len(alias) > available - used - 5:
                self.truncated = True
                continue

            output_name = alias
            omitted_body = (
                resource
                and (alias == "blob" or (alias == "text" and not self.complete))
            ) or (block and kind in {"image", "audio"} and alias == "data")
            if omitted_body:
                output_name = (
                    "textCharsOmitted" if alias == "text" else "encodedBytesOmitted"
                )

            safe_key = cast(str, redact_known_secrets(output_name, self.secrets))
            if len(safe_key) > available - used - 5:
                self.truncated = True
                continue

            encoded_key = _json(safe_key)
            remaining = available - used - len(encoded_key) - 1 - bool(parts)
            if remaining < 2:
                self.truncated = True
                break

            if is_secret_key(alias):
                encoded = self._text("[redacted]", remaining)
            else:
                item = _field(value, name)
                if omitted_body:
                    item = len(item) if isinstance(item, str) else 0

                encoded = self.value(
                    item,
                    remaining,
                    depth + 1,
                    resource=block and kind == "resource" and alias == "resource",
                    charged=charged,
                )

            if encoded is None:
                break

            part = encoded_key + ":" + encoded
            used += len(part) + bool(parts)
            parts.append(part)
            kept.add(alias)

        if block and ("type" not in kept or (kind == "text" and "text" not in kept)):
            self.truncated = True
            return None

        return "{" + ",".join(parts) + "}"

    @staticmethod
    def _content_fields(
        fields: Iterator[tuple[str, str]], *, text: bool
    ) -> Iterator[tuple[str, str]]:
        yield "type", "type"
        if text:
            yield "text", "text"

        for name, alias in fields:
            if alias != "type" and not (text and alias == "text"):
                yield name, alias

    def _array(
        self, value: Sequence[object], available: int, depth: int, *, charged: bool
    ) -> str:
        parts: list[str] = []
        used = 2
        for index in range(len(value)):
            remaining = available - used - bool(parts)
            if self.nodes <= 0 or remaining < 2:
                self.truncated = True
                break

            encoded = self.value(value[index], remaining, depth + 1, charged=charged)
            if encoded is None:
                break

            used += len(encoded) + bool(parts)
            parts.append(encoded)

        return "[" + ",".join(parts) + "]"

    def content(
        self,
        value: Sequence[object],
        available: int,
        max_blocks: int,
        *,
        skip_text: bool,
    ) -> tuple[str, int]:
        parts: list[str] = []
        used = 2
        kept = 0
        for index in range(min(len(value), max_blocks)):
            block = value[index]

            # A skipped text block is a repeated copy, not an omitted block.
            if skip_text and _block_type(block) == "text":
                kept += 1
                continue

            remaining = available - used - bool(parts)
            if self.nodes <= 0 or remaining < 2:
                self.truncated = True
                break

            encoded = self.value(block, remaining, block=True)
            if encoded is None:
                break

            used += len(encoded) + bool(parts)
            parts.append(encoded)
            kept += 1

        return "[" + ",".join(parts) + "]", kept
