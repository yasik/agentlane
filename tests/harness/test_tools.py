import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from agentlane.harness import RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    GitignoreMatcher,
    HarnessToolDefinition,
    HarnessToolsShim,
    ToolPathResolver,
    base_harness_tools,
    truncate_output,
)
from agentlane.models import Tool, Tools
from agentlane.runtime import CancellationToken


class _EchoArgs(BaseModel):
    text: str


async def _echo(args: _EchoArgs, cancellation_token: CancellationToken) -> str:
    del cancellation_token
    return args.text


def _tool(name: str) -> Tool[_EchoArgs, str]:
    return Tool(
        name=name,
        description=f"{name} tool.",
        args_model=_EchoArgs,
        handler=_echo,
    )


def _run_state(*, turn_count: int = 1) -> RunState:
    return RunState(
        instructions="Base",
        history=[],
        responses=[],
        turn_count=turn_count,
    )


def test_base_harness_tools_starts_empty_until_tool_slices_land() -> None:
    assert base_harness_tools() == ()


def test_harness_tools_shim_merges_tools_and_appends_prompt_once() -> None:
    async def scenario() -> None:
        existing = _tool("existing")
        definition = HarnessToolDefinition(
            tool=_tool("read"),
            prompt_snippet="Read file contents",
            prompt_guidelines=[
                "Use read to examine files instead of cat or sed.",
                "Use read to examine files instead of cat or sed.",
            ],
        )
        assert definition.prompt_guidelines == (
            "Use read to examine files instead of cat or sed.",
            "Use read to examine files instead of cat or sed.",
        )
        shim = HarnessToolsShim((definition,))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = _run_state()

        first_turn = PreparedTurn(
            run_state=state,
            tools=Tools(tools=[existing]),
            model_args=None,
        )
        await bound.prepare_turn(first_turn)

        assert first_turn.tools is not None
        assert [tool.name for tool in first_turn.tools.normalized_tools] == [
            "existing",
            "read",
        ]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- read: Read file contents\n\n"
            "Guidelines:\n"
            "- Use read to examine files instead of cat or sed.\n"
            "</default_tools>"
        )
        instructions_after_first = state.instructions

        second_turn = PreparedTurn(
            run_state=state,
            tools=Tools(tools=[existing]),
            model_args=None,
        )
        await bound.prepare_turn(second_turn)

        assert state.instructions == instructions_after_first

    asyncio.run(scenario())


def test_harness_tools_shim_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="Duplicate harness tool name: read"):
        HarnessToolsShim(
            (
                HarnessToolDefinition(tool=_tool("read")),
                HarnessToolDefinition(tool=_tool("read")),
            )
        )


def test_tool_path_resolver_captures_and_normalizes_cwd(tmp_path: Path) -> None:
    resolver = ToolPathResolver(cwd=tmp_path)

    assert resolver.resolve("notes/today.md") == tmp_path / "notes" / "today.md"
    assert resolver.resolve(tmp_path / "absolute.txt") == tmp_path / "absolute.txt"


def test_tool_path_resolver_rejects_empty_string(tmp_path: Path) -> None:
    resolver = ToolPathResolver(cwd=tmp_path)

    with pytest.raises(ValueError, match="path must not be empty"):
        resolver.resolve("")


def test_truncate_output_limits_head_by_line_count() -> None:
    output = truncate_output("a\nb\nc\n", max_lines=2, max_bytes=100)

    assert output.truncated is True
    assert output.text == (
        "[output truncated: showing first 2 lines or 100 bytes]\n" "a\nb\n"
    )


def test_truncate_output_limits_tail_by_byte_count() -> None:
    output = truncate_output(
        "alpha\nbravo\ncharlie\n", max_lines=10, max_bytes=8, tail=True
    )

    assert output.truncated is True
    assert output.text == (
        "[output truncated: showing last 10 lines or 8 bytes]\n" "charlie\n"
    )


def test_gitignore_matcher_respects_root_rules_and_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(
        "ignored.txt\nbuild/\n*.log\n",
        encoding="utf-8",
    )
    (tmp_path / "build").mkdir()

    matcher = GitignoreMatcher.from_path(tmp_path)

    assert matcher.is_ignored("ignored.txt") is True
    assert matcher.is_ignored(tmp_path / "ignored.txt") is True
    assert matcher.is_ignored(tmp_path / "build", is_dir=True) is True
    assert matcher.is_ignored(tmp_path / "build" / "artifact.js") is True
    assert matcher.is_ignored(tmp_path / "debug.log") is True
    assert matcher.is_ignored(tmp_path / ".git" / "config") is True
    assert matcher.is_ignored(tmp_path / "visible.txt") is False
