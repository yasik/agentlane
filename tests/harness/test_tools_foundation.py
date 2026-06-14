import asyncio
from pathlib import Path
from typing import cast

import pytest

from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    BASE_TOOL_NAMES,
    GitignoreMatcher,
    HarnessToolDefinition,
    HarnessToolsShim,
    ToolPathResolver,
    ToolPermissionDecision,
    ToolPermissionRequest,
    WorkspaceToolPermissionPolicy,
    base_harness_tools,
    bash_tool,
    truncate_output,
)
from agentlane.models import Tools

from .tools_test_utils import echo_tool, run_state, run_tool


def test_base_harness_tools_includes_current_tool_set() -> None:
    definitions = base_harness_tools()

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "find",
        "grep",
        "patch",
        "write",
        "write_plan",
        "bash",
        "agent",
    ]


def test_base_harness_tools_selects_included_names_in_standard_order() -> None:
    definitions = base_harness_tools(include=("bash", "read", "grep"))

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "grep",
        "bash",
    ]


def test_base_harness_tools_excludes_selected_names() -> None:
    definitions = base_harness_tools(exclude=("bash", "agent"))

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "find",
        "grep",
        "patch",
        "write",
        "write_plan",
    ]


@pytest.mark.parametrize("selector", ["include", "exclude"])
def test_base_harness_tools_rejects_unknown_selector_name(selector: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Unknown base_harness_tools "
            f"{selector} selector\\(s\\): ls\\. Expected one of:"
        ),
    ):
        if selector == "include":
            base_harness_tools(include=("ls",))
        else:
            base_harness_tools(exclude=("ls",))


def test_base_harness_tools_rejects_overlapping_include_exclude() -> None:
    with pytest.raises(ValueError, match="include/exclude selectors overlap: read"):
        base_harness_tools(include=("read", "grep"), exclude=("read",))


def test_base_tool_names_constant_matches_built_set() -> None:
    assert BASE_TOOL_NAMES == (
        "read",
        "find",
        "grep",
        "patch",
        "write",
        "write_plan",
        "bash",
        "agent",
    )
    assert tuple(definition.tool.name for definition in base_harness_tools()) == (
        BASE_TOOL_NAMES
    )


def test_base_harness_tools_include_accepts_extra_names_without_building_them() -> None:
    definitions = base_harness_tools(
        include=("read", "web_search"),
        extra_names=("web_search",),
    )

    assert [definition.tool.name for definition in definitions] == ["read"]


def test_base_harness_tools_exclude_accepts_extra_names_without_raising() -> None:
    definitions = base_harness_tools(
        exclude=("bash", "web_search"),
        extra_names=("web_search",),
    )

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "find",
        "grep",
        "patch",
        "write",
        "write_plan",
        "agent",
    ]


def test_base_harness_tools_accepts_single_string_extra_name() -> None:
    definitions = base_harness_tools(
        include=("read", "web_search"),
        extra_names="web_search",
    )

    assert [definition.tool.name for definition in definitions] == ["read"]


def test_base_harness_tools_still_rejects_unknown_name_outside_extra_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"Unknown base_harness_tools include selector\(s\): mystery",
    ):
        base_harness_tools(include=("read", "mystery"), extra_names=("web_search",))


def test_base_harness_tools_overlap_check_ignores_extra_names() -> None:
    # An extra name appearing in both selectors is dropped before the overlap
    # check because it is not built here; only base-tool overlap should raise.
    definitions = base_harness_tools(
        include=("read", "web_search"),
        exclude=("web_search",),
        extra_names=("web_search",),
    )

    assert [definition.tool.name for definition in definitions] == ["read"]


def test_base_harness_tools_constructs_current_tool_set_with_common_options(
    tmp_path: Path,
) -> None:
    definitions = base_harness_tools(
        cwd=tmp_path,
        permissions=WorkspaceToolPermissionPolicy(tmp_path),
        approval_callback=_approval_callback,
    )

    assert [definition.tool.name for definition in definitions] == [
        "read",
        "find",
        "grep",
        "patch",
        "write",
        "write_plan",
        "bash",
        "agent",
    ]
    bash_definition = next(
        definition for definition in definitions if definition.tool.name == "bash"
    )
    assert bash_definition.tool.name == "bash"


def test_bash_tool_constructs_with_optional_cwd_regression(tmp_path: Path) -> None:
    definition = bash_tool(
        cwd=tmp_path,
        permissions=WorkspaceToolPermissionPolicy(tmp_path),
        approval_callback=_approval_callback,
    )

    assert definition.tool.name == "bash"


def test_base_harness_tools_threads_cwd_and_permissions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    definitions = base_harness_tools(
        cwd=workspace,
        permissions=WorkspaceToolPermissionPolicy(workspace),
    )
    read_definition = next(
        definition for definition in definitions if definition.tool.name == "read"
    )

    output = run_tool(read_definition, path=str(outside))

    assert output == f"permission denied: read is not allowed for `{outside}`"


def test_harness_tools_shim_merges_tools_and_appends_prompt_once() -> None:
    async def scenario() -> None:
        existing = echo_tool("existing")
        definition = HarnessToolDefinition(
            tool=echo_tool("read"),
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
        state = run_state()

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


def test_harness_tools_shim_renders_shim_level_prompt_guidelines() -> None:
    async def scenario() -> None:
        definition = HarnessToolDefinition(
            tool=echo_tool("read"),
            prompt_snippet="Read file contents",
            prompt_guidelines=(
                "Use workspace-relative paths.",
                "Use read to examine files instead of cat or sed.",
            ),
        )
        shim = HarnessToolsShim(
            (definition,),
            prompt_guidelines=(
                "Tool paths are relative to the workspace root.",
                "Use workspace-relative paths.",
            ),
        )
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert definition.prompt_guidelines == (
            "Use workspace-relative paths.",
            "Use read to examine files instead of cat or sed.",
        )
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- read: Read file contents\n\n"
            "Guidelines:\n"
            "- Tool paths are relative to the workspace root.\n"
            "- Use workspace-relative paths.\n"
            "- Use read to examine files instead of cat or sed.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())


def test_harness_tools_shim_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="Duplicate harness tool name: read"):
        HarnessToolsShim(
            (
                HarnessToolDefinition(tool=echo_tool("read")),
                HarnessToolDefinition(tool=echo_tool("read")),
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
        "[output truncated: showing first 2 lines or 100 bytes]\na\nb\n"
    )


def test_truncate_output_limits_tail_by_byte_count() -> None:
    output = truncate_output(
        "alpha\nbravo\ncharlie\n", max_lines=10, max_bytes=8, tail=True
    )

    assert output.truncated is True
    assert output.text == (
        "[output truncated: showing last 10 lines or 8 bytes]\ncharlie\n"
    )


def test_truncate_output_can_omit_marker() -> None:
    output = truncate_output(
        "a\nb\nc\n",
        max_lines=2,
        max_bytes=100,
        include_marker=False,
    )

    assert output.truncated is True
    assert output.text == "a\nb\n"


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


def _approval_callback(
    request: ToolPermissionRequest,
    decision: ToolPermissionDecision,
) -> ToolPermissionDecision:
    del request
    return decision
