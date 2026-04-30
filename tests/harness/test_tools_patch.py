import asyncio
from pathlib import Path
from typing import cast

import patch_tool as patch_engine
import pytest
from tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_state,
    run_tool,
)

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import HarnessToolsShim, ToolPathResolver, patch_tool
from agentlane.models import Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


def test_patch_tool_applies_search_replace_block(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbravo\ncharlie\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits=("<<<<<<< SEARCH\n" "bravo\n" "=======\n" "beta\n" ">>>>>>> REPLACE\n"),
    )

    assert output == f"Applied 1 edit to {target}."
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\ncharlie\n"


def test_patch_tool_applies_multiple_blocks_atomically(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits=(
            "<<<<<<< SEARCH\n"
            "one\n"
            "=======\n"
            "ONE\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "three\n"
            "=======\n"
            "THREE\n"
            ">>>>>>> REPLACE\n"
        ),
    )

    assert output == f"Applied 2 edits to {target}."
    assert target.read_text(encoding="utf-8") == "ONE\ntwo\nTHREE\nfour\n"


def test_patch_tool_applies_fuzzy_matches_without_model_facing_detail(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("quote = \u201cold\u201d\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits='<<<<<<< SEARCH\nquote = "old"\n=======\nquote = "new"\n>>>>>>> REPLACE\n',
    )

    assert output == f"Applied 1 edit to {target}."
    assert target.read_text(encoding="utf-8") == 'quote = "new"\n'


def test_patch_tool_leaves_file_unchanged_when_any_edit_fails(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    original = "one\ntwo\n"
    target.write_text(original, encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits=(
            "<<<<<<< SEARCH\n"
            "one\n"
            "=======\n"
            "ONE\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "missing\n"
            "=======\n"
            "replacement\n"
            ">>>>>>> REPLACE\n"
        ),
    )

    assert output == (
        f"Could not find edit 2 in {target}. The SEARCH text must match "
        "exactly including all whitespace and newlines."
    )
    assert target.read_text(encoding="utf-8") == original


def test_patch_tool_resolves_relative_paths_from_configured_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "notes.txt"
    target.parent.mkdir()
    target.write_text("relative\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=workspace),
        path="notes.txt",
        edits="<<<<<<< SEARCH\nrelative\n=======\npatched\n>>>>>>> REPLACE\n",
    )

    assert output == f"Applied 1 edit to {target}."
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_patch_tool_accepts_absolute_paths(tmp_path: Path) -> None:
    target = tmp_path / "absolute.txt"
    target.write_text("absolute\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path / "elsewhere"),
        path=str(target),
        edits="<<<<<<< SEARCH\nabsolute\n=======\npatched\n>>>>>>> REPLACE\n",
    )

    assert output == f"Applied 1 edit to {target}."
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_patch_tool_reports_parser_errors_and_empty_patch(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n", encoding="utf-8")
    definition = patch_tool(cwd=tmp_path)

    assert run_tool(definition, path="notes.txt", edits="no blocks") == (
        "Patch tool input is invalid. edits must contain at least one replacement."
    )
    assert run_tool(definition, path="notes.txt", edits="<<<<<<< SEARCH\nalpha\n") == (
        "Invalid patch format: Unterminated SEARCH block at line 1: missing '======='"
    )


def test_patch_tool_reports_semantic_edit_errors(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("repeat\nrepeat\n", encoding="utf-8")
    definition = patch_tool(cwd=tmp_path)

    assert run_tool(
        definition,
        path="notes.txt",
        edits="<<<<<<< SEARCH\nmissing\n=======\nreplacement\n>>>>>>> REPLACE\n",
    ) == (
        f"Could not find the SEARCH text in {target}. The SEARCH text must "
        "match exactly including all whitespace and newlines."
    )
    assert run_tool(
        definition,
        path="notes.txt",
        edits="<<<<<<< SEARCH\nrepeat\n=======\nreplacement\n>>>>>>> REPLACE\n",
    ) == (
        f"Found 2 occurrences of the SEARCH text in {target}. The SEARCH text "
        "must be unique. Please provide more context to make it unique."
    )
    assert (
        run_tool(
            definition,
            path="notes.txt",
            edits="<<<<<<< SEARCH\n=======\nreplacement\n>>>>>>> REPLACE\n",
        )
        == f"SEARCH text must not be empty in {target}."
    )
    assert (
        run_tool(
            definition,
            path="notes.txt",
            edits="<<<<<<< SEARCH\nrepeat\nrepeat\n=======\nrepeat\nrepeat\n>>>>>>> REPLACE\n",
        )
        == f"No changes made to {target}. The replacement produced identical content."
    )


def test_patch_tool_reports_overlapping_edits(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("abcdef\n", encoding="utf-8")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits=(
            "<<<<<<< SEARCH\n"
            "bcd\n"
            "=======\n"
            "BCD\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "cde\n"
            "=======\n"
            "CDE\n"
            ">>>>>>> REPLACE\n"
        ),
    )

    assert output == (
        f"edit 1 and edit 2 overlap in {target}. Merge them into one edit or "
        "target disjoint regions."
    )
    assert target.read_text(encoding="utf-8") == "abcdef\n"


def test_patch_tool_rejects_invalid_path_and_content(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("content\n", encoding="utf-8")
    definition = patch_tool(cwd=tmp_path)

    assert run_tool(definition, path="", edits="content") == "path must not be empty"
    assert run_tool(definition, path="bad\x00name.txt", edits="content") == (
        "path contains a null byte"
    )
    assert run_tool(definition, path="notes.txt", edits="\ud800") == (
        "patch is not valid UTF-8"
    )


def test_patch_tool_reports_missing_file_and_directory(tmp_path: Path) -> None:
    assert run_tool(patch_tool(cwd=tmp_path), path="missing.txt", edits="content") == (
        f"File not found: {tmp_path / 'missing.txt'}"
    )
    assert run_tool(patch_tool(cwd=tmp_path), path=".", edits="content") == (
        f"Path is a directory: {tmp_path}"
    )


def test_patch_tool_reports_invalid_utf8_file(tmp_path: Path) -> None:
    (tmp_path / "latin1.txt").write_bytes(b"\xff")

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="latin1.txt",
        edits="<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n",
    )

    assert output == f"File is not valid UTF-8: {tmp_path / 'latin1.txt'}"


def test_patch_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n", encoding="utf-8")

    def raise_unexpected_error(
        path: str | Path,
        edits: object,
    ) -> patch_engine.EditResult:
        del path
        del edits
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(patch_engine, "apply_edits", raise_unexpected_error)

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits="<<<<<<< SEARCH\nalpha\n=======\nbeta\n>>>>>>> REPLACE\n",
    )

    assert output == "failed to patch file"


def test_patch_tool_sanitizes_path_resolver_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(self: ToolPathResolver, path: str | Path) -> Path:
        del self
        del path
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(ToolPathResolver, "resolve", raise_unexpected_error)

    output = run_tool(
        patch_tool(cwd=tmp_path),
        path="notes.txt",
        edits="<<<<<<< SEARCH\nalpha\n=======\nbeta\n>>>>>>> REPLACE\n",
    )

    assert output == "failed to patch file"


def test_patch_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        target = tmp_path / "runner.txt"
        target.write_text("before\n", encoding="utf-8")
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="patch",
                            arguments=(
                                '{"path":"runner.txt","edits":"<<<<<<< SEARCH\\n'
                                'before\\n=======\\nafter\\n>>>>>>> REPLACE\\n"}'
                            ),
                        )
                    ],
                ),
                make_assistant_response(content="done"),
            ]
        )
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="PatchRunner",
                model=model,
                tools=Tools(
                    tools=[patch_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"patch": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Patch the requested file before answering.",
            history=["edit runner.txt"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        assert target.read_text(encoding="utf-8") == "after\n"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["patch"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "patch"
        assert tool_message["content"] == f"Applied 1 edit to {target}."

    asyncio.run(scenario())


def test_patch_tool_prompt_metadata_renders_through_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((patch_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()

        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["patch"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- patch: Apply search/replace edits to existing files\n"
            "\n"
            "Guidelines:\n"
            "- Use patch for precise edits to existing files after reading "
            "them; use write for new files or complete rewrites.\n"
            "- Each patch SEARCH block must match exactly one location; "
            "include enough surrounding lines to make it unique.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())
