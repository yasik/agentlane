import asyncio
from pathlib import Path
from typing import cast

import pytest
from tools_test_utils import (
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_state,
    run_tool,
    set_mtime,
    touch,
)

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    FIND_DEFAULT_LIMIT,
    TEXT_MAX_BYTES,
    HarnessToolsShim,
    ToolPathResolver,
    find_tool,
)
from agentlane.models import Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


def test_find_tool_matches_simple_glob(tmp_path: Path) -> None:
    touch(tmp_path / "alpha.py")
    touch(tmp_path / "nested" / "beta.py")
    touch(tmp_path / "notes.md")

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.py")

    assert output == f"Search directory: {tmp_path}\nalpha.py"


def test_find_tool_matches_recursive_glob(tmp_path: Path) -> None:
    touch(tmp_path / "alpha.json")
    touch(tmp_path / "nested" / "beta.json")
    touch(tmp_path / "nested" / "notes.md")
    set_mtime(tmp_path / "alpha.json", 2000)
    set_mtime(tmp_path / "nested" / "beta.json", 1000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="**/*.json")

    assert output == f"Search directory: {tmp_path}\nalpha.json\nnested/beta.json"


def test_find_tool_uses_explicit_search_path(tmp_path: Path) -> None:
    touch(tmp_path / "outside.py")
    touch(tmp_path / "src" / "inside.py")

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.py", path="src")

    assert output == f"Search directory: {tmp_path / 'src'}\ninside.py"


def test_find_tool_includes_dotfiles(tmp_path: Path) -> None:
    touch(tmp_path / ".env")
    touch(tmp_path / "visible.txt")
    set_mtime(tmp_path / ".env", 2000)
    set_mtime(tmp_path / "visible.txt", 1000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*")

    assert output == f"Search directory: {tmp_path}\n.env\nvisible.txt"


def test_find_tool_matches_case_insensitively(tmp_path: Path) -> None:
    touch(tmp_path / "Alpha.py")

    output = run_tool(find_tool(cwd=tmp_path), pattern="alpha.py")

    assert output == f"Search directory: {tmp_path}\nAlpha.py"


def test_find_tool_supports_brace_expansion(tmp_path: Path) -> None:
    touch(tmp_path / "alpha.py")
    touch(tmp_path / "beta.ts")
    touch(tmp_path / "gamma.md")
    set_mtime(tmp_path / "alpha.py", 1000)
    set_mtime(tmp_path / "beta.ts", 2000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.{py,ts}")

    assert output == f"Search directory: {tmp_path}\nbeta.ts\nalpha.py"


def test_find_tool_strips_leading_slash_from_pattern(tmp_path: Path) -> None:
    touch(tmp_path / "src" / "alpha.py")

    output = run_tool(find_tool(cwd=tmp_path), pattern="/src/*.py")

    assert output == f"Search directory: {tmp_path}\nsrc/alpha.py"


def test_find_tool_respects_gitignore_and_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(
        "ignored/\n*.tmp\n",
        encoding="utf-8",
    )
    touch(tmp_path / ".git" / "hidden.py")
    touch(tmp_path / "ignored" / "skip.py")
    touch(tmp_path / "visible" / "keep.py")
    touch(tmp_path / "visible" / "skip.tmp")
    set_mtime(tmp_path / ".gitignore", 2000)
    set_mtime(tmp_path / "visible" / "keep.py", 1000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="**/*")

    assert output == f"Search directory: {tmp_path}\n.gitignore\nvisible/keep.py"


def test_find_tool_sorts_by_mtime_newest_first(tmp_path: Path) -> None:
    touch(tmp_path / "old.txt")
    touch(tmp_path / "middle.txt")
    touch(tmp_path / "newest.txt")
    set_mtime(tmp_path / "old.txt", 1000)
    set_mtime(tmp_path / "middle.txt", 2000)
    set_mtime(tmp_path / "newest.txt", 3000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.txt")

    assert output == (f"Search directory: {tmp_path}\nnewest.txt\nmiddle.txt\nold.txt")


def test_find_tool_breaks_mtime_ties_alphabetically(tmp_path: Path) -> None:
    touch(tmp_path / "zeta.txt")
    touch(tmp_path / "alpha.txt")
    touch(tmp_path / "middle.txt")
    set_mtime(tmp_path / "zeta.txt", 2000)
    set_mtime(tmp_path / "alpha.txt", 2000)
    set_mtime(tmp_path / "middle.txt", 1000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.txt")

    assert output == (f"Search directory: {tmp_path}\nalpha.txt\nzeta.txt\nmiddle.txt")


def test_find_tool_marks_result_limit_truncation(tmp_path: Path) -> None:
    touch(tmp_path / "a.py")
    touch(tmp_path / "b.py")
    touch(tmp_path / "c.py")
    set_mtime(tmp_path / "a.py", 3000)
    set_mtime(tmp_path / "b.py", 2000)
    set_mtime(tmp_path / "c.py", 1000)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.py", limit=2)

    assert output == (
        f"Search directory: {tmp_path}\n"
        "a.py\n"
        "b.py\n"
        "3 files matched; returned first 2. "
        f"Refine the pattern or raise `limit` (max {FIND_DEFAULT_LIMIT})."
    )


def test_find_tool_marks_limit_capped_at_maximum(tmp_path: Path) -> None:
    for index in range(FIND_DEFAULT_LIMIT + 1):
        path = tmp_path / f"file_{index:04d}.py"
        touch(path)
        set_mtime(path, 1000)

    output = run_tool(
        find_tool(cwd=tmp_path),
        pattern="*.py",
        limit=FIND_DEFAULT_LIMIT + 10,
    )

    lines = output.splitlines()
    assert lines[0] == f"Search directory: {tmp_path}"
    assert lines[1] == "file_0000.py"
    assert lines[FIND_DEFAULT_LIMIT] == f"file_{FIND_DEFAULT_LIMIT - 1:04d}.py"
    assert lines[-1] == (
        f"{FIND_DEFAULT_LIMIT + 1} files matched; "
        f"returned first {FIND_DEFAULT_LIMIT} (maximum). "
        "Refine the pattern or narrow `path`."
    )


def test_find_tool_marks_byte_limit_truncation(tmp_path: Path) -> None:
    for index in range(800):
        touch(tmp_path / f"{index:04d}-{'x' * 72}.txt")

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.txt")
    lines = output.splitlines()

    assert lines[0] == f"Search directory: {tmp_path}"
    assert lines[-1] == (
        f"Output truncated at {TEXT_MAX_BYTES} bytes; "
        "refine the pattern or narrow `path`."
    )
    assert len("\n".join(lines[1:-1]).encode("utf-8")) <= TEXT_MAX_BYTES


def test_find_tool_reports_no_matches(tmp_path: Path) -> None:
    touch(tmp_path / "notes.md")

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.py")

    assert output == f"Search directory: {tmp_path}\nNo files matched."


def test_find_tool_reports_non_directory_path(tmp_path: Path) -> None:
    touch(tmp_path / "notes.md")

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.md", path="notes.md")

    assert output == f"path is not a directory: `{tmp_path / 'notes.md'}`"


def test_find_tool_rejects_invalid_pattern_path_and_limit(tmp_path: Path) -> None:
    definition = find_tool(cwd=tmp_path)

    assert run_tool(definition, pattern=" ") == "pattern must not be empty"
    assert run_tool(definition, pattern="*.py", path="") == "path must not be empty"
    assert run_tool(definition, pattern="*.py", limit=0) == (
        "limit must be greater than zero"
    )


def test_find_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(self: ToolPathResolver, path: str | Path) -> Path:
        del self
        del path
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(ToolPathResolver, "resolve", raise_unexpected_error)

    output = run_tool(find_tool(cwd=tmp_path), pattern="*.py", path=".")

    assert output == "failed to find files"


def test_find_tool_runs_through_runner_tool_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        touch(tmp_path / "alpha.py")
        tool_call = make_tool_call(
            tool_id="call_1",
            name="find",
            arguments='{"pattern":"*.py"}',
        )
        model = SequenceModel(
            [
                make_assistant_response(None, tool_calls=[tool_call]),
                make_assistant_response("complete"),
            ]
        )
        runner = Runner()
        runtime = SingleThreadedRuntimeEngine()
        agent = Agent(
            runtime,
            runner,
            descriptor=AgentDescriptor(
                name="Finder",
                model=model,
                tools=Tools(
                    tools=[find_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"find": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Find requested files before answering.",
            history=["Find Python files."],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "complete"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["find"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "find"
        assert "alpha.py" in cast(str, tool_message["content"])

    asyncio.run(scenario())


def test_find_tool_prompt_snippet_through_harness_tools_shim(tmp_path: Path) -> None:
    async def scenario() -> None:
        shim = HarnessToolsShim((find_tool(cwd=tmp_path),))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()

        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )
        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["find"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- find: Find files by glob pattern (use `**/` for recursion)\n\n"
            "Guidelines:\n"
            "- Use find to locate files instead of shelling out to find or ls.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())


def test_find_tool_limit_schema_documents_default_result_limit() -> None:
    schema = find_tool().tool.schema

    assert schema["parameters"]["properties"]["limit"]["default"] == FIND_DEFAULT_LIMIT
