import asyncio
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from ripgrepy import RipGrepNotFound, Ripgrepy
from tools_test_utils import (
    FakeRipgrepResult,
    SequenceModel,
    make_assistant_response,
    make_tool_call,
    run_state,
    run_tool,
)

from agentlane.harness import Agent, AgentDescriptor, Runner, RunState
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import (
    TEXT_MAX_BYTES,
    HarnessToolsShim,
    ToolPathResolver,
    grep_tool,
)
from agentlane.models import Tools
from agentlane.runtime import SingleThreadedRuntimeEngine


def test_grep_tool_searches_directory_with_regex(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def main() -> None:\n    return None\nmain()\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("nothing here\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern=r"main\(", path=".")

    assert result == (
        "Search path: .\n" "app.py:1:def main() -> None:\n" "app.py:3:main()"
    )


def test_grep_tool_searches_explicit_file_path(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path="second.txt")

    assert result == "Search path: second.txt\nsecond.txt:1:needle"


def test_grep_tool_supports_literal_and_ignore_case(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(
        "axb\nA.B\n",
        encoding="utf-8",
    )

    literal_result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="a.b",
        path="notes.txt",
        literal=True,
        ignoreCase=True,
    )
    regex_result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="a.b",
        path="notes.txt",
        ignoreCase=True,
    )

    assert literal_result == "Search path: notes.txt\nnotes.txt:2:A.B"
    assert regex_result == (
        "Search path: notes.txt\n" "notes.txt:1:axb\n" "notes.txt:2:A.B"
    )


def test_grep_tool_filters_candidates_by_glob(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".", glob="*.py")

    assert result == "Search path: .\napp.py:1:needle"


def test_grep_tool_filters_explicit_file_path_by_glob(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="notes.txt",
        glob="*.py",
    )

    assert result == "Search path: notes.txt\nNo matches."


def test_grep_tool_includes_context_and_merges_overlapping_groups(
    tmp_path: Path,
) -> None:
    (tmp_path / "story.txt").write_text(
        "\n".join(
            [
                "zero",
                "one needle",
                "two",
                "three needle",
                "four",
                "five",
                "six",
                "seven",
                "eight needle",
                "nine",
            ]
        ),
        encoding="utf-8",
    )

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="story.txt",
        context=1,
    )

    assert result == (
        "Search path: story.txt\n"
        "story.txt:1:zero\n"
        "story.txt:2:one needle\n"
        "story.txt:3:two\n"
        "story.txt:4:three needle\n"
        "story.txt:5:four\n"
        "--\n"
        "story.txt:8:seven\n"
        "story.txt:9:eight needle\n"
        "story.txt:10:nine"
    )


def test_grep_tool_respects_gitignore_and_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\nvisible.txt:1:needle"


def test_grep_tool_respects_gitignore_for_explicit_file_path(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path="ignored.txt")

    assert result == "Search path: ignored.txt\nNo matches."


def test_grep_tool_searches_hidden_files_but_skips_git_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == ("Search path: .\n.env:1:needle\nvisible.txt:1:needle")


def test_grep_tool_truncates_long_lines_and_match_limit(tmp_path: Path) -> None:
    long_line = "needle " + ("x" * 600)
    (tmp_path / "long.txt").write_text(
        f"{long_line}\nneedle second\n",
        encoding="utf-8",
    )

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="long.txt",
        limit=1,
    )
    lines = result.splitlines()

    assert lines[0] == "Search path: long.txt"
    assert lines[1].startswith("long.txt:1:needle ")
    assert "needle second" not in result
    assert len(lines[1].split(":", maxsplit=2)[2]) == 500
    assert lines[-2] == "Showing first 1 matches; more remain."
    assert lines[-1] == (
        "One or more matching lines were truncated after 500 characters"
    )


def test_grep_tool_truncates_by_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "\n".join(f"needle {index:04d} " + ("x" * 90) for index in range(700)),
        encoding="utf-8",
    )

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path="large.txt",
        limit=1000,
    )

    assert result.endswith(f"\nOutput truncated after {TEXT_MAX_BYTES} bytes")
    assert len(result.encode("utf-8")) < 53 * 1024


def test_grep_tool_reports_no_matches_and_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("plain text\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\nNo matches."


def test_grep_tool_continues_when_directory_contains_binary_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")
    (tmp_path / "text.txt").write_text("needle in text\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\ntext.txt:1:needle in text"


def test_grep_tool_ignores_binary_warnings_alongside_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_path = tmp_path / "text.txt"
    text_path.write_text("needle\n", encoding="utf-8")
    output = "\n".join(
        (
            'rg: ./binary.bin: binary file matches (found "\\0" byte around offset 0)',
            (
                f'{{"type":"match","data":{{"path":{{"text":"{text_path.as_posix()}"}},'
                '"lines":{"text":"needle\\n"},"line_number":1,'
                '"absolute_offset":0,"submatches":[]}}'
            ),
        )
    )

    def fake_run(self: Ripgrepy) -> FakeRipgrepResult:
        del self
        return FakeRipgrepResult(output)

    monkeypatch.setattr(Ripgrepy, "run", fake_run)

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "Search path: .\ntext.txt:1:needle"


def test_grep_tool_ignores_per_file_warnings_alongside_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible_path = tmp_path / "visible.py"
    visible_path.write_text("hit\n", encoding="utf-8")
    output = "\n".join(
        (
            "rg: ./locked.txt: Permission denied (os error 13)",
            (
                f'{{"type":"match","data":{{"path":{{"text":"{visible_path.as_posix()}"}},'
                '"lines":{"text":"hit\\n"},"line_number":1,'
                '"absolute_offset":0,"submatches":[]}}'
            ),
        )
    )

    def fake_run(self: Ripgrepy) -> FakeRipgrepResult:
        del self
        return FakeRipgrepResult(output)

    monkeypatch.setattr(Ripgrepy, "run", fake_run)

    result = run_tool(grep_tool(cwd=tmp_path), pattern="hit", path=".")

    assert result == "Search path: .\nvisible.py:1:hit"


def test_grep_tool_listing_mode_filters_warning_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "alpha.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "beta.py").write_text("needle\n", encoding="utf-8")

    fake_output = "\n".join(
        [
            f"{tmp_path / 'alpha.py'}",
            "rg: ./locked.txt: Permission denied (os error 13)",
            f"{tmp_path / 'beta.py'}",
        ]
    )

    def fake_run(self: Ripgrepy) -> FakeRipgrepResult:
        del self
        return FakeRipgrepResult(fake_output)

    monkeypatch.setattr(Ripgrepy, "run", fake_run)
    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
    )

    assert result == "Search path: .\nalpha.py\nbeta.py"


def test_grep_tool_reports_binary_file_for_explicit_file_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\x00needle\x00")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path="binary.bin")

    assert result == (
        "file appears to be binary and cannot be searched as text: "
        f"`{tmp_path / 'binary.bin'}`"
    )


def test_grep_tool_reports_invalid_regex(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="(", path=".")

    assert result == "invalid regex pattern"


def test_grep_tool_reports_invalid_glob(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".", glob="[")

    assert result == "invalid glob pattern"


@pytest.mark.parametrize(
    "output_mode",
    ["content", "files_with_matches", "count"],
)
def test_grep_tool_reports_invalid_file_type(
    tmp_path: Path,
    output_mode: str,
) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode=output_mode,
        type="notatype",
    )

    assert result == "invalid file type"


def test_grep_tool_reports_missing_ripgrep_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_not_found(self: Ripgrepy, pattern: str, path: str) -> None:
        del self
        del pattern
        del path
        raise RipGrepNotFound("private details")

    monkeypatch.setattr(Ripgrepy, "__init__", raise_not_found)

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "ripgrep executable not found"


def test_grep_tool_rejects_negative_context() -> None:
    args_model = grep_tool().tool.args_type()

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        args_model(pattern="needle", context=-1)


def test_grep_tool_sanitizes_unexpected_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_unexpected_error(self: ToolPathResolver, path: str | Path) -> Path:
        del self
        del path
        raise RuntimeError("Traceback (most recent call last): private details")

    monkeypatch.setattr(ToolPathResolver, "resolve", raise_unexpected_error)

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path=".")

    assert result == "failed to search files"


def test_grep_tool_sanitizes_os_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")

    def raise_os_error(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self
        del args
        del kwargs
        raise OSError("Traceback (most recent call last): private details")

    monkeypatch.setattr(Path, "open", raise_os_error)

    result = run_tool(grep_tool(cwd=tmp_path), pattern="needle", path="notes.txt")

    assert result == f"failed to read file: `{tmp_path / 'notes.txt'}`"


def test_grep_tool_files_with_matches_mode(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("needle in haystack\n", encoding="utf-8")
    (tmp_path / "bravo.py").write_text("another needle\n", encoding="utf-8")
    (tmp_path / "charlie.py").write_text("no match here\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
    )

    assert result == "Search path: .\nalpha.py\nbravo.py"


def test_grep_tool_files_with_matches_mode_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("plain text\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
    )

    assert result == "Search path: .\nNo matches."


def test_grep_tool_count_mode(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text(
        "needle one\nneedle two\nneedle three\n",
        encoding="utf-8",
    )
    (tmp_path / "bravo.py").write_text("needle solo\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="count",
    )

    assert result == "Search path: .\nalpha.py:3\nbravo.py:1"


def test_grep_tool_filters_by_type(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle\n", encoding="utf-8")
    (tmp_path / "script.js").write_text("needle\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        type="py",
    )

    assert result == "Search path: .\napp.py:1:needle"


def test_grep_tool_multiline_pattern_matches_across_newlines(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text(
        "def foo():\n    return 42\n\ndef bar():\n    return 99\n",
        encoding="utf-8",
    )

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern=r"def foo\(\):.+?return",
        path="code.py",
        multiline=True,
    )

    assert "code.py:1:def foo():" in result


def test_grep_tool_caps_files_mode_with_limit(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"file_{index}.txt").write_text("needle\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="files_with_matches",
        limit=2,
    )

    lines = result.splitlines()
    assert lines[0] == "Search path: ."
    assert lines[1] == "file_0.txt"
    assert lines[2] == "file_1.txt"
    assert lines[-1] == "Showing first 2 files; more remain."


def test_grep_tool_count_mode_renders_relative_paths(tmp_path: Path) -> None:
    nested = tmp_path / "src"
    nested.mkdir()
    (nested / "deep.py").write_text("needle one\nneedle two\n", encoding="utf-8")

    result = run_tool(
        grep_tool(cwd=tmp_path),
        pattern="needle",
        path=".",
        outputMode="count",
    )

    assert result == "Search path: .\nsrc/deep.py:2"


def test_grep_tool_executes_through_runner_tool_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
        runtime = SingleThreadedRuntimeEngine()
        runner = Runner()
        model = SequenceModel(
            [
                make_assistant_response(
                    content=None,
                    tool_calls=[
                        make_tool_call(
                            tool_id="call_1",
                            name="grep",
                            arguments='{"pattern":"needle","path":"."}',
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
                name="GrepRunner",
                model=model,
                tools=Tools(
                    tools=[grep_tool(cwd=tmp_path).tool],
                    tool_choice="required",
                    tool_call_limits={"grep": 1},
                ),
            ),
        )
        state = RunState(
            instructions="Search the requested files before answering.",
            history=["inspect TODO mentions"],
            responses=[],
        )

        result = await runner.run(agent, state)

        assert result.final_output == "done"
        first_turn_tools = model.call_tools[0]
        assert first_turn_tools is not None
        assert [tool.name for tool in first_turn_tools.normalized_tools] == ["grep"]
        tool_message = cast(dict[str, object], state.history[2])
        assert tool_message["role"] == "tool"
        assert tool_message["name"] == "grep"
        assert "notes.txt:1:needle" in cast(str, tool_message["content"])

    asyncio.run(scenario())


def test_grep_tool_prompt_metadata_renders_through_harness_tools_shim(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        definition = grep_tool(cwd=tmp_path)
        shim = HarnessToolsShim((definition,))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()

        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["grep"]
        assert state.instructions == (
            "Base\n\n"
            "<default_tools>\n"
            "Available tools:\n"
            "- grep: Search file contents for patterns\n"
            "\n"
            "Guidelines:\n"
            '- Use grep with output_mode="files_with_matches" or "count" to '
            "discover where a symbol appears; switch to content mode when you "
            "need surrounding lines.\n"
            "</default_tools>"
        )

    asyncio.run(scenario())
