import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import RunState, Task
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.skills import (
    SKILL_PATH_PROMPT_GUIDANCE,
    FilesystemSkillLoader,
    SkillCatalog,
    SkillManifest,
    SkillRelativePathShim,
    SkillsShim,
    discover_skill_catalog,
    resolve_skill_relative_path,
)
from agentlane.harness.tools import HarnessToolDefinition, base_harness_tools
from agentlane.models import Tool, ToolExecutionContext, Tools
from agentlane.models.run import DefaultRunContext
from agentlane.runtime import CancellationToken


def _write_skill(root: Path, name: str, body_files: dict[str, str]) -> Path:
    skill_root = root / name
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                f"description: Skill {name}.",
                "---",
                "",
                f"# {name}",
                "",
            )
        ),
        encoding="utf-8",
    )
    for relative, content in body_files.items():
        target = skill_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_root


def test_resolve_skill_relative_path_returns_absolute_for_existing_resource(
    tmp_path: Path,
) -> None:
    skill_root = _write_skill(tmp_path, "report", {"references/guide.md": "guide body"})

    resolved = resolve_skill_relative_path(
        "references/guide.md",
        skill_roots=[skill_root],
    )

    assert resolved == str((skill_root / "references" / "guide.md").resolve())


def test_resolve_skill_relative_path_passes_through_when_resource_missing(
    tmp_path: Path,
) -> None:
    skill_root = _write_skill(tmp_path, "report", {})

    assert (
        resolve_skill_relative_path("references/missing.md", skill_roots=[skill_root])
        == "references/missing.md"
    )


def test_resolve_skill_relative_path_ignores_absolute_input(tmp_path: Path) -> None:
    skill_root = _write_skill(tmp_path, "report", {"a.md": "x"})

    absolute_input = str(tmp_path / "report" / "a.md")
    assert (
        resolve_skill_relative_path(absolute_input, skill_roots=[skill_root])
        == absolute_input
    )


@pytest.mark.parametrize(
    "anchored",
    ["./a.md", "../a.md", "~/a.md", "~user/a.md"],
)
def test_resolve_skill_relative_path_ignores_anchored_paths(
    anchored: str,
    tmp_path: Path,
) -> None:
    skill_root = _write_skill(tmp_path, "report", {"a.md": "x"})

    assert resolve_skill_relative_path(anchored, skill_roots=[skill_root]) == anchored


@pytest.mark.parametrize(
    "globby",
    ["references/*.md", "refs/file?.md", "data/[abc].md", "files/{a,b}.md"],
)
def test_resolve_skill_relative_path_ignores_glob_and_brace_patterns(
    globby: str,
    tmp_path: Path,
) -> None:
    skill_root = _write_skill(tmp_path, "report", {})

    assert resolve_skill_relative_path(globby, skill_roots=[skill_root]) == globby


def test_resolve_skill_relative_path_ignores_empty_value(tmp_path: Path) -> None:
    skill_root = _write_skill(tmp_path, "report", {})

    assert resolve_skill_relative_path("   ", skill_roots=[skill_root]) == "   "


def test_resolve_skill_relative_path_refuses_to_escape_skill_root(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    skill_root = _write_skill(tmp_path, "report", {})

    # A path whose `..` collapse climbs above the skill root must not resolve to
    # the escaping target even though the target exists on disk.
    escaping = "nested/../../secret.md"
    assert resolve_skill_relative_path(escaping, skill_roots=[skill_root]) == escaping


def test_resolve_skill_relative_path_prefers_last_active_skill(tmp_path: Path) -> None:
    first = _write_skill(tmp_path / "a", "first", {"shared.md": "first"})
    second = _write_skill(tmp_path / "b", "second", {"shared.md": "second"})

    resolved = resolve_skill_relative_path(
        "shared.md",
        skill_roots=[first, second],
    )

    assert resolved == str((second / "shared.md").resolve())


def test_resolve_skill_relative_path_returns_input_with_no_active_roots() -> None:
    assert resolve_skill_relative_path("a.md", skill_roots=[]) == "a.md"


def _binding_context() -> ShimBindingContext:
    return ShimBindingContext(task=cast(Task, SimpleNamespace(task_id="run-skill")))


def test_skill_relative_path_shim_resolves_read_after_activation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_root = tmp_path / "skills"
    skill_root = _write_skill(
        skills_root, "report-generator", {"references/guide.md": "guide body marker"}
    )
    assert skill_root.exists()

    async def scenario() -> None:
        loader = FilesystemSkillLoader(
            roots=(skills_root,),
            include_default_roots=False,
        )
        catalog = await discover_skill_catalog(loader)

        skills_shim = SkillsShim(catalog=catalog)
        resolver_shim = SkillRelativePathShim(
            base_harness_tools(cwd=workspace, include=("read",)),
            catalog=catalog,
        )

        bound_skills = await skills_shim.bind(_binding_context())
        bound_resolver = await resolver_shim.bind(_binding_context())

        run_state = RunState(instructions=None, history=[], responses=[])
        await bound_skills.on_run_start(run_state, DefaultRunContext())
        await bound_resolver.on_run_start(run_state, DefaultRunContext())
        turn = PreparedTurn(run_state=run_state, tools=None, model_args=None)
        await bound_skills.prepare_turn(turn)
        await bound_resolver.prepare_turn(turn)

        assert turn.tools is not None
        tools = {tool.name: tool for tool in turn.tools.executable_tools}

        # Before activation a skill-relative read does not resolve.
        unresolved = await tools["read"].run(
            tools["read"].args_type()(path="references/guide.md"),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "guide body marker" not in unresolved

        activation = await tools["activate_skill"].run(
            tools["activate_skill"].args_type()(name="report-generator"),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "report-generator" in activation

        resolved = await tools["read"].run(
            tools["read"].args_type()(path="references/guide.md"),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "guide body marker" in resolved

    asyncio.run(scenario())


def test_skill_relative_path_shim_leaves_absolute_workspace_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("workspace note", encoding="utf-8")
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "report-generator", {"notes.txt": "skill note"})

    async def scenario() -> None:
        loader = FilesystemSkillLoader(
            roots=(skills_root,),
            include_default_roots=False,
        )
        catalog = await discover_skill_catalog(loader)
        skills_shim = SkillsShim(catalog=catalog)
        resolver_shim = SkillRelativePathShim(
            base_harness_tools(cwd=workspace, include=("read",)),
            catalog=catalog,
        )
        bound_skills = await skills_shim.bind(_binding_context())
        bound_resolver = await resolver_shim.bind(_binding_context())

        run_state = RunState(instructions=None, history=[], responses=[])
        await bound_skills.on_run_start(run_state, DefaultRunContext())
        await bound_resolver.on_run_start(run_state, DefaultRunContext())
        turn = PreparedTurn(run_state=run_state, tools=None, model_args=None)
        await bound_skills.prepare_turn(turn)
        await bound_resolver.prepare_turn(turn)

        assert turn.tools is not None
        tools = {tool.name: tool for tool in turn.tools.executable_tools}
        await tools["activate_skill"].run(
            tools["activate_skill"].args_type()(name="report-generator"),
            CancellationToken(),
            ToolExecutionContext(),
        )

        # An absolute workspace path still reads the workspace file, not the
        # same-named skill resource, because resolution only fires for plain
        # relative paths.
        result = await tools["read"].run(
            tools["read"].args_type()(path=str(workspace / "notes.txt")),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "workspace note" in result

    asyncio.run(scenario())


def test_skill_relative_path_shim_rejects_renamed_argument() -> None:
    # A renamed/absent path argument must fail the bind loudly instead of
    # silently disabling resolution: here the configured field does not exist on
    # the read tool's argument model.
    skill_root = Path("/skills/report")
    manifest = SkillManifest(
        name="report",
        description="Report skill.",
        skill_file=skill_root / "SKILL.md",
        root=skill_root,
    )
    catalog = SkillCatalog(manifests=[manifest], loader=cast(Any, _DummyLoader()))

    with pytest.raises(ValueError, match="does not declare the 'filepath' argument"):
        SkillRelativePathShim(
            base_harness_tools(include=("read",)),
            catalog=catalog,
            path_arg_fields={"read": "filepath"},
        )


def test_skill_relative_path_shim_passes_through_unwrapped_tools() -> None:
    catalog = SkillCatalog(manifests=[], loader=cast(Any, _DummyLoader()))

    async def scenario() -> None:
        shim = SkillRelativePathShim(
            base_harness_tools(include=("read", "bash")),
            catalog=catalog,
        )
        bound = await shim.bind(_binding_context())

        run_state = RunState(instructions=None, history=[], responses=[])
        turn = PreparedTurn(run_state=run_state, tools=None, model_args=None)
        await bound.prepare_turn(turn)

        assert isinstance(turn.tools, Tools)
        names = {tool.name for tool in turn.tools.executable_tools}
        assert names == {"read", "bash"}

    asyncio.run(scenario())


class _PathToolArgs(BaseModel):
    """Arguments for a custom path-taking tool used to check field fidelity."""

    path: str


def test_skill_relative_path_shim_preserves_wrapped_tool_formatter_and_schema() -> None:
    # The wrapper must copy every Tool field by construction, not re-list them.
    # A custom formatter and an explicit parameters schema are the two fields a
    # field-by-field reconstruction historically dropped or diverged on.
    catalog = SkillCatalog(manifests=[], loader=cast(Any, _DummyLoader()))
    explicit_schema = {
        "type": "object",
        "properties": {"path": {"type": "string", "title": "Custom Path"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def custom_formatter(value: str) -> str:
        return f"formatted::{value}"

    async def handler(
        args: _PathToolArgs,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del cancellation_token, context
        return args.path

    source_tool: Tool[_PathToolArgs, str] = Tool(
        name="read",
        description="Read a path.",
        args_model=_PathToolArgs,
        handler=handler,
        formatter=custom_formatter,
        parameters_schema=explicit_schema,
    )
    definition = HarnessToolDefinition(tool=source_tool)

    async def scenario() -> None:
        shim = SkillRelativePathShim([definition], catalog=catalog)
        bound = await shim.bind(_binding_context())

        run_state = RunState(instructions=None, history=[], responses=[])
        turn = PreparedTurn(run_state=run_state, tools=None, model_args=None)
        await bound.prepare_turn(turn)

        assert isinstance(turn.tools, Tools)
        wrapped = {tool.name: tool for tool in turn.tools.executable_tools}["read"]
        assert isinstance(wrapped, Tool)
        # The custom formatter is carried over, not replaced by the source tool's
        # bound render method.
        assert wrapped.formatter is custom_formatter
        assert wrapped.return_value_as_string("x") == "formatted::x"
        # The explicit parameters schema survives the copy unchanged.
        assert wrapped.schema["parameters"] == explicit_schema

    asyncio.run(scenario())


def test_skill_relative_path_shim_rejects_declarative_tool() -> None:
    # The declarative `agent` tool is executed by the runner, not callable here,
    # so selecting it for path rewriting must fail at construction.
    catalog = SkillCatalog(manifests=[], loader=cast(Any, _DummyLoader()))

    with pytest.raises(TypeError, match="is not an executable native tool"):
        SkillRelativePathShim(
            base_harness_tools(include=("agent",)),
            catalog=catalog,
            path_arg_fields={"agent": "path"},
        )


def test_skill_relative_path_shim_resolves_optional_find_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "report-generator",
        {"references/guide.md": "guide body"},
    )

    async def scenario() -> None:
        loader = FilesystemSkillLoader(
            roots=(skills_root,),
            include_default_roots=False,
        )
        catalog = await discover_skill_catalog(loader)
        skills_shim = SkillsShim(catalog=catalog)
        resolver_shim = SkillRelativePathShim(
            base_harness_tools(cwd=workspace, include=("find",)),
            catalog=catalog,
        )
        bound_skills = await skills_shim.bind(_binding_context())
        bound_resolver = await resolver_shim.bind(_binding_context())

        run_state = RunState(instructions=None, history=[], responses=[])
        await bound_skills.on_run_start(run_state, DefaultRunContext())
        await bound_resolver.on_run_start(run_state, DefaultRunContext())
        turn = PreparedTurn(run_state=run_state, tools=None, model_args=None)
        await bound_skills.prepare_turn(turn)
        await bound_resolver.prepare_turn(turn)

        assert turn.tools is not None
        tools = {tool.name: tool for tool in turn.tools.executable_tools}
        await tools["activate_skill"].run(
            tools["activate_skill"].args_type()(name="report-generator"),
            CancellationToken(),
            ToolExecutionContext(),
        )

        # A relative directory that exists only in the active skill resolves to
        # the skill root so find lists the bundled resource.
        result = await tools["find"].run(
            tools["find"].args_type()(pattern="*.md", path="references"),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "guide.md" in result

        # The optional None path is left untouched and find falls back to cwd.
        none_path_result = await tools["find"].run(
            tools["find"].args_type()(pattern="*.md", path=None),
            CancellationToken(),
            ToolExecutionContext(),
        )
        assert "guide.md" not in none_path_result

    asyncio.run(scenario())


def test_skill_path_prompt_guidance_mentions_bash_limit() -> None:
    assert "bash" in SKILL_PATH_PROMPT_GUIDANCE
    assert "read tool" in SKILL_PATH_PROMPT_GUIDANCE


class _DummyLoader:
    async def discover(self) -> tuple[object, ...]:
        return ()

    async def load(self, name: str) -> object:
        raise KeyError(name)
