"""Skill locations stay independent from the file tools' session directory."""

import asyncio
from pathlib import Path, PurePosixPath
from typing import cast

from pydantic import BaseModel

from agentlane.harness import RunState, Task
from agentlane.harness.filesystem import MountedFileSystem
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.harness.tools import HarnessToolsShim, read_tool, write_tool
from agentlane.messaging import AgentId
from agentlane.models import Tool
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine


def test_discover_mounted_equivalent_roots_returns_canonical_locations(
    mounted_skill_filesystem: MountedFileSystem,
) -> None:
    loader = FilesystemSkillLoader(
        roots=("tenant/skills", "/tenant/skills/./", "/packaged"),
        reader=mounted_skill_filesystem,
    )

    manifests = asyncio.run(loader.discover())
    refund = asyncio.run(loader.load("refund"))

    assert {manifest.name for manifest in manifests} == {"refund", "review"}
    assert refund.manifest.root == PurePosixPath("/tenant/skills/refund")
    assert refund.manifest.skill_file == PurePosixPath("/tenant/skills/refund/SKILL.md")
    assert "Follow the refund policy." in refund.instructions
    assert "[Policy](policy.md)" in refund.instructions
    assert all(manifest.root.is_absolute() for manifest in manifests)


def test_activate_two_mounted_skills_keeps_resource_origins_and_session_writes(
    mounted_skill_filesystem: MountedFileSystem, tmp_path: Path
) -> None:
    async def scenario() -> None:
        skills = SkillsShim(
            loader=FilesystemSkillLoader(
                roots=("/tenant/skills", "/packaged"), reader=mounted_skill_filesystem
            )
        )
        cwd = "/tenant/sessions/current"
        tools = HarnessToolsShim(
            (
                read_tool(cwd=cwd, reader=mounted_skill_filesystem),
                write_tool(cwd=cwd, writer=mounted_skill_filesystem),
            )
        )
        context = ShimBindingContext(
            task=Task(
                SingleThreadedRuntimeEngine(),
                bind_id=AgentId.from_values("skill-path-test", "agent"),
            )
        )
        state = RunState(instructions=None, history=[], responses=[], turn_count=1)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        for shim in (skills, tools):
            bound = await shim.bind(context)
            await bound.prepare_turn(turn)

        assert turn.tools is not None
        native_tools: dict[str, Tool[BaseModel, object]] = {}
        for tool in turn.tools.tools:
            assert isinstance(tool, Tool)
            native_tools[tool.name] = cast(Tool[BaseModel, object], tool)

        # Activation preserves authored references and supplies their full anchor.
        for name, path, expected in (
            ("refund", "/tenant/skills/refund/policy.md", "Tenant refund policy."),
            ("review", "/packaged/review/policy.md", "Local review policy."),
        ):
            activation = await _run_tool(native_tools["activate_skill"], name=name)
            assert "[Policy](policy.md)" in activation
            assert f'read_path="{path}"' in activation
            assert "does not change the file tools' working directory" in activation
            assert await _run_tool(native_tools["read"], path=path) == expected

        assert skills.active_skill_names(state) == ("refund", "review")

        # Neither activation can redirect a bare output name into a skill folder.
        result = await _run_tool(
            native_tools["write"], path="notes.md", content="Session notes."
        )
        assert result == "Wrote 14 bytes to /tenant/sessions/current/notes.md."
        assert (
            await _run_tool(native_tools["read"], path="notes.md") == "Session notes."
        )
        missing = await _run_tool(native_tools["read"], path="policy.md")
        assert "file not found: `/tenant/sessions/current/policy.md`" in missing

    asyncio.run(scenario())

    assert (tmp_path / "tenant/sessions/current/notes.md").read_text(
        encoding="utf-8"
    ) == "Session notes."
    assert (tmp_path / "tenant/skills/refund/notes.md").read_text(
        encoding="utf-8"
    ) == "Notes\n"
    assert not (tmp_path / ".agents/skills/review/notes.md").exists()


async def _run_tool(tool: Tool[BaseModel, object], **arguments: object) -> str:
    result = await tool.run(
        tool.args_type().model_validate(arguments), CancellationToken()
    )
    assert isinstance(result, str)
    return result
