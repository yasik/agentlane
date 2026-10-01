"""Behavior of the native loader with non-local file I/O."""

import asyncio
import os
from pathlib import Path, PurePosixPath
from threading import get_ident
from typing import cast

import pytest
from pydantic import BaseModel
from pytest import MonkeyPatch

from agentlane.harness import RunState, Task
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.messaging import AgentId
from agentlane.models import Tool
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from ._reader import MemorySkillReader


def test_discover_injected_reader_returns_relative_metadata_off_event_loop(
    skill_reader: MemorySkillReader, monkeypatch: MonkeyPatch
) -> None:
    main_thread = get_ident()
    loader = FilesystemSkillLoader(roots=("skills",), reader=skill_reader)

    def reject_local_access(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Injected skill readers must not access local paths")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", reject_local_access)
        patch.setattr(Path, "open", reject_local_access)
        patch.setattr(Path, "iterdir", reject_local_access)
        manifests = asyncio.run(loader.discover())
        loaded = asyncio.run(loader.load("refund"))

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.skill_file == PurePosixPath("skills/refund/SKILL.md")
    assert manifest.root == PurePosixPath("skills/refund")
    assert not isinstance(manifest.root, Path)
    assert manifest.tools == ("read", "write")
    assert manifest.disallowed_tools == ("bash",)
    assert manifest.metadata == {"version": "2"}
    assert loaded.instructions == "Follow the refund policy."
    assert skill_reader.opened == ["skills/refund/SKILL.md"]
    assert skill_reader.closed == skill_reader.opened
    assert all(thread_id != main_thread for thread_id in skill_reader.thread_ids)
    assert [resource.path for resource in loaded.resources] == [
        "scripts/run.py",
        "references/nested/policy.md",
        "assets/logo.bin",
        "notes.md",
    ]


def test_load_changed_storage_keeps_discovered_metadata_and_lists_new_resources(
    skill_reader: MemorySkillReader,
) -> None:
    loader = FilesystemSkillLoader(roots=("skills",), reader=skill_reader)
    manifests = asyncio.run(loader.discover())
    skill_reader.files["skills/refund/SKILL.md"] = b"not valid frontmatter"
    skill_reader.files["skills/refund/references/new.md"] = b"new"

    loaded = asyncio.run(loader.load("refund"))

    assert loaded.manifest == manifests[0]
    assert loaded.instructions == "Follow the refund policy."
    assert "references/new.md" in [resource.path for resource in loaded.resources]
    assert skill_reader.opened == ["skills/refund/SKILL.md"]


def test_discover_injected_default_root_never_adds_local_roots(
    skill_reader: MemorySkillReader,
) -> None:
    skill_reader.files = {
        "refund/SKILL.md": skill_reader.files["skills/refund/SKILL.md"]
    }
    loader = FilesystemSkillLoader(reader=skill_reader)

    manifests = asyncio.run(loader.discover())

    assert [manifest.name for manifest in manifests] == ["refund"]
    assert skill_reader.listed == ["."]
    assert manifests[0].root == PurePosixPath("refund")


def test_load_without_discovery_reads_skill_and_missing_name_raises(
    skill_reader: MemorySkillReader,
) -> None:
    loader = FilesystemSkillLoader(roots=("skills",), reader=skill_reader)

    assert (
        asyncio.run(loader.load("refund")).instructions == "Follow the refund policy."
    )
    with pytest.raises(KeyError, match="unknown"):
        asyncio.run(loader.load("unknown"))


def test_discover_malformed_and_duplicate_skills_keeps_first_valid_name(
    skill_reader: MemorySkillReader,
) -> None:
    skill_reader.files.update(
        {
            "skills/invalid/SKILL.md": b"\xff",
            "skills/malformed/SKILL.md": b"---\nname: [\n---\nBad",
            "skills/missing/notes.md": b"No manifest",
            "skills/zduplicate/SKILL.md": (
                b"---\nname: refund\ndescription: Duplicate\n---\nOther"
            ),
        }
    )
    loader = FilesystemSkillLoader(roots=("skills", "skills"), reader=skill_reader)

    manifests = asyncio.run(loader.discover())

    assert len(manifests) == 1
    assert manifests[0].description == "Handle refunds."
    assert skill_reader.listed == ["skills"]


@pytest.mark.parametrize("root", ["/skills", "../skills", "C:\\skills", "bad\x00root"])
def test_loader_injected_unsafe_root_rejected(
    skill_reader: MemorySkillReader, root: str
) -> None:
    with pytest.raises(ValueError):
        FilesystemSkillLoader(roots=(root,), reader=skill_reader)


def test_discover_local_symlink_skill_preserves_canonical_root_and_skips_cycles(
    tmp_path: Path,
) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    target = tmp_path / "shared" / "refund"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\nname: refund\ndescription: Handle refunds.\n---\nPolicy",
        encoding="utf-8",
    )
    (target / "reference.md").write_text("details", encoding="utf-8")
    (target / "loop").symlink_to(target, target_is_directory=True)
    (skills / "refund").symlink_to(target, target_is_directory=True)
    loader = FilesystemSkillLoader(roots=(skills,), include_default_roots=False)

    manifests = asyncio.run(loader.discover())
    loaded = asyncio.run(loader.load("refund"))

    assert manifests[0].root == target.resolve()
    assert [resource.path for resource in loaded.resources] == ["reference.md"]


def test_activate_injected_skill_renders_paths_for_the_same_reader(
    skill_reader: MemorySkillReader,
) -> None:
    async def scenario() -> str:
        shim = SkillsShim(
            loader=FilesystemSkillLoader(roots=("skills",), reader=skill_reader)
        )
        task = Task(
            SingleThreadedRuntimeEngine(),
            bind_id=AgentId.from_values("skill-test", "agent"),
        )
        bound = await shim.bind(ShimBindingContext(task=task))
        state = RunState(instructions=None, history=[], responses=[], turn_count=1)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        await bound.prepare_turn(turn)
        assert turn.tools is not None
        tool = turn.tools.tools[0]
        assert isinstance(tool, Tool)
        tool = cast(Tool[BaseModel, object], tool)
        assert tool.name == "activate_skill"

        args = tool.args_type().model_validate({"name": "refund"})
        result = await tool.run(args, CancellationToken())
        assert isinstance(result, str)
        assert shim.active_skill_names(state) == ("refund",)
        return result

    result = asyncio.run(scenario())

    assert "Follow the refund policy." in result
    assert 'read_path="skills/refund/references/nested/policy.md"' in result
    assert "absolute_path" not in result
    assert skill_reader.opened == ["skills/refund/SKILL.md"]


@pytest.mark.skipif(os.name != "posix", reason="These names are valid on POSIX systems")
def test_local_names_with_backslashes_and_colons_remain_supported(
    tmp_path: Path,
) -> None:
    skill_root = tmp_path / r"refund\policy"
    skill_root.mkdir()
    (skill_root / "SKILL.md").write_text(
        "---\nname: refund\ndescription: Handle refunds.\n---\nPolicy",
        encoding="utf-8",
    )
    resources = (r"references\note.md", "a:b.md")
    for name in resources:
        (skill_root / name).write_text("details", encoding="utf-8")

    loader = FilesystemSkillLoader(roots=(tmp_path,), include_default_roots=False)
    manifests = asyncio.run(loader.discover())
    loaded = asyncio.run(loader.load("refund"))

    assert manifests[0].root == skill_root
    assert [resource.path for resource in loaded.resources] == sorted(resources)


@pytest.mark.parametrize("name", [r"bad\name", "a:b", "../escape", "/escape"])
def test_injected_malformed_child_name_rejected_before_read(
    skill_reader: MemorySkillReader, name: str
) -> None:
    skill_reader.files[f"skills/{name}/SKILL.md"] = b"malformed adapter entry"
    loader = FilesystemSkillLoader(roots=("skills",), reader=skill_reader)

    with pytest.raises(ValueError):
        asyncio.run(loader.discover())

    assert skill_reader.opened == []
