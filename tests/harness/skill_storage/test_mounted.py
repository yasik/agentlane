"""Mounted storage routes skill discovery and native reads across backends."""

import asyncio
from pathlib import Path, PurePosixPath
from typing import Never, cast

import pytest
from pydantic import BaseModel

from agentlane.harness import RunState, Task
from agentlane.harness.filesystem import (
    DirectoryEntry,
    FilePathResolver,
    LocalFileSystem,
    MountedFileSystem,
    MountedReader,
    SkillReader,
)
from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.skills import FilesystemSkillLoader, SkillsShim
from agentlane.harness.tools import HarnessToolsShim, ToolPermissionDecision, read_tool
from agentlane.messaging import AgentId
from agentlane.models import Tool
from agentlane.runtime import CancellationToken, SingleThreadedRuntimeEngine

from ..tool_storage.conftest import RecordingPolicy
from ..tools_test_utils import run_tool
from ._reader import MemorySkillReader


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        "/tenant",
        "a/b",
        "a\\b",
        "C:",
        "a:b",
        "./tenant",
        "tenant/",
        "tenant/.",
        "bad\x00name",
    ],
)
def test_mount_invalid_name_rejected(
    name: str, skill_reader: MemorySkillReader
) -> None:
    with pytest.raises(ValueError):
        MountedReader({name: skill_reader})

    assert skill_reader.thread_ids == []


def test_mount_mapping_copied_and_root_listing_sorted(
    skill_reader: MemorySkillReader,
) -> None:
    mounts: dict[str, SkillReader] = {"tenant": skill_reader, "archive": skill_reader}
    reader = MountedReader(mounts)
    mounts.clear()

    assert tuple(reader.list_directory(".")) == (
        DirectoryEntry(name="archive", is_directory=True),
        DirectoryEntry(name="tenant", is_directory=True),
    )
    assert skill_reader.thread_ids == []
    with reader.open_read("tenant/skills/refund/notes.md") as stream:
        assert stream.read() == b"Notes\n"


def test_empty_mounts_returns_empty_listing_and_rejects_unknown_path() -> None:
    reader = MountedReader({})

    assert tuple(reader.list_directory(".")) == ()
    with pytest.raises(FileNotFoundError):
        reader.list_directory("tenant")


def test_mounted_resolver_new_mount_does_not_change_relative_path(
    skill_reader: MemorySkillReader,
) -> None:
    first = MountedReader({"tenant": skill_reader})
    second = MountedReader({"tenant": skill_reader, "local": skill_reader})

    assert isinstance(first, FilePathResolver)
    for reader in (first, second):
        assert reader.resolve_path(
            "local/notes.md", cwd="/tenant/sessions/current"
        ) == (PurePosixPath("/tenant/sessions/current/local/notes.md"))
        assert reader.resolve_path(
            "/local/notes.md", cwd="/tenant/sessions/current"
        ) == (PurePosixPath("/local/notes.md"))


def test_rooted_mount_paths_route_reads_writes_and_metadata(tmp_path: Path) -> None:
    filesystem = MountedFileSystem({"workspace": LocalFileSystem(tmp_path)})

    filesystem.write("/workspace/session/notes.md", b"Session notes")

    with filesystem.open_read("/workspace/session/notes.md") as stream:
        assert stream.read() == b"Session notes"
    assert (tmp_path / "session" / "notes.md").read_bytes() == b"Session notes"
    assert tuple(filesystem.list_directory("/")) == (
        DirectoryEntry(name="workspace", is_directory=True),
    )
    assert [
        entry.name for entry in filesystem.list_directory("/workspace/session")
    ] == ["notes.md"]
    root_info = filesystem.stat("/")
    file_info = filesystem.stat("/workspace/session/notes.md")
    assert root_info is not None and root_info.is_directory
    assert file_info is not None and not file_info.is_directory


def test_unknown_absolute_mount_never_falls_back_to_host(tmp_path: Path) -> None:
    host_file = tmp_path / "host.txt"
    host_file.write_text("Host file", encoding="utf-8")
    filesystem = MountedFileSystem({"workspace": LocalFileSystem(tmp_path)})

    with pytest.raises(FileNotFoundError), filesystem.open_read(str(host_file)):
        pytest.fail("Host path must not open")
    with pytest.raises(FileNotFoundError):
        filesystem.write(str(host_file), b"Changed")
    assert filesystem.stat(str(host_file)) is None
    assert host_file.read_text(encoding="utf-8") == "Host file"


def test_list_mount_root_and_nested_directory_strips_prefix(
    skill_reader: MemorySkillReader,
) -> None:
    reader = MountedReader({"tenant": skill_reader})

    assert tuple(reader.list_directory("tenant")) == (
        DirectoryEntry(name="skills", is_directory=True),
    )
    assert tuple(reader.list_directory("tenant/skills/./refund/../")) == (
        DirectoryEntry(name="refund", is_directory=True),
    )
    assert skill_reader.listed == [".", "skills"]


@pytest.mark.parametrize(
    "path", [".", "/", "tenant", "/tenant", "tenant/skills/..", "tenant/.."]
)
def test_read_namespace_or_mount_root_rejected_before_backend_access(
    skill_reader: MemorySkillReader, path: str
) -> None:
    reader = MountedReader({"tenant": skill_reader})

    with pytest.raises(IsADirectoryError), reader.open_read(path):
        pytest.fail("Directory must not open")

    assert skill_reader.thread_ids == []


@pytest.mark.parametrize("path", ["ten", "ten/skills", "tenants/skills"])
def test_unknown_mount_never_uses_prefix_match_or_fallback(
    skill_reader: MemorySkillReader, path: str
) -> None:
    reader = MountedReader({"tenant": skill_reader})

    with pytest.raises(FileNotFoundError):
        reader.list_directory(path)
    with pytest.raises(FileNotFoundError), reader.open_read(path):
        pytest.fail("Unknown mount must not open")

    assert skill_reader.thread_ids == []


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../tenant",
        "tenant/../../escape",
        "tenant\\skills",
        "tenant/bad\x00name",
        "C:/tenant",
        "tenant/C:/file",
        "tenant/a:b",
    ],
)
def test_invalid_logical_or_child_path_rejected_before_backend_access(
    skill_reader: MemorySkillReader, path: str
) -> None:
    reader = MountedReader({"tenant": skill_reader})

    with pytest.raises(ValueError):
        reader.list_directory(path)
    with pytest.raises(ValueError), reader.open_read(path):
        pytest.fail("Invalid path must not open")

    assert skill_reader.thread_ids == []


def test_normalized_path_selects_one_mount_and_child_closes_stream(
    skill_reader: MemorySkillReader,
) -> None:
    other = MemorySkillReader({"skills/refund/notes.md": b"Other"})
    reader = MountedReader({"tenant": skill_reader, "workspace": other})

    with reader.open_read("workspace/../tenant//skills/./refund/notes.md") as stream:
        assert stream.read() == b"Notes\n"
        assert skill_reader.closed == []

    assert skill_reader.opened == skill_reader.closed == ["skills/refund/notes.md"]
    assert other.thread_ids == []

    with pytest.raises(RuntimeError, match="consumer failed"):
        with reader.open_read("tenant/skills/refund/notes.md"):
            raise RuntimeError("consumer failed")

    assert skill_reader.closed == skill_reader.opened


def test_child_missing_file_does_not_fall_back_to_another_mount(
    skill_reader: MemorySkillReader,
) -> None:
    empty = MemorySkillReader({})
    reader = MountedReader({"tenant": empty, "workspace": skill_reader})

    with (
        pytest.raises(FileNotFoundError),
        reader.open_read("tenant/skills/refund/notes.md"),
    ):
        pytest.fail("Missing file must not open")
    with pytest.raises(FileNotFoundError):
        reader.list_directory("tenant/skills")

    assert empty.opened == ["skills/refund/notes.md"]
    assert empty.listed == ["skills"]
    assert skill_reader.thread_ids == []


@pytest.mark.parametrize("operation", ["open_read", "list_directory"])
@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("unavailable")])
def test_child_failure_propagates_without_fallback(
    skill_reader: MemorySkillReader,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error: OSError,
) -> None:
    failing = MemorySkillReader({})
    reader = MountedReader({"tenant": failing, "workspace": skill_reader})

    def fail(_path: str) -> Never:
        raise error

    monkeypatch.setattr(failing, operation, fail)

    with pytest.raises(type(error)) as raised:
        if operation == "list_directory":
            reader.list_directory("tenant/skills")
        else:
            with reader.open_read("tenant/skills/refund/notes.md"):
                pytest.fail("Failed backend must not open")

    assert raised.value is error
    assert skill_reader.thread_ids == []


@pytest.mark.parametrize("remote_first", [True, False])
def test_mixed_skill_roots_preserve_duplicate_precedence_and_resources(
    local_skill_root: Path, skill_reader: MemorySkillReader, remote_first: bool
) -> None:
    reader = MountedReader(
        {
            "workspace": LocalFileSystem(local_skill_root.parent.parent),
            "tenant": skill_reader,
        }
    )
    roots = ("tenant/skills", "workspace/.agents/skills")
    loader = FilesystemSkillLoader(
        roots=roots if remote_first else roots[::-1], reader=reader
    )

    manifests = asyncio.run(loader.discover())
    loaded = asyncio.run(loader.load("refund"))

    assert {manifest.name for manifest in manifests} == {"refund", "review"}
    assert loaded.manifest == next(
        manifest for manifest in manifests if manifest.name == "refund"
    )
    root = f"/{roots[0] if remote_first else roots[1]}"
    assert loaded.manifest.root == PurePosixPath(root) / "refund"
    assert loaded.instructions == (
        "Follow the refund policy."
        if remote_first
        else "Follow the local refund instructions."
    )
    resource = "references/nested/policy.md" if remote_first else "policy.md"
    assert resource in [item.path for item in loaded.resources]
    with reader.open_read(f"{root}/refund/{resource}") as stream:
        assert stream.read() == (
            b"Refund within 30 days.\n" if remote_first else b"Local refund policy.\n"
        )


def test_single_read_tool_reads_activated_local_and_remote_resources(
    local_skill_root: Path, skill_reader: MemorySkillReader
) -> None:
    reader = MountedReader(
        {
            "workspace": LocalFileSystem(local_skill_root.parent.parent),
            "tenant": skill_reader,
        }
    )

    async def scenario() -> None:
        skills = SkillsShim(
            loader=FilesystemSkillLoader(
                roots=("tenant/skills", "workspace/.agents/skills"), reader=reader
            )
        )
        tools_shim = HarnessToolsShim(
            (read_tool(reader=reader, cwd="/tenant/sessions/current"),)
        )
        context = ShimBindingContext(
            task=Task(
                SingleThreadedRuntimeEngine(),
                bind_id=AgentId.from_values("mounted-test", "agent"),
            )
        )
        state = RunState(instructions=None, history=[], responses=[], turn_count=1)
        turn = PreparedTurn(run_state=state, tools=None, model_args=None)
        for shim in (skills, tools_shim):
            bound = await shim.bind(context)
            await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert sorted(tool.name for tool in turn.tools.normalized_tools) == [
            "activate_skill",
            "read",
        ]
        native_tools: dict[str, Tool[BaseModel, object]] = {}
        for tool in turn.tools.tools:
            assert isinstance(tool, Tool)
            native_tools[tool.name] = cast(Tool[BaseModel, object], tool)

        activate = native_tools["activate_skill"]
        read = native_tools["read"]
        for name, path, expected in (
            (
                "refund",
                "/tenant/skills/refund/references/nested/policy.md",
                "Refund within 30 days.",
            ),
            (
                "review",
                "/workspace/.agents/skills/review/policy.md",
                "Local review policy.",
            ),
        ):
            activation = await activate.run(
                activate.args_type().model_validate({"name": name}), CancellationToken()
            )
            assert isinstance(activation, str)
            assert f'read_path="{path}"' in activation
            result = await read.run(
                read.args_type().model_validate({"path": path}), CancellationToken()
            )
            assert result == expected

        assert skills.active_skill_names(state) == ("refund", "review")
        report = await read.run(
            read.args_type().model_validate({"path": "/workspace/report.txt"}),
            CancellationToken(),
        )
        assert report == "Workspace report."

    asyncio.run(scenario())


def test_read_permission_checks_full_normalized_path_before_backend_access(
    skill_reader: MemorySkillReader,
) -> None:
    reader = MountedReader({"tenant": skill_reader})
    policy = RecordingPolicy()
    definition = read_tool(reader=reader, cwd="tenant/skills", permissions=policy)

    assert run_tool(definition, path="refund/./notes.md") == "Notes"
    assert policy.requests[0].path == PurePosixPath("/tenant/skills/refund/notes.md")
    assert policy.requests[0].cwd == PurePosixPath("/tenant/skills")
    skill_reader.opened.clear()
    policy.decision = ToolPermissionDecision.deny()

    denied = run_tool(definition, path="refund/notes.md")

    assert denied.startswith("permission denied:")
    assert policy.requests[1].path == PurePosixPath("/tenant/skills/refund/notes.md")
    assert skill_reader.opened == []
