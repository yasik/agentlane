"""Run read, write, find, and patch against one mounted namespace without a model."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.tools import base_harness_tools
from agentlane.models import Tool
from agentlane.runtime import CancellationToken


async def main() -> None:
    """Use a session cwd and explicit paths across mounted file tools."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace"
        tenant = root / "tenant"
        workspace.mkdir()
        tenant.mkdir()
        (workspace / "process.txt").write_text("process workspace\n")

        # Virtual mounts do not create process filesystem mounts.
        storage = MountedFileSystem(
            {
                "workspace": LocalFileSystem(workspace),
                # Replace this child with a remote storage adapter as needed.
                "tenant": LocalFileSystem(tenant),
            }
        )

        # Each file tool captures this session cwd. Grep and bash use the
        # separate process workspace. The filesystem itself has no mutable cwd.
        definitions = {
            definition.tool.name: definition
            for definition in base_harness_tools(
                reader=storage,
                cwd=workspace,
                storage_cwd="/tenant/sessions/session-123",
            )
        }
        calls: tuple[tuple[str, dict[str, object]], ...] = (
            ("write", {"path": "notes.txt", "content": "old\n"}),
            ("find", {"pattern": "**/*.txt"}),
            ("find", {"pattern": "*.txt", "path": "/workspace"}),
            # Compose the search directory with its relative result name.
            ("read", {"path": "/workspace/process.txt"}),
            ("grep", {"pattern": "process"}),
            (
                "patch",
                {
                    "path": "notes.txt",
                    "edits": "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
                },
            ),
            ("read", {"path": "notes.txt"}),
            ("bash", {"command": "cat process.txt"}),
        )
        for name, arguments in calls:
            tool = definitions[name].tool
            assert isinstance(tool, Tool)
            result = await tool.run(
                tool.args_type().model_validate(arguments), CancellationToken()
            )
            print(f"{name}: {result}")

        # The tenant edit must reach its backend without a workspace copy.
        assert (tenant / "sessions/session-123/notes.txt").read_bytes() == b"new\n"
        assert sorted(path.name for path in workspace.iterdir()) == ["process.txt"]


if __name__ == "__main__":
    asyncio.run(main())
