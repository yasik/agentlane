"""Run read, write, find, and patch against one mounted namespace without a model."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.tools import base_harness_tools
from agentlane.models import Tool
from agentlane.runtime import CancellationToken


async def main() -> None:
    """Show that a mounted file keeps the same path through every file tool."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = root / "workspace"
        tenant = root / "tenant"
        workspace.mkdir()
        tenant.mkdir()
        (workspace / "process.txt").write_text("process workspace\n")

        # Mount names are tool path prefixes; they do not create process mounts.
        storage = MountedFileSystem(
            {
                "workspace": LocalFileSystem(workspace),
                # Replace this child with a remote storage adapter as needed.
                "tenant": LocalFileSystem(tenant),
            }
        )

        # The factory infers write access from storage. File tools start at its
        # root, while grep and bash use the separate process workspace.
        definitions = {
            definition.tool.name: definition
            for definition in base_harness_tools(reader=storage, cwd=workspace)
        }
        calls: tuple[tuple[str, dict[str, object]], ...] = (
            ("write", {"path": "tenant/notes.txt", "content": "old\n"}),
            ("find", {"pattern": "**/*.txt"}),
            ("grep", {"pattern": "process"}),
            (
                "patch",
                {
                    "path": "tenant/notes.txt",
                    "edits": "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
                },
            ),
            ("read", {"path": "tenant/notes.txt"}),
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
        assert (tenant / "notes.txt").read_bytes() == b"new\n"
        assert sorted(path.name for path in workspace.iterdir()) == ["process.txt"]


if __name__ == "__main__":
    asyncio.run(main())
