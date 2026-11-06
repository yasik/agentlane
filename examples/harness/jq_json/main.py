"""Query and save local and mounted JSON files without a model or API key."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.tools import jq_tool
from agentlane.models import Tool, ToolFailure
from agentlane.runtime import CancellationToken


async def main() -> None:
    """Query a local and mounted file, edit the source, and save a copy."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "results.json").write_text(
            json.dumps(
                {"items": [{"id": 1, "active": True}, {"id": 2, "active": False}]}
            ),
            encoding="utf-8",
        )

        # The mounted path is logical. jq receives its contents through stdin.
        storage = MountedFileSystem({"data": LocalFileSystem(root)})
        cases = (
            (jq_tool(cwd=root), "results.json"),
            (
                jq_tool(
                    reader=storage,
                    writer=storage,
                    cwd="/data",
                    process_cwd=root,
                ),
                "saved/results.json",
            ),
        )
        for definition, output_path in cases:
            tool = definition.tool
            assert isinstance(tool, Tool)
            result = await tool.run(
                tool.args_type().model_validate(
                    {
                        "path": "results.json",
                        "filter": ".items[] | select(.active) | .id",
                    }
                ),
                CancellationToken(),
            )
            if isinstance(result, ToolFailure):
                raise RuntimeError(result.error.message)

            assert result == "1"
            print(result)

            # The local call replaces its source. The mounted call creates a
            # copy through the same filesystem used to read the input.
            saved = await tool.run(
                tool.args_type().model_validate(
                    {
                        "path": "results.json",
                        "filter": ".processed = true",
                        "output_path": output_path,
                    }
                ),
                CancellationToken(),
            )
            if isinstance(saved, ToolFailure):
                raise RuntimeError(saved.error.message)

            content = json.loads((root / output_path).read_text(encoding="utf-8"))
            assert content["processed"] is True
            assert len(content["items"]) == 2
            assert saved.startswith("Wrote ")
            print(saved.splitlines()[0])


if __name__ == "__main__":
    asyncio.run(main())
