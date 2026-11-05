"""Query local and mounted JSON files with jq, without a model or API key."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agentlane.harness.filesystem import LocalFileSystem, MountedFileSystem
from agentlane.harness.tools import jq_tool
from agentlane.models import Tool, ToolFailure
from agentlane.runtime import CancellationToken


async def main() -> None:
    """Use the same jq filter with local and mounted input paths."""
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
        definitions = (
            jq_tool(cwd=root),
            jq_tool(reader=storage, cwd="/data", process_cwd=root),
        )
        for definition in definitions:
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


if __name__ == "__main__":
    asyncio.run(main())
