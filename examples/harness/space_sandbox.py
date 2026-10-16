"""Check six tools in an existing SPACE sandbox without an LLM.

Set SPACE_API_URL, SPACE_DATA_PLANE_URL, SPACE_TOKEN, and SPACE_SANDBOX_ID.
SPACE_CWD defaults to /workspace. SPACE_PYTHON defaults to python3.
Install the same AgentLane build in the host and guest, then run:

    uv run python examples/harness/space_sandbox.py

The application must prepare and manage the sandbox. This example creates
one temporary file, checks the tools, and removes that file on success.
"""

import asyncio
import os
from typing import Any, cast
from uuid import uuid4

from agentlane_space import SpaceSandbox, space_harness_tools

from agentlane.models import Tool, ToolFailure
from agentlane.runtime import CancellationToken


async def run_smoke() -> None:
    """Run file and Bash checks using application-supplied connection values."""
    cwd = os.environ.get("SPACE_CWD", "/workspace")
    async with SpaceSandbox(
        api_url=os.environ["SPACE_API_URL"],
        data_plane_url=os.environ["SPACE_DATA_PLANE_URL"],
        token=lambda: os.environ["SPACE_TOKEN"],
        sandbox_id=os.environ["SPACE_SANDBOX_ID"],
        python_executable=os.environ.get("SPACE_PYTHON", "python3"),
    ) as sandbox:
        await sandbox.check(cwd=cwd)
        tools = {
            definition.tool.name: cast(Tool[Any, Any], definition.tool)
            for definition in space_harness_tools(
                sandbox,
                cwd=cwd,
                include=("read", "write", "find", "grep", "patch", "bash"),
            )
        }

        async def invoke(name: str, arguments: dict[str, Any]) -> str:
            tool = tools[name]
            result = await tool.run(
                tool.args_type().model_validate(arguments), CancellationToken()
            )
            if isinstance(result, ToolFailure):
                raise RuntimeError(
                    f"{name}: {result.error.kind}: {result.error.message}"
                )
            return str(result)

        filename = f"agentlane-space-smoke-{uuid4().hex}.txt"
        await invoke("write", {"path": filename, "content": "alpha\nβeta\n"})
        assert await invoke("read", {"path": filename}) == "alpha\nβeta"
        assert filename in await invoke("find", {"pattern": filename})
        assert "alpha" in await invoke("grep", {"pattern": "alpha", "path": filename})
        await invoke(
            "patch",
            {
                "path": filename,
                "edits": "<<<<<<< SEARCH\nalpha\n=======\ngamma\n>>>>>>> REPLACE",
            },
        )
        assert await invoke("bash", {"command": f"cat -- {filename}"}) == "gamma\nβeta"
        await invoke("bash", {"command": f"rm -- {filename}"})
        print("SPACE check passed: read, write, find, grep, patch, and bash.")


if __name__ == "__main__":
    asyncio.run(run_smoke())
