import asyncio
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from agentlane.harness.shims import PreparedTurn, ShimBindingContext
from agentlane.harness.tools import WorkspaceToolsShim
from agentlane.models import Tool
from agentlane.util import CancellationToken

from .tools_test_utils import run_state


def test_workspace_tools_shim_renders_workspace_relative_guidance(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shim = WorkspaceToolsShim(workspace, include=("write",))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert turn.tools is not None
        assert [tool.name for tool in turn.tools.normalized_tools] == ["write"]
        assert isinstance(state.instructions, str)
        assert (
            "Tool paths are relative to the workspace root; do not prefix the "
            "workspace directory name."
        ) in state.instructions

    asyncio.run(scenario())


def test_workspace_tools_shim_write_uses_workspace_relative_path(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        shim = WorkspaceToolsShim(workspace, include=("write",))
        bound = await shim.bind(cast(ShimBindingContext, object()))
        state = run_state()
        turn = PreparedTurn(
            run_state=state,
            tools=None,
            model_args=None,
        )

        await bound.prepare_turn(turn)

        assert turn.tools is not None
        write_spec = turn.tools.normalized_tools[0]
        write = cast(Tool[BaseModel, str], write_spec)
        args_model = write.args_type()
        output = await write.run(
            args_model(path="notes/today.md", content="today\n"),
            CancellationToken(),
        )

        expected_path = workspace / "notes" / "today.md"
        double_prefixed_path = workspace / "workspace" / "notes" / "today.md"
        assert output.endswith(f"to {expected_path}.")
        assert expected_path.read_text(encoding="utf-8") == "today\n"
        assert not double_prefixed_path.exists()

    asyncio.run(scenario())
