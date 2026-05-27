"""Workspace-oriented convenience shim for first-party harness tools."""

from collections.abc import Iterable
from pathlib import Path

from ..shims import BoundShim, Shim, ShimBindingContext
from ._permissions import ToolApprovalCallback, ToolPermissionPolicy
from ._shim import HarnessToolsShim, base_harness_tools

WORKSPACE_PATH_GUIDANCE = (
    "Tool paths are relative to the workspace root; do not prefix the workspace "
    "directory name."
)


class WorkspaceToolsShim(Shim):
    """Convenience shim for workspace-rooted first-party local tools."""

    def __init__(
        self,
        root: str | Path,
        *,
        permissions: ToolPermissionPolicy | None = None,
        approval_callback: ToolApprovalCallback | None = None,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        name: str = "workspace-tools",
    ) -> None:
        self._root = Path(root).expanduser().resolve(strict=False)
        self._name = name
        definitions = base_harness_tools(
            cwd=self._root,
            permissions=permissions,
            approval_callback=approval_callback,
            include=include,
            exclude=exclude,
        )
        self._shim = HarnessToolsShim(
            definitions,
            name=name,
            prompt_guidelines=(WORKSPACE_PATH_GUIDANCE,) if definitions else (),
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def root(self) -> Path:
        """Return the normalized workspace root used for local tool paths."""
        return self._root

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        return await self._shim.bind(context)
