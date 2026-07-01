"""Generic shim that removes named tools from the visible set each turn."""

from collections.abc import Iterable

from .._tooling import exclude_tools
from ._base import Shim
from ._types import PreparedTurn


class ExcludeToolsShim(Shim):
    """Remove a fixed set of tool names from the visible set on every turn.

    `ToolConfig` expresses tool allow-lists but has no deny variant, so a
    "remove these tools" rule is applied as a shim over `exclude_tools`. It runs
    every turn rather than once, so a later shim that re-adds a tool cannot
    defeat the exclusion. Names that are not currently visible are ignored,
    which is correct for custom or MCP tool names the parent may not expose.

    This is the single, reusable exclusion primitive. Features that need a
    static tool denylist (for example markdown agent definitions) compose this
    shim instead of re-implementing per-turn exclusion.

    When composing more than one `ExcludeToolsShim` on the same agent, pass a
    distinct `name=` to each so their persisted shim-state keys do not collide.
    """

    def __init__(
        self,
        *,
        names: Iterable[str],
        name: str = "exclude-tools",
    ) -> None:
        """Initialize one exclusion shim.

        Args:
            names: Tool names to remove from the visible set each turn.
            name: Stable shim name used for persisted state keys.
        """
        self._names = frozenset(names)
        self._shim_name = name

    @property
    def name(self) -> str:
        return self._shim_name

    @property
    def excluded_names(self) -> frozenset[str]:
        """Return the tool names this shim removes."""
        return self._names

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        if not self._names:
            return

        turn.tools = exclude_tools(turn.tools, names=self._names)
