"""First-party harness shim for skill discovery and activation."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel, Field

from agentlane.models import Tool, ToolExecutionContext, Tools
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken

from .._run import RunState, ShimState
from .._tooling import exclude_tools, filter_tools, merge_tools
from ..shims import BoundShim, PreparedTurn, Shim, ShimBindingContext
from ._catalog import SkillCatalog
from ._loader import SkillLoader
from ._loader_fs import FilesystemSkillLoader
from ._prompt import (
    ACTIVATE_SKILL_TOOL_DESCRIPTION,
    DEFAULT_SKILLS_SYSTEM_PROMPT,
    SkillsSystemPromptContext,
    render_loaded_skill,
    render_skills_system_prompt,
)


class ActivateSkillInput(BaseModel):
    """Arguments for the `activate_skill` tool."""

    name: str = Field(
        min_length=1,
        description="Exact skill name to activate.",
    )


class _BoundSkillsShim(BoundShim):
    """Bound skills shim session for one concrete agent instance."""

    def __init__(
        self,
        *,
        shim_name: str,
        catalog: SkillCatalog,
        system_prompt: str,
        tool_name: str,
    ) -> None:
        self._shim_name = shim_name
        self._catalog = catalog
        self._system_prompt = system_prompt
        self._tool_name = tool_name
        self._current_run_state: RunState | None = None
        self._tool = _build_activate_skill_tool(
            tool_name=tool_name,
            handler=self._activate_skill,
        )

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        del transient_state
        self._current_run_state = state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self._current_run_state = turn.run_state
        if len(self._catalog) == 0:
            return
        if turn.run_state.turn_count == 1:
            skills_prompt = render_skills_system_prompt(
                template=self._system_prompt,
                context=SkillsSystemPromptContext(
                    tool_name=self._tool_name,
                    skills=tuple(self._catalog),
                ),
            )
            turn.append_system_instruction(skills_prompt)

        turn.tools = merge_tools(turn.tools, (self._tool,))
        turn.tools = _apply_active_skill_tool_filters(
            tools=turn.tools,
            catalog=self._catalog,
            shim_state=turn.run_state.shim_state,
            active_names_key=_active_names_key(self._shim_name),
        )

    async def _activate_skill(
        self,
        args: ActivateSkillInput,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        del context
        del cancellation_token

        skill_name = args.name
        if not self._catalog.has(skill_name):
            available_skill_names = ", ".join(self._catalog.names())
            return (
                f"Skill `{skill_name}` was not found."
                f" Available skills: {available_skill_names}."
            )

        shim_state = self._require_shim_state()
        active_names_key = _active_names_key(self._shim_name)
        if skill_name in _active_skill_names(
            shim_state=shim_state, key=active_names_key
        ):
            return _already_active_message(skill_name)

        loaded_skill = await self._catalog.load(skill_name)
        appended = await shim_state.append_if_unique(
            active_names_key,
            skill_name,
            lambda value: value,
        )
        if not appended:
            return _already_active_message(skill_name)
        return render_loaded_skill(loaded_skill)

    def _require_shim_state(self) -> ShimState:
        """Return the persisted shim state for the current run."""
        if self._current_run_state is None:
            raise RuntimeError("SkillsShim activation requires an active run state.")
        return self._current_run_state.shim_state


class SkillsShim(Shim):
    """First-party shim that exposes an agent skills to the model."""

    def __init__(
        self,
        *,
        loader: SkillLoader | None = None,
        catalog: SkillCatalog | None = None,
        system_prompt: str | None = None,
        tool_name: str = "activate_skill",
        name: str = "skills",
    ) -> None:
        """Initialize one skills shim definition.

        Args:
            loader: Optional skill loader used to discover skills at bind time.
                Defaults to a `FilesystemSkillLoader`. Ignored when `catalog`
                is provided.
            catalog: Optional already-discovered catalog to reuse. Pass the
                value returned by `discover_skill_catalog(...)` so an
                application can share one discovered catalog with this shim and
                with its own tools instead of re-discovering skills with a
                second loader and asserting the name pairing at assembly.
            system_prompt: Optional system-prompt template override.
            tool_name: Name of the contributed activation tool.
            name: Stable shim name used for persisted state keys.

        Raises:
            ValueError: When both `loader` and `catalog` are provided.
        """
        if loader is not None and catalog is not None:
            raise ValueError(
                "SkillsShim accepts loader or catalog, not both; a catalog "
                "already carries its own loader."
            )
        self._loader = loader
        self._catalog = catalog
        self._system_prompt = system_prompt or DEFAULT_SKILLS_SYSTEM_PROMPT
        self._tool_name = tool_name
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def active_skill_names(self, run_state: RunState) -> tuple[str, ...]:
        """Return the skill names activated so far in `run_state`.

        This reads the persisted activation state this shim owns and returns
        the names in activation order. Use it instead of reconstructing the
        internal `{name}:active-skill-names` shim-state key; the key format is
        private and may change.

        Args:
            run_state: Persisted run state for the run to inspect.

        Returns:
            tuple[str, ...]: Activated skill names in activation order.
        """
        return _active_skill_names(
            shim_state=run_state.shim_state,
            key=_active_names_key(self._name),
        )

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        del context
        return _BoundSkillsShim(
            shim_name=self._name,
            catalog=await self._resolve_catalog(),
            system_prompt=self._system_prompt,
            tool_name=self._tool_name,
        )

    async def _resolve_catalog(self) -> SkillCatalog:
        """Return the shared catalog, or discover one through the loader."""
        if self._catalog is not None:
            return self._catalog
        return await discover_skill_catalog(self._loader or FilesystemSkillLoader())


async def discover_skill_catalog(loader: SkillLoader) -> SkillCatalog:
    """Discover skills once and return a shareable `SkillCatalog`.

    `SkillsShim` discovers skills inside `bind(...)`, so an application that
    needs the discovered manifests elsewhere (for example to map skill names to
    their roots) would otherwise have to discover a second time with a parallel
    loader and assert the two name sets agree. Building the catalog here and
    passing it to `SkillsShim(catalog=...)` retires that duplicate discovery:
    the same catalog backs the shim and is available to the application.

    Args:
        loader: Loader whose `discover()` result seeds the catalog.

    Returns:
        SkillCatalog: Read-only catalog over the discovered manifests, bound to
        `loader` for later activation.
    """
    return SkillCatalog(manifests=await loader.discover(), loader=loader)


def _build_activate_skill_tool(
    *,
    tool_name: str,
    handler: Callable[
        [ActivateSkillInput, CancellationToken, ToolExecutionContext],
        Awaitable[str],
    ],
) -> Tool[ActivateSkillInput, str]:
    """Return the normal skill-activation tool contributed by the shim."""

    async def run_tool(
        args: ActivateSkillInput,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> str:
        return await handler(args, cancellation_token, context)

    return Tool(
        name=tool_name,
        description=ACTIVATE_SKILL_TOOL_DESCRIPTION,
        args_model=ActivateSkillInput,
        handler=run_tool,
    )


def _active_names_key(shim_name: str) -> str:
    """Return the persisted shim-state key for activated skill names."""
    return f"{shim_name}:active-skill-names"


def _active_skill_names(
    *,
    shim_state: ShimState,
    key: str,
) -> tuple[str, ...]:
    """Return the currently active skill names for deduplication."""
    raw_value = shim_state.get(key, [])
    if not isinstance(raw_value, list):
        return ()
    raw_items = cast(list[object], raw_value)
    values = [value for value in raw_items if isinstance(value, str)]
    return tuple(values)


def _already_active_message(skill_name: str) -> str:
    """Return a directive idempotent response for repeated activation."""
    return (
        f"Skill `{skill_name}` is already active in this run. "
        f'Continue using the existing `<skill_content name="{skill_name}">`; '
        "do not call `activate_skill` for this skill again."
    )


def _apply_active_skill_tool_filters(
    *,
    tools: Tools | None,
    catalog: SkillCatalog,
    shim_state: ShimState,
    active_names_key: str,
) -> Tools | None:
    """Apply active skill tool replacement and deny rules before model exposure."""
    active_manifests = [
        manifest
        for skill_name in _active_skill_names(
            shim_state=shim_state,
            key=active_names_key,
        )
        if (manifest := catalog.get(skill_name)) is not None
    ]
    if not active_manifests:
        return tools

    filtered_tools = tools
    denied_names = frozenset(
        tool_name
        for manifest in active_manifests
        for tool_name in manifest.disallowed_tools
    )
    if denied_names:
        filtered_tools = exclude_tools(filtered_tools, names=denied_names)

    explicit_tool_sets = [
        manifest.tools for manifest in active_manifests if manifest.tools is not None
    ]
    if explicit_tool_sets:
        allowed_names = frozenset(
            tool_name for tool_names in explicit_tool_sets for tool_name in tool_names
        )
        filtered_tools = filter_tools(filtered_tools, names=allowed_names)

    return filtered_tools
