"""First-party harness shim for skill discovery and activation."""

from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic import BaseModel, Field

from agentlane.models import Tool
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken

from .._run import RunState, ShimState
from .._tooling import merge_tools
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

    async def _activate_skill(
        self,
        args: ActivateSkillInput,
        cancellation_token: CancellationToken,
    ) -> str:
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
        system_prompt: str | None = None,
        tool_name: str = "activate_skill",
        name: str = "skills",
    ) -> None:
        self._loader = loader
        self._system_prompt = system_prompt or DEFAULT_SKILLS_SYSTEM_PROMPT
        self._tool_name = tool_name
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        del context
        loader = self._loader or FilesystemSkillLoader()
        catalog = SkillCatalog(
            manifests=await loader.discover(),
            loader=loader,
        )
        return _BoundSkillsShim(
            shim_name=self._name,
            catalog=catalog,
            system_prompt=self._system_prompt,
            tool_name=self._tool_name,
        )


def _build_activate_skill_tool(
    *,
    tool_name: str,
    handler: Callable[[ActivateSkillInput, CancellationToken], Awaitable[str]],
) -> Tool[ActivateSkillInput, str]:
    """Return the normal skill-activation tool contributed by the shim."""

    async def run_tool(
        args: ActivateSkillInput,
        cancellation_token: CancellationToken,
    ) -> str:
        return await handler(args, cancellation_token)

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
