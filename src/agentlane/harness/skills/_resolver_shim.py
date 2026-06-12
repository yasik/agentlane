"""Opt-in shim that resolves skill-relative path arguments for harness tools.

`SkillRelativePathShim` wraps first-party path-taking tools (`read`, `grep`,
`find`) so that, while a skill is active, a plain relative path the model passes
is resolved against that skill's root before the underlying tool runs. This lets
an installed skill reference its own bundled files by their in-skill relative
path without the application teaching every tool where each skill lives.

The contract is deliberately narrow and safe:

1. Only a single, declared string path argument per tool is resolved.
2. Resolution only fires for plain relative paths (see `_resolver`); absolute,
   anchored, glob, and brace values pass through untouched.
3. Shell-command arguments are never rewritten. The `bash` tool is out of scope
   by design; skill roots are surfaced to the model through prompt guidance so
   it can use absolute paths or the `read` tool inside commands.

See `docs/harness/skills.md` for the rationale and the full limit list.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from agentlane.models import Tool, ToolExecutionContext
from agentlane.models.run import RunContext
from agentlane.runtime import CancellationToken

from .._run import RunState
from .._tooling import merge_tools
from ..shims import BoundShim, PreparedTurn, Shim, ShimBindingContext
from ..tools import HarnessToolDefinition
from ._catalog import SkillCatalog
from ._resolver import active_skill_roots, resolve_skill_relative_path
from ._shim import SkillsShim

SKILL_PATH_PROMPT_GUIDANCE = (
    "When a skill is active, a relative resource path you pass to read, grep, or "
    "find resolves from that skill's root directory. For bash commands, the "
    "skill root is shown in the activation result as `Skill directory: ...`; "
    "reference resources by that absolute path or with the read tool instead of "
    "a skill-relative path."
)
"""Prompt guidance describing the narrowed skill-relative path behavior."""

_DEFAULT_PATH_ARG_FIELD = "path"
_DEFAULT_PATH_ARG_FIELDS: dict[str, str] = {
    "read": _DEFAULT_PATH_ARG_FIELD,
    "grep": _DEFAULT_PATH_ARG_FIELD,
    "find": _DEFAULT_PATH_ARG_FIELD,
}

type _RootsProvider = Callable[[], tuple[Path, ...]]
"""Callable returning the active skill roots at tool-execution time."""


class SkillRelativePathShim(Shim):
    """Resolve skill-relative path arguments for selected harness tools.

    Pair this shim with a `SkillsShim` (matched by `skills_shim_name`). It reads
    the skills the paired shim has activated in the current run and, for each
    wrapped tool, resolves the tool's path argument against the active skill
    roots before the tool executes.
    """

    def __init__(
        self,
        definitions: Sequence[HarnessToolDefinition],
        *,
        catalog: SkillCatalog,
        skills_shim_name: str = "skills",
        path_arg_fields: Mapping[str, str] | None = None,
        name: str = "skill-relative-paths",
    ) -> None:
        """Initialize the resolver shim over a set of tool definitions.

        Args:
            definitions: Tool definitions to expose. Definitions whose tool name
                appears in the path-argument map are wrapped to resolve that
                argument; all others pass through unchanged.
            catalog: Discovered skill catalog providing skill roots. Share the
                same catalog with the paired `SkillsShim` via
                `discover_skill_catalog(...)` so skills are discovered once.
            skills_shim_name: Name of the paired `SkillsShim` whose activation
                state is read.
            path_arg_fields: Optional mapping of tool name to the string
                argument resolved for that tool. Defaults to resolving the
                `path` argument of `read`, `grep`, and `find`. A tool only has
                its argument resolved when it appears in this map.
            name: Stable shim name.

        Raises:
            ValueError: When a tool selected for rewriting does not declare its
                configured path argument, so a renamed argument fails the bind
                loudly instead of silently disabling resolution.
            TypeError: When a tool selected for rewriting is a declarative spec
                rather than an executable tool.
        """
        self._name = name
        self._skill_roots = {manifest.name: manifest.root for manifest in catalog}
        # A name-only SkillsShim is used purely as the public accessor for the
        # paired shim's active-skill names; this avoids re-deriving the private
        # activation state-key format here.
        self._active_names_reader = SkillsShim(name=skills_shim_name)
        self._path_arg_fields = (
            dict(_DEFAULT_PATH_ARG_FIELDS)
            if path_arg_fields is None
            else dict(path_arg_fields)
        )
        self._definitions = tuple(definitions)
        _validate_path_arguments(self._definitions, self._path_arg_fields)

    @property
    def name(self) -> str:
        return self._name

    async def bind(self, context: ShimBindingContext) -> BoundShim:
        del context
        return _BoundSkillRelativePathShim(
            definitions=self._definitions,
            path_arg_fields=self._path_arg_fields,
            skill_roots=self._skill_roots,
            active_names_reader=self._active_names_reader,
        )


class _BoundSkillRelativePathShim(BoundShim):
    """Bound resolver session that exposes wrapped tools and tracks run state."""

    def __init__(
        self,
        *,
        definitions: tuple[HarnessToolDefinition, ...],
        path_arg_fields: Mapping[str, str],
        skill_roots: Mapping[str, Path],
        active_names_reader: SkillsShim,
    ) -> None:
        self._skill_roots = skill_roots
        self._active_names_reader = active_names_reader
        self._run_state: RunState | None = None
        self._definitions = tuple(
            (
                _wrap_definition(
                    definition,
                    arg_field=path_arg_fields[definition.tool.name],
                    roots_provider=self._active_skill_roots,
                )
                if definition.tool.name in path_arg_fields
                else definition
            )
            for definition in definitions
        )

    async def on_run_start(
        self,
        state: RunState,
        transient_state: RunContext[Any],
    ) -> None:
        del transient_state
        self._run_state = state

    async def prepare_turn(self, turn: PreparedTurn) -> None:
        self._run_state = turn.run_state
        tool_specs = tuple(definition.tool for definition in self._definitions)
        turn.tools = merge_tools(turn.tools, tool_specs)

    def _active_skill_roots(self) -> tuple[Path, ...]:
        """Return resolved roots for the skills active in the current run.

        One bound shim processes its agent's runs sequentially, so the latest
        captured run state is always the active one; no per-call run identity is
        needed for the lookup.
        """
        if self._run_state is None:
            return ()
        active_names = self._active_names_reader.active_skill_names(self._run_state)
        return active_skill_roots(
            active_skill_names=active_names,
            skill_roots=self._skill_roots,
        )


def _validate_path_arguments(
    definitions: Sequence[HarnessToolDefinition],
    path_arg_fields: Mapping[str, str],
) -> None:
    """Reject a selected tool that does not declare its configured path field."""
    for definition in definitions:
        arg_field = path_arg_fields.get(definition.tool.name)
        if arg_field is None:
            continue
        tool = definition.tool
        if not isinstance(tool, Tool):
            raise TypeError(
                f"tool {tool.name!r} is not an executable native tool and "
                "cannot resolve skill-relative paths"
            )
        if arg_field not in tool.args_type().model_fields:
            raise ValueError(
                f"tool {tool.name!r} does not declare the {arg_field!r} "
                "argument required for skill-relative path resolution"
            )


def _wrap_definition(
    definition: HarnessToolDefinition,
    *,
    arg_field: str,
    roots_provider: _RootsProvider,
) -> HarnessToolDefinition:
    """Return a definition whose tool resolves `arg_field` against skill roots."""
    tool = cast(Tool[BaseModel, Any], definition.tool)
    wrapped = _wrap_tool(tool, arg_field=arg_field, roots_provider=roots_provider)
    return HarnessToolDefinition(
        tool=wrapped,
        prompt_snippet=definition.prompt_snippet,
        prompt_guidelines=tuple(definition.prompt_guidelines),
    )


def _wrap_tool(
    tool: Tool[BaseModel, Any],
    *,
    arg_field: str,
    roots_provider: _RootsProvider,
) -> Tool[BaseModel, Any]:
    """Return a tool that resolves one path argument before delegating."""

    async def run_tool(
        args: BaseModel,
        cancellation_token: CancellationToken,
        context: ToolExecutionContext,
    ) -> Any:
        resolved_args = _resolve_args(
            args,
            arg_field=arg_field,
            skill_roots=roots_provider(),
        )
        return await tool.run(resolved_args, cancellation_token, context)

    return Tool(
        name=tool.name,
        description=tool.description,
        args_model=tool.args_type(),
        handler=run_tool,
        formatter=tool.return_value_as_string,
        parameters_schema=tool.schema["parameters"],
    )


def _resolve_args(
    args: BaseModel,
    *,
    arg_field: str,
    skill_roots: Sequence[Path],
) -> BaseModel:
    """Return args with the path field resolved against active skill roots."""
    raw_value = getattr(args, arg_field, None)
    if not isinstance(raw_value, str):
        return args
    resolved_value = resolve_skill_relative_path(raw_value, skill_roots=skill_roots)
    if resolved_value == raw_value:
        return args
    return args.model_copy(update={arg_field: resolved_value})
