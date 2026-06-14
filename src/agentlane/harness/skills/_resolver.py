"""Pure skill-relative path resolution for active skills.

This module owns the narrow, side-effect-free rule for turning a model-supplied
relative path into an absolute path under an active skill's root. It is the
core of the opt-in `SkillRelativePathShim`: given the active skill roots and one
candidate path, decide whether the path names a resource bundled with an active
skill and, if so, return its absolute location.

The contract is intentionally conservative. See `docs/harness/skills.md` for the
design rationale and the explicit limits (notably: no shell-command rewriting).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

# Glob, brace, and home-expansion characters whose meaning depends on a shell or
# glob layer the framework does not run here. A path carrying any of them is left
# untouched so resolution never silently changes a pattern's meaning.
_AMBIGUOUS_PATH_CHARACTERS = frozenset("*?[]{}~")


def resolve_skill_relative_path(
    path: str,
    *,
    skill_roots: Sequence[Path],
) -> str:
    """Return the absolute skill path when `path` names an active skill resource.

    Resolution only fires for a plain workspace-style relative path. Absolute
    paths, explicitly anchored paths (`./`, `../`, `~`), empty strings, and
    values carrying glob or brace metacharacters are returned unchanged so this
    helper never reinterprets a path the caller anchored deliberately.

    Active skill roots are tried in reverse order, so the most recently
    activated skill wins when two skills bundle the same relative path. A
    candidate that escapes its skill root through `..` collapse, or that does
    not exist on disk, is skipped; when no active skill provides the resource
    the original `path` is returned unchanged.

    Args:
        path: Candidate path supplied by the model, as received by the tool.
        skill_roots: Resolved roots of the currently active skills, in
            activation order.

    Returns:
        str: The absolute path under an active skill root, or the unchanged
        input when the path is not an active skill resource.
    """
    if not _is_plain_relative_path(path):
        return path

    relative_path = Path(path)
    for skill_root in reversed(tuple(skill_roots)):
        resolved_root = skill_root.expanduser().resolve(strict=False)
        candidate = (resolved_root / relative_path).resolve(strict=False)
        # `..` segments collapse during resolution and can climb above the skill
        # root; skip escaping candidates so the existence probe never leaks
        # files outside the skill. Final reads stay permission-gated regardless.
        if not candidate.is_relative_to(resolved_root):
            continue
        if candidate.exists():
            return str(candidate)
    return path


def active_skill_roots(
    *,
    active_skill_names: Sequence[str],
    skill_roots: Mapping[str, Path],
) -> tuple[Path, ...]:
    """Return resolved roots for the active skills that are known to the map.

    Names without a known root are dropped silently: a skill can be active for
    its instructions without contributing any resolvable resource root.

    Args:
        active_skill_names: Skill names activated in the current run.
        skill_roots: Mapping of skill name to its root directory.

    Returns:
        tuple[Path, ...]: Roots for the active, known skills in activation
        order.
    """
    return tuple(
        skill_roots[name] for name in active_skill_names if name in skill_roots
    )


def _is_plain_relative_path(path: str) -> bool:
    """Return whether `path` is a bare relative path safe to re-root."""
    if path.strip() == "":
        return False
    if not _AMBIGUOUS_PATH_CHARACTERS.isdisjoint(path):
        return False

    raw_path = Path(path)
    if raw_path.is_absolute():
        return False
    if path.startswith(("./", "../")):
        return False
    return True
