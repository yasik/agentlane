"""Skills support built on top of the harness shim system."""

from ._catalog import SkillCatalog
from ._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
    SKILL_MAX_NAME_LENGTH,
)
from ._loader import SkillLoader
from ._loader_fs import FilesystemSkillLoader
from ._prompt import DEFAULT_SKILLS_SYSTEM_PROMPT
from ._shim import SkillsShim
from ._types import LoadedSkill, SkillManifest, SkillResource

__all__ = [
    "DEFAULT_SKILLS_SYSTEM_PROMPT",
    "FilesystemSkillLoader",
    "LoadedSkill",
    "SkillCatalog",
    "SkillLoader",
    "SkillManifest",
    "SkillResource",
    "SkillsShim",
    "SKILL_MAX_COMPATIBILITY_LENGTH",
    "SKILL_MAX_DESCRIPTION_LENGTH",
    "SKILL_MAX_FILE_LINES",
    "SKILL_MAX_NAME_LENGTH",
]
