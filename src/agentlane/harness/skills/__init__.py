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
from ._parser import ParsedSkillFile, parse_skill_text
from ._prompt import DEFAULT_SKILLS_SYSTEM_PROMPT
from ._resolver import resolve_skill_relative_path
from ._resolver_shim import SKILL_PATH_PROMPT_GUIDANCE, SkillRelativePathShim
from ._shim import SkillsShim, discover_skill_catalog
from ._types import LoadedSkill, SkillManifest, SkillResource

__all__ = [
    "DEFAULT_SKILLS_SYSTEM_PROMPT",
    "FilesystemSkillLoader",
    "LoadedSkill",
    "ParsedSkillFile",
    "SKILL_PATH_PROMPT_GUIDANCE",
    "SkillCatalog",
    "SkillLoader",
    "SkillManifest",
    "SkillRelativePathShim",
    "SkillResource",
    "SkillsShim",
    "discover_skill_catalog",
    "parse_skill_text",
    "resolve_skill_relative_path",
    "SKILL_MAX_COMPATIBILITY_LENGTH",
    "SKILL_MAX_DESCRIPTION_LENGTH",
    "SKILL_MAX_FILE_LINES",
    "SKILL_MAX_NAME_LENGTH",
]
