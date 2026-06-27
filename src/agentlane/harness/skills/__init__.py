"""Skills support built on top of the harness shim system."""

from ._catalog import SkillCatalog
from ._constraints import (
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
    SKILL_MAX_NAME_LENGTH,
)
from ._filesystem import LocalSkillFilesystem, SkillFilesystem, SkillFilesystemEntry
from ._loader import SkillLoader
from ._loader_fs import FilesystemSkillLoader
from ._parser import ParsedSkillFile, parse_skill_text
from ._prompt import DEFAULT_SKILLS_SYSTEM_PROMPT
from ._read_tool import filesystem_read_tool
from ._resolver import resolve_skill_relative_path
from ._resolver_shim import SKILL_PATH_PROMPT_GUIDANCE, SkillRelativePathShim
from ._shim import SkillsShim, discover_skill_catalog
from ._types import LoadedSkill, SkillManifest, SkillResource

__all__ = [
    "DEFAULT_SKILLS_SYSTEM_PROMPT",
    "FilesystemSkillLoader",
    "LoadedSkill",
    "LocalSkillFilesystem",
    "ParsedSkillFile",
    "SKILL_PATH_PROMPT_GUIDANCE",
    "SkillCatalog",
    "SkillFilesystem",
    "SkillFilesystemEntry",
    "SkillLoader",
    "SkillManifest",
    "SkillRelativePathShim",
    "SkillResource",
    "SkillsShim",
    "discover_skill_catalog",
    "filesystem_read_tool",
    "parse_skill_text",
    "resolve_skill_relative_path",
    "SKILL_MAX_COMPATIBILITY_LENGTH",
    "SKILL_MAX_DESCRIPTION_LENGTH",
    "SKILL_MAX_FILE_LINES",
    "SKILL_MAX_NAME_LENGTH",
]
