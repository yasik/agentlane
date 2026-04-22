"""Shared constraints for the harness skills package
based on public specification https://agentskills.io/client-implementation/adding-skills-support.
"""

SKILL_MAX_NAME_LENGTH = 64
"""Maximum allowed skill name length from the Agent Skills spec."""

SKILL_MAX_DESCRIPTION_LENGTH = 1024
"""Maximum allowed skill description length from the Agent Skills spec."""

SKILL_MAX_COMPATIBILITY_LENGTH = 500
"""Maximum allowed compatibility field length from the Agent Skills spec."""

SKILL_MAX_FILE_LINES = 500
"""Maximum accepted `SKILL.md` line count for best-effort loading."""
