"""Prompt and payload rendering for harness skills."""

from dataclasses import dataclass

from jinja2 import Template

from ._types import LoadedSkill, SkillManifest

DEFAULT_SKILLS_SYSTEM_PROMPT = """
<skills_system>
You have access to the following skills that provide specialized instructions for specific tasks.
When a task matches a skill's description, call the {{ tool_name }} tool with the skill's name to load its full instructions.

<available_skills>
{% for skill in skills %}
  <skill>
    <name>{{ skill.name }}</name>
    <description>{{ skill.description }}</description>
    <location>{{ skill.skill_file }}</location>
  </skill>
{% endfor %}
</available_skills>
</skills_system>
"""
"""Default skills guidance appended to the effective system instructions."""

ACTIVATE_SKILL_TOOL_DESCRIPTION = (
    "Activate a discovered skill by exact name. Use this when the current task "
    "matches one of the available skill descriptions in the system instructions."
)
"""Model-visible description for the `activate_skill` tool."""

LOADED_SKILL_TEMPLATE = """
<skill_content name="{{ skill.manifest.name }}">
{{ skill.instructions }}

Skill directory: {{ skill.manifest.root }}
Relative paths in this skill are relative to the skill directory.

<skill_resources>
{% for resource in skill.resources %}
  <file>{{ resource.path }}</file>
{% endfor %}
</skill_resources>
</skill_content>
"""


@dataclass(frozen=True, slots=True)
class SkillsSystemPromptContext:
    """Typed context used to render the skills system prompt."""

    tool_name: str
    """Name of the activation tool exposed to the model."""

    skills: tuple[SkillManifest, ...]
    """Discovered skills visible to the model before activation."""


def render_skills_system_prompt(
    *,
    template: str,
    context: SkillsSystemPromptContext,
) -> str:
    """Render the skills system block for the discovered catalog."""
    return (
        Template(template)
        .render(
            tool_name=context.tool_name,
            skills=context.skills,
        )
        .strip()
    )


def render_loaded_skill(loaded_skill: LoadedSkill) -> str:
    """Render activated skill content into the tool-result payload."""
    template = Template(LOADED_SKILL_TEMPLATE, trim_blocks=True, lstrip_blocks=True)
    return template.render(skill=loaded_skill).strip()
