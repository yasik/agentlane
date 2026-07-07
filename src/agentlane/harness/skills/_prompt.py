"""Prompt and payload rendering for harness skills."""

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

from ._types import LoadedSkill, SkillManifest, SkillResource

DEFAULT_SKILLS_SYSTEM_PROMPT = """
<skills_system>
You have access to the following skills that provide specialized instructions for specific tasks.
When a task matches a skill's description, call the {{ tool_name }} tool with the skill's name to load its full instructions.
Call {{ tool_name }} at most once per skill in a run. If the conversation already contains a <skill_content name="..."> block for a skill, continue using that existing skill content instead of activating the same skill again.

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
    "matches one of the available skill descriptions in the system instructions. "
    "Do not call this again for a skill whose skill_content block is already "
    "visible in the conversation."
)
"""Model-visible description for the `activate_skill` tool."""

LOADED_SKILL_TEMPLATE = """
<skill_content name="{{ skill.manifest.name }}">
{{ skill.instructions }}

Skill directory: {{ skill.manifest.root }}
Use absolute_path values below with filesystem tools. The path attribute is the skill-relative display path.

<skill_resources>
{% for resource in skill.resources %}
  <file path="{{ resource.path }}" absolute_path="{{ resource.absolute_path }}" />
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


@dataclass(frozen=True, slots=True)
class _LoadedSkillTemplateContext:
    """Template context with absolute resource paths precomputed."""

    manifest: SkillManifest
    instructions: str
    resources: tuple["_SkillResourceTemplateContext", ...]


@dataclass(frozen=True, slots=True)
class _SkillResourceTemplateContext:
    """Template context for one skill resource."""

    path: str
    absolute_path: Path


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
    return template.render(skill=_loaded_skill_template_context(loaded_skill)).strip()


def _loaded_skill_template_context(
    loaded_skill: LoadedSkill,
) -> _LoadedSkillTemplateContext:
    """Return a render context with absolute resource paths."""
    return _LoadedSkillTemplateContext(
        manifest=loaded_skill.manifest,
        instructions=loaded_skill.instructions,
        resources=tuple(
            _skill_resource_template_context(
                resource,
                root=loaded_skill.manifest.root,
            )
            for resource in loaded_skill.resources
        ),
    )


def _skill_resource_template_context(
    resource: SkillResource,
    *,
    root: Path,
) -> _SkillResourceTemplateContext:
    """Return a render context for one skill resource."""
    return _SkillResourceTemplateContext(
        path=resource.path,
        absolute_path=(root / resource.path).resolve(strict=False),
    )
