# Harness Skills

Skills are the first major first-party capability built on top of
[Harness Shims](./shims.md).

They let an agent expose a skills library to the model, activate a matching
skill through the normal tool loop, and keep the activated skill content in
later turns of the same run.

## Import Path

```python
from agentlane.harness.skills import (
    DEFAULT_SKILLS_SYSTEM_PROMPT,
    FilesystemSkillLoader,
    SkillCatalog,
    SkillLoader,
    SkillsShim,
)
```

## Core Shape

The main integration point is `SkillsShim`.

Attach it through `AgentDescriptor.shims`:

```python
descriptor = AgentDescriptor(
    name="Support",
    model=model,
    shims=(SkillsShim(),),
)
```

That shim:

1. discovers available skills before the first model call,
2. augments the effective system instructions during turn preparation,
3. contributes one `activate_skill(name: str)` tool,
4. loads the full skill content only when the model activates a skill,
5. persists activated-skill state in `RunState.shim_state`.

## Loader Interface

The harness does not hard-code the filesystem as the only source of skills.

`SkillsShim` depends on the `SkillLoader` interface. The built-in
`FilesystemSkillLoader` is only the default implementation.

That means applications may provide custom loaders for skills stored in:

1. a database,
2. a remote service,
3. an application-specific in-memory source.

Example:

```python
shim = SkillsShim(loader=my_loader)
```

## Filesystem Loader

`FilesystemSkillLoader` is the default loader.

It discovers skills from local directories rooted in `SKILL.md`.

You can point it at explicit roots:

```python
loader = FilesystemSkillLoader(
    roots=(Path("/app/skills"),),
    include_default_roots=False,
)
```

Or let it include the standard local roots:

1. `./.agents/skills`
2. `~/.agents/skills`

Discovered `SKILL.md` files are normalized to absolute paths. Activated skill
payloads then expose resource files relative to the skill directory so the
model can resolve them against the emitted `Skill directory: ...` line.

## Progressive Disclosure

Skills use progressive disclosure.

Before activation, the model sees only:

1. the skill name,
2. the skill description,
3. the absolute `SKILL.md` location,
4. the fact that it can call `activate_skill` with the exact skill name,
5. a short example workflow that shows the typical activation pattern.

After activation, the shim returns:

1. the full `SKILL.md` body without frontmatter,
2. the skill directory path,
3. bundled resource file paths relative to the skill directory,
4. one `<skill_content>` payload that keeps those pieces grouped together.

If no skills are discovered, the shim does not modify instructions and does not
register `activate_skill`.

The loader does not eagerly read `scripts/`, `references/`, or `assets/`
during discovery.

## State

Activated skill names are persisted in `RunState.shim_state`.

That gives two important properties:

1. repeated activation can be deduplicated,
2. later turns in the same run continue with the activated skill already in
   context.

## Example

For a runnable example, see
[examples/harness/default_agent_skills_quickstart](../../examples/harness/default_agent_skills_quickstart/README.md).
