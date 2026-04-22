# Harness Skills

Skills are the first major first-party capability built on top of
[Harness Shims](./shims.md).

They let an agent expose a skills library to the model, activate a matching
skill through the normal tool loop, and keep the loaded skill content in the
append-only conversation history for later turns in the same run.

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

## Mental Model

The main integration point is `SkillsShim`.

Attach it through `AgentDescriptor.shims`:

```python
descriptor = AgentDescriptor(
    name="Support",
    model=model,
    shims=(SkillsShim(),),
)
```

That shim does five things:

1. discovers skills once when it binds to a concrete agent instance,
2. appends one skills guidance block to the system instruction before the first
   model turn, if any skills were discovered,
3. contributes one `activate_skill(name: str)` tool,
4. loads the full skill content only when the model activates a skill,
5. deduplicates repeated activation through `RunState.shim_state`.

## Before Activation

Before the model activates any skill, it sees:

1. the skills system prompt appended by `SkillsShim`,
2. the available skill names,
3. the skill descriptions,
4. the absolute `SKILL.md` paths,
5. the `activate_skill` tool.

If no skills are discovered, the shim does not modify the system instruction
and does not register the activation tool.

## After Activation

When the model calls `activate_skill`, the shim returns one tool result that
contains:

1. the full `SKILL.md` body without frontmatter,
2. the skill directory path,
3. resource file paths relative to the skill directory,
4. one `<skill_content>` block that groups those pieces together.

That tool result becomes part of the normal tool loop and is preserved in the
conversation history. Later turns in the same run continue with that skill
content already visible to the model.

Repeated activation of the same skill returns a plain tool-result message
instead of injecting the same skill content again.

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
payloads expose resource file paths relative to the skill directory so the
model can resolve them against the emitted `Skill directory: ...` line.

### Filesystem Parsing Policy

The filesystem loader is best-effort by design.

It skips a skill entirely when:

1. the file cannot be read,
2. YAML frontmatter is missing or malformed,
3. frontmatter is not a mapping,
4. `name` is missing or empty,
5. `description` is missing or empty,
6. the file exceeds the configured line limit.

It logs a warning and continues for softer issues such as:

1. name-spec drift,
2. oversized `description` or `compatibility` values,
3. non-mapping metadata,
4. loose field types that can be coerced safely.

One malformed skill does not fail discovery or break the agent loop. The loader
skips that skill and continues with the rest.

## State

Activated skill names are persisted in `RunState.shim_state`.

That gives two important properties:

1. repeated activation can be deduplicated,
2. later turns in the same run continue without reloading the same skill
   instructions.

The actual skill content remains visible because the activation tool result is
already part of the persisted conversation history.

## Customization

You can customize both the skill source and the system prompt:

```python
shim = SkillsShim(
    loader=my_loader,
    system_prompt=my_prompt_template,
    tool_name="activate_skill",
)
```

`DEFAULT_SKILLS_SYSTEM_PROMPT` is the built-in template used when
`system_prompt` is not provided.

## Example

For a runnable example, see
[examples/harness/default_agent_skills_quickstart](../../examples/harness/default_agent_skills_quickstart/README.md).
