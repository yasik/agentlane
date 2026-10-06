# Harness Skills

Skills are a first-party capability built on top of
[Harness Shims](./shims.md).

They let an agent expose a skills library to the model, activate a matching
skill through the normal tool loop, and keep the loaded skill content in the
append-only conversation history for later turns in the same run.

## Import Path

```python
from agentlane.harness.skills import (
    DEFAULT_SKILLS_SYSTEM_PROMPT,
    FilesystemSkillLoader,
    LoadedSkill,
    SkillCatalog,
    SkillLoader,
    SkillManifest,
    SkillResource,
    SkillsShim,
    discover_skill_catalog,
    SKILL_MAX_COMPATIBILITY_LENGTH,
    SKILL_MAX_DESCRIPTION_LENGTH,
    SKILL_MAX_FILE_LINES,
    SKILL_MAX_NAME_LENGTH,
)
```

## Public Types

A custom `SkillLoader` produces and returns these typed primitives:

1. `SkillManifest` is the discovered metadata for one skill plus its canonical
   file locations: `name`, `description`, `skill_file`, `root`, and the optional
   `license`, `compatibility`, `metadata`, `tools`, and `disallowed_tools`
   fields. `SkillLoader.discover()` returns a sequence of these.
2. `SkillResource` is one bundled file that belongs to an activated skill,
   carrying a `path` relative to the skill directory. `SkillsShim` also renders
   a `read_path` in the reader's namespace when it activates the skill.
3. `LoadedSkill` is the activated payload returned by `SkillLoader.load(name)`:
   the `manifest`, the rendered `instructions` body, and the bundled
   `resources`.

## Constants

The skills package exposes the limits used by the filesystem parser, taken from
the [Agent Skills spec](https://agentskills.io/client-implementation/adding-skills-support):

1. `SKILL_MAX_NAME_LENGTH` (`64`),
2. `SKILL_MAX_DESCRIPTION_LENGTH` (`1024`),
3. `SKILL_MAX_COMPATIBILITY_LENGTH` (`500`),
4. `SKILL_MAX_FILE_LINES` (`500`).

## Mental Model

The main integration point is `SkillsShim`.

Attach it through `AgentDescriptor.shims`:

```python
descriptor = AgentDescriptor(
    name="Clinical Review",
    model=model,
    shims=(SkillsShim(),),
)
```

That shim does five core things:

1. discovers skills once when it binds to a concrete agent instance,
2. appends one skills guidance block to the system instruction before the first
   model turn, if any skills were discovered,
3. contributes one cache-stable `activate_skill(name: str)` tool,
4. adds the full skill instructions to the model context when the model
   activates a skill,
5. deduplicates repeated activation through `RunState.shim_state`.

## Before Activation

Before the model activates any skill, it sees:

1. the skills system prompt appended by `SkillsShim`,
2. the available skill names,
3. the skill descriptions,
4. the `SKILL.md` paths in the selected filesystem namespace,
5. the `activate_skill` tool.

If no skills are discovered, the shim does not modify the system instruction and
does not register the activation tool.

## After Activation

When the model calls `activate_skill`, the shim returns one tool result that
contains:

1. the full `SKILL.md` body without frontmatter,
2. the `Skill directory: ...` in the reader's namespace,
3. a `<skill_resources>` list where each resource includes its skill-relative
   `path` and a `read_path` suitable for a read tool in the same namespace,
4. one `<skill_content name="<skill-name>">` block that groups those pieces
   together, where the `name` attribute matches the dedup directive below.

That tool result becomes part of the normal tool loop and is preserved in the
conversation history. Later turns in the same run continue with that skill
content already visible to the model.

Repeated activation of the same skill returns a plain tool-result message
instead of injecting the same skill content again when duplicate tool calls
arrive in the same model response, through a race, or on a later model turn.

On later model turns, the activation tool schema stays stable for prompt-cache
reuse. The loaded skill content remains available in history, and the repeated
activation result tells the model to continue using the existing
`<skill_content>` block instead of calling `activate_skill` for that skill
again.

## Hooks Around Skill Activation

`activate_skill` is a normal tool contributed by `SkillsShim`.

That means standard [`RunnerHooks`](../../src/agentlane/harness/_hooks.py)
can react to skill loading the same way they react to any other tool call.
Applications can use that for logging, tracing, metering, audits, policy
checks, or other workflow-specific side effects.

Example:

```python
class SkillLoggingHooks(RunnerHooks):
    async def on_agent_start(
        self,
        task: Task,
        state: RunState,
    ) -> None:
        logger.info("agent_started", agent=task.name, next_turn=state.turn_count + 1)

    async def on_tool_call_start(
        self,
        task: Task,
        tool_call: ToolCall,
    ) -> None:
        if tool_call.function.name == "activate_skill":
            logger.info("skill_activation_started", tool_call_id=tool_call.id)

    async def on_agent_end(
        self,
        task: Task,
        result: RunResult | None,
    ) -> None:
        logger.info("agent_finished", agent=task.name)
```

Pass that hook into the agent exactly like any other runner hook:

```python
agent = DefaultAgent(
    descriptor=descriptor,
    hooks=SkillLoggingHooks(),
)
```

## Loader Interface

The harness does not hard-code the filesystem as the only source of skills.

`SkillsShim` depends on the `SkillLoader` interface. The built-in
`FilesystemSkillLoader` is only the default implementation.

For a different storage backend, pass a `SkillReader` to
`FilesystemSkillLoader`. AgentLane then retains discovery, its private
frontmatter parser, resource listing, and activation. See
[File I/O interfaces](filesystem.md) for the contracts and composition.

Use a custom `SkillLoader` when the application needs different discovery or
loading behavior, not just different storage. For example:

```python
shim = SkillsShim(loader=my_loader)
```

### Catalog

`SkillsShim` builds a `SkillCatalog` from `await loader.discover()` when it binds
to an agent instance. The catalog is a read-only container over the discovered
`SkillManifest` values and exposes:

1. `get(name)` returns the manifest for one skill name, or `None`,
2. `has(name)` returns whether the named skill exists,
3. `names()` returns the discovered skill names in stable order,
4. `await load(name)` loads one named skill into a `LoadedSkill` through the
   catalog's loader,
5. iteration and `len()` over the discovered manifests.

### Sharing a Discovered Catalog

By default `SkillsShim` discovers skills internally when it binds. When an
application also needs the discovered manifests — for example to map skill names
to their root directories — discover the catalog once with
`discover_skill_catalog(...)` and pass it to the shim. The shim then reuses that
catalog instead of discovering again, so the application and the shim cannot
disagree about which skills exist:

```python
loader = FilesystemSkillLoader(roots=(SKILLS_ROOT,), include_default_roots=False)
catalog = await discover_skill_catalog(loader)

descriptor = AgentDescriptor(
    name="Clinical Review",
    model=model,
    shims=(SkillsShim(catalog=catalog),),
)

# The same catalog is available to the application without a second loader.
skill_roots = {manifest.name: manifest.root for manifest in catalog}
```

`SkillsShim` rejects being given both a `loader` and a `catalog`, since a catalog
already carries its own loader.

### Reading Active Skill Names

`SkillsShim.active_skill_names(run_state)` returns the skill names activated so
far in a run, in activation order. It is a convenience accessor over the
documented state contract: the shim writes active names under a `shim_state` key
formed from the shim `name` and
[`ACTIVE_SKILL_NAMES_STATE_KEY_SUFFIX`](../../src/agentlane/harness/_run.py),
which is also what `RunStateView.active_skill_names` reads. Prefer this accessor
(or the run-state view, for the union across multiple skills shims) over
hand-building the key:

```python
shim = SkillsShim(catalog=catalog)
# ... after a run ...
active = shim.active_skill_names(agent.run_state)  # ("report-generator", ...)
```

The accessor honors a custom shim `name`, so a shim configured with
`SkillsShim(name="docs")` reads the activation state it owns.

## Filesystem Loader

`FilesystemSkillLoader` is the default loader.

It discovers skill directories that contain `SKILL.md`, using local files by
default. Pass `reader=` to supply file reading and directory listing.

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

Local `SKILL.md` paths are normalized to absolute paths. With an injected
reader, `roots` default to `(".",)` in its storage namespace. Plain readers
use relative POSIX paths. A reader with `FilePathResolver` normalizes roots in
its own namespace; mounted roots and manifests use rooted virtual paths.
`include_default_roots` is ignored for injected readers; no local home or working
directory is inspected. Earlier roots win when skill names repeat.

To combine local and remote roots, share a `MountedReader` between the loader
and one read tool. Use rooted paths such as
`roots=("/tenant/skills", "/local/skills")`. Relative mounted roots also start
at `/`, independently of the read tool's cwd. See
[Mixed local and remote filesystems](filesystem.md#mixed-local-and-remote-filesystems)
for configuration and resource paths.

Manifest `root` and `skill_file` use `PurePath`: local discovery returns `Path`,
and injected discovery returns `PurePosixPath`. Resource `path` values stay
relative to the skill directory. Activation adds `read_path` by joining the
manifest root and resource path.

Discovery checks the immediate child directories of each root for `SKILL.md`.
It reads and caches the parsed metadata and instructions. Activation uses this
cached text and lists the current resource names without reading their content.
Changes to `SKILL.md` appear after another discovery call. Resource listing is
recursive, excludes the skill's own `SKILL.md`, and skips directory symlinks.
Local skill directories can themselves be symlinks; their manifest paths resolve
to the target directory.

Calls run on worker threads; configure storage timeouts and thread safety in
the reader. See
[File I/O interfaces](filesystem.md#permissions-and-execution).

### Filesystem Parsing Policy

The loader skips invalid skill files. Missing directory roots and roots that
are files contribute no skills. Other directory listing errors propagate.

It skips a skill entirely when:

1. the file cannot be read or is not valid UTF-8,
2. YAML frontmatter is missing or malformed,
3. frontmatter is not a mapping,
4. `name` is missing or empty,
5. `description` is missing or empty,
6. the file exceeds `SKILL_MAX_FILE_LINES`.

It silently tolerates softer issues, keeping the skill:

1. name-spec drift (a non-compliant or directory-mismatched name is kept as-is),
2. oversized `description` or `compatibility` values (truncated to their limit),
3. non-mapping metadata (ignored),
4. loose field types that can be coerced safely.

One malformed skill does not fail discovery or break the agent loop. The loader
skips that skill and continues with the rest.

### Frontmatter Schema

The filesystem parser reads the following `SKILL.md` frontmatter fields:

| Field             | Required | Notes                                                              |
| ----------------- | -------- | ------------------------------------------------------------------ |
| `name`            | yes      | Stable skill name; should match the skill directory name.          |
| `description`     | yes      | Truncated to `SKILL_MAX_DESCRIPTION_LENGTH`.                       |
| `license`         | no       | Optional license string.                                           |
| `compatibility`   | no       | Optional note; truncated to `SKILL_MAX_COMPATIBILITY_LENGTH`.      |
| `metadata`        | no       | Mapping; keys and values are coerced to strings.                   |
| `tools`           | no       | Replacement tool allowlist (see below).                            |
| `disallowedTools` | no       | Tool names removed before the model sees the active skill context. |

Unknown frontmatter keys are ignored.

## Tool Selection Metadata

Skills may include optional `tools` and `disallowedTools` frontmatter fields.
These fields control the model-visible tool pool after a skill is active. They
do not grant host permissions and cannot override an application's outer
permission policy.

```markdown
---
name: workspace-editor
description: Edit files in the active workspace.
tools: read, grep, write, patch
disallowedTools: bash
---
```

Tool selection follows deny-first replacement rules:

1. Missing `tools` inherits the current tool pool, subject to deny filters.
2. Present `tools` replaces the current tool pool with the declared names.
3. `disallowedTools` subtracts tool names before the model sees the active
   skill context.
4. If a tool is both in `tools` and `disallowedTools`, the deny rule wins.

Both fields accept a comma-separated string or a YAML list:

```markdown
---
name: workspace-reader
description: Read files in the active workspace.
tools:
  - read
  - grep
disallowedTools:
  - bash
  - write
---
```

There is no alternate allowlist field. `tools` is the framework's only
allowlist/replacement frontmatter field.

These fields filter model exposure. The host application's sandbox,
permissions, and approval callbacks still decide whether an exposed tool call
is allowed to execute.

`SkillsShim` applies these filters to the tools that are already present in the
prepared turn. A name listed in `disallowedTools` stays hidden from the model
even when the tool was contributed by another shim.

## State

Activated skill names are persisted in `RunState.shim_state`.

That gives three important properties:

1. repeated activation can be deduplicated,
2. later turns keep a cache-stable activation tool schema,
3. later turns in the same run continue without reloading the same skill
   instructions.

The actual skill content remains visible because the activation tool result is
already part of the persisted conversation history.

## Bundled Resource Paths

Installed skills often reference their own bundled files by paths relative to
the skill root, such as `references/policy.md` or `scripts/run.py`.

`SkillsShim` does not change tool path resolution. Activation renders a
`read_path` beside each skill-relative display path. For an injected reader
with a plain backend and the skill at `skills/refund-policy`, it emits:

```xml
<skill_resources>
  <file path="references/policy.md" read_path="skills/refund-policy/references/policy.md" />
</skill_resources>
```

For this plain backend, use `read_path` with a read tool that shares the
loader's reader and has `cwd="."`.

A mounted loader emits a rooted path, such as
`/local/skills/refund-policy/references/policy.md`. This path works with a
mounted read tool from any cwd, including `/tenant/sessions/session-123`.
References authored in the skill stay relative. The activation result gives
the skill directory and tells the model to resolve those references from it.
Generic file tools do not parse Markdown links or remember an active skill
as their cwd. Reading resources from two skills leaves subsequent
`write("notes.md")` calls relative to the tool's captured cwd.

For local skills, `read_path` is absolute and works with the default local
read tool:

```python
from agentlane.harness.skills import (
    SkillsShim,
    FilesystemSkillLoader,
)
from agentlane.harness.tools import HarnessToolsShim, read_tool

loader = FilesystemSkillLoader(roots=(SKILLS_ROOT,), include_default_roots=False)

descriptor = AgentDescriptor(
    name="Clinical Review",
    model=model,
    shims=(
        SkillsShim(loader=loader),
        HarnessToolsShim((read_tool(),)),
    ),
)
```

Use the rendered `read_path` attribute for both local and injected storage.
Host permission policies and approval callbacks decide whether the read tool
can open a resource. An injected reader does not make local shell or patch
tools operate on its storage.

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
That quickstart includes several clinical-case skills and a logging hook that
records agent lifecycle events and `activate_skill` calls during a real run.
