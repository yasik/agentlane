# Harness Markdown Agent Definitions

`agents.definitions` lets you define an agent in a Claude-Code-style markdown file — YAML
frontmatter plus a system-prompt body — and turn it into an
[`AgentDescriptor`](./agents.md) or a runnable
[`DefaultAgent`](./default-agents.md) without constructing one in code.


## Entry Points

```python
from pathlib import Path

from agentlane.harness.agents import DefaultAgent
from agentlane.harness.agents.definitions import descriptor_from_markdown

# A descriptor you can compose, inspect, or wire as a sub-agent.
descriptor = descriptor_from_markdown(Path("agents/chart_reviewer.md"))

# A runnable agent (requires a model — see Model Resolution).
agent = DefaultAgent.from_markdown(Path("agents/chart_reviewer.md"), model=model)
```

## Import Path

```python
from agentlane.harness.agents.definitions import (
    AGENT_MAX_DESCRIPTION_LENGTH,
    AGENT_MAX_INSTRUCTIONS_LINES,
    AGENT_MAX_SUBAGENT_DEPTH,
    AgentFileError,
    AgentManifest,
    FactoryModelResolver,
    ModelResolver,
    ParsedAgentFile,
    SubagentLink,
    descriptor_from_markdown,
    parse_agent_file,
    resolve_tool_config,
)
```

The static tool-exclusion shim that backs `disallowedTools` is the shared
`ExcludeToolsShim` from `agentlane.harness.shims`, not a feature-local type.

## Frontmatter Schema

Everything after the closing `---` is the Markdown body and becomes the agent's
`instructions` (its system prompt). The frontmatter maps onto descriptor fields:

| Field             | Required | Maps to                  | Notes                                                                                              |
| ----------------- | -------- | ------------------------ | -------------------------------------------------------------------------------------------------- |
| `name`            | no       | `name`                   | Identity comes from this field, not the filename. Blank/absent → generated fallback name.          |
| `description`     | no       | `description`            | Used as the tool/handoff description when this agent is a sub-agent.                                |
| `model`           | no       | resolved client          | Provider/model spec (`anthropic/<m>`, `openai/<m>`, `azure/<m>`, …). `inherit`/omitted → no model.  |
| `model_args`      | no       | `model_args`             | Free-form mapping forwarded verbatim as request kwargs (temperature, reasoning_effort, …).         |
| `tools`           | no       | `tools` (`ToolConfig`)   | Allowlist of native AgentLane tool names. Omitted → inherit-all; `[]` → no tools.                  |
| `disallowedTools` | no       | an `ExcludeToolsShim` in `shims` | Denylist removed from the visible tool set each turn.                                           |

Tool names are AgentLane's native lowercase names (`read`, `find`, `grep`,
`patch`, `write`, `write_plan`, `bash`, `agent`); custom/MCP tool names the
parent exposes also work. Both `tools` and `disallowedTools` accept a comma-separated string or a YAML list.
Unknown frontmatter keys are ignored.

### Example `AGENT.md`

```markdown
---
name: chart-reviewer
description: Reviews a patient chart for medication-safety flags. Use before clinician sign-off.
model: anthropic/claude-sonnet-4-5
model_args:
  temperature: 0.2
  max_tokens: 4096
tools: [read, grep]
disallowedTools: write
---
You are a meticulous clinical chart reviewer. Read the chart, surface
medication-safety flags first, then note guideline gaps. Be concise and cite the
section you used.
```

## Constants

Limits used by the parser:

1. `AGENT_MAX_DESCRIPTION_LENGTH` (`1024`) — agent description limit; 
2. an over-length description is truncated to it,
2. `AGENT_MAX_INSTRUCTIONS_LINES` (`1000`) — a longer instruction body is
   truncated to this many lines with a pointer back to the source file, rather
   than the agent being dropped,
3. `AGENT_MAX_SUBAGENT_DEPTH` (`4`) — load-time sub-agent nesting cap, matching
   the runner's default `agent_max_depth`.

## Model Resolution

`AgentDescriptor.model` is a live model client, not a string, and credentials
are deliberately not in the file — so a frontmatter `model` spec is turned into
a client through an injected `ModelResolver`:

```python
class ModelResolver(Protocol):
    def resolve(self, model_spec: str, *, model_args: dict[str, Any]) -> Model[ModelResponse]: ...
```

The built-in `FactoryModelResolver` adapts any AgentLane `Factory` (which owns
credentials and provider routing):

```python
from agentlane.harness.agents.definitions import FactoryModelResolver

resolver = FactoryModelResolver(factory=my_factory)
agent = DefaultAgent.from_markdown(Path("agents/chart_reviewer.md"), model_resolver=resolver)
```

Rules:

1. `model: inherit` or an omitted `model` → the descriptor's model stays `None`.
   A sub-agent then inherits its parent's model at runtime.
2. A `model` spec with no resolver → the descriptor's model stays `None`. It is
   not resolved here; a top-level agent without a model is rejected at the
   `DefaultAgent.from_markdown` boundary instead.
3. `model_args` is attached only to `descriptor.model_args` (forwarded by the
   runner as `extra_call_args`); it is never folded into the client. A child's
   `model_args` are independent of the parent's — the child's values win.
4. `DefaultAgent.from_markdown` enforces that the **root** agent resolves to a
   model: the model comes from the explicit `model=` argument, or from the
   frontmatter spec via `model_resolver`. If neither yields a model, it raises
   `AgentFileError` — an agent cannot run without one. `descriptor_from_markdown`
   does not apply this check, so it can build sub-agent descriptors that inherit.

Core ships only the protocol and the factory adapter; it never imports the
optional provider packages. The host supplies the `Factory` (and credentials),
so the same `.md` runs in dev/staging/prod by swapping the injected factory.

## Tool Resolution

`tools`/`disallowedTools` produce native tool policy:

1. `tools` omitted → `INHERIT_TOOLS` (inherit the parent's tools),
2. `tools` non-empty → `RESTRICT_TOOLS.only(*names)` (allowlist over inherited tools),
3. `tools: []` → `OVERRIDE_TOOLS` (expose no tools),
4. `disallowedTools` non-empty → an `ExcludeToolsShim` in `shims` that removes the
   denied names from the visible set on every turn (re-applied so a later shim
   cannot silently re-add a denied tool).

The markdown produces only name-based policy. The actual tool instances (with
`cwd`, permissions, approval callbacks) come from the host's tools — for example
a `HarnessToolsShim` built with `base_harness_tools(...)`. Names not present in
the visible set are tolerated, which is correct for custom and MCP tool names.

## Sub-agents

`subagents=` is the one way to attach a sub-agent — `DefaultAgent` does the
`as_tool()` conversion for you. It is available wherever you build a
`DefaultAgent`:

- `DefaultAgent.from_markdown(..., subagents=[...])` and the module-level
  `descriptor_from_markdown(..., subagents=[...])` accept markdown **paths** or
  `AgentDescriptor` values (paths are resolved with the same `model_resolver`).
- `DefaultAgent(descriptor=..., subagents=[...])` accepts `AgentDescriptor`
  values for fully programmatic agents (no markdown). Pass a built agent's
  `resolved_descriptor` to reuse it.

```python
# From markdown:
agent = DefaultAgent.from_markdown(
    Path("agents/triage_lead.md"),
    model_resolver=resolver,
    subagents=[Path("agents/med_safety.md"), Path("agents/guidelines.md")],
)

# Programmatic:
agent = DefaultAgent(
    descriptor=triage_lead,
    subagents=[med_safety, guidelines],
)
```

`subagent_link` selects how they attach:

1. `SubagentLink.AS_TOOL` (default) — each sub-agent becomes an agent-as-tool;
   the parent calls it, gets the result back, and continues,
2. `SubagentLink.HANDOFF` — each sub-agent becomes a first-class handoff target
   (control transfers to the child).

For fine-grained control, `tools=[child.as_tool(...)]` on the descriptor remains
an alternative — `as_tool` takes an explicit name, description, and args model.

There is no inline `subagents:` frontmatter key: one file describes one agent,
and trees are composed by the caller. Attaching sub-agents is guarded against two
sub-agents whose names normalize to the same delegation tool name; markdown
loading additionally guards against cycles and nesting deeper than
`AGENT_MAX_SUBAGENT_DEPTH` — all raising `AgentFileError`.

## Scope and Limitations

- **`disallowedTools` (and `tools`) scope to the declaring agent only — they do
  not cascade to sub-agents.** Each agent owns its own tool policy.
  A parent's `disallowedTools: bash` does not stop
  an attached sub-agent that inherits tools from reaching `bash`; constrain the
  child by giving its own file a `tools` allowlist or `disallowedTools` entry.
  The deny-list is a per-turn model-visibility filter (`ExcludeToolsShim`), not a
  runtime authorization boundary — tool permissions/approvals remain the real
  authz control. Naming a sub-agent's own delegation tool in `disallowedTools`
  will hide that sub-agent.
- **One `subagent_link` per call.** `subagent_link` applies uniformly to every
  entry in `subagents`. To mix attachment styles (one child as a tool, another
  as a handoff), compose `AgentDescriptor` values directly rather than loading a
  single mixed tree from one call.
- **Not every descriptor field is expressible in frontmatter.** The frontmatter
  covers `name`, `description`, `model`, `model_args`, `tools`, and
  `disallowedTools`. Fields such as the structured-output `schema` and
  `default_handoff` are Python-level and must be set programmatically on the
  returned `AgentDescriptor`.

## Validation

Two tiers, and the parser does not log:

1. `parse_agent_file(path)` returns `None` only when the file has no parseable
   frontmatter (missing/unterminated fence, invalid YAML, or non-mapping
   frontmatter). It does not catch read errors — a missing or unreadable file
   raises `OSError` so the cause is visible rather than swallowed into a silent
   `None`. An oversized instruction body is truncated with a pointer back to the
   source file, never dropped; conflicting `model_args` (such as `temperature`
   with `reasoning_effort`) pass through and surface at the model call.
2. The `from_markdown` entry points and `descriptor_from_markdown(...)` raise on
   caller-facing failures: `OSError`/`FileNotFoundError` propagating from a
   missing path, and `AgentFileError` (a `ValueError` subclass) for an
   existing-but-unparseable file, a sub-agent cycle, depth-cap violation, or a
   root agent that resolves to no model.

## Relationship to Skills

`agents.definitions` and [Harness Skills](./skills.md) share the same frontmatter parsing
helpers and follow the same warn-and-skip parser policy. Skills expose a library
the model activates mid-run; `agents.definitions` defines whole agents (and sub-agents)
up front from files.
