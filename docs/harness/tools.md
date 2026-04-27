# Harness Tools

`agentlane.harness.tools` provides first-party harness tool definitions for
common local workspace actions. Each helper returns a
`HarnessToolDefinition`, which wraps an executable `agentlane.models.Tool` plus
optional prompt metadata for `HarnessToolsShim`.

## Import Path

```python
from agentlane.harness.tools import (
    HarnessToolsShim,
    find_tool,
    grep_tool,
    plan_tool,
    read_tool,
    write_tool,
)
```

## Tool Definitions

Tool helpers return definitions, not raw model tools:

```python
definition = write_tool(cwd=WORKSPACE)
tool = definition.tool
```

Use the definition when you want prompt snippets and guidelines to be rendered
by `HarnessToolsShim`. Use `definition.tool` when you only need the executable
`Tool` value.

## HarnessToolsShim

`HarnessToolsShim` merges executable tools into each prepared turn and appends
the definitions' prompt metadata to the first turn's system instructions:

```python
from agentlane.harness import AgentDescriptor
from agentlane.harness.tools import (
    HarnessToolsShim,
    find_tool,
    grep_tool,
    read_tool,
    write_tool,
)
from agentlane.models import Tools

descriptor = AgentDescriptor(
    name="Workspace Agent",
    model=model,
    instructions="Use workspace tools before answering workspace questions.",
    tools=Tools(
        tools=[],
        tool_call_limits={"find": 1, "grep": 1, "read": 1, "write": 1},
    ),
    shims=(
        HarnessToolsShim(
            (
                read_tool(cwd=WORKSPACE),
                find_tool(cwd=WORKSPACE),
                grep_tool(cwd=WORKSPACE),
                write_tool(cwd=WORKSPACE),
            )
        ),
    ),
)
```

`base_harness_tools()` returns the current standard tool set. It contains
`read`, `find`, `grep`, `write`, and `update_plan`.

## Path Policy

Filesystem tools use `ToolPathResolver`. Relative paths resolve against the
`cwd` captured when the tool is constructed. Absolute paths are allowed in
the current implementation. Paths are normalized with
`Path.resolve(strict=False)`.

The first-party path resolver does not enforce a sandbox boundary, permission
allowlist, or approval workflow.

## Output Limits

Text output is capped at shared deterministic limits. `read` output is capped
at 2000 lines or 51200 bytes, whichever limit is reached first. `find` output
is capped at 1000 matching paths or 51200 bytes, whichever limit is reached
first. `grep` output is capped at 100 matching entries or 51200 bytes,
whichever limit is reached first.

Caller-provided limits are applied before the global caps. For large files, call
`read` repeatedly with `offset` and `limit`. For large search results, narrow
the `find` pattern or search path.

## read

`read_tool()` exposes a `read` tool for UTF-8 text files.

Parameters:

1. `path: str`
2. `offset: int | None = None`
3. `limit: int | None = None`

Example tool result:

```text
Absolute path: /workspace/notes.txt
L1: alpha
L2: bravo
L3: charlie
```

When more lines remain after a caller limit or global line cap, the result adds
a continuation note:

```text
More than 2000 lines found
```

When the byte cap is reached, the result reports:

```text
Output truncated after 51200 bytes
```

The tool returns clear text errors for directories, missing files, likely binary
files, invalid offsets, invalid limits, and unreadable paths. Invalid UTF-8 byte
sequences are decoded with replacement characters so the model can still use
the surrounding text.

## find

`find_tool()` exposes a `find` tool for local file search by glob pattern.

Parameters:

1. `pattern: str`
2. `path: str | None = None`
3. `limit: int = 1000`

Example tool result:

```text
Search directory: /workspace
README.md
docs/notes.md
```

### Pattern semantics

Patterns use path-aware glob matching backed by
[`wcmatch.glob`](https://facelessuser.github.io/wcmatch/glob/). The matcher is
compiled with `GLOBSTAR | DOTMATCH | BRACE | IGNORECASE | FORCEUNIX`, which
means:

- `**` matches zero or more directory segments. Use `**/` for recursive
  matches (`*.py` is *not* recursive — it matches only top-level files).
- `{a,b}` brace expansion is supported (`**/*.{ts,tsx}`).
- Matching is **case-insensitive** on every platform. This keeps results
  consistent across Linux (case-sensitive) and macOS / Windows
  (case-insensitive) filesystems.
- Dotfiles (`.env`, `.gitignore`, etc.) are included unless ignored.
- Only files are returned, never directories.
- Leading `./` and `/` are stripped from the pattern, so `/src/*.py` and
  `./src/*.py` both behave like `src/*.py`.

Examples:

```text
**/*.py
**/*.{ts,tsx}
src/**/*.spec.ts
```

### Search root, ordering, and traversal

By default `path` is the configured `cwd`. If `path` is provided, output paths
are relative to that search directory.

Results are sorted by **modification time, newest first**, with ties broken
alphabetically. This mirrors the ordering used by editor file pickers and is
the most useful default for "what changed recently?" queries.

Symlinked directories are **not** followed during traversal. This avoids
cycles and prevents pattern matching from escaping the search directory
through symlinks.

`find` respects `.gitignore` files from the search root up to the nearest
repository boundary and always skips `.git/`.

### No-match and truncation

When no files match, the result includes the resolved search directory:

```text
Search directory: /workspace
No files matched.
```

When more files match than the caller-provided `limit` (and `limit` is below
the 1000 maximum), the result reports the total count and how to recover:

```text
N files matched; returned first <limit>. Refine the pattern or raise `limit` (max 1000).
```

When the caller-provided `limit` is at or above the 1000 maximum and there are
still more matches, the result tells the model the cap was hit:

```text
N files matched; returned first 1000 (maximum). Refine the pattern or narrow `path`.
```

When the byte cap is reached, the result reports:

```text
Output truncated at 51200 bytes; refine the pattern or narrow `path`.
```

The tool returns clear text errors for empty patterns, empty paths, invalid
limits, and paths that do not resolve to a directory.

## grep

`grep_tool()` exposes a `grep` tool for searching local UTF-8 text files. It is
backed by `ripgrepy`, so the `rg` executable from ripgrep must be available on
`PATH`.

Parameters:

1. `pattern: str`
2. `path: str | None = None`
3. `outputMode: "content" | "files_with_matches" | "count" = "content"`
4. `glob: str | None = None`
5. `type: str | None = None`
6. `ignoreCase: bool = False`
7. `literal: bool = False`
8. `multiline: bool = False`
9. `context: int = 0`
10. `limit: int = 100`

`pattern` is a ripgrep regular expression by default. Set `literal=True` to
search for exact text. Set `ignoreCase=True` for case-insensitive matching.
Set `multiline=True` to allow patterns to span newlines (`.` also matches
newlines).

`path` may point to one file or one directory. When omitted, grep searches the
tool's configured `cwd`. The result header and rendered match paths are
relative to that `cwd` when possible.

### Output modes

`content` (default) returns matching lines:

```text
Search path: .
src/app.py:12:def main() -> None:
src/app.py:20:main()
```

`files_with_matches` returns one path per matching file — the cheapest mode for
discovering where a symbol lives:

```text
Search path: .
src/app.py
src/lib.py
```

`count` returns `path:N` rows with the number of matches per file:

```text
Search path: .
src/app.py:2
src/lib.py:5
```

### Filtering

`type` filters candidates by ripgrep file-type name (e.g. `py`, `js`, `rust`).
Prefer it over `glob` when one of ripgrep's predefined types fits — it is more
compact and intent-clear.

`glob` filters candidates by glob. Directory searches use ripgrep glob syntax
(supports `**` and `!`-negation); explicit file paths fall back to `fnmatch`,
which does not support `**` or negation.

### Context and limits

When `context > 0` (content mode only), grep includes that many lines before
and after each match. Overlapping context groups are merged and distinct groups
are separated with:

```text
--
```

`limit` caps the number of returned entries: matching lines in `content`,
matching files in `files_with_matches`, and rows in `count`. When the limit is
reached and more entries remain, the result appends:

```text
Showing first 100 matches; more remain.
```

```text
Showing first 100 files; more remain.
```

When the byte cap is reached, the result reports:

```text
Output truncated after 51200 bytes
```

Long matching lines (content mode) are capped at 500 characters and reported
with:

```text
One or more matching lines were truncated after 500 characters
```

### Errors and edge cases

Grep respects `.gitignore`, searches hidden files except `.git/`, and relies on
ripgrep's binary-file filtering when searching directories. Directory searches
are best effort: binary-file warnings are ignored so matches from other text
files can still be returned. If the requested path is a binary file, grep
returns a clear text error. Invalid regular expressions, invalid globs, invalid
file types, missing paths, empty inputs, invalid contexts, invalid limits,
missing ripgrep, and unreadable explicit file paths also return clear text
errors.

## write

`write_tool()` exposes a `write` tool for creating or overwriting UTF-8 text
files.

Parameters:

1. `path: str`
2. `content: str`

Example tool result:

```text
Wrote 128 bytes to /workspace/notes.txt.
```

The tool creates parent directories automatically. Existing files are replaced
through a sibling temporary file where practical.

Use `write` for new files or complete rewrites. It does not provide append mode
or precise patch operations.

The tool returns clear text errors for empty paths, paths containing null bytes,
directory targets, parent paths that are files, invalid UTF-8 content,
permission failures, and other failed writes. Unexpected implementation errors
return a stable generic failure message so the agent loop can continue.

## plan

`plan_tool()` exposes an `update_plan` tool for creating or replacing the current
task plan.

Parameters:

1. `explanation: str | None = None`
2. `plan: list[PlanItem]`

Each plan item has:

1. `step: str`
2. `status: "pending" | "in_progress" | "completed"`

Each call replaces the previous plan. Partial item updates are intentionally
not part of the Phase 11 contract. The model should keep at most one item
`in_progress`.

Successful model-facing tool result:

```text
Plan updated
```

The plan payload itself is intended for clients and shims to render. The tool
does not echo the full checklist back to the model after a successful update.
Malformed arguments are rejected by the normal tool argument validation path.

When used through `HarnessToolsShim`, the latest plan update is persisted in
`RunState.shim_state` under `harness-tools:plan` for the default shim name.
Custom shim names use the same pattern: `{shim_name}:plan`.
