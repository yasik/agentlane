# find Tool

`find_tool()` exposes a `find` tool for local file search by glob pattern.

Parameters:

1. `pattern: str`
2. `path: str | None = None`
3. `limit: int = 1000`

## Permissions

`find` resolves `path` through `ToolPathResolver` and checks
`ToolOperation.SEARCH_FILES` before directory validation or traversal. A denied
request returns:

```text
permission denied: find is not allowed for `/workspace/private`
```

The bundled side-effect approval policy does not request approval for searches.
A custom policy may still return `require_approval`, in which case the tool
uses the configured `approval_callback=`.

```text
approval required: find requires application approval for `/workspace` before execution
```

Example tool result:

```text
Search directory: /workspace
README.md
docs/notes.md
```

## Pattern semantics

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

## Search root, ordering, and traversal

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

## No-match and truncation

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
