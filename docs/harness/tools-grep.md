# grep Tool

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

## Permissions

`grep` resolves `path` through `ToolPathResolver` and checks
`ToolOperation.READ_FILE` when the resolved path is an existing file, otherwise
`ToolOperation.SEARCH_FILES`. That means missing paths use the search
operation before the later missing-path error. A denied request returns:

```text
permission denied: grep is not allowed for `/workspace/private`
```

The bundled side-effect approval policy does not request approval for grep. A
custom policy may still return `require_approval`, in which case the tool uses
the configured `approval_callback=`.

```text
approval required: grep requires application approval for `/workspace` before execution
```

## Output modes

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

## Filtering

`type` filters candidates by ripgrep file-type name (e.g. `py`, `js`, `rust`).
Prefer it over `glob` when one of ripgrep's predefined types fits — it is more
compact and intent-clear.

`glob` filters candidates by glob. Directory searches use ripgrep glob syntax
(supports `**` and `!`-negation); explicit file paths fall back to `fnmatch`,
which does not support `**` or negation.

## Context and limits

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

Both `files_with_matches` and `count` modes share the same overflow message
(`count` does not emit a distinct rows-oriented notice):

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

## Errors and edge cases

Grep respects `.gitignore`, searches hidden files except `.git/`, and relies on
ripgrep's binary-file filtering when searching directories. Directory searches
are best effort: binary-file warnings are ignored so matches from other text
files can still be returned. If the requested path is a binary file, grep
returns a clear text error. Invalid regular expressions, invalid globs, invalid
file types, missing paths, empty inputs, invalid contexts, invalid limits,
missing ripgrep, and unreadable explicit file paths also return clear text
errors.
