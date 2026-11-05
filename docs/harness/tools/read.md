# read

`read_tool()` exposes a `read` tool for UTF-8 text files.

Parameters:

1. `path: str`
2. `offset: int | None = None`
3. `limit: int | None = None`
4. `column: int | None = None`

`offset` is a one-based line number. `column` is a one-based Unicode character
position within that line. Both default to 1. The column applies only to the
first requested line; later lines start at their first character. File content
is limited to 2,000 lines or 51,200 bytes per call.

At construction, pass `reader=` to read from application storage. The native
schema, text limits, and result format stay the same. The reader receives
paths in its logical namespace; `cwd` defaults to the storage root. Mounted
readers accept rooted virtual paths and resolve relative paths from `cwd`.
Plain injected readers retain relative POSIX paths. Omit `reader` for local
files and the process cwd captured at construction. See [File I/O adapters](../filesystem.md) for the stream contract and example.

## Permissions

For local files, `read` resolves `path` through `ToolPathResolver`. An injected
reader uses its normalized logical paths. Both modes check
`ToolOperation.READ_FILE` before opening the file. A denied request returns:

```text
permission denied: read is not allowed for `/workspace/private.txt`
```

The bundled side-effect approval policy does not request approval for reads.
A custom policy may still return `require_approval`, in which case the tool
uses the configured `approval_callback=`.

```text
approval required: read requires application approval for `/workspace/notes.txt` before execution
```

Example tool result:

```text
alpha
bravo
charlie
```

When more lines remain after a caller limit or global line cap, the result adds
a continuation note:

```text
alpha
bravo

[Showing lines 1-2. Use offset=3 to continue.]
```

When the byte cap is reached, the result reports:

```text
[Showing lines 1-128 (51200 byte limit). Use offset=129 to continue.]
```

If the first requested line exceeds the byte cap, the result returns the part
that fits without splitting a UTF-8 character, followed by a continuation note.
For an ASCII line, the note is:

```text
[Showing line 1, columns 1-51200 (51200 byte limit). Use offset=1, column=51201 to continue.]
```

Follow both values to read the rest of the same line. Concatenate parts of that
line without adding a newline. Once the line is complete, the reader continues
with later lines and uses the usual line continuation notes. The file itself
is unchanged.

The tool returns clear text errors for directories, missing files, likely binary
files, invalid offsets, invalid columns, invalid limits, and unreadable paths. A file is flagged
as binary when a NUL byte appears in its first 4096 bytes (not a full-file
scan). Invalid UTF-8 byte sequences are decoded with replacement characters so
the model can still use the surrounding text.
