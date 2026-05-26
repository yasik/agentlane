# read Tool

`read_tool()` exposes a `read` tool for UTF-8 text files.

Parameters:

1. `path: str`
2. `offset: int | None = None`
3. `limit: int | None = None`

## Permissions

`read` resolves `path` through `ToolPathResolver` and checks
`ToolOperation.READ_FILE` before opening the file. A denied request returns:

```text
permission denied: read is not allowed for `/workspace/private.txt`
```

The bundled side-effect approval policy does not request approval for reads.
A custom policy may still return `require_approval`, in which case the tool
uses the configured `approval_callback=`.

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

If the first requested line exceeds the byte cap by itself, the result reports:

```text
[Line 1 is 51201 bytes, exceeds 51200 byte limit. Use bash to inspect it.]
```

The tool returns clear text errors for directories, missing files, likely binary
files, invalid offsets, invalid limits, and unreadable paths. Invalid UTF-8 byte
sequences are decoded with replacement characters so the model can still use
the surrounding text.
