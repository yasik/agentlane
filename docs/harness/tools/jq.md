# jq

`jq_tool()` exposes a `jq` tool that selects and transforms JSON from one file.
The executor must have the `jq` executable on `PATH`.

## Parameters

| Parameter | Default | jq behavior |
| --- | --- | --- |
| `path: str` | Required | One literal file path; wildcards are not expanded |
| `filter: str` | Required | jq program, such as `.items[] \| {id, name}` |
| `output_path: str \| None` | `None` | Create or replace this file with complete stdout after success; use `path` to edit the input |
| `raw_output: bool` | `False` | `--raw-output`: write strings without JSON quotes |
| `slurp: bool` | `False` | `--slurp`: collect input values into one array |
| `compact_output: bool` | `False` | `--compact-output`: write compact JSON |
| `timeout: float \| None` | Executor default | Positive, finite execution deadline in seconds |

Pass the filter as jq source, without a `jq` command prefix or shell quoting:

```json
{
  "path": "results.json",
  "filter": ".items[] | select(.active) | {id, name}"
}
```

The tool uses the installed jq implementation. It accepts a stream of JSON
values, including JSONL, and uses jq's normal pretty output by default.
`null`, `false`, and an empty result are successful outcomes; the tool does not
enable `--exit-status`. Malformed input is reported by jq and is not repaired.
For filters and data types, see the [jq manual](https://jqlang.org/manual/).

Only the options in the table are exposed. The tool fixes monochrome output
and sends the selected file's unchanged bytes to stdin. Thus `input_filename`
returns `<stdin>`. It does not add MCP-specific parsing. Without `output_path`,
the selected file remains unchanged.

To edit the source file, use the same input and output path:

```json
{
  "path": "results.json",
  "filter": ".processed = true | del(.debug)",
  "output_path": "results.json"
}
```

Use a different `output_path` to save a copy. Shell startup is disabled, and the
saved content is only the exact stdout from jq, including trailing newlines.
Stderr and display notices are excluded. Raw output, multiple values, and empty
output keep their jq semantics; the saved file can contain plain text, a JSON
stream, or zero bytes.

## Storage and execution

`cwd=` and `reader=` follow the [read tool](./read.md) path rules. The default
reads local files. An injected `FileReader` can read mounted or remote files.
The tool buffers at most `max_input_bytes=8 * 1024 * 1024` bytes before execution;
a larger file returns an error. The reader owns storage timeouts and retries.

Local defaults also support saving. With an injected reader, supply
`writer=` as a `WritableFileSystem` in the same logical namespace. A reader
without a writer supports queries and rejects saves. The tool never substitutes
a local writer for injected storage. `base_harness_tools()` supplies its writer
to jq and can infer that writer from a `WritableFileSystem` reader.

Saving buffers at most `max_output_bytes=8 * 1024 * 1024` bytes of stdout.
The local executor stops the command if this limit is exceeded. A failed,
timed-out, cancelled, or incomplete jq result is not written. The output limit
does not change query preview limits. Set larger host limits explicitly when
needed.

Each tool instance serializes saving calls from the input read through the
settled write. Other tool instances and external writers require coordination
by the host or backend. As with [write](./write.md), a started write must settle
before task cancellation propagates. Cancellation during commit does not undo
that write.

`process_cwd=` controls the executor directory, including relative jq imports.
It defaults to the file cwd for local files and to the captured host cwd for
injected storage. An explicit directory for a custom executor passes through
without host path resolution. Storage paths and process paths remain separate.
Local execution reports a missing process directory or a path that is not a
directory before reading the input or starting jq. Custom executors validate
their own working directory.

```python
from agentlane.harness.tools import HarnessToolsShim, jq_tool

shims = (
    HarnessToolsShim((
        jq_tool(
            reader=filesystem,
            writer=filesystem,
            cwd="/workspace",
            process_cwd="/tmp",
        ),
    )),
)
```

The default executor is `LocalBashExecutor`. It runs a shell-quoted jq command
with binary stdin. Every execution sets `BashExecutionRequest.shell_startup`
to `False`. Executors must disable startup files and inherited shell functions
and options, or reject the request before execution.

`executor=` accepts an implementation of the optional, runtime-checkable
`StdinBashExecutor` protocol with `supports_stdin = True`.
It must send `BashExecutionRequest.stdin` and close the input stream. An
unsupported executor returns a tool failure; jq never falls back to local
execution. `base_harness_tools(bash_executor=...)` uses that executor for both
bash and jq. The [SPACE tool bundle](../sandboxes/space.md) currently excludes jq.

Saving also requires `StdoutBashExecutor` with `supports_stdout_capture = True`.
The request sets `stdout_capture_limit`; the result supplies raw `stdout` and
`stdout_truncated`. A capture is incomplete if it exceeds the limit or does
not reach EOF, even if the command's leader exits successfully. Jq rejects
incomplete capture before opening the writer.

`default_timeout=None` leaves the locally created executor without a default
deadline. Set a positive value to configure that executor. A supplied executor
owns its default timeout; the tool passes the model's `timeout` unchanged,
including `None`. This matches [bash](./bash.md). The deadline covers command
execution, after the input read. The local executor terminates the process
group on timeout or cancellation.

## Permissions

Before opening the input, jq checks `ToolOperation.READ_FILE` for the resolved
path and `ToolOperation.EXECUTE_COMMAND` for the generated command and process
cwd. A denied file request returns:

```text
permission denied: jq is not allowed for `/workspace/private.json`
```

A denied command request returns `permission denied: jq is not allowed`.
When `output_path` is supplied, jq also checks `CREATE_FILE` or `OVERWRITE_FILE`
for the destination, and `CREATE_DIRECTORY` if its parent is missing. These
checks use the same metadata and permission rules as [write](./write.md) and
finish before reading the input or starting the command.

An approval-required decision uses the configured `approval_callback=`. A
whole-tool `jq` grant includes all five operations. A read grant alone does not
admit jq execution or saving.

Ordinary jq expressions can read process environment variables and import
process files. The default executor is not filesystem-confined. Path checks
on the selected input do not restrict imports. Applications that require
isolation must supply a suitable executor or run the tool in a sandbox.

## Results

Results use the [bash output contract](./bash.md): combined stdout and stderr,
with at most the last 2,000 lines or 51,200 bytes. Empty successful output is
`(no output)`, and trailing newlines are removed. Truncation adds a notice and
the log path in the executor filesystem. The notice says `Partial output` when
raw stdout capture is incomplete; otherwise it says `Full output`. Injected
readers cannot automatically access that path. Narrow the filter for a smaller
result; the host can also provide access to the log. A truncated preview can
contain incomplete JSON. The host owns log retention.

A successful save starts with `Wrote N bytes to path.`, followed by the same
bounded command output. Preview truncation does not truncate the saved file.

Non-zero exits, timeouts, and cancellation results returned by the executor
carry `ToolFailure` metadata while retaining their readable output. For example,
a malformed final JSONL record can produce earlier results followed by a
failure. The runner reports that call as failed. Missing files, input-size or
save-output limits, and permission errors also return tool failures.
Cancellation before execution or during the input read, and task cancellation,
propagate `asyncio.CancelledError`.

Run the [local and mounted JSON example](../../../examples/harness/jq_json/README.md)
without a model or API key.
