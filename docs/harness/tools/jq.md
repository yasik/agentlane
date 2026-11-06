# jq

`jq_tool()` exposes a `jq` tool that selects and transforms JSON from one file.
The executor must have the `jq` executable on `PATH`.

## Parameters

| Parameter | Default | jq behavior |
| --- | --- | --- |
| `path: str` | Required | One literal file path; wildcards are not expanded |
| `filter: str` | Required | jq program, such as `.items[] | {id, name}` |
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
returns `<stdin>`. It does not add MCP-specific parsing or change the source
file.

## Storage and execution

`cwd=` and `reader=` follow the [read tool](./read.md) path rules. The default
reads local files. An injected `FileReader` can read mounted or remote files.
The tool buffers at most `max_input_bytes=8 * 1024 * 1024` bytes before execution;
a larger file returns an error. The reader owns storage timeouts and retries.

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
        jq_tool(reader=filesystem, cwd="/workspace", process_cwd="/tmp"),
    )),
)
```

The default executor is `LocalBashExecutor`. It runs a shell-quoted jq command
with binary stdin. `executor=` accepts an implementation of the optional,
runtime-checkable `StdinBashExecutor` protocol with `supports_stdin = True`.
It must send `BashExecutionRequest.stdin` and close the input stream. An
unsupported executor returns a tool failure; jq never falls back to local
execution. `base_harness_tools(bash_executor=...)` uses that executor for both
bash and jq. The [SPACE tool bundle](../sandboxes/space.md) currently excludes jq.

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
An approval-required decision uses the configured `approval_callback=`. A
whole-tool `jq` grant includes both operations. A read grant alone does not
admit jq execution.

Ordinary jq expressions can read process environment variables and import
process files. The default executor is not filesystem-confined. Path checks
on the selected input do not restrict imports. Applications that require
isolation must supply a suitable executor or run the tool in a sandbox.

## Results

Results use the [bash output contract](./bash.md): combined stdout and stderr,
with at most the last 2,000 lines or 51,200 bytes. Empty successful output is
`(no output)`, and trailing newlines are removed. Truncation adds a notice and
the full-output log path in the executor filesystem. Injected readers cannot
automatically access that path. Narrow the filter for a smaller result; the
host can also provide access to the log. A truncated preview can contain
incomplete JSON. The host owns log retention.

Non-zero exits, timeouts, and cancellation results returned by the executor
carry `ToolFailure` metadata while retaining their readable output. For example,
a malformed final JSONL record can produce earlier results followed by a
failure. The runner reports that call as failed. Missing files, input-size
limits, and permission errors also return tool failures. Cancellation before
execution or during the input read, and task cancellation, propagate
`asyncio.CancelledError`.

Run the [local and mounted JSON example](../../../examples/harness/jq_json/README.md)
without a model or API key.
