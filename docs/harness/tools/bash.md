# bash

`bash_tool()` exposes a `bash` tool for bounded non-interactive shell commands.

Parameters:

1. `command: str`
2. `timeout: float | None = None`

## Permissions

`bash` checks `ToolOperation.EXECUTE_COMMAND` before the executor starts. A
workspace policy denies command execution unless it is explicitly included,
and `workspace_tool_policy(require_bash_approval=True)` makes execution
approval-required. A denied request returns:

```text
permission denied: bash is not allowed
```

An approval-required request returns:

```text
approval required: bash command requires application approval before execution
```

Commands run through `bash -lc` in the `cwd` captured when the tool is
constructed. The result is the combined stdout/stderr output in terminal
arrival order.

Example tool result:

```text
/workspace
total 8
drwxr-xr-x  3 user  staff   96 Apr 27 09:00 .
drwxr-xr-x  5 user  staff  160 Apr 27 09:00 ..
-rw-r--r--  1 user  staff   18 Apr 27 09:00 notes.txt
```

`bash_tool(default_timeout=...)` sets the default timeout on its locally created
executor for calls that omit `timeout`. A supplied executor owns its default;
the tool passes the model's timeout unchanged, including `None`. A model call
can override the executor default with a positive per-call timeout. Invalid
empty commands and non-positive timeouts return stable text errors before any
process starts.

Host applications can import the executor-facing contracts
`BashExecutor`, `StdinBashExecutor`, `StdoutBashExecutor`, `LocalBashExecutor`,
`BashExecutionRequest`, `BashExecutionResult`, `BashShellConfig`, and
`resolve_bash_shell` from `agentlane.harness.tools` when they need to wrap or
replace local execution.

`BashExecutionRequest.shell_startup` defaults to `True`, which preserves normal
Bash startup behavior. Set it to `False` to disable startup files and inherited
shell functions and options. Executors must honor this setting or reject the
request before execution. The local executor uses `--noprofile`, `--norc`, and
`-p` while preserving the configured executable, explicit shell options,
environment, and default timeout.

`BashExecutionRequest.stdin` accepts binary input from host code. The local
executor sends those bytes and closes stdin; empty bytes send EOF, while
`None` preserves inherited stdin. This field is not a model-facing Bash
argument. Executors that support it implement the optional, runtime-checkable
`StdinBashExecutor` protocol and declare `supports_stdin = True`. Other
executors must reject supplied input.

Host code can set `BashExecutionRequest.stdout_capture_limit` to a positive byte
limit to capture raw stdout separately from the combined display output. This
requires `StdoutBashExecutor` with `supports_stdout_capture = True`. The local
executor stops execution on overflow and returns the retained bytes in
`BashExecutionResult.stdout`. `stdout_truncated` is true on overflow or if
stdout did not reach EOF. Callers must check this flag and the command outcome
before using the bytes as a complete result. With capture disabled, `stdout`
is `None`. These fields do not change the model-facing Bash arguments.

`bash_tool(permissions=...)` evaluates the shared permission policy before the
executor starts. Pair it with `approval_callback=` when `require_approval`
decisions should be resolved by a host-controlled prompt.

Empty successful commands return `(no output)`. Non-zero exits, timeouts,
cancellations, and truncation add short bracketed notices after the output:

```text
before failure

[Command exited with code 7]
```

For timed-out, cancelled, and non-zero-exit outcomes the tool returns a
[`ToolFailure`](../runner.md#tool-success-and-failure) instead of a plain string.
`ToolFailure` is a `str` subclass, so the model-facing text is unchanged, while
the runner reads the attached `ToolError` and reports `ok=False` (with a stable
`kind` of `timeout`, `cancelled`, or `nonzero_exit`) on the `RunToolEndEvent`.
Consumers read that envelope directly rather than reflecting over
`BashExecutionResult` fields.

If display output is truncated, the result includes a temporary log path for
the combined output. The notice says `Partial output` when `stdout_truncated`
is true; otherwise it says `Full output`. On timeout or cancellation, the tool
terminates the process group and kills it if graceful termination does not
finish promptly. On POSIX, both graceful and forced termination target the
process group. On Windows, graceful termination sends `CTRL_BREAK_EVENT` to the
new process group when
available, then falls back to leader-only termination; forced termination uses
`taskkill /F /T` for the process tree. The tool is intentionally
non-interactive: it does not stream partial output to the model and does not
accept follow-up stdin for a running command. The default local executor is
not a process sandbox; hosts with strict data boundaries should provide an
executor that controls execution and full-output log storage.

## Process Workspace

Bash operates on files visible to its executing process. Pass the workspace
through `cwd=`. A host can provide `executor=` directly or `bash_executor=` in
`base_harness_tools()` to run commands in a sandbox or another process
environment. That executor owns directory validation and process isolation;
the tool does not check the remote directory on the local machine. An explicit
`cwd` passes to a custom executor without host symlink resolution or tilde
expansion. The executor resolves relative paths. When `cwd` is omitted, the
tool captures the host current directory at construction.

Filesystem readers and logical mounts are independent of process mounts.
Bash never obtains remote access through a reader or writer. The host must
expose data to the process when commands need it. Grep runs in the harness
process environment, so run both tool handlers there when they must share a
sandbox workspace. See [File I/O interfaces](../filesystem.md).
