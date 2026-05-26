# bash Tool

`bash_tool()` exposes a `bash` tool for bounded non-interactive shell commands.

Parameters:

1. `command: str`
2. `timeout: float | None = None`

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

`bash_tool(default_timeout=...)` sets a construction-time default timeout for
calls that omit `timeout`. A model call can override it with a positive
per-call timeout. Invalid empty commands and non-positive timeouts return
stable text errors before any process starts.

Host applications can import the executor-facing contracts
`BashExecutor`, `LocalBashExecutor`, `BashExecutionRequest`,
`BashExecutionResult`, `BashShellConfig`, and `resolve_bash_shell` from
`agentlane.harness.tools` when they need to wrap or replace local execution.

`bash_tool(permissions=...)` evaluates the shared permission policy before the
executor starts. Pair it with `approval_callback=` when `require_approval`
decisions should be resolved by a host-controlled prompt.

Empty successful commands return `(no output)`. Non-zero exits, timeouts,
cancellations, and truncation add short bracketed notices after the output:

```text
before failure

[Command exited with code 7]
```

If output is truncated, the result includes a temporary log path with the full
combined output. On timeout or cancellation, the tool terminates the process
group and kills it if graceful termination does not finish promptly. On POSIX,
both graceful and forced termination target the process group. On Windows,
graceful termination sends `CTRL_BREAK_EVENT` to the new process group when
available, then falls back to leader-only termination; forced termination uses
`taskkill /F /T` for the process tree. The tool is intentionally
non-interactive: it does not stream partial output to the model and does not
accept follow-up stdin for a running command. The default local executor is
not a process sandbox; hosts with strict data boundaries should provide an
executor that controls execution and full-output log storage.
