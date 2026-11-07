# Process execution

`ProcessExecutor` runs an executable with an argument vector and byte streams.
`LocalProcessExecutor` starts that executable directly. Arguments are passed as
separate values, without shell parsing, expansion, or startup files.

The executor owns the execution directory, environment, streams, timeout, and
process cleanup. File storage and application permission checks remain with the
calling tool. Applications can implement the same contract for another execution
environment. A failed executor does not select another backend automatically.

## Local execution

Import the public contracts from `agentlane.harness.tools`:

```python
import sys
from pathlib import Path

from agentlane.harness.tools import (
    LocalProcessExecutor,
    ProcessExecutionRequest,
    ProcessStopReason,
)
from agentlane.runtime import CancellationToken

executor = LocalProcessExecutor(default_timeout=5)
request = ProcessExecutionRequest(
    argv=(
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
    ),
    cwd=str(Path.cwd()),
    stdin=b"input bytes\n",
    stdout_capture_limit=1024,
)
result = await executor.run(request, CancellationToken())
assert result.stop_reason == ProcessStopReason.EXITED
assert result.exit_code == 0
assert result.stdout_complete
assert result.stdout == b"input bytes\n"
```

Use the example inside an async function. A custom executor implements
`async run(request, cancellation_token) -> ProcessExecutionResult`. Binary stdin
and requested bounded stdout capture are part of this contract.

## Request

`ProcessExecutionRequest` has these fields:

| Field | Default | Behavior |
| --- | --- | --- |
| `argv: tuple[str, ...]` | Required | Executable followed by literal arguments; the executable must be nonempty and values cannot contain NUL |
| `cwd: str` | Required | Working directory in the executor's filesystem |
| `stdin: bytes \| None` | `b""` | Send the bytes and EOF; `None` inherits input in the execution environment |
| `timeout_seconds: float \| None` | `None` | Positive, finite override; `None` uses the executor default |
| `env: Mapping[str, str] \| None` | `None` | Complete replacement environment; `None` uses the executor default |
| `stdout_capture_limit: int \| None` | `None` | Positive byte limit for raw stdout; `None` selects diagnostic output only |

The backend resolves executable names and validates its working directory.
Callers must not resolve a remote cwd against the host filesystem. An environment
mapping replaces the environment; an empty mapping remains empty.

`LocalProcessExecutor(default_timeout=None, default_env=None)` has no default
deadline and inherits the local environment. Supply `default_env=` to configure
a replacement environment. A request's explicit `env` takes precedence over
that default. Explicit mappings are copied before use so subsequent caller
changes cannot change a running request. Native execution does not update
`PWD`, `SHLVL`, or `_` to imitate a shell.

The execution deadline covers the process and pipe transfer. It does not cover
permission waits, source storage reads, or a destination write. Cleanup can
extend beyond the execution deadline.

## Result

`ProcessExecutionResult` describes a settled process:

| Field | Meaning |
| --- | --- |
| `exit_code` | Known child exit code after the owned process is reaped |
| `stop_reason` | `EXITED`, `TIMEOUT`, `CANCELLED`, or `STDOUT_LIMIT` from `ProcessStopReason` |
| `timeout_seconds` | Effective execution timeout |
| `stdout` | Captured raw stdout, or `None` when capture was not requested |
| `stdout_complete` | Capture reached EOF without discarded stdout bytes |
| `output` | Bounded combined stdout/stderr preview as `TruncatedOutput` |
| `log_path` | Optional diagnostic log path in the executor's filesystem |
| `log_complete` | Whether the log contains all diagnostic output read through stream EOF |

Exactly reaching the capture limit is valid if stdout then reaches EOF. Another
byte stops execution and makes the capture incomplete. Empty captured output is
`b""`. With capture disabled, `stdout` is `None` and `stdout_complete` is false.
Use the stop reason, exit code, and completeness together before treating stdout
as a successful complete result.

The preview keeps at most the last 2,000 lines or 51,200 bytes, controlled by
`PROCESS_MAX_LINES` and `PROCESS_MAX_BYTES`. Combined output follows the observed
order of asynchronous stream reads. Raw stdout capture excludes stderr and log
headers. The result does not expose a separate raw stderr payload.

Preview truncation, raw capture completeness, and log completeness are separate.
A full diagnostic log can exceed the preview limit. The host owns log retention
and access; a storage reader cannot automatically read an executor-local log.

## Cancellation and cleanup

The local executor drains stdout and stderr while sending stdin. On timeout,
token cancellation, or capture overflow, it terminates the owned process group
and escalates if graceful termination does not finish. It settles stream tasks
and pipe cleanup before returning a result.

Startup and transport failures raise exceptions. Token cancellation before
startup raises `asyncio.CancelledError`, because no child exit code exists.
After a child starts, cooperative token cancellation can return `CANCELLED`.
Task cancellation settles cleanup and propagates `asyncio.CancelledError`,
including when cancellation is repeated during cleanup.

The local executor is not a sandbox. The executable can access its environment
and filesystem according to the operating system's permissions. Applications
own isolation, resource policy, and admission of process tools.

The existing [Bash executor](./tools/bash.md) uses the same local process
lifecycle and retains its shell-command API. Native clients use the process
contract directly.
