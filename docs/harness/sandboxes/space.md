# SPACE

Use `agentlane_space` to connect AgentLane tools to a running
[SPACE sandbox](https://space.docs.perplexity.ai/guide/introduction.html).
Your app creates, manages, and deletes the sandbox.

## How it works

`space_harness_tools()` builds a sandbox version of the
[`base_harness_tools()` bundle](../tools.md#standard-set). It keeps
the standard tool names and argument schemas and changes where the
environment tools run:

| Tools | Execution |
| --- | --- |
| `read`, `find`, `grep`, `patch`, `write` | A guest Python worker runs the original AgentLane file handlers |
| `bash` | The core Bash tool uses `SpaceBashExecutor` to run commands in the guest |
| `write_plan`, `agent` | The host harness retains the core definitions |

Pass this bundle to `HarnessToolsShim` in place of the local base bundle.
The shim registers the tools; their handlers determine where execution
occurs. Registering both bundles causes duplicate tool names. Passing
`SpaceBashExecutor` through `base_harness_tools(bash_executor=...)` changes
only Bash execution.

The sandbox and guest user permissions form the security boundary. `cwd`
sets the working directory; it does not restrict file access. Guest file
operations do not call host permission callbacks.

## Configure the connection

Run the host client on Linux or macOS. The `agentlane` wheel includes
`agentlane_space`. Prepare the guest with these requirements:

- Python 3.12 with the same AgentLane build as the host, including the extension
- `/bin/bash`, `rg`, and `tee`
- An absolute, writable working directory, such as `/workspace`
- A writable `/tmp` directory for Bash output logs

Set these environment variables for the usage example:

| Variable | Value |
| --- | --- |
| `SPACE_API_URL` | Management endpoint |
| `SPACE_DATA_PLANE_URL` | Event endpoint |
| `SPACE_TOKEN` | Management bearer token |
| `SPACE_SANDBOX_ID` | ID of the running sandbox |

Endpoints require HTTPS; local deployments can use loopback HTTP. A bare
origin gets a `/v1` suffix. Include the API prefix if your deployment uses
one. Credentials stay on the host.

`SpaceSandbox` uses `python3` in the guest by default. Set
`python_executable=` to select another interpreter and `user=` to select the
guest user for checks and tool calls. `default_timeout=` sets the command
deadline in seconds; the default is 60. A Bash `timeout` argument overrides
that deadline.

## Use the tools

Call `check()` for the guest working directory before you construct the
tools. It verifies the guest dependencies and compares source fingerprints;
matching version numbers alone are insufficient. Keep the connection open
for the full agent run.

Use your app's configured AgentLane `model` in this example, inside an async
function:

```python
import os

from agentlane.harness import AgentDescriptor
from agentlane.harness.agents import DefaultAgent
from agentlane.harness.tools import HarnessToolsShim
from agentlane_space import SpaceSandbox, space_harness_tools

async with SpaceSandbox(
    api_url=os.environ["SPACE_API_URL"],
    data_plane_url=os.environ["SPACE_DATA_PLANE_URL"],
    token=lambda: os.environ["SPACE_TOKEN"],
    sandbox_id=os.environ["SPACE_SANDBOX_ID"],
) as sandbox:
    await sandbox.check(cwd="/workspace")
    agent = DefaultAgent(
        descriptor=AgentDescriptor(
            name="Sandbox assistant",
            model=model,
            shims=(
                HarnessToolsShim(
                    space_harness_tools(sandbox, cwd="/workspace"),
                ),
            ),
        ),
    )
    result = await agent.run("List the files in the workspace.")
    print(result.final_output)
```

The token function runs for each management request. Your app owns token
refresh. You can also pass a token string. Closing the context closes the
HTTP client; your app must stop or delete the sandbox separately.

Inside the open sandbox context, use `include=` or `exclude=` when you
construct the bundle to select tools:

```python
tools = space_harness_tools(
    sandbox,
    cwd="/workspace",
    include=("read", "find", "grep"),
)
```

File-tool JSON requests and Bash command strings have a 48 KiB UTF-8 input
limit. Split larger writes or patches into smaller operations. Run
foreground, non-interactive shell commands. Bash saves full output in guest
files under `/tmp/agentlane-bash-*.log`; your app owns log cleanup.

To check all six environment tools without a model, run the
[connection example](../../../examples/harness/space_sandbox.py) from a checkout:

```bash
uv run python examples/harness/space_sandbox.py
```

It uses the connection variables, `/workspace`, and `python3`. Set
`SPACE_CWD` and `SPACE_PYTHON` to change those defaults. It removes its
temporary test file on success; your app owns cleanup after failure.

## Recover after a failure

The extension does not automatically retry tools or switch to host execution.
If `sandbox.needs_recovery` is true, the handle blocks further commands.
Use `sandbox.last_command` to identify the command, then inspect the sandbox
and affected files before repeating an operation. After your app completes
recovery, create a fresh handle and call `check()` again.
