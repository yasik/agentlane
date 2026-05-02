# Base Tools Quickstart

This example wires the first-party `bash`, `find`, `grep`, `patch`, `read`,
`write`, and `write_plan` tools through `HarnessToolsShim`, then runs the agent
with `DefaultAgent.run_stream(...)`.

It creates a temporary workspace, asks the model to create a finished portfolio
risk note plus a controls checklist, verify the workspace, and summarize the
result. The prompt describes the workspace outcome instead of an exact tool
sequence, so the model chooses how to use the tools. It must create a plan,
update progress while working, and mark the plan complete before the final
answer. The console streams model text and hook-based tool start/result logs
while the agent is working.

## Run

Install ripgrep first so the `rg` executable is available on `PATH`.

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/base_tools_quickstart/main.py
```
