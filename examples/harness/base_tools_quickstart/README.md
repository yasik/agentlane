# Base Tools Quickstart

This example wires the first-party `find`, `read`, and `write` tools through
`HarnessToolsShim`.

It creates a temporary workspace, asks the model to write a file, find it,
read it back, and summarize, then prints the assistant answer plus the
observed tool calls.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/base_tools_quickstart/main.py
```
