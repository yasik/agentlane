# Base Tools Quickstart

This example wires the first-party `read` and `write` tools through
`HarnessToolsShim`.

It creates a temporary workspace, asks the model to write a file and read it
back, and prints the assistant answer plus the observed tool calls.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/base_tools_quickstart/main.py
```
