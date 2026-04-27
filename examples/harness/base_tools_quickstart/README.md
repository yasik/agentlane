# Base Tools Quickstart

This example wires the first-party `read` tool through `HarnessToolsShim`.

It creates a temporary workspace file, exposes only `read`, asks the model to
inspect that file, and prints the assistant answer plus the observed tool call.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/base_tools_quickstart/main.py
```
