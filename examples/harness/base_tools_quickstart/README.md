# Base Tools Quickstart

This example wires the first-party `find`, `grep`, `read`, and `write` tools
through `HarnessToolsShim`.

It creates a temporary workspace, asks the model to write a portfolio risk note,
find it, search inside it with `grep`, read it back, and summarize, then prints
the assistant answer plus the observed tool calls.

## Run

Install ripgrep first so the `rg` executable is available on `PATH`.

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/base_tools_quickstart/main.py
```
