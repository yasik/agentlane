# Base Tools Plan Quickstart

This example shows the first-party `plan` tool exposed through
`HarnessToolsShim`.

The agent creates a tracked plan for a portfolio concentration review, updates
it in a later run, and prints the latest serialized plan stored in
`RunState.shim_state`.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/base_tools_plan_quickstart/main.py
```

## What It Shows

1. exposing `plan_tool()` through `HarnessToolsShim`
2. forcing one plan call per run with `Tools(tool_choice="required")`
3. replacing the previous plan with each successful tool call
4. persisting the latest plan under `harness-tools:plan`
