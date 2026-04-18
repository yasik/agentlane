# Default Agent Shims Quickstart

This example shows the new generic harness shim seam.

It keeps the core agent surface unchanged and adds behavior through
`AgentDescriptor.shims`.

The script defines two small shims:

1. one appends an extra instruction line before each model turn,
2. one persists a completed-turn counter in `RunState.shim_state`.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_shims_quickstart/main.py
```

## What It Shows

1. custom `HarnessShim` and `BoundHarnessShim` definitions
2. `PreparedTurn` instruction mutation
3. persisted shim-owned state in `RunState.shim_state`
4. repeated `run(...)` calls continuing the same shim-aware conversation
