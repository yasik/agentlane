# Default Agent Shims Quickstart

This example shows the new generic harness shim seam.

It keeps the core agent surface unchanged and adds behavior through
`AgentDescriptor.shims`.

The script defines two small shims:

1. one appends an extra instruction line before each model turn request,
2. one persists a completed-turn counter through the `ShimState` helper methods stored at `RunState.shim_state`.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_shims_quickstart/main.py
```

## What It Shows

1. simple one-class `Shim` definitions for the common path
2. `PreparedTurn` instruction mutation
3. persisted shim-owned state in the mapping-backed `ShimState` stored at `RunState.shim_state`
4. repeated `run(...)` calls continuing the same shim-aware conversation

## Important Detail

`PreparedTurn.instructions` starts from the agent descriptor on each model
turn. In this example, the extra `Support:` instruction is appended once per
turn request. It does not keep stacking up across later `run(...)` calls.
