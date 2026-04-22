# Default Agent Shims Quickstart

This example shows the new generic harness shim seam.

It keeps the core agent surface unchanged and adds behavior through
`AgentDescriptor.shims`.

The script defines two small shims:

1. one bootstraps the persisted system instruction before the first model turn,
2. one persists a completed-turn counter through the `ShimState` helper methods stored at `RunState.shim_state`.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_shims_quickstart/main.py
```

## What It Shows

1. simple one-class `Shim` definitions for the common path
2. explicit system-instruction mutation through `PreparedTurn.append_system_instruction(...)`
3. persisted shim-owned state in the mapping-backed `ShimState` stored at `RunState.shim_state`
4. repeated `run(...)` calls continuing the same shim-aware conversation

## Important Detail

`RunState.instructions` is the single persisted system instruction for the
conversation. In this example, the extra `Support:` line is appended once at
bootstrap time, before the first model turn. Later runs continue from that same
saved instruction without re-appending it.
