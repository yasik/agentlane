# Default Agent Quickstart

This example shows the highest-level local harness path:

1. define one `DefaultAgent` subclass,
2. set one `AgentDescriptor`,
3. call `run(...)`.

The script does not create a runtime, a runner, an `AgentId`, or call
`send_message(...)` directly. `DefaultAgent` provisions the default local
runtime and the default runner automatically.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_quickstart/main.py
```

## What It Shows

1. class-based `DefaultAgent` definition
2. templated instructions with `PromptSpec`
3. repeated `run(...)` calls continuing the same conversation
4. persisted `RunState` without manual lifecycle wiring
