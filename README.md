# AgentLane

`agentlane` is a UV workspace monorepo for an AI agent runtime and messaging framework.

## Development Model

This repository keeps two parallel layouts:

1. `src/agentlane` is the **primary development surface** (OpenAI SDK style `src/...` workflow).
2. `packages/*` stays configured as a UV workspace for later extraction/modularization.

## Repository Structure

```text
agentlane/
├── src/agentlane/                    # active development (src-first)
│   ├── agents/
│   ├── extensions/
│   ├── handoffs/
│   ├── mcp/
│   ├── memory/
│   ├── messaging/
│   ├── models/
│   ├── runtime/
│   ├── tracing/
│   └── util/
├── tests/                            # tests for src/agentlane
├── packages/                         # workspace packages kept configured
│   ├── contracts/
│   ├── runtime/
│   ├── agents/
│   ├── transport/
│   ├── tracing/
│   └── cli/
├── docs/
├── examples/
└── pyproject.toml
```

## Quickstart

```bash
make sync
make format
make lint
make tests
```
