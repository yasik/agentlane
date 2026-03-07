# Multi-Agent Workflow Demo

This runtime demo simulates a 4-agent workflow with both messaging patterns:

1. direct send (`IngressAgent -> PlannerAgent`)
2. publish fan-out (`PlannerAgent -> WorkerAAgent + WorkerBAgent`)
3. single stateful in-order aggregator (`AggregatorAgent`)

All agents print to console so message flow is easy to follow.

## Run

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
```

## Optional Flags

```bash
uv run python examples/runtime/multi_agent_workflow/main.py \
  --workflow-count 3 \
  --worker-count 8 \
  --timeout-seconds 20 \
  --aggregator-route-key global
```
