# Distributed Publish Fan-In Demo

This example shows explicit distributed runtime topology with:

1. one `WorkerAgentRuntimeHost`
2. five `WorkerAgentRuntime` instances
3. publish fan-out from planner to two specialist workers
4. fan-in into one stateful aggregator agent keyed by workflow id

Message flow:

1. `IngressAgent -> PlannerAgent` via direct RPC
2. `PlannerAgent -> InventoryWorkerAgent + PricingWorkerAgent` via publish
3. `InventoryWorkerAgent + PricingWorkerAgent -> AggregatorAgent` via publish
4. `AggregatorAgent` merges both results and resolves external completion

The aggregator demonstrates the key fan-in pattern: one agent instance collects
multiple result events and produces one merged workflow summary.

## Run

```bash
uv run python examples/runtime/distributed_publish_fan_in/main.py
```

## Optional Flags

```bash
uv run python examples/runtime/distributed_publish_fan_in/main.py \
  --workflow-count 4 \
  --timeout-seconds 15
```
