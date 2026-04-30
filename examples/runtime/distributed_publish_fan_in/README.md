# Distributed Portfolio Analysis Publish Fan-In Demo

This example shows explicit distributed runtime topology for portfolio analysis
with:

1. one `WorkerAgentRuntimeHost`
2. five `WorkerAgentRuntime` instances
3. publish fan-out from planner to market-data and risk workers
4. fan-in into one stateful aggregator agent keyed by analysis id

Message flow:

1. `IngressAgent -> PlannerAgent` via direct RPC
2. `PlannerAgent -> MarketDataWorkerAgent + RiskWorkerAgent` via publish
3. `MarketDataWorkerAgent + RiskWorkerAgent -> AggregatorAgent` via publish
4. `AggregatorAgent` merges both results and resolves external completion

The aggregator demonstrates the key fan-in pattern: one agent instance collects
multiple result events and produces one merged portfolio analysis summary.

## ASCII Flow

```text
main
  |
  | direct RPC recipient=IngressAgent(workflow_id)
  v
[ingress_worker]
  IngressAgent
      |
      | direct RPC recipient=PlannerAgent("planner")
      v
    [host]
      |
      v
[planner_worker]
  PlannerAgent
      |
      | publish PLAN_TOPIC_TYPE(route_key=workflow_id)
      v
    [host]
    /    \
   v      v
[market_data_worker]      [risk_worker]
  MarketDataWorkerAgent    RiskWorkerAgent
           \              /
            \            /
             v          v
               [host]
                 |
                 | publish RESULT_TOPIC_TYPE(route_key=workflow_id)
                 v
        [aggregator_worker]
          AggregatorAgent(key=workflow_id)
                 |
                 | tracker.complete(summary)
                 v
          CompletionTracker
                 |
                 v
                main
```

The host appears more than once in the diagram for readability. In the real
cluster it is one service, and every cross-worker hop routes through it.

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
