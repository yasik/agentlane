# Distributed Trade Analysis Scatter / Gather Demo

This example shows explicit distributed runtime topology for a trade analysis
request with:

1. one `WorkerAgentRuntimeHost`
2. four `WorkerAgentRuntime` instances
3. one coordinator agent that fans direct RPCs out to finance specialists
4. direct fan-in where the coordinator gathers all responses into one result

Message flow:

1. `main -> CoordinatorAgent` via direct RPC
2. `CoordinatorAgent -> PositionAgent + ValuationAgent + RiskAgent` via direct RPC
3. `CoordinatorAgent` waits for all outcomes with `asyncio.gather(...)`
4. `CoordinatorAgent` returns one merged `TradeAnalysis`

This demonstrates the other common fan-in pattern: one agent explicitly
aggregates responses from multiple peer agents rather than aggregating through
publish events.

## ASCII Flow

```text
main
  |
  | direct RPC recipient=CoordinatorAgent(request_id)
  v
[coordinator_worker]
  CoordinatorAgent
      |
      | direct RPCs started concurrently
      v
                   [host]
                /     |      \
               v      v       v
[position_worker] [valuation_worker] [risk_worker]
  PositionAgent     ValuationAgent     RiskAgent
               \      |       /
                \     |      /
                 v    v     v
                   [host]
                     |
                     | delivered outcomes
                     v
            [coordinator_worker]
              CoordinatorAgent
                     |
                     | asyncio.gather(...) merges responses
                     v
                    main
             (TradeAnalysis)
```

The host is drawn twice to separate outbound request routing from inbound
response delivery. In the real runtime both paths go through the same host
service.

## Run

```bash
uv run python examples/runtime/distributed_scatter_gather/main.py
```

## Optional Flags

```bash
uv run python examples/runtime/distributed_scatter_gather/main.py \
  --request-count 3
```
