# Distributed Scatter / Gather Demo

This example shows explicit distributed runtime topology with:

1. one `WorkerAgentRuntimeHost`
2. four `WorkerAgentRuntime` instances
3. one coordinator agent that fans direct RPCs out to specialist agents
4. direct fan-in where the coordinator gathers all responses into one result

Message flow:

1. `main -> CoordinatorAgent` via direct RPC
2. `CoordinatorAgent -> InventoryAgent + PricingAgent + ShippingAgent` via direct RPC
3. `CoordinatorAgent` waits for all outcomes with `asyncio.gather(...)`
4. `CoordinatorAgent` returns one merged `AggregatedQuote`

This demonstrates the other common fan-in pattern: one agent explicitly
aggregates responses from multiple peer agents rather than aggregating through
publish events.

## Run

```bash
uv run python examples/runtime/distributed_scatter_gather/main.py
```

## Optional Flags

```bash
uv run python examples/runtime/distributed_scatter_gather/main.py \
  --request-count 3
```
