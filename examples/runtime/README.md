# Runtime Examples

This category is reserved for runtime-focused demonstrations.

## Available demos

1. [multi_agent_workflow](./multi_agent_workflow/): direct send + publish fan-out with one stateful in-order aggregator.
2. [distributed_publish_fan_in](./distributed_publish_fan_in/): explicit distributed host/workers with publish fan-out and one stateful aggregator agent for fan-in.
3. [distributed_scatter_gather](./distributed_scatter_gather/): explicit distributed host/workers with one coordinator agent aggregating multiple direct RPC responses.

## Run

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
uv run python examples/runtime/distributed_publish_fan_in/main.py
uv run python examples/runtime/distributed_scatter_gather/main.py
```
