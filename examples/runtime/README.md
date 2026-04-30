# Runtime Examples

This category is reserved for runtime-focused finance workflow demonstrations.

## Available demos

1. [multi_agent_workflow](./multi_agent_workflow/): trading analysis with direct send, publish fan-out, execution/risk workers, and one stateful in-order aggregator.
2. [distributed_publish_fan_in](./distributed_publish_fan_in/): distributed portfolio analysis with market-data/risk workers and one stateful aggregator agent for fan-in.
3. [distributed_scatter_gather](./distributed_scatter_gather/): distributed trade analysis with one coordinator aggregating position, valuation, and risk RPC responses.
4. [simple/](./simple/): stripped-down distributed finance runtime starters with plain `print(...)` output.

## Run

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
uv run python examples/runtime/distributed_publish_fan_in/main.py
uv run python examples/runtime/distributed_scatter_gather/main.py
uv run python examples/runtime/simple/distributed_publish_fan_in.py
uv run python examples/runtime/simple/distributed_scatter_gather.py
```
