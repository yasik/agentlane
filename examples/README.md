# Examples

AgentLane runnable demos are grouped by category.

## Categories

1. [`throughput/`](./throughput/) for high-load messaging demonstrations.
2. [`runtime/`](./runtime/) for lifecycle, scheduling, and distributed runtime demos.
3. [`harness/`](./harness/) for direct agentic-harness demonstrations.

## Quick Start

Run the high-throughput demo:

```bash
uv run python examples/throughput/high_throughput_messaging/main.py
```

Run the runtime multi-agent workflow demo:

```bash
uv run python examples/runtime/multi_agent_workflow/main.py
```

Run the harness customer-support demo:

```bash
uv run python examples/harness/customer_support_conversation/main.py
```

Run the distributed publish fan-out / fan-in demo:

```bash
uv run python examples/runtime/distributed_publish_fan_in/main.py
```

Run the distributed scatter / gather demo:

```bash
uv run python examples/runtime/distributed_scatter_gather/main.py
```

Run the simple distributed runtime starter examples:

```bash
uv run python examples/runtime/simple/distributed_publish_fan_in.py
uv run python examples/runtime/simple/distributed_scatter_gather.py
```
