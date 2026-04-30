# High Throughput Market Workflow Demo

This example stress-tests AgentLane messaging for market workflows using both:

1. execution-risk RPC-style direct sends (`send_message`)
2. market-data fan-out publishes (`publish_message`)

It streams progress to stdout while running.

## Run

```bash
uv run python examples/throughput/high_throughput_messaging/main.py
```

## Tunable Flags

```bash
uv run python examples/throughput/high_throughput_messaging/main.py \
  --duration-seconds 10 \
  --rpc-concurrency 8 \
  --publish-concurrency 8 \
  --worker-count 16 \
  --shard-count 64 \
  --progress-interval-seconds 1.0 \
  --publish-pause-seconds 0.002
```

## What You Should See

1. Startup configuration banner.
2. Rolling per-interval rates for risk RPC send/deliver and market-data publish enqueue.
3. Final summary with totals and aggregate throughput.

Note:
If `publish_failed` grows quickly, reduce publish pressure by increasing `--publish-pause-seconds` or lowering `--publish-concurrency`.
