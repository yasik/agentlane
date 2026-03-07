# High Throughput Messaging Demo

This example stress-tests AgentLane messaging using both:

1. RPC-style direct sends (`send_message`)
2. event fan-out publishes (`publish_message`)

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
2. Rolling per-interval rates for RPC send/deliver and publish enqueue.
3. Final summary with totals and aggregate throughput.

Note:
If `publish_failed` grows quickly, reduce publish pressure by increasing `--publish-pause-seconds` or lowering `--publish-concurrency`.
