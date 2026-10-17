# OpenAI steering

This example changes a trip request while OpenAI is generating a response.
It uses `DefaultAgent.run_stream()` and the stream's `steer()` method over
the native SDK WebSocket connection.

## Run

Set `OPENAI_API_KEY` in your environment, then run from the repository root:

```bash
uv run python examples/harness/openai_steering/main.py
```

The default model is `gpt-6-astra`. Set `OPENAI_MODEL` to select another model
that supports steering. The account must have access to that model and feature.

## What to observe

The initial request asks for a Tokyo itinerary. After `response.created`, the
example submits one update that changes the destination to Kyoto. It prints
the receipt transitions and the final answer.

`accepted` means the provider queued the update. The example reports
`verified: true` only after the receipt reaches `applied`, the update appears
once in committed history, and the final answer starts with Kyoto. The report
includes the original and successor response IDs.

The run has a 90-second timeout and closes the stream on failure. Provider
rejections and failed verification exit with an error; they are not treated
as successful steering.

See [steering and async tools](../../../docs/models/overview.md#steering-and-async-tools)
for the adapter's lifecycle and replay rules.
