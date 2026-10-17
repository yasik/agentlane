# OpenAI async tools

This example starts two synthetic order lookups while OpenAI continues to
write packing guidance. It uses HTTP streaming and registers one function
through `Tools`, with its name selected in `ResponsesClient(async_tools=...)`.
The harness owns tool execution and result delivery.

## Run

Set `OPENAI_API_KEY` in your environment, then run from the repository root:

```bash
uv run python examples/harness/openai_async_tools/main.py
```

The default model is `gpt-6-astra`. Set `OPENAI_MODEL` to select another model
that supports native async functions. The account must have access to that
model and feature.

## What to observe

The stock lookup takes four seconds; delivery takes eight seconds. These
delays simulate two services without external service dependencies. The model
writes packing guidance while the lookups run, ends that response, and uses
the results in later turns. Ending the first response lets the harness deliver
ready results without waiting for further model output.

The script prints text and tool timestamps, then reports `verified: true`
only when all of these checks pass:

- Two native async call IDs were observed, and each lookup executed once.
- The two tool executions overlapped.
- A tool started before the first model response finished.
- Model text arrived while a tool was still running.
- The final JSON answer contains both source values from the tools.

This checks native async events as well as concurrent Python execution.
The run has a 90-second timeout and closes the stream on failure. A model
that does not emit async calls or overlapping text fails verification.

The output reports measured timings; model latency and wording vary between
runs. See [steering and async tools](../../../docs/models/overview.md#steering-and-async-tools)
for cancellation, tool policy, and continuation rules.
