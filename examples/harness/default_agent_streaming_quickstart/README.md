# Default Agent Streaming Quickstart

This example shows the smallest high-level streaming surface in the harness:
`agentlane.harness.agents.DefaultAgent.run_stream(...)`. The scenario is a
portfolio risk review that looks up a mocked risk policy before answering.

It demonstrates:

1. one `DefaultAgent` subclass with a normal `AgentDescriptor`,
2. one mocked search tool,
3. live text deltas from the model,
4. streamed reasoning summaries, when the provider emits them, plus OpenAI
   phase / preamble details,
5. streamed tool-call argument deltas, and
6. final `RunResult` access after the stream completes.

Run:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/default_agent_streaming_quickstart/main.py
```
