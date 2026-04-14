# Streaming Escalation Flow

This example shows one streamed harness run that exercises three mechanics in
sequence:

1. a normal tool call,
2. a predefined agent-as-tool delegation, and
3. a first-class handoff.

The example uses `agentlane.harness.agents.DefaultAgent.run_stream(...)` and a
real OpenAI model. It prints reasoning summaries, OpenAI phase changes, streamed
tool-call arguments, and the final specialist answer after handoff.

One important detail in the current implementation:

1. first-class handoff keeps the outer stream going because control transfers,
2. agent-as-tool remains internal, so the child agent's own model events are
   not surfaced separately on the outer stream yet.

Run:

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/streaming_escalation_flow/main.py
```
