# Claude Streaming Tool Thinking

This example uses `agentlane_litellm.Client.stream_response(...)` directly with
`anthropic/claude-sonnet-4-6` in an interactive clinical triage chat loop.

It demonstrates:

1. streamed LiteLLM chunk events,
2. printed Claude thinking output and thinking blocks,
3. a streamed tool call,
4. manual local tool execution, and
5. internal continuation that resends the assistant tool-call message together
   with the tool result so one user message produces one complete assistant
   turn.

The second turn also preserves `thinking_blocks` on the assistant message. That
matches Anthropic's documented tool-use continuation requirements for extended
thinking. The terminal still pauses only at user input boundaries, so the demo
behaves like a normal chat session.

The example uses `tool_choice="auto"` with a strong instruction to call the
tool. Anthropic rejects forced tool choice when extended thinking is enabled.
It also sets `max_tokens` higher than `thinking.budget_tokens`, which Anthropic
requires for extended thinking requests.

Run:

```bash
ANTHROPIC_API_KEY=... uv run python examples/models/claude_streaming_tool_thinking/main.py
```

Then type a question such as:

```text
A patient reports dizziness and two glucose readings in the 60s after starting a new diabetes medication. Look up the triage protocol first.
```

Type `exit` to quit.
