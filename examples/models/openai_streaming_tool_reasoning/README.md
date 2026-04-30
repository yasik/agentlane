# OpenAI Streaming Tool Reasoning

This example uses `agentlane_openai.ResponsesClient.stream_response(...)`
directly in an interactive chat loop for a portfolio risk policy lookup.

It demonstrates:

1. streamed OpenAI reasoning output,
2. a short assistant preamble before the tool call,
3. streamed function-call argument deltas,
4. manual local tool execution, and
5. internal continuation after the tool result so one user message produces one
   complete assistant turn.

The continuation stays local so the example runs cleanly in organizations that
use Zero Data Retention and do not allow `previous_response_id`. That internal
continuation is hidden behind the chat loop, so the terminal still behaves like
a normal chat session.

Run:

```bash
OPENAI_API_KEY=sk-... uv run python examples/models/openai_streaming_tool_reasoning/main.py
```

Then type a question such as:

```text
The portfolio is 42% semiconductors and 12% leveraged ETFs. Please look up the risk policy before answering.
```

Type `exit` to quit.
