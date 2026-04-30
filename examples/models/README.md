# Models Examples

This category is reserved for runnable examples that exercise the provider and
models layer directly, without the harness.

## Available demos

1. [openai_streaming_tool_reasoning](./openai_streaming_tool_reasoning/): OpenAI Responses API streaming in a portfolio risk chat loop with streamed reasoning, a short assistant preamble before a tool call, and an internal continuation after the tool result.
2. [claude_streaming_tool_thinking](./claude_streaming_tool_thinking/): Claude Sonnet 4.6 via LiteLLM in a clinical triage chat loop with streamed tool-call output, printed thinking blocks, and an internal continuation that preserves Anthropic thinking blocks with the tool result.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/models/openai_streaming_tool_reasoning/main.py
ANTHROPIC_API_KEY=... uv run python examples/models/claude_streaming_tool_thinking/main.py
```
