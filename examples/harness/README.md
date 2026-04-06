# Harness Examples

This category is reserved for runnable examples that exercise the agentic
harness directly.

## Available demos

1. [customer_support_conversation](./customer_support_conversation/): real OpenAI-backed multi-turn support conversation with templated instructions and `RunState` resume.
2. [tool_calling_search_answer](./tool_calling_search_answer/): real OpenAI-backed single-question demo showing the runner-owned tool loop with one mocked search result.

## Run

```bash
uv run python examples/harness/customer_support_conversation/main.py
uv run python examples/harness/tool_calling_search_answer/main.py
```
