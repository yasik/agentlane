# MCP Meeting Assistant

This example runs a `DefaultAgent` that searches meeting notes, reads the
matching record, and reports a launch decision. `MCPToolsShim` discovers the
two tools from a local stdio server built with the official MCP SDK. The
records in [`meetings.json`](./meetings.json) are synthetic.

## Run

From the repository root, set your OpenRouter key and a model that supports
tool calls in the ignored `.env.local` file:

```dotenv
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=your-tool-capable-model-id
```

The model ID can include or omit the `openrouter/` prefix.

```bash
uv run --env-file .env.local --extra mcp --extra litellm \
  python examples/harness/mcp_meeting_assistant/main.py
```

The script prints a JSON report with the observed `tool_calls` and the final
`answer`. It sets `verified` to `true` after checking that the agent called
`meetings__search_meetings`, then `meetings__get_meeting`, and returned the
source record's facts. The manager closes before the report is returned.

The report includes tool arguments and results. Adapt that output before using
private meeting data.

## Local test

Run the example with a deterministic test model and the real stdio MCP server,
without model credentials:

```bash
uv run --extra mcp --extra litellm \
  pytest tests/harness/mcp/test_meeting_example.py -s
```

See [Harness MCP Tools](../../../docs/harness/mcp.md) for remote servers,
authorization, and connection lifecycle.
