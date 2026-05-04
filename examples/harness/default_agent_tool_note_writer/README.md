# Generic `agent` Tool

This example shows the default spawned-helper path in a portfolio operations
scenario.

The model sees one normal tool named `agent` with this schema:

1. `name`
2. `task`

When the model calls that tool, the runner validates those arguments into a
Pydantic model, creates a fresh child agent descriptor from that payload, and
sends the task to the child agent as its input.

## Why This Example Exists

This is the generic version of `agent-as-tool`:

```python
tools = Tools(
    tools=[],
    tool_choice="required",
    tool_call_limits={"agent": 1},
)
shims = (HarnessToolsShim((agent_tool(model=model),)),)
```

The manager does not predefine a specific child specialist. Instead, the model
chooses the helper `name` and the concrete `task`.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/default_agent_tool_note_writer/main.py
```
