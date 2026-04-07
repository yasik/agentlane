# Predefined Handoff

This example shows a first-class handoff to a predefined specialist agent.

The model still sees the handoff as a tool choice, but the runner intercepts
it specially:

1. the parent assistant turn is preserved,
2. a transfer acknowledgement is added,
3. conversation history moves to the child agent, and
4. the child agent takes over and produces the final answer.

The caller does not resume after a handoff. Control is transferred.

## Why This Example Exists

The important part is that the parent does not call the specialist as a
subroutine. It transfers the conversation:

```python
descriptor = AgentDescriptor(
    name="Support Triage",
    tools=Tools(tools=[], tool_choice="required"),
    handoffs=(returns_specialist,),
)
```

The child specialist owns its own instructions. The handoff itself does not
invent a new system prompt for the child.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/handoff_to_returns_specialist/main.py
```
