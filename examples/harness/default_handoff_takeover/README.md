# Default Handoff

This example shows the generic `handoff` path in a patient triage scenario.

Unlike a predefined handoff target, the parent does not point at one named
child descriptor ahead of time. Instead, the runner creates a fresh child
descriptor from `DefaultHandoff(...)` when the model calls `handoff`.

The transfer semantics are the same as any other first-class handoff:

1. conversation history moves to the child,
2. the trigger turn is preserved,
3. the child agent takes over, and
4. the parent does not resume.

## Why This Example Exists

The configuration is still small:

```python
descriptor = AgentDescriptor(
    name="Patient Triage",
    tools=Tools(tools=[], tool_choice="required"),
    default_handoff=DefaultHandoff(
        model=model,
        instructions="You are the clinical escalation specialist.",
    ),
)
```

`DefaultHandoff(...)` is the transfer version of a generic spawned helper.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/default_handoff_takeover/main.py
```
