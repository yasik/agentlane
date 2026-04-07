# Predefined Agent-As-Tool

This example shows the simplest predefined `agent-as-tool` flow:

1. define a child `AgentDescriptor`,
2. expose it to the parent with `.as_tool(...)`,
3. give that tool an explicit Pydantic args model, and
4. let the runner route the validated payload to the child agent through runtime messaging.

The important idea is that the model sees this exactly like a normal tool call.
The only difference is where the call executes: instead of running an in-process
function, the runner sends the validated arguments to another agent.

## Why This Example Exists

The core harness surface is small:

```python
class WarrantyLookupArgs(BaseModel):
    product_name: str
    issue_summary: str


policy_specialist = AgentDescriptor(
    name="Policy Specialist",
    model=model,
    instructions="You are Acme's warranty policy specialist.",
)


tools = Tools(
    tools=[
        policy_specialist.as_tool(
            name="policy_specialist",
            args_model=WarrantyLookupArgs,
        )
    ],
    tool_choice="required",
    tool_call_limits={"policy_specialist": 1},
)
```

`tool_choice="required"` guarantees the first model turn uses the specialist
tool. `tool_call_limits={"policy_specialist": 1}` lets the manager answer on the
next turn instead of delegating forever.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/agent_as_tool_policy_specialist/main.py
```
