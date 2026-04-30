# Predefined Agent-As-Tool

This example shows the simplest predefined `agent-as-tool` flow in a portfolio
risk-review scenario:

1. define a child `AgentDescriptor`,
2. expose it to the parent with `.as_tool(...)`,
3. give that tool an explicit Pydantic args model, and
4. let the harness runner execute the child agent through the normal tool loop.

The important idea is that the model sees this exactly like a normal tool call.
The only difference is where the call executes: instead of running an in-process
function, the runner sends the validated arguments to another agent.

The core harness surface is small:

```python
class RiskReviewArgs(BaseModel):
    portfolio_name: str
    exposure_summary: str


risk_specialist = AgentDescriptor(
    name="Risk Specialist",
    model=model,
    instructions="You are a professional portfolio risk specialist.",
)


tools = Tools(
    tools=[
        risk_specialist.as_tool(
            name="risk_specialist",
            args_model=RiskReviewArgs,
        )
    ],
    tool_choice="required",
    tool_call_limits={"risk_specialist": 1},
)

agent = DefaultAgent(
    descriptor=AgentDescriptor(
        name="Portfolio Manager Assistant",
        tools=tools,
        ...
    ),
    runner=Runner(max_attempts=2),
)
result = await agent.run(user_message)
```

`tool_choice="required"` guarantees the first model turn uses the specialist
tool. `tool_call_limits={"risk_specialist": 1}` lets the manager answer on the
next turn instead of delegating forever.

## Run

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/agent_as_tool_risk_specialist/main.py
```
