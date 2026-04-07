# Customer Support Conversation Demo

This harness demo runs a real multi-turn customer-support conversation against
OpenAI using `gpt-5.4-mini`.

Important:

This example simulates customer follow-up turns by sending them directly from
the demo script. It does **not** implement proper user handoff yet. That is an
intentional limitation of the current harness phase. The assistant responses
are real model outputs, but the user side is still mocked by the example.

It demonstrates:

1. templated agent instructions via `PromptSpec`
2. a templated initial customer message
3. real assistant replies returned through the harness runner
4. simulated process restart by carrying `RunState` into a new agent instance

The demo requires `OPENAI_API_KEY` in the environment.

## Why This Example Matters

The main file is now intentionally self-contained. The core harness flow is just:

```python
model = ResponsesClient(config=Config(api_key=api_key, model="gpt-5.4-mini"))
descriptor = AgentDescriptor(
    name="Acme Support",
    model=model,
    instructions=PromptSpec(template=SUPPORT_INSTRUCTIONS_TEMPLATE, values=...),
)

runtime = SingleThreadedRuntimeEngine()
agent = Agent.bind(runtime, agent_id, runner=Runner(max_attempts=2), descriptor=descriptor)

first_outcome = await runtime.send_message([...], recipient=agent_id)
second_outcome = await runtime.send_message("...", recipient=agent_id)
saved_run_state = agent.run_state

resumed_runtime = SingleThreadedRuntimeEngine()
Agent.bind(
    resumed_runtime,
    agent_id,
    runner=Runner(max_attempts=2),
    descriptor=descriptor,
    run_state=saved_run_state,
)

resumed_outcome = await resumed_runtime.send_message("...", recipient=agent_id)
```

The point is that the script reads top-to-bottom without demo-only helper
layers. The missing piece is true user handoff orchestration, which will
eventually replace the mocked follow-up turns used by this demo.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/customer_support_conversation/main.py
```

Or export it once and run without the prefix:

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/customer_support_conversation/main.py
```
