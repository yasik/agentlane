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

The demo loads `OPENAI_API_KEY` from the repository `.env` file automatically
when the variable is not already present in the environment.

## Why This Example Matters

The main file is intentionally small. The core harness flow is just:

```python
descriptor = AgentDescriptor(
    name="Acme Support",
    model=ResponsesClient(config=Config(api_key=api_key, model="gpt-5.4-mini")),
    instructions=PromptSpec(template=SUPPORT_INSTRUCTIONS_TEMPLATE, values=...),
)

runtime = SingleThreadedRuntimeEngine()
agent = Agent.bind(runtime, agent_id, runner=Runner(max_attempts=2), descriptor=descriptor)

await send_turn(runtime=runtime, agent_id=agent_id, customer_text="...", payload="...")
saved_run_state = agent.run_state

resumed_runtime = SingleThreadedRuntimeEngine()
resumed_agent = Agent.bind(
    resumed_runtime,
    agent_id,
    runner=Runner(max_attempts=2),
    descriptor=descriptor,
    run_state=saved_run_state,
)
```

In other words: the impressive part here is how little harness code is needed
to run and resume the agent. The missing piece is true user handoff orchestration,
which will eventually replace the mocked follow-up turns used by this demo.

## Run

```bash
uv run python examples/harness/customer_support_conversation/main.py
```
