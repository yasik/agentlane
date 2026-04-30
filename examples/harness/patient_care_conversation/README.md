# Patient Care Conversation Demo

This harness demo runs a real multi-turn patient-care navigation conversation
against OpenAI using `gpt-5.4-mini`.

Important:

This example simulates patient follow-up turns by sending them directly from
the demo script. It does **not** implement proper clinician handoff yet. That
is an intentional limitation of the current harness phase. The assistant
responses are real model outputs, but the patient side is still mocked by the
example.

It demonstrates:

1. templated agent instructions via `PromptSpec`
2. a templated initial patient message
3. real assistant replies returned through the harness runner
4. simulated process restart by carrying `RunState` into a new agent instance

The demo requires `OPENAI_API_KEY` in the environment.

## Why This Example Matters

The main file is intentionally self-contained. The core harness flow is just:

```python
model = ResponsesClient(config=Config(api_key=api_key, model="gpt-5.4-mini"))
descriptor = AgentDescriptor(
    name="Patient Care Navigator",
    model=model,
    instructions=PromptSpec(template=CARE_INSTRUCTIONS_TEMPLATE, values=...),
)

agent = DefaultAgent(
    descriptor=descriptor,
    runner=Runner(max_attempts=2),
)

first_result = await agent.run([...])
second_result = await agent.run("...")
saved_run_state = agent.run_state

resumed_agent = DefaultAgent(
    runner=Runner(max_attempts=2),
    descriptor=descriptor,
    run_state=saved_run_state,
)

resumed_result = await resumed_agent.run("...")
```

The point is that the script reads top-to-bottom without demo-only helper
layers. The missing piece is true clinician handoff orchestration, which will
eventually replace the mocked follow-up turns used by this demo.

## Run

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/patient_care_conversation/main.py
```

Or export it once and run without the prefix:

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/patient_care_conversation/main.py
```
