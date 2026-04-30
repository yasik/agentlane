# Tool-Calling Protocol Answer

This example shows the smallest useful harness tool-calling flow in a clinical
protocol lookup scenario:

1. define templated agent instructions,
2. decorate one typed Python function with `@as_tool`,
3. send one user message, and
4. let the harness runner execute the tool loop before returning a final answer.

The search tool is mocked on purpose. It returns one fixed clinical protocol
snippet so the demo stays focused on the harness API. The assistant answer is
still generated live by `gpt-5.4-mini`.

The main thing to notice is how little framework code is needed in the main
script itself:

```python
@as_tool
async def search_clinical_protocol(question: str) -> str:
    """Search the clinical protocol library for the current care guidance."""
    del question
    return MOCK_SEARCH_RESULT


model = ResponsesClient(config=Config(api_key=api_key, model="gpt-5.4-mini"))
descriptor = AgentDescriptor(
    name="Clinical Protocol Assistant",
    model=model,
    instructions=PromptSpec(
        template=INSTRUCTIONS_TEMPLATE,
        values={
            "clinic_name": "Harborview Anticoagulation Clinic",
            "knowledge_source": "protocol-library",
            "tone": "clear and practical",
        },
    ),
    tools=Tools(
        tools=[search_clinical_protocol],
        tool_choice="required",
        tool_call_limits={"search_clinical_protocol": 1},
    ),
)

agent = DefaultAgent(descriptor=descriptor, runner=Runner(max_attempts=2))
result = await agent.run(question)
```

`tool_choice="required"` guarantees the first model turn calls the protocol
search tool. `tool_call_limits={"search_clinical_protocol": 1}` lets the runner
remove that tool on the next turn so the model answers from the returned search
result instead of calling the tool forever. `@as_tool` makes the function itself
become a native `Tool`, so the framework still derives the tool name, schema,
and description for you without extra boilerplate.

## Run

The example requires `OPENAI_API_KEY` in the environment.

```bash
OPENAI_API_KEY=sk-... uv run python examples/harness/tool_calling_search_answer/main.py
```

Or export it once and run without the prefix:

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/harness/tool_calling_search_answer/main.py
```

## What You Should See

1. one clinical protocol question,
2. one real assistant answer,
3. a short summary showing:
   - the tool name,
   - the tool arguments chosen by the model, and
   - the mocked protocol result that the harness fed back into the loop.
