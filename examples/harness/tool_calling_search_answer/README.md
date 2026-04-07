# Tool-Calling Search Answer

This example shows the smallest useful Phase 5 harness flow:

1. define templated agent instructions,
2. decorate one typed Python function with `@as_tool`,
3. send one user message, and
4. let the harness runner execute the tool loop before returning a final answer.

The search tool is mocked on purpose. It returns one fixed string so the demo
stays focused on the harness API. The assistant answer is still generated live
by `gpt-5.4-mini`.

## Why This Example Exists

The main thing to notice is how little framework code is needed in the main
script itself:

```python
@as_tool
async def search_help_center(question: str) -> str:
    """Search the Acme help center for the current policy answer."""
    del question
    return MOCK_SEARCH_RESULT


model = ResponsesClient(config=Config(api_key=api_key, model="gpt-5.4-mini"))
descriptor = AgentDescriptor(
    name="Acme Policy Assistant",
    model=model,
    instructions=PromptSpec(
        template=INSTRUCTIONS_TEMPLATE,
        values={
            "company_name": "Acme",
            "knowledge_source": "help-center",
            "tone": "clear and practical",
        },
    ),
    tools=Tools(
        tools=[search_help_center],
        tool_choice="required",
        tool_call_limits={"search_help_center": 1},
    ),
)

agent = Agent.bind(runtime, agent_id, runner=Runner(max_attempts=2), descriptor=descriptor)
outcome = await runtime.send_message(question, recipient=agent_id)
if outcome.status != DeliveryStatus.DELIVERED:
    raise RuntimeError(...)
```

`tool_choice="required"` guarantees the first model turn calls the search tool.
`tool_call_limits={"search_help_center": 1}` lets the runner remove that tool on
the next turn so the model answers from the returned search result instead of
calling the tool forever. `@as_tool` makes the function itself become a native
`Tool`, so the framework still derives the tool name, schema, and description
for you without extra boilerplate.

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

1. one user policy question,
2. one real assistant answer,
3. a short summary showing:
   - the tool name,
   - the tool arguments chosen by the model, and
   - the mocked search result that the harness fed back into the loop.
