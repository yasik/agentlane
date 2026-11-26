# Spawn Model Routing

This example uses `DefaultAgent` and real OpenRouter clients to delegate one
calculation. `Runner(spawn_model_provider=provider)` selects the helper's client
and request settings when the helper is created.

## Policy settings

The application configures `RoutingProvider` in `main.py`:

```python
provider = RoutingProvider(
    routes,
    model_can_choose=True,
    mandatory_route=None,
    default_route="quick",
)
```

`model_can_choose=True` adds `SpawnChoice` to the tool schema. The parent can
suggest `quick` or `reasoning`. Set it to `False` to omit those preferences.
`mandatory_route` overrides any suggestion. `default_route` applies when neither
an override nor a suggestion is present.

The configured routes contain real model clients and their reasoning settings.
The runner applies the selected configuration to the new child.

## Run

Set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` in `.env.local`, then run from the
repository root:

```bash
uv run --env-file .env.local --extra litellm \
  python examples/harness/spawn_model_routing/main.py
```

`OPENROUTER_MODEL` selects the parent. Optional `SPAWN_QUICK_MODEL` defaults to
`openai/gpt-5.4-mini`; `SPAWN_REASONING_MODEL` defaults to the parent model.
Use OpenRouter model IDs and models that support the configured reasoning effort.

The example makes paid API requests and prints the task and final answer.
With the default settings, the parent suggests the `reasoning` route.
