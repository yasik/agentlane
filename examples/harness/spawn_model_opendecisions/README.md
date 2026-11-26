# Spawn Model Selection with OpenDecisions

This example uses [OpenDecisions](https://github.com/OpenDecisions/OpenDecisions)
to choose a configured child model before the helper starts. The parent supplies
only the task: `RoutingProvider.parameters_model=None` keeps model preferences
out of its tool schema.

`Runner(spawn_model_provider=provider)` calls the provider's async `resolve()`.
It sends the task to `POST /v1/decisions`, validates the returned `quick` or
`reasoning` choice, and maps that choice to a `SpawnModelConfig`. The runner
retains that client and its request settings for the child's lifetime. The
provider uses `httpx.AsyncClient` because the OpenDecisions Python SDK is
synchronous.

## Run

Set these values in `.env.local` at the AgentLane repository root. Replace `...`
with your OpenRouter key and a parent model ID that supports tools:

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
OPENDECISIONS_MODEL=openrouter/typesafe/jev-1.14
```

Start the OpenDecisions server in one terminal:

```bash
uvx --env-file .env.local \
  --from git+https://github.com/OpenDecisions/OpenDecisions.git@c9fb35b2d859170cc830815c8fee199eac3fba0f \
  opendecisions serve --no-local
```

This setup uses the remote decision model through OpenRouter. It does not load
local model weights. Keep the server running, then run the standalone example
from the repository root in another terminal:

```bash
uv run --env-file .env.local --extra litellm \
  python examples/harness/spawn_model_opendecisions/main.py
```

The example makes paid requests and prints the task and final answer.
`OPENDECISIONS_MODEL` selects the decision model; `OPENROUTER_MODEL` selects the
parent. The decision service chooses between the child routes configured in
`main.py`.

## Optional settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENDECISIONS_BASE_URL` | `http://127.0.0.1:8000` | Decision service address. |
| `OPENDECISIONS_API_KEY` | Unset | Sends a Bearer token when the service requires authentication. |
| `OPENDECISIONS_MODEL` | `auto` | Decision model; the run setup above selects a remote model explicitly. |
| `SPAWN_QUICK_MODEL` | `openai/gpt-5.4-mini` | Child for simple tasks, with low reasoning effort. |
| `SPAWN_REASONING_MODEL` | Parent model | Child for complex tasks, with medium reasoning effort. |

Use child models that support their configured reasoning effort. HTTP errors
and invalid decision responses stop selection; this example has no fallback route.
