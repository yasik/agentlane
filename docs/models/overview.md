# Models Overview

`agentlane.models` is the shared model-facing layer in AgentLane. It holds the
public pieces that describe how prompts are rendered through
[`PromptTemplate`](../../src/agentlane/models/_prompts.py) and
[`PromptSpec`](../../src/agentlane/models/_prompts.py), how tools are exposed
through [`Tool`](../../src/agentlane/models/_tool.py) and
[`Tools`](../../src/agentlane/models/_interface.py), how outputs are validated
through [`OutputSchema`](../../src/agentlane/models/_output_schema.py), and how
provider clients conform to the shared [`Model`](../../src/agentlane/models/_interface.py)
and [`Config`](../../src/agentlane/models/_interface.py) contract.

If you are working on model requests or responses, this is usually the right
layer to start with. Run-scoped helpers such as
[`DefaultRunContext`](../../src/agentlane/models/run/_context.py) and
[`TraceCtxManager`](../../src/agentlane/models/run/_ctx_managers.py) also live
here.

## What It Includes

1. [`Model`](../../src/agentlane/models/_interface.py),
   [`Config`](../../src/agentlane/models/_interface.py), and
   [`ModelResponse`](../../src/agentlane/models/_types.py) as the shared model
   client contract.
2. [`PromptTemplate`](../../src/agentlane/models/_prompts.py),
   [`MultiPartPromptTemplate`](../../src/agentlane/models/_prompts.py), and
   [`PromptSpec`](../../src/agentlane/models/_prompts.py) for typed prompt
   input.
3. [`Tool`](../../src/agentlane/models/_tool.py),
   [`Tools`](../../src/agentlane/models/_interface.py),
   [`ToolExecutor`](../../src/agentlane/models/_tool_executor.py), and
   [`@as_tool`](../../src/agentlane/models/_tool.py) for native tool
   definition and execution.
4. [`OutputSchema`](../../src/agentlane/models/_output_schema.py) and strict
   JSON-schema helpers for structured outputs.
5. Retry and rate-limiting helpers for provider clients.
6. `agentlane.models.run` helpers for run-scoped context and trace ownership.

## Boundaries

1. This package defines reusable model primitives.
2. It does not implement agent orchestration or runtime delivery.
3. Application code should usually provide plain payloads or prompt primitives,
   not low-level message dictionaries.
4. Provider-specific request arguments should flow through per-call model args
   instead of being normalized into `Config`.
5. Provider packages such as `agentlane-openai` and `agentlane-litellm` build
   on this layer instead of redefining the shared contract.

## Prompt Templating

Prompt templating is the part of the model layer that keeps prompt structure
separate from prompt values. That makes prompt definitions easier to reuse and
easier to reason about when a run is built from typed input.

See [Models: Prompt Templating](./prompt-templating.md) for the full guide.

## Tool Ergonomics

The common tool paths are intentionally lightweight:

1. decorate a typed function with
   [`@as_tool`](../../src/agentlane/models/_tool.py),
2. pass typed callables into
   [`Tools(...)`](../../src/agentlane/models/_interface.py),
3. use [`Tool.from_function(...)`](../../src/agentlane/models/_tool.py) when
   you want an explicit native tool value.

All three paths share the same schema inference logic for name, description, and
arguments.

`Tools(...)` also owns the model-facing tool policy for a request:

1. `tool_choice` controls whether tool use is automatic, required, or disabled.
2. `parallel_tool_calls` controls whether the model may issue parallel tool
   calls.
3. `tool_call_limits` and `max_tool_round_trips` provide loop-safety controls
   for higher-level harness usage.

## Run-Scoped Helpers

Run-scoped helpers live under `agentlane.models.run`.

Use them when you need state that should exist for one agent run or one tool
chain without becoming part of the LLM-visible prompt.

The public helpers are:

1. [`RunContext[T]`](../../src/agentlane/models/run/_context.py) for carrying
   typed dependencies or in-memory state.
2. [`DefaultRunContext`](../../src/agentlane/models/run/_context.py) for a
   dictionary-backed context with async-safe helper methods such as
   `increment(...)`, `set(...)`, and `append_to_list(...)`.
3. [`TraceCtxManager`](../../src/agentlane/models/run/_ctx_managers.py) for
   creating a trace only when the current code path does not already have one.

## Minimal Example

```python
from typing import TypedDict

from agentlane.models import OutputSchema, PromptSpec, PromptTemplate, Tools, as_tool
from agentlane.models.run import DefaultRunContext


class SupportVars(TypedDict):
    customer_tier: str


support_prompt = PromptTemplate[SupportVars, str](
    system_template="You support {{ customer_tier }} customers.",
    user_template=None,
    output_schema=OutputSchema(str),
)


@as_tool
async def search_help_center(question: str) -> str:
    """Search the help center."""
    return "Reset your password from Settings > Security."


instructions = PromptSpec(
    template=support_prompt,
    values={"customer_tier": "business"},
)

tools = Tools(tools=[search_help_center])
run_context = DefaultRunContext()
```

## Related Docs

1. [Harness Architecture](../harness/architecture.md)
2. [Harness Runner](../harness/runner.md)
3. [Models: Prompt Templating](./prompt-templating.md)
4. [Transport Serialization](../transport/serialization.md)
