# Models Overview

`agentlane.models` provides the low-level LLM primitives used by provider
packages and the harness.

It keeps model-facing concerns separate from runtime delivery and agent
orchestration. If you are implementing model clients, prompts, tool schemas, or
output validation, this is the layer to build on.

## What It Includes

1. `Model`, `Config`, and `ModelResponse` as the shared model client contract.
2. `PromptTemplate`, `MultiPartPromptTemplate`, and `PromptSpec` for typed
   prompt input.
3. `Tool`, `Tools`, `ToolExecutor`, and `@as_tool` for native tool definition
   and execution.
4. `OutputSchema` and strict JSON-schema helpers for structured outputs.
5. Retry and rate-limiting helpers for provider clients.

## Boundaries

1. This package defines reusable model primitives.
2. It does not implement agent orchestration or runtime delivery.
3. Application code should usually provide plain payloads or prompt primitives,
   not low-level message dictionaries.
4. Provider-specific request arguments should flow through per-call model args
   instead of being normalized into `Config`.

## Tool Ergonomics

The common tool paths are intentionally lightweight:

1. decorate a typed function with `@as_tool`,
2. pass typed callables into `Tools(...)`,
3. use `Tool.from_function(...)` when you want an explicit native `Tool`.

All three paths share the same schema inference logic for name, description, and
arguments.

## Minimal Example

```python
from typing import TypedDict

from agentlane.models import OutputSchema, PromptSpec, PromptTemplate, Tools, as_tool


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
```

## Related Docs

1. [Harness Architecture](../harness/architecture.md)
2. [Harness Runner](../harness/runner.md)
3. [Transport Serialization](../transport/serialization.md)
