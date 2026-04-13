# agentlane.models

`agentlane.models` is the shared low-level foundation for working with LLM calls in AgentLane.

It exists to keep model-facing concerns separate from both the runtime and the higher-level harness layer. The runtime should not need to know how prompts, schemas, retries, tool payloads, or provider adapters are represented. The harness should build on stable primitives instead of embedding provider logic directly.

At a high level, this package provides:

1. client-facing model primitives such as `Model`, `Factory`, `Config`, and the shared `ModelResponse` contract,
2. `ModelStreamEvent` and `Model.stream_response(...)` for provider-grounded streaming,
3. prompt-template helpers such as `PromptTemplate`, `MultiPartPromptTemplate`, and `PromptSpec` for building typed LLM message content,
4. the native `Tool` primitive and tool execution helpers,
5. retry and rate-limiting helpers for model clients,
6. `RunContext` primitives for ephemeral per-run state,
7. a clean dependency boundary so provider packages can build on the same core model contract.

For tooling ergonomics, the common application path is intentionally lightweight:
decorate a typed Python function with `@as_tool`, pass a normal typed callable
into `Tools(...)`, or use `Tool.from_function(...)` when you want an explicit
native `Tool` value. All three paths share the same inference logic for tool
name, description, and strict argument schema, while still preserving the
lower-level `Tool(...)` constructor for full manual control.

Core principles:

1. Keep the contract provider-agnostic. Concrete clients adapt to this surface instead of forcing the rest of the framework to understand each provider separately.
2. Reuse OpenAI-compatible structures where practical rather than inventing parallel result models.
3. Keep orchestration out of this package. Tasks, agents, and runners belong in higher layers.
4. Put reusable LLM mechanics here once so runtime, harness, and provider packages do not drift apart.

`Config` is intentionally the shared control-plane and networking surface for model
clients. Model-specific request parameters such as temperature, reasoning effort,
cache retention, and provider-specific extras should be passed through client kwargs
or per-call model args rather than being normalized into `Config`.

The shared cancellation token intentionally lives in `agentlane.runtime`, not here. Model clients and tools consume that runtime primitive instead of growing a second copy.

If you are building orchestration, use the harness or runtime layers. Application developers should provide plain payloads or higher-level prompt primitives such as `PromptSpec`, not assemble low-level message dictionaries themselves. The harness runner owns request construction and decides how typed prompt input and accumulated run state become canonical model messages.

Streaming follows the same boundary. Provider adapters own event fidelity and
final response assembly. The shared models surface keeps the normalized event
envelope intentionally small and preserves raw provider payloads on each
`ModelStreamEvent`. That means OpenAI-native adapters can keep semantic
Responses API events, while LiteLLM adapters preserve the documented chunk
shape they actually expose without pretending to provide richer provider-native
stream semantics than LiteLLM publishes.

Likewise, application developers should usually define tools from typed Python
functions rather than hand-writing argument models for every simple tool. Reach
for `@as_tool` when you want the function declaration itself to read like a
tool definition, `Tools(tools=[my_function])` for lightweight one-offs, and the
explicit `Tool(...)` constructor only when you need low-level control over the
schema, handler shape, or output formatting.

If you are defining how the framework talks to models, validates outputs, executes tools, or carries ephemeral model-call state, it belongs here.
