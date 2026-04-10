# Harness Tasks

This page explains the smallest harness abstraction. A task is useful when you
want runtime-oriented orchestration code with a clearer name and a few helper
methods, but without opting into the default LLM loop.

[`Task`](../../src/agentlane/harness/_task.py) itself is a thin layer over
[`BaseAgent`](../../src/agentlane/runtime/_base_agent.py), so it keeps the
runtime model intact while adding a clearer place for application orchestration
code.

## What A Task Is

[`Task`](../../src/agentlane/harness/_task.py) is the top-level harness
primitive for application work that sits upstream of LLM-driven agents.

Use a task when you need orchestration logic that:

1. receives runtime messages,
2. talks to other runtime recipients,
3. may call databases, webhooks, filesystems, or service clients, and
4. should not itself imply an LLM loop.

The default harness `Agent` builds directly on top of `Task`.

## Runtime Model

Tasks reuse the existing runtime model directly:

1. a task extends `agentlane.runtime.BaseAgent`
2. message handlers are still declared with `@on_message`
3. orchestration still uses `send_message` and `publish_message`
4. runtime instance reuse is still keyed by `AgentId`

There is no second task-specific execution engine.

## Registration Patterns

### Lazy factory registration

Use `Task.register(runtime, agent_type, ...)` when the runtime should create
task instances on demand.

This is the right default when:

1. task construction is cheap,
2. state should be tied to `AgentId` reuse, and
3. you want runtime factories rather than pre-created instances.

### Stateful instance binding

Use `Task.bind(runtime, agent_id, ...)` when you want to construct one concrete
task instance and keep reusing it for a specific `AgentId`.

This is the right choice when:

1. task state must survive across multiple deliveries for the same id,
2. dependencies or state are initialized up front, or
3. tests need direct access to the concrete task instance.

## Stateful vs Stateless Guidance

Tasks may be either stateful or effectively stateless depending on how they are
addressed.

1. Reusing the same `AgentId` means the runtime will reuse the same task
   instance.
2. Using a new `AgentId` creates an isolated task instance for that key.
3. A task registered via factory can still behave statefully if callers reuse
   the same `AgentId`.
4. A task bound as one explicit instance is intentionally stateful for that
   exact `AgentId`.

The runtime semantics stay explicit: statefulness is driven by instance reuse,
not hidden task logic.

## Design Constraint

Tasks remain intentionally thin. They add semantic clarity and small
registration helpers, but they do not replace or wrap the runtime's routing,
scheduler, or dispatcher contracts.
