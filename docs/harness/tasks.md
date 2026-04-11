# Harness Tasks

Tasks are the smallest harness abstraction. They are useful when you want a
clear place for application orchestration logic, but you do not want to opt
into the default model loop.

[`Task`](../../src/agentlane/harness/_task.py) is a thin layer over
[`BaseAgent`](../../src/agentlane/runtime/_base_agent.py). That is why it feels
familiar if you already understand the runtime: it keeps the same message
handlers, the same delivery model, and the same instance reuse rules.

## When To Use A Task

Use a task when the work is application logic rather than model-driven
reasoning. Typical examples include:

1. coordinating multiple runtime recipients
2. calling databases, services, or webhooks
3. shaping a workflow before or after a model-backed agent runs

If the code needs the runtime but not an LLM loop, a task is usually the right
level.

## Why Tasks Stay Thin

Tasks do not introduce a second scheduler or a second execution model. They
exist mostly to make intent clearer and to provide a few small registration
helpers.

That means the important runtime ideas still apply:

1. one concrete `AgentId` means one reusable instance
2. message handlers are still declared with `@on_message`
3. orchestration still uses `send_message(...)` and `publish_message(...)`

## Registration And State

There are two common patterns:

1. `Task.register(...)` when the runtime should create instances lazily
2. `Task.bind(...)` when you want to create and bind one concrete instance

The choice is really about state ownership.

Use registration when state should follow the normal `AgentId` reuse model. Use
binding when you already have a concrete stateful instance and want that exact
instance tied to one identity.

## A Useful Rule Of Thumb

If you start adding prompt construction, model configuration, or tool policies
to a task, it is probably time to move up to the default harness
[`Agent`](../../src/agentlane/harness/_agent.py).
