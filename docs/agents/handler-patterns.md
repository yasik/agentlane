# Agent Handler Patterns (External SDK Friendly)

This document shows how to implement AgentLane handlers while keeping internal orchestration unopinionated.

## Core Contract

1. Mark handler methods with `@on_message`.
2. Use signature `(payload, context)`.
3. Annotate `payload` with a concrete type.
4. Keep handlers `async def`.

## Pattern 1: RPC Handler With External SDK Loop

Use `send_message` when caller expects a terminal response.

```python
from dataclasses import dataclass

from agentlane.agents import on_message
from agentlane.messaging import MessageContext


@dataclass(slots=True)
class SummarizeRequest:
    text: str


class SummarizerAgent:
    def __init__(self, llm_client: object) -> None:
        self._llm_client = llm_client

    @on_message
    async def handle(self, payload: SummarizeRequest, context: MessageContext) -> object:
        _ = context
        # Replace with your own SDK call (OpenAI/Claude/LangChain/custom).
        summary = await self._llm_client.summarize(payload.text)
        return {"summary": summary}
```

## Pattern 2: Event Handler Publishing Downstream

`MessageContext` carries correlation metadata, but runtime publish is done through your injected runtime dependency.

```python
from dataclasses import dataclass

from agentlane.agents import on_message
from agentlane.messaging import MessageContext, TopicId
from agentlane.runtime import RuntimeEngine


@dataclass(slots=True)
class PlanReady:
    goal: str


@dataclass(slots=True)
class WorkItemCreated:
    goal: str
    steps: list[str]


class PlannerAgent:
    def __init__(self, runtime: RuntimeEngine, planner_sdk: object) -> None:
        self._runtime = runtime
        self._planner_sdk = planner_sdk

    @on_message
    async def handle(self, payload: PlanReady, context: MessageContext) -> object:
        steps = await self._planner_sdk.plan(payload.goal)
        event = WorkItemCreated(goal=payload.goal, steps=steps)
        await self._runtime.publish_message(
            event,
            topic=TopicId.from_values(type_value="workflow.work_item_created", route_key="default"),
            correlation_id=context.correlation_id,
        )
        return {"step_count": len(steps)}
```

## Pattern 3: Long-Running Handler With Cooperative Cancellation

For long-running work, check `context.cancellation_token` between expensive steps and pass cancellation into SDK calls when possible.

```python
from dataclasses import dataclass

from agentlane.agents import on_message
from agentlane.messaging import MessageContext


@dataclass(slots=True)
class DeepTask:
    task_id: str


class DeepWorkerAgent:
    def __init__(self, sdk: object) -> None:
        self._sdk = sdk

    @on_message
    async def handle(self, payload: DeepTask, context: MessageContext) -> object:
        checkpoints: list[str] = []

        for step_name in ["analyze", "plan", "execute"]:
            if context.cancellation_token.is_cancelled:
                return {
                    "task_id": payload.task_id,
                    "status": "cancelled",
                    "completed_steps": checkpoints,
                }

            # Replace with a real SDK call. If SDK supports cancellation,
            # pass the token/signal into the call as well.
            await self._sdk.run_step(step_name)
            checkpoints.append(step_name)

        return {
            "task_id": payload.task_id,
            "status": "completed",
            "completed_steps": checkpoints,
        }
```

## Practical Guidance

1. Preserve `context.correlation_id` when sending/publishing downstream to keep one logical workflow trace.
2. Keep payload models small and explicit to avoid ambiguous handler routing.
3. Treat handler code as orchestration glue; keep provider-specific logic in dedicated service clients.
