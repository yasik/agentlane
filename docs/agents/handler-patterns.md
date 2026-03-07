# Agent Handler Patterns (External SDK Friendly)

This document shows how to implement AgentLane handlers while keeping internal orchestration unopinionated.

## Core Contract

1. Mark handler methods with `@on_message`.
2. Use signature `(payload, context)`.
3. Annotate `payload` with a concrete type.
4. Keep handlers `async def`.
5. Prefer inheriting `BaseAgent` so runtime wiring (`id`, `send_message`, `publish_message`) is provided.

## Pattern 1: RPC Handler With External SDK Loop

Use `send_message` when caller expects a terminal response.

```python
from dataclasses import dataclass

from agentlane.messaging import MessageContext
from agentlane.runtime import BaseAgent, Engine, on_message


@dataclass(slots=True)
class SummarizeRequest:
    text: str


class SummarizerAgent(BaseAgent):
    def __init__(self, engine: Engine, llm_client: object) -> None:
        super().__init__(engine)
        self._llm_client = llm_client

    @on_message
    async def handle(self, payload: SummarizeRequest, context: MessageContext) -> object:
        _ = context
        # Replace with your own SDK call (OpenAI/Claude/LangChain/custom).
        summary = await self._llm_client.summarize(payload.text)
        return {"summary": summary}
```

## Pattern 2: Event Handler Publishing Downstream

`MessageContext` carries correlation metadata. Use `BaseAgent.publish_message` for downstream fan-out.

```python
from dataclasses import dataclass

from agentlane.messaging import MessageContext, TopicId
from agentlane.runtime import BaseAgent, Engine, on_message


@dataclass(slots=True)
class PlanReady:
    goal: str


@dataclass(slots=True)
class WorkItemCreated:
    goal: str
    steps: list[str]


class PlannerAgent(BaseAgent):
    def __init__(self, engine: Engine, planner_sdk: object) -> None:
        super().__init__(engine)
        self._planner_sdk = planner_sdk

    @on_message
    async def handle(self, payload: PlanReady, context: MessageContext) -> object:
        steps = await self._planner_sdk.plan(payload.goal)
        event = WorkItemCreated(goal=payload.goal, steps=steps)
        await self.publish_message(
            event,
            topic=TopicId.from_values(
                type_value="workflow.work_item_created",
                route_key="default",
            ),
            correlation_id=context.correlation_id,
        )
        return {"step_count": len(steps)}
```

## Pattern 3: Long-Running Handler With Checkpoints

For long-running work, structure logic in explicit checkpoints and persist intermediate progress.
This keeps retries and restarts predictable even when runtime stops in-flight work.

```python
from dataclasses import dataclass

from agentlane.messaging import MessageContext
from agentlane.runtime import BaseAgent, Engine, on_message


@dataclass(slots=True)
class DeepTask:
    task_id: str


class DeepWorkerAgent(BaseAgent):
    def __init__(self, engine: Engine, sdk: object, state_store: object) -> None:
        super().__init__(engine)
        self._sdk = sdk
        self._state_store = state_store

    @on_message
    async def handle(self, payload: DeepTask, context: MessageContext) -> object:
        _ = context
        checkpoints: list[str] = []

        for step_name in ["analyze", "plan", "execute"]:
            await self._sdk.run_step(step_name)
            checkpoints.append(step_name)
            await self._state_store.save(payload.task_id, {"steps": checkpoints})

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
