"""Runtime primitives and engine exports."""

from typing import Any

from ._shared import Engine

__all__ = [
    "AgentRegistry",
    "Engine",
    "DistributedRuntimeEngine",
    "distributed_runtime",
    "PerAgentMailboxScheduler",
    "RuntimeEngine",
    "SingleThreadedRuntimeEngine",
    "runtime_scope",
    "SchedulerRejectedError",
    "single_threaded_runtime",
]


def __getattr__(name: str) -> Any:
    if name in ("distributed_runtime", "runtime_scope", "single_threaded_runtime"):
        from ._context import (
            distributed_runtime,
            runtime_scope,
            single_threaded_runtime,
        )

        mapping = {
            "distributed_runtime": distributed_runtime,
            "runtime_scope": runtime_scope,
            "single_threaded_runtime": single_threaded_runtime,
        }
        return mapping[name]

    if name in ("DistributedRuntimeEngine", "RuntimeEngine", "SingleThreadedRuntimeEngine"):
        from ._engine import (
            DistributedRuntimeEngine,
            RuntimeEngine,
            SingleThreadedRuntimeEngine,
        )

        mapping = {
            "DistributedRuntimeEngine": DistributedRuntimeEngine,
            "RuntimeEngine": RuntimeEngine,
            "SingleThreadedRuntimeEngine": SingleThreadedRuntimeEngine,
        }
        return mapping[name]

    if name == "AgentRegistry":
        from ._registry import AgentRegistry

        return AgentRegistry

    if name in ("PerAgentMailboxScheduler", "SchedulerRejectedError"):
        from ._scheduler import (
            PerAgentMailboxScheduler,
            SchedulerRejectedError,
        )

        mapping = {
            "PerAgentMailboxScheduler": PerAgentMailboxScheduler,
            "SchedulerRejectedError": SchedulerRejectedError,
        }
        return mapping[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
