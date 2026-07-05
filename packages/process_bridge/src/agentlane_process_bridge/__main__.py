"""Command-line entrypoint for packaged AgentLane process-bridge backends."""

import argparse
import asyncio
import importlib
import inspect
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

from ._backend import AgentRuntime
from ._stdio import AgentBackend, run_stdio

type BackendFactory = Callable[
    [], AgentBackend | AgentRuntime | Awaitable[AgentBackend | AgentRuntime]
]


def parse_app_reference(reference: str) -> tuple[str, str]:
    """Parse a `module:attribute` backend factory reference."""
    module_name, separator, attribute = reference.partition(":")

    if not module_name or separator != ":" or not attribute:
        raise ValueError("Expected --app as module:attribute.")

    return module_name, attribute


def load_backend_factory(reference: str) -> BackendFactory:
    """Load a backend factory from a `module:attribute` reference."""
    module_name, attribute = parse_app_reference(reference)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)

    if not callable(factory):
        raise TypeError(f"Backend reference is not callable: {reference}")

    return cast(BackendFactory, factory)


async def resolve_agent_backend(
    value: AgentBackend | AgentRuntime | Awaitable[AgentBackend | AgentRuntime],
) -> AgentBackend:
    """Resolve a factory return value into an `AgentBackend`."""
    resolved = await value if inspect.isawaitable(value) else value

    if isinstance(resolved, AgentBackend):
        return resolved

    return AgentBackend(agent=resolved)


async def run_app_reference(reference: str) -> None:
    """Load an app factory and serve it over stdio."""
    factory = load_backend_factory(reference)
    backend = await resolve_agent_backend(factory())
    await run_stdio(
        agent=backend.agent,
        approvals=backend.approvals,
        ready_metadata=backend.ready_metadata,
        config=backend.config,
    )


async def amain(argv: Sequence[str] | None = None) -> int:
    """Run the process-bridge command-line entrypoint."""
    parser = argparse.ArgumentParser(
        prog="python -m agentlane_process_bridge",
        description="Serve an AgentLane backend over the local stdio bridge.",
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Backend factory reference as module:attribute.",
    )
    args = parser.parse_args(argv)

    try:
        await run_app_reference(args.app)
    except Exception as exc:
        print(f"agentlane_process_bridge: {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous CLI wrapper used by `python -m agentlane_process_bridge`."""
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
