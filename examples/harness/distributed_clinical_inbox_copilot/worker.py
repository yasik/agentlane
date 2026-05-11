"""Subprocess worker entrypoint for the distributed clinical inbox demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import structlog

from agentlane.messaging import TopicId
from agentlane.runtime import WorkerAgentRuntime

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.harness.distributed_clinical_inbox_copilot.agents import (  # noqa: E402
    register_worker_role,
)
from examples.harness.distributed_clinical_inbox_copilot.messages import (  # noqa: E402
    TELEMETRY_TOPIC_TYPE,
    WORKER_ROLE_ORDER,
    WORKER_ROLE_TO_LABEL,
    DemoEvent,
    register_demo_message_types,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the subprocess worker CLI."""
    parser = argparse.ArgumentParser(
        description="Clinical inbox distributed worker subprocess.",
    )
    parser.add_argument(
        "--role",
        choices=WORKER_ROLE_ORDER,
        required=True,
        help="Worker role to run in this process.",
    )
    parser.add_argument(
        "--host-address",
        required=True,
        help="Distributed runtime host address.",
    )
    parser.add_argument(
        "--bind-address",
        default="127.0.0.1:0",
        help="Local bind address for this worker runtime.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=4,
        help="Scheduler worker count for this runtime node.",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Demo session id used for telemetry routing.",
    )
    return parser


def _print_json(payload: dict[str, object]) -> None:
    """Print one structured line for the parent controller."""
    print(json.dumps(payload, sort_keys=True), flush=True)


async def _publish_worker_event(
    worker: WorkerAgentRuntime,
    *,
    session_id: str,
    actor: str,
    message: str,
) -> None:
    """Best-effort publish of a worker lifecycle event."""
    await worker.publish_message(
        DemoEvent(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            actor=actor,
            message=message,
        ),
        topic=TopicId.from_values(
            type_value=TELEMETRY_TOPIC_TYPE,
            route_key=session_id,
        ),
    )


async def run_worker(args: argparse.Namespace) -> None:
    """Start one worker runtime and wait until the process is signaled."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    worker = WorkerAgentRuntime(
        host_address=args.host_address,
        address=args.bind_address,
        worker_count=args.worker_count,
    )
    register_demo_message_types(worker)
    register_worker_role(worker, args.role)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await worker.start()
        label = WORKER_ROLE_TO_LABEL[args.role]
        worker_id = worker.worker_id or "unknown"
        _print_json(
            {
                "type": "worker_ready",
                "role": args.role,
                "label": label,
                "address": worker.address,
                "worker_id": worker_id,
                "pid": os.getpid(),
                "host_address": args.host_address,
            }
        )
        await _publish_worker_event(
            worker,
            session_id=args.session_id,
            actor="topology",
            message=f"→ {label} process ready pid={os.getpid()}",
        )
        await stop_event.wait()
        await _publish_worker_event(
            worker,
            session_id=args.session_id,
            actor="topology",
            message=f"→ {label} process stopping pid={os.getpid()}",
        )
    except Exception as exc:
        _print_json(
            {
                "type": "worker_error",
                "role": args.role,
                "error": str(exc),
            }
        )
        raise
    finally:
        if worker.is_running:
            await worker.stop_when_idle()


def main() -> None:
    """Run the worker subprocess."""
    parser = build_parser()
    asyncio.run(run_worker(parser.parse_args()))


if __name__ == "__main__":
    main()
