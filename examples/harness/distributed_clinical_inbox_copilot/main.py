"""Distributed streamed clinical copilot demo with optional worker processes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from uuid import uuid4

os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import structlog
from agentlane_openai import ResponsesClient
from rich.live import Live

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agentlane.harness import AgentDescriptor, RunResult  # noqa: E402
from agentlane.harness.agents import DefaultAgent  # noqa: E402
from agentlane.messaging import DeliveryMode, MessageContext, TopicId  # noqa: E402
from agentlane.models import Config, Tools, as_tool  # noqa: E402
from agentlane.runtime import (  # noqa: E402
    BaseAgent,
    Engine,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)
from examples.harness.distributed_clinical_inbox_copilot.agents import (  # noqa: E402
    register_worker_role,
)
from examples.harness.distributed_clinical_inbox_copilot.messages import (  # noqa: E402
    MODEL_NAME,
    REVIEW_COMPLETED_TOPIC_TYPE,
    REVIEW_RESULT_AGENT_TYPE,
    REVIEW_TOPIC_TYPE,
    SPECIALIST_NAMES,
    SPECIALIST_WORKER_LABELS,
    TELEMETRY_AGENT_TYPE,
    TELEMETRY_TOPIC_TYPE,
    WORKER_ROLE_ORDER,
    WORKER_ROLE_TO_LABEL,
    AggregatedClinicalReview,
    ClinicalReviewTask,
    DemoEvent,
    DemoInputs,
    register_demo_message_types,
)
from examples.harness.distributed_clinical_inbox_copilot.ui import (  # noqa: E402
    CONSOLE,
    DemoTelemetry,
    DemoUIState,
    InboxCopilotHooks,
    build_dashboard,
    consume_stream,
    drain_telemetry,
    print_final_summary,
    refresh_dashboard,
    update_runtime_node,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class SessionState:
    """Mutable session state shared between top-level tools."""

    session_id: str
    chart_snapshot: str | None = None
    review_counter: int = 0


class ReviewCompletionTracker:
    """Waits for controller-local result receiver to finish a review id."""

    def __init__(self) -> None:
        """Initialize tracker storage."""
        self._futures: dict[str, asyncio.Future[AggregatedClinicalReview]] = {}

    def register(self, review_id: str) -> None:
        """Register one review id."""
        if review_id in self._futures:
            return
        self._futures[review_id] = asyncio.get_running_loop().create_future()

    def complete(self, review: AggregatedClinicalReview) -> None:
        """Complete one review future if it is still pending."""
        self.register(review.review_id)
        future = self._futures[review.review_id]
        if future.done():
            return
        future.set_result(review)

    async def wait_for_result(
        self,
        review_id: str,
        *,
        timeout_seconds: float,
    ) -> AggregatedClinicalReview:
        """Wait for one aggregated review to complete."""
        self.register(review_id)
        return await asyncio.wait_for(self._futures[review_id], timeout=timeout_seconds)


class TelemetryReceiverAgent(BaseAgent):
    """Controller-local sink for distributed telemetry messages."""

    def __init__(self, engine: Engine, telemetry: DemoTelemetry) -> None:
        """Initialize the receiver with a local UI queue."""
        super().__init__(engine)
        self._telemetry = telemetry

    @on_message
    async def handle(self, payload: DemoEvent, context: MessageContext) -> object:
        """Forward distributed telemetry into the controller-local UI queue."""
        _ = context
        self._telemetry.queue.put_nowait(payload)
        return None


class ReviewResultAgent(BaseAgent):
    """Controller-local sink for completed aggregate reviews."""

    def __init__(
        self,
        engine: Engine,
        tracker: ReviewCompletionTracker,
        telemetry: DemoTelemetry,
    ) -> None:
        """Initialize the receiver with the local completion tracker."""
        super().__init__(engine)
        self._tracker = tracker
        self._telemetry = telemetry

    @on_message
    async def handle(
        self,
        payload: AggregatedClinicalReview,
        context: MessageContext,
    ) -> object:
        """Complete the local waiter for one distributed review result."""
        _ = context
        self._tracker.complete(payload)
        self._telemetry.emit(
            "aggregator",
            f"→ aggregated review ready for {payload.review_id} from distributed fan-in",
        )
        return None


@dataclass(slots=True)
class WorkerProcessHandle:
    """Controller-owned subprocess metadata."""

    role: str
    label: str
    process: asyncio.subprocess.Process
    stderr_task: asyncio.Task[None] | None = None


def _new_local_workers() -> list[tuple[str, WorkerAgentRuntime]]:
    """Return an empty local worker list with a concrete type."""
    return []


def _new_worker_processes() -> list[WorkerProcessHandle]:
    """Return an empty worker process list with a concrete type."""
    return []


@dataclass(slots=True)
class RuntimeCluster:
    """Runtime objects owned by the controller process."""

    host: WorkerAgentRuntimeHost
    copilot_worker: WorkerAgentRuntime
    local_workers: list[tuple[str, WorkerAgentRuntime]] = field(
        default_factory=_new_local_workers
    )
    worker_processes: list[WorkerProcessHandle] = field(
        default_factory=_new_worker_processes
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the demo."""
    parser = argparse.ArgumentParser(
        description="Distributed clinical inbox copilot demo.",
    )
    parser.add_argument(
        "--clinician-name",
        type=str,
        default=None,
        help="Clinician name shown in the demo prompt flow.",
    )
    parser.add_argument(
        "--patient-label",
        type=str,
        default=None,
        help="Patient label displayed throughout the demo.",
    )
    parser.add_argument(
        "--patient-message",
        type=str,
        default=None,
        help="Inbox message sent to the top-level copilot.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="OpenAI model name for the streamed top-level agent.",
    )
    parser.add_argument(
        "--mode",
        choices=("in-process", "multiprocess"),
        default="in-process",
        help="Run specialist workers in this process or as subprocesses.",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Shortcut for --mode multiprocess.",
    )
    parser.add_argument(
        "--smoke-review",
        action="store_true",
        help="Run a model-free distributed review smoke test and exit.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=4,
        help="Scheduler worker count for each distributed runtime node.",
    )
    parser.add_argument(
        "--worker-bind-address",
        type=str,
        default="127.0.0.1:0",
        help="Bind address used by worker runtime nodes.",
    )
    parser.add_argument(
        "--worker-start-timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout when waiting for worker subprocess readiness.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout when waiting for the specialist aggregator.",
    )
    return parser


def prompt_with_default(label: str, default: str) -> str:
    """Prompt the user for one interactive value with a default."""
    value = CONSOLE.input(f"[bold cyan]{label}[/bold cyan] [[dim]{default}[/dim]]: ")
    stripped = value.strip()
    return stripped or default


def resolve_inputs(args: argparse.Namespace) -> DemoInputs:
    """Resolve interactive inputs from CLI arguments or terminal prompts."""
    default_inputs = DemoInputs(
        clinician_name="Dr. Rivera",
        patient_label="Maya R., 54F",
        patient_message=(
            "I started the new injection and now I feel dizzy. My sugar was 64 "
            "this morning and 68 after lunch. Should I stop anything or go in?"
        ),
    )
    if not sys.stdin.isatty():
        return DemoInputs(
            clinician_name=args.clinician_name or default_inputs.clinician_name,
            patient_label=args.patient_label or default_inputs.patient_label,
            patient_message=args.patient_message or default_inputs.patient_message,
        )

    clinician_name = args.clinician_name or prompt_with_default(
        "Clinician name",
        default_inputs.clinician_name,
    )
    patient_label = args.patient_label or prompt_with_default(
        "Patient label",
        default_inputs.patient_label,
    )
    patient_message = args.patient_message or prompt_with_default(
        "Patient inbox message",
        default_inputs.patient_message,
    )
    return DemoInputs(
        clinician_name=clinician_name,
        patient_label=patient_label,
        patient_message=patient_message,
    )


def build_chart_snapshot(inputs: DemoInputs) -> str:
    """Return a concise mock chart snapshot for the selected patient."""
    return textwrap.dedent(f"""
        Patient: {inputs.patient_label}
        Problem list: type 2 diabetes, hypertension
        Current meds: metformin 1000 mg BID, glipizide 10 mg BID, semaglutide 0.5 mg weekly started 2 weeks ago
        Recent labs: A1c 7.1% three weeks ago, creatinine 0.9 mg/dL
        Recent patient-reported data: glucose 63-68 in the last 24 hours, dizziness after meals, no chest pain, no fever
        Team note: review for likely medication-related hypoglycemia and provide a patient-safe same-day plan
        """).strip()


def build_user_prompt(inputs: DemoInputs) -> str:
    """Build the single user turn sent to the streamed top-level agent."""
    return textwrap.dedent(f"""
        Clinician: {inputs.clinician_name}
        Patient: {inputs.patient_label}
        Inbox message: {inputs.patient_message}

        Please review this inbox message, tell me the main safety concern, and
        draft a patient reply I can send.
        """).strip()


def format_review_for_model(review: AggregatedClinicalReview) -> str:
    """Render the aggregated specialist findings into one tool result string."""
    sections = [
        f"review_id: {review.review_id}",
        f"urgent_flag: {'yes' if review.urgent_flag else 'no'}",
        "specialist_findings:",
    ]
    for finding in review.findings.values():
        sections.append(
            "- "
            f"{expect_finding_str(finding, 'agent_name')} | "
            f"{expect_finding_str(finding, 'headline')}: "
            f"{expect_finding_str(finding, 'detail')}"
        )
    sections.append("patient_reply_guidance:")
    for finding in review.findings.values():
        sections.append(
            f"- {expect_finding_str(finding, 'agent_name')}: "
            f"{expect_finding_str(finding, 'patient_reply_guidance')}"
        )
    return "\n".join(sections)


def expect_finding_str(finding: dict[str, object], field_name: str) -> str:
    """Return one string field from a serialized aggregate finding."""
    value = finding.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"Expected aggregate finding field '{field_name}' as string.")
    return value


async def start_cluster(
    *,
    args: argparse.Namespace,
    state: DemoUIState,
    telemetry: DemoTelemetry,
    tracker: ReviewCompletionTracker,
    session_id: str,
) -> RuntimeCluster:
    """Start the host, controller worker, and role workers."""
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    copilot_worker: WorkerAgentRuntime | None = None
    cluster: RuntimeCluster | None = None
    try:
        await host.start()
        host_node = update_runtime_node(
            state,
            label="host",
            address=host.address,
            worker_id="control-plane",
            pid=str(os.getpid()),
            node_state="ready",
        )
        telemetry.emit("host", f"→ distributed host listening at {host_node.address}")

        copilot_worker = WorkerAgentRuntime(
            host_address=host.address,
            address=args.worker_bind_address,
            worker_count=args.worker_count,
        )
        register_demo_message_types(copilot_worker)
        copilot_worker.register_factory(
            TELEMETRY_AGENT_TYPE,
            lambda engine: TelemetryReceiverAgent(engine, telemetry),
        )
        copilot_worker.subscribe_exact(
            topic_type=TELEMETRY_TOPIC_TYPE,
            agent_type=TELEMETRY_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATEFUL,
        )
        copilot_worker.register_factory(
            REVIEW_RESULT_AGENT_TYPE,
            lambda engine: ReviewResultAgent(engine, tracker, telemetry),
        )
        copilot_worker.subscribe_exact(
            topic_type=REVIEW_COMPLETED_TOPIC_TYPE,
            agent_type=REVIEW_RESULT_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATEFUL,
        )
        await copilot_worker.start()
        copilot_node = update_runtime_node(
            state,
            label="copilot-worker",
            address=copilot_worker.address,
            worker_id=copilot_worker.worker_id or "unknown",
            pid=str(os.getpid()),
            node_state="ready",
        )
        telemetry.emit(
            "topology",
            (
                f"→ copilot-worker ready at {copilot_node.address} "
                f"worker_id={_short_worker_id(copilot_node.worker_id)} "
                f"pid={os.getpid()}"
            ),
        )

        cluster = RuntimeCluster(host=host, copilot_worker=copilot_worker)
        if args.mode == "multiprocess":
            for role in WORKER_ROLE_ORDER:
                handle = await launch_worker_process(
                    args=args,
                    host_address=host.address,
                    role=role,
                    session_id=session_id,
                    state=state,
                    telemetry=telemetry,
                )
                cluster.worker_processes.append(handle)
        else:
            cluster.local_workers = await start_in_process_workers(
                args=args,
                host_address=host.address,
                state=state,
                telemetry=telemetry,
            )
        return cluster
    except Exception:
        if cluster is not None:
            await stop_cluster(cluster, state)
        else:
            if copilot_worker is not None and copilot_worker.is_running:
                await copilot_worker.stop_when_idle()
            if host.is_running:
                await host.stop_when_idle()
        raise


async def start_in_process_workers(
    *,
    args: argparse.Namespace,
    host_address: str,
    state: DemoUIState,
    telemetry: DemoTelemetry,
) -> list[tuple[str, WorkerAgentRuntime]]:
    """Start specialist and aggregator workers inside the controller process."""
    workers: list[tuple[str, WorkerAgentRuntime]] = []
    for role in WORKER_ROLE_ORDER:
        worker = WorkerAgentRuntime(
            host_address=host_address,
            address=args.worker_bind_address,
            worker_count=args.worker_count,
        )
        register_demo_message_types(worker)
        register_worker_role(worker, role)
        workers.append((WORKER_ROLE_TO_LABEL[role], worker))

    await asyncio.gather(*(worker.start() for _, worker in workers))
    for label, worker in workers:
        node = update_runtime_node(
            state,
            label=label,
            address=worker.address,
            worker_id=worker.worker_id or "unknown",
            pid=str(os.getpid()),
            node_state="ready",
        )
        telemetry.emit(
            "topology",
            (
                f"→ {label} ready at {node.address} "
                f"worker_id={_short_worker_id(node.worker_id)} pid={os.getpid()}"
            ),
        )
    return workers


async def launch_worker_process(
    *,
    args: argparse.Namespace,
    host_address: str,
    role: str,
    session_id: str,
    state: DemoUIState,
    telemetry: DemoTelemetry,
) -> WorkerProcessHandle:
    """Launch one role worker as a subprocess and wait for readiness."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    existing_python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_python_path
        else f"{REPO_ROOT}{os.pathsep}{existing_python_path}"
    )
    command = [
        sys.executable,
        "-m",
        "examples.harness.distributed_clinical_inbox_copilot.worker",
        "--role",
        role,
        "--host-address",
        host_address,
        "--bind-address",
        args.worker_bind_address,
        "--worker-count",
        str(args.worker_count),
        "--session-id",
        session_id,
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=REPO_ROOT,
        env=env,
    )
    try:
        ready = await read_worker_ready(
            process,
            role=role,
            timeout_seconds=args.worker_start_timeout_seconds,
        )
    except Exception:
        await stop_worker_process(
            WorkerProcessHandle(
                role=role,
                label=WORKER_ROLE_TO_LABEL[role],
                process=process,
            )
        )
        raise
    label = str(ready["label"])
    node = update_runtime_node(
        state,
        label=label,
        address=str(ready["address"]),
        worker_id=str(ready["worker_id"]),
        pid=str(ready["pid"]),
        node_state="ready",
    )
    telemetry.emit(
        "topology",
        (
            f"→ {label} subprocess ready at {node.address} "
            f"worker_id={_short_worker_id(node.worker_id)} pid={node.pid}"
        ),
    )
    handle = WorkerProcessHandle(
        role=role,
        label=label,
        process=process,
    )
    handle.stderr_task = asyncio.create_task(drain_process_stderr(handle, telemetry))
    return handle


async def read_worker_ready(
    process: asyncio.subprocess.Process,
    *,
    role: str,
    timeout_seconds: float,
) -> dict[str, object]:
    """Read and validate one subprocess readiness record."""
    if process.stdout is None:
        raise RuntimeError(f"Worker process for role '{role}' has no stdout pipe.")
    try:
        raw_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timed out waiting for worker process role '{role}' to become ready."
        ) from exc
    if not raw_line:
        return_code = await process.wait()
        raise RuntimeError(
            f"Worker process role '{role}' exited before readiness "
            f"with code {return_code}."
        )

    decoded: object = json.loads(raw_line.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Worker process role '{role}' returned invalid readiness.")
    payload = cast(dict[str, object], decoded)
    event_type = payload.get("type")
    if not isinstance(event_type, str):
        raise RuntimeError(
            f"Worker process role '{role}' returned readiness without an event type."
        )
    if event_type == "worker_error":
        error = payload.get("error", "unknown error")
        raise RuntimeError(
            f"Worker process role '{role}' failed before readiness: {error}"
        )
    if event_type != "worker_ready":
        raise RuntimeError(
            f"Worker process role '{role}' returned unexpected event: {payload!r}"
        )
    return payload


async def drain_process_stderr(
    handle: WorkerProcessHandle,
    telemetry: DemoTelemetry,
) -> None:
    """Forward subprocess stderr lines into the UI timeline."""
    if handle.process.stderr is None:
        return
    while True:
        raw_line = await handle.process.stderr.readline()
        if not raw_line:
            return
        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            telemetry.emit("topology", f"{handle.label} stderr: {line}")


async def stop_cluster(cluster: RuntimeCluster, state: DemoUIState) -> None:
    """Stop child processes, local workers, controller worker, and host."""
    for handle in reversed(cluster.worker_processes):
        await stop_worker_process(handle)
        update_runtime_node(state, label=handle.label, node_state="stopped")

    for label, worker in reversed(cluster.local_workers):
        if worker.is_running:
            await worker.stop_when_idle()
        update_runtime_node(state, label=label, node_state="stopped")

    if cluster.copilot_worker.is_running:
        await cluster.copilot_worker.stop_when_idle()
    update_runtime_node(state, label="copilot-worker", node_state="stopped")

    if cluster.host.is_running:
        await cluster.host.stop_when_idle()
    update_runtime_node(state, label="host", node_state="stopped")


async def stop_worker_process(handle: WorkerProcessHandle) -> None:
    """Terminate one subprocess worker and clean up drain tasks."""
    if handle.process.returncode is None:
        handle.process.terminate()
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=5.0)
        except TimeoutError:
            handle.process.kill()
            await handle.process.wait()

    if handle.stderr_task is not None:
        handle.stderr_task.cancel()
        await asyncio.gather(handle.stderr_task, return_exceptions=True)


async def publish_review_and_wait(
    *,
    args: argparse.Namespace,
    inputs: DemoInputs,
    session: SessionState,
    telemetry: DemoTelemetry,
    state: DemoUIState,
    tracker: ReviewCompletionTracker,
    copilot_worker: WorkerAgentRuntime,
    patient_message: str,
) -> AggregatedClinicalReview:
    """Publish one clinical review request and wait for distributed fan-in."""
    if session.chart_snapshot is None:
        raise RuntimeError("Chart snapshot must be loaded before parallel review.")

    session.review_counter += 1
    review_id = f"review-{session.review_counter:02d}"
    tracker.register(review_id)
    telemetry.emit(
        "system",
        f"→ publishing {review_id} through distributed host session={session.session_id}",
    )
    for specialist_name in SPECIALIST_NAMES:
        worker_label = SPECIALIST_WORKER_LABELS[specialist_name]
        worker_node = state.topology_nodes[worker_label]
        telemetry.emit(
            "system",
            f"• Queued {specialist_name} on {worker_label}",
            status_target=specialist_name,
            status_state="queued",
            status_detail=f"Queued on {worker_node.address} pid={worker_node.pid}",
        )

    request = ClinicalReviewTask(
        session_id=session.session_id,
        review_id=review_id,
        clinician_name=inputs.clinician_name,
        patient_label=inputs.patient_label,
        patient_message=patient_message,
        chart_snapshot=session.chart_snapshot,
    )
    ack = await copilot_worker.publish_message(
        request,
        topic=TopicId.from_values(
            type_value=REVIEW_TOPIC_TYPE,
            route_key=review_id,
        ),
    )
    telemetry.emit(
        "system",
        f"→ distributed fan-out enqueued {ack.enqueued_recipient_count} deliveries",
    )
    if ack.enqueued_recipient_count != len(SPECIALIST_NAMES):
        raise RuntimeError(
            "Distributed review fan-out expected "
            f"{len(SPECIALIST_NAMES)} deliveries, got "
            f"{ack.enqueued_recipient_count}."
        )

    return await tracker.wait_for_result(
        review_id,
        timeout_seconds=args.timeout_seconds,
    )


async def run_smoke_review(
    *,
    args: argparse.Namespace,
    inputs: DemoInputs,
    state: DemoUIState,
    telemetry: DemoTelemetry,
    tracker: ReviewCompletionTracker,
    session: SessionState,
    cluster: RuntimeCluster,
) -> None:
    """Run a model-free distributed review through the same worker topology."""
    session.chart_snapshot = build_chart_snapshot(inputs)
    review = await publish_review_and_wait(
        args=args,
        inputs=inputs,
        session=session,
        telemetry=telemetry,
        state=state,
        tracker=tracker,
        copilot_worker=cluster.copilot_worker,
        patient_message=inputs.patient_message,
    )
    state.final_output = format_review_for_model(review)
    CONSOLE.print(
        f"smoke_review={review.review_id} findings={len(review.findings)} "
        f"urgent={review.urgent_flag} mode={state.process_mode}"
    )


async def run_model_demo(
    *,
    args: argparse.Namespace,
    inputs: DemoInputs,
    state: DemoUIState,
    telemetry: DemoTelemetry,
    tracker: ReviewCompletionTracker,
    session: SessionState,
    cluster: RuntimeCluster,
) -> None:
    """Run the streamed OpenAI-backed clinical copilot demo."""
    api_key = os.environ["OPENAI_API_KEY"]
    model = ResponsesClient(
        config=Config(
            api_key=api_key,
            model=args.model,
        )
    )

    @as_tool
    async def load_patient_snapshot(patient_label: str) -> str:
        """Load the concise chart snapshot for the current patient."""
        snapshot = build_chart_snapshot(inputs)
        session.chart_snapshot = snapshot
        telemetry.emit(
            "tool",
            f"→ loaded chart snapshot for {patient_label}",
        )
        return snapshot

    @as_tool
    async def launch_parallel_review(patient_message: str) -> str:
        """Launch the parallel clinical specialist review for the inbox message."""
        review = await publish_review_and_wait(
            args=args,
            inputs=inputs,
            session=session,
            telemetry=telemetry,
            state=state,
            tracker=tracker,
            copilot_worker=cluster.copilot_worker,
            patient_message=patient_message,
        )
        return format_review_for_model(review)

    descriptor = AgentDescriptor(
        name="Clinical Inbox Copilot",
        description=(
            "Streams clinician-facing reasoning, tools, and synthesis for one "
            "mock patient inbox review."
        ),
        model=model,
        model_args={"reasoning": {"effort": "medium", "summary": "detailed"}},
        instructions=textwrap.dedent(f"""
            You are Clinical Inbox Copilot assisting {inputs.clinician_name}.
            This is a mock clinical workflow demo, not autonomous diagnosis.

            Follow this exact sequence:
            1. Start with a short two-step preamble about what you will check.
            2. Call `load_patient_snapshot` exactly once using the patient label.
            3. Call `launch_parallel_review` exactly once using the patient inbox message.
            4. After both tools return, produce:
               - `Clinician View` with exactly 3 bullets
               - `Suggested Next Step` with one sentence
               - `Draft Reply To Patient` with one concise paragraph under 90 words

            Constraints:
            - Use only the facts returned by the tools.
            - If the review shows urgent risk, say same-day escalation is needed.
            - Keep the tone concise, clinical, and operational.
            """).strip(),
        tools=Tools(
            tools=[load_patient_snapshot, launch_parallel_review],
            parallel_tool_calls=False,
            tool_call_timeout=max(args.timeout_seconds + 10.0, 30.0),
            tool_call_max_retries=0,
            tool_call_limits={
                "load_patient_snapshot": 1,
                "launch_parallel_review": 1,
            },
        ),
    )
    agent = DefaultAgent(
        descriptor=descriptor,
        runtime=cluster.copilot_worker,
        hooks=InboxCopilotHooks(telemetry),
    )

    stream = await agent.run_stream(build_user_prompt(inputs))
    telemetry_done = asyncio.Event()
    refresh_done = asyncio.Event()

    with Live(
        build_dashboard(state),
        console=CONSOLE,
        refresh_per_second=10,
        screen=False,
    ) as live:
        telemetry_task = asyncio.create_task(
            drain_telemetry(
                telemetry,
                state,
                stop_when_idle=telemetry_done,
            )
        )
        refresh_task = asyncio.create_task(
            refresh_dashboard(
                live,
                state,
                stop_when_done=refresh_done,
            )
        )
        try:
            await consume_stream(stream, state)
            result: RunResult = await stream.result()
            final_output = (
                result.final_output
                if isinstance(result.final_output, str)
                else str(result.final_output)
            )
            state.final_output = final_output
        finally:
            telemetry_done.set()
            await telemetry_task
            refresh_done.set()
            await refresh_task


async def run_demo(args: argparse.Namespace) -> None:
    """Run the complete clinical inbox copilot demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    if args.multiprocess:
        args.mode = "multiprocess"

    inputs = resolve_inputs(args)
    telemetry = DemoTelemetry()
    state = DemoUIState(inputs=inputs, process_mode=args.mode)
    tracker = ReviewCompletionTracker()
    session = SessionState(session_id=f"clinical-{uuid4().hex[:10]}")
    cluster: RuntimeCluster | None = None
    try:
        cluster = await start_cluster(
            args=args,
            state=state,
            telemetry=telemetry,
            tracker=tracker,
            session_id=session.session_id,
        )
        if args.smoke_review:
            await run_smoke_review(
                args=args,
                inputs=inputs,
                state=state,
                telemetry=telemetry,
                tracker=tracker,
                session=session,
                cluster=cluster,
            )
        else:
            await run_model_demo(
                args=args,
                inputs=inputs,
                state=state,
                telemetry=telemetry,
                tracker=tracker,
                session=session,
                cluster=cluster,
            )
    finally:
        if cluster is not None:
            await stop_cluster(cluster, state)

    print_final_summary(state)


def _short_worker_id(worker_id: str) -> str:
    """Return a compact worker id for dense terminal tables."""
    if worker_id in {"pending", "control-plane"}:
        return worker_id
    return worker_id[:8]


def main() -> None:
    """Run the example from the command line."""
    parser = build_parser()
    asyncio.run(run_demo(parser.parse_args()))


if __name__ == "__main__":
    main()
