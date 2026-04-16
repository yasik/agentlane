"""Interactive streamed clinical copilot demo with parallel specialist review."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from agentlane_openai import ResponsesClient
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentlane.harness import (
    AgentDescriptor,
    RunnerHooks,
    RunResult,
    RunState,
    RunStream,
    Task,
)
from agentlane.harness.agents import DefaultAgent
from agentlane.messaging import (
    AgentId,
    DeliveryMode,
    DeliveryStatus,
    MessageContext,
    TopicId,
)
from agentlane.models import (
    Config,
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
    Tools,
    as_tool,
)
from agentlane.runtime import BaseAgent, Engine, SingleThreadedRuntimeEngine, on_message

MODEL_NAME = "gpt-5.4-mini"

MED_SAFETY_AGENT_TYPE = "demo.clinical.med_safety"
GUIDELINE_AGENT_TYPE = "demo.clinical.guideline"
CHART_HISTORY_AGENT_TYPE = "demo.clinical.chart_history"
PATIENT_COMMS_AGENT_TYPE = "demo.clinical.patient_comms"
AGGREGATOR_AGENT_TYPE = "demo.clinical.aggregator"

REVIEW_TOPIC_TYPE = "demo.clinical.review_requested"
RESULT_TOPIC_TYPE = "demo.clinical.review_result"

SPECIALIST_NAMES = (
    "med-safety-agent",
    "guideline-agent",
    "chart-history-agent",
    "patient-comms-agent",
)

ACTOR_STYLES = {
    "system": "bold cyan",
    "tool": "bold magenta",
    "stream": "bold blue",
    "aggregator": "bold white",
    "med-safety-agent": "bold yellow",
    "guideline-agent": "bold green",
    "chart-history-agent": "bold blue",
    "patient-comms-agent": "bold magenta",
}

STATUS_STYLES = {
    "idle": "dim",
    "queued": "yellow",
    "running": "cyan",
    "done": "green",
}

CONSOLE = Console()


@dataclass(slots=True, frozen=True)
class DemoInputs:
    """Interactive inputs gathered from the clinician."""

    clinician_name: str
    patient_label: str
    patient_message: str


@dataclass(slots=True)
class SessionState:
    """Mutable session state shared between tools."""

    chart_snapshot: str | None = None
    review_counter: int = 0


@dataclass(slots=True)
class ClinicalReviewTask:
    """Publish payload delivered to each specialist reviewer."""

    review_id: str
    clinician_name: str
    patient_label: str
    patient_message: str
    chart_snapshot: str


@dataclass(slots=True)
class SpecialistFinding:
    """One specialist output published back to the aggregator."""

    review_id: str
    agent_name: str
    headline: str
    detail: str
    patient_reply_guidance: str
    urgent_flag: bool


@dataclass(slots=True)
class AggregatedClinicalReview:
    """Merged specialist review for one inbox message."""

    review_id: str
    findings: dict[str, SpecialistFinding]
    urgent_flag: bool


@dataclass(slots=True, frozen=True)
class DemoEvent:
    """One structured UI event emitted by hooks or runtime agents."""

    timestamp: str
    actor: str
    message: str
    status_target: str | None = None
    status_state: str | None = None
    status_detail: str | None = None


@dataclass(slots=True)
class DemoUIState:
    """Live render state for the demo dashboard."""

    inputs: DemoInputs
    stream_text: str = ""
    reasoning_text: str = ""
    last_reasoning_text: str = ""
    tool_arguments_text: str = ""
    phases: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    timeline: deque[str] = field(default_factory=lambda: deque(maxlen=16))
    specialist_statuses: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            name: ("idle", "Waiting for delegation") for name in SPECIALIST_NAMES
        }
    )
    final_output: str | None = None

    def append_timeline(
        self, actor: str, message: str, *, style: str | None = None
    ) -> None:
        """Append one formatted timeline line."""
        actor_style = style or ACTOR_STYLES.get(actor, "white")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.timeline.append(
            f"[dim][{timestamp}][/dim] [{actor_style}]{actor:<20}[/{actor_style}] "
            f"[dim]|[/dim] {message}"
        )


class DemoTelemetry:
    """Queue-backed event emitter for the live dashboard."""

    def __init__(self) -> None:
        """Initialize the async event queue."""
        self._queue: asyncio.Queue[DemoEvent] = asyncio.Queue()

    @property
    def queue(self) -> asyncio.Queue[DemoEvent]:
        """Return the underlying event queue."""
        return self._queue

    def emit(
        self,
        actor: str,
        message: str,
        *,
        status_target: str | None = None,
        status_state: str | None = None,
        status_detail: str | None = None,
    ) -> None:
        """Emit one event immediately without blocking the caller."""
        self._queue.put_nowait(
            DemoEvent(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                actor=actor,
                message=message,
                status_target=status_target,
                status_state=status_state,
                status_detail=status_detail,
            )
        )


class InboxCopilotHooks(RunnerHooks):
    """Surface top-level harness lifecycle events in the dashboard."""

    def __init__(self, telemetry: DemoTelemetry) -> None:
        """Initialize hooks with one shared telemetry sink."""
        self._telemetry = telemetry

    async def on_agent_start(self, task: Task, state: RunState) -> None:
        """Record the start of the top-level run."""
        _ = task
        _ = state
        self._telemetry.emit("system", "→ top-level agent accepted the inbox message")

    async def on_llm_start(
        self,
        task: Task,
        messages: list[MessageDict],
    ) -> None:
        """Record the start of one model turn."""
        _ = task
        _ = messages
        self._telemetry.emit("system", "→ preparing the next top-level model turn")

    async def on_llm_end(self, task: Task, response: ModelResponse) -> None:
        """Record the end of one model turn."""
        _ = task
        _ = response
        self._telemetry.emit("system", "→ top-level model turn completed")

    async def on_tool_call_start(self, task: Task, tool_call: ToolCall) -> None:
        """Record one tool invocation start."""
        _ = task
        tool_name = _tool_name(tool_call)
        self._telemetry.emit("tool", f"→ calling tool `{tool_name}`")

    async def on_tool_call_end(
        self,
        task: Task,
        tool_call: ToolCall,
        result: object,
    ) -> None:
        """Record one tool invocation finish."""
        _ = task
        _ = result
        tool_name = _tool_name(tool_call)
        self._telemetry.emit("tool", f"→ tool `{tool_name}` completed")


class ReviewCompletionTracker:
    """Waits for the stateful aggregator to finish a review id."""

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


class MedSafetyAgent(BaseAgent):
    """Specialist that reviews likely medication-safety signals."""

    def __init__(self, engine: Engine, telemetry: DemoTelemetry) -> None:
        """Initialize agent dependencies."""
        super().__init__(engine)
        self._telemetry = telemetry

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Review medication-safety concerns and publish one finding."""
        self._telemetry.emit(
            "med-safety-agent",
            "│ med-safety-agent      checking for medication-related hypoglycemia",
            status_target="med-safety-agent",
            status_state="running",
            status_detail="Reviewing medication safety signals",
        )
        await asyncio.sleep(0.35)
        finding = SpecialistFinding(
            review_id=payload.review_id,
            agent_name="med-safety-agent",
            headline="Medication safety",
            detail=(
                "The reported dizziness plus glucose values under 70 raise concern "
                "for symptomatic hypoglycemia. The chart pattern makes glipizide the "
                "most likely immediate contributor to low readings after the recent "
                "regimen change."
            ),
            patient_reply_guidance=(
                "Tell the patient not to make unsupervised medication changes, but to "
                "treat any low sugar per their plan and wait for a same-day clinician "
                "reply."
            ),
            urgent_flag=True,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        self._telemetry.emit(
            "med-safety-agent",
            "└ med-safety-agent      flagged hypoglycemia risk and likely culprit",
            status_target="med-safety-agent",
            status_state="done",
            status_detail="Flagged symptomatic hypoglycemia risk",
        )
        return finding


class GuidelineAgent(BaseAgent):
    """Specialist that summarizes the relevant care-path guidance."""

    def __init__(self, engine: Engine, telemetry: DemoTelemetry) -> None:
        """Initialize agent dependencies."""
        super().__init__(engine)
        self._telemetry = telemetry

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Review the message against guideline-like escalation logic."""
        self._telemetry.emit(
            "guideline-agent",
            "│ guideline-agent       reviewing escalation guidance for low glucose",
            status_target="guideline-agent",
            status_state="running",
            status_detail="Matching guideline-style escalation signals",
        )
        await asyncio.sleep(0.2)
        finding = SpecialistFinding(
            review_id=payload.review_id,
            agent_name="guideline-agent",
            headline="Guideline alignment",
            detail=(
                "Recurrent symptomatic glucose values below 70 should trigger "
                "same-day clinical review and clear escalation instructions if "
                "symptoms worsen, the patient cannot keep glucose above target, "
                "or new confusion, syncope, or chest pain appears."
            ),
            patient_reply_guidance=(
                "Include explicit red-flag instructions for severe symptoms or "
                "persistent lows that do not improve quickly."
            ),
            urgent_flag=True,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        self._telemetry.emit(
            "guideline-agent",
            "└ guideline-agent       recommended same-day clinician review",
            status_target="guideline-agent",
            status_state="done",
            status_detail="Recommended same-day clinical review",
        )
        return finding


class ChartHistoryAgent(BaseAgent):
    """Specialist that extracts relevant chart context."""

    def __init__(self, engine: Engine, telemetry: DemoTelemetry) -> None:
        """Initialize agent dependencies."""
        super().__init__(engine)
        self._telemetry = telemetry

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Summarize the most relevant chart history for the inbox message."""
        self._telemetry.emit(
            "chart-history-agent",
            "│ chart-history-agent   extracting recent labs, meds, and symptoms",
            status_target="chart-history-agent",
            status_state="running",
            status_detail="Pulling chart context",
        )
        await asyncio.sleep(0.28)
        finding = SpecialistFinding(
            review_id=payload.review_id,
            agent_name="chart-history-agent",
            headline="Chart context",
            detail=(
                f"The chart snapshot for {payload.patient_label} shows type 2 diabetes, "
                "glipizide plus metformin, and semaglutide started two weeks ago. Recent "
                "home readings in the low 60s line up with the patient message and there "
                "is no ED visit or severe-event documentation in the mock chart."
            ),
            patient_reply_guidance=(
                "Ask the patient to confirm their latest glucose reading and whether the "
                "dizziness is improving after eating or treating the low."
            ),
            urgent_flag=False,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        self._telemetry.emit(
            "chart-history-agent",
            "└ chart-history-agent   summarized the mock chart and recent glucose trend",
            status_target="chart-history-agent",
            status_state="done",
            status_detail="Summarized chart context",
        )
        return finding


class PatientCommsAgent(BaseAgent):
    """Specialist that suggests patient-friendly messaging."""

    def __init__(self, engine: Engine, telemetry: DemoTelemetry) -> None:
        """Initialize agent dependencies."""
        super().__init__(engine)
        self._telemetry = telemetry

    @on_message
    async def handle(
        self,
        payload: ClinicalReviewTask,
        context: MessageContext,
    ) -> object:
        """Draft plain-language patient communication guidance."""
        self._telemetry.emit(
            "patient-comms-agent",
            "│ patient-comms-agent   drafting a plain-language patient response",
            status_target="patient-comms-agent",
            status_state="running",
            status_detail="Drafting patient-friendly guidance",
        )
        await asyncio.sleep(0.32)
        finding = SpecialistFinding(
            review_id=payload.review_id,
            agent_name="patient-comms-agent",
            headline="Patient communication",
            detail=(
                "The response should acknowledge the symptoms, advise the patient to "
                "treat any low sugar per their plan, and set expectations that the care "
                "team will review medication safety the same day."
            ),
            patient_reply_guidance=(
                "Use calm plain language, ask about the current glucose value, and tell "
                "the patient to seek urgent help for severe weakness, confusion, fainting, "
                "or symptoms that do not improve."
            ),
            urgent_flag=False,
        )
        await self.publish_message(
            finding,
            topic=TopicId.from_values(
                type_value=RESULT_TOPIC_TYPE,
                route_key=payload.review_id,
            ),
            correlation_id=context.correlation_id,
        )
        self._telemetry.emit(
            "patient-comms-agent",
            "└ patient-comms-agent   prepared patient-friendly response guidance",
            status_target="patient-comms-agent",
            status_state="done",
            status_detail="Prepared patient response guidance",
        )
        return finding


class ReviewAggregatorAgent(BaseAgent):
    """Stateful aggregator keyed by review id."""

    def __init__(
        self,
        engine: Engine,
        telemetry: DemoTelemetry,
        expected_finding_count: int,
    ) -> None:
        """Initialize the aggregator dependencies."""
        super().__init__(engine)
        self._telemetry = telemetry
        self._expected_finding_count = expected_finding_count
        self._findings: dict[str, SpecialistFinding] = {}

    @on_message
    async def handle(
        self,
        payload: SpecialistFinding,
        context: MessageContext,
    ) -> object:
        """Collect specialist findings and resolve the tracker when complete."""
        _ = context
        self._findings[payload.agent_name] = payload
        self._telemetry.emit(
            "aggregator",
            (
                "→ aggregator collected "
                f"{len(self._findings)}/{self._expected_finding_count} findings"
            ),
        )
        if len(self._findings) < self._expected_finding_count:
            return None

        ordered_findings = {
            name: self._findings[name] for name in sorted(self._findings)
        }
        review = AggregatedClinicalReview(
            review_id=self.id.key.value,
            findings=ordered_findings,
            urgent_flag=any(item.urgent_flag for item in ordered_findings.values()),
        )
        self._telemetry.emit(
            "aggregator",
            f"→ review {review.review_id} merged into one summary",
        )
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the demo."""
    parser = argparse.ArgumentParser(
        description="Interactive clinical inbox copilot demo.",
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
        "--worker-count",
        type=int,
        default=8,
        help="Worker count for the local runtime handling specialist reviews.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout when waiting for the specialist aggregator.",
    )
    return parser


def _tool_name(tool_call: object) -> str:
    """Extract the function name from one canonical tool call."""
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    return name if isinstance(name, str) and name else "unknown_tool"


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
            f"- {finding.agent_name} | {finding.headline}: {finding.detail}"
        )
    sections.append("patient_reply_guidance:")
    for finding in review.findings.values():
        sections.append(f"- {finding.agent_name}: {finding.patient_reply_guidance}")
    return "\n".join(sections)


def build_aggregated_review(
    review_id: str,
    findings: list[SpecialistFinding],
) -> AggregatedClinicalReview:
    """Merge direct specialist findings into one aggregated review."""
    ordered_findings = {
        item.agent_name: item
        for item in sorted(findings, key=lambda item: item.agent_name)
    }
    return AggregatedClinicalReview(
        review_id=review_id,
        findings=ordered_findings,
        urgent_flag=any(item.urgent_flag for item in ordered_findings.values()),
    )


def build_dashboard(state: DemoUIState) -> Layout:
    """Render the current live dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="stream", ratio=3),
        Layout(name="bottom", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="specialists", ratio=2),
        Layout(name="timeline", ratio=3),
    )
    layout["header"].update(build_header_panel(state.inputs))
    layout["stream"].update(build_stream_panel(state))
    layout["specialists"].update(build_specialist_panel(state))
    layout["timeline"].update(build_timeline_panel(state))
    return layout


def build_header_panel(inputs: DemoInputs) -> Panel:
    """Render the top banner with scenario context."""
    header_table = Table.grid(padding=(0, 2))
    header_table.add_row("[bold]Clinician[/bold]", inputs.clinician_name)
    header_table.add_row("[bold]Patient[/bold]", inputs.patient_label)
    header_table.add_row("[bold]Message[/bold]", inputs.patient_message)
    return Panel(
        header_table,
        title="[bold cyan]Clinical Inbox Copilot[/bold cyan]",
        subtitle="Top-level streaming + publish fan-out specialist review",
        border_style="cyan",
    )


def build_stream_panel(state: DemoUIState) -> Panel:
    """Render the top-level streamed agent panel."""
    phases_text = (
        " → ".join(state.phases) if state.phases else "awaiting first model phase"
    )
    if state.reasoning_text:
        reasoning_body: object = Text(
            state.reasoning_text,
            style="white",
            overflow="fold",
            no_wrap=False,
        )
    else:
        reasoning_body = Text("Reasoning will appear here as it streams.", style="dim")
    tool_args_text = (
        _tail_text(state.tool_arguments_text, 500) or "No tool arguments emitted yet."
    )
    stream_text = state.stream_text or "Awaiting top-level stream..."
    body = Group(
        Text(f"Model phases: {phases_text}", style="dim"),
        Text(""),
        Text("Reasoning Stream", style="bold yellow"),
        reasoning_body,
        Text(""),
        Text("Tool Arguments", style="bold magenta"),
        Text(tool_args_text, style="white"),
        Text(""),
        Text("Assistant Stream", style="bold green"),
        Text(stream_text, style="white"),
    )
    return Panel(
        body, title="[bold green]Top-Level Agent[/bold green]", border_style="green"
    )


def build_specialist_panel(state: DemoUIState) -> Panel:
    """Render the parallel specialist status board."""
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Agent")
    table.add_column("State")
    table.add_column("Detail")
    for agent_name in SPECIALIST_NAMES:
        status, detail = state.specialist_statuses[agent_name]
        state_style = STATUS_STYLES.get(status, "white")
        table.add_row(
            agent_name,
            f"[{state_style}]{status}[/{state_style}]",
            detail,
        )
    return Panel(
        table,
        title="[bold blue]Parallel Specialist Review[/bold blue]",
        border_style="blue",
    )


def build_timeline_panel(state: DemoUIState) -> Panel:
    """Render the rolling timeline panel."""
    if not state.timeline:
        body: object = Text("Awaiting activity...", style="dim")
    else:
        body = Group(*[Text.from_markup(line) for line in state.timeline])
    return Panel(
        body, title="[bold white]Event Timeline[/bold white]", border_style="white"
    )


def _tail_text(value: str, max_chars: int) -> str:
    """Return the last `max_chars` characters from a string."""
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


def append_reasoning_chunk(state: DemoUIState, reasoning_chunk: str) -> None:
    """Append one reasoning event without rewriting the earlier stream."""
    if not reasoning_chunk:
        return

    if reasoning_chunk == state.last_reasoning_text:
        return

    if state.last_reasoning_text and reasoning_chunk.startswith(
        state.last_reasoning_text
    ):
        state.reasoning_text = reasoning_chunk
    else:
        state.reasoning_text += reasoning_chunk

    state.last_reasoning_text = reasoning_chunk


async def consume_stream(stream: RunStream, state: DemoUIState) -> None:
    """Consume top-level model stream events into the dashboard state."""
    last_openai_phase: str | None = None
    async for event in stream:
        model_event = _coerce_stream_event(event)
        if (
            model_event.kind == ModelStreamEventKind.REASONING
            and model_event.reasoning is not None
        ):
            reasoning_chunk = (
                model_event.reasoning
                if isinstance(model_event.reasoning, str)
                else str(model_event.reasoning)
            )
            append_reasoning_chunk(state, reasoning_chunk)
            continue
        if model_event.kind == ModelStreamEventKind.TEXT_DELTA and model_event.text:
            state.stream_text += model_event.text
            continue
        if (
            model_event.kind == ModelStreamEventKind.TOOL_CALL_ARGUMENTS_DELTA
            and model_event.arguments_delta
        ):
            state.tool_arguments_text += model_event.arguments_delta
            continue
        if (
            model_event.kind == ModelStreamEventKind.PROVIDER
            and model_event.provider_event_type is not None
        ):
            raw_item = getattr(model_event.raw, "item", None)
            raw_phase = getattr(raw_item, "phase", None)
            if isinstance(raw_phase, str) and raw_phase != last_openai_phase:
                last_openai_phase = raw_phase
                state.phases.append(raw_phase)
                state.append_timeline("stream", f"→ model phase `{raw_phase}`")


async def drain_telemetry(
    telemetry: DemoTelemetry,
    state: DemoUIState,
    *,
    stop_when_idle: asyncio.Event,
) -> None:
    """Drain queued telemetry into the live dashboard state."""
    while True:
        if stop_when_idle.is_set() and telemetry.queue.empty():
            return
        try:
            event = await asyncio.wait_for(telemetry.queue.get(), timeout=0.1)
        except TimeoutError:
            continue

        actor_style = ACTOR_STYLES.get(event.actor, "white")
        state.timeline.append(
            f"[dim][{event.timestamp}][/dim] "
            f"[{actor_style}]{event.actor:<20}[/{actor_style}] "
            f"[dim]|[/dim] {event.message}"
        )
        if event.status_target is not None:
            current_state, current_detail = state.specialist_statuses[
                event.status_target
            ]
            state.specialist_statuses[event.status_target] = (
                event.status_state or current_state,
                event.status_detail or current_detail,
            )


async def refresh_dashboard(
    live: Live,
    state: DemoUIState,
    *,
    stop_when_done: asyncio.Event,
) -> None:
    """Refresh the live dashboard at a steady cadence."""
    while not stop_when_done.is_set():
        live.update(build_dashboard(state), refresh=True)
        await asyncio.sleep(0.1)
    live.update(build_dashboard(state), refresh=True)


def _coerce_stream_event(event: object) -> ModelStreamEvent:
    """Assert that one streamed item is a `ModelStreamEvent`."""
    if not isinstance(event, ModelStreamEvent):
        raise TypeError(f"Expected ModelStreamEvent, got {type(event)!r}.")
    return event


def require_specialist_finding(
    agent_name: str,
    outcome: object,
) -> SpecialistFinding:
    """Validate one specialist direct-send outcome and extract the finding."""
    status = getattr(outcome, "status", None)
    if status != DeliveryStatus.DELIVERED:
        error = getattr(outcome, "error", None)
        error_message = getattr(error, "message", "unknown specialist error")
        raise RuntimeError(
            f"{agent_name} failed with status `{status}`: {error_message}"
        )

    response_payload = getattr(outcome, "response_payload", None)
    if not isinstance(response_payload, SpecialistFinding):
        raise RuntimeError(
            f"{agent_name} returned unexpected payload type: {type(response_payload)!r}"
        )
    return response_payload


async def run_demo(args: argparse.Namespace) -> None:
    """Run the complete clinical inbox copilot demo."""
    logging.basicConfig(level=logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
    )
    api_key = os.environ["OPENAI_API_KEY"]
    inputs = resolve_inputs(args)
    telemetry = DemoTelemetry()
    state = DemoUIState(inputs=inputs)
    session = SessionState()

    model = ResponsesClient(
        config=Config(
            api_key=api_key,
            model=args.model,
        )
    )

    runtime = SingleThreadedRuntimeEngine(worker_count=args.worker_count)
    try:
        runtime.register_factory(
            MED_SAFETY_AGENT_TYPE,
            lambda engine: MedSafetyAgent(engine, telemetry),
        )
        runtime.register_factory(
            GUIDELINE_AGENT_TYPE,
            lambda engine: GuidelineAgent(engine, telemetry),
        )
        runtime.register_factory(
            CHART_HISTORY_AGENT_TYPE,
            lambda engine: ChartHistoryAgent(engine, telemetry),
        )
        runtime.register_factory(
            PATIENT_COMMS_AGENT_TYPE,
            lambda engine: PatientCommsAgent(engine, telemetry),
        )
        runtime.register_factory(
            AGGREGATOR_AGENT_TYPE,
            lambda engine: ReviewAggregatorAgent(
                engine,
                telemetry,
                expected_finding_count=len(SPECIALIST_NAMES),
            ),
        )
        runtime.subscribe_exact(
            topic_type=RESULT_TOPIC_TYPE,
            agent_type=AGGREGATOR_AGENT_TYPE,
            delivery_mode=DeliveryMode.STATEFUL,
        )
        await runtime.start()

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
            if session.chart_snapshot is None:
                raise RuntimeError(
                    "Chart snapshot must be loaded before parallel review."
                )

            session.review_counter += 1
            review_id = f"review-{session.review_counter:02d}"
            telemetry.emit(
                "system", f"→ launching parallel specialist review for {review_id}"
            )
            for specialist_name in SPECIALIST_NAMES:
                telemetry.emit(
                    "system",
                    f"• Delegated to {specialist_name}",
                    status_target=specialist_name,
                    status_state="queued",
                    status_detail="Queued for specialist review",
                )

            request = ClinicalReviewTask(
                review_id=review_id,
                clinician_name=inputs.clinician_name,
                patient_label=inputs.patient_label,
                patient_message=patient_message,
                chart_snapshot=session.chart_snapshot,
            )
            specialist_recipients = [
                (
                    "med-safety-agent",
                    AgentId.from_values(MED_SAFETY_AGENT_TYPE, review_id),
                ),
                (
                    "guideline-agent",
                    AgentId.from_values(GUIDELINE_AGENT_TYPE, review_id),
                ),
                (
                    "chart-history-agent",
                    AgentId.from_values(CHART_HISTORY_AGENT_TYPE, review_id),
                ),
                (
                    "patient-comms-agent",
                    AgentId.from_values(PATIENT_COMMS_AGENT_TYPE, review_id),
                ),
            ]
            specialist_outcomes = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        runtime.send_message(request, recipient=recipient)
                        for _, recipient in specialist_recipients
                    ]
                ),
                timeout=args.timeout_seconds,
            )
            findings = [
                require_specialist_finding(agent_name, outcome)
                for (agent_name, _), outcome in zip(
                    specialist_recipients,
                    specialist_outcomes,
                    strict=True,
                )
            ]
            review = build_aggregated_review(review_id, findings)
            telemetry.emit(
                "aggregator",
                f"→ aggregated review ready for {review.review_id} from direct specialist results",
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
            hooks=InboxCopilotHooks(telemetry),
        )

        user_prompt = build_user_prompt(inputs)
        stream = await agent.run_stream(user_prompt)
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
    finally:
        if runtime.is_running:
            await runtime.stop_when_idle()

    print_final_summary(state)


def print_final_summary(state: DemoUIState) -> None:
    """Print the final persisted result after the live dashboard exits."""
    CONSOLE.print()
    CONSOLE.print(
        Panel(
            Text(state.final_output or "No final output captured.", style="white"),
            title="[bold green]Final Response[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    """Run the example from the command line."""
    parser = build_parser()
    asyncio.run(run_demo(parser.parse_args()))


if __name__ == "__main__":
    main()
