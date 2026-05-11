"""Rich terminal UI helpers for the distributed clinical inbox demo."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentlane.harness import RunnerHooks, RunState, RunStream, Task
from agentlane.models import (
    MessageDict,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventKind,
    ToolCall,
)
from examples.harness.distributed_clinical_inbox_copilot.messages import (
    SPECIALIST_NAMES,
    SPECIALIST_WORKER_LABELS,
    TOPOLOGY_NODE_ORDER,
    DemoEvent,
    DemoInputs,
    RuntimeNode,
    build_topology_nodes,
)

ACTOR_STYLES = {
    "host": "bold cyan",
    "system": "bold cyan",
    "tool": "bold magenta",
    "stream": "bold blue",
    "topology": "bold cyan",
    "copilot-worker": "bold green",
    "aggregator": "bold white",
    "med-safety-agent": "bold yellow",
    "guideline-agent": "bold green",
    "chart-history-agent": "bold blue",
    "patient-comms-agent": "bold magenta",
}

STATUS_STYLES = {
    "idle": "dim",
    "starting": "yellow",
    "queued": "yellow",
    "running": "cyan",
    "ready": "green",
    "done": "green",
    "stopped": "dim",
}

CONSOLE = Console()


@dataclass(slots=True)
class DemoUIState:
    """Live render state for the demo dashboard."""

    inputs: DemoInputs
    process_mode: str
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
    topology_nodes: dict[str, RuntimeNode] = field(default_factory=build_topology_nodes)
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


def build_dashboard(state: DemoUIState) -> Layout:
    """Render the current live dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="stream", ratio=3),
        Layout(name="bottom", ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="topology", ratio=3),
        Layout(name="specialists", ratio=3),
        Layout(name="timeline", ratio=4),
    )
    layout["header"].update(build_header_panel(state))
    layout["stream"].update(build_stream_panel(state))
    layout["topology"].update(build_topology_panel(state))
    layout["specialists"].update(build_specialist_panel(state))
    layout["timeline"].update(build_timeline_panel(state))
    return layout


def build_header_panel(state: DemoUIState) -> Panel:
    """Render the top banner with scenario context."""
    header_table = Table.grid(padding=(0, 2))
    header_table.add_row("[bold]Clinician[/bold]", state.inputs.clinician_name)
    header_table.add_row("[bold]Patient[/bold]", state.inputs.patient_label)
    header_table.add_row("[bold]Message[/bold]", state.inputs.patient_message)
    header_table.add_row("[bold]Process mode[/bold]", state.process_mode)
    return Panel(
        header_table,
        title="[bold cyan]Distributed Clinical Inbox Copilot[/bold cyan]",
        subtitle="Distributed host/workers + streamed harness run + fan-out/fan-in",
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


def build_topology_panel(state: DemoUIState) -> Panel:
    """Render the distributed host and worker topology."""
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Node")
    table.add_column("State")
    table.add_column("Host:Port")
    table.add_column("Worker ID")
    table.add_column("PID")
    for label in TOPOLOGY_NODE_ORDER:
        node = state.topology_nodes[label]
        state_style = STATUS_STYLES.get(node.state, "white")
        table.add_row(
            node.label,
            f"[{state_style}]{node.state}[/{state_style}]",
            node.address,
            _short_worker_id(node.worker_id),
            node.pid,
        )
    return Panel(
        table,
        title="[bold cyan]Distributed Runtime[/bold cyan]",
        subtitle="host:port, worker id, and process id",
        border_style="cyan",
    )


def build_specialist_panel(state: DemoUIState) -> Panel:
    """Render the parallel specialist status board."""
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Agent")
    table.add_column("State")
    table.add_column("Worker")
    table.add_column("Detail")
    for agent_name in SPECIALIST_NAMES:
        status, detail = state.specialist_statuses[agent_name]
        state_style = STATUS_STYLES.get(status, "white")
        worker_label = SPECIALIST_WORKER_LABELS[agent_name]
        node = state.topology_nodes[worker_label]
        table.add_row(
            agent_name,
            f"[{state_style}]{status}[/{state_style}]",
            (
                f"{worker_label}\n{node.address}\n"
                f"{_short_worker_id(node.worker_id)}\npid={node.pid}"
            ),
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


def _short_worker_id(worker_id: str) -> str:
    """Return a compact worker id for dense terminal tables."""
    if worker_id in {"pending", "control-plane"}:
        return worker_id
    return worker_id[:8]


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

        apply_demo_event(state, event)


def apply_demo_event(state: DemoUIState, event: DemoEvent) -> None:
    """Apply one distributed or local UI event to render state."""
    actor_style = ACTOR_STYLES.get(event.actor, "white")
    state.timeline.append(
        f"[dim][{event.timestamp}][/dim] "
        f"[{actor_style}]{event.actor:<20}[/{actor_style}] "
        f"[dim]|[/dim] {event.message}"
    )
    if (
        event.status_target is not None
        and event.status_target in state.specialist_statuses
    ):
        current_state, current_detail = state.specialist_statuses[event.status_target]
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


def update_runtime_node(
    state: DemoUIState,
    *,
    label: str,
    address: str | None = None,
    worker_id: str | None = None,
    pid: str | None = None,
    node_state: str | None = None,
) -> RuntimeNode:
    """Update one topology node shown in the live dashboard."""
    node = state.topology_nodes[label]
    if address is not None:
        node.address = address
    if worker_id is not None:
        node.worker_id = worker_id
    if pid is not None:
        node.pid = pid
    if node_state is not None:
        node.state = node_state
    return node


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


def _coerce_stream_event(event: object) -> ModelStreamEvent:
    """Assert that one streamed item is a `ModelStreamEvent`."""
    if not isinstance(event, ModelStreamEvent):
        raise TypeError(f"Expected ModelStreamEvent, got {type(event)!r}.")
    return event


def _tool_name(tool_call: object) -> str:
    """Extract the function name from one canonical tool call."""
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    return name if isinstance(name, str) and name else "unknown_tool"
