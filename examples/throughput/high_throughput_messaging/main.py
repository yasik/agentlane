"""High-throughput demo for execution checks and market data fan-out."""

import argparse
import asyncio
from dataclasses import dataclass
from time import perf_counter, time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentlane.messaging import (
    AgentId,
    DeliveryMode,
    DeliveryStatus,
    MessageContext,
    TopicId,
)
from agentlane.runtime import (
    BaseAgent,
    Engine,
    SingleThreadedRuntimeEngine,
    on_message,
    single_threaded_runtime,
)

RPC_AGENT_TYPE = "demo.execution_risk_worker"
EVENT_AGENT_TYPE = "demo.market_data_worker"
EVENT_TOPIC_TYPE = "demo.throughput.market_data"
CONSOLE = Console()


@dataclass(slots=True, frozen=True)
class DemoConfig:
    """Runtime configuration for the high-throughput market workflow."""

    duration_seconds: float
    """How long the producers should run."""

    rpc_concurrency: int
    """Number of concurrent execution-risk RPC producer tasks."""

    publish_concurrency: int
    """Number of concurrent market-data publish producer tasks."""

    worker_count: int
    """Worker count passed to `SingleThreadedRuntimeEngine`."""

    shard_count: int
    """Number of route-key shards used to spread load across AgentIds."""

    progress_interval_seconds: float
    """How often to print streaming progress output."""

    publish_pause_seconds: float
    """Delay inserted after each publish call to avoid scheduler starvation."""


@dataclass(slots=True)
class RpcRequest:
    """Execution-risk RPC payload model used by producer tasks."""

    producer_id: int
    """Producer task identifier."""

    sequence: int
    """Monotonic sequence number within one producer."""

    created_at_ms: int
    """Producer-side creation timestamp in epoch milliseconds."""


@dataclass(slots=True)
class EventMessage:
    """Market-data publish payload model used by producer tasks."""

    producer_id: int
    """Producer task identifier."""

    sequence: int
    """Monotonic sequence number within one producer."""

    created_at_ms: int
    """Producer-side creation timestamp in epoch milliseconds."""


@dataclass(slots=True, frozen=True)
class StatsSnapshot:
    """Immutable snapshot of demo counters used for reporting."""

    rpc_sent: int
    """Total execution-risk RPC requests emitted by producers."""

    rpc_delivered: int
    """Total execution-risk RPC requests with `DELIVERED` outcome."""

    rpc_failed: int
    """Total execution-risk RPC requests with non-delivered outcome."""

    rpc_handled: int
    """Total execution-risk RPC requests handled by worker agents."""

    publish_sent: int
    """Total market-data publish calls emitted by producers."""

    publish_acked_recipients: int
    """Total market-data recipient enqueue count returned by publish acknowledgments."""

    publish_failed: int
    """Total market-data publish calls that raised an exception."""

    events_handled: int
    """Total market-data publish events handled by worker agents."""


class ThroughputStats:
    """Shared in-process counter store for market workflow metrics."""

    def __init__(self) -> None:
        """Initialize all counters to zero."""
        self._rpc_sent = 0
        self._rpc_delivered = 0
        self._rpc_failed = 0
        self._rpc_handled = 0
        self._publish_sent = 0
        self._publish_acked_recipients = 0
        self._publish_failed = 0
        self._events_handled = 0

    def record_rpc_sent(self) -> None:
        """Increment emitted RPC request counter."""
        self._rpc_sent += 1

    def record_rpc_delivered(self) -> None:
        """Increment successful RPC outcome counter."""
        self._rpc_delivered += 1

    def record_rpc_failed(self) -> None:
        """Increment failed RPC outcome counter."""
        self._rpc_failed += 1

    def record_rpc_handled(self) -> None:
        """Increment RPC handler execution counter."""
        self._rpc_handled += 1

    def record_publish_sent(self) -> None:
        """Increment emitted publish call counter."""
        self._publish_sent += 1

    def record_publish_acked_recipients(self, recipient_count: int) -> None:
        """Increment acknowledged recipient enqueue counter."""
        self._publish_acked_recipients += recipient_count

    def record_publish_failed(self) -> None:
        """Increment publish exception counter."""
        self._publish_failed += 1

    def record_event_handled(self) -> None:
        """Increment publish event handler execution counter."""
        self._events_handled += 1

    def snapshot(self) -> StatsSnapshot:
        """Return immutable snapshot of all counters."""
        return StatsSnapshot(
            rpc_sent=self._rpc_sent,
            rpc_delivered=self._rpc_delivered,
            rpc_failed=self._rpc_failed,
            rpc_handled=self._rpc_handled,
            publish_sent=self._publish_sent,
            publish_acked_recipients=self._publish_acked_recipients,
            publish_failed=self._publish_failed,
            events_handled=self._events_handled,
        )


class RpcWorkerAgent(BaseAgent):
    """Worker handling direct execution-risk RPC requests."""

    def __init__(self, engine: Engine, stats: ThroughputStats) -> None:
        """Initialize execution-risk worker with runtime engine and shared stats."""
        super().__init__(engine)
        self._stats = stats

    @on_message
    async def handle(self, payload: RpcRequest, context: MessageContext) -> object:
        """Process one execution-risk RPC request payload."""
        _ = context
        _ = payload.created_at_ms
        self._stats.record_rpc_handled()
        return payload.sequence


class EventWorkerAgent(BaseAgent):
    """Worker handling market-data publish messages."""

    def __init__(self, engine: Engine, stats: ThroughputStats) -> None:
        """Initialize market-data worker with runtime engine and shared stats."""
        super().__init__(engine)
        self._stats = stats

    @on_message
    async def handle(self, payload: EventMessage, context: MessageContext) -> object:
        """Process one market-data publish payload."""
        _ = context
        _ = payload.created_at_ms
        self._stats.record_event_handled()
        return None


async def run_rpc_producer(
    *,
    producer_id: int,
    runtime: SingleThreadedRuntimeEngine,
    stats: ThroughputStats,
    stop_at: float,
    shard_count: int,
) -> None:
    """Emit direct execution-risk RPC requests until stop time is reached."""
    sequence = 0
    while perf_counter() < stop_at:
        route_key = f"rpc-{(producer_id + sequence) % shard_count}"
        payload = RpcRequest(
            producer_id=producer_id,
            sequence=sequence,
            created_at_ms=int(time() * 1000),
        )
        stats.record_rpc_sent()
        outcome = await runtime.send_message(
            payload,
            recipient=AgentId.from_values(RPC_AGENT_TYPE, route_key),
        )
        if outcome.status == DeliveryStatus.DELIVERED:
            stats.record_rpc_delivered()
        else:
            stats.record_rpc_failed()
        sequence += 1


async def run_publish_producer(
    *,
    producer_id: int,
    runtime: SingleThreadedRuntimeEngine,
    stats: ThroughputStats,
    stop_at: float,
    shard_count: int,
    publish_pause_seconds: float,
) -> None:
    """Emit market-data publish events until stop time is reached."""
    sequence = 0
    while perf_counter() < stop_at:
        route_key = f"pub-{(producer_id + sequence) % shard_count}"
        payload = EventMessage(
            producer_id=producer_id,
            sequence=sequence,
            created_at_ms=int(time() * 1000),
        )
        stats.record_publish_sent()
        try:
            ack = await runtime.publish_message(
                payload,
                topic=TopicId.from_values(
                    type_value=EVENT_TOPIC_TYPE,
                    route_key=route_key,
                ),
            )
        except Exception:  # noqa: BLE001
            stats.record_publish_failed()
        else:
            stats.record_publish_acked_recipients(ack.enqueued_recipient_count)
        if publish_pause_seconds > 0:
            await asyncio.sleep(publish_pause_seconds)
        sequence += 1


def compute_delta(current: StatsSnapshot, previous: StatsSnapshot) -> StatsSnapshot:
    """Return field-wise difference between two snapshots."""
    return StatsSnapshot(
        rpc_sent=current.rpc_sent - previous.rpc_sent,
        rpc_delivered=current.rpc_delivered - previous.rpc_delivered,
        rpc_failed=current.rpc_failed - previous.rpc_failed,
        rpc_handled=current.rpc_handled - previous.rpc_handled,
        publish_sent=current.publish_sent - previous.publish_sent,
        publish_acked_recipients=(
            current.publish_acked_recipients - previous.publish_acked_recipients
        ),
        publish_failed=current.publish_failed - previous.publish_failed,
        events_handled=current.events_handled - previous.events_handled,
    )


def format_count(value: int) -> str:
    """Format integer counters with thousands separators."""
    return f"{value:,}"


def format_rate(value: float) -> str:
    """Format floating-point rates with one decimal place."""
    return f"{value:,.1f}/s"


def print_progress_header() -> None:
    """Print one progress table header for streaming rows."""
    CONSOLE.print(
        "[bold cyan] elapsed[/bold cyan] | "
        "[bold cyan]risk rpc sent/delivered/failed[/bold cyan] | "
        "[bold cyan]market data sent/acked/failed[/bold cyan] | "
        "[bold cyan]rates (risk/market)[/bold cyan]"
    )
    CONSOLE.rule(style="dim")


async def stream_progress(
    *,
    stats: ThroughputStats,
    started_at: float,
    stop_event: asyncio.Event,
    interval_seconds: float,
) -> None:
    """Print periodic streaming progress until stop event is set."""
    previous_snapshot = stats.snapshot()
    previous_tick_at = perf_counter()
    while True:
        should_stop = False
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            should_stop = True
        except TimeoutError:
            should_stop = False

        now = perf_counter()
        current_snapshot = stats.snapshot()
        delta = compute_delta(current_snapshot, previous_snapshot)
        tick_elapsed = max(now - previous_tick_at, 1e-9)
        total_elapsed = max(now - started_at, 1e-9)
        rpc_group = (
            f"{format_count(current_snapshot.rpc_sent)}/"
            f"{format_count(current_snapshot.rpc_delivered)}/"
            f"{format_count(current_snapshot.rpc_failed)}"
        )
        publish_group = (
            f"{format_count(current_snapshot.publish_sent)}/"
            f"{format_count(current_snapshot.publish_acked_recipients)}/"
            f"{format_count(current_snapshot.publish_failed)}"
        )
        rates_group = (
            f"{format_rate(delta.rpc_sent / tick_elapsed)}"
            f" / {format_rate(delta.publish_sent / tick_elapsed)}"
        )

        CONSOLE.print(
            f"[bold]{total_elapsed:7.2f}s[/bold] | "
            f"[green]{rpc_group:>34}[/green] | "
            f"[magenta]{publish_group:>29}[/magenta] | "
            f"[yellow]{rates_group:>22}[/yellow]"
        )

        previous_snapshot = current_snapshot
        previous_tick_at = now
        if should_stop:
            break


def print_startup(config: DemoConfig) -> None:
    """Print startup banner and effective configuration."""
    config_table = Table.grid(padding=(0, 2))
    config_table.add_row(
        "[bold]duration[/bold]",
        f"{config.duration_seconds}s",
        "[bold]worker_count[/bold]",
        str(config.worker_count),
    )
    config_table.add_row(
        "[bold]rpc_concurrency[/bold]",
        str(config.rpc_concurrency),
        "[bold]publish_concurrency[/bold]",
        str(config.publish_concurrency),
    )
    config_table.add_row(
        "[bold]shard_count[/bold]",
        str(config.shard_count),
        "[bold]publish_pause[/bold]",
        f"{config.publish_pause_seconds}s",
    )
    config_table.add_row(
        "[bold]progress_interval[/bold]",
        f"{config.progress_interval_seconds}s",
        "",
        "",
    )
    CONSOLE.print("")
    CONSOLE.print(
        Panel(
            config_table,
            title="[bold cyan]AgentLane Market Workflow Throughput Demo[/bold cyan]",
            border_style="cyan",
        )
    )
    CONSOLE.print("[bold]Streaming progress[/bold]")
    print_progress_header()


def print_summary(
    *,
    config: DemoConfig,
    snapshot: StatsSnapshot,
    elapsed_seconds: float,
) -> None:
    """Print final run summary with aggregate throughput."""
    elapsed = max(elapsed_seconds, 1e-9)
    summary_table = Table(show_header=False, box=None, pad_edge=False)
    summary_table.add_row("elapsed_seconds", f"{elapsed_seconds:.2f}")
    summary_table.add_row(
        "risk rpc sent/delivered/failed",
        (
            f"{format_count(snapshot.rpc_sent)}/"
            f"{format_count(snapshot.rpc_delivered)}/"
            f"{format_count(snapshot.rpc_failed)}"
        ),
    )
    summary_table.add_row("risk rpc handled", f"{format_count(snapshot.rpc_handled)}")
    summary_table.add_row(
        "market data sent/acked/failed",
        (
            f"{format_count(snapshot.publish_sent)}/"
            f"{format_count(snapshot.publish_acked_recipients)}/"
            f"{format_count(snapshot.publish_failed)}"
        ),
    )
    summary_table.add_row(
        "market data handled",
        f"{format_count(snapshot.events_handled)}",
    )
    summary_table.add_row(
        "avg risk_rpc_sent rate",
        format_rate(snapshot.rpc_sent / elapsed),
    )
    summary_table.add_row(
        "avg risk_rpc_delivered rate",
        format_rate(snapshot.rpc_delivered / elapsed),
    )
    summary_table.add_row(
        "avg market_data_sent rate",
        format_rate(snapshot.publish_sent / elapsed),
    )
    summary_table.add_row("workers", str(config.worker_count))
    CONSOLE.print(
        Panel(
            summary_table,
            title="[bold green]Final Summary[/bold green]",
            border_style="green",
        )
    )


async def run_demo(config: DemoConfig) -> None:
    """Run high-throughput market workflow demo with live progress reporting."""
    stats = ThroughputStats()
    runtime = SingleThreadedRuntimeEngine(worker_count=config.worker_count)
    runtime.register_factory(
        RPC_AGENT_TYPE,
        lambda engine: RpcWorkerAgent(engine=engine, stats=stats),
    )
    runtime.register_factory(
        EVENT_AGENT_TYPE,
        lambda engine: EventWorkerAgent(engine=engine, stats=stats),
    )
    runtime.subscribe_exact(
        topic_type=EVENT_TOPIC_TYPE,
        agent_type=EVENT_AGENT_TYPE,
        delivery_mode=DeliveryMode.STATEFUL,
    )

    stop_event = asyncio.Event()
    started_at = 0.0
    stop_at = 0.0

    print_startup(config)
    async with single_threaded_runtime(runtime):
        started_at = perf_counter()
        stop_at = started_at + config.duration_seconds
        reporter_task = asyncio.create_task(
            stream_progress(
                stats=stats,
                started_at=started_at,
                stop_event=stop_event,
                interval_seconds=config.progress_interval_seconds,
            )
        )
        producer_tasks: list[asyncio.Task[None]] = []
        for producer_id in range(config.rpc_concurrency):
            producer_tasks.append(
                asyncio.create_task(
                    run_rpc_producer(
                        producer_id=producer_id,
                        runtime=runtime,
                        stats=stats,
                        stop_at=stop_at,
                        shard_count=config.shard_count,
                    )
                )
            )
        for producer_id in range(config.publish_concurrency):
            producer_tasks.append(
                asyncio.create_task(
                    run_publish_producer(
                        producer_id=producer_id,
                        runtime=runtime,
                        stats=stats,
                        stop_at=stop_at,
                        shard_count=config.shard_count,
                        publish_pause_seconds=config.publish_pause_seconds,
                    )
                )
            )

        await asyncio.gather(*producer_tasks)
        stop_event.set()
        await reporter_task

    elapsed_seconds = perf_counter() - started_at
    print_summary(
        config=config,
        snapshot=stats.snapshot(),
        elapsed_seconds=elapsed_seconds,
    )


def parse_args() -> DemoConfig:
    """Parse command line arguments into immutable demo config."""
    parser = argparse.ArgumentParser(
        description="AgentLane high-throughput market workflow demonstration.",
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--rpc-concurrency", type=int, default=8)
    parser.add_argument("--publish-concurrency", type=int, default=8)
    parser.add_argument("--worker-count", type=int, default=16)
    parser.add_argument("--shard-count", type=int, default=64)
    parser.add_argument("--progress-interval-seconds", type=float, default=1.0)
    parser.add_argument("--publish-pause-seconds", type=float, default=0.002)
    args = parser.parse_args()
    return DemoConfig(
        duration_seconds=args.duration_seconds,
        rpc_concurrency=args.rpc_concurrency,
        publish_concurrency=args.publish_concurrency,
        worker_count=args.worker_count,
        shard_count=args.shard_count,
        progress_interval_seconds=args.progress_interval_seconds,
        publish_pause_seconds=args.publish_pause_seconds,
    )


def main() -> None:
    """CLI entrypoint for high-throughput market workflow demo."""
    config = parse_args()
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
