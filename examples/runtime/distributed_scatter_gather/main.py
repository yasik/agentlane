"""Distributed direct scatter / gather demo for trade analysis."""

import argparse
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentlane.messaging import AgentId, DeliveryOutcome, DeliveryStatus, MessageContext
from agentlane.runtime import (
    BaseAgent,
    Engine,
    WorkerAgentRuntime,
    WorkerAgentRuntimeHost,
    on_message,
)

COORDINATOR_AGENT_TYPE = "demo.scatter.coordinator"
POSITION_AGENT_TYPE = "demo.scatter.position"
VALUATION_AGENT_TYPE = "demo.scatter.valuation"
RISK_AGENT_TYPE = "demo.scatter.risk"

CONSOLE = Console()
COMPONENT_STYLES = {
    "host": "bold cyan",
    "coordinator": "bold magenta",
    "position": "bold blue",
    "valuation": "bold yellow",
    "risk": "bold green",
}


def log_line(component: str, message: str) -> None:
    """Print one timestamped demo log line."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    component_style = COMPONENT_STYLES.get(component, "white")
    CONSOLE.print(
        f"[dim][{timestamp}][/dim] "
        f"[{component_style}]{component:<18}[/{component_style}] "
        f"[dim]|[/dim] {message}"
    )


@dataclass(slots=True, frozen=True)
class DemoConfig:
    """Configuration for the distributed scatter / gather demo."""

    request_count: int
    """Number of trade analysis requests to run in parallel."""


@dataclass(slots=True)
class TradeAnalysis:
    """Merged trade analysis returned to the demo runner."""

    request_id: str
    """Unique request identifier."""

    symbol: str
    """Analyzed instrument symbol."""

    shares: int
    """Trade size in shares."""

    position_status: str
    """Portfolio position summary."""

    notional_usd: float
    """Estimated notional value for the trade."""

    risk_score: int
    """Synthetic risk score for the proposed trade."""

    risk_band: str
    """Selected risk review band."""


def expect_mapping(payload: object, context: str) -> dict[str, object]:
    """Return one JSON-like mapping payload."""
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping payload for {context}.")
    return cast(dict[str, object], payload)


def expect_str(payload: Mapping[str, object], field_name: str) -> str:
    """Return one required string field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"Expected string field '{field_name}'.")
    return value


def expect_int(payload: Mapping[str, object], field_name: str) -> int:
    """Return one required integer field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Expected integer field '{field_name}'.")
    return value


def expect_float(payload: Mapping[str, object], field_name: str) -> float:
    """Return one required numeric field from a JSON-like payload."""
    value = payload.get(field_name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected numeric field '{field_name}'.")
    return float(value)


def expect_delivered(
    *,
    label: str,
    outcome: DeliveryOutcome,
) -> dict[str, object]:
    """Return response payload mapping for one delivered outcome."""
    if outcome.status != DeliveryStatus.DELIVERED:
        raise RuntimeError(f"{label} RPC failed with status={outcome.status.value}")
    if outcome.response_payload is None:
        raise RuntimeError(f"{label} RPC returned no response payload.")
    return expect_mapping(outcome.response_payload, label)


def build_trade_analysis(payload: dict[str, object]) -> TradeAnalysis:
    """Construct one typed trade analysis view from a JSON-like payload."""
    return TradeAnalysis(
        request_id=expect_str(payload, "request_id"),
        symbol=expect_str(payload, "symbol"),
        shares=expect_int(payload, "shares"),
        position_status=expect_str(payload, "position_status"),
        notional_usd=expect_float(payload, "notional_usd"),
        risk_score=expect_int(payload, "risk_score"),
        risk_band=expect_str(payload, "risk_band"),
    )


class CoordinatorAgent(BaseAgent):
    """Coordinator that scatters direct RPCs and gathers responses."""

    def __init__(
        self,
        engine: Engine,
        position_recipient: AgentId,
        valuation_recipient: AgentId,
        risk_recipient: AgentId,
    ) -> None:
        """Initialize coordinator with specialist recipients."""
        super().__init__(engine)
        self._position_recipient = position_recipient
        self._valuation_recipient = valuation_recipient
        self._risk_recipient = risk_recipient

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Scatter direct RPCs to specialists and gather one trade analysis."""
        request_id = expect_str(payload, "request_id")
        symbol = expect_str(payload, "symbol")
        shares = expect_int(payload, "shares")
        request_payload = {
            "request_id": request_id,
            "symbol": symbol,
            "shares": shares,
        }
        log_line(
            "coordinator",
            f"request={request_id} scattering to position, valuation, and risk",
        )
        # create_task starts all three RPCs immediately so the specialist
        # workers can process them in parallel instead of serially.
        position_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._position_recipient,
                correlation_id=context.correlation_id,
            )
        )
        valuation_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._valuation_recipient,
                correlation_id=context.correlation_id,
            )
        )
        risk_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._risk_recipient,
                correlation_id=context.correlation_id,
            )
        )

        position_outcome, valuation_outcome, risk_outcome = await asyncio.gather(
            position_task,
            valuation_task,
            risk_task,
        )

        position = expect_delivered(label="position", outcome=position_outcome)
        valuation = expect_delivered(label="valuation", outcome=valuation_outcome)
        risk = expect_delivered(label="risk", outcome=risk_outcome)

        log_line(
            "coordinator",
            f"request={request_id} gathered all specialist responses",
        )
        return {
            "request_id": request_id,
            "symbol": symbol,
            "shares": shares,
            "position_status": expect_str(position, "status"),
            "notional_usd": expect_float(valuation, "notional_usd"),
            "risk_score": expect_int(risk, "risk_score"),
            "risk_band": expect_str(risk, "risk_band"),
        }


class PositionAgent(BaseAgent):
    """Portfolio position specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one portfolio position assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        shares = expect_int(payload, "shares")
        await asyncio.sleep(0.08)
        log_line("position", f"request={request_id} checked portfolio capacity")
        return {
            "request_id": request_id,
            "reserved_shares": shares,
            "status": f"capacity available for {shares} shares",
        }


class ValuationAgent(BaseAgent):
    """Market valuation specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one valuation assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        shares = expect_int(payload, "shares")
        await asyncio.sleep(0.12)
        mark_price_usd = 187.42
        notional_usd = round(mark_price_usd * shares, 2)
        log_line("valuation", f"request={request_id} priced notional")
        return {
            "request_id": request_id,
            "mark_price_usd": mark_price_usd,
            "notional_usd": notional_usd,
        }


class RiskAgent(BaseAgent):
    """Trade risk specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one risk assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        shares = expect_int(payload, "shares")
        await asyncio.sleep(0.05)
        risk_score = 18 if shares <= 5 else 42
        log_line("risk", f"request={request_id} scored trade risk")
        return {
            "request_id": request_id,
            "risk_score": risk_score,
            "risk_band": "standard-review" if risk_score <= 25 else "senior-review",
        }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the demo."""
    parser = argparse.ArgumentParser(
        description="Distributed direct scatter / gather demo.",
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=2,
        help="Number of trade analysis requests to run in parallel.",
    )
    return parser


def print_cluster_layout(
    *,
    host: WorkerAgentRuntimeHost,
    coordinator_worker: WorkerAgentRuntime,
    position_worker: WorkerAgentRuntime,
    valuation_worker: WorkerAgentRuntime,
    risk_worker: WorkerAgentRuntime,
) -> None:
    """Print resolved host and worker addresses."""
    table = Table(title="Distributed Scatter / Gather Layout", show_lines=False)
    table.add_column("Component", style="bold cyan")
    table.add_column("Address", style="white")
    table.add_row("host", host.address)
    table.add_row("coordinator_worker", coordinator_worker.address)
    table.add_row("position_worker", position_worker.address)
    table.add_row("valuation_worker", valuation_worker.address)
    table.add_row("risk_worker", risk_worker.address)
    CONSOLE.print(table)


def print_trade_analyses(analyses: list[TradeAnalysis]) -> None:
    """Print aggregated trade analysis results."""
    table = Table(title="Aggregated Trade Analyses", show_lines=False)
    table.add_column("Request", style="bold green")
    table.add_column("Symbol", style="white")
    table.add_column("Shares", justify="right")
    table.add_column("Position", style="white")
    table.add_column("Notional USD", justify="right")
    table.add_column("Risk Score", justify="right")
    table.add_column("Risk Band", style="white")
    for analysis in analyses:
        table.add_row(
            analysis.request_id,
            analysis.symbol,
            str(analysis.shares),
            analysis.position_status,
            f"{analysis.notional_usd:.2f}",
            str(analysis.risk_score),
            analysis.risk_band,
        )
    CONSOLE.print(table)


async def run_demo(config: DemoConfig) -> None:
    """Run the distributed scatter / gather demo."""
    # The host is the single control plane for the cluster. Each worker below
    # connects back to this address and uses the host for cross-worker routing.
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    await host.start()
    log_line("host", f"started distributed host at {host.address}")

    coordinator_worker = WorkerAgentRuntime(host_address=host.address)
    position_worker = WorkerAgentRuntime(host_address=host.address)
    valuation_worker = WorkerAgentRuntime(host_address=host.address)
    risk_worker = WorkerAgentRuntime(host_address=host.address)

    # The coordinator keeps only logical recipient ids. It does not need to know
    # which worker currently owns each specialist.
    coordinator_worker.register_factory(
        COORDINATOR_AGENT_TYPE,
        lambda engine: CoordinatorAgent(
            engine,
            position_recipient=AgentId.from_values(POSITION_AGENT_TYPE, "position"),
            valuation_recipient=AgentId.from_values(
                VALUATION_AGENT_TYPE,
                "valuation",
            ),
            risk_recipient=AgentId.from_values(RISK_AGENT_TYPE, "risk"),
        ),
    )
    position_worker.register_factory(POSITION_AGENT_TYPE, PositionAgent)
    valuation_worker.register_factory(VALUATION_AGENT_TYPE, ValuationAgent)
    risk_worker.register_factory(RISK_AGENT_TYPE, RiskAgent)

    workers = [
        coordinator_worker,
        position_worker,
        valuation_worker,
        risk_worker,
    ]
    # Worker startup registers the available agent types with the host so direct
    # RPC routing can begin.
    await asyncio.gather(*(worker.start() for worker in workers))
    print_cluster_layout(
        host=host,
        coordinator_worker=coordinator_worker,
        position_worker=position_worker,
        valuation_worker=valuation_worker,
        risk_worker=risk_worker,
    )

    try:
        requests = [
            {
                "request_id": f"trade-{index + 1}",
                "symbol": "AAPL",
                "shares": index + 2,
            }
            for index in range(config.request_count)
        ]
        # Each trade analysis targets a coordinator agent keyed by request id.
        # That keeps the logical recipient stable if the coordinator ever needs
        # per-request state later.
        outcomes = await asyncio.gather(
            *(
                coordinator_worker.send_message(
                    request,
                    recipient=AgentId.from_values(
                        COORDINATOR_AGENT_TYPE,
                        expect_str(request, "request_id"),
                    ),
                )
                for request in requests
            )
        )
        analyses = [
            build_trade_analysis(
                expect_delivered(
                    label=expect_str(request, "request_id"),
                    outcome=outcome,
                )
            )
            for request, outcome in zip(requests, outcomes, strict=True)
        ]
        print_trade_analyses(analyses)
    finally:
        await asyncio.gather(*(worker.stop_when_idle() for worker in workers))
        await host.stop_when_idle()


def main() -> None:
    """Parse CLI args and run the demo."""
    parser = build_parser()
    args = parser.parse_args()
    config = DemoConfig(request_count=args.request_count)
    CONSOLE.print(
        Panel.fit(
            (
                "Distributed trade-analysis scatter / gather demo\n"
                "- coordinator sends direct RPCs to specialist workers\n"
                "- specialist workers reply independently\n"
                "- coordinator aggregates responses into one trade analysis"
            ),
            border_style="magenta",
        )
    )
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
