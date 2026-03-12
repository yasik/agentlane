"""Distributed direct scatter / gather demo with an aggregating coordinator."""

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
INVENTORY_AGENT_TYPE = "demo.scatter.inventory"
PRICING_AGENT_TYPE = "demo.scatter.pricing"
SHIPPING_AGENT_TYPE = "demo.scatter.shipping"

CONSOLE = Console()
COMPONENT_STYLES = {
    "host": "bold cyan",
    "coordinator": "bold magenta",
    "inventory": "bold blue",
    "pricing": "bold yellow",
    "shipping": "bold green",
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
    """Number of quote requests to run in parallel."""


@dataclass(slots=True)
class AggregatedQuote:
    """Merged quote returned to the demo runner."""

    request_id: str
    """Unique request identifier."""

    product_name: str
    """Quoted product."""

    quantity: int
    """Quoted quantity."""

    inventory_status: str
    """Inventory summary."""

    subtotal_usd: float
    """Total price for the request."""

    shipping_eta_days: int
    """Estimated shipping time."""

    service_level: str
    """Selected shipping service level."""


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


def build_aggregated_quote(payload: dict[str, object]) -> AggregatedQuote:
    """Construct one typed quote view from a JSON-like payload."""
    return AggregatedQuote(
        request_id=expect_str(payload, "request_id"),
        product_name=expect_str(payload, "product_name"),
        quantity=expect_int(payload, "quantity"),
        inventory_status=expect_str(payload, "inventory_status"),
        subtotal_usd=expect_float(payload, "subtotal_usd"),
        shipping_eta_days=expect_int(payload, "shipping_eta_days"),
        service_level=expect_str(payload, "service_level"),
    )


class CoordinatorAgent(BaseAgent):
    """Coordinator that scatters direct RPCs and gathers responses."""

    def __init__(
        self,
        engine: Engine,
        inventory_recipient: AgentId,
        pricing_recipient: AgentId,
        shipping_recipient: AgentId,
    ) -> None:
        """Initialize coordinator with specialist recipients."""
        super().__init__(engine)
        self._inventory_recipient = inventory_recipient
        self._pricing_recipient = pricing_recipient
        self._shipping_recipient = shipping_recipient

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Scatter direct RPCs to specialists and gather one quote."""
        request_id = expect_str(payload, "request_id")
        product_name = expect_str(payload, "product_name")
        quantity = expect_int(payload, "quantity")
        request_payload = {
            "request_id": request_id,
            "product_name": product_name,
            "quantity": quantity,
        }
        log_line(
            "coordinator",
            f"request={request_id} scattering to inventory, pricing, and shipping",
        )
        inventory_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._inventory_recipient,
                correlation_id=context.correlation_id,
            )
        )
        pricing_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._pricing_recipient,
                correlation_id=context.correlation_id,
            )
        )
        shipping_task = asyncio.create_task(
            self.send_message(
                request_payload,
                recipient=self._shipping_recipient,
                correlation_id=context.correlation_id,
            )
        )

        inventory_outcome, pricing_outcome, shipping_outcome = await asyncio.gather(
            inventory_task,
            pricing_task,
            shipping_task,
        )

        inventory = expect_delivered(label="inventory", outcome=inventory_outcome)
        pricing = expect_delivered(label="pricing", outcome=pricing_outcome)
        shipping = expect_delivered(label="shipping", outcome=shipping_outcome)

        log_line(
            "coordinator",
            f"request={request_id} gathered all specialist responses",
        )
        return {
            "request_id": request_id,
            "product_name": product_name,
            "quantity": quantity,
            "inventory_status": expect_str(inventory, "status"),
            "subtotal_usd": expect_float(pricing, "subtotal_usd"),
            "shipping_eta_days": expect_int(shipping, "eta_days"),
            "service_level": expect_str(shipping, "service_level"),
        }


class InventoryAgent(BaseAgent):
    """Inventory specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one inventory assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        quantity = expect_int(payload, "quantity")
        await asyncio.sleep(0.08)
        log_line("inventory", f"request={request_id} reserved inventory")
        return {
            "request_id": request_id,
            "reserved_units": quantity,
            "status": f"reserved {quantity} units",
        }


class PricingAgent(BaseAgent):
    """Pricing specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one pricing assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        quantity = expect_int(payload, "quantity")
        await asyncio.sleep(0.12)
        unit_price_usd = 39.5
        subtotal_usd = round(unit_price_usd * quantity, 2)
        log_line("pricing", f"request={request_id} priced subtotal")
        return {
            "request_id": request_id,
            "unit_price_usd": unit_price_usd,
            "subtotal_usd": subtotal_usd,
        }


class ShippingAgent(BaseAgent):
    """Shipping specialist running on its own worker."""

    @on_message
    async def handle(
        self,
        payload: dict[str, object],
        context: MessageContext,
    ) -> object:
        """Return one shipping assessment."""
        _ = context
        request_id = expect_str(payload, "request_id")
        quantity = expect_int(payload, "quantity")
        await asyncio.sleep(0.05)
        eta_days = 2 if quantity <= 5 else 4
        log_line("shipping", f"request={request_id} estimated shipping")
        return {
            "request_id": request_id,
            "eta_days": eta_days,
            "service_level": "priority" if eta_days == 2 else "standard",
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
        help="Number of quote requests to run in parallel.",
    )
    return parser


def print_cluster_layout(
    *,
    host: WorkerAgentRuntimeHost,
    coordinator_worker: WorkerAgentRuntime,
    inventory_worker: WorkerAgentRuntime,
    pricing_worker: WorkerAgentRuntime,
    shipping_worker: WorkerAgentRuntime,
) -> None:
    """Print resolved host and worker addresses."""
    table = Table(title="Distributed Scatter / Gather Layout", show_lines=False)
    table.add_column("Component", style="bold cyan")
    table.add_column("Address", style="white")
    table.add_row("host", host.address)
    table.add_row("coordinator_worker", coordinator_worker.address)
    table.add_row("inventory_worker", inventory_worker.address)
    table.add_row("pricing_worker", pricing_worker.address)
    table.add_row("shipping_worker", shipping_worker.address)
    CONSOLE.print(table)


def print_quotes(quotes: list[AggregatedQuote]) -> None:
    """Print aggregated quote results."""
    table = Table(title="Aggregated Quotes", show_lines=False)
    table.add_column("Request", style="bold green")
    table.add_column("Product", style="white")
    table.add_column("Qty", justify="right")
    table.add_column("Inventory", style="white")
    table.add_column("Subtotal USD", justify="right")
    table.add_column("ETA (days)", justify="right")
    table.add_column("Service", style="white")
    for quote in quotes:
        table.add_row(
            quote.request_id,
            quote.product_name,
            str(quote.quantity),
            quote.inventory_status,
            f"{quote.subtotal_usd:.2f}",
            str(quote.shipping_eta_days),
            quote.service_level,
        )
    CONSOLE.print(table)


async def run_demo(config: DemoConfig) -> None:
    """Run the distributed scatter / gather demo."""
    host = WorkerAgentRuntimeHost(address="127.0.0.1:0")
    await host.start()
    log_line("host", f"started distributed host at {host.address}")

    coordinator_worker = WorkerAgentRuntime(host_address=host.address)
    inventory_worker = WorkerAgentRuntime(host_address=host.address)
    pricing_worker = WorkerAgentRuntime(host_address=host.address)
    shipping_worker = WorkerAgentRuntime(host_address=host.address)

    coordinator_worker.register_factory(
        COORDINATOR_AGENT_TYPE,
        lambda engine: CoordinatorAgent(
            engine,
            inventory_recipient=AgentId.from_values(INVENTORY_AGENT_TYPE, "inventory"),
            pricing_recipient=AgentId.from_values(PRICING_AGENT_TYPE, "pricing"),
            shipping_recipient=AgentId.from_values(SHIPPING_AGENT_TYPE, "shipping"),
        ),
    )
    inventory_worker.register_factory(INVENTORY_AGENT_TYPE, InventoryAgent)
    pricing_worker.register_factory(PRICING_AGENT_TYPE, PricingAgent)
    shipping_worker.register_factory(SHIPPING_AGENT_TYPE, ShippingAgent)

    workers = [
        coordinator_worker,
        inventory_worker,
        pricing_worker,
        shipping_worker,
    ]
    await asyncio.gather(*(worker.start() for worker in workers))
    print_cluster_layout(
        host=host,
        coordinator_worker=coordinator_worker,
        inventory_worker=inventory_worker,
        pricing_worker=pricing_worker,
        shipping_worker=shipping_worker,
    )

    try:
        requests = [
            {
                "request_id": f"quote-{index + 1}",
                "product_name": "agentlane-widget",
                "quantity": index + 2,
            }
            for index in range(config.request_count)
        ]
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
        quotes = [
            build_aggregated_quote(
                expect_delivered(
                    label=expect_str(request, "request_id"),
                    outcome=outcome,
                )
            )
            for request, outcome in zip(requests, outcomes, strict=True)
        ]
        print_quotes(quotes)
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
                "Distributed direct scatter / gather demo\n"
                "- coordinator sends direct RPCs to specialist workers\n"
                "- specialist workers reply independently\n"
                "- coordinator aggregates the responses into one quote"
            ),
            border_style="magenta",
        )
    )
    asyncio.run(run_demo(config))


if __name__ == "__main__":
    main()
