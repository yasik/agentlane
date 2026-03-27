"""Core metric types for the tracing metrics system."""

from dataclasses import dataclass, field
from typing import Any, Literal

MetricValue = int | float
"""Supported metric value types."""

AggregationType = Literal["sum", "count", "avg", "min", "max", "first", "last"]
"""Available aggregation strategy names."""


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """A single metric emission from a span.

    Attributes:
        name: The metric name (e.g., "unsupported_components_count").
        value: The numeric value.
        span_id: The span that emitted this metric.
        trace_id: The trace containing the span.
        timestamp: ISO timestamp when metric was emitted.
    """

    name: str
    value: MetricValue
    span_id: str
    trace_id: str
    timestamp: str


def _default_raw_values() -> list[MetricValue]:
    """Create an empty raw-value list with concrete metric typing."""
    return []


@dataclass(slots=True)
class AggregatedMetric:
    """Result of aggregating metrics at trace level.

    Attributes:
        name: The metric name.
        aggregation: The aggregation type applied.
        value: The aggregated value.
        count: Number of values aggregated.
        raw_values: Optional list of original values (for debugging).
    """

    name: str
    aggregation: AggregationType
    value: MetricValue
    count: int
    raw_values: list[MetricValue] = field(default_factory=_default_raw_values)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for processor output."""
        return {
            "name": self.name,
            "aggregation": self.aggregation,
            "value": self.value,
            "count": self.count,
        }
