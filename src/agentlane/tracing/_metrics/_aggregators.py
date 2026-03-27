# pylint: disable=W2301

"""Metric aggregation protocols and built-in implementations."""

import threading
from typing import Protocol, runtime_checkable

from ._types import AggregatedMetric, AggregationType, MetricValue


@runtime_checkable
class MetricAggregator(Protocol):
    """Protocol for metric aggregation strategies.

    Aggregators receive a list of values from span emissions and produce
    a single aggregated metric result.
    """

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        ...

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Aggregate a list of metric values into a single result.

        Args:
            name: The metric name.
            values: List of values to aggregate (guaranteed non-empty).

        Returns:
            The aggregated metric result.
        """
        ...


class SumAggregator:
    """Sum all metric values."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "sum"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Sum all values in the list."""
        return AggregatedMetric(
            name=name,
            aggregation="sum",
            value=sum(values),
            count=len(values),
            raw_values=list(values),
        )


class CountAggregator:
    """Count the number of emissions."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "count"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Count the number of values."""
        return AggregatedMetric(
            name=name,
            aggregation="count",
            value=len(values),
            count=len(values),
            raw_values=list(values),
        )


class AvgAggregator:
    """Calculate average of all values."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "avg"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Calculate the average of all values."""
        avg_value = sum(values) / len(values) if values else 0.0
        return AggregatedMetric(
            name=name,
            aggregation="avg",
            value=avg_value,
            count=len(values),
            raw_values=list(values),
        )


class MinAggregator:
    """Return minimum value."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "min"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Return the minimum value."""
        return AggregatedMetric(
            name=name,
            aggregation="min",
            value=min(values),
            count=len(values),
            raw_values=list(values),
        )


class MaxAggregator:
    """Return maximum value."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "max"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Return the maximum value."""
        return AggregatedMetric(
            name=name,
            aggregation="max",
            value=max(values),
            count=len(values),
            raw_values=list(values),
        )


class FirstAggregator:
    """Return first emitted value."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "first"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Return the first emitted value."""
        return AggregatedMetric(
            name=name,
            aggregation="first",
            value=values[0],
            count=len(values),
            raw_values=list(values),
        )


class LastAggregator:
    """Return last emitted value."""

    @property
    def aggregation_type(self) -> AggregationType:
        """Return the aggregation type identifier."""
        return "last"

    def aggregate(self, name: str, values: list[MetricValue]) -> AggregatedMetric:
        """Return the last emitted value."""
        return AggregatedMetric(
            name=name,
            aggregation="last",
            value=values[-1],
            count=len(values),
            raw_values=list(values),
        )


_aggregators: dict[AggregationType, MetricAggregator] = {
    "sum": SumAggregator(),
    "count": CountAggregator(),
    "avg": AvgAggregator(),
    "min": MinAggregator(),
    "max": MaxAggregator(),
    "first": FirstAggregator(),
    "last": LastAggregator(),
}
"""Pre-instantiated aggregators for common use."""

_aggregators_lock = threading.Lock()
"""Lock for the aggregators dictionary."""


def get_aggregator(aggregation_type: AggregationType) -> MetricAggregator:
    """Get a built-in aggregator by type.

    Args:
        aggregation_type: The aggregation type identifier.

    Returns:
        The aggregator instance.

    Raises:
        KeyError: If aggregation type is not found.
    """
    with _aggregators_lock:
        return _aggregators[aggregation_type]


def add_aggregator(
    aggregation_type: AggregationType, aggregator: MetricAggregator
) -> None:
    """Add a custom aggregator to the registry.

    Args:
        aggregation_type: The aggregation type identifier.
        aggregator: The aggregator instance.
    """
    with _aggregators_lock:
        _aggregators[aggregation_type] = aggregator


def get_aggregators() -> dict[AggregationType, MetricAggregator]:
    """Get the aggregators dictionary.

    Returns:
        The aggregators dictionary.
    """
    with _aggregators_lock:
        return _aggregators
