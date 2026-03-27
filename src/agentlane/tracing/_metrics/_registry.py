# pylint: disable=W0603

"""Registry for mapping metric names to aggregation strategies."""

import threading
from typing import Any

from ._aggregators import MetricAggregator, get_aggregator
from ._types import AggregationType


class MetricsRegistry:
    """Registry mapping metric names to their aggregation strategies.

    The registry is thread-safe and supports:
    - Registering metrics with specific aggregation types
    - Registering metrics with custom aggregator instances
    - Default aggregation for unregistered metrics

    Example:
        >>> registry = MetricsRegistry(default_aggregation="sum")
        >>> registry.register("search_count", "count")
        >>> registry.register("relevance_score", "avg")
        >>> registry.register("custom_metric", custom_aggregator)
    """

    def __init__(
        self,
        default_aggregation: AggregationType = "sum",
    ) -> None:
        """Initialize the registry.

        Args:
            default_aggregation: Default aggregation for unregistered metrics.
        """
        self._default_aggregation: AggregationType = default_aggregation
        self._metrics: dict[str, MetricAggregator] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        aggregation: AggregationType | MetricAggregator,
    ) -> None:
        """Register a metric with its aggregation strategy.

        Args:
            name: The metric name.
            aggregation: Either an AggregationType string or a custom aggregator.
        """
        with self._lock:
            if isinstance(aggregation, str):
                self._metrics[name] = get_aggregator(aggregation)
            else:
                self._metrics[name] = aggregation

    def register_many(
        self,
        metrics: dict[str, AggregationType | MetricAggregator],
    ) -> None:
        """Register multiple metrics at once.

        Args:
            metrics: Dictionary mapping metric names to aggregation strategies.
        """
        with self._lock:
            for name, aggregation in metrics.items():
                if isinstance(aggregation, str):
                    self._metrics[name] = get_aggregator(aggregation)
                else:
                    self._metrics[name] = aggregation

    def get_aggregator(self, name: str) -> MetricAggregator:
        """Get the aggregator for a metric name.

        Args:
            name: The metric name.

        Returns:
            The registered aggregator or default aggregator.
        """
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]
            return get_aggregator(self._default_aggregation)

    def is_registered(self, name: str) -> bool:
        """Check if a metric is explicitly registered.

        Args:
            name: The metric name.

        Returns:
            True if explicitly registered.
        """
        with self._lock:
            return name in self._metrics

    def export(self) -> dict[str, Any]:
        """Export registry configuration for debugging.

        Returns:
            Dictionary of metric configurations.
        """
        with self._lock:
            return {
                "default_aggregation": self._default_aggregation,
                "registered_metrics": {
                    name: agg.aggregation_type for name, agg in self._metrics.items()
                },
            }


_default_registry: MetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry, creating if needed.

    Returns:
        The global MetricsRegistry instance.
    """
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = MetricsRegistry()
        return _default_registry


def set_metrics_registry(registry: MetricsRegistry) -> None:
    """Set the global metrics registry.

    Args:
        registry: The registry to use globally.
    """
    global _default_registry
    with _registry_lock:
        _default_registry = registry


def reset_metrics_registry() -> None:
    """Reset the global metrics registry to None.

    Useful for testing to ensure clean state between tests.
    """
    global _default_registry
    with _registry_lock:
        _default_registry = None
