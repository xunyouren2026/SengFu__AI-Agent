"""
Metrics Module

This module provides metrics collection and export capabilities for the AGI Unified Framework,
including Prometheus-compatible metric types and aggregation strategies.
"""

from core.metrics.registry import (
    # Enums
    MetricType,
    AggregationType,
    # Core classes
    MetricFamily,
    Metric,
    GaugeMetric,
    CounterMetric,
    HistogramMetric,
    # Aggregation
    AggregationStrategy,
    SumAggregation,
    CountAggregation,
    MinAggregation,
    MaxAggregation,
    AverageAggregation,
    QuantileAggregation,
    create_aggregation,
    # Registry
    MetricsRegistry,
    PrometheusExporter,
    get_registry,
    # Helpers
    create_request_metrics,
    create_system_metrics,
    # Types
    Labels,
    Value,
    Observation,
)


__all__ = [
    # Enums
    "MetricType",
    "AggregationType",
    # Core classes
    "MetricFamily",
    "Metric",
    "GaugeMetric",
    "CounterMetric",
    "HistogramMetric",
    # Aggregation
    "AggregationStrategy",
    "SumAggregation",
    "CountAggregation",
    "MinAggregation",
    "MaxAggregation",
    "AverageAggregation",
    "QuantileAggregation",
    "create_aggregation",
    # Registry
    "MetricsRegistry",
    "PrometheusExporter",
    "get_registry",
    # Helpers
    "create_request_metrics",
    "create_system_metrics",
    # Types
    "Labels",
    "Value",
    "Observation",
]
