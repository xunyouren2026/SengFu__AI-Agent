"""
AGI Unified Framework - Evaluation Module

Provides benchmarking and evaluation infrastructure for measuring
model performance across various dimensions.
"""

from .benchmark import (
    BenchmarkSuite,
    Benchmark,
    BenchmarkResult,
    ELOBenchmark,
    ForgettingBenchmark,
    AccuracyBenchmark,
    LatencyBenchmark,
    ThroughputBenchmark,
    MemoryBenchmark,
)
from .metrics import (
    Metric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUC,
    BLEUScore,
    ROUGEScore,
    Perplexity,
    EditDistance,
    SemanticSimilarity,
    MetricsTracker,
)

__all__ = [
    # Benchmarks
    "BenchmarkSuite",
    "Benchmark",
    "BenchmarkResult",
    "ELOBenchmark",
    "ForgettingBenchmark",
    "AccuracyBenchmark",
    "LatencyBenchmark",
    "ThroughputBenchmark",
    "MemoryBenchmark",
    # Metrics
    "Metric",
    "Accuracy",
    "Precision",
    "Recall",
    "F1Score",
    "AUC",
    "BLEUScore",
    "ROUGEScore",
    "Perplexity",
    "EditDistance",
    "SemanticSimilarity",
    "MetricsTracker",
]
