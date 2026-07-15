"""
Evaluation Module - Benchmark Suite

Provides a comprehensive benchmarking framework including benchmark suites,
ELO rating systems, forgetting measurement, and various performance benchmarks.
"""

import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark Result
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
            "duration": self.duration,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create result from dictionary."""
        return cls(
            name=data.get("name", ""),
            score=data.get("score", 0.0),
            details=data.get("details", {}),
            duration=data.get("duration", 0.0),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Benchmark Base
# ---------------------------------------------------------------------------
class Benchmark(ABC):
    """Abstract base class for all benchmarks."""

    def __init__(
        self,
        name: str,
        description: str = "",
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.metrics = metrics or []

    @abstractmethod
    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """
        Run the benchmark on an agent/model.

        Args:
            agent: The agent or model to evaluate.
            **kwargs: Additional arguments.

        Returns:
            BenchmarkResult with scores and details.
        """
        ...

    @abstractmethod
    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """
        Evaluate predictions against targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth values.

        Returns:
            Dictionary of metric names to scores.
        """
        ...


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """Manages and runs a collection of benchmarks."""

    def __init__(self, name: str = "default_suite") -> None:
        self.name = name
        self._benchmarks: Dict[str, Benchmark] = {}
        self._results: List[BenchmarkResult] = []

    def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark with the suite."""
        self._benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name}")

    def unregister_benchmark(self, name: str) -> bool:
        """Remove a benchmark by name. Returns True if found."""
        if name in self._benchmarks:
            del self._benchmarks[name]
            return True
        return False

    def list_benchmarks(self) -> List[str]:
        """List all registered benchmark names."""
        return list(self._benchmarks.keys())

    def run_all(self, agent: Any, **kwargs) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        self._results = []
        for name, benchmark in self._benchmarks.items():
            try:
                result = self.run_single(name, agent, **kwargs)
                self._results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                error_result = BenchmarkResult(
                    name=name, score=0.0, error=str(e)
                )
                self._results.append(error_result)
        return self._results

    def run_single(self, name: str, agent: Any, **kwargs) -> BenchmarkResult:
        """Run a single benchmark by name."""
        if name not in self._benchmarks:
            raise ValueError(f"Benchmark '{name}' not registered")
        benchmark = self._benchmarks[name]
        start_time = time.time()
        result = benchmark.run(agent, **kwargs)
        result.duration = time.time() - start_time
        return result

    def get_results(self) -> List[BenchmarkResult]:
        """Get all results from the last run."""
        return list(self._results)

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result for a specific benchmark."""
        for r in self._results:
            if r.name == name:
                return r
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all results."""
        if not self._results:
            return {"suite": self.name, "benchmarks": 0}

        scores = [r.score for r in self._results if r.error is None]
        summary = {
            "suite": self.name,
            "benchmarks_run": len(self._results),
            "benchmarks_passed": len(scores),
            "benchmarks_failed": len(self._results) - len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "best_score": max(scores) if scores else 0.0,
            "worst_score": min(scores) if scores else 0.0,
            "total_duration": sum(r.duration for r in self._results),
        }
        return summary

    def export_results(self, filepath: str, format: str = "json") -> str:
        """
        Export results to a file.

        Args:
            filepath: Output file path.
            format: Export format ('json').

        Returns:
            Path to the exported file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        export_data = {
            "suite": self.name,
            "summary": self.get_summary(),
            "results": [r.to_dict() for r in self._results],
        }

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results exported to {filepath}")
        return filepath

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results = []


# ---------------------------------------------------------------------------
# ELO Benchmark
# ---------------------------------------------------------------------------
class ELOBenchmark:
    """ELO rating system for comparing agents through pairwise matches."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
    ) -> None:
        """
        Args:
            initial_rating: Default starting rating for new agents.
            k_factor: K-factor controlling rating change magnitude.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}
        self._match_history: List[Dict[str, Any]] = []

    def register_agent(self, agent_id: str, initial_rating: Optional[float] = None) -> None:
        """Register a new agent with an optional initial rating."""
        if agent_id not in self._ratings:
            self._ratings[agent_id] = initial_rating or self.initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for agent A against agent B.

        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Args:
            rating_a: Rating of agent A.
            rating_b: Rating of agent B.

        Returns:
            Expected score for agent A (0 to 1).
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + 10.0 ** exponent)

    def update_rating(
        self,
        winner: str,
        loser: str,
        k_factor: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.

        Args:
            winner: ID of the winning agent.
            loser: ID of the losing agent.
            k_factor: Override K-factor for this match.

        Returns:
            Tuple of (new_winner_rating, new_loser_rating).
        """
        self.register_agent(winner)
        self.register_agent(loser)

        k = k_factor or self.k_factor
        rating_w = self._ratings[winner]
        rating_l = self._ratings[loser]

        expected_w = self.expected_score(rating_w, rating_l)
        expected_l = self.expected_score(rating_l, rating_w)

        # Winner gets score 1.0, loser gets score 0.0
        new_rating_w = rating_w + k * (1.0 - expected_w)
        new_rating_l = rating_l + k * (0.0 - expected_l)

        self._ratings[winner] = new_rating_w
        self._ratings[loser] = new_rating_l

        self._match_history.append({
            "winner": winner,
            "loser": loser,
            "winner_rating_before": rating_w,
            "loser_rating_before": rating_l,
            "winner_rating_after": new_rating_w,
            "loser_rating_after": new_rating_l,
            "k_factor": k,
        })

        return new_rating_w, new_rating_l

    def update_rating_draw(
        self,
        agent_a: str,
        agent_b: str,
        k_factor: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Update ratings after a draw (both get score 0.5)."""
        self.register_agent(agent_a)
        self.register_agent(agent_b)

        k = k_factor or self.k_factor
        rating_a = self._ratings[agent_a]
        rating_b = self._ratings[agent_b]

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        new_rating_a = rating_a + k * (0.5 - expected_a)
        new_rating_b = rating_b + k * (0.5 - expected_b)

        self._ratings[agent_a] = new_rating_a
        self._ratings[agent_b] = new_rating_b

        self._match_history.append({
            "winner": agent_a,
            "loser": agent_b,
            "draw": True,
            "agent_a_rating_before": rating_a,
            "agent_b_rating_before": rating_b,
            "agent_a_rating_after": new_rating_a,
            "agent_b_rating_after": new_rating_b,
        })

        return new_rating_a, new_rating_b

    def get_rating(self, agent_id: str) -> float:
        """Get the current rating of an agent."""
        return self._ratings.get(agent_id, self.initial_rating)

    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get all agents ranked by rating (highest first)."""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def get_match_history(self) -> List[Dict[str, Any]]:
        """Get the full match history."""
        return list(self._match_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get ELO system statistics."""
        if not self._ratings:
            return {"agents": 0, "matches": 0}

        ratings = list(self._ratings.values())
        return {
            "agents": len(self._ratings),
            "matches": len(self._match_history),
            "average_rating": sum(ratings) / len(ratings),
            "max_rating": max(ratings),
            "min_rating": min(ratings),
            "rating_std": self._std(ratings),
        }

    @staticmethod
    def _std(values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Forgetting Benchmark
# ---------------------------------------------------------------------------
class ForgettingBenchmark(Benchmark):
    """Measures catastrophic forgetting in continual learning scenarios.

    Tracks how model performance on previously learned tasks degrades
    after learning new tasks.
    """

    def __init__(self) -> None:
        super().__init__(
            name="forgetting_benchmark",
            description="Measures catastrophic forgetting and transfer",
            metrics=["forgetting_rate", "forward_transfer", "backward_transfer"],
        )
        self._task_results: Dict[int, List[float]] = {}

    def record_task_performance(self, task_id: int, accuracy: float) -> None:
        """Record model accuracy on a task after training."""
        if task_id not in self._task_results:
            self._task_results[task_id] = []
        self._task_results[task_id].append(accuracy)

    def measure_forgetting(self) -> Dict[int, float]:
        """
        Measure forgetting rate for each task.

        Forgetting(i) = max_accuracy(i) - final_accuracy(i)

        Returns:
            Dictionary mapping task IDs to forgetting rates.
        """
        forgetting = {}
        for task_id, accuracies in self._task_results.items():
            if len(accuracies) >= 2:
                max_acc = max(accuracies)
                final_acc = accuracies[-1]
                forgetting[task_id] = max_acc - final_acc
            else:
                forgetting[task_id] = 0.0
        return forgetting

    def forward_transfer(self) -> Dict[int, float]:
        """
        Measure forward transfer for each task.

        Forward transfer(i) = accuracy_on_task_i_after_seeing_0_samples - random_baseline

        Since we track performance after each training step, the first entry
        for a task that was evaluated before being trained represents zero-shot.
        Returns 0.0 for tasks where no zero-shot evaluation was recorded.

        Returns:
            Dictionary mapping task IDs to forward transfer values.
        """
        transfer = {}
        random_baseline = 1.0 / max(len(self._task_results), 1)

        for task_id, accuracies in self._task_results.items():
            if len(accuracies) >= 1:
                # Use first recorded accuracy as zero-shot proxy
                transfer[task_id] = accuracies[0] - random_baseline
            else:
                transfer[task_id] = 0.0
        return transfer

    def backward_transfer(self) -> Dict[int, float]:
        """
        Measure backward transfer for each task.

        Backward transfer(i) = final_accuracy(i) - accuracy_after_first_training(i)

        Positive values indicate improvement, negative values indicate forgetting.

        Returns:
            Dictionary mapping task IDs to backward transfer values.
        """
        transfer = {}
        for task_id, accuracies in self._task_results.items():
            if len(accuracies) >= 2:
                transfer[task_id] = accuracies[-1] - accuracies[0]
            else:
                transfer[task_id] = 0.0
        return transfer

    def get_average_forgetting(self) -> float:
        """Get the average forgetting rate across all tasks."""
        forgetting = self.measure_forgetting()
        if not forgetting:
            return 0.0
        return sum(forgetting.values()) / len(forgetting)

    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """Run the forgetting benchmark."""
        forgetting = self.measure_forgetting()
        avg_forgetting = self.get_average_forgetting()
        fw_transfer = self.forward_transfer()
        bw_transfer = self.backward_transfer()

        return BenchmarkResult(
            name=self.name,
            score=-avg_forgetting,  # Higher is better (less forgetting)
            details={
                "forgetting_rates": forgetting,
                "average_forgetting": avg_forgetting,
                "forward_transfer": fw_transfer,
                "backward_transfer": bw_transfer,
            },
        )

    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """Evaluate predictions against targets for a single task."""
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        total = len(targets) if targets else 1
        accuracy = correct / total
        return {"accuracy": accuracy}

    def reset(self) -> None:
        """Reset all recorded task results."""
        self._task_results = {}


# ---------------------------------------------------------------------------
# Accuracy Benchmark
# ---------------------------------------------------------------------------
class AccuracyBenchmark(Benchmark):
    """Benchmark for measuring classification/regression accuracy."""

    def __init__(self, name: str = "accuracy_benchmark") -> None:
        super().__init__(
            name=name,
            description="Measures prediction accuracy",
            metrics=["accuracy", "top_k_accuracy"],
        )

    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """
        Run accuracy benchmark.

        Args:
            agent: Must have a predict() method or be a callable.
            kwargs: 'test_data' (list of (input, target) pairs),
                    'top_k' (int, default 1).
        """
        test_data = kwargs.get("test_data", [])
        top_k = kwargs.get("top_k", 1)

        if not test_data:
            return BenchmarkResult(name=self.name, score=0.0, error="No test data")

        correct = 0
        total = len(test_data)

        for input_data, target in test_data:
            if callable(agent):
                pred = agent(input_data)
            elif hasattr(agent, "predict"):
                pred = agent.predict(input_data)
            else:
                pred = agent

            if top_k == 1:
                if pred == target:
                    correct += 1
            else:
                if isinstance(pred, (list, tuple)):
                    if target in pred[:top_k]:
                        correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return BenchmarkResult(
            name=self.name,
            score=accuracy,
            details={
                "correct": correct,
                "total": total,
                "top_k": top_k,
            },
        )

    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """Evaluate accuracy of predictions."""
        if isinstance(predictions[0], (list, tuple)):
            # Top-k predictions
            correct = sum(
                1 for p, t in zip(predictions, targets) if t in p
            )
        else:
            correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        total = len(targets) if targets else 1
        return {"accuracy": correct / total}


# ---------------------------------------------------------------------------
# Latency Benchmark
# ---------------------------------------------------------------------------
class LatencyBenchmark(Benchmark):
    """Benchmark for measuring inference latency."""

    def __init__(self, name: str = "latency_benchmark") -> None:
        super().__init__(
            name=name,
            description="Measures inference latency",
            metrics=["mean_latency", "median_latency", "p95_latency", "p99_latency"],
        )

    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """
        Run latency benchmark.

        Args:
            agent: Callable or object with predict() method.
            kwargs: 'inputs' (list of inputs), 'warmup_runs' (int, default 3).
        """
        inputs = kwargs.get("inputs", [])
        warmup_runs = kwargs.get("warmup_runs", 3)

        if not inputs:
            return BenchmarkResult(name=self.name, score=0.0, error="No inputs")

        # Warmup
        for inp in inputs[:warmup_runs]:
            if callable(agent):
                agent(inp)
            elif hasattr(agent, "predict"):
                agent.predict(inp)

        # Measure
        latencies = []
        for inp in inputs:
            start = time.time()
            if callable(agent):
                agent(inp)
            elif hasattr(agent, "predict"):
                agent.predict(inp)
            latencies.append(time.time() - start)

        latencies.sort()
        n = len(latencies)

        mean_lat = sum(latencies) / n
        median_lat = latencies[n // 2]
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        p95_lat = latencies[min(p95_idx, n - 1)]
        p99_lat = latencies[min(p99_idx, n - 1)]

        return BenchmarkResult(
            name=self.name,
            score=mean_lat,
            details={
                "mean_latency_ms": mean_lat * 1000,
                "median_latency_ms": median_lat * 1000,
                "p95_latency_ms": p95_lat * 1000,
                "p99_latency_ms": p99_lat * 1000,
                "min_latency_ms": latencies[0] * 1000,
                "max_latency_ms": latencies[-1] * 1000,
                "num_runs": n,
            },
        )

    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """Not applicable for latency benchmark."""
        return {"latency": 0.0}


# ---------------------------------------------------------------------------
# Throughput Benchmark
# ---------------------------------------------------------------------------
class ThroughputBenchmark(Benchmark):
    """Benchmark for measuring processing throughput."""

    def __init__(self, name: str = "throughput_benchmark") -> None:
        super().__init__(
            name=name,
            description="Measures processing throughput",
            metrics=["items_per_second", "batch_throughput"],
        )

    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """
        Run throughput benchmark.

        Args:
            agent: Callable or object with predict() method.
            kwargs: 'inputs' (list of inputs), 'batch_sizes' (list of ints to test).
        """
        inputs = kwargs.get("inputs", [])
        batch_sizes = kwargs.get("batch_sizes", [1, 8, 32, 64])

        if not inputs:
            return BenchmarkResult(name=self.name, score=0.0, error="No inputs")

        results = {}
        for bs in batch_sizes:
            # Create batches
            batches = [inputs[i:i + bs] for i in range(0, len(inputs), bs)]
            if not batches:
                continue

            start = time.time()
            processed = 0
            for batch in batches:
                if callable(agent):
                    agent(batch)
                elif hasattr(agent, "predict"):
                    agent.predict(batch)
                processed += len(batch)
            elapsed = time.time() - start

            throughput = processed / elapsed if elapsed > 0 else 0.0
            results[f"batch_{bs}"] = {
                "items_per_second": throughput,
                "batch_size": bs,
                "total_items": processed,
                "elapsed_seconds": elapsed,
            }

        # Use the largest batch throughput as the primary score
        best_throughput = 0.0
        for v in results.values():
            if v["items_per_second"] > best_throughput:
                best_throughput = v["items_per_second"]

        return BenchmarkResult(
            name=self.name,
            score=best_throughput,
            details=results,
        )

    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """Not applicable for throughput benchmark."""
        return {"throughput": 0.0}


# ---------------------------------------------------------------------------
# Memory Benchmark
# ---------------------------------------------------------------------------
class MemoryBenchmark(Benchmark):
    """Benchmark for measuring memory usage."""

    def __init__(self, name: str = "memory_benchmark") -> None:
        super().__init__(
            name=name,
            description="Measures memory usage during inference",
            metrics=["peak_memory_mb", "avg_memory_mb"],
        )

    def run(self, agent: Any, **kwargs) -> BenchmarkResult:
        """
        Run memory benchmark.

        Args:
            agent: Callable or object with predict() method.
            kwargs: 'inputs' (list of inputs), 'get_memory_fn' (optional custom function).
        """
        inputs = kwargs.get("inputs", [])
        get_memory_fn = kwargs.get("get_memory_fn")

        if not inputs:
            return BenchmarkResult(name=self.name, score=0.0, error="No inputs")

        memory_readings = []

        for inp in inputs:
            if get_memory_fn:
                mem_before = get_memory_fn()
            else:
                # Use resource module if available, otherwise estimate
                try:
                    import resource
                    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                except ImportError:
                    # Fallback: estimate based on input size
                    mem_before = self._estimate_memory(inp)

            if callable(agent):
                agent(inp)
            elif hasattr(agent, "predict"):
                agent.predict(inp)

            if get_memory_fn:
                mem_after = get_memory_fn()
            else:
                try:
                    import resource
                    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                except ImportError:
                    mem_after = self._estimate_memory(inp) * 1.5

            memory_readings.append(max(0, mem_after - mem_before))

        if not memory_readings:
            return BenchmarkResult(name=self.name, score=0.0)

        peak_mem = max(memory_readings)
        avg_mem = sum(memory_readings) / len(memory_readings)

        # Convert to MB if from resource (returns KB on Linux)
        try:
            import resource
            scale = 1024.0  # KB to MB
        except ImportError:
            scale = 1.0

        return BenchmarkResult(
            name=self.name,
            score=peak_mem / scale,
            details={
                "peak_memory_mb": peak_mem / scale,
                "avg_memory_mb": avg_mem / scale,
                "min_memory_mb": min(memory_readings) / scale,
                "max_memory_mb": peak_mem / scale,
                "num_samples": len(memory_readings),
            },
        )

    def evaluate(self, predictions: Any, targets: Any) -> Dict[str, float]:
        """Not applicable for memory benchmark."""
        return {"memory_mb": 0.0}

    @staticmethod
    def _estimate_memory(data: Any) -> float:
        """Rough memory estimate based on data size."""
        if isinstance(data, (list, tuple)):
            return len(data) * 8.0  # Assume 8 bytes per element
        if isinstance(data, dict):
            return len(data) * 64.0  # Assume 64 bytes per entry
        return 64.0
