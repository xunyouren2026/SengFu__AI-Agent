"""Audit Anomaly Detection Module - Statistical baseline, z-score, moving average, isolation forest, behavioral profiling, alert generation."""

from __future__ import annotations
import math, random, time, uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    FREQUENCY = "frequency"
    TEMPORAL = "temporal"
    SEQUENCE = "sequence"

class AlertSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyAlert:
    alert_id: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    description: str
    detected_at: float = field(default_factory=time.time)
    source: str = ""
    metric_name: str = ""
    observed_value: float = 0.0
    expected_value: float = 0.0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {"alert_id": self.alert_id, "type": self.anomaly_type.value,
                "severity": self.severity.value, "description": self.description,
                "detected_at": self.detected_at, "score": round(self.score, 4)}

class StatisticalBaseline:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._stats: Dict[str, Dict[str, float]] = {}
    def add_observation(self, metric: str, value: float) -> None:
        self._metrics[metric].append(value)
        if len(self._metrics[metric]) > self.window_size * 2:
            self._metrics[metric] = self._metrics[metric][-self.window_size * 2:]
        self._update_stats(metric)
    def _update_stats(self, metric: str) -> None:
        values = self._metrics[metric][-self.window_size:]
        if not values:
            return
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0
        std = math.sqrt(variance)
        sorted_v = sorted(values)
        self._stats[metric] = {
            "mean": mean, "std": std, "min": sorted_v[0], "max": sorted_v[-1],
            "median": sorted_v[n // 2], "p25": sorted_v[n // 4],
            "p75": sorted_v[3 * n // 4], "count": n,
        }
    def get_stats(self, metric: str) -> Optional[Dict[str, float]]:
        return self._stats.get(metric)
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        return dict(self._stats)

class ZScoreAnalyzer:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
    def analyze(self, value: float, baseline: Dict[str, float]) -> Tuple[float, bool]:
        mean = baseline.get("mean", 0)
        std = baseline.get("std", 1)
        if std == 0:
            return 0.0, False
        z_score = abs(value - mean) / std
        return z_score, z_score > self.threshold

class MovingAverage:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._values: Dict[str, List[float]] = defaultdict(list)
    def add(self, metric: str, value: float) -> float:
        self._values[metric].append(value)
        if len(self._values[metric]) > self.window_size * 2:
            self._values[metric] = self._values[metric][-self.window_size * 2:]
        return self.compute(metric)
    def compute(self, metric: str) -> float:
        values = self._values[metric][-self.window_size:]
        return sum(values) / len(values) if values else 0.0
    def detect_anomaly(self, metric: str, value: float, threshold: float = 2.0) -> Tuple[float, bool]:
        ma = self.compute(metric)
        values = self._values[metric][-self.window_size:]
        if len(values) < 3:
            return 0.0, False
        variance = sum((x - ma) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0, False
        deviation = abs(value - ma) / std
        return deviation, deviation > threshold

class IsolationForest:
    def __init__(self, n_trees: int = 100, max_depth: int = 10, sample_size: int = 256):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self._trees: List[Dict[str, Any]] = []
        self._fitted = False
    def fit(self, data: List[List[float]]) -> None:
        self._trees = []
        n_features = len(data[0]) if data else 0
        for _ in range(self.n_trees):
            sample = random.sample(data, min(self.sample_size, len(data)))
            tree = self._build_tree(sample, 0, n_features)
            self._trees.append(tree)
        self._fitted = True
    def _build_tree(self, data: List[List[float]], depth: int, n_features: int) -> Dict[str, Any]:
        if depth >= self.max_depth or len(data) <= 1:
            return {"type": "leaf", "size": len(data)}
        feature = random.randint(0, n_features - 1)
        values = sorted(set(row[feature] for row in data))
        if len(values) <= 1:
            return {"type": "leaf", "size": len(data)}
        split = values[len(values) // 2]
        left = [row for row in data if row[feature] < split]
        right = [row for row in data if row[feature] >= split]
        if not left or not right:
            return {"type": "leaf", "size": len(data)}
        return {"type": "node", "feature": feature, "split": split,
                "left": self._build_tree(left, depth + 1, n_features),
                "right": self._build_tree(right, depth + 1, n_features)}
    def score(self, point: List[float]) -> float:
        if not self._fitted:
            return 0.0
        path_lengths = []
        for tree in self._trees:
            path_lengths.append(self._path_length(point, tree, 0))
        avg_path = sum(path_lengths) / len(path_lengths)
        n = self.sample_size
        c_n = 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
        if avg_path == 0:
            return 0.0
        return 2.0 ** (-avg_path / c_n)
    def _path_length(self, point: List[float], node: Dict[str, Any], depth: int) -> int:
        if node["type"] == "leaf":
            return depth + self._avg_path_length(node["size"])
        feature = node["feature"]
        if point[feature] < node["split"]:
            return self._path_length(point, node["left"], depth + 1)
        return self._path_length(point, node["right"], depth + 1)
    @staticmethod
    def _avg_path_length(size: int) -> float:
        if size <= 1:
            return 0
        if size == 2:
            return 1
        return 2.0 * (math.log(size - 1) + 0.5772156649) - (2.0 * (size - 1) / size)
    def is_anomaly(self, point: List[float], threshold: float = 0.6) -> Tuple[bool, float]:
        s = self.score(point)
        return s > threshold, s

class BehavioralProfiler:
    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "action_counts": Counter(), "resource_counts": Counter(),
            "timestamps": [], "action_sequence": [],
        })
    def record(self, actor: str, action: str, resource: str, timestamp: float) -> None:
        profile = self._profiles[actor]
        profile["action_counts"][action] += 1
        profile["resource_counts"][resource] += 1
        profile["timestamps"].append(timestamp)
        profile["action_sequence"].append(action)
        if len(profile["timestamps"]) > 10000:
            profile["timestamps"] = profile["timestamps"][-5000:]
            profile["action_sequence"] = profile["action_sequence"][-5000:]
    def get_profile(self, actor: str) -> Dict[str, Any]:
        profile = self._profiles[actor]
        timestamps = profile["timestamps"]
        if len(timestamps) >= 2:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 0
        return {
            "actor": actor, "total_actions": sum(profile["action_counts"].values()),
            "unique_actions": len(profile["action_counts"]),
            "top_actions": profile["action_counts"].most_common(10),
            "unique_resources": len(profile["resource_counts"]),
            "avg_interval": avg_interval,
        }
    def detect_deviation(self, actor: str, action: str, resource: str) -> Tuple[float, bool]:
        profile = self._profiles[actor]
        total = sum(profile["action_counts"].values())
        if total == 0:
            return 0.0, False
        action_prob = profile["action_counts"].get(action, 0) / total
        resource_prob = profile["resource_counts"].get(resource, 0) / total
        combined = (action_prob + resource_prob) / 2
        is_rare = combined < 0.01 and total > 10
        return combined, is_rare

class AlertGenerator:
    def __init__(self):
        self._alerts: List[AnomalyAlert] = []
        self._max_alerts = 10000
        self._callbacks: List[Callable[[AnomalyAlert], None]] = []
    def add_callback(self, callback: Callable[[AnomalyAlert], None]) -> None:
        self._callbacks.append(callback)
    def create_alert(self, anomaly_type: AnomalyType, severity: AlertSeverity,
                     description: str, metric_name: str = "", observed: float = 0.0,
                     expected: float = 0.0, score: float = 0.0, source: str = "") -> AnomalyAlert:
        alert = AnomalyAlert(
            alert_id=uuid.uuid4().hex[:12], anomaly_type=anomaly_type,
            severity=severity, description=description, source=source,
            metric_name=metric_name, observed_value=observed,
            expected_value=expected, score=score,
        )
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                pass
        return alert
    def get_alerts(self, severity: Optional[AlertSeverity] = None, limit: int = 100) -> List[AnomalyAlert]:
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

class AnomalyDetector:
    def __init__(self, z_threshold: float = 3.0, isolation_threshold: float = 0.6):
        self.baseline = StatisticalBaseline()
        self.z_analyzer = ZScoreAnalyzer(z_threshold)
        self.moving_avg = MovingAverage()
        self.isolation_forest = IsolationForest()
        self.profiler = BehavioralProfiler()
        self.alert_generator = AlertGenerator()
        self._detection_count = 0
    def add_observation(self, metric: str, value: float) -> Optional[AnomalyAlert]:
        self.baseline.add_observation(metric, value)
        ma_value = self.moving_avg.add(metric, value)
        stats = self.baseline.get_stats(metric)
        if stats:
            z_score, is_anomaly = self.z_analyzer.analyze(value, stats)
            if is_anomaly:
                self._detection_count += 1
                severity = AlertSeverity.CRITICAL if z_score > 5 else AlertSeverity.HIGH if z_score > 4 else AlertSeverity.MEDIUM
                return self.alert_generator.create_alert(
                    AnomalyType.STATISTICAL, severity,
                    f"Z-score anomaly in '{metric}': {value:.2f} (z={z_score:.2f}, mean={stats['mean']:.2f})",
                    metric_name=metric, observed=value, expected=stats["mean"], score=z_score,
                )
        return None
    def record_action(self, actor: str, action: str, resource: str, timestamp: Optional[float] = None) -> Optional[AnomalyAlert]:
        ts = timestamp or time.time()
        self.profiler.record(actor, action, resource, ts)
        prob, is_deviation = self.profiler.detect_deviation(actor, action, resource)
        if is_deviation:
            self._detection_count += 1
            return self.alert_generator.create_alert(
                AnomalyType.BEHAVIORAL, AlertSeverity.MEDIUM,
                f"Behavioral deviation for '{actor}': unusual action '{action}' on '{resource}' (prob={prob:.4f})",
                source=actor, score=1.0 - prob,
            )
        return None
    def detect_batch(self, data: List[List[float]]) -> List[Tuple[int, bool, float]]:
        if not data:
            return []
        self.isolation_forest.fit(data)
        results = []
        for i, point in enumerate(data):
            is_anom, score = self.isolation_forest.is_anomaly(point)
            results.append((i, is_anom, score))
        return results
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_detections": self._detection_count,
            "total_alerts": len(self.alert_generator.get_alerts()),
            "metrics_tracked": len(self.baseline.get_all_stats()),
        }
