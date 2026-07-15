"""
Synthetic Data Generation Module

Statistical profile extraction, marginal distribution preservation,
correlation maintenance, differential privacy noise, CTGAN simulation,
and data quality metrics.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PrivacyModel(Enum):
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    IDENTIFIER_REMOVAL = "identifier_removal"


class DistributionType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    name: str
    data_type: DistributionType
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Any] = None
    categories: List[Tuple[str, float]] = field(default_factory=list)
    histogram: List[Tuple[float, int]] = field(default_factory=list)
    skewness: float = 0.0
    kurtosis: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "data_type": self.data_type.value,
            "null_ratio": self.null_ratio, "unique_ratio": self.unique_ratio,
            "min": self.min_value, "max": self.max_value,
            "mean": self.mean, "std": self.std, "median": self.median,
            "mode": self.mode, "category_count": len(self.categories),
        }


@dataclass
class CorrelationPair:
    col1: str
    col2: str
    correlation: float
    method: str = "pearson"

    def to_dict(self) -> Dict[str, Any]:
        return {"col1": self.col1, "col2": self.col2, "correlation": self.correlation, "method": self.method}


class StatisticalProfiler:
    def __init__(self):
        self._profiles: Dict[str, ColumnProfile] = {}
        self._correlations: List[CorrelationPair] = []
        self._row_count: int = 0

    def profile(self, data: List[Dict[str, Any]]) -> Dict[str, ColumnProfile]:
        if not data:
            return {}
        self._row_count = len(data)
        columns = list(data[0].keys())
        for col in columns:
            values = [r.get(col) for r in data]
            self._profiles[col] = self._profile_column(col, values)
        self._compute_correlations(data)
        return dict(self._profiles)

    def _profile_column(self, name: str, values: List[Any]) -> ColumnProfile:
        non_null = [v for v in values if v is not None]
        null_count = len(values) - len(non_null)
        total = len(values)
        profile = ColumnProfile(
            name=name, data_type=DistributionType.UNKNOWN,
            null_count=null_count, null_ratio=null_count / total if total else 0,
            unique_count=len(set(str(v) for v in non_null)),
            unique_ratio=len(set(str(v) for v in non_null)) / len(non_null) if non_null else 0,
        )
        if not non_null:
            return profile
        try:
            nums = [float(v) for v in non_null]
            profile.data_type = DistributionType.INTEGER if all(float(v) == int(float(v)) for v in non_null[:100]) else DistributionType.GAUSSIAN
            profile.min_value = min(nums)
            profile.max_value = max(nums)
            profile.mean = sum(nums) / len(nums)
            variance = sum((x - profile.mean) ** 2 for x in nums) / len(nums)
            profile.std = math.sqrt(variance) if variance >= 0 else 0
            sorted_nums = sorted(nums)
            n = len(sorted_nums)
            if n % 2 == 0:
                profile.median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
            else:
                profile.median = sorted_nums[n // 2]
            profile.skewness = self._compute_skewness(nums, profile.mean, profile.std)
            profile.kurtosis = self._compute_kurtosis(nums, profile.mean, profile.std)
            num_bins = min(20, max(5, len(nums) // 10))
            bin_width = (max(nums) - min(nums)) / num_bins if max(nums) != min(nums) else 1
            for i in range(num_bins):
                lower = min(nums) + i * bin_width
                upper = lower + bin_width
                count = sum(1 for x in nums if lower <= x < upper)
                profile.histogram.append((lower, count))
        except (TypeError, ValueError):
            str_values = [str(v) for v in non_null]
            counts = Counter(str_values)
            total_str = len(str_values)
            profile.categories = [(v, c / total_str) for v, c in counts.most_common(50)]
            profile.mode = counts.most_common(1)[0][0] if counts else None
            if len(counts) <= 10:
                profile.data_type = DistributionType.CATEGORICAL
            else:
                profile.data_type = DistributionType.UNKNOWN
        return profile

    @staticmethod
    def _compute_skewness(nums: List[float], mean: float, std: float) -> float:
        if std == 0 or len(nums) < 3:
            return 0.0
        n = len(nums)
        return (n / ((n - 1) * (n - 2) * std ** 3)) * sum((x - mean) ** 3 for x in nums)

    @staticmethod
    def _compute_kurtosis(nums: List[float], mean: float, std: float) -> float:
        if std == 0 or len(nums) < 4:
            return 0.0
        n = len(nums)
        m4 = sum((x - mean) ** 4 for x in nums) / n
        return m4 / (std ** 4) - 3

    def _compute_correlations(self, data: List[Dict[str, Any]]) -> None:
        self._correlations = []
        numeric_cols = [name for name, p in self._profiles.items() if p.data_type in (DistributionType.GAUSSIAN, DistributionType.INTEGER)]
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                corr = self._pearson_correlation(
                    [r.get(col1) for r in data],
                    [r.get(col2) for r in data],
                )
                if abs(corr) > 0.05:
                    self._correlations.append(CorrelationPair(col1, col2, corr))

    @staticmethod
    def _pearson_correlation(x: List[Any], y: List[Any]) -> float:
        pairs = [(float(a), float(b)) for a, b in zip(x, y) if a is not None and b is not None]
        if len(pairs) < 3:
            return 0.0
        n = len(pairs)
        sum_x = sum(p[0] for p in pairs)
        sum_y = sum(p[1] for p in pairs)
        sum_xy = sum(p[0] * p[1] for p in pairs)
        sum_x2 = sum(p[0] ** 2 for p in pairs)
        sum_y2 = sum(p[1] ** 2 for p in pairs)
        denom = math.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    def get_profile(self, col: str) -> Optional[ColumnProfile]:
        return self._profiles.get(col)

    def get_correlations(self) -> List[CorrelationPair]:
        return list(self._correlations)


class DistributionFitter:
    def __init__(self):
        self._fitted: Dict[str, Dict[str, Any]] = {}

    def fit(self, profiles: Dict[str, ColumnProfile]) -> Dict[str, Dict[str, Any]]:
        for name, profile in profiles.items():
            if profile.data_type in (DistributionType.GAUSSIAN, DistributionType.INTEGER):
                self._fitted[name] = {
                    "type": "gaussian", "mean": profile.mean, "std": profile.std,
                    "min": profile.min_value, "max": profile.max_value,
                }
            elif profile.data_type == DistributionType.CATEGORICAL:
                self._fitted[name] = {
                    "type": "categorical", "categories": profile.categories,
                    "null_ratio": profile.null_ratio,
                }
            else:
                self._fitted[name] = {"type": "unknown", "null_ratio": profile.null_ratio}
        return dict(self._fitted)

    def sample(self, col: str) -> Any:
        params = self._fitted.get(col)
        if not params:
            return None
        if params["type"] == "gaussian":
            value = random.gauss(params["mean"] or 0, params["std"] or 1)
            if params["min"] is not None:
                value = max(params["min"], value)
            if params["max"] is not None:
                value = min(params["max"], value)
            if col in str(self._fitted.get("_int_cols", set())):
                return int(round(value))
            return round(value, 4)
        elif params["type"] == "categorical":
            r = random.random()
            cumulative = 0.0
            for cat, prob in params.get("categories", []):
                cumulative += prob
                if r <= cumulative:
                    return cat
            return params["categories"][0][0] if params.get("categories") else None
        return None


class CorrelationPreserver:
    def __init__(self):
        self._correlations: List[CorrelationPair] = []

    def set_correlations(self, correlations: List[CorrelationPair]) -> None:
        self._correlations = correlations

    def apply_correlations(self, synthetic: List[Dict[str, Any]], profiles: Dict[str, ColumnProfile]) -> List[Dict[str, Any]]:
        if not self._correlations:
            return synthetic
        for record in synthetic:
            for corr in self._correlations:
                if abs(corr.correlation) < 0.1:
                    continue
                p1 = profiles.get(corr.col1)
                p2 = profiles.get(corr.col2)
                if not p1 or not p2:
                    continue
                v1 = record.get(corr.col1)
                if v1 is None:
                    continue
                try:
                    v1_norm = (float(v1) - (p1.mean or 0)) / (p1.std or 1)
                    v2_norm = v1_norm * corr.correlation + random.gauss(0, math.sqrt(max(0, 1 - corr.correlation ** 2)))
                    v2 = v2_norm * (p2.std or 1) + (p2.mean or 0)
                    v2 = max(p2.min_value or 0, min(p2.max_value or 0, v2))
                    record[corr.col2] = round(v2, 4)
                except (TypeError, ValueError):
                    pass
        return synthetic


class PrivacyNoise:
    def __init__(self, epsilon: float = 1.0, mechanism: str = "laplacian"):
        self.epsilon = epsilon
        self.mechanism = mechanism

    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        if self.mechanism == "laplacian":
            scale = sensitivity / self.epsilon
            noise = random.laplace(0, scale)
            return value + noise
        elif self.mechanism == "gaussian":
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / 0.00001)) / self.epsilon
            noise = random.gauss(0, sigma)
            return value + noise
        return value

    def apply_to_record(self, record: Dict[str, Any], numeric_cols: List[str], sensitivity: float = 1.0) -> Dict[str, Any]:
        result = dict(record)
        for col in numeric_cols:
            try:
                result[col] = self.add_noise(float(record.get(col, 0)), sensitivity)
            except (TypeError, ValueError):
                pass
        return result


class CTGANSimulator:
    def __init__(self, epochs: int = 10, batch_size: int = 100, generator_lr: float = 0.001, discriminator_lr: float = 0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self._trained = False
        self._column_stats: Dict[str, Dict[str, Any]] = {}

    def train(self, data: List[Dict[str, Any]]) -> None:
        self._column_stats = {}
        for col in list(data[0].keys()):
            values = [r.get(col) for r in data if r.get(col) is not None]
            try:
                nums = [float(v) for v in values]
                self._column_stats[col] = {
                    "type": "numeric", "mean": sum(nums) / len(nums),
                    "std": math.sqrt(sum((x - sum(nums) / len(nums)) ** 2 for x in nums) / len(nums)),
                    "min": min(nums), "max": max(nums),
                    "histogram": self._build_histogram(nums, 20),
                }
            except (TypeError, ValueError):
                counts = Counter(str(v) for v in values)
                total = len(values)
                self._column_stats[col] = {
                    "type": "categorical",
                    "distribution": {v: c / total for v, c in counts.items()},
                }
        self._trained = True

    def generate(self, n: int) -> List[Dict[str, Any]]:
        if not self._trained:
            return []
        synthetic: List[Dict[str, Any]] = []
        for _ in range(n):
            record: Dict[str, Any] = {}
            for col, stats in self._column_stats.items():
                if stats["type"] == "numeric":
                    hist = stats.get("histogram", [])
                    if hist:
                        bins = [h[0] for h in hist]
                        weights = [h[1] for h in hist]
                        chosen_bin = random.choices(bins, weights=weights, k=1)[0]
                        value = chosen_bin + random.uniform(0, (stats["max"] - stats["min"]) / len(hist))
                        value = max(stats["min"], min(stats["max"], value))
                        record[col] = round(value, 4)
                    else:
                        record[col] = random.gauss(stats["mean"], stats["std"])
                elif stats["type"] == "categorical":
                    dist = stats.get("distribution", {})
                    if dist:
                        cats = list(dist.keys())
                        probs = list(dist.values())
                        record[col] = random.choices(cats, weights=probs, k=1)[0]
                    else:
                        record[col] = None
            synthetic.append(record)
        return synthetic

    @staticmethod
    def _build_histogram(nums: List[float], bins: int) -> List[Tuple[float, int]]:
        if not nums:
            return []
        min_v, max_v = min(nums), max(nums)
        if min_v == max_v:
            return [(min_v, len(nums))]
        bin_width = (max_v - min_v) / bins
        histogram: List[Tuple[float, int]] = []
        for i in range(bins):
            lower = min_v + i * bin_width
            count = sum(1 for x in nums if lower <= x < lower + bin_width)
            histogram.append((lower, count))
        return histogram


class QualityMetrics:
    def __init__(self):
        pass

    def evaluate(self, original: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not original or not synthetic:
            return {"error": "Empty data"}
        cols = list(original[0].keys())
        column_metrics: Dict[str, Dict[str, float]] = {}
        for col in cols:
            orig_vals = [r.get(col) for r in original if r.get(col) is not None]
            synth_vals = [r.get(col) for r in synthetic if r.get(col) is not None]
            try:
                orig_nums = [float(v) for v in orig_vals]
                synth_nums = [float(v) for v in synth_vals]
                metrics = {
                    "mean_diff": abs(sum(orig_nums) / len(orig_nums) - sum(synth_nums) / len(synth_nums)),
                    "std_diff": abs(self._std(orig_nums) - self._std(synth_nums)),
                    "min_diff": abs(min(orig_nums) - min(synth_nums)),
                    "max_diff": abs(max(orig_nums) - max(synth_nums)),
                }
                metrics["column_similarity"] = max(0, 1 - metrics["mean_diff"] / (abs(sum(orig_nums) / len(orig_nums)) + 1e-9))
                column_metrics[col] = metrics
            except (TypeError, ValueError):
                orig_counts = Counter(str(v) for v in orig_vals)
                synth_counts = Counter(str(v) for v in synth_vals)
                all_cats = set(orig_counts.keys()) | set(synth_counts.keys())
                total_orig = len(orig_vals)
                total_synth = len(synth_vals)
                kl_div = 0.0
                for cat in all_cats:
                    p = orig_counts.get(cat, 0) / total_orig
                    q = synth_counts.get(cat, 0) / total_synth
                    if p > 0 and q > 0:
                        kl_div += p * math.log(p / q)
                column_metrics[col] = {"kl_divergence": kl_div, "column_similarity": max(0, 1 - kl_div)}
        overall = self._compute_overall(column_metrics)
        return {"column_metrics": column_metrics, "overall": overall, "rows_original": len(original), "rows_synthetic": len(synthetic)}

    @staticmethod
    def _std(nums: List[float]) -> float:
        if len(nums) < 2:
            return 0.0
        mean = sum(nums) / len(nums)
        return math.sqrt(sum((x - mean) ** 2 for x in nums) / (len(nums) - 1))

    def _compute_overall(self, column_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not column_metrics:
            return {"similarity": 0.0}
        similarities = [m.get("column_similarity", 0) for m in column_metrics.values()]
        return {
            "similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
        }


class SyntheticValidator:
    def __init__(self):
        self.quality_metrics = QualityMetrics()

    def validate(self, original: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> Dict[str, Any]:
        quality = self.quality_metrics.evaluate(original, synthetic)
        privacy = self._check_privacy(original, synthetic)
        completeness = self._check_completeness(original, synthetic)
        return {
            "quality": quality,
            "privacy": privacy,
            "completeness": completeness,
            "is_valid": privacy["privacy_score"] > 0.5 and quality["overall"]["similarity"] > 0.3,
        }

    def _check_privacy(self, original: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> Dict[str, Any]:
        exact_matches = 0
        for o in original:
            o_str = json.dumps(o, sort_keys=True, default=str)
            for s in synthetic:
                s_str = json.dumps(s, sort_keys=True, default=str)
                if o_str == s_str:
                    exact_matches += 1
                    break
        ratio = exact_matches / len(original) if original else 0
        return {"exact_match_count": exact_matches, "exact_match_ratio": ratio, "privacy_score": max(0, 1 - ratio * 5)}

    def _check_completeness(self, original: List[Dict[str, Any]], synthetic: List[Dict[str, Any]]) -> Dict[str, Any]:
        orig_cols = set(original[0].keys()) if original else set()
        synth_cols = set(synthetic[0].keys()) if synthetic else set()
        return {
            "column_coverage": len(orig_cols & synth_cols) / len(orig_cols) if orig_cols else 0,
            "missing_columns": list(orig_cols - synth_cols),
            "extra_columns": list(synth_cols - orig_cols),
        }


class SyntheticDataGenerator:
    def __init__(self, epsilon: float = 1.0, privacy_model: PrivacyModel = PrivacyModel.DIFFERENTIAL_PRIVACY):
        self.profiler = StatisticalProfiler()
        self.fitter = DistributionFitter()
        self.correlation_preserver = CorrelationPreserver()
        self.privacy_noise = PrivacyNoise(epsilon=epsilon)
        self.ctgan_simulator = CTGANSimulator()
        self.quality_metrics = QualityMetrics()
        self.validator = SyntheticValidator()
        self.privacy_model = privacy_model
        self._last_profile: Dict[str, ColumnProfile] = {}

    def generate(self, data: List[Dict[str, Any]], n: int = 100, method: str = "statistical") -> Dict[str, Any]:
        if not data:
            return {"synthetic": [], "quality": {}}
        self._last_profile = self.profiler.profile(data)
        self.fitter.fit(self._last_profile)
        self.correlation_preserver.set_correlations(self.profiler.get_correlations())
        if method == "ctgan":
            self.ctgan_simulator.train(data)
            synthetic = self.ctgan_simulator.generate(n)
        else:
            synthetic = self._generate_statistical(n)
        synthetic = self.correlation_preserver.apply_correlations(synthetic, self._last_profile)
        if self.privacy_model == PrivacyModel.DIFFERENTIAL_PRIVACY:
            numeric_cols = [name for name, p in self._last_profile.items() if p.data_type in (DistributionType.GAUSSIAN, DistributionType.INTEGER)]
            synthetic = [self.privacy_noise.apply_to_record(r, numeric_cols) for r in synthetic]
        quality = self.quality_metrics.evaluate(data, synthetic)
        validation = self.validator.validate(data, synthetic)
        return {
            "synthetic": synthetic,
            "quality": quality,
            "validation": validation,
            "method": method,
            "privacy_model": self.privacy_model.value,
            "rows_generated": len(synthetic),
        }

    def _generate_statistical(self, n: int) -> List[Dict[str, Any]]:
        synthetic: List[Dict[str, Any]] = []
        for _ in range(n):
            record: Dict[str, Any] = {}
            for col in self._last_profile:
                value = self.fitter.sample(col)
                record[col] = value
            synthetic.append(record)
        return synthetic
