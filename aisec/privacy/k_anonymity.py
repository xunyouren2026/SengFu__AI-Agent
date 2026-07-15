"""
K-Anonymity Implementation Module

Quasi-identifier detection, generalization hierarchies, k-anonymity check,
l-diversity, t-closeness, and anonymization algorithms (Mondrian).
"""

from __future__ import annotations

import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class AnonymityLevel(Enum):
    NONE = "none"
    K_ANONYMOUS = "k_anonymous"
    L_DIVerse = "l_diverse"
    T_CLOSE = "t_close"


@dataclass
class QuasiIdentifier:
    name: str
    data_type: str = "categorical"
    min_value: Any = None
    max_value: Any = None
    categories: List[str] = field(default_factory=list)
    generalization_levels: int = 5
    hierarchy: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.hierarchy and self.categories:
            self.hierarchy = self._build_default_hierarchy()

    def _build_default_hierarchy(self) -> List[List[str]]:
        if not self.categories:
            return []
        levels: List[List[str]] = [list(self.categories)]
        current = list(self.categories)
        while len(current) > 1:
            next_level: List[str] = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    next_level.append(f"{current[i]}|{current[i+1]}")
                else:
                    next_level.append(current[i])
            levels.append(next_level)
            current = next_level
        return levels

    def generalize(self, value: Any, level: int) -> str:
        if level == 0:
            return str(value)
        if self.data_type == "numerical" and self.min_value is not None:
            return self._generalize_numerical(value, level)
        if self.hierarchy:
            return self._generalize_categorical(value, level)
        return "*"

    def _generalize_numerical(self, value: Any, level: int) -> str:
        try:
            num = float(value)
            span = float(self.max_value) - float(self.min_value)
            if span == 0:
                return str(num)
            interval = span / (2 ** level)
            lower = float(self.min_value) + (int((num - float(self.min_value)) / interval) * interval)
            upper = lower + interval
            if level >= self.generalization_levels:
                return f"[{self.min_value}, {self.max_value}]"
            return f"[{lower:.1f}, {upper:.1f})"
        except (TypeError, ValueError):
            return str(value)

    def _generalize_categorical(self, value: Any, level: int) -> str:
        val_str = str(value)
        if level < len(self.hierarchy):
            for item in self.hierarchy[level]:
                if val_str in item:
                    return item
        return "*"


@dataclass
class GeneralizationHierarchy:
    quasi_identifiers: List[QuasiIdentifier] = field(default_factory=list)

    def add_qi(self, qi: QuasiIdentifier) -> None:
        self.quasi_identifiers.append(qi)

    def get_qi(self, name: str) -> Optional[QuasiIdentifier]:
        for qi in self.quasi_identifiers:
            if qi.name == name:
                return qi
        return None

    def generalize_record(self, record: Dict[str, Any], levels: Dict[str, int]) -> Dict[str, Any]:
        result = dict(record)
        for qi in self.quasi_identifiers:
            level = levels.get(qi.name, 0)
            if qi.name in result:
                result[qi.name] = qi.generalize(result[qi.name], level)
        return result


@dataclass
class EquivalenceClass:
    qi_values: Tuple[str, ...]
    records: List[Dict[str, Any]] = field(default_factory=list)
    sensitive_values: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def diversity(self) -> int:
        return len(set(self.sensitive_values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qi_values": self.qi_values,
            "size": self.size,
            "diversity": self.diversity,
            "sensitive_distribution": dict(Counter(self.sensitive_values)),
        }


class QuasiIdentifierDetector:
    def __init__(self):
        self._common_qi_names = {
            "age", "zip", "zipcode", "zip_code", "gender", "sex",
            "race", "ethnicity", "education", "occupation", "salary",
            "income", "country", "state", "city", "address", "birth_date",
            "date_of_birth", "marital_status", "nationality", "religion",
            "phone", "email_domain", "workplace",
        }
        self._numerical_patterns = {
            "age": (0, 120), "salary": (0, 10000000), "income": (0, 10000000),
            "zip": (10000, 99999), "year": (1900, 2030),
        }

    def detect(self, data: List[Dict[str, Any]], sensitive_attrs: Optional[List[str]] = None) -> List[str]:
        if not data:
            return []
        sensitive = set(sensitive_attrs or [])
        all_keys = set(data[0].keys())
        detected: List[str] = []
        for key in all_keys:
            if key in sensitive:
                continue
            if key.lower() in self._common_qi_names:
                detected.append(key)
                continue
            values = [record.get(key) for record in data if record.get(key) is not None]
            if len(values) < 2:
                continue
            unique_ratio = len(set(str(v) for v in values)) / len(values)
            if unique_ratio > 0.5 and unique_ratio < 0.99:
                detected.append(key)
                continue
            try:
                nums = [float(v) for v in values]
                if all(0 <= n <= 1000 for n in nums):
                    detected.append(key)
            except (TypeError, ValueError):
                pass
        return detected

    def build_qi(self, name: str, data: List[Dict[str, Any]]) -> QuasiIdentifier:
        values = [record.get(name) for record in data if record.get(name) is not None]
        if not values:
            return QuasiIdentifier(name=name)
        try:
            nums = [float(v) for v in values]
            return QuasiIdentifier(
                name=name, data_type="numerical",
                min_value=min(nums), max_value=max(nums),
            )
        except (TypeError, ValueError):
            cats = list(set(str(v) for v in values))
            return QuasiIdentifier(
                name=name, data_type="categorical", categories=cats,
            )


class KAnonymityChecker:
    def __init__(self):
        pass

    def check(self, data: List[Dict[str, Any]], qi_names: List[str], k: int) -> Tuple[bool, List[EquivalenceClass]]:
        classes = self._build_equivalence_classes(data, qi_names)
        violating = [ec for ec in classes if ec.size < k]
        return len(violating) == 0, classes

    def compute_k(self, data: List[Dict[str, Any]], qi_names: List[str]) -> int:
        classes = self._build_equivalence_classes(data, qi_names)
        if not classes:
            return 0
        return min(ec.size for ec in classes)

    def _build_equivalence_classes(self, data: List[Dict[str, Any]], qi_names: List[str]) -> List[EquivalenceClass]:
        groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for record in data:
            key = tuple(str(record.get(qi, "")) for qi in qi_names)
            groups[key].append(record)
        classes: List[EquivalenceClass] = []
        for qi_vals, records in groups.items():
            ec = EquivalenceClass(qi_values=qi_vals, records=records)
            classes.append(ec)
        return classes


class LDiversityChecker:
    def __init__(self):
        pass

    def check(self, classes: List[EquivalenceClass], sensitive_attr: str, l: int) -> Tuple[bool, List[EquivalenceClass]]:
        for ec in classes:
            ec.sensitive_values = [str(r.get(sensitive_attr, "")) for r in ec.records]
        violating = [ec for ec in classes if ec.diversity < l]
        return len(violating) == 0, violating

    def compute_l(self, classes: List[EquivalenceClass], sensitive_attr: str) -> int:
        for ec in classes:
            ec.sensitive_values = [str(r.get(sensitive_attr, "")) for r in ec.records]
        if not classes:
            return 0
        return min(ec.diversity for ec in classes)

    def entropy_l(self, classes: List[EquivalenceClass], sensitive_attr: str, l: float) -> Tuple[bool, float]:
        import math
        min_entropy = float("inf")
        for ec in classes:
            ec.sensitive_values = [str(r.get(sensitive_attr, "")) for r in ec.records]
            if ec.size == 0:
                continue
            counts = Counter(ec.sensitive_values)
            entropy = -sum((c / ec.size) * math.log(c / ec.size) for c in counts.values() if c > 0)
            min_entropy = min(min_entropy, entropy)
        return min_entropy >= math.log(l), min_entropy


class TCloseChecker:
    def __init__(self):
        pass

    def check(self, classes: List[EquivalenceClass], sensitive_attr: str, t: float, global_dist: Optional[Dict[str, float]] = None) -> Tuple[bool, float]:
        all_values = [str(r.get(sensitive_attr, "")) for ec in classes for r in ec.records]
        if global_dist is None:
            counts = Counter(all_values)
            total = len(all_values)
            global_dist = {v: c / total for v, c in counts.items()}
        max_distance = 0.0
        for ec in classes:
            ec.sensitive_values = [str(r.get(sensitive_attr, "")) for r in ec.records]
            if ec.size == 0:
                continue
            local_counts = Counter(ec.sensitive_values)
            local_dist = {v: c / ec.size for v, c in local_counts.items()}
            distance = sum(abs(local_dist.get(v, 0) - global_dist.get(v, 0)) for v in global_dist)
            max_distance = max(max_distance, distance)
        return max_distance <= t, max_distance


class MondrianAlgorithm:
    def __init__(self, k: int, qi_names: List[str], hierarchy: Optional[GeneralizationHierarchy] = None):
        self.k = k
        self.qi_names = qi_names
        self.hierarchy = hierarchy or GeneralizationHierarchy()

    def anonymize(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            return []
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        for qi_name in self.qi_names:
            qi = self.hierarchy.get_qi(qi_name)
            if qi and qi.data_type == "numerical":
                numeric_cols.append(qi_name)
            else:
                categorical_cols.append(qi_name)
        records = [dict(r) for r in data]
        partitions = self._mondrian(records, numeric_cols, categorical_cols)
        result: List[Dict[str, Any]] = []
        for partition in partitions:
            generalized = self._generalize_partition(partition, numeric_cols, categorical_cols)
            result.extend(generalized)
        return result

    def _mondrian(self, records: List[Dict[str, Any]], numeric_cols: List[str], categorical_cols: List[str]) -> List[List[Dict[str, Any]]]:
        if len(records) < 2 * self.k:
            return [records]
        best_col, best_split = self._find_best_split(records, numeric_cols, categorical_cols)
        if best_col is None:
            return [records]
        try:
            split_val = float(best_split)
            left = [r for r in records if float(r.get(best_col, 0)) < split_val]
            right = [r for r in records if float(r.get(best_col, 0)) >= split_val]
        except (TypeError, ValueError):
            left = [r for r in records if str(r.get(best_col, "")) == best_split]
            right = [r for r in records if str(r.get(best_col, "")) != best_split]
        if len(left) < self.k or len(right) < self.k:
            return [records]
        left_parts = self._mondrian(left, numeric_cols, categorical_cols)
        right_parts = self._mondrian(right, numeric_cols, categorical_cols)
        return left_parts + right_parts

    def _find_best_split(self, records: List[Dict[str, Any]], numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
        best_col: Optional[str] = None
        best_split: Optional[str] = None
        best_score = -1
        for col in numeric_cols:
            try:
                values = sorted(set(float(r.get(col, 0)) for r in records))
                for i in range(len(values) - 1):
                    mid = (values[i] + values[i + 1]) / 2
                    left_count = sum(1 for r in records if float(r.get(col, 0)) < mid)
                    right_count = len(records) - left_count
                    if left_count >= self.k and right_count >= self.k:
                        score = abs(left_count - right_count)
                        if best_score == -1 or score < best_score:
                            best_score = score
                            best_col = col
                            best_split = str(mid)
            except (TypeError, ValueError):
                continue
        if best_col is None:
            for col in categorical_cols:
                values = list(set(str(r.get(col, "")) for r in records))
                if len(values) >= 2:
                    best_col = col
                    best_split = values[0]
                    break
        return best_col, best_split

    def _generalize_partition(self, partition: List[Dict[str, Any]], numeric_cols: List[str], categorical_cols: List[str]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for qi_name in self.qi_names:
            qi = self.hierarchy.get_qi(qi_name)
            if qi and qi.data_type == "numerical" and qi_name in numeric_cols:
                try:
                    vals = [float(r.get(qi_name, 0)) for r in partition]
                    min_v, max_v = min(vals), max(vals)
                    for r in partition:
                        r[qi_name] = f"[{min_v:.1f}, {max_v:.1f})" if min_v != max_v else str(min_v)
                except (TypeError, ValueError):
                    pass
            elif qi and qi_name in categorical_cols:
                cats = list(set(str(r.get(qi_name, "")) for r in partition))
                gen_val = "|".join(cats) if len(cats) <= 3 else "*"
                for r in partition:
                    r[qi_name] = gen_val
        result.extend(partition)
        return result


class KAnonymizer:
    def __init__(self):
        self.qi_detector = QuasiIdentifierDetector()
        self.k_checker = KAnonymityChecker()
        self.l_checker = LDiversityChecker()
        self.t_checker = TCloseChecker()

    def anonymize(self, data: List[Dict[str, Any]], k: int = 2, qi_names: Optional[List[str]] = None,
                  sensitive_attrs: Optional[List[str]] = None) -> Dict[str, Any]:
        if not data:
            return {"anonymized": [], "k_achieved": 0, "original_count": 0}
        if qi_names is None:
            qi_names = self.qi_detector.detect(data, sensitive_attrs)
        if not qi_names:
            return {"anonymized": data, "k_achieved": len(data), "original_count": len(data), "qi_used": []}
        hierarchy = GeneralizationHierarchy()
        for qi_name in qi_names:
            qi = self.qi_detector.build_qi(qi_name, data)
            hierarchy.add_qi(qi)
        mondrian = MondrianAlgorithm(k=k, qi_names=qi_names, hierarchy=hierarchy)
        anonymized = mondrian.anonymize(data)
        achieved_k = self.k_checker.compute_k(anonymized, qi_names)
        return {
            "anonymized": anonymized,
            "k_achieved": achieved_k,
            "original_count": len(data),
            "anonymized_count": len(anonymized),
            "qi_used": qi_names,
            "sensitive_attrs": sensitive_attrs or [],
        }

    def check_k_anonymity(self, data: List[Dict[str, Any]], qi_names: List[str], k: int) -> Dict[str, Any]:
        is_k, classes = self.k_checker.check(data, qi_names, k)
        return {
            "is_k_anonymous": is_k,
            "k": k,
            "achieved_k": self.k_checker.compute_k(data, qi_names),
            "equivalence_classes": len(classes),
            "min_class_size": min(ec.size for ec in classes) if classes else 0,
            "max_class_size": max(ec.size for ec in classes) if classes else 0,
            "violating_classes": sum(1 for ec in classes if ec.size < k),
        }

    def check_l_diversity(self, data: List[Dict[str, Any]], qi_names: List[str],
                           sensitive_attr: str, l: int) -> Dict[str, Any]:
        _, classes = self.k_checker.check(data, qi_names, 1)
        is_l, violating = self.l_checker.check(classes, sensitive_attr, l)
        achieved_l = self.l_checker.compute_l(classes, sensitive_attr)
        return {
            "is_l_diverse": is_l,
            "l": l,
            "achieved_l": achieved_l,
            "violating_classes": len(violating),
        }

    def check_t_closeness(self, data: List[Dict[str, Any]], qi_names: List[str],
                           sensitive_attr: str, t: float) -> Dict[str, Any]:
        _, classes = self.k_checker.check(data, qi_names, 1)
        is_t, distance = self.t_checker.check(classes, sensitive_attr, t)
        return {
            "is_t_close": is_t,
            "t": t,
            "max_distance": distance,
        }


@dataclass
class AnonymizedDataset:
    data: List[Dict[str, Any]] = field(default_factory=list)
    k_achieved: int = 0
    l_achieved: int = 0
    t_distance: float = 0.0
    qi_names: List[str] = field(default_factory=list)
    sensitive_attrs: List[str] = field(default_factory=list)
    generalization_levels: Dict[str, int] = field(default_factory=dict)
    original_count: int = 0
    anonymized_count: int = 0
    information_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_achieved": self.k_achieved,
            "l_achieved": self.l_achieved,
            "t_distance": self.t_distance,
            "qi_names": self.qi_names,
            "sensitive_attrs": self.sensitive_attrs,
            "original_count": self.original_count,
            "anonymized_count": self.anonymized_count,
            "information_loss": self.information_loss,
            "record_count": len(self.data),
        }

    def compute_information_loss(self, original: List[Dict[str, Any]]) -> float:
        if not original or not self.data:
            return 0.0
        total_loss = 0.0
        total_fields = 0
        for qi in self.qi_names:
            orig_values = [str(r.get(qi, "")) for r in original]
            anon_values = [str(r.get(qi, "")) for r in self.data[:len(original)]]
            for ov, av in zip(orig_values, anon_values):
                if ov != av:
                    total_loss += 1.0
                total_fields += 1
        self.information_loss = total_loss / total_fields if total_fields else 0.0
        return self.information_loss
