"""
Evaluation Module - Metrics

Provides a comprehensive set of evaluation metrics for classification,
regression, NLP, and similarity tasks, all implemented using only
the Python standard library.
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Metric Base
# ---------------------------------------------------------------------------
class Metric(ABC):
    """Abstract base class for all evaluation metrics."""

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self._value: Optional[float] = None

    @abstractmethod
    def compute(self, predictions: Any, targets: Any = None) -> float:
        """
        Compute the metric value.

        Args:
            predictions: Model predictions.
            targets: Ground truth values.

        Returns:
            Metric value as a float.
        """
        ...

    def reset(self) -> None:
        """Reset the metric state."""
        self._value = None

    def get_value(self) -> float:
        """Get the last computed value."""
        return self._value if self._value is not None else 0.0

    def __call__(self, predictions: Any, targets: Any = None) -> float:
        """Allow calling the metric directly."""
        self._value = self.compute(predictions, targets)
        return self._value


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
class Accuracy(Metric):
    """Classification accuracy metric."""

    def __init__(self) -> None:
        super().__init__(name="accuracy")

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute accuracy as fraction of correct predictions."""
        if not targets:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct / len(targets)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
class Precision(Metric):
    """Precision metric with support for macro/micro/weighted averaging."""

    def __init__(self, average: str = "macro", pos_label: Any = 1) -> None:
        """
        Args:
            average: One of 'macro', 'micro', 'weighted', or 'binary'.
            pos_label: The positive class label for binary mode.
        """
        super().__init__(name=f"precision_{average}")
        self.average = average
        self.pos_label = pos_label

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute precision."""
        if not targets:
            return 0.0

        if self.average == "binary":
            tp = sum(1 for p, t in zip(predictions, targets) if p == self.pos_label and t == self.pos_label)
            fp = sum(1 for p, t in zip(predictions, targets) if p == self.pos_label and t != self.pos_label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        labels = set(targets) | set(predictions)
        precisions = {}

        for label in labels:
            tp = sum(1 for p, t in zip(predictions, targets) if p == label and t == label)
            fp = sum(1 for p, t in zip(predictions, targets) if p == label and t != label)
            precisions[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if self.average == "macro":
            return sum(precisions.values()) / len(precisions) if precisions else 0.0
        elif self.average == "micro":
            total_tp = sum(
                sum(1 for p, t in zip(predictions, targets) if p == label and t == label)
                for label in labels
            )
            total_fp = sum(
                sum(1 for p, t in zip(predictions, targets) if p == label and t != label)
                for label in labels
            )
            return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        elif self.average == "weighted":
            target_counts = Counter(targets)
            total = len(targets)
            weighted_sum = sum(precisions[label] * target_counts.get(label, 0) for label in labels)
            return weighted_sum / total if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown average mode: {self.average}")


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
class Recall(Metric):
    """Recall metric with support for macro/micro/weighted averaging."""

    def __init__(self, average: str = "macro", pos_label: Any = 1) -> None:
        super().__init__(name=f"recall_{average}")
        self.average = average
        self.pos_label = pos_label

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute recall."""
        if not targets:
            return 0.0

        if self.average == "binary":
            tp = sum(1 for p, t in zip(predictions, targets) if p == self.pos_label and t == self.pos_label)
            fn = sum(1 for p, t in zip(predictions, targets) if p != self.pos_label and t == self.pos_label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        labels = set(targets) | set(predictions)
        recalls = {}

        for label in labels:
            tp = sum(1 for p, t in zip(predictions, targets) if p == label and t == label)
            fn = sum(1 for p, t in zip(predictions, targets) if p != label and t == label)
            recalls[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if self.average == "macro":
            return sum(recalls.values()) / len(recalls) if recalls else 0.0
        elif self.average == "micro":
            total_tp = sum(
                sum(1 for p, t in zip(predictions, targets) if p == label and t == label)
                for label in labels
            )
            total_fn = sum(
                sum(1 for p, t in zip(predictions, targets) if p != label and t == label)
                for label in labels
            )
            return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        elif self.average == "weighted":
            target_counts = Counter(targets)
            total = len(targets)
            weighted_sum = sum(recalls[label] * target_counts.get(label, 0) for label in labels)
            return weighted_sum / total if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown average mode: {self.average}")


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------
class F1Score(Metric):
    """F1 score metric with support for macro/micro/weighted averaging."""

    def __init__(self, average: str = "macro", pos_label: Any = 1) -> None:
        super().__init__(name=f"f1_{average}")
        self.average = average
        self.pos_label = pos_label

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute F1 score."""
        precision_metric = Precision(average=self.average, pos_label=self.pos_label)
        recall_metric = Recall(average=self.average, pos_label=self.pos_label)

        p = precision_metric.compute(predictions, targets)
        r = recall_metric.compute(predictions, targets)

        if p + r == 0:
            return 0.0
        return 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------
class AUC(Metric):
    """Area Under the ROC Curve using the trapezoidal rule."""

    def __init__(self) -> None:
        super().__init__(name="auc_roc")

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute AUC-ROC.

        Args:
            predictions: List of predicted scores/probabilities.
            targets: List of binary labels (0 or 1).

        Returns:
            AUC score.
        """
        if not predictions or not targets:
            return 0.0

        # Pair scores with labels and sort by score descending
        pairs = list(zip(predictions, targets))
        pairs.sort(key=lambda x: x[0], reverse=True)

        total_pos = sum(1 for _, t in pairs if t == 1)
        total_neg = sum(1 for _, t in pairs if t == 0)

        if total_pos == 0 or total_neg == 0:
            return 0.0

        # Compute ROC curve points using trapezoidal rule
        tp = 0
        fp = 0
        prev_fpr = 0.0
        prev_tpr = 0.0
        auc = 0.0

        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1

            tpr = tp / total_pos
            fpr = fp / total_neg

            # Trapezoidal area
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0

            prev_fpr = fpr
            prev_tpr = tpr

        return auc


# ---------------------------------------------------------------------------
# BLEU Score
# ---------------------------------------------------------------------------
class BLEUScore(Metric):
    """BLEU (Bilingual Evaluation Understudy) score for text generation."""

    def __init__(self, max_n: int = 4) -> None:
        """
        Args:
            max_n: Maximum n-gram order to consider (typically 4).
        """
        super().__init__(name=f"bleu_{max_n}")
        self.max_n = max_n

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-gram counts from a token list."""
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    def _brevity_penalty(self, hypothesis: List[str], reference: List[str]) -> float:
        """Compute brevity penalty."""
        hyp_len = len(hypothesis)
        ref_len = len(reference)
        if hyp_len > ref_len:
            return 1.0
        elif hyp_len == 0:
            return 0.0
        else:
            return math.exp(1.0 - ref_len / hyp_len)

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute BLEU score.

        Args:
            predictions: List of hypothesis strings or token lists.
            targets: List of reference strings or token lists.
        """
        if not predictions or not targets:
            return 0.0

        total_bleu = 0.0

        for hyp, ref in zip(predictions, targets):
            # Tokenize if strings
            if isinstance(hyp, str):
                hyp_tokens = hyp.lower().split()
            else:
                hyp_tokens = list(hyp)

            if isinstance(ref, str):
                ref_tokens = ref.lower().split()
            else:
                ref_tokens = list(ref)

            if len(hyp_tokens) == 0:
                continue

            # Compute modified n-gram precision for each n
            precisions = []
            for n in range(1, self.max_n + 1):
                if len(hyp_tokens) < n:
                    precisions.append(0.0)
                    continue

                hyp_ngrams = self._get_ngrams(hyp_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)

                # Clipped count
                clipped = 0
                total = 0
                for ngram, count in hyp_ngrams.items():
                    clipped += min(count, ref_ngrams.get(ngram, 0))
                    total += count

                if total == 0:
                    precisions.append(0.0)
                else:
                    precisions.append(clipped / total)

            # Compute geometric mean of precisions
            log_avg = 0.0
            weight = 1.0 / self.max_n
            valid = True
            for p in precisions:
                if p == 0:
                    valid = False
                    break
                log_avg += weight * math.log(p)

            if not valid:
                total_bleu += 0.0
            else:
                bp = self._brevity_penalty(hyp_tokens, ref_tokens)
                total_bleu += bp * math.exp(log_avg)

        return total_bleu / len(predictions)


# ---------------------------------------------------------------------------
# ROUGE Score
# ---------------------------------------------------------------------------
class ROUGEScore(Metric):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score."""

    def __init__(self, rouge_type: str = "rouge_l") -> None:
        """
        Args:
            rouge_type: One of 'rouge_1', 'rouge_2', 'rouge_l'.
        """
        super().__init__(name=rouge_type)
        self.rouge_type = rouge_type

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Compute the length of the longest common subsequence."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0

        # Use two-row DP for space efficiency
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute ROUGE score.

        Args:
            predictions: List of hypothesis strings or token lists.
            targets: List of reference strings or token lists.
        """
        if not predictions or not targets:
            return 0.0

        total_f1 = 0.0

        for hyp, ref in zip(predictions, targets):
            if isinstance(hyp, str):
                hyp_tokens = hyp.lower().split()
            else:
                hyp_tokens = list(hyp)

            if isinstance(ref, str):
                ref_tokens = ref.lower().split()
            else:
                ref_tokens = list(ref)

            if not hyp_tokens or not ref_tokens:
                continue

            if self.rouge_type == "rouge_1":
                # Unigram overlap
                hyp_counts = Counter(hyp_tokens)
                ref_counts = Counter(ref_tokens)
                overlap = sum((hyp_counts & ref_counts).values())
                precision = overlap / len(hyp_tokens)
                recall = overlap / len(ref_tokens)
            elif self.rouge_type == "rouge_2":
                # Bigram overlap
                hyp_bigrams = self._get_ngrams(hyp_tokens, 2)
                ref_bigrams = self._get_ngrams(ref_tokens, 2)
                overlap = sum((hyp_bigrams & ref_bigrams).values())
                hyp_total = sum(hyp_bigrams.values())
                ref_total = sum(ref_bigrams.values())
                precision = overlap / hyp_total if hyp_total > 0 else 0.0
                recall = overlap / ref_total if ref_total > 0 else 0.0
            elif self.rouge_type == "rouge_l":
                # Longest common subsequence
                lcs_len = self._lcs_length(hyp_tokens, ref_tokens)
                precision = lcs_len / len(hyp_tokens)
                recall = lcs_len / len(ref_tokens)
            else:
                raise ValueError(f"Unknown ROUGE type: {self.rouge_type}")

            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            total_f1 += f1

        return total_f1 / len(predictions)

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-gram counts."""
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------
class Perplexity(Metric):
    """Perplexity metric for language models."""

    def __init__(self) -> None:
        super().__init__(name="perplexity")

    def compute(self, predictions: Any, targets: Any = None) -> float:
        """
        Compute perplexity from cross-entropy loss.

        Args:
            predictions: Cross-entropy loss values (list of floats) or probabilities.
            targets: Not used when predictions are losses; used when predictions are probs.

        Returns:
            Perplexity value (lower is better).
        """
        if not predictions:
            return float("inf")

        # If predictions are loss values
        if isinstance(predictions, (int, float)):
            return math.exp(min(predictions, 100))  # Clamp to avoid overflow

        if isinstance(predictions[0], (int, float)) and not isinstance(predictions[0], bool):
            # Assume these are loss values
            avg_loss = sum(predictions) / len(predictions)
            return math.exp(min(avg_loss, 100))

        # If predictions are probability distributions
        if isinstance(predictions[0], list):
            total_log_prob = 0.0
            count = 0
            for probs, target in zip(predictions, targets):
                target = int(target)
                if 0 <= target < len(probs):
                    prob = max(probs[target], 1e-12)
                    total_log_prob += math.log(prob)
                    count += 1
            if count == 0:
                return float("inf")
            avg_log_prob = total_log_prob / count
            return math.exp(-avg_log_prob)

        return float("inf")


# ---------------------------------------------------------------------------
# Edit Distance (Levenshtein)
# ---------------------------------------------------------------------------
class EditDistance(Metric):
    """Levenshtein edit distance metric."""

    def __init__(self, normalize: bool = True) -> None:
        """
        Args:
            normalize: If True, return normalized distance (0 to 1).
        """
        super().__init__(name="edit_distance")
        self.normalize = normalize

    @staticmethod
    def _levenshtein(s1: List[Any], s2: List[Any]) -> int:
        """Compute Levenshtein distance between two sequences."""
        m, n = len(s1), len(s2)
        if m == 0:
            return n
        if n == 0:
            return m

        # Two-row DP
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute edit distance.

        Args:
            predictions: List of predicted strings or token lists.
            targets: List of reference strings or token lists.
        """
        if not predictions or not targets:
            return 0.0

        total_distance = 0.0
        total_length = 0

        for hyp, ref in zip(predictions, targets):
            if isinstance(hyp, str):
                hyp_seq = list(hyp)
            else:
                hyp_seq = list(hyp)

            if isinstance(ref, str):
                ref_seq = list(ref)
            else:
                ref_seq = list(ref)

            dist = self._levenshtein(hyp_seq, ref_seq)
            total_distance += dist
            total_length += max(len(hyp_seq), len(ref_seq))

        if self.normalize:
            return total_distance / total_length if total_length > 0 else 0.0
        return total_distance / len(predictions)


# ---------------------------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------------------------
class SemanticSimilarity(Metric):
    """Semantic similarity using cosine similarity and Jaccard index."""

    def __init__(self, method: str = "cosine") -> None:
        """
        Args:
            method: 'cosine' for cosine similarity, 'jaccard' for Jaccard index.
        """
        super().__init__(name=f"semantic_similarity_{method}")
        self.method = method

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    @staticmethod
    def _jaccard_index(set_a: set, set_b: set) -> float:
        """Compute Jaccard index between two sets."""
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute semantic similarity.

        Args:
            predictions: List of embedding vectors or token sets.
            targets: List of embedding vectors or token sets.
        """
        if not predictions or not targets:
            return 0.0

        total_sim = 0.0

        for pred, target in zip(predictions, targets):
            if self.method == "cosine":
                sim = self._cosine_similarity(
                    list(pred) if not isinstance(pred, list) else pred,
                    list(target) if not isinstance(target, list) else target,
                )
            elif self.method == "jaccard":
                if isinstance(pred, str):
                    pred_set = set(pred.lower().split())
                else:
                    pred_set = set(pred)

                if isinstance(target, str):
                    target_set = set(target.lower().split())
                else:
                    target_set = set(target)

                sim = self._jaccard_index(pred_set, target_set)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            total_sim += sim

        return total_sim / len(predictions)


# ---------------------------------------------------------------------------
# Metrics Tracker
# ---------------------------------------------------------------------------
class MetricsTracker:
    """Tracks metric values over time and provides analysis utilities."""

    def __init__(self) -> None:
        self._history: Dict[str, List[float]] = {}
        self._step_history: Dict[str, List[int]] = {}

    def track(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Track a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if name not in self._history:
            self._history[name] = []
            self._step_history[name] = []

        self._history[name].append(value)
        if step is not None:
            self._step_history[name].append(step)
        else:
            self._step_history[name].append(len(self._history[name]))

    def get_history(self, name: str) -> List[float]:
        """Get the full history of a metric."""
        return list(self._history.get(name, []))

    def get_all_histories(self) -> Dict[str, List[float]]:
        """Get histories for all tracked metrics."""
        return {k: list(v) for k, v in self._history.items()}

    def get_latest(self, name: str) -> Optional[float]:
        """Get the most recent value of a metric."""
        history = self._history.get(name, [])
        return history[-1] if history else None

    def get_best(self, name: str, mode: str = "max") -> Optional[float]:
        """
        Get the best value of a metric.

        Args:
            name: Metric name.
            mode: 'max' for highest value, 'min' for lowest value.
        """
        history = self._history.get(name, [])
        if not history:
            return None
        if mode == "max":
            return max(history)
        return min(history)

    def get_best_step(self, name: str, mode: str = "max") -> Optional[int]:
        """Get the step at which the best value occurred."""
        history = self._history.get(name, [])
        steps = self._step_history.get(name, [])
        if not history:
            return None
        if mode == "max":
            best_idx = history.index(max(history))
        else:
            best_idx = history.index(min(history))
        return steps[best_idx] if best_idx < len(steps) else None

    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """
        Get the average of a metric.

        Args:
            name: Metric name.
            last_n: If set, average only the last N values.
        """
        history = self._history.get(name, [])
        if not history:
            return 0.0
        if last_n is not None:
            history = history[-last_n:]
        return sum(history) / len(history)

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get descriptive statistics for a metric."""
        history = self._history.get(name, [])
        if not history:
            return {}

        n = len(history)
        mean = sum(history) / n
        variance = sum((v - mean) ** 2 for v in history) / n
        std = math.sqrt(variance)

        sorted_h = sorted(history)
        median = sorted_h[n // 2] if n % 2 == 1 else (sorted_h[n // 2 - 1] + sorted_h[n // 2]) / 2

        return {
            "count": n,
            "mean": mean,
            "std": std,
            "min": min(history),
            "max": max(history),
            "median": median,
            "last": history[-1],
        }

    def list_metrics(self) -> List[str]:
        """List all tracked metric names."""
        return list(self._history.keys())

    def reset(self) -> None:
        """Clear all tracked metrics."""
        self._history = {}
        self._step_history = {}

    def plot(self, name: str, width: int = 60, height: int = 15) -> str:
        """
        Generate an ASCII chart of a metric's history.

        Args:
            name: Metric name.
            width: Chart width in characters.
            height: Chart height in rows.

        Returns:
            ASCII string representation of the chart.
        """
        history = self._history.get(name, [])
        if not history:
            return f"No data for metric '{name}'"

        min_val = min(history)
        max_val = max(history)
        val_range = max_val - min_val
        if val_range == 0:
            val_range = 1.0

        lines = []
        lines.append(f"  Metric: {name}")
        lines.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
        lines.append(f"  Points: {len(history)}")
        lines.append("")

        # Downsample if needed
        if len(history) > width:
            step = len(history) / width
            sampled = [history[int(i * step)] for i in range(width)]
        else:
            sampled = history
            width = len(sampled)

        for row in range(height, -1, -1):
            threshold = min_val + (row / height) * val_range
            label = f"{threshold:8.4f} |"
            bar = ""
            for val in sampled:
                normalized = (val - min_val) / val_range
                char_row = int(normalized * height)
                if char_row >= row:
                    bar += "#"
                else:
                    bar += " "
            lines.append(f"{label}{bar}")

        lines.append(f"{'':>10}+{'-' * width}")
        return "\n".join(lines)
