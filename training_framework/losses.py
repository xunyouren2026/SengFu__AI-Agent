"""
Training Framework - Loss Functions Module

Provides various loss functions implemented from scratch using only
the Python standard library. Supports scalar, list, and batch inputs.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Numerical Helpers
# ---------------------------------------------------------------------------
def _ensure_float(x: Any) -> float:
    """Convert a value to float."""
    if isinstance(x, (list, tuple)):
        return float(x[0]) if len(x) == 1 else float(sum(x) / len(x))
    return float(x)


def _softmax(logits: List[float]) -> List[float]:
    """Compute softmax probabilities from logits."""
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _log_softmax(logits: List[float]) -> List[float]:
    """Compute log-softmax from logits (numerically stable)."""
    max_val = max(logits)
    shifted = [x - max_val for x in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    return [s - log_sum_exp for s in shifted]


def _sigmoid(x: float) -> float:
    """Compute sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def _mean(values: List[float]) -> float:
    """Compute mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Loss Function Base
# ---------------------------------------------------------------------------
class LossFunction(ABC):
    """Abstract base class for all loss functions."""

    @abstractmethod
    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute the loss.

        Args:
            predictions: Model predictions (scalar, list, or nested structure).
            targets: Ground truth values.

        Returns:
            Scalar loss value.
        """
        ...

    def __call__(self, predictions: Any, targets: Any) -> float:
        """Allow calling the loss function directly."""
        return self.compute(predictions, targets)


# ---------------------------------------------------------------------------
# Cross Entropy Loss
# ---------------------------------------------------------------------------
class CrossEntropyLoss(LossFunction):
    """Cross-entropy loss with optional label smoothing and class weights."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: Optional[List[float]] = None,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            label_smoothing: Smoothing factor in [0, 1).
            class_weights: Optional per-class weights.
            reduction: 'mean', 'sum', or 'none'.
        """
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute cross-entropy loss.

        Args:
            predictions: List of logits (batch_size, num_classes) or single list.
            targets: List of class indices or single index.
        """
        # Handle single sample
        if isinstance(targets, int):
            predictions = [predictions] if not isinstance(predictions[0], list) else [predictions]
            targets = [targets]

        # Handle batch of logits
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            # predictions is already a batch
            batch_logits = predictions
            batch_targets = targets if isinstance(targets, list) else [targets]
        else:
            batch_logits = [predictions]
            batch_targets = [targets]

        num_classes = len(batch_logits[0])
        losses = []

        for logits, target in zip(batch_logits, batch_targets):
            log_probs = _log_softmax(logits)
            target = int(target)

            if self.class_weights is not None:
                weight = self.class_weights[target]
            else:
                weight = 1.0

            if self.label_smoothing > 0:
                # Smoothed: -sum(q * log(p)) where q = (1-smooth)*one_hot + smooth/num_classes
                smooth_loss = 0.0
                for c in range(num_classes):
                    q = self.label_smoothing / num_classes
                    if c == target:
                        q += 1.0 - self.label_smoothing
                    smooth_loss -= q * log_probs[c]
                losses.append(smooth_loss * weight)
            else:
                losses.append(-log_probs[target] * weight)

        if self.reduction == "sum":
            return sum(losses)
        elif self.reduction == "none":
            return _mean(losses)  # Return as average for scalar output
        else:  # mean
            return _mean(losses)


# ---------------------------------------------------------------------------
# MSE Loss
# ---------------------------------------------------------------------------
class MSELoss(LossFunction):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute MSE loss.

        Args:
            predictions: Predicted values (scalar, list, or list of lists).
            targets: Ground truth values.
        """
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
            targets = [targets]

        if isinstance(predictions[0], list):
            # Batch of sequences
            losses = []
            for pred_seq, tgt_seq in zip(predictions, targets):
                for p, t in zip(pred_seq, tgt_seq):
                    losses.append((float(p) - float(t)) ** 2)
        else:
            losses = [(float(p) - float(t)) ** 2 for p, t in zip(predictions, targets)]

        if self.reduction == "sum":
            return sum(losses)
        elif self.reduction == "none":
            return _mean(losses)
        else:
            return _mean(losses)


# ---------------------------------------------------------------------------
# L1 Loss
# ---------------------------------------------------------------------------
class L1Loss(LossFunction):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute L1 loss."""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
            targets = [targets]

        if isinstance(predictions[0], list):
            losses = []
            for pred_seq, tgt_seq in zip(predictions, targets):
                for p, t in zip(pred_seq, tgt_seq):
                    losses.append(abs(float(p) - float(t)))
        else:
            losses = [abs(float(p) - float(t)) for p, t in zip(predictions, targets)]

        if self.reduction == "sum":
            return sum(losses)
        elif self.reduction == "none":
            return _mean(losses)
        else:
            return _mean(losses)


# ---------------------------------------------------------------------------
# Huber Loss (Smooth L1)
# ---------------------------------------------------------------------------
class HuberLoss(LossFunction):
    """Huber loss (Smooth L1 loss).

    Less sensitive to outliers than MSE.
    L(x) = 0.5 * x^2                  if |x| <= delta
    L(x) = delta * |x| - 0.5 * delta^2  if |x| > delta
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        self.delta = delta
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """Compute Huber loss."""
        if isinstance(predictions, (int, float)):
            predictions = [predictions]
            targets = [targets]

        if isinstance(predictions[0], list):
            losses = []
            for pred_seq, tgt_seq in zip(predictions, targets):
                for p, t in zip(pred_seq, tgt_seq):
                    diff = abs(float(p) - float(t))
                    if diff <= self.delta:
                        losses.append(0.5 * diff * diff)
                    else:
                        losses.append(self.delta * diff - 0.5 * self.delta * self.delta)
        else:
            losses = []
            for p, t in zip(predictions, targets):
                diff = abs(float(p) - float(t))
                if diff <= self.delta:
                    losses.append(0.5 * diff * diff)
                else:
                    losses.append(self.delta * diff - 0.5 * self.delta * self.delta)

        if self.reduction == "sum":
            return sum(losses)
        elif self.reduction == "none":
            return _mean(losses)
        else:
            return _mean(losses)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(LossFunction):
    """Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples and focuses on hard examples.
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha: Per-class weights. If None, all classes weighted equally.
            gamma: Focusing parameter. Higher = more focus on hard examples.
            reduction: 'mean' or 'sum'.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute focal loss.

        Args:
            predictions: List of logits or batch of logits.
            targets: Class indices.
        """
        if isinstance(targets, int):
            predictions = [predictions] if not isinstance(predictions[0], list) else [predictions]
            targets = [targets]

        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            batch_logits = predictions
            batch_targets = targets if isinstance(targets, list) else [targets]
        else:
            batch_logits = [predictions]
            batch_targets = [targets]

        losses = []
        for logits, target in zip(batch_logits, batch_targets):
            probs = _softmax(logits)
            target = int(target)
            p_t = probs[target]

            # Compute focal weight: (1 - p_t)^gamma
            focal_weight = (1.0 - p_t) ** self.gamma

            # Compute cross-entropy: -log(p_t)
            ce_loss = -math.log(max(p_t, 1e-12))

            # Apply alpha weighting
            if self.alpha is not None:
                alpha_t = self.alpha[target]
            else:
                alpha_t = 1.0

            losses.append(alpha_t * focal_weight * ce_loss)

        if self.reduction == "sum":
            return sum(losses)
        else:
            return _mean(losses)


# ---------------------------------------------------------------------------
# Contrastive Loss
# ---------------------------------------------------------------------------
class ContrastiveLoss(LossFunction):
    """Contrastive loss for siamese networks.

    L = (1 - y) * 0.5 * d^2 + y * 0.5 * max(margin - d, 0)^2

    where d is the Euclidean distance, y is 1 for similar pairs, 0 for dissimilar.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        self.margin = margin
        self.reduction = reduction

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute contrastive loss.

        Args:
            predictions: Tuple of (embedding_a, embedding_b) or distance.
            targets: 1 for similar pair, 0 for dissimilar.
        """
        if isinstance(predictions, tuple):
            emb_a, emb_b = predictions
            # Euclidean distance
            dist_sq = sum((float(a) - float(b)) ** 2 for a, b in zip(emb_a, emb_b))
            dist = math.sqrt(dist_sq)
        else:
            dist = float(predictions)

        label = int(targets)

        if label == 1:
            # Similar pair: minimize distance
            loss = 0.5 * dist * dist
        else:
            # Dissimilar pair: push apart to at least margin
            loss = 0.5 * max(0.0, self.margin - dist) ** 2

        return loss


# ---------------------------------------------------------------------------
# KL Divergence Loss
# ---------------------------------------------------------------------------
class KLDivergenceLoss(LossFunction):
    """KL Divergence loss: KL(P || Q) = sum(P * log(P / Q))."""

    def __init__(self, reduction: str = "mean", epsilon: float = 1e-12) -> None:
        self.reduction = reduction
        self.epsilon = epsilon

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute KL divergence.

        Args:
            predictions: Logits or probabilities from the model (Q).
            targets: Target probabilities or logits (P).
        """
        # Convert to probabilities if they look like logits
        if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
            # Batch mode
            batch_losses = []
            for pred, tgt in zip(predictions, targets):
                batch_losses.append(self._compute_single(pred, tgt))
            if self.reduction == "sum":
                return sum(batch_losses)
            return _mean(batch_losses)
        else:
            return self._compute_single(predictions, targets)

    def _compute_single(self, predictions: List[float], targets: List[float]) -> float:
        """Compute KL divergence for a single sample."""
        # Normalize targets to sum to 1 (treat as probabilities)
        target_sum = sum(abs(t) for t in targets)
        if target_sum > 0:
            p = [abs(t) / target_sum for t in targets]
        else:
            p = [1.0 / len(targets)] * len(targets)

        # Softmax predictions
        q = _softmax(predictions)

        kl = 0.0
        for pi, qi in zip(p, q):
            qi = max(qi, self.epsilon)
            if pi > self.epsilon:
                kl += pi * math.log(pi / qi)

        return kl


# ---------------------------------------------------------------------------
# Triplet Loss
# ---------------------------------------------------------------------------
class TripletLoss(LossFunction):
    """Triplet loss for metric learning.

    L = max(d(a, p) - d(a, n) + margin, 0)

    where a=anchor, p=positive, n=negative.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean") -> None:
        self.margin = margin
        self.reduction = reduction

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Compute Euclidean distance."""
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))

    def compute(self, predictions: Any, targets: Any = None) -> float:
        """
        Compute triplet loss.

        Args:
            predictions: Tuple of (anchor, positive, negative) embeddings,
                        or list of such tuples.
            targets: Not used, kept for API consistency.
        """
        if isinstance(predictions, tuple) and not isinstance(predictions[0], tuple):
            # Single triplet
            anchor, positive, negative = predictions
            d_ap = self._euclidean_distance(anchor, positive)
            d_an = self._euclidean_distance(anchor, negative)
            loss = max(0.0, d_ap - d_an + self.margin)
            return loss
        else:
            # Batch of triplets
            losses = []
            for triplet in predictions:
                anchor, positive, negative = triplet
                d_ap = self._euclidean_distance(anchor, positive)
                d_an = self._euclidean_distance(anchor, negative)
                losses.append(max(0.0, d_ap - d_an + self.margin))

            if self.reduction == "sum":
                return sum(losses)
            return _mean(losses)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------
class CombinedLoss(LossFunction):
    """Combines multiple loss functions with configurable weights."""

    def __init__(
        self,
        losses: Optional[List[Tuple[LossFunction, float]]] = None,
    ) -> None:
        """
        Args:
            losses: List of (loss_function, weight) tuples.
        """
        self.losses: List[Tuple[LossFunction, float]] = losses or []

    def add_loss(self, loss_fn: LossFunction, weight: float = 1.0) -> None:
        """Add a loss function with its weight."""
        self.losses.append((loss_fn, weight))

    def remove_loss(self, index: int) -> None:
        """Remove a loss function by index."""
        if 0 <= index < len(self.losses):
            self.losses.pop(index)

    def compute(self, predictions: Any, targets: Any) -> float:
        """
        Compute weighted combination of all losses.

        Args:
            predictions: Model predictions.
            targets: Ground truth values.

        Returns:
            Combined weighted loss.
        """
        if not self.losses:
            return 0.0

        total_loss = 0.0
        total_weight = 0.0

        for loss_fn, weight in self.losses:
            if weight == 0.0:
                continue
            try:
                loss_val = loss_fn.compute(predictions, targets)
            except (TypeError, ValueError):
                # Skip loss functions that are incompatible with the current input format
                continue
            total_loss += weight * loss_val
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_loss / total_weight

    def get_individual_losses(self, predictions: Any, targets: Any) -> List[float]:
        """Compute and return each individual loss value."""
        return [loss_fn.compute(predictions, targets) for loss_fn, _ in self.losses]

    def __len__(self) -> int:
        return len(self.losses)

    def __repr__(self) -> str:
        parts = [f"{loss_fn.__class__.__name__}(w={w:.2f})" for loss_fn, w in self.losses]
        return f"CombinedLoss([{', '.join(parts)}])"
