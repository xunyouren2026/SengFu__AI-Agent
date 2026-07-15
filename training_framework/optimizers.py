"""
Training Framework - Optimizers Module

Provides various optimization algorithms and learning rate schedulers
implemented from scratch using only the Python standard library.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Helper: deep copy of nested parameter dicts
# ---------------------------------------------------------------------------
def _deep_add(a: Any, b: Any) -> Any:
    """Add two values, lists, or nested dicts element-wise."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    if isinstance(a, list) and isinstance(b, list):
        return [_deep_add(x, y) for x, y in zip(a, b)]
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        return {k: _deep_add(a.get(k, 0), b.get(k, 0)) for k in keys}
    return a


def _deep_scale(a: Any, scalar: float) -> Any:
    """Scale a value, list, or nested dict by a scalar."""
    if isinstance(a, (int, float)):
        return a * scalar
    if isinstance(a, list):
        return [_deep_scale(x, scalar) for x in a]
    if isinstance(a, dict):
        return {k: _deep_scale(v, scalar) for k, v in a.items()}
    return a


def _deep_zeros_like(a: Any) -> Any:
    """Create a zero structure matching the shape of a."""
    if isinstance(a, (int, float)):
        return 0.0
    if isinstance(a, list):
        return [_deep_zeros_like(x) for x in a]
    if isinstance(a, dict):
        return {k: _deep_zeros_like(v) for k, v in a.items()}
    return 0.0


def _deep_clamp(a: Any, min_val: float, max_val: float) -> Any:
    """Clamp values between min_val and max_val."""
    if isinstance(a, (int, float)):
        return max(min_val, min(max_val, a))
    if isinstance(a, list):
        return [_deep_clamp(x, min_val, max_val) for x in a]
    if isinstance(a, dict):
        return {k: _deep_clamp(v, min_val, max_val) for k, v in a.items()}
    return a


# ---------------------------------------------------------------------------
# Optimizer Base
# ---------------------------------------------------------------------------
class Optimizer(ABC):
    """Abstract base class for all optimizers."""

    def __init__(self, lr: float = 1e-3) -> None:
        self.lr = lr
        self.step_count = 0
        self._grad_buffer: Dict[str, Any] = {}

    @abstractmethod
    def step(self, gradients: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single optimization step.

        Args:
            gradients: Dictionary mapping parameter names to gradients.
            params: Dictionary mapping parameter names to current values.

        Returns:
            Updated parameters.
        """
        ...

    def zero_grad(self) -> None:
        """Clear the gradient buffer."""
        self._grad_buffer = {}

    def accumulate_grad(self, gradients: Dict[str, Any]) -> None:
        """Accumulate gradients into the buffer."""
        for key, value in gradients.items():
            if key in self._grad_buffer:
                self._grad_buffer[key] = _deep_add(self._grad_buffer[key], value)
            else:
                self._grad_buffer[key] = value

    def get_accumulated_grad(self) -> Dict[str, Any]:
        """Get accumulated gradients."""
        return dict(self._grad_buffer)

    @property
    def param_count(self) -> int:
        """Return the number of parameters being tracked."""
        return len(self._grad_buffer)


# ---------------------------------------------------------------------------
# SGD Optimizer
# ---------------------------------------------------------------------------
class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent with optional momentum, Nesterov, and weight decay."""

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr=lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._velocity: Dict[str, Any] = {}

    def step(self, gradients: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform SGD update with momentum and weight decay."""
        updated = {}
        for key in params:
            grad = gradients.get(key, 0.0)

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0.0:
                grad = _deep_add(grad, _deep_scale(params[key], self.weight_decay))

            # Momentum update
            if self.momentum != 0.0:
                if key not in self._velocity:
                    self._velocity[key] = _deep_zeros_like(params[key])
                self._velocity[key] = _deep_add(
                    _deep_scale(self._velocity[key], self.momentum), grad
                )
                if self.nesterov:
                    update = _deep_add(
                        grad, _deep_scale(self._velocity[key], self.momentum)
                    )
                else:
                    update = self._velocity[key]
            else:
                update = grad

            updated[key] = _deep_add(params[key], _deep_scale(update, -self.lr))

        self.step_count += 1
        return updated


# ---------------------------------------------------------------------------
# Adam Optimizer
# ---------------------------------------------------------------------------
class AdamOptimizer(Optimizer):
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m: Dict[str, Any] = {}  # First moment estimates
        self._v: Dict[str, Any] = {}  # Second moment estimates

    def step(self, gradients: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Adam update with bias-corrected moment estimates."""
        updated = {}
        t = self.step_count + 1

        for key in params:
            grad = gradients.get(key, 0.0)

            # Apply weight decay
            if self.weight_decay != 0.0:
                grad = _deep_add(grad, _deep_scale(params[key], self.weight_decay))

            # Initialize moment estimates
            if key not in self._m:
                self._m[key] = _deep_zeros_like(params[key])
                self._v[key] = _deep_zeros_like(params[key])

            # Update biased first moment estimate
            self._m[key] = _deep_add(
                _deep_scale(self._m[key], self.beta1),
                _deep_scale(grad, 1.0 - self.beta1),
            )

            # Update biased second raw moment estimate
            self._v[key] = _deep_add(
                _deep_scale(self._v[key], self.beta2),
                _deep_scale(_deep_scale(grad, grad), 1.0 - self.beta2),
            )

            # Compute bias-corrected first moment estimate
            bias_correction1 = 1.0 - self.beta1 ** t
            m_hat = _deep_scale(self._m[key], 1.0 / bias_correction1)

            # Compute bias-corrected second raw moment estimate
            bias_correction2 = 1.0 - self.beta2 ** t
            v_hat = _deep_scale(self._v[key], 1.0 / bias_correction2)

            # Update parameters: param -= lr * m_hat / (sqrt(v_hat) + epsilon)
            # For scalar: param - lr * m / (sqrt(v) + eps)
            if isinstance(m_hat, (int, float)):
                denom = math.sqrt(abs(v_hat)) + self.epsilon
                update = m_hat / denom
            elif isinstance(m_hat, list):
                update = []
                for mh, vh in zip(m_hat, v_hat):
                    if isinstance(mh, (int, float)):
                        denom = math.sqrt(abs(vh)) + self.epsilon
                        update.append(mh / denom)
                    else:
                        update.append(mh)
            elif isinstance(m_hat, dict):
                update = {}
                for mk, mv in m_hat.items():
                    vv = v_hat.get(mk, 0.0)
                    if isinstance(mv, (int, float)):
                        denom = math.sqrt(abs(vv)) + self.epsilon
                        update[mk] = mv / denom
                    else:
                        update[mk] = mv
            else:
                update = m_hat

            updated[key] = _deep_add(params[key], _deep_scale(update, -self.lr))

        self.step_count += 1
        return updated


# ---------------------------------------------------------------------------
# AdamW Optimizer (Decoupled Weight Decay)
# ---------------------------------------------------------------------------
class AdamWOptimizer(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Unlike Adam with L2 regularization, AdamW decouples the weight decay
    from the gradient-based update, applying it directly to the parameters.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self._m: Dict[str, Any] = {}
        self._v: Dict[str, Any] = {}

    def step(self, gradients: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AdamW update with decoupled weight decay."""
        updated = {}
        t = self.step_count + 1
        wd = self.weight_decay * self.lr  # Decoupled: scale by lr

        for key in params:
            grad = gradients.get(key, 0.0)

            # Initialize moment estimates
            if key not in self._m:
                self._m[key] = _deep_zeros_like(params[key])
                self._v[key] = _deep_zeros_like(params[key])

            # Update biased first moment estimate
            self._m[key] = _deep_add(
                _deep_scale(self._m[key], self.beta1),
                _deep_scale(grad, 1.0 - self.beta1),
            )

            # Update biased second raw moment estimate
            self._v[key] = _deep_add(
                _deep_scale(self._v[key], self.beta2),
                _deep_scale(_deep_scale(grad, grad), 1.0 - self.beta2),
            )

            # Bias correction
            bias_correction1 = 1.0 - self.beta1 ** t
            bias_correction2 = 1.0 - self.beta2 ** t
            m_hat = _deep_scale(self._m[key], 1.0 / bias_correction1)
            v_hat = _deep_scale(self._v[key], 1.0 / bias_correction2)

            # Adam update
            if isinstance(m_hat, (int, float)):
                denom = math.sqrt(abs(v_hat)) + self.epsilon
                adam_update = m_hat / denom
            elif isinstance(m_hat, list):
                adam_update = []
                for mh, vh in zip(m_hat, v_hat):
                    if isinstance(mh, (int, float)):
                        denom = math.sqrt(abs(vh)) + self.epsilon
                        adam_update.append(mh / denom)
                    else:
                        adam_update.append(mh)
            elif isinstance(m_hat, dict):
                adam_update = {}
                for mk, mv in m_hat.items():
                    vv = v_hat.get(mk, 0.0)
                    if isinstance(mv, (int, float)):
                        denom = math.sqrt(abs(vv)) + self.epsilon
                        adam_update[mk] = mv / denom
                    else:
                        adam_update[mk] = mv
            else:
                adam_update = m_hat

            # Decoupled weight decay applied directly to parameters
            param_wd = _deep_scale(params[key], -wd)
            adam_step = _deep_scale(adam_update, -self.lr)

            updated[key] = _deep_add(_deep_add(params[key], adam_step), param_wd)

        self.step_count += 1
        return updated


# ---------------------------------------------------------------------------
# Learning Rate Scheduler Base
# ---------------------------------------------------------------------------
class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers."""

    def __init__(self, base_lr: float) -> None:
        self.base_lr = base_lr
        self.step_count = 0

    @abstractmethod
    def get_lr(self, step: Optional[int] = None) -> float:
        """Get the learning rate for the given step."""
        ...

    def step(self) -> float:
        """Advance the scheduler by one step and return the new learning rate."""
        self.step_count += 1
        return self.get_lr(self.step_count)


# ---------------------------------------------------------------------------
# Cosine Annealing LR
# ---------------------------------------------------------------------------
class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(self, base_lr: float, T_max: int, eta_min: float = 0.0) -> None:
        """
        Args:
            base_lr: Initial learning rate.
            T_max: Maximum number of steps/epochs.
            eta_min: Minimum learning rate.
        """
        super().__init__(base_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute cosine annealed learning rate."""
        t = step if step is not None else self.step_count
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * t / self.T_max)
        )


# ---------------------------------------------------------------------------
# Linear Warmup + Cosine Decay LR
# ---------------------------------------------------------------------------
class LinearWarmupLR(LRScheduler):
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
    ) -> None:
        """
        Args:
            base_lr: Peak learning rate after warmup.
            warmup_steps: Number of linear warmup steps.
            total_steps: Total number of training steps.
            eta_min: Minimum learning rate after decay.
        """
        super().__init__(base_lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        t = step if step is not None else self.step_count
        if t < self.warmup_steps:
            # Linear warmup
            return self.base_lr * t / max(1, self.warmup_steps)
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            progress = (t - self.warmup_steps) / max(1, decay_steps)
            progress = min(1.0, progress)
            return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1.0 + math.cos(math.pi * progress)
            )


# ---------------------------------------------------------------------------
# Step LR
# ---------------------------------------------------------------------------
class StepLR(LRScheduler):
    """Step learning rate scheduler: decay by gamma every step_size steps."""

    def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1) -> None:
        """
        Args:
            base_lr: Initial learning rate.
            step_size: Number of steps between decays.
            gamma: Multiplicative factor of decay.
        """
        super().__init__(base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute stepped learning rate."""
        t = step if step is not None else self.step_count
        return self.base_lr * (self.gamma ** (t // self.step_size))


# ---------------------------------------------------------------------------
# Exponential LR
# ---------------------------------------------------------------------------
class ExponentialLR(LRScheduler):
    """Exponential learning rate decay."""

    def __init__(self, base_lr: float, gamma: float = 0.95) -> None:
        """
        Args:
            base_lr: Initial learning rate.
            gamma: Multiplicative decay factor per step.
        """
        super().__init__(base_lr)
        self.gamma = gamma

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute exponentially decayed learning rate."""
        t = step if step is not None else self.step_count
        return self.base_lr * (self.gamma ** t)


# ---------------------------------------------------------------------------
# OneCycle LR
# ---------------------------------------------------------------------------
class OneCycleLR(LRScheduler):
    """1Cycle learning rate policy.

    Implements the 1Cycle policy from "Super-Convergence" (Smith & Topin, 2019):
    1. Linear warmup from initial_lr to max_lr over warmup_steps
    2. Linear (or cosine) annealing from max_lr to final_lr over remaining steps
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        total_steps: int,
        warmup_fraction: float = 0.3,
        final_lr: float = 1e-6,
        anneal_strategy: str = "cosine",
    ) -> None:
        """
        Args:
            base_lr: Initial learning rate.
            max_lr: Maximum learning rate.
            total_steps: Total number of training steps.
            warmup_fraction: Fraction of steps for warmup phase.
            final_lr: Final learning rate at the end.
            anneal_strategy: 'cosine' or 'linear' for the annealing phase.
        """
        super().__init__(base_lr)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.final_lr = final_lr
        self.anneal_strategy = anneal_strategy

    def get_lr(self, step: Optional[int] = None) -> float:
        """Compute 1Cycle learning rate."""
        t = step if step is not None else self.step_count

        if t < self.warmup_steps:
            # Phase 1: Linear warmup from base_lr to max_lr
            if self.warmup_steps == 0:
                return self.max_lr
            return self.base_lr + (self.max_lr - self.base_lr) * t / self.warmup_steps
        else:
            # Phase 2: Anneal from max_lr to final_lr
            remaining = self.total_steps - self.warmup_steps
            if remaining <= 0:
                return self.final_lr
            progress = (t - self.warmup_steps) / remaining
            progress = min(1.0, progress)

            if self.anneal_strategy == "cosine":
                return self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            else:
                # Linear annealing
                return self.max_lr - (self.max_lr - self.final_lr) * progress
