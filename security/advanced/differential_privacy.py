"""
Differential Privacy Module
============================

Implements differential privacy mechanisms (Laplace, Gaussian), gradient
clipping, and privacy budget accountants (RDP, Moments) using only the
Python standard library.
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPConfig:
    """Configuration for differential privacy mechanisms."""

    epsilon: float = 1.0
    """Total privacy budget (epsilon)."""

    delta: float = 1e-5
    """Failure probability delta (for Gaussian mechanism)."""

    max_grad_norm: float = 1.0
    """Maximum L2 norm for gradient clipping."""

    noise_multiplier: float = 1.1
    """Multiplier for the noise scale (sigma = multiplier * clip_norm / epsilon)."""

    mechanism: str = "gaussian"
    """Noise mechanism: 'laplace' or 'gaussian'."""

    secure_rng: bool = True
    """Whether to use ``secrets`` for random number generation."""


# ---------------------------------------------------------------------------
# Differential Privacy Mechanism
# ---------------------------------------------------------------------------

class DifferentialPrivacy:
    """Core differential privacy mechanisms.

    Supports the Laplace and Gaussian mechanisms for adding calibrated
    noise to individual values or gradient vectors, as well as gradient
    clipping and a built-in privacy budget accountant.
    """

    def __init__(self, config: Optional[DPConfig] = None):
        self.config = config or DPConfig()
        self._accountant = _SimpleAccountant(self.config.epsilon)

    # -- noise mechanisms -------------------------------------------------

    def add_laplace_noise(
        self, value: float, sensitivity: float, epsilon: float
    ) -> float:
        """Add Laplace noise calibrated for (epsilon, 0)-DP.

        The Laplace distribution has scale b = sensitivity / epsilon.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")
        if sensitivity == 0:
            return value

        scale = sensitivity / epsilon
        # Laplace sampling via inverse CDF: u ~ Uniform(-0.5, 0.5)
        u = random.uniform(-0.5, 0.5)
        noise = -scale * (1 if u < 0 else -1) * math.log(1 - 2 * abs(u))
        return value + noise

    def add_gaussian_noise(
        self,
        value: float,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> float:
        """Add Gaussian noise calibrated for (epsilon, delta)-DP.

        The standard deviation is
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")
        if sensitivity == 0:
            return value

        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        noise = self._sample_gaussian(0.0, sigma)
        return value + noise

    def add_vector_laplace(
        self, values: List[float], sensitivity: float, epsilon: float
    ) -> List[float]:
        """Add Laplace noise to every element of *values*."""
        return [
            self.add_laplace_noise(v, sensitivity, epsilon) for v in values
        ]

    def add_vector_gaussian(
        self,
        values: List[float],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> List[float]:
        """Add Gaussian noise to every element of *values*."""
        return [
            self.add_gaussian_noise(v, sensitivity, epsilon, delta)
            for v in values
        ]

    # -- gradient clipping ------------------------------------------------

    def clip_gradient(
        self, gradients: List[float], max_norm: Optional[float] = None
    ) -> Tuple[List[float], float]:
        """Clip gradients so that their L2 norm does not exceed *max_norm*.

        Returns (clipped_gradients, original_norm).
        """
        max_norm = max_norm or self.config.max_grad_norm
        norm = math.sqrt(sum(g * g for g in gradients))
        if norm <= max_norm or norm == 0:
            return list(gradients), norm
        factor = max_norm / norm
        return [g * factor for g in gradients], norm

    def clip_and_noise(
        self, gradients: List[float], max_norm: Optional[float] = None
    ) -> List[float]:
        """Clip gradients and add DP noise in one step."""
        clipped, _ = self.clip_gradient(gradients, max_norm)
        if self.config.mechanism == "laplace":
            return self.add_vector_laplace(
                clipped,
                sensitivity=max_norm or self.config.max_grad_norm,
                epsilon=self.config.epsilon,
            )
        else:
            return self.add_vector_gaussian(
                clipped,
                sensitivity=max_norm or self.config.max_grad_norm,
                epsilon=self.config.epsilon,
                delta=self.config.delta,
            )

    # -- privacy budget ---------------------------------------------------

    def privacy_budget_accountant(self) -> "_SimpleAccountant":
        """Return the built-in privacy budget accountant."""
        return self._accountant

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _sample_gaussian(mean: float, std: float) -> float:
        """Sample from a Gaussian distribution using the Box-Muller transform."""
        u1 = random.random()
        u2 = random.random()
        # Avoid log(0)
        while u1 == 0:
            u1 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def compute_noise_scale(
        self, mechanism: Optional[str] = None
    ) -> float:
        """Compute the noise standard deviation for the current config."""
        mech = mechanism or self.config.mechanism
        clip = self.config.max_grad_norm
        if mech == "laplace":
            return clip / self.config.epsilon
        else:
            return clip * math.sqrt(
                2 * math.log(1.25 / self.config.delta)
            ) / self.config.epsilon


# ---------------------------------------------------------------------------
# Simple (Advanced Composition) Accountant
# ---------------------------------------------------------------------------

class _SimpleAccountant:
    """Tracks privacy budget using basic composition and advanced composition.

    * Serial composition: eps_total = sum(eps_i)
    * Parallel composition: eps_total = max(eps_i)
    """

    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon: float = 0.0
        self.spent_delta: float = 0.0
        self.steps: List[Tuple[float, float, str]] = []

    def track_spent(
        self, epsilon: float, delta: float = 0.0, mechanism: str = "generic"
    ) -> None:
        """Record a privacy spend."""
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.steps.append((epsilon, delta, mechanism))

    def get_remaining_budget(self) -> Tuple[float, float]:
        """Return (remaining_epsilon, remaining_delta)."""
        return (
            max(0.0, self.total_epsilon - self.spent_epsilon),
            max(0.0, self.total_delta - self.spent_delta),
        )

    def check_budget(self, epsilon: float, delta: float = 0.0) -> bool:
        """Return True if the requested budget is available."""
        rem_eps, rem_delta = self.get_remaining_budget()
        return epsilon <= rem_eps and delta <= rem_delta

    def get_spent_budget(self) -> Tuple[float, float]:
        """Return (spent_epsilon, spent_delta)."""
        return self.spent_epsilon, self.spent_delta

    def reset(self) -> None:
        """Reset the accountant."""
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.steps.clear()

    def get_history(self) -> List[Tuple[float, float, str]]:
        """Return the full history of privacy spends."""
        return list(self.steps)

    def advanced_composition_epsilon(
        self, k: int, target_delta: float
    ) -> float:
        """Compute epsilon for k-fold adaptive composition.

        Uses the advanced composition theorem:
        eps_total = sqrt(2*k*ln(1/delta)) * eps + k*eps*(e^eps - 1)
        """
        eps_step = self.total_epsilon / max(k, 1)
        term1 = math.sqrt(2 * k * math.log(1 / target_delta)) * eps_step
        term2 = k * eps_step * (math.exp(eps_step) - 1)
        return term1 + term2


# ---------------------------------------------------------------------------
# RDP (Renyi Differential Privacy) Accountant
# ---------------------------------------------------------------------------

class RDPAccountant:
    """Renyi Differential Privacy accountant.

    Tracks the RDP bound at each step and converts to a (epsilon, delta)-DP
    guarantee using the RDP-to-DP conversion.
    """

    def __init__(self, orders: Optional[List[float]] = None):
        self.orders = orders or [
            1 + i / 2.0 for i in range(1, 65)
        ]  # [1.5, 2.0, ..., 32.5]
        self._rdp_values: Dict[float, float] = {o: 0.0 for o in self.orders}
        self._steps: int = 0

    def compute_rdp(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int = 1,
    ) -> Dict[float, float]:
        """Compute the RDP for Poisson-sampled Gaussian mechanism.

        Uses the closed-form RDP bound for the Gaussian mechanism:
        RDP(alpha) = alpha * sampling_rate^2 * q / (2 * sigma^2)
        where q = sampling_rate and sigma = noise_multiplier.

        For each order alpha we compute:
        RDP(alpha) = (alpha * q^2) / (2 * noise_multiplier^2)
        """
        result: Dict[float, float] = {}
        q = sampling_rate
        sigma_sq = noise_multiplier ** 2
        for alpha in self.orders:
            rdp = (alpha * q * q) / (2.0 * sigma_sq) * steps
            result[alpha] = rdp
        return result

    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int = 1,
    ) -> None:
        """Accumulate RDP values for additional training steps."""
        new_rdp = self.compute_rdp(noise_multiplier, sampling_rate, steps)
        for alpha in self.orders:
            self._rdp_values[alpha] += new_rdp[alpha]
        self._steps += steps

    def rdp_to_dp(self, delta: float) -> float:
        """Convert accumulated RDP to (epsilon, delta)-DP.

        epsilon = min over orders alpha of:
            RDP(alpha) + log(1/delta) / (alpha - 1)
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")

        best_epsilon = float("inf")
        for alpha in self.orders:
            rdp_val = self._rdp_values[alpha]
            if alpha <= 1.0:
                continue
            eps = rdp_val + math.log(1.0 / delta) / (alpha - 1.0)
            if eps < best_epsilon:
                best_epsilon = eps
        return best_epsilon

    def get_epsilon(self, delta: float) -> float:
        """Convenience alias for :meth:`rdp_to_dp`."""
        return self.rdp_to_dp(delta)

    def reset(self) -> None:
        """Reset the accountant."""
        self._rdp_values = {o: 0.0 for o in self.orders}
        self._steps = 0

    def get_steps(self) -> int:
        """Return the number of accumulated steps."""
        return self._steps


# ---------------------------------------------------------------------------
# Moments Accountant
# ---------------------------------------------------------------------------

class MomentsAccountant:
    """Moments accountant for tracking privacy budget.

    Tracks the log of the moments of the privacy loss random variable
    and converts them to an (epsilon, delta)-DP guarantee.
    """

    def __init__(self, max_moment: int = 64):
        self.max_moment = max_moment
        self._moments: List[float] = [0.0] * (max_moment + 1)
        self._steps: int = 0

    def log_moment(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int = 1,
    ) -> List[float]:
        """Compute the log moments for the Gaussian mechanism.

        For the Gaussian mechanism with Poisson sampling:
        log E[exp(t * Z)] ~ t^2 * q^2 * sigma^{-2} / 2

        We compute this for t = 1, 2, ..., max_moment.
        """
        q = sampling_rate
        sigma_sq = noise_multiplier ** 2
        moments: List[float] = []
        for t in range(1, self.max_moment + 1):
            log_moment_val = (t * t * q * q) / (2.0 * sigma_sq) * steps
            moments.append(log_moment_val)
        return moments

    def accumulate(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int = 1,
    ) -> None:
        """Accumulate log moments for additional training steps."""
        new_moments = self.log_moment(noise_multiplier, sampling_rate, steps)
        for i, lm in enumerate(new_moments):
            # Use log-sum-exp for stable accumulation
            old = self._moments[i + 1]
            self._moments[i + 1] = (
                old + math.log1p(math.exp(lm - old))
                if old > lm
                else lm + math.log1p(math.exp(old - lm))
            )
        self._steps += steps

    def get_epsilon(self, delta: float) -> float:
        """Convert accumulated moments to epsilon.

        Uses the conversion:
        epsilon = min_t { log_moment(t) - log(delta) / t }
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        log_delta = math.log(delta)
        best_epsilon = float("inf")
        for t in range(1, self.max_moment + 1):
            lm = self._moments[t]
            eps = lm - log_delta / t
            if eps < best_epsilon:
                best_epsilon = eps
        return best_epsilon

    def reset(self) -> None:
        """Reset the accountant."""
        self._moments = [0.0] * (self.max_moment + 1)
        self._steps = 0

    def get_steps(self) -> int:
        """Return the number of accumulated steps."""
        return self._steps


# ---------------------------------------------------------------------------
# Privacy Accountant Factory
# ---------------------------------------------------------------------------

class PrivacyAccountant:
    """Factory for creating privacy accountants."""

    @staticmethod
    def create(
        method: str = "rdp", **kwargs
    ) -> object:
        """Create a privacy accountant.

        Args:
            method: One of 'rdp', 'moments', 'simple'.
            **kwargs: Additional arguments passed to the accountant constructor.

        Returns:
            An accountant instance.
        """
        method = method.lower()
        if method == "rdp":
            return RDPAccountant(**kwargs)
        elif method == "moments":
            return MomentsAccountant(**kwargs)
        elif method == "simple":
            return _SimpleAccountant(**kwargs)
        else:
            raise ValueError(f"Unknown accountant method: {method}")
