"""
Training Framework - Trainer Module

Provides the core training loop, configuration, state management,
callbacks, early stopping, gradient accumulation, and checkpoint management.
"""

import copy
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    """Training configuration with all hyperparameters and settings."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adam"
    scheduler: str = "cosine"
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    checkpoint_interval: int = 1
    log_interval: int = 10
    warmup_steps: int = 0
    weight_decay: float = 0.0
    seed: int = 42
    output_dir: str = "./checkpoints"
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "early_stopping_patience": self.early_stopping_patience,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """Create configuration from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------
@dataclass
class TrainingState:
    """Tracks the current state of training progress."""

    epoch: int = 0
    step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    learning_rate: float = 0.0
    total_steps: int = 0
    start_time: float = 0.0
    is_best: bool = False

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value at the current step."""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        self.metrics_history[name].append(value)

    def update_best(self, metric: float, mode: str = "min") -> bool:
        """Update best metric. Returns True if this is a new best."""
        improved = False
        if mode == "min":
            if metric < self.best_metric:
                improved = True
                self.best_metric = metric
        else:
            if metric > self.best_metric:
                improved = True
                self.best_metric = metric
        if improved:
            self.best_epoch = self.epoch
            self.is_best = True
        else:
            self.is_best = False
        return improved

    def get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metrics_history": self.metrics_history,
            "learning_rate": self.learning_rate,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "TrainingState":
        """Create state from dictionary."""
        metrics_history = state_dict.get("metrics_history", {})
        return cls(
            epoch=state_dict.get("epoch", 0),
            step=state_dict.get("step", 0),
            best_metric=state_dict.get("best_metric", float("inf")),
            best_epoch=state_dict.get("best_epoch", 0),
            metrics_history=metrics_history,
            learning_rate=state_dict.get("learning_rate", 0.0),
            total_steps=state_dict.get("total_steps", 0),
        )


# ---------------------------------------------------------------------------
# Training Callback
# ---------------------------------------------------------------------------
class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""

    def on_train_begin(self, config: TrainerConfig) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        ...

    @abstractmethod
    def on_epoch_end(self, epoch: int, state: TrainingState, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_begin(self, batch_idx: int, batch: Any) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, metrics: Dict[str, float]) -> None:
        """Called at the end of each batch."""
        pass

    @abstractmethod
    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        """Called after validation completes."""
        ...


class LoggingCallback(TrainingCallback):
    """Callback that logs training progress."""

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval

    def on_epoch_begin(self, epoch: int, state: TrainingState) -> None:
        logger.info(f"Epoch {epoch + 1}/{state.total_steps} starting")

    def on_epoch_end(self, epoch: int, state: TrainingState, metrics: Dict[str, float]) -> None:
        elapsed = state.get_elapsed_time()
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(
            f"Epoch {epoch + 1} completed | {metric_str} | "
            f"LR: {state.learning_rate:.6f} | Time: {elapsed:.1f}s"
        )

    def on_batch_end(self, batch_idx: int, loss: float, metrics: Dict[str, float]) -> None:
        if (batch_idx + 1) % self.log_interval == 0:
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"  Batch {batch_idx + 1} | Loss: {loss:.4f} | {metric_str}")

    def on_validation_end(self, metrics: Dict[str, float]) -> None:
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Validation | {metric_str}")


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Monitors a metric and stops training when no improvement is observed."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ) -> None:
        """
        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' for metrics like loss, 'max' for metrics like accuracy.
            restore_best: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_state: Optional[Dict[str, Any]] = None

    def check(self, metric: float, current_state: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current metric value.
            current_state: Optional model state to save if best.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            if current_state is not None:
                self.best_state = copy.deepcopy(current_state)
            return False

        improved = False
        if self.mode == "min":
            if metric < self.best_score - self.min_delta:
                improved = True
        else:
            if metric > self.best_score + self.min_delta:
                improved = True

        if improved:
            self.best_score = metric
            self.counter = 0
            if current_state is not None:
                self.best_state = copy.deepcopy(current_state)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement (best: {self.best_score:.4f})"
                )
                return True

        return False

    def get_best_state(self) -> Optional[Dict[str, Any]]:
        """Return the best saved state."""
        return self.best_state

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_state = None


# ---------------------------------------------------------------------------
# Gradient Accumulator
# ---------------------------------------------------------------------------
class GradientAccumulator:
    """Accumulates gradients over multiple steps before applying an update."""

    def __init__(self, accumulation_steps: int = 1) -> None:
        """
        Args:
            accumulation_steps: Number of steps to accumulate before updating.
        """
        self.accumulation_steps = accumulation_steps
        self._buffer: List[Dict[str, Any]] = []
        self._step_count = 0

    def accumulate(self, gradients: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Accumulate gradients. Returns averaged gradients when buffer is full.

        Args:
            gradients: Dictionary mapping parameter names to gradient values.

        Returns:
            Averaged gradients if buffer is full, None otherwise.
        """
        self._buffer.append(gradients)
        self._step_count += 1

        if self._step_count >= self.accumulation_steps:
            return self.get_average()
        return None

    def get_average(self) -> Dict[str, Any]:
        """Compute the average of all accumulated gradients."""
        if not self._buffer:
            return {}

        averaged: Dict[str, Any] = {}
        all_keys = set()
        for grad in self._buffer:
            all_keys.update(grad.keys())

        for key in all_keys:
            values = [g[key] for g in self._buffer if key in g]
            if not values:
                continue
            if isinstance(values[0], (int, float)):
                averaged[key] = sum(values) / len(values)
            elif isinstance(values[0], list):
                length = len(values[0])
                averaged[key] = [
                    sum(v[i] for v in values) / len(values) for i in range(length)
                ]
            elif isinstance(values[0], dict):
                # Nested dict averaging
                sub_keys = set()
                for v in values:
                    sub_keys.update(v.keys())
                averaged[key] = {}
                for sk in sub_keys:
                    sub_vals = [v[sk] for v in values if sk in v]
                    if sub_vals and isinstance(sub_vals[0], (int, float)):
                        averaged[key][sk] = sum(sub_vals) / len(sub_vals)

        self.reset()
        return averaged

    def reset(self) -> None:
        """Reset the accumulation buffer."""
        self._buffer = []
        self._step_count = 0

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has accumulated enough steps."""
        return self._step_count >= self.accumulation_steps

    @property
    def step_count(self) -> int:
        """Current accumulation step count."""
        return self._step_count


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Manages saving, loading, and cleanup of training checkpoints."""

    def __init__(
        self,
        output_dir: str = "./checkpoints",
        max_keep: int = 5,
        prefix: str = "checkpoint",
    ) -> None:
        """
        Args:
            output_dir: Directory to save checkpoints.
            max_keep: Maximum number of checkpoints to keep.
            prefix: Filename prefix for checkpoints.
        """
        self.output_dir = output_dir
        self.max_keep = max_keep
        self.prefix = prefix
        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        state: TrainingState,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            state: Training state.
            model_state: Model parameters/state dict.
            optimizer_state: Optimizer state.
            extra: Additional data to save.

        Returns:
            Path to the saved checkpoint.
        """
        filename = f"{self.prefix}_epoch{state.epoch:04d}_step{state.step:06d}.json"
        filepath = os.path.join(self.output_dir, filename)

        checkpoint = {
            "training_state": state.to_dict(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "timestamp": time.time(),
        }
        if extra:
            checkpoint["extra"] = extra

        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {filepath}")
        self._cleanup_old_checkpoints()
        return filepath

    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            filepath: Path to the checkpoint file.

        Returns:
            Dictionary with training_state, model_state, optimizer_state, extra.
        """
        with open(filepath, "r") as f:
            checkpoint = json.load(f)

        state = TrainingState.from_dict(checkpoint.get("training_state", {}))
        logger.info(
            f"Checkpoint loaded: {filepath} "
            f"(epoch={state.epoch}, step={state.step})"
        )

        return {
            "training_state": state,
            "model_state": checkpoint.get("model_state", {}),
            "optimizer_state": checkpoint.get("optimizer_state"),
            "extra": checkpoint.get("extra"),
            "timestamp": checkpoint.get("timestamp"),
        }

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return self.load(checkpoints[-1])

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint files sorted by modification time."""
        if not os.path.exists(self.output_dir):
            return []
        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.startswith(self.prefix) and f.endswith(".json")
        ]
        files.sort(key=lambda x: os.path.getmtime(x))
        return files

    def keep_last_n(self, n: Optional[int] = None) -> int:
        """
        Keep only the last n checkpoints, deleting older ones.

        Args:
            n: Number to keep. Uses max_keep if not specified.

        Returns:
            Number of checkpoints deleted.
        """
        if n is None:
            n = self.max_keep
        checkpoints = self.list_checkpoints()
        to_delete = checkpoints[: max(0, len(checkpoints) - n)]
        for path in to_delete:
            os.remove(path)
            logger.debug(f"Deleted old checkpoint: {path}")
        return len(to_delete)

    def _cleanup_old_checkpoints(self) -> None:
        """Automatically clean up old checkpoints beyond max_keep."""
        self.keep_last_n(self.max_keep)

    def get_best_checkpoint(self, mode: str = "min") -> Optional[str]:
        """
        Find the checkpoint with the best metric.

        Args:
            mode: 'min' or 'max' to determine best.

        Returns:
            Path to the best checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best_path = None
        best_metric = float("inf") if mode == "min" else float("-inf")

        for path in checkpoints:
            try:
                data = self.load(path)
                metric = data["training_state"].best_metric
                if mode == "min" and metric < best_metric:
                    best_metric = metric
                    best_path = path
                elif mode == "max" and metric > best_metric:
                    best_metric = metric
                    best_path = path
            except (json.JSONDecodeError, KeyError):
                continue

        return best_path


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------
class BaseTrainer(ABC):
    """Abstract base class for all trainers."""

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ) -> None:
        """
        Args:
            config: Training configuration.
            callbacks: List of training callbacks.
        """
        self.config = config or TrainerConfig()
        self.state = TrainingState(learning_rate=self.config.learning_rate)
        self.callbacks = callbacks or []
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience
        )
        self.gradient_accumulator = GradientAccumulator(
            accumulation_steps=self.config.gradient_accumulation_steps
        )
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.config.output_dir
        )
        self._is_training = False

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for a single epoch. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Run validation. Must be implemented by subclasses."""
        ...

    def train(
        self,
        train_data: Any = None,
        val_data: Any = None,
        num_epochs: Optional[int] = None,
    ) -> TrainingState:
        """
        Main training loop.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            num_epochs: Override number of epochs.

        Returns:
            Final training state.
        """
        epochs = num_epochs or self.config.epochs
        self.state.start_time = time.time()
        self.state.total_steps = epochs
        self._is_training = True

        for callback in self.callbacks:
            callback.on_train_begin(self.config)

        for epoch in range(self.state.epoch, epochs):
            self.state.epoch = epoch

            # Epoch begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self.state)

            # Train one epoch
            train_metrics = self.train_epoch(epoch)

            # Record training metrics
            for name, value in train_metrics.items():
                self.state.record_metric(f"train_{name}", value)

            # Validation
            val_metrics = {}
            if val_data is not None:
                val_metrics = self.validate()
                for name, value in val_metrics.items():
                    self.state.record_metric(f"val_{name}", value)
                for callback in self.callbacks:
                    callback.on_validation_end(val_metrics)

            # Check early stopping
            monitor_metric = val_metrics.get("loss", train_metrics.get("loss", 0.0))
            if self.early_stopping.check(monitor_metric):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Epoch end callbacks
            all_metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.state, all_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

        self._is_training = False
        for callback in self.callbacks:
            callback.on_train_end(self.state)

        return self.state

    def save_checkpoint(self) -> str:
        """Save current training checkpoint."""
        model_state = self._get_model_state()
        return self.checkpoint_manager.save(
            state=self.state,
            model_state=model_state,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a training checkpoint.

        Args:
            filepath: Path to checkpoint file.
        """
        data = self.checkpoint_manager.load(filepath)
        self.state = data["training_state"]
        self._set_model_state(data["model_state"])

    def get_training_state(self) -> TrainingState:
        """Return a copy of the current training state."""
        return copy.deepcopy(self.state)

    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get serializable model state. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model from state. Must be implemented by subclasses."""
        ...

    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a training callback."""
        self.callbacks.append(callback)

    def set_early_stopping(
        self, patience: int, min_delta: float = 0.0, mode: str = "min"
    ) -> None:
        """Configure early stopping."""
        self.early_stopping = EarlyStopping(
            patience=patience, min_delta=min_delta, mode=mode
        )

    def resume_training(self) -> bool:
        """Try to resume from the latest checkpoint. Returns True if successful."""
        data = self.checkpoint_manager.load_latest()
        if data is not None:
            self.state = data["training_state"]
            self._set_model_state(data["model_state"])
            logger.info(f"Resumed training from epoch {self.state.epoch}")
            return True
        return False
