"""
Long Task Progress Publisher Module

This module provides a comprehensive progress tracking system for long-running tasks.
It supports percentage-based progress, ETA calculation, step-by-step tracking, and
cancellation support.

Key Components:
- TaskProgressPublisher: Main class for publishing progress events
- ProgressEvent: Event data for progress updates
- StepTracker: Track progress through discrete steps
- ETACalculator: Calculate estimated time to completion
- CancellationToken: Token for task cancellation support
- ProgressAggregator: Aggregate multiple progress sources

Author: AGI Unified Framework Team
Version: 1.0.0
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from uuid import UUID, uuid4


class ProgressState(Enum):
    """States of a task in progress."""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class ProgressEvent:
    """
    Event data for progress updates.
    
    Contains all information about a progress update including
    percentage, current step, ETA, and additional context.
    """
    event_id: UUID = field(default_factory=uuid4)
    task_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    percentage: float = 0.0
    current_step: int = 0
    total_steps: int = 1
    step_name: str = ""
    step_description: str = ""
    message: str = ""
    eta_seconds: Optional[float] = None
    elapsed_seconds: float = 0.0
    state: ProgressState = ProgressState.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self.percentage = max(0.0, min(100.0, self.percentage))
        self.current_step = max(0, self.current_step)
        self.total_steps = max(1, self.total_steps)
    
    @property
    def is_active(self) -> bool:
        """Check if the task is in an active state."""
        return self.state in {ProgressState.RUNNING, ProgressState.PAUSED}
    
    @property
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        return self.state in {ProgressState.COMPLETED, ProgressState.CANCELLED, ProgressState.FAILED}
    
    @property
    def progress_ratio(self) -> float:
        """Get progress as a ratio (0.0 to 1.0)."""
        return self.percentage / 100.0


@dataclass
class StepDefinition:
    """Definition of a single step in a multi-step task."""
    name: str
    description: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CancellationToken:
    """
    Token for checking and requesting cancellation of a task.
    
    This class provides thread-safe cancellation support.
    """
    
    def __init__(self) -> None:
        self._cancelled = False
        self._cancellation_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[], None]] = []
    
    @property
    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested."""
        with self._lock:
            return self._cancelled
    
    @property
    def cancellation_reason(self) -> Optional[str]:
        """Get the reason for cancellation."""
        with self._lock:
            return self._cancellation_reason
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """
        Request cancellation.
        
        Args:
            reason: Optional reason for cancellation
        """
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                self._cancellation_reason = reason
                self._notify_callbacks()
    
    def throw_if_cancelled(self) -> None:
        """Raise CancellationError if cancellation has been requested."""
        if self.is_cancellation_requested:
            raise CancellationError(self._cancellation_reason or "Task was cancelled")
    
    def register_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called on cancellation.
        
        Args:
            callback: Callback function
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[], None]) -> None:
        """
        Unregister a callback.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception:
                pass
    
    def reset(self) -> None:
        """Reset the cancellation state."""
        with self._lock:
            self._cancelled = False
            self._cancellation_reason = None


class ETACalculator:
    """
    Calculator for estimating time to completion.
    
    Uses historical data to estimate how long a task will take
    to complete based on current progress.
    """
    
    def __init__(self, window_size: int = 10) -> None:
        """
        Initialize the ETA calculator.
        
        Args:
            window_size: Number of recent data points to use for calculation
        """
        self._window_size = window_size
        self._progress_history: List[Tuple[float, float]] = []
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
        self._last_percentage: float = 0.0
    
    def start(self) -> None:
        """Start the ETA calculation timer."""
        with self._lock:
            self._start_time = time.monotonic()
            self._progress_history.clear()
            self._last_percentage = 0.0
    
    def update(self, percentage: float) -> Optional[float]:
        """
        Update the calculator with current progress.
        
        Args:
            percentage: Current progress percentage (0.0 to 100.0)
            
        Returns:
            Estimated seconds to completion, or None if cannot calculate
        """
        with self._lock:
            current_time = time.monotonic()
            
            if self._start_time is None:
                self._start_time = current_time
                self._last_percentage = percentage
                return None
            
            elapsed = current_time - self._start_time
            
            self._progress_history.append((percentage, elapsed))
            
            if len(self._progress_history) > self._window_size:
                self._progress_history.pop(0)
            
            self._last_percentage = percentage
            
            if len(self._progress_history) < 2:
                return None
            
            first_progress, first_time = self._progress_history[0]
            last_progress, last_time = self._progress_history[-1]
            
            progress_delta = last_progress - first_progress
            time_delta = last_time - first_time
            
            if progress_delta <= 0 or time_delta <= 0:
                return None
            
            rate = progress_delta / time_delta
            
            if rate <= 0:
                return None
            
            remaining_progress = 100.0 - percentage
            eta_seconds = remaining_progress / rate
            
            return eta_seconds
    
    def get_eta(self) -> Optional[float]:
        """
        Get the current ETA based on accumulated data.
        
        Returns:
            Estimated seconds to completion, or None
        """
        with self._lock:
            if self._start_time is None or len(self._progress_history) < 2:
                return None
            
            current_time = time.monotonic()
            last_progress, _ = self._progress_history[-1]
            
            total_time = current_time - self._start_time
            total_progress = last_progress
            
            if total_progress <= 0 or total_time <= 0:
                return None
            
            rate = total_progress / total_time
            
            if rate <= 0:
                return None
            
            remaining_progress = 100.0 - total_progress
            return remaining_progress / rate
    
    def reset(self) -> None:
        """Reset the calculator."""
        with self._lock:
            self._start_time = None
            self._progress_history.clear()
            self._last_percentage = 0.0
    
    def get_average_rate(self) -> Optional[float]:
        """
        Get the average progress rate in percent per second.
        
        Returns:
            Average rate or None
        """
        with self._lock:
            if len(self._progress_history) < 2:
                return None
            
            first_progress, first_time = self._progress_history[0]
            last_progress, last_time = self._progress_history[-1]
            
            progress_delta = last_progress - first_progress
            time_delta = last_time - first_time
            
            if time_delta <= 0:
                return None
            
            return progress_delta / time_delta


class StepTracker:
    """
    Track progress through discrete steps.
    
    This class manages a series of steps and calculates overall
    progress based on step completion.
    """
    
    def __init__(self, steps: Optional[List[StepDefinition]] = None) -> None:
        """
        Initialize the step tracker.
        
        Args:
            steps: Optional list of step definitions
        """
        self._steps: List[StepDefinition] = steps or []
        self._current_step_index: int = -1
        self._step_start_times: Dict[int, float] = {}
        self._step_completion_times: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._total_weight: float = sum(s.weight for s in self._steps) or 1.0
    
    @property
    def total_steps(self) -> int:
        """Get the total number of steps."""
        return len(self._steps)
    
    @property
    def current_step(self) -> Optional[StepDefinition]:
        """Get the current step definition."""
        with self._lock:
            if 0 <= self._current_step_index < len(self._steps):
                return self._steps[self._current_step_index]
            return None
    
    @property
    def current_step_index(self) -> int:
        """Get the current step index (0-based)."""
        return self._current_step_index
    
    @property
    def completed_steps(self) -> int:
        """Get the number of completed steps."""
        return self._current_step_index
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        with self._lock:
            if not self._steps:
                return 0.0
            
            completed_weight = 0.0
            for i in range(self._current_step_index):
                completed_weight += self._steps[i].weight
            
            if self._current_step_index >= 0 and self._current_step_index < len(self._steps):
                current_weight = self._steps[self._current_step_index].weight
                estimated_completed = current_weight * 0.5
                completed_weight += estimated_completed
            
            return (completed_weight / self._total_weight) * 100.0
    
    def add_step(self, step: StepDefinition) -> None:
        """
        Add a step to the tracker.
        
        Args:
            step: Step definition to add
        """
        with self._lock:
            self._steps.append(step)
            self._total_weight = sum(s.weight for s in self._steps)
    
    def start_step(self, step_index: int) -> StepDefinition:
        """
        Start a specific step.
        
        Args:
            step_index: Index of the step to start
            
        Returns:
            The step definition
            
        Raises:
            IndexError: If step index is invalid
        """
        with self._lock:
            if step_index < 0 or step_index >= len(self._steps):
                raise IndexError(f"Invalid step index: {step_index}")
            
            self._current_step_index = step_index
            self._step_start_times[step_index] = time.monotonic()
            return self._steps[step_index]
    
    def complete_step(self, step_index: Optional[int] = None) -> None:
        """
        Mark a step as completed.
        
        Args:
            step_index: Index of the step to complete. If None, completes current.
        """
        with self._lock:
            if step_index is None:
                step_index = self._current_step_index
            
            if step_index < 0 or step_index >= len(self._steps):
                raise IndexError(f"Invalid step index: {step_index}")
            
            self._step_completion_times[step_index] = time.monotonic()
    
    def next_step(self) -> Optional[StepDefinition]:
        """
        Move to the next step.
        
        Returns:
            The next step definition, or None if no more steps
        """
        with self._lock:
            if self._current_step_index >= 0:
                self.complete_step(self._current_step_index)
            
            next_index = self._current_step_index + 1
            if next_index >= len(self._steps):
                return None
            
            return self.start_step(next_index)
    
    def get_step_duration(self, step_index: int) -> Optional[float]:
        """
        Get the duration of a completed step.
        
        Args:
            step_index: Index of the step
            
        Returns:
            Duration in seconds, or None if step not completed
        """
        with self._lock:
            if step_index not in self._step_completion_times:
                return None
            
            start_time = self._step_start_times.get(step_index)
            if start_time is None:
                return None
            
            return self._step_completion_times[step_index] - start_time
    
    def get_average_step_duration(self) -> Optional[float]:
        """
        Get the average duration of completed steps.
        
        Returns:
            Average duration in seconds, or None if no completed steps
        """
        with self._lock:
            completed_durations = []
            for i in range(self._current_step_index):
                duration = self.get_step_duration(i)
                if duration is not None:
                    completed_durations.append(duration)
            
            if not completed_durations:
                return None
            
            return sum(completed_durations) / len(completed_durations)
    
    def estimate_remaining_time(self) -> Optional[float]:
        """
        Estimate remaining time based on average step duration.
        
        Returns:
            Estimated seconds remaining, or None
        """
        with self._lock:
            avg_duration = self.get_average_step_duration()
            if avg_duration is None:
                return None
            
            remaining_steps = len(self._steps) - self._current_step_index - 1
            return avg_duration * remaining_steps
    
    def reset(self) -> None:
        """Reset the tracker to the beginning."""
        with self._lock:
            self._current_step_index = -1
            self._step_start_times.clear()
            self._step_completion_times.clear()


ProgressCallback = Callable[[ProgressEvent], None]


class TaskProgressPublisher:
    """
    Main class for publishing progress events through a message bus.
    
    This class provides a high-level interface for tracking and publishing
    progress of long-running tasks.
    """
    
    def __init__(
        self,
        task_id: Optional[UUID] = None,
        message_bus: Optional[Any] = None,
        auto_report_interval: float = 0.0,
    ) -> None:
        """
        Initialize the progress publisher.
        
        Args:
            task_id: Optional task identifier
            message_bus: Optional message bus for publishing events
            auto_report_interval: Interval in seconds for auto-reporting
        """
        self.task_id = task_id or uuid4()
        self._message_bus = message_bus
        self._auto_report_interval = auto_report_interval
        self._last_report_time: float = 0.0
        
        self._state = ProgressState.PENDING
        self._percentage: float = 0.0
        self._current_step: int = 0
        self._total_steps: int = 1
        self._step_name: str = ""
        self._step_description: str = ""
        self._message: str = ""
        self._metadata: Dict[str, Any] = {}
        
        self._start_time: Optional[float] = None
        self._lock = threading.RLock()
        
        self._step_tracker: Optional[StepTracker] = None
        self._eta_calculator = ETACalculator()
        self._cancellation_token = CancellationToken()
        
        self._subscribers: List[ProgressCallback] = []
        self._event_history: List[ProgressEvent] = []
        self._max_history_size: int = 1000
    
    @property
    def state(self) -> ProgressState:
        """Get the current state."""
        with self._lock:
            return self._state
    
    @property
    def percentage(self) -> float:
        """Get the current progress percentage."""
        with self._lock:
            return self._percentage
    
    @property
    def cancellation_token(self) -> CancellationToken:
        """Get the cancellation token."""
        return self._cancellation_token
    
    @property
    def is_active(self) -> bool:
        """Check if the task is active."""
        with self._lock:
            return self._state in {ProgressState.RUNNING, ProgressState.PAUSED}
    
    @property
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        with self._lock:
            return self._state in {ProgressState.COMPLETED, ProgressState.CANCELLED, ProgressState.FAILED}
    
    def subscribe(self, callback: ProgressCallback) -> None:
        """
        Subscribe to progress updates.
        
        Args:
            callback: Callback function to receive updates
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
    
    def unsubscribe(self, callback: ProgressCallback) -> None:
        """
        Unsubscribe from progress updates.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def start(
        self,
        total_steps: int = 1,
        initial_message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start tracking progress.
        
        Args:
            total_steps: Total number of steps
            initial_message: Initial progress message
            metadata: Optional initial metadata
        """
        with self._lock:
            self._state = ProgressState.RUNNING
            self._start_time = time.monotonic()
            self._percentage = 0.0
            self._current_step = 0
            self._total_steps = max(1, total_steps)
            self._message = initial_message
            self._metadata = metadata or {}
            
            self._eta_calculator.start()
            self._cancellation_token.reset()
            
            self._publish_update()
    
    def set_steps(self, steps: List[StepDefinition]) -> None:
        """
        Set up step tracking.
        
        Args:
            steps: List of step definitions
        """
        with self._lock:
            self._step_tracker = StepTracker(steps)
            self._total_steps = len(steps)
    
    def update(
        self,
        percentage: Optional[float] = None,
        step: Optional[int] = None,
        message: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """
        Update progress.
        
        Args:
            percentage: New progress percentage
            step: New step number
            message: New progress message
            **metadata: Additional metadata
        """
        with self._lock:
            if self._state != ProgressState.RUNNING:
                return
            
            if percentage is not None:
                self._percentage = max(0.0, min(100.0, percentage))
                self._eta_calculator.update(self._percentage)
            
            if step is not None:
                self._current_step = max(0, min(step, self._total_steps - 1))
                if self._step_tracker and self._current_step < self._step_tracker.total_steps:
                    step_def = self._step_tracker.start_step(self._current_step)
                    self._step_name = step_def.name
                    self._step_description = step_def.description
            
            if message is not None:
                self._message = message
            
            self._metadata.update(metadata)
            
            self._publish_update()
    
    def advance(self, step_name: str = "", message: str = "", delta: float = 0.0) -> None:
        """
        Advance progress by a delta or to the next step.
        
        Args:
            step_name: Name of the new step
            message: Progress message
            delta: Percentage to advance (0 = use step-based)
        """
        with self._lock:
            if self._state != ProgressState.RUNNING:
                return
            
            if delta > 0:
                self._percentage = min(100.0, self._percentage + delta)
            elif self._total_steps > 1:
                step_progress = 100.0 / self._total_steps
                self._percentage = min(100.0, self._percentage + step_progress)
                self._current_step = min(self._current_step + 1, self._total_steps - 1)
            else:
                self._percentage = min(100.0, self._percentage + 10.0)
            
            self._eta_calculator.update(self._percentage)
            
            if step_name:
                self._step_name = step_name
            if message:
                self._message = message
            
            self._publish_update()
    
    def pause(self, message: str = "Paused") -> None:
        """
        Pause progress tracking.
        
        Args:
            message: Pause message
        """
        with self._lock:
            if self._state == ProgressState.RUNNING:
                self._state = ProgressState.PAUSED
                self._message = message
                self._publish_update()
    
    def resume(self, message: str = "Resumed") -> None:
        """
        Resume progress tracking.
        
        Args:
            message: Resume message
        """
        with self._lock:
            if self._state == ProgressState.PAUSED:
                self._state = ProgressState.RUNNING
                self._message = message
                self._eta_calculator.update(self._percentage)
                self._publish_update()
    
    def complete(self, message: str = "Completed") -> None:
        """
        Mark the task as completed.
        
        Args:
            message: Completion message
        """
        with self._lock:
            self._state = ProgressState.COMPLETED
            self._percentage = 100.0
            self._message = message
            self._publish_update()
    
    def fail(self, message: str = "Failed") -> None:
        """
        Mark the task as failed.
        
        Args:
            message: Failure message
        """
        with self._lock:
            self._state = ProgressState.FAILED
            self._message = message
            self._publish_update()
    
    def cancel(self, reason: str = "Cancelled") -> None:
        """
        Cancel the task.
        
        Args:
            reason: Cancellation reason
        """
        with self._lock:
            self._state = ProgressState.CANCELLED
            self._message = reason
            self._cancellation_token.cancel(reason)
            self._publish_update()
    
    def get_current_event(self) -> ProgressEvent:
        """
        Get the current progress event.
        
        Returns:
            Current ProgressEvent
        """
        with self._lock:
            return self._create_event()
    
    def _create_event(self) -> ProgressEvent:
        """Create a progress event from current state."""
        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
        
        eta = self._eta_calculator.get_eta()
        
        return ProgressEvent(
            task_id=self.task_id,
            percentage=self._percentage,
            current_step=self._current_step,
            total_steps=self._total_steps,
            step_name=self._step_name,
            step_description=self._step_description,
            message=self._message,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            state=self._state,
            metadata=dict(self._metadata),
        )
    
    def _publish_update(self) -> None:
        """Publish a progress update to all subscribers."""
        event = self._create_event()
        
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)
        
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                pass
        
        if self._message_bus is not None:
            try:
                self._message_bus.publish(event)
            except Exception:
                pass
    
    def get_event_history(self) -> List[ProgressEvent]:
        """
        Get the history of progress events.
        
        Returns:
            List of progress events
        """
        with self._lock:
            return list(self._event_history)


class ProgressAggregator:
    """
    Aggregate progress from multiple sources.
    
    This class combines progress from multiple TaskProgressPublisher instances
    into a single view.
    """
    
    def __init__(self) -> None:
        self._publishers: Dict[UUID, TaskProgressPublisher] = {}
        self._weights: Dict[UUID, float] = {}
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[UUID, ProgressEvent], None]] = []
    
    def add_publisher(
        self,
        publisher: TaskProgressPublisher,
        weight: float = 1.0,
    ) -> None:
        """
        Add a progress publisher.
        
        Args:
            publisher: The publisher to add
            weight: Weight for aggregated calculation
        """
        with self._lock:
            self._publishers[publisher.task_id] = publisher
            self._weights[publisher.task_id] = weight
            
            publisher.subscribe(self._on_child_update)
    
    def remove_publisher(self, task_id: UUID) -> None:
        """
        Remove a progress publisher.
        
        Args:
            task_id: ID of the publisher to remove
        """
        with self._lock:
            if task_id in self._publishers:
                publisher = self._publishers[task_id]
                publisher.unsubscribe(self._on_child_update)
                del self._publishers[task_id]
                del self._weights[task_id]
    
    def subscribe(self, callback: Callable[[UUID, ProgressEvent], None]) -> None:
        """
        Subscribe to aggregated progress updates.
        
        Args:
            callback: Callback function
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[UUID, ProgressEvent], None]) -> None:
        """
        Unsubscribe from updates.
        
        Args:
            callback: Callback function
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def get_aggregated_percentage(self) -> float:
        """
        Get the weighted average of all progress percentages.
        
        Returns:
            Aggregated percentage
        """
        with self._lock:
            if not self._publishers:
                return 0.0
            
            total_weight = sum(self._weights.values())
            if total_weight == 0:
                return 0.0
            
            weighted_sum = sum(
                publisher.percentage * self._weights[task_id]
                for task_id, publisher in self._publishers.items()
            )
            
            return weighted_sum / total_weight
    
    def get_active_count(self) -> int:
        """
        Get the number of active tasks.
        
        Returns:
            Count of active tasks
        """
        with self._lock:
            return sum(1 for p in self._publishers.values() if p.is_active)
    
    def get_completed_count(self) -> int:
        """
        Get the number of completed tasks.
        
        Returns:
            Count of completed tasks
        """
        with self._lock:
            return sum(1 for p in self._publishers.values() if p.is_complete)
    
    def _on_child_update(self, event: ProgressEvent) -> None:
        """Handle update from a child publisher."""
        with self._lock:
            for subscriber in self._subscribers:
                try:
                    subscriber(event.task_id, event)
                except Exception:
                    pass
    
    def cancel_all(self) -> None:
        """Cancel all tracked tasks."""
        with self._lock:
            for publisher in self._publishers.values():
                publisher.cancel()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of aggregated progress.
        
        Returns:
            Summary dictionary
        """
        with self._lock:
            return {
                "total_tasks": len(self._publishers),
                "active_tasks": self.get_active_count(),
                "completed_tasks": self.get_completed_count(),
                "aggregated_percentage": self.get_aggregated_percentage(),
            }
