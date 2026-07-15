"""
AGI Unified Framework - Events Module

This module provides comprehensive event handling capabilities for the AGI framework,
including event registration, progress tracking, and signal management.
"""

from __future__ import annotations

from .registry import (
    EventType,
    EventCategory,
    EventPriority,
    EventMetadata,
    EventDefinition,
    EventRegistry,
    EventFilter,
    BaseEvent,
    EventHandler,
    get_global_registry,
    register_event,
    get_event_definition,
)
from .task_progress import (
    ProgressState,
    ProgressEvent,
    StepDefinition,
    StepTracker,
    ETACalculator,
    CancellationToken,
    TaskProgressPublisher,
    ProgressAggregator,
    ProgressCallback,
)
from .signals import (
    SignalType,
    SignalAction,
    SignalConfig,
    CleanupHook,
    SignalInfo,
    SignalHandler,
    SignalManager,
    get_global_signal_manager,
    setup_default_handlers,
    register_cleanup,
    shutdown_signals,
    is_shutting_down,
    get_current_signal,
)


__all__ = [
    # Registry exports
    "EventType",
    "EventCategory",
    "EventPriority",
    "EventMetadata",
    "EventDefinition",
    "EventRegistry",
    "EventFilter",
    "BaseEvent",
    "EventHandler",
    "get_global_registry",
    "register_event",
    "get_event_definition",
    # Task progress exports
    "ProgressState",
    "ProgressEvent",
    "StepDefinition",
    "StepTracker",
    "ETACalculator",
    "CancellationToken",
    "TaskProgressPublisher",
    "ProgressAggregator",
    "ProgressCallback",
    # Signal exports
    "SignalType",
    "SignalAction",
    "SignalConfig",
    "CleanupHook",
    "SignalInfo",
    "SignalHandler",
    "SignalManager",
    "get_global_signal_manager",
    "setup_default_handlers",
    "register_cleanup",
    "shutdown_signals",
    "is_shutting_down",
    "get_current_signal",
]
