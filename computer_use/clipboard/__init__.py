"""
Clipboard History Module

Provides clipboard history management with ring buffer storage,
search, deduplication, and persistence.
"""

from .history import (
    ClipboardType,
    DeduplicationStrategy,
    HistoryEntry,
    HistorySearch,
    SizeManager,
    HistoryPersistence,
    ClipboardHistory,
)

__all__ = [
    "ClipboardType",
    "DeduplicationStrategy",
    "HistoryEntry",
    "HistorySearch",
    "SizeManager",
    "HistoryPersistence",
    "ClipboardHistory",
]
