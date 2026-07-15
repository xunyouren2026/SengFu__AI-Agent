"""
Clipboard History Management Module

Provides clipboard history with:
- Ring buffer storage
- Search by content, type, and time
- Deduplication strategies
- Size limits and management
- Persistence to JSON

Pure Python standard library only.
"""

from __future__ import annotations

import json
import time
import os
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Callable


class ClipboardType(Enum):
    """Types of clipboard content."""
    TEXT = "text"
    HTML = "html"
    IMAGE = "image"
    FILE_LIST = "file_list"
    RTF = "rtf"
    UNKNOWN = "unknown"


class DeduplicationStrategy(Enum):
    """Deduplication strategies."""
    EXACT = "exact"
    NORMALIZED_TEXT = "normalized_text"
    CONTENT_HASH = "content_hash"
    FUZZY = "fuzzy"
    NONE = "none"


@dataclass
class HistoryEntry:
    """A single entry in the clipboard history."""
    entry_id: str
    content: str
    content_type: ClipboardType
    timestamp: float
    source_app: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = 0.0
    pinned: bool = False
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.size_bytes == 0:
            self.size_bytes = len(self.content.encode("utf-8", errors="replace"))
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def is_text(self) -> bool:
        return self.content_type in (ClipboardType.TEXT, ClipboardType.HTML, ClipboardType.RTF)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "content_type": self.content_type.value,
            "timestamp": self.timestamp,
            "source_app": self.source_app,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "pinned": self.pinned,
            "tags": self.tags,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HistoryEntry:
        return cls(
            entry_id=data["entry_id"],
            content=data["content"],
            content_type=ClipboardType(data.get("content_type", "text")),
            timestamp=data["timestamp"],
            source_app=data.get("source_app", ""),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", data.get("timestamp", 0)),
            pinned=data.get("pinned", False),
            tags=data.get("tags", []),
            size_bytes=data.get("size_bytes", 0),
        )


class HistorySearch:
    """
    Search functionality for clipboard history.

    Supports text search, type filtering, time range, and tag filtering.
    """

    def __init__(self, history: List[HistoryEntry]) -> None:
        self._history = history

    def search(self, query: str = "",
               content_type: Optional[ClipboardType] = None,
               start_time: Optional[float] = None,
               end_time: Optional[float] = None,
               tags: Optional[List[str]] = None,
               source_app: Optional[str] = None,
               max_results: int = 100,
               sort_by: str = "timestamp") -> List[HistoryEntry]:
        """Search clipboard history with multiple filters."""
        results: List[HistoryEntry] = []

        for entry in self._history:
            # Skip pinned entries only if not searching specifically
            if not self._matches_query(entry, query):
                continue
            if content_type and entry.content_type != content_type:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if tags and not any(t in entry.tags for t in tags):
                continue
            if source_app and entry.source_app != source_app:
                continue
            results.append(entry)

        # Sort
        if sort_by == "timestamp":
            results.sort(key=lambda e: e.timestamp, reverse=True)
        elif sort_by == "access_count":
            results.sort(key=lambda e: e.access_count, reverse=True)
        elif sort_by == "size":
            results.sort(key=lambda e: e.size_bytes, reverse=True)
        elif sort_by == "age":
            results.sort(key=lambda e: e.age_seconds)

        return results[:max_results]

    def _matches_query(self, entry: HistoryEntry, query: str) -> bool:
        """Check if an entry matches a text query."""
        if not query:
            return True
        query_lower = query.lower()
        return (query_lower in entry.content.lower() or
                query_lower in entry.source_app.lower() or
                any(query_lower in tag.lower() for tag in entry.tags))

    def find_duplicates(self, strategy: DeduplicationStrategy = DeduplicationStrategy.EXACT) -> List[List[HistoryEntry]]:
        """Find duplicate entries in the history."""
        groups: Dict[str, List[HistoryEntry]] = {}

        for entry in self._history:
            key = self._get_dedup_key(entry, strategy)
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)

        return [group for group in groups.values() if len(group) > 1]

    def _get_dedup_key(self, entry: HistoryEntry,
                       strategy: DeduplicationStrategy) -> str:
        """Get the deduplication key for an entry."""
        if strategy == DeduplicationStrategy.EXACT:
            return entry.content
        elif strategy == DeduplicationStrategy.NORMALIZED_TEXT:
            return self._normalize_text(entry.content)
        elif strategy == DeduplicationStrategy.CONTENT_HASH:
            return hashlib.md5(entry.content.encode()).hexdigest()
        elif strategy == DeduplicationStrategy.FUZZY:
            return self._normalize_text(entry.content)[:50]
        return entry.content

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().split())

    def get_frequent_entries(self, top_n: int = 10) -> List[HistoryEntry]:
        """Get the most frequently accessed entries."""
        sorted_entries = sorted(self._history, key=lambda e: e.access_count, reverse=True)
        return sorted_entries[:top_n]

    def get_recent_entries(self, count: int = 10) -> List[HistoryEntry]:
        """Get the most recent entries."""
        sorted_entries = sorted(self._history, key=lambda e: e.timestamp, reverse=True)
        return sorted_entries[:count]

    def get_entries_by_app(self, app_name: str) -> List[HistoryEntry]:
        """Get entries from a specific application."""
        return [e for e in self._history if e.source_app == app_name]


class SizeManager:
    """
    Manages clipboard history size limits.

    Handles total size limits, per-entry limits, and eviction strategies.
    """

    def __init__(self, max_entries: int = 1000,
                 max_total_size_mb: float = 50.0,
                 max_entry_size_mb: float = 10.0,
                 max_text_length: int = 100000) -> None:
        self.max_entries = max_entries
        self.max_total_size_bytes = int(max_total_size_mb * 1024 * 1024)
        self.max_entry_size_bytes = int(max_entry_size_mb * 1024 * 1024)
        self.max_text_length = max_text_length

    def check_entry_size(self, content: str) -> Tuple[bool, str]:
        """Check if an entry is within size limits."""
        size = len(content.encode("utf-8", errors="replace"))
        if size > self.max_entry_size_bytes:
            return False, f"Entry size {size} exceeds limit {self.max_entry_size_bytes}"
        if len(content) > self.max_text_length:
            return False, f"Text length {len(content)} exceeds limit {self.max_text_length}"
        return True, ""

    def needs_eviction(self, entries: List[HistoryEntry]) -> bool:
        """Check if eviction is needed."""
        if len(entries) > self.max_entries:
            return True
        total_size = sum(e.size_bytes for e in entries)
        if total_size > self.max_total_size_bytes:
            return True
        return False

    def evict(self, entries: List[HistoryEntry],
              strategy: str = "lru") -> List[HistoryEntry]:
        """
        Evict entries to meet size limits.

        Strategies: lru (least recently used), fifo (first in first out),
        lfrequent (least frequently used), largest (remove largest first).
        """
        entries = list(entries)

        while self.needs_eviction(entries):
            if not entries:
                break

            if strategy == "lru":
                # Remove least recently accessed non-pinned entry
                candidates = [e for e in entries if not e.pinned]
                if not candidates:
                    break
                victim = min(candidates, key=lambda e: e.last_accessed)
                entries.remove(victim)

            elif strategy == "fifo":
                # Remove oldest non-pinned entry
                candidates = [e for e in entries if not e.pinned]
                if not candidates:
                    break
                victim = min(candidates, key=lambda e: e.timestamp)
                entries.remove(victim)

            elif strategy == "lfrequent":
                # Remove least frequently accessed non-pinned entry
                candidates = [e for e in entries if not e.pinned]
                if not candidates:
                    break
                victim = min(candidates, key=lambda e: e.access_count)
                entries.remove(victim)

            elif strategy == "largest":
                # Remove largest non-pinned entry
                candidates = [e for e in entries if not e.pinned]
                if not candidates:
                    break
                victim = max(candidates, key=lambda e: e.size_bytes)
                entries.remove(victim)

        return entries

    def get_total_size(self, entries: List[HistoryEntry]) -> int:
        """Get total size of all entries in bytes."""
        return sum(e.size_bytes for e in entries)

    def get_size_stats(self, entries: List[HistoryEntry]) -> Dict[str, Any]:
        """Get size statistics."""
        total = self.get_total_size(entries)
        return {
            "total_entries": len(entries),
            "total_size_bytes": total,
            "total_size_mb": total / (1024 * 1024),
            "max_entries": self.max_entries,
            "max_total_size_mb": self.max_total_size_bytes / (1024 * 1024),
            "utilization": len(entries) / max(1, self.max_entries),
            "size_utilization": total / max(1, self.max_total_size_bytes),
        }


class HistoryPersistence:
    """
    Persistence for clipboard history.

    Saves and loads history to/from JSON files.
    """

    def __init__(self, file_path: str = "clipboard_history.json") -> None:
        self.file_path = file_path

    def save(self, entries: List[HistoryEntry]) -> bool:
        """Save history to file."""
        try:
            data = {
                "version": 1,
                "saved_at": time.time(),
                "entries": [e.to_dict() for e in entries],
            }
            # Write atomically using temp file
            temp_path = self.file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
            return True
        except (OSError, IOError, TypeError):
            return False

    def load(self) -> List[HistoryEntry]:
        """Load history from file."""
        try:
            if not os.path.exists(self.file_path):
                return []
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = [HistoryEntry.from_dict(e) for e in data.get("entries", [])]
            return entries
        except (OSError, IOError, json.JSONDecodeError, KeyError, TypeError):
            return []

    def export_to_text(self, entries: List[HistoryEntry],
                       output_path: str) -> bool:
        """Export history to a readable text file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    timestamp_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp)
                    )
                    f.write(f"[{timestamp_str}] ({entry.content_type.value})")
                    if entry.source_app:
                        f.write(f" [{entry.source_app}]")
                    f.write(f"\n{entry.content[:200]}\n")
                    if len(entry.content) > 200:
                        f.write(f"... (truncated, {entry.size_bytes} bytes)\n")
                    f.write("\n")
            return True
        except (OSError, IOError):
            return False

    def get_file_size(self) -> int:
        """Get the size of the persistence file."""
        try:
            return os.path.getsize(self.file_path)
        except OSError:
            return 0


class ClipboardHistory:
    """
    Clipboard history manager.

    Provides a ring buffer for clipboard entries with search,
    deduplication, size management, and persistence.
    """

    def __init__(self, max_entries: int = 1000,
                 max_total_size_mb: float = 50.0,
                 dedup_strategy: DeduplicationStrategy = DeduplicationStrategy.EXACT,
                 persistence_path: Optional[str] = None,
                 auto_save: bool = False) -> None:
        self._entries: List[HistoryEntry] = []
        self.size_manager = SizeManager(
            max_entries=max_entries,
            max_total_size_mb=max_total_size_mb,
        )
        self.dedup_strategy = dedup_strategy
        self.persistence = HistoryPersistence(
            persistence_path or "clipboard_history.json"
        )
        self.auto_save = auto_save
        self._entry_counter = 0

        # Load existing history
        loaded = self.persistence.load()
        self._entries = loaded
        if loaded:
            self._entry_counter = max(
                int(e.entry_id.split("-")[-1]) for e in loaded
            ) + 1 if loaded else 0

    def add(self, content: str, content_type: ClipboardType = ClipboardType.TEXT,
            source_app: str = "", metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None) -> Optional[HistoryEntry]:
        """Add a new entry to the history."""
        # Check size
        valid, error = self.size_manager.check_entry_size(content)
        if not valid:
            return None

        # Check for duplicates
        if self.dedup_strategy != DeduplicationStrategy.NONE:
            if self._is_duplicate(content, content_type):
                return None

        self._entry_counter += 1
        entry = HistoryEntry(
            entry_id=f"clip-{self._entry_counter}",
            content=content,
            content_type=content_type,
            timestamp=time.time(),
            source_app=source_app,
            metadata=metadata or {},
            tags=tags or [],
        )

        self._entries.append(entry)

        # Evict if needed
        if self.size_manager.needs_eviction(self._entries):
            self._entries = self.size_manager.evict(self._entries)

        if self.auto_save:
            self.save()

        return entry

    def _is_duplicate(self, content: str, content_type: ClipboardType) -> bool:
        """Check if content is a duplicate of a recent entry."""
        if not self._entries:
            return False

        # Only check recent entries (last 50)
        recent = self._entries[-50:]

        for entry in reversed(recent):
            if entry.content_type != content_type:
                continue

            if self.dedup_strategy == DeduplicationStrategy.EXACT:
                if entry.content == content:
                    return True
            elif self.dedup_strategy == DeduplicationStrategy.NORMALIZED_TEXT:
                if " ".join(content.lower().split()) == " ".join(entry.content.lower().split()):
                    return True
            elif self.dedup_strategy == DeduplicationStrategy.CONTENT_HASH:
                if hashlib.md5(content.encode()).hexdigest() == \
                   hashlib.md5(entry.content.encode()).hexdigest():
                    return True
            elif self.dedup_strategy == DeduplicationStrategy.FUZZY:
                norm_content = " ".join(content.lower().split())[:100]
                norm_entry = " ".join(entry.content.lower().split())[:100]
                if norm_content == norm_entry:
                    return True

        return False

    def get_latest(self) -> Optional[HistoryEntry]:
        """Get the latest clipboard entry."""
        return self._entries[-1] if self._entries else None

    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Get an entry by ID."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry
        return None

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        for i, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                self._entries.pop(i)
                if self.auto_save:
                    self.save()
                return True
        return False

    def pin(self, entry_id: str) -> bool:
        """Pin an entry to prevent eviction."""
        entry = self.get_entry(entry_id)
        if entry:
            entry.pinned = True
            return True
        return False

    def unpin(self, entry_id: str) -> bool:
        """Unpin an entry."""
        entry = self.get_entry(entry_id)
        if entry:
            entry.pinned = False
            return True
        return False

    def get_pinned(self) -> List[HistoryEntry]:
        """Get all pinned entries."""
        return [e for e in self._entries if e.pinned]

    def add_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to an entry."""
        entry = self.get_entry(entry_id)
        if entry and tag not in entry.tags:
            entry.tags.append(tag)
            return True
        return False

    def remove_tag(self, entry_id: str, tag: str) -> bool:
        """Remove a tag from an entry."""
        entry = self.get_entry(entry_id)
        if entry and tag in entry.tags:
            entry.tags.remove(tag)
            return True
        return False

    def search(self, query: str = "", **kwargs: Any) -> List[HistoryEntry]:
        """Search the clipboard history."""
        searcher = HistorySearch(self._entries)
        return searcher.search(query, **kwargs)

    def clear(self, keep_pinned: bool = True) -> int:
        """Clear the history."""
        if keep_pinned:
            pinned = [e for e in self._entries if e.pinned]
            removed = len(self._entries) - len(pinned)
            self._entries = pinned
        else:
            removed = len(self._entries)
            self._entries = []
        if self.auto_save:
            self.save()
        return removed

    def save(self) -> bool:
        """Save history to disk."""
        return self.persistence.save(self._entries)

    def load(self) -> int:
        """Load history from disk."""
        self._entries = self.persistence.load()
        return len(self._entries)

    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics."""
        size_stats = self.size_manager.get_size_stats(self._entries)
        type_counts: Dict[str, int] = {}
        app_counts: Dict[str, int] = {}
        for entry in self._entries:
            type_counts[entry.content_type.value] = type_counts.get(entry.content_type.value, 0) + 1
            if entry.source_app:
                app_counts[entry.source_app] = app_counts.get(entry.source_app, 0) + 1

        return {
            **size_stats,
            "type_distribution": type_counts,
            "app_distribution": app_counts,
            "pinned_count": sum(1 for e in self._entries if e.pinned),
            "total_accesses": sum(e.access_count for e in self._entries),
        }

    @property
    def count(self) -> int:
        """Get the number of entries."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Any:
        return iter(self._entries)
