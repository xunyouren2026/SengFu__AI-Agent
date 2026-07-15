"""
Scheduled Task Management Module

Provides cron expression parsing, task scheduling, persistence,
execution logging, timezone handling, and task dependency chains.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import pickle
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

try:
    import zoneinfo
except ImportError:
    try:
        from backports import zoneinfo  # type: ignore
    except ImportError:
        zoneinfo = None  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_NAMES: Dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

DOW_NAMES: Dict[str, int] = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
}

FIELD_RANGES: Dict[str, Tuple[int, int]] = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day_of_month": (1, 31),
    "month": (1, 12),
    "day_of_week": (0, 6),
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CronField:
    """Represents a single cron field with its allowed values."""
    name: str
    raw: str
    values: FrozenSet[int] = field(default_factory=frozenset)

    def __contains__(self, value: int) -> bool:
        return value in self.values

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self.values))


@dataclass
class ExecutionRecord:
    """Record of a single task execution."""
    task_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    return_value: Any = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class TaskDefinition:
    """Definition of a scheduled task."""
    task_id: str
    name: str
    cron_expression: str
    callback: Optional[Callable[..., Any]] = None
    callback_name: Optional[str] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 5.0
    timeout: Optional[float] = None
    enabled: bool = True
    timezone_str: str = "UTC"
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    fail_count: int = 0


# ---------------------------------------------------------------------------
# Cron Expression Parser
# ---------------------------------------------------------------------------

class CronParser:
    """Parses cron expressions into structured field representations.

    Supports standard 5-field cron:
        minute  hour  day-of-month  month  day-of-week

    Special characters:
        *   any value
        ,   value list separator
        -   range
        /   step values
        L   last day of month (day-of-month)
        W   nearest weekday (day-of-month)
        #   nth day of week in month (day-of-week)
        ?   no specific value (day-of-month or day-of-week)
    """

    FIELD_NAMES: List[str] = [
        "minute", "hour", "day_of_month", "month", "day_of_week"
    ]

    def __init__(self, expression: str) -> None:
        self.raw_expression = expression.strip()
        self.fields: Dict[str, CronField] = {}
        self._parse(self.raw_expression)

    def _parse(self, expression: str) -> None:
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: expected 5 fields, got {len(parts)}: '{expression}'"
            )
        for name, part in zip(self.FIELD_NAMES, parts):
            self.fields[name] = self._parse_field(name, part)

    def _parse_field(self, name: str, raw: str) -> CronField:
        lo, hi = FIELD_RANGES[name]
        values: Set[int] = set()

        for segment in raw.split(","):
            segment = segment.strip()
            if not segment:
                continue
            values.update(self._parse_segment(name, segment, lo, hi))

        if not values:
            raise ValueError(f"No values parsed for field '{name}' from '{raw}'")

        return CronField(name=name, raw=raw, values=frozenset(values))

    def _parse_segment(
        self, name: str, segment: str, lo: int, hi: int
    ) -> Set[int]:
        step: Optional[int] = None
        if "/" in segment:
            range_part, step_str = segment.split("/", 1)
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"Step must be positive, got {step}")
        else:
            range_part = segment

        range_part = range_part.strip()

        if range_part == "*":
            start, end = lo, hi
        elif range_part == "?":
            return set(range(lo, hi + 1))
        elif range_part == "L" and name == "day_of_month":
            return {-1}  # sentinel for "last day"
        elif range_part.startswith("L") and name == "day_of_month":
            offset = int(range_part[1:]) if len(range_part) > 1 else 0
            return {-2 - offset}  # sentinel for "L-N"
        elif range_part.endswith("W") and name == "day_of_month":
            day = int(range_part[:-1])
            return {-10 - day}  # sentinel for nearest weekday
        elif "#" in range_part and name == "day_of_week":
            dow_str, nth_str = range_part.split("#", 1)
            dow = self._resolve_name(dow_str, DOW_NAMES)
            nth = int(nth_str)
            if nth < 1 or nth > 5:
                raise ValueError(f"Day-of-week # must be 1-5, got {nth}")
            return {-20 - dow * 10 - nth}  # sentinel
        elif "-" in range_part:
            start_str, end_str = range_part.split("-", 1)
            start = self._resolve_name(start_str, MONTH_NAMES if name == "month" else DOW_NAMES)
            end = self._resolve_name(end_str, MONTH_NAMES if name == "month" else DOW_NAMES)
        else:
            start = end = self._resolve_name(
                range_part, MONTH_NAMES if name == "month" else DOW_NAMES
            )

        if step is None:
            step = 1

        result: Set[int] = set()
        current = start
        while current <= end:
            if lo <= current <= hi:
                result.add(current)
            current += step

        return result

    def _resolve_name(self, value: str, name_map: Dict[str, int]) -> int:
        value = value.strip().lower()
        if value in name_map:
            return name_map[value]
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot resolve field value: '{value}'")

    def matches(self, dt: datetime) -> bool:
        """Check if a datetime matches this cron expression."""
        minute_val = self.fields["minute"]
        hour_val = self.fields["hour"]
        dom_val = self.fields["day_of_month"]
        month_val = self.fields["month"]
        dow_val = self.fields["day_of_week"]

        if dt.minute not in minute_val:
            return False
        if dt.hour not in hour_val:
            return False
        if dt.month not in month_val:
            return False

        # Day-of-month and day-of-week handling with special characters
        dom_match = self._matches_dom(dom_val, dt)
        dow_match = self._matches_dow(dow_val, dt)

        # Standard cron: if both are restricted, either match is ok
        # If only one is restricted, that one must match
        dom_is_star = len(dom_val) == (FIELD_RANGES["day_of_month"][1] - FIELD_RANGES["day_of_month"][0] + 1)
        dow_is_star = len(dow_val) == (FIELD_RANGES["day_of_week"][1] - FIELD_RANGES["day_of_week"][0] + 1)

        if dom_is_star and dow_is_star:
            return True
        if dom_is_star:
            return dow_match
        if dow_is_star:
            return dom_match
        return dom_match or dow_match

    def _matches_dom(self, dom_field: CronField, dt: datetime) -> bool:
        for v in dom_field:
            if v == -1:  # L
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                if dt.day == last_day:
                    return True
            elif v == -2:  # L-0
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                if dt.day == last_day:
                    return True
            elif v < -2 and v >= -32:  # L-N
                offset = -2 - v
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                target = last_day - offset
                if dt.day == target and target >= 1:
                    return True
            elif v < -10 and v >= -42:  # N W (nearest weekday)
                day = -10 - v
                if self._nearest_weekday(dt.year, dt.month, day) == dt.day:
                    return True
            elif v == dt.day:
                return True
        return False

    def _matches_dow(self, dow_field: CronField, dt: datetime) -> bool:
        actual_dow = dt.isoweekday() % 7  # 0=Sun
        for v in dow_field:
            if v < -20:  # DOW#N
                encoded = -20 - v
                dow = encoded // 10
                nth = encoded % 10
                if self._nth_weekday(dt.year, dt.month, dow, nth) == dt.day:
                    return True
            elif v == actual_dow:
                return True
        return False

    def _nearest_weekday(self, year: int, month: int, day: int) -> int:
        if day < 1:
            day = 1
        last_day = calendar.monthrange(year, month)[1]
        if day > last_day:
            day = last_day
        dt = datetime(year, month, day)
        dow = dt.weekday()  # 0=Mon
        if dow == 5:  # Saturday
            day -= 1
        elif dow == 6:  # Sunday
            day += 1
        day = max(1, min(day, last_day))
        return day

    def _nth_weekday(self, year: int, month: int, dow: int, nth: int) -> Optional[int]:
        """Find the nth occurrence of dow (0=Sun) in the given month."""
        count = 0
        for day in range(1, 32):
            try:
                dt = datetime(year, month, day)
            except ValueError:
                break
            if dt.isoweekday() % 7 == dow:
                count += 1
                if count == nth:
                    return day
        return None

    def next_occurrence(self, after: Optional[datetime] = None) -> datetime:
        """Calculate the next occurrence after the given datetime."""
        if after is None:
            after = datetime.utcnow()
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        max_iterations = 525960  # ~1 year of minutes
        for _ in range(max_iterations):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise RuntimeError(
            f"No matching time found within 1 year for cron: {self.raw_expression}"
        )

    def next_n_occurrences(self, n: int, after: Optional[datetime] = None) -> List[datetime]:
        """Calculate the next n occurrences."""
        results: List[datetime] = []
        current = after
        for _ in range(n):
            nxt = self.next_occurrence(current)
            results.append(nxt)
            current = nxt
        return results

    def describe(self) -> str:
        """Human-readable description of the cron expression."""
        parts: List[str] = []
        minute_field = self.fields["minute"]
        hour_field = self.fields["hour"]
        dom_field = self.fields["day_of_month"]
        month_field = self.fields["month"]
        dow_field = self.fields["day_of_week"]

        if len(month_field) < 12:
            months = ", ".join(str(m) for m in month_field)
            parts.append(f"in months {months}")
        if len(dom_field) < 31:
            days = ", ".join(str(d) for d in dom_field)
            parts.append(f"on days {days}")
        if len(dow_field) < 7:
            dows = ", ".join(str(d) for d in dow_field)
            parts.append(f"on weekdays {dows}")
        if len(hour_field) < 24:
            hours = ", ".join(f"{h:02d}" for h in hour_field)
            parts.append(f"at hours {hours}")
        if len(minute_field) < 60:
            minutes = ", ".join(f"{m:02d}" for m in minute_field)
            parts.append(f"at minutes {minutes}")

        if not parts:
            return "Every minute"
        return "Runs " + ", ".join(parts)


class CronExpression:
    """High-level wrapper around CronParser with timezone support."""

    def __init__(
        self,
        expression: str,
        tz: Optional[str] = None,
    ) -> None:
        self.parser = CronParser(expression)
        self.expression = expression
        self.timezone_name = tz or "UTC"
        self._tz = self._load_timezone(self.timezone_name)

    @staticmethod
    def _load_timezone(tz_name: str):
        if zoneinfo is not None:
            try:
                return zoneinfo.ZoneInfo(tz_name)
            except (KeyError, AttributeError):
                pass
        return timezone.utc

    def matches(self, dt: datetime) -> bool:
        if dt.tzinfo is not None:
            dt = dt.astimezone(self._tz)
        else:
            dt = dt.replace(tzinfo=timezone.utc).astimezone(self._tz)
        naive = dt.replace(tzinfo=None)
        return self.parser.matches(naive)

    def next_occurrence(self, after: Optional[datetime] = None) -> datetime:
        if after is None:
            after = datetime.now(self._tz)
        elif after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc).astimezone(self._tz)
        naive = after.replace(tzinfo=None)
        result = self.parser.next_occurrence(naive)
        return result.replace(tzinfo=self._tz)

    def next_n_occurrences(self, n: int, after: Optional[datetime] = None) -> List[datetime]:
        if after is None:
            after = datetime.now(self._tz)
        elif after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc).astimezone(self._tz)
        naive = after.replace(tzinfo=None)
        results = self.parser.next_n_occurrences(n, naive)
        return [r.replace(tzinfo=self._tz) for r in results]

    def describe(self) -> str:
        return self.parser.describe()


# ---------------------------------------------------------------------------
# Scheduled Task
# ---------------------------------------------------------------------------

class ScheduledTask:
    """Represents a single scheduled task with full lifecycle management."""

    def __init__(
        self,
        name: str,
        cron_expression: str,
        callback: Optional[Callable[..., Any]] = None,
        *,
        task_id: Optional[str] = None,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: Optional[float] = None,
        timezone_str: str = "UTC",
        tags: Optional[Set[str]] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.definition = TaskDefinition(
            task_id=task_id or str(uuid.uuid4()),
            name=name,
            cron_expression=cron_expression,
            callback=callback,
            callback_name=callback.__name__ if callback else None,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            timezone_str=timezone_str,
            tags=tags or set(),
            dependencies=dependencies or [],
            metadata=metadata or {},
        )
        self.cron = CronExpression(cron_expression, tz=timezone_str)
        self.status: TaskStatus = TaskStatus.PENDING
        self._lock = threading.Lock()

    @property
    def task_id(self) -> str:
        return self.definition.task_id

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def enabled(self) -> bool:
        return self.definition.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.definition.enabled = value

    def should_run(self, now: Optional[datetime] = None) -> bool:
        if not self.definition.enabled:
            return False
        return self.cron.matches(now or datetime.utcnow())

    def calculate_next_run(self) -> datetime:
        after = self.definition.last_run or datetime.utcnow()
        next_run = self.cron.next_occurrence(after)
        self.definition.next_run = next_run
        return next_run

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.definition.task_id,
            "name": self.definition.name,
            "cron_expression": self.definition.cron_expression,
            "callback_name": self.definition.callback_name,
            "priority": self.definition.priority.value,
            "max_retries": self.definition.max_retries,
            "retry_delay": self.definition.retry_delay,
            "timeout": self.definition.timeout,
            "enabled": self.definition.enabled,
            "timezone": self.definition.timezone_str,
            "tags": list(self.definition.tags),
            "dependencies": self.definition.dependencies,
            "status": self.status.value,
            "last_run": self.definition.last_run.isoformat() if self.definition.last_run else None,
            "next_run": self.definition.next_run.isoformat() if self.definition.next_run else None,
            "run_count": self.definition.run_count,
            "fail_count": self.definition.fail_count,
            "metadata": self.definition.metadata,
        }


# ---------------------------------------------------------------------------
# Task Logger
# ---------------------------------------------------------------------------

class TaskLogger:
    """Execution logging for scheduled tasks with persistence."""

    def __init__(
        self,
        max_records: int = 10000,
        persist_path: Optional[str] = None,
    ) -> None:
        self.max_records = max_records
        self.persist_path = persist_path
        self._records: Dict[str, List[ExecutionRecord]] = {}
        self._lock = threading.Lock()

    def log_start(
        self,
        task_id: str,
        execution_id: Optional[str] = None,
    ) -> ExecutionRecord:
        record = ExecutionRecord(
            task_id=task_id,
            execution_id=execution_id or str(uuid.uuid4()),
            start_time=datetime.utcnow(),
            status=TaskStatus.RUNNING,
        )
        with self._lock:
            if task_id not in self._records:
                self._records[task_id] = []
            self._records[task_id].append(record)
            self._trim_records(task_id)
        return record

    def log_success(
        self,
        task_id: str,
        execution_id: str,
        return_value: Any = None,
    ) -> None:
        with self._lock:
            record = self._find_record(task_id, execution_id)
            if record:
                record.end_time = datetime.utcnow()
                record.status = TaskStatus.SUCCESS
                record.return_value = return_value

    def log_failure(
        self,
        task_id: str,
        execution_id: str,
        error_message: str,
    ) -> None:
        with self._lock:
            record = self._find_record(task_id, execution_id)
            if record:
                record.end_time = datetime.utcnow()
                record.status = TaskStatus.FAILED
                record.error_message = error_message

    def log_retry(
        self,
        task_id: str,
        execution_id: str,
        retry_count: int,
        error_message: str,
    ) -> None:
        with self._lock:
            record = self._find_record(task_id, execution_id)
            if record:
                record.retry_count = retry_count
                record.status = TaskStatus.RETRYING
                record.error_message = error_message

    def _find_record(
        self, task_id: str, execution_id: str
    ) -> Optional[ExecutionRecord]:
        records = self._records.get(task_id, [])
        for r in records:
            if r.execution_id == execution_id:
                return r
        return None

    def _trim_records(self, task_id: str) -> None:
        records = self._records.get(task_id, [])
        if len(records) > self.max_records:
            self._records[task_id] = records[-self.max_records:]

    def get_records(
        self,
        task_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> List[ExecutionRecord]:
        with self._lock:
            if task_id:
                records = list(self._records.get(task_id, []))
            else:
                records = []
                for recs in self._records.values():
                    records.extend(recs)
                records.sort(key=lambda r: r.start_time, reverse=True)

            if status:
                records = [r for r in records if r.status == status]
            return records[:limit]

    def get_statistics(self, task_id: str) -> Dict[str, Any]:
        records = self._records.get(task_id, [])
        if not records:
            return {"total": 0}
        total = len(records)
        success = sum(1 for r in records if r.status == TaskStatus.SUCCESS)
        failed = sum(1 for r in records if r.status == TaskStatus.FAILED)
        durations = [r.duration for r in records if r.duration is not None]
        avg_duration = (
            sum(d.total_seconds() for d in durations) / len(durations)
            if durations
            else 0
        )
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "avg_duration_seconds": avg_duration,
        }

    def persist(self) -> None:
        if not self.persist_path:
            return
        data = {}
        with self._lock:
            for task_id, records in self._records.items():
                data[task_id] = [r.to_dict() for r in records]
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        with open(self.persist_path, "r") as f:
            data = json.load(f)
        with self._lock:
            self._records.clear()
            for task_id, record_dicts in data.items():
                self._records[task_id] = []
                for rd in record_dicts:
                    record = ExecutionRecord(
                        task_id=rd["task_id"],
                        execution_id=rd["execution_id"],
                        start_time=datetime.fromisoformat(rd["start_time"]),
                        end_time=(
                            datetime.fromisoformat(rd["end_time"])
                            if rd.get("end_time")
                            else None
                        ),
                        status=TaskStatus(rd["status"]),
                        error_message=rd.get("error_message"),
                        retry_count=rd.get("retry_count", 0),
                        metadata=rd.get("metadata", {}),
                    )
                    self._records[task_id].append(record)


# ---------------------------------------------------------------------------
# Dependency Chain
# ---------------------------------------------------------------------------

class DependencyChain:
    """Manages task dependency graphs and topological ordering."""

    def __init__(self) -> None:
        self._graph: Dict[str, Set[str]] = {}  # task_id -> set of dependency task_ids
        self._dependents: Dict[str, Set[str]] = {}  # task_id -> set of tasks that depend on it
        self._lock = threading.Lock()

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        if task_id == depends_on:
            raise ValueError(f"Task '{task_id}' cannot depend on itself")
        with self._lock:
            self._graph.setdefault(task_id, set()).add(depends_on)
            self._dependents.setdefault(depends_on, set()).add(task_id)

    def remove_dependency(self, task_id: str, depends_on: str) -> None:
        with self._lock:
            if task_id in self._graph:
                self._graph[task_id].discard(depends_on)
            if depends_on in self._dependents:
                self._dependents[depends_on].discard(task_id)

    def remove_task(self, task_id: str) -> None:
        with self._lock:
            deps = self._graph.pop(task_id, set())
            for dep in deps:
                if dep in self._dependents:
                    self._dependents[dep].discard(task_id)
            dependents = self._dependents.pop(task_id, set())
            for dependent in dependents:
                if dependent in self._graph:
                    self._graph[dependent].discard(task_id)

    def get_dependencies(self, task_id: str) -> Set[str]:
        with self._lock:
            return set(self._graph.get(task_id, set()))

    def get_dependents(self, task_id: str) -> Set[str]:
        with self._lock:
            return set(self._dependents.get(task_id, set()))

    def get_all_dependencies(self, task_id: str) -> Set[str]:
        """Get transitive closure of all dependencies."""
        visited: Set[str] = set()
        stack = list(self.get_dependencies(task_id))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.get_dependencies(current))
        return visited

    def topological_sort(self, task_ids: Optional[List[str]] = None) -> List[str]:
        """Return tasks in topological order (dependencies first)."""
        with self._lock:
            if task_ids is None:
                task_ids = list(self._graph.keys())

            in_degree: Dict[str, int] = {t: 0 for t in task_ids}
            adj: Dict[str, List[str]] = {t: [] for t in task_ids}

            for tid in task_ids:
                for dep in self._graph.get(tid, set()):
                    if dep in in_degree:
                        in_degree[tid] += 1
                        adj[dep].append(tid)

            queue = [t for t, d in in_degree.items() if d == 0]
            queue.sort()
            result: List[str] = []

            while queue:
                node = queue.pop(0)
                result.append(node)
                for neighbor in sorted(adj[node]):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
                        queue.sort()

            if len(result) != len(task_ids):
                cycle_tasks = set(task_ids) - set(result)
                raise ValueError(f"Circular dependency detected among: {cycle_tasks}")

            return result

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def resolve_order(self, task_id: str) -> List[str]:
        """Get the execution order for a task and all its dependencies."""
        all_deps = self.get_all_dependencies(task_id)
        all_deps.add(task_id)
        return self.topological_sort(list(all_deps))


# ---------------------------------------------------------------------------
# Task Executor
# ---------------------------------------------------------------------------

class TaskExecutor:
    """Executes scheduled tasks with retry, timeout, and error handling."""

    def __init__(
        self,
        logger: Optional[TaskLogger] = None,
        dependency_chain: Optional[DependencyChain] = None,
    ) -> None:
        self.logger = logger or TaskLogger()
        self.dependency_chain = dependency_chain or DependencyChain()
        self._running_tasks: Dict[str, threading.Thread] = {}
        self._completed_flags: Dict[str, bool] = {}
        self._lock = threading.Lock()

    def execute(self, task: ScheduledTask) -> ExecutionRecord:
        """Execute a task synchronously."""
        record = self.logger.log_start(task.task_id)
        callback = task.definition.callback

        if callback is None:
            self.logger.log_failure(
                task.task_id, record.execution_id, "No callback registered"
            )
            return record

        # Check dependencies
        deps = self.dependency_chain.get_dependencies(task.task_id)
        for dep_id in deps:
            if dep_id not in self._completed_flags or not self._completed_flags[dep_id]:
                self.logger.log_failure(
                    task.task_id,
                    record.execution_id,
                    f"Dependency '{dep_id}' not yet completed",
                )
                record.status = TaskStatus.SKIPPED
                return record

        last_error: Optional[str] = None
        for attempt in range(task.definition.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.log_retry(
                        task.task_id,
                        record.execution_id,
                        attempt,
                        last_error or "Unknown error",
                    )
                    time.sleep(task.definition.retry_delay * attempt)

                result = self._run_with_timeout(
                    callback,
                    task.definition.args,
                    task.definition.kwargs,
                    task.definition.timeout,
                )

                self.logger.log_success(task.task_id, record.execution_id, result)
                with self._lock:
                    self._completed_flags[task.task_id] = True
                task.definition.run_count += 1
                task.definition.last_run = datetime.utcnow()
                return record

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Task %s attempt %d failed: %s",
                    task.task_id, attempt + 1, last_error,
                )

        self.logger.log_failure(task.task_id, record.execution_id, last_error or "Unknown error")
        task.definition.fail_count += 1
        task.definition.last_run = datetime.utcnow()
        return record

    def execute_async(self, task: ScheduledTask) -> threading.Thread:
        """Execute a task in a background thread."""
        thread = threading.Thread(
            target=self.execute,
            args=(task,),
            daemon=True,
            name=f"task-{task.task_id}",
        )
        with self._lock:
            self._running_tasks[task.task_id] = thread
        thread.start()
        return thread

    def _run_with_timeout(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        timeout: Optional[float],
    ) -> Any:
        if timeout is None or timeout <= 0:
            return func(*args, **kwargs)

        result: Any = None
        exception: Optional[Exception] = None

        def target() -> None:
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                exception = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Task timed out after {timeout} seconds"
            )
        if exception is not None:
            raise exception
        return result

    def is_running(self, task_id: str) -> bool:
        with self._lock:
            thread = self._running_tasks.get(task_id)
            if thread is None:
                return False
            return thread.is_alive()

    def wait_for(self, task_id: str, timeout: Optional[float] = None) -> bool:
        with self._lock:
            thread = self._running_tasks.get(task_id)
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            thread = self._running_tasks.pop(task_id, None)
        return thread is not None and not thread.is_alive()


# ---------------------------------------------------------------------------
# Task Scheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Main scheduler that manages and runs scheduled tasks."""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        max_records: int = 10000,
        check_interval: float = 30.0,
    ) -> None:
        self.logger = TaskLogger(
            max_records=max_records,
            persist_path=persist_path,
        )
        self.dependency_chain = DependencyChain()
        self.executor = TaskExecutor(self.logger, self.dependency_chain)
        self.check_interval = check_interval
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_check: Optional[datetime] = None

    def register_task(self, task: ScheduledTask) -> str:
        with self._lock:
            self._tasks[task.task_id] = task
            for dep_id in task.definition.dependencies:
                self.dependency_chain.add_dependency(task.task_id, dep_id)
            task.calculate_next_run()
        logger.info("Registered task '%s' (id=%s)", task.name, task.task_id)
        return task.task_id

    def unregister_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                self.dependency_chain.remove_task(task_id)
                return True
            return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        tag: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[ScheduledTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        if tag:
            tasks = [t for t in tasks if tag in t.definition.tags]
        if enabled_only:
            tasks = [t for t in tasks if t.definition.enabled]
        return tasks

    def enable_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False

    def run_task_now(self, task_id: str) -> Optional[ExecutionRecord]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return self.executor.execute(task)

    def run_task_now_async(self, task_id: str) -> Optional[threading.Thread]:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return self.executor.execute_async(task)

    def _check_and_run(self) -> None:
        now = datetime.utcnow()
        self._last_check = now
        with self._lock:
            tasks_snapshot = list(self._tasks.values())

        for task in tasks_snapshot:
            if not task.definition.enabled:
                continue
            if self.executor.is_running(task.task_id):
                continue
            if task.should_run(now):
                try:
                    self.executor.execute_async(task)
                    task.calculate_next_run()
                except Exception as exc:
                    logger.error(
                        "Failed to schedule task '%s': %s", task.task_id, exc
                    )

    def start(self, blocking: bool = False) -> None:
        self._running = True
        logger.info("Task scheduler started (interval=%.1fs)", self.check_interval)

        def _loop() -> None:
            while self._running:
                try:
                    self._check_and_run()
                except Exception as exc:
                    logger.error("Scheduler loop error: %s", exc)
                time.sleep(self.check_interval)

        self._scheduler_thread = threading.Thread(
            target=_loop, daemon=True, name="task-scheduler"
        )
        self._scheduler_thread.start()

        if blocking:
            self._scheduler_thread.join()

    def stop(self) -> None:
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self.logger.persist()
        logger.info("Task scheduler stopped")

    def save_state(self, path: str) -> None:
        """Persist all task definitions to a file."""
        state = {
            "tasks": [],
            "dependencies": {},
        }
        with self._lock:
            for task in self._tasks.values():
                state["tasks"].append(task.to_dict())
            for tid, deps in self.dependency_chain._graph.items():
                state["dependencies"][tid] = list(deps)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str) -> int:
        """Load task definitions from a file. Returns count of loaded tasks."""
        if not os.path.exists(path):
            return 0
        with open(path, "r") as f:
            state = json.load(f)

        count = 0
        for td in state.get("tasks", []):
            task = ScheduledTask(
                name=td["name"],
                cron_expression=td["cron_expression"],
                task_id=td["task_id"],
                priority=TaskPriority(td.get("priority", 2)),
                max_retries=td.get("max_retries", 3),
                retry_delay=td.get("retry_delay", 5.0),
                timeout=td.get("timeout"),
                timezone_str=td.get("timezone", "UTC"),
                tags=set(td.get("tags", [])),
                dependencies=td.get("dependencies", []),
                metadata=td.get("metadata", {}),
            )
            task.definition.run_count = td.get("run_count", 0)
            task.definition.fail_count = td.get("fail_count", 0)
            if td.get("last_run"):
                task.definition.last_run = datetime.fromisoformat(td["last_run"])
            if td.get("next_run"):
                task.definition.next_run = datetime.fromisoformat(td["next_run"])
            task.enabled = td.get("enabled", True)
            self.register_task(task)
            count += 1

        return count

    def get_overview(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._tasks)
            enabled = sum(1 for t in self._tasks.values() if t.definition.enabled)
            running = sum(
                1 for t in self._tasks.values()
                if self.executor.is_running(t.task_id)
            )
        return {
            "total_tasks": total,
            "enabled_tasks": enabled,
            "running_tasks": running,
            "scheduler_running": self._running,
            "last_check": (
                self._last_check.isoformat() if self._last_check else None
            ),
            "check_interval": self.check_interval,
        }
