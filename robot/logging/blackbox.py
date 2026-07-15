"""
Robot Black Box Recorder Module

Provides flight-recorder-style logging for robots:
- Joint data logging at high frequency
- Event recording with timestamps
- Crash detection and pre-crash data capture
- Data compression for long-term storage
- Playback analysis and visualization data extraction
"""

from __future__ import annotations

import math
import time
import struct
import threading
import hashlib
import zlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator
from enum import Enum, IntEnum
from collections import deque
from datetime import datetime, timezone

try:
    import gzip
    _HAS_GZIP = True
except ImportError:
    _HAS_GZIP = False

try:
    import io
except ImportError:
    io = None  # type: ignore


class LogLevel(IntEnum):
    """Log severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class EventType(IntEnum):
    """Types of recorded events."""
    JOINT_UPDATE = 0
    JOINT_LIMIT_REACHED = 1
    COLLISION_DETECTED = 2
    EMERGENCY_STOP = 3
    PROGRAM_START = 4
    PROGRAM_END = 5
    WAYPOINT_REACHED = 6
    GRIPPER_ACTION = 7
    SENSOR_READING = 8
    COMMUNICATION_ERROR = 9
    CALIBRATION_EVENT = 10
    MODE_CHANGE = 11
    FAULT_DETECTED = 12
    RECOVERY_ACTION = 13
    CUSTOM = 99


class CrashSeverity(IntEnum):
    """Severity levels for crash events."""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class JointState:
    """Snapshot of joint states at a point in time."""
    positions: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    torques: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "velocities": self.velocities,
            "accelerations": self.accelerations,
            "torques": self.torques,
            "temperatures": self.temperatures,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JointState:
        return cls(
            positions=data.get("positions", []),
            velocities=data.get("velocities", []),
            accelerations=data.get("accelerations", []),
            torques=data.get("torques", []),
            temperatures=data.get("temperatures", []),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class LogEvent:
    """A logged event with metadata."""
    event_type: EventType
    timestamp: float
    level: LogLevel = LogLevel.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp,
            "level": self.level.name,
            "message": self.message,
            "data": self.data,
            "source": self.source,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LogEvent:
        event_type = EventType[data.get("event_type", "CUSTOM")]
        level = LogLevel[data.get("level", "INFO")]
        return cls(
            event_type=event_type,
            timestamp=data.get("timestamp", 0.0),
            level=level,
            message=data.get("message", ""),
            data=data.get("data", {}),
            source=data.get("source", ""),
            sequence_number=data.get("sequence_number", 0),
        )


@dataclass
class CrashRecord:
    """Record of a crash or near-crash event."""
    timestamp: float
    severity: CrashSeverity
    description: str
    pre_crash_data: List[JointState] = field(default_factory=list)
    post_crash_data: List[JointState] = field(default_factory=list)
    related_events: List[LogEvent] = field(default_factory=list)
    joint_states_at_crash: Optional[JointState] = None
    estimated_cause: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.name,
            "description": self.description,
            "pre_crash_data": [js.to_dict() for js in self.pre_crash_data],
            "post_crash_data": [js.to_dict() for js in self.post_crash_data],
            "related_events": [ev.to_dict() for ev in self.related_events],
            "joint_states_at_crash": (
                self.joint_states_at_crash.to_dict() if self.joint_states_at_crash else None
            ),
            "estimated_cause": self.estimated_cause,
        }


class JointDataLogger:
    """High-frequency joint data logger.

    Records joint states in a ring buffer for efficient memory usage.
    Supports configurable sampling rates and buffer sizes.
    """

    def __init__(
        self,
        num_joints: int = 6,
        buffer_size: int = 10000,
        sample_rate_hz: float = 100.0,
    ) -> None:
        self.num_joints = num_joints
        self.buffer_size = buffer_size
        self.sample_rate_hz = sample_rate_hz
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._recording = False
        self._sample_interval = 1.0 / sample_rate_hz
        self._last_sample_time: float = 0.0
        self._total_samples: int = 0
        self._dropped_samples: int = 0

    def record(self, state: JointState) -> bool:
        """Record a joint state sample."""
        current_time = time.monotonic()
        elapsed = current_time - self._last_sample_time

        if elapsed < self._sample_interval * 0.9:
            self._dropped_samples += 1
            return False

        with self._lock:
            state.timestamp = state.timestamp or time.time()
            self._buffer.append(state)
            self._total_samples += 1
            self._last_sample_time = current_time
            return True

    def record_raw(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
        torques: Optional[List[float]] = None,
    ) -> bool:
        """Record joint state from raw values."""
        state = JointState(
            positions=positions[:self.num_joints],
            velocities=(velocities or [])[:self.num_joints],
            torques=(torques or [])[:self.num_joints],
            timestamp=time.time(),
        )
        return self.record(state)

    def get_recent(self, count: int = 100) -> List[JointState]:
        """Get the most recent N samples."""
        with self._lock:
            buf_list = list(self._buffer)
            return buf_list[-count:]

    def get_range(self, start_time: float, end_time: float) -> List[JointState]:
        """Get samples within a time range."""
        with self._lock:
            return [
                js for js in self._buffer
                if start_time <= js.timestamp <= end_time
            ]

    def get_all(self) -> List[JointState]:
        """Get all buffered samples."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_samples = 0
            self._dropped_samples = 0

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def dropped_samples(self) -> int:
        return self._dropped_samples

    @property
    def buffer_usage(self) -> float:
        return len(self._buffer) / max(self.buffer_size, 1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._lock:
            if not self._buffer:
                return {
                    "total_samples": self._total_samples,
                    "dropped_samples": self._dropped_samples,
                    "buffer_usage": 0.0,
                    "buffer_size": self.buffer_size,
                }

            positions = [js.positions for js in self._buffer if js.positions]
            stats: Dict[str, Any] = {
                "total_samples": self._total_samples,
                "dropped_samples": self._dropped_samples,
                "buffer_usage": self.buffer_usage,
                "buffer_size": self.buffer_size,
                "time_span": (
                    self._buffer[-1].timestamp - self._buffer[0].timestamp
                    if len(self._buffer) > 1 else 0.0
                ),
            }

            if positions:
                n_joints = len(positions[0])
                for j in range(n_joints):
                    vals = [p[j] for p in positions if j < len(p)]
                    if vals:
                        stats[f"joint_{j}_min"] = min(vals)
                        stats[f"joint_{j}_max"] = max(vals)
                        stats[f"joint_{j}_mean"] = sum(vals) / len(vals)

            return stats


class EventRecorder:
    """Records events with timestamps and metadata."""

    def __init__(
        self,
        buffer_size: int = 5000,
        auto_flush_interval: float = 60.0,
    ) -> None:
        self.buffer_size = buffer_size
        self.auto_flush_interval = auto_flush_interval
        self._events: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._sequence_counter: int = 0
        self._filters: Dict[EventType, bool] = {}
        self._last_flush_time: float = time.time()

    def record_event(
        self,
        event_type: EventType,
        message: str = "",
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> LogEvent:
        """Record an event."""
        if self._filters.get(event_type, True) is False:
            return LogEvent(event_type, 0.0)

        with self._lock:
            self._sequence_counter += 1
            event = LogEvent(
                event_type=event_type,
                timestamp=time.time(),
                level=level,
                message=message,
                data=data or {},
                source=source,
                sequence_number=self._sequence_counter,
            )
            self._events.append(event)
            return event

    def record_joint_limit(
        self, joint_index: int, limit_type: str, value: float
    ) -> LogEvent:
        """Record a joint limit event."""
        return self.record_event(
            EventType.JOINT_LIMIT_REACHED,
            f"Joint {joint_index} {limit_type} limit reached: {value:.4f}",
            LogLevel.WARNING,
            {"joint_index": joint_index, "limit_type": limit_type, "value": value},
        )

    def record_collision(
        self, location: str, force: float, severity: str = "minor"
    ) -> LogEvent:
        """Record a collision event."""
        return self.record_event(
            EventType.COLLISION_DETECTED,
            f"Collision at {location}, force: {force:.2f}N, severity: {severity}",
            LogLevel.ERROR,
            {"location": location, "force": force, "severity": severity},
        )

    def record_emergency_stop(self, reason: str = "") -> LogEvent:
        """Record an emergency stop event."""
        return self.record_event(
            EventType.EMERGENCY_STOP,
            f"Emergency stop activated: {reason}",
            LogLevel.CRITICAL,
            {"reason": reason},
        )

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000,
    ) -> List[LogEvent]:
        """Query events with optional filters."""
        with self._lock:
            events = list(self._events)

        filtered: List[LogEvent] = []
        for ev in events:
            if event_type is not None and ev.event_type != event_type:
                continue
            if level is not None and ev.level < level:
                continue
            if start_time is not None and ev.timestamp < start_time:
                continue
            if end_time is not None and ev.timestamp > end_time:
                continue
            filtered.append(ev)
            if len(filtered) >= limit:
                break

        return filtered

    def get_recent(self, count: int = 100) -> List[LogEvent]:
        """Get the most recent events."""
        with self._lock:
            events = list(self._events)
        return events[-count:]

    def set_filter(self, event_type: EventType, enabled: bool) -> None:
        """Enable or disable recording of a specific event type."""
        self._filters[event_type] = enabled

    def clear(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()
            self._sequence_counter = 0

    @property
    def event_count(self) -> int:
        return len(self._events)

    def get_event_counts_by_type(self) -> Dict[str, int]:
        """Get count of events grouped by type."""
        with self._lock:
            counts: Dict[str, int] = {}
            for ev in self._events:
                name = ev.event_type.name
                counts[name] = counts.get(name, 0) + 1
            return counts


class CrashDetector:
    """Detects crashes and captures pre/post crash data.

    Monitors joint data and events for crash indicators such as
    sudden acceleration changes, collision events, and emergency stops.
    """

    def __init__(
        self,
        acceleration_threshold: float = 50.0,
        jerk_threshold: float = 500.0,
        pre_crash_window_seconds: float = 5.0,
        post_crash_window_seconds: float = 2.0,
    ) -> None:
        self.acceleration_threshold = acceleration_threshold
        self.jerk_threshold = jerk_threshold
        self.pre_crash_window = pre_crash_window_seconds
        self.post_crash_window = post_crash_window_seconds
        self._crashes: List[CrashRecord] = []
        self._prev_velocities: Optional[List[float]] = None
        self._prev_accelerations: Optional[List[float]] = None
        self._lock = threading.Lock()
        self._crash_callbacks: List[Callable[[CrashRecord], None]] = []

    def analyze_joint_state(
        self,
        state: JointState,
        joint_logger: JointDataLogger,
        event_recorder: EventRecorder,
    ) -> Optional[CrashRecord]:
        """Analyze a joint state for crash indicators."""
        crash_severity = CrashSeverity.NONE
        description = ""

        if state.accelerations:
            for i, acc in enumerate(state.accelerations):
                if abs(acc) > self.acceleration_threshold:
                    crash_severity = max(crash_severity, CrashSeverity.MODERATE)
                    description = f"High acceleration on joint {i}: {acc:.2f} rad/s^2"

        if self._prev_accelerations and state.accelerations:
            for i, (curr, prev) in enumerate(
                zip(state.accelerations, self._prev_accelerations)
            ):
                jerk = abs(curr - prev)
                if jerk > self.jerk_threshold:
                    crash_severity = max(crash_severity, CrashSeverity.SEVERE)
                    description = f"High jerk on joint {i}: {jerk:.2f} rad/s^3"

        self._prev_velocities = state.velocities[:]
        self._prev_accelerations = state.accelerations[:]

        if crash_severity == CrashSeverity.NONE:
            return None

        crash_time = state.timestamp
        pre_data = joint_logger.get_range(
            crash_time - self.pre_crash_window, crash_time
        )
        post_data = joint_logger.get_range(
            crash_time, crash_time + self.post_crash_window
        )
        related_events = event_recorder.get_events(
            start_time=crash_time - self.pre_crash_window,
            end_time=crash_time + self.post_crash_window,
        )

        cause = self._estimate_cause(state, related_events)

        record = CrashRecord(
            timestamp=crash_time,
            severity=crash_severity,
            description=description,
            pre_crash_data=pre_data,
            post_crash_data=post_data,
            related_events=related_events,
            joint_states_at_crash=state,
            estimated_cause=cause,
        )

        with self._lock:
            self._crashes.append(record)

        for cb in self._crash_callbacks:
            try:
                cb(record)
            except Exception:
                pass

        return record

    def check_event_crash(
        self,
        event: LogEvent,
        joint_logger: JointDataLogger,
    ) -> Optional[CrashRecord]:
        """Check if an event indicates a crash."""
        if event.event_type == EventType.EMERGENCY_STOP:
            pre_data = joint_logger.get_range(
                event.timestamp - self.pre_crash_window, event.timestamp
            )
            record = CrashRecord(
                timestamp=event.timestamp,
                severity=CrashSeverity.SEVERE,
                description=f"Emergency stop: {event.message}",
                pre_crash_data=pre_data,
                related_events=[event],
                estimated_cause="Emergency stop activation",
            )
            with self._lock:
                self._crashes.append(record)
            return record

        if event.event_type == EventType.COLLISION_DETECTED:
            pre_data = joint_logger.get_range(
                event.timestamp - self.pre_crash_window, event.timestamp
            )
            severity = CrashSeverity.MODERATE
            force = event.data.get("force", 0)
            if force > 100:
                severity = CrashSeverity.SEVERE
            if force > 500:
                severity = CrashSeverity.CRITICAL
            record = CrashRecord(
                timestamp=event.timestamp,
                severity=severity,
                description=f"Collision: {event.message}",
                pre_crash_data=pre_data,
                related_events=[event],
                estimated_cause=f"Physical collision (force={force:.1f}N)",
            )
            with self._lock:
                self._crashes.append(record)
            return record

        return None

    def _estimate_cause(
        self, state: JointState, events: List[LogEvent]
    ) -> str:
        """Estimate the likely cause of a crash."""
        for ev in reversed(events):
            if ev.event_type == EventType.COLLISION_DETECTED:
                return "Collision detected"
            if ev.event_type == EventType.COMMUNICATION_ERROR:
                return "Communication failure"
            if ev.event_type == EventType.FAULT_DETECTED:
                return f"Hardware fault: {ev.message}"

        if state.torques:
            max_torque_idx = max(
                range(len(state.torques)), key=lambda i: abs(state.torques[i])
            )
            max_torque = abs(state.torques[max_torque_idx])
            if max_torque > 100:
                return f"Over-torque on joint {max_torque_idx} ({max_torque:.1f}Nm)"

        if state.accelerations:
            max_acc_idx = max(
                range(len(state.accelerations)),
                key=lambda i: abs(state.accelerations[i]),
            )
            return f"Sudden acceleration on joint {max_acc_idx}"

        return "Unknown cause"

    def register_callback(
        self, callback: Callable[[CrashRecord], None]
    ) -> None:
        """Register a callback for crash detection events."""
        self._crash_callbacks.append(callback)

    def get_crashes(
        self, min_severity: Optional[CrashSeverity] = None
    ) -> List[CrashRecord]:
        """Get recorded crashes, optionally filtered by severity."""
        with self._lock:
            crashes = list(self._crashes)
        if min_severity is not None:
            crashes = [c for c in crashes if c.severity >= min_severity]
        return crashes

    def clear(self) -> None:
        """Clear all crash records."""
        with self._lock:
            self._crashes.clear()


class DataCompressor:
    """Compresses and decompresses black box data for storage."""

    def __init__(self, compression_level: int = 6) -> None:
        self.compression_level = compression_level

    def compress_joint_states(
        self, states: List[JointState]
    ) -> bytes:
        """Compress a list of joint states to bytes."""
        data_parts: List[bytes] = []
        data_parts.append(struct.pack(">I", len(states)))

        for state in states:
            data_parts.append(struct.pack(">d", state.timestamp))
            n_pos = len(state.positions)
            data_parts.append(struct.pack(">H", n_pos))

            for val in state.positions:
                data_parts.append(struct.pack(">f", val))
            for val in state.velocities:
                data_parts.append(struct.pack(">f", val))
            for val in state.torques:
                data_parts.append(struct.pack(">f", val))

        raw = b"".join(data_parts)
        return zlib.compress(raw, self.compression_level)

    def decompress_joint_states(self, data: bytes) -> List[JointState]:
        """Decompress joint states from bytes."""
        raw = zlib.decompress(data)
        offset = 0

        num_states = struct.unpack_from(">I", raw, offset)[0]
        offset += 4

        states: List[JointState] = []
        for _ in range(num_states):
            timestamp = struct.unpack_from(">d", raw, offset)[0]
            offset += 8
            n_pos = struct.unpack_from(">H", raw, offset)[0]
            offset += 2

            positions = []
            for _ in range(n_pos):
                val = struct.unpack_from(">f", raw, offset)[0]
                offset += 4
                positions.append(val)

            velocities = []
            for _ in range(n_pos):
                if offset + 4 > len(raw):
                    break
                val = struct.unpack_from(">f", raw, offset)[0]
                offset += 4
                velocities.append(val)

            torques = []
            for _ in range(n_pos):
                if offset + 4 > len(raw):
                    break
                val = struct.unpack_from(">f", raw, offset)[0]
                offset += 4
                torques.append(val)

            states.append(JointState(
                positions=positions,
                velocities=velocities,
                torques=torques,
                timestamp=timestamp,
            ))

        return states

    def compress_events(self, events: List[LogEvent]) -> bytes:
        """Compress events to bytes."""
        json_str = json.dumps(
            [ev.to_dict() for ev in events],
            separators=(",", ":"),
        )
        raw = json_str.encode("utf-8")
        return zlib.compress(raw, self.compression_level)

    def decompress_events(self, data: bytes) -> List[LogEvent]:
        """Decompress events from bytes."""
        raw = zlib.decompress(data)
        json_str = raw.decode("utf-8")
        event_dicts = json.loads(json_str)
        return [LogEvent.from_dict(d) for d in event_dicts]

    def compress_crash_records(self, crashes: List[CrashRecord]) -> bytes:
        """Compress crash records to bytes."""
        json_str = json.dumps(
            [cr.to_dict() for cr in crashes],
            separators=(",", ":"),
        )
        raw = json_str.encode("utf-8")
        return zlib.compress(raw, self.compression_level)

    def decompress_crash_records(self, data: bytes) -> List[CrashRecord]:
        """Decompress crash records from bytes."""
        raw = zlib.decompress(data)
        json_str = raw.decode("utf-8")
        crash_dicts = json.loads(json_str)
        return [CrashRecord(
            timestamp=d["timestamp"],
            severity=CrashSeverity[d["severity"]],
            description=d["description"],
            pre_crash_data=[JointState.from_dict(js) for js in d.get("pre_crash_data", [])],
            post_crash_data=[JointState.from_dict(js) for js in d.get("post_crash_data", [])],
            related_events=[LogEvent.from_dict(ev) for ev in d.get("related_events", [])],
            joint_states_at_crash=(
                JointState.from_dict(d["joint_states_at_crash"])
                if d.get("joint_states_at_crash") else None
            ),
            estimated_cause=d.get("estimated_cause", ""),
        ) for d in crash_dicts]

    def compute_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Compute compression ratio."""
        if original_size == 0:
            return 0.0
        return compressed_size / original_size


class PlaybackAnalyzer:
    """Analyzes recorded black box data for playback and diagnostics."""

    def __init__(self) -> None:
        self._compressor = DataCompressor()

    def compute_trajectory_statistics(
        self, states: List[JointState]
    ) -> Dict[str, Any]:
        """Compute statistics about a joint trajectory."""
        if not states:
            return {"error": "No data"}

        n_joints = len(states[0].positions) if states[0].positions else 0
        stats: Dict[str, Any] = {
            "num_samples": len(states),
            "num_joints": n_joints,
            "duration": states[-1].timestamp - states[0].timestamp,
            "start_time": states[0].timestamp,
            "end_time": states[-1].timestamp,
        }

        for j in range(n_joints):
            positions = [s.positions[j] for s in states if j < len(s.positions)]
            if not positions:
                continue

            total_distance = sum(
                abs(positions[i + 1] - positions[i])
                for i in range(len(positions) - 1)
            )
            max_velocity = 0.0
            for s in states:
                if s.velocities and j < len(s.velocities):
                    max_velocity = max(max_velocity, abs(s.velocities[j]))

            stats[f"joint_{j}"] = {
                "min_pos": min(positions),
                "max_pos": max(positions),
                "range": max(positions) - min(positions),
                "total_distance": total_distance,
                "max_velocity": max_velocity,
            }

        return stats

    def detect_anomalies(
        self,
        states: List[JointState],
        velocity_threshold: float = 10.0,
        acceleration_threshold: float = 100.0,
        torque_threshold: float = 200.0,
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in recorded joint data."""
        anomalies: List[Dict[str, Any]] = []

        for i, state in enumerate(states):
            if state.velocities:
                for j, vel in enumerate(state.velocities):
                    if abs(vel) > velocity_threshold:
                        anomalies.append({
                            "sample_index": i,
                            "timestamp": state.timestamp,
                            "joint": j,
                            "type": "high_velocity",
                            "value": vel,
                            "threshold": velocity_threshold,
                        })

            if state.accelerations:
                for j, acc in enumerate(state.accelerations):
                    if abs(acc) > acceleration_threshold:
                        anomalies.append({
                            "sample_index": i,
                            "timestamp": state.timestamp,
                            "joint": j,
                            "type": "high_acceleration",
                            "value": acc,
                            "threshold": acceleration_threshold,
                        })

            if state.torques:
                for j, tq in enumerate(state.torques):
                    if abs(tq) > torque_threshold:
                        anomalies.append({
                            "sample_index": i,
                            "timestamp": state.timestamp,
                            "joint": j,
                            "type": "high_torque",
                            "value": tq,
                            "threshold": torque_threshold,
                        })

        return anomalies

    def extract_timeline(
        self,
        events: List[LogEvent],
        crashes: Optional[List[CrashRecord]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract a unified timeline from events and crashes."""
        timeline: List[Dict[str, Any]] = []

        for ev in events:
            timeline.append({
                "timestamp": ev.timestamp,
                "type": "event",
                "event_type": ev.event_type.name,
                "level": ev.level.name,
                "message": ev.message,
                "data": ev.data,
            })

        if crashes:
            for cr in crashes:
                timeline.append({
                    "timestamp": cr.timestamp,
                    "type": "crash",
                    "severity": cr.severity.name,
                    "description": cr.description,
                    "cause": cr.estimated_cause,
                })

        timeline.sort(key=lambda x: x["timestamp"])
        return timeline

    def compute_duty_cycle(
        self,
        states: List[JointState],
        window_seconds: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """Compute duty cycle over time windows."""
        if not states:
            return []

        start_time = states[0].timestamp
        end_time = states[-1].timestamp
        windows: List[Dict[str, Any]] = []

        window_start = start_time
        while window_start < end_time:
            window_end = window_start + window_seconds
            window_states = [
                s for s in states
                if window_start <= s.timestamp < window_end
            ]

            if window_states:
                n_joints = len(window_states[0].positions) if window_states[0].positions else 0
                movement: Dict[int, float] = {}
                for j in range(n_joints):
                    positions = [s.positions[j] for s in window_states if j < len(s.positions)]
                    total = sum(
                        abs(positions[i + 1] - positions[i])
                        for i in range(len(positions) - 1)
                    )
                    movement[j] = total

                windows.append({
                    "start_time": window_start,
                    "end_time": window_end,
                    "num_samples": len(window_states),
                    "joint_movement": movement,
                })

            window_start = window_end

        return windows

    def generate_summary_report(
        self,
        joint_logger: JointDataLogger,
        event_recorder: EventRecorder,
        crash_detector: CrashDetector,
    ) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        states = joint_logger.get_all()
        events = event_recorder.get_recent(event_recorder.event_count)
        crashes = crash_detector.get_crashes()

        trajectory_stats = self.compute_trajectory_statistics(states)
        anomalies = self.detect_anomalies(states)
        timeline = self.extract_timeline(events, crashes)
        event_counts = event_recorder.get_event_counts_by_type()

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trajectory_statistics": trajectory_stats,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[:50],
            "crash_count": len(crashes),
            "crash_severities": [c.severity.name for c in crashes],
            "event_counts": event_counts,
            "total_events": len(events),
            "timeline_entries": len(timeline),
            "joint_logger_stats": joint_logger.get_statistics(),
        }


class BlackBoxRecorder:
    """Main black box recorder for robots.

    Integrates joint data logging, event recording, crash detection,
    data compression, and playback analysis into a unified system.
    """

    def __init__(
        self,
        robot_name: str = "default_robot",
        num_joints: int = 6,
        joint_buffer_size: int = 10000,
        event_buffer_size: int = 5000,
        sample_rate_hz: float = 100.0,
        compression_level: int = 6,
    ) -> None:
        self.robot_name = robot_name
        self.joint_logger = JointDataLogger(
            num_joints=num_joints,
            buffer_size=joint_buffer_size,
            sample_rate_hz=sample_rate_hz,
        )
        self.event_recorder = EventRecorder(buffer_size=event_buffer_size)
        self.crash_detector = CrashDetector()
        self.compressor = DataCompressor(compression_level=compression_level)
        self.analyzer = PlaybackAnalyzer()
        self._recording = False
        self._lock = threading.Lock()
        self._session_start: Optional[float] = None

    def start_recording(self) -> None:
        """Start recording session."""
        with self._lock:
            self._recording = True
            self._session_start = time.time()
            self.event_recorder.record_event(
                EventType.PROGRAM_START,
                f"Recording started for {self.robot_name}",
                LogLevel.INFO,
            )

    def stop_recording(self) -> None:
        """Stop recording session."""
        with self._lock:
            if self._recording:
                self._recording = False
                self.event_recorder.record_event(
                    EventType.PROGRAM_END,
                    f"Recording stopped for {self.robot_name}",
                    LogLevel.INFO,
                )

    def is_recording(self) -> bool:
        return self._recording

    def log_joint_state(self, state: JointState) -> Optional[CrashRecord]:
        """Log a joint state and check for crashes."""
        if not self._recording:
            return None

        self.joint_logger.record(state)
        crash = self.crash_detector.analyze_joint_state(
            state, self.joint_logger, self.event_recorder
        )
        return crash

    def log_event(
        self,
        event_type: EventType,
        message: str = "",
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict[str, Any]] = None,
    ) -> LogEvent:
        """Log an event."""
        event = self.event_recorder.record_event(
            event_type, message, level, data, self.robot_name
        )
        crash = self.crash_detector.check_event_crash(event, self.joint_logger)
        if crash is not None:
            self.event_recorder.record_event(
                EventType.FAULT_DETECTED,
                f"Crash detected: {crash.description}",
                LogLevel.CRITICAL,
                {"severity": crash.severity.name, "cause": crash.estimated_cause},
            )
        return event

    def export_data(self) -> Dict[str, bytes]:
        """Export all recorded data as compressed bytes."""
        states = self.joint_logger.get_all()
        events = self.event_recorder.get_recent(self.event_recorder.event_count)
        crashes = self.crash_detector.get_crashes()

        return {
            "joint_states": self.compressor.compress_joint_states(states),
            "events": self.compressor.compress_events(events),
            "crashes": self.compressor.compress_crash_records(crashes),
            "metadata": json.dumps({
                "robot_name": self.robot_name,
                "export_time": datetime.now(timezone.utc).isoformat(),
                "num_joint_states": len(states),
                "num_events": len(events),
                "num_crashes": len(crashes),
            }).encode("utf-8"),
        }

    def import_data(self, data: Dict[str, bytes]) -> bool:
        """Import previously exported data."""
        try:
            if "joint_states" in data:
                states = self.compressor.decompress_joint_states(data["joint_states"])
                for state in states:
                    self.joint_logger.record(state)

            if "events" in data:
                events = self.compressor.decompress_events(data["events"])
                for event in events:
                    self.event_recorder.record_event(
                        event.event_type,
                        event.message,
                        event.level,
                        event.data,
                        event.source,
                    )

            if "crashes" in data:
                crashes = self.compressor.decompress_crash_records(data["crashes"])
                with self.crash_detector._lock:
                    self.crash_detector._crashes.extend(crashes)

            return True
        except Exception:
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        return self.analyzer.generate_summary_report(
            self.joint_logger, self.event_recorder, self.crash_detector
        )

    def clear(self) -> None:
        """Clear all recorded data."""
        self.joint_logger.clear()
        self.event_recorder.clear()
        self.crash_detector.clear()

    def get_session_duration(self) -> float:
        """Get the duration of the current recording session."""
        if self._session_start is None:
            return 0.0
        return time.time() - self._session_start
