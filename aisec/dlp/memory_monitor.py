"""
Process Memory Monitoring Module

Heap scanning, string extraction, pattern matching in memory regions,
memory dump analysis, leak detection, and cleanup triggers.
"""

from __future__ import annotations

import hashlib
import re
import struct
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class MemoryRegion(Enum):
    """Types of memory regions."""
    HEAP = "heap"
    STACK = "stack"
    CODE = "code"
    DATA = "data"
    SHARED = "shared"
    MAPPED = "mapped"
    UNKNOWN = "unknown"


class LeakSeverity(Enum):
    """Severity of memory leaks."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryRegionInfo:
    """Information about a memory region."""
    region_id: str
    start_address: int
    end_address: int
    size: int
    region_type: MemoryRegion
    permissions: str = "rw-"
    process_id: int = 0
    label: str = ""

    @property
    def address_range(self) -> str:
        return f"0x{self.start_address:016x}-0x{self.end_address:016x}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "address_range": self.address_range,
            "size": self.size,
            "region_type": self.region_type.value,
            "permissions": self.permissions,
            "process_id": self.process_id,
            "label": self.label,
        }


@dataclass
class ExtractedString:
    """A string extracted from memory."""
    string_id: str
    value: str
    offset: int
    length: int
    region_id: str
    region_type: MemoryRegion
    is_printable: bool = True
    encoding: str = "utf-8"
    entropy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "string_id": self.string_id,
            "value_preview": self.value[:100],
            "offset": f"0x{self.offset:08x}",
            "length": self.length,
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "encoding": self.encoding,
            "entropy": round(self.entropy, 2),
        }


@dataclass
class PatternMatch:
    """A pattern match found in memory."""
    match_id: str
    pattern_name: str
    matched_text: str
    offset: int
    region_id: str
    region_type: MemoryRegion
    severity: LeakSeverity = LeakSeverity.MEDIUM
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "pattern_name": self.pattern_name,
            "matched_text_preview": self.matched_text[:100],
            "offset": f"0x{self.offset:08x}",
            "region_id": self.region_id,
            "region_type": self.region_type.value,
            "severity": self.severity.value,
        }


@dataclass
class LeakReport:
    """Report of detected memory leaks."""
    report_id: str
    timestamp: float = field(default_factory=time.time)
    leaks: List[Dict[str, Any]] = field(default_factory=list)
    total_leaked_bytes: int = 0
    severity: LeakSeverity = LeakSeverity.NONE
    recommendations: List[str] = field(default_factory=list)

    def add_leak(self, leak: Dict[str, Any]) -> None:
        self.leaks.append(leak)
        self.total_leaked_bytes += leak.get("size", 0)
        self._update_severity()

    def _update_severity(self) -> None:
        if self.total_leaked_bytes > 100 * 1024 * 1024:
            self.severity = LeakSeverity.CRITICAL
        elif self.total_leaked_bytes > 10 * 1024 * 1024:
            self.severity = LeakSeverity.HIGH
        elif self.total_leaked_bytes > 1024 * 1024:
            self.severity = LeakSeverity.MEDIUM
        elif self.total_leaked_bytes > 1024:
            self.severity = LeakSeverity.LOW
        else:
            self.severity = LeakSeverity.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "leak_count": len(self.leaks),
            "total_leaked_bytes": self.total_leaked_bytes,
            "severity": self.severity.value,
            "recommendations": self.recommendations,
        }


class HeapScanner:
    """Scans heap memory regions for sensitive data."""

    def __init__(self) -> None:
        self._regions: Dict[str, MemoryRegionInfo] = {}
        self._scan_results: List[Dict[str, Any]] = []
        self._max_results: int = 5000
        self._min_string_length: int = 4
        self._max_string_length: int = 4096

    def register_region(self, region: MemoryRegionInfo) -> None:
        self._regions[region.region_id] = region

    def unregister_region(self, region_id: str) -> Optional[MemoryRegionInfo]:
        return self._regions.pop(region_id, None)

    def scan_region(
        self, region_id: str, data: bytes, label: str = ""
    ) -> List[ExtractedString]:
        region = self._regions.get(region_id)
        if region is None:
            return []
        strings = self._extract_strings(data, region_id, region.region_type)
        result = {
            "region_id": region_id,
            "strings_found": len(strings),
            "data_size": len(data),
            "timestamp": time.time(),
            "label": label,
        }
        self._scan_results.append(result)
        if len(self._scan_results) > self._max_results:
            self._scan_results = self._scan_results[-self._max_results:]
        return strings

    def scan_all_regions(
        self, data_map: Dict[str, bytes]
    ) -> Dict[str, List[ExtractedString]]:
        results: Dict[str, List[ExtractedString]] = {}
        for region_id, data in data_map.items():
            results[region_id] = self.scan_region(region_id, data)
        return results

    def _extract_strings(
        self, data: bytes, region_id: str, region_type: MemoryRegion
    ) -> List[ExtractedString]:
        strings: List[ExtractedString] = []
        current_string: List[int] = []
        start_offset: int = 0
        for i, byte in enumerate(data):
            if 32 <= byte <= 126 or byte in (9, 10, 13):
                if not current_string:
                    start_offset = i
                current_string.append(byte)
            else:
                if len(current_string) >= self._min_string_length:
                    raw = bytes(current_string)
                    try:
                        value = raw.decode("utf-8", errors="replace")
                    except Exception:
                        value = raw.decode("latin-1")
                    if len(value) <= self._max_string_length:
                        entropy = self._compute_entropy(raw)
                        strings.append(ExtractedString(
                            string_id=uuid.uuid4().hex[:12],
                            value=value,
                            offset=start_offset,
                            length=len(value),
                            region_id=region_id,
                            region_type=region_type,
                            entropy=entropy,
                        ))
                current_string = []
        if len(current_string) >= self._min_string_length:
            raw = bytes(current_string)
            try:
                value = raw.decode("utf-8", errors="replace")
            except Exception:
                value = raw.decode("latin-1")
            if len(value) <= self._max_string_length:
                entropy = self._compute_entropy(raw)
                strings.append(ExtractedString(
                    string_id=uuid.uuid4().hex[:12],
                    value=value,
                    offset=start_offset,
                    length=len(value),
                    region_id=region_id,
                    region_type=region_type,
                    entropy=entropy,
                ))
        return strings

    @staticmethod
    def _compute_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        freq: Dict[int, int] = Counter(data)
        length = len(data)
        entropy = 0.0
        import math
        for count in freq.values():
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)
        return entropy

    def get_regions(self) -> List[MemoryRegionInfo]:
        return list(self._regions.values())

    def get_scan_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._scan_results[-limit:]


class StringExtractor:
    """Extracts and classifies strings from memory data."""

    def __init__(self) -> None:
        self._classification_rules: List[Tuple[str, re.Pattern, str]] = [
            ("email", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), "sensitive"),
            ("ip_address", re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), "network"),
            ("url", re.compile(r'https?://[^\s<>"]+'), "network"),
            ("file_path", re.compile(r'(?:/[\w/.-]+|\w:\\[\w\\.-]+)'), "filesystem"),
            ("api_key", re.compile(r'(?:api[_-]?key|apikey)\s*[:=]\s*\S{8,}', re.I), "credential"),
            ("password", re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*\S+', re.I), "credential"),
            ("token", re.compile(r'(?:token|bearer)\s*[:=]\s*\S{8,}', re.I), "credential"),
            ("sql_query", re.compile(r'\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)\b.*\b(?:FROM|INTO|TABLE)\b', re.I), "query"),
            ("json_data", re.compile(r'\{[^{}]*"[^"]+"\s*:\s*[^{}]*\}'), "structured"),
            ("uuid", re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), "identifier"),
        ]
        self._sensitive_keywords: Set[str] = {
            "password", "secret", "token", "api_key", "private_key",
            "credential", "auth", "session", "cookie",
        }

    def extract_and_classify(
        self, strings: List[ExtractedString]
    ) -> Dict[str, List[ExtractedString]]:
        classified: Dict[str, List[ExtractedString]] = defaultdict(list)
        for s in strings:
            categories = self._classify_string(s.value)
            for cat in categories:
                classified[cat].append(s)
        return dict(classified)

    def _classify_string(self, value: str) -> List[str]:
        categories: List[str] = ["unclassified"]
        value_lower = value.lower()
        for name, pattern, category in self._classification_rules:
            if pattern.search(value):
                if categories[0] == "unclassified":
                    categories[0] = category
                else:
                    categories.append(category)
        if any(kw in value_lower for kw in self._sensitive_keywords):
            if "sensitive" not in categories:
                categories.append("sensitive")
        return categories

    def find_sensitive_strings(
        self, strings: List[ExtractedString]
    ) -> List[ExtractedString]:
        sensitive: List[ExtractedString] = []
        classified = self.extract_and_classify(strings)
        for cat in ["sensitive", "credential"]:
            if cat in classified:
                sensitive.extend(classified[cat])
        return sensitive

    def extract_structured_data(
        self, strings: List[ExtractedString]
    ) -> List[Dict[str, Any]]:
        structured: List[Dict[str, Any]] = []
        for s in strings:
            try:
                import json
                data = json.loads(s.value)
                if isinstance(data, dict):
                    structured.append({
                        "string_id": s.string_id,
                        "offset": s.offset,
                        "data": data,
                        "keys": list(data.keys()),
                    })
            except (json.JSONDecodeError, ValueError):
                pass
        return structured


class PatternMatcher:
    """Matches patterns in memory-extracted strings."""

    def __init__(self) -> None:
        self._patterns: List[Tuple[str, re.Pattern, LeakSeverity]] = [
            ("aws_key", re.compile(r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}'), LeakSeverity.CRITICAL),
            ("private_key", re.compile(r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----'), LeakSeverity.CRITICAL),
            ("connection_string", re.compile(r'(?:mongodb|postgres|mysql|redis)://[^\s]+'), LeakSeverity.HIGH),
            ("jwt_token", re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'), LeakSeverity.HIGH),
            ("password_assignment", re.compile(r'(?:password|passwd|pwd)\s*[:=]\s*\S+', re.I), LeakSeverity.HIGH),
            ("secret_assignment", re.compile(r'(?:secret|token|api_key)\s*[:=]\s*\S{8,}', re.I), LeakSeverity.HIGH),
            ("ssn", re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), LeakSeverity.CRITICAL),
            ("credit_card", re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), LeakSeverity.CRITICAL),
            ("html_content", re.compile(r'<(?:html|head|body|div|script)\b', re.I), LeakSeverity.LOW),
            ("sql_injection", re.compile(r"(?:UNION\s+SELECT|DROP\s+TABLE|';\s*--)", re.I), LeakSeverity.HIGH),
        ]
        self._custom_patterns: List[Tuple[str, re.Pattern, LeakSeverity]] = []

    def add_pattern(
        self, name: str, pattern: str, severity: LeakSeverity = LeakSeverity.MEDIUM
    ) -> None:
        compiled = re.compile(pattern)
        self._custom_patterns.append((name, compiled, severity))

    def match_strings(
        self, strings: List[ExtractedString]
    ) -> List[PatternMatch]:
        matches: List[PatternMatch] = []
        all_patterns = self._patterns + self._custom_patterns
        for s in strings:
            for name, pattern, severity in all_patterns:
                m = pattern.search(s.value)
                if m:
                    start = max(0, m.start() - 20)
                    end = min(len(s.value), m.end() + 20)
                    matches.append(PatternMatch(
                        match_id=uuid.uuid4().hex[:12],
                        pattern_name=name,
                        matched_text=m.group(0),
                        offset=s.offset + m.start(),
                        region_id=s.region_id,
                        region_type=s.region_type,
                        severity=severity,
                        context_before=s.value[start:m.start()],
                        context_after=s.value[m.end():end],
                    ))
        return matches

    def match_data(self, data: bytes, region_id: str = "", region_type: MemoryRegion = MemoryRegion.UNKNOWN) -> List[PatternMatch]:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        strings = [ExtractedString(
            string_id=uuid.uuid4().hex[:12],
            value=text,
            offset=0,
            length=len(text),
            region_id=region_id,
            region_type=region_type,
        )]
        return self.match_strings(strings)


class MemoryDumpAnalyzer:
    """Analyzes memory dumps for security issues."""

    def __init__(self) -> None:
        self._heap_scanner = HeapScanner()
        self._string_extractor = StringExtractor()
        self._pattern_matcher = PatternMatcher()
        self._analysis_history: List[Dict[str, Any]] = []

    def analyze_dump(
        self,
        dump_data: bytes,
        region_id: str = "dump_0",
        region_type: MemoryRegion = MemoryRegion.HEAP,
    ) -> Dict[str, Any]:
        start_time = time.time()
        region = MemoryRegionInfo(
            region_id=region_id,
            start_address=0,
            end_address=len(dump_data),
            size=len(dump_data),
            region_type=region_type,
        )
        self._heap_scanner.register_region(region)
        strings = self._heap_scanner.scan_region(region_id, dump_data)
        classified = self._string_extractor.extract_and_classify(strings)
        sensitive = self._string_extractor.find_sensitive_strings(strings)
        pattern_matches = self._pattern_matcher.match_strings(strings)
        structured = self._string_extractor.extract_structured_data(strings)
        high_entropy_strings = [s for s in strings if s.entropy > 6.0]
        duration_ms = (time.time() - start_time) * 1000
        analysis = {
            "dump_size": len(dump_data),
            "strings_extracted": len(strings),
            "classified_strings": {k: len(v) for k, v in classified.items()},
            "sensitive_strings": len(sensitive),
            "pattern_matches": len(pattern_matches),
            "structured_data_found": len(structured),
            "high_entropy_strings": len(high_entropy_strings),
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
        self._analysis_history.append(analysis)
        return analysis

    def compare_dumps(
        self, dump1: bytes, dump2: bytes
    ) -> Dict[str, Any]:
        hash1 = hashlib.sha256(dump1).hexdigest()
        hash2 = hashlib.sha256(dump2).hexdigest()
        if hash1 == hash2:
            return {"identical": True, "difference_count": 0}
        diff_count = sum(1 for a, b in zip(dump1, dump2) if a != b)
        common_prefix = 0
        for a, b in zip(dump1, dump2):
            if a == b:
                common_prefix += 1
            else:
                break
        strings1 = set(self._heap_scanner._extract_strings(dump1, "dump1", MemoryRegion.HEAP))
        strings2 = set(self._heap_scanner._extract_strings(dump2, "dump2", MemoryRegion.HEAP))
        vals1 = {s.value for s in strings1}
        vals2 = {s.value for s in strings2}
        new_strings = vals2 - vals1
        removed_strings = vals1 - vals2
        return {
            "identical": False,
            "byte_differences": diff_count,
            "common_prefix_bytes": common_prefix,
            "new_strings": len(new_strings),
            "removed_strings": len(removed_strings),
            "common_strings": len(vals1 & vals2),
        }


class LeakDetector:
    """Detects memory leaks and sensitive data accumulation."""

    def __init__(
        self,
        check_interval: float = 60.0,
        high_watermark: int = 100 * 1024 * 1024,
        growth_threshold: float = 1.5,
    ) -> None:
        self._check_interval: float = check_interval
        self._high_watermark: int = high_watermark
        self._growth_threshold: float = growth_threshold
        self._snapshots: List[Dict[str, Any]] = []
        self._max_snapshots: int = 100
        self._object_tracker: Dict[str, List[Dict[str, Any]]] = {}
        self._leak_thresholds: Dict[str, int] = {
            "string_accumulation": 1000,
            "large_objects": 100,
            "sensitive_data": 10,
        }

    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        snapshot = {
            "snapshot_id": uuid.uuid4().hex[:12],
            "timestamp": time.time(),
            "label": label,
            "tracked_objects": sum(len(v) for v in self._object_tracker.values()),
            "total_tracked_size": sum(
                obj.get("size", 0)
                for objs in self._object_tracker.values()
                for obj in objs
            ),
        }
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]
        return snapshot

    def track_object(
        self, category: str, obj_id: str, size: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if category not in self._object_tracker:
            self._object_tracker[category] = []
        self._object_tracker[category].append({
            "obj_id": obj_id,
            "size": size,
            "created_at": time.time(),
            "metadata": metadata or {},
        })

    def release_object(self, category: str, obj_id: str) -> bool:
        if category not in self._object_tracker:
            return False
        for i, obj in enumerate(self._object_tracker[category]):
            if obj["obj_id"] == obj_id:
                self._object_tracker[category].pop(i)
                return True
        return False

    def detect_leaks(self) -> LeakReport:
        report = LeakReport(report_id=uuid.uuid4().hex[:12])
        for category, objects in self._object_tracker.items():
            total_size = sum(obj["size"] for obj in objects)
            if total_size > 1024:
                report.add_leak({
                    "category": category,
                    "object_count": len(objects),
                    "total_size": total_size,
                    "severity": self._assess_leak_severity(category, len(objects), total_size),
                })
        if len(self._snapshots) >= 2:
            latest = self._snapshots[-1]
            previous = self._snapshots[-2]
            if previous["total_tracked_size"] > 0:
                growth = latest["total_tracked_size"] / previous["total_tracked_size"]
                if growth > self._growth_threshold:
                    report.add_leak({
                        "category": "memory_growth",
                        "object_count": latest["tracked_objects"],
                        "total_size": latest["total_tracked_size"],
                        "growth_ratio": growth,
                        "severity": LeakSeverity.HIGH if growth > 2.0 else LeakSeverity.MEDIUM,
                    })
        report.recommendations = self._generate_recommendations(report)
        return report

    def _assess_leak_severity(self, category: str, count: int, size: int) -> LeakSeverity:
        threshold = self._leak_thresholds.get(category, 100)
        if count > threshold * 10:
            return LeakSeverity.CRITICAL
        elif count > threshold * 5:
            return LeakSeverity.HIGH
        elif count > threshold:
            return LeakSeverity.MEDIUM
        elif count > threshold // 2:
            return LeakSeverity.LOW
        return LeakSeverity.NONE

    def _generate_recommendations(self, report: LeakReport) -> List[str]:
        recs: List[str] = []
        if report.total_leaked_bytes > 50 * 1024 * 1024:
            recs.append("Critical memory leak detected. Immediate investigation required.")
        for leak in report.leaks:
            cat = leak.get("category", "")
            if cat == "string_accumulation":
                recs.append("Large string accumulation detected. Consider string interning or pooling.")
            elif cat == "sensitive_data":
                recs.append("Sensitive data accumulating in memory. Implement secure cleanup.")
            elif cat == "memory_growth":
                recs.append("Rapid memory growth detected. Check for unintended object retention.")
        if not recs and report.leaks:
            recs.append("Minor memory usage patterns detected. Continue monitoring.")
        return recs

    def get_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._snapshots[-limit:]


class CleanupTrigger:
    """Triggers cleanup actions based on memory monitoring."""

    def __init__(self) -> None:
        self._triggers: List[Dict[str, Any]] = []
        self._cleanup_actions: Dict[str, Callable[[], int]] = {}
        self._trigger_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

    def add_trigger(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        action_name: str,
        cooldown: float = 300.0,
    ) -> None:
        self._triggers.append({
            "name": name,
            "condition": condition,
            "action_name": action_name,
            "cooldown": cooldown,
            "last_triggered": 0.0,
        })

    def register_cleanup_action(
        self, name: str, action: Callable[[], int]
    ) -> None:
        self._cleanup_actions[name] = action

    def evaluate(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        triggered: List[Dict[str, Any]] = []
        now = time.time()
        for trigger in self._triggers:
            if now - trigger["last_triggered"] < trigger["cooldown"]:
                continue
            try:
                if trigger["condition"](context):
                    action = self._cleanup_actions.get(trigger["action_name"])
                    result = 0
                    if action:
                        result = action()
                    trigger["last_triggered"] = now
                    record = {
                        "trigger_name": trigger["name"],
                        "action_name": trigger["action_name"],
                        "result": result,
                        "timestamp": now,
                    }
                    triggered.append(record)
                    self._trigger_history.append(record)
            except Exception:
                pass
        if len(self._trigger_history) > self._max_history:
            self._trigger_history = self._trigger_history[-self._max_history:]
        return triggered

    def get_trigger_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._trigger_history[-limit:]


class MemoryMonitor:
    """Main memory monitoring orchestrator."""

    def __init__(self) -> None:
        self.heap_scanner = HeapScanner()
        self.string_extractor = StringExtractor()
        self.pattern_matcher = PatternMatcher()
        self.dump_analyzer = MemoryDumpAnalyzer()
        self.leak_detector = LeakDetector()
        self.cleanup_trigger = CleanupTrigger()
        self._monitoring_active: bool = False
        self._scan_count: int = 0

    def scan_memory(
        self,
        data: bytes,
        region_id: str = "mem_0",
        region_type: MemoryRegion = MemoryRegion.HEAP,
    ) -> Dict[str, Any]:
        self._scan_count += 1
        region = MemoryRegionInfo(
            region_id=region_id,
            start_address=0,
            end_address=len(data),
            size=len(data),
            region_type=region_type,
        )
        self.heap_scanner.register_region(region)
        strings = self.heap_scanner.scan_region(region_id, data)
        classified = self.string_extractor.extract_and_classify(strings)
        sensitive = self.string_extractor.find_sensitive_strings(strings)
        pattern_matches = self.pattern_matcher.match_strings(strings)
        leak_report = self.leak_detector.detect_leaks()
        return {
            "scan_id": self._scan_count,
            "data_size": len(data),
            "strings_extracted": len(strings),
            "classified": {k: len(v) for k, v in classified.items()},
            "sensitive_count": len(sensitive),
            "pattern_matches": len(pattern_matches),
            "leak_report": leak_report.to_dict(),
        }

    def start_monitoring(self) -> None:
        self._monitoring_active = True

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "monitoring_active": self._monitoring_active,
            "total_scans": self._scan_count,
            "registered_regions": len(self.heap_scanner.get_regions()),
            "leak_report": self.leak_detector.detect_leaks().to_dict(),
        }
