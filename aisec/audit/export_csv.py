"""Audit Log Export Module - CSV/JSON/ELK format export, field selection, date formatting, anonymization, compression, scheduling."""

from __future__ import annotations
import csv, io, json, gzip, time, uuid, zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

class ExportFormat(Enum):
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    ELK = "elk"

@dataclass
class FieldSelector:
    included_fields: List[str] = field(default_factory=list)
    excluded_fields: List[str] = field(default_factory=list)
    def select(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        if self.included_fields:
            return {k: v for k, v in entry.items() if k in self.included_fields}
        result = dict(entry)
        for f in self.excluded_fields:
            result.pop(f, None)
        return result

class DateFormatter:
    def __init__(self, format_str: str = "%Y-%m-%d %H:%M:%S", timezone_offset: float = 0.0):
        self.format_str = format_str
        self.tz_offset = timezone_offset
    def format(self, timestamp: float) -> str:
        t = time.localtime(timestamp + self.tz_offset)
        return time.strftime(self.format_str, t)
    def format_entry(self, entry: Dict[str, Any], date_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        result = dict(entry)
        fields = date_fields or ["timestamp", "created_at", "updated_at"]
        for f in fields:
            if f in result and isinstance(result[f], (int, float)):
                result[f] = self.format(result[f])
        return result

class AnonymizationFilter:
    def __init__(self):
        self._sensitive_fields: Set[str] = {"password", "secret", "token", "api_key", "private_key"}
        self._anonymize_fields: Set[str] = set()
        self._hash_fields: Set[str] = set()
    def add_sensitive_field(self, field_name: str) -> None:
        self._sensitive_fields.add(field_name)
    def add_anonymize_field(self, field_name: str) -> None:
        self._anonymize_fields.add(field_name)
    def add_hash_field(self, field_name: str) -> None:
        self._hash_fields.add(field_name)
    def filter(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(entry)
        for f in self._sensitive_fields | self._anonymize_fields:
            if f in result:
                val = result[f]
                if isinstance(val, str):
                    result[f] = "***" if len(val) <= 4 else val[:2] + "***" + val[-2:]
                else:
                    result[f] = "[REDACTED]"
        for f in self._hash_fields:
            if f in result:
                import hashlib
                result[f] = hashlib.sha256(str(result[f]).encode()).hexdigest()[:16]
        return result

class CSVExporter:
    def __init__(self, field_selector: Optional[FieldSelector] = None,
                 date_formatter: Optional[DateFormatter] = None,
                 anonymizer: Optional[AnonymizationFilter] = None):
        self.field_selector = field_selector or FieldSelector()
        self.date_formatter = date_formatter or DateFormatter()
        self.anonymizer = anonymizer or AnonymizationFilter()
    def export(self, entries: List[Dict[str, Any]]) -> str:
        if not entries:
            return ""
        processed = self._process_entries(entries)
        output = io.StringIO()
        all_fields: List[str] = []
        for e in processed:
            for k in e.keys():
                if k not in all_fields:
                    all_fields.append(k)
        writer = csv.DictWriter(output, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for e in processed:
            row = {k: self._format_value(v) for k, v in e.items()}
            writer.writerow(row)
        return output.getvalue()
    def _process_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for e in entries:
            e = self.field_selector.select(e)
            e = self.date_formatter.format_entry(e)
            e = self.anonymizer.filter(e)
            results.append(e)
        return results
    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, dict):
            return json.dumps(value, default=str)
        if isinstance(value, list):
            return json.dumps(value, default=str)
        return str(value) if value is not None else ""

class JSONExporter:
    def __init__(self, field_selector: Optional[FieldSelector] = None,
                 date_formatter: Optional[DateFormatter] = None,
                 anonymizer: Optional[AnonymizationFilter] = None,
                 pretty: bool = True):
        self.field_selector = field_selector or FieldSelector()
        self.date_formatter = date_formatter or DateFormatter()
        self.anonymizer = anonymizer or AnonymizationFilter()
        self.pretty = pretty
    def export(self, entries: List[Dict[str, Any]]) -> str:
        processed = []
        for e in entries:
            e = self.field_selector.select(e)
            e = self.date_formatter.format_entry(e)
            e = self.anonymizer.filter(e)
            processed.append(e)
        indent = 2 if self.pretty else None
        return json.dumps(processed, indent=indent, default=str, ensure_ascii=False)

class ELKExporter:
    def __init__(self, index_name: str = "audit-logs", field_selector: Optional[FieldSelector] = None,
                 date_formatter: Optional[DateFormatter] = None):
        self.index_name = index_name
        self.field_selector = field_selector or FieldSelector()
        self.date_formatter = date_formatter or DateFormatter()
    def export(self, entries: List[Dict[str, Any]]) -> str:
        bulk_lines: List[str] = []
        for e in entries:
            e = self.field_selector.select(e)
            e = self.date_formatter.format_entry(e)
            action = {"index": {"_index": self.index_name}}
            bulk_lines.append(json.dumps(action))
            bulk_lines.append(json.dumps(e, default=str))
        return "\n".join(bulk_lines) + "\n"
    def export_single(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for e in entries:
            e = self.field_selector.select(e)
            e = self.date_formatter.format_entry(e)
            e["_index"] = self.index_name
            results.append(e)
        return results

class ExportScheduler:
    def __init__(self):
        self._schedules: Dict[str, Dict[str, Any]] = {}
    def add_schedule(self, name: str, cron_expression: str, format_type: ExportFormat,
                     output_path: str, query_params: Optional[Dict[str, Any]] = None) -> str:
        schedule_id = uuid.uuid4().hex[:12]
        self._schedules[schedule_id] = {
            "name": name, "cron_expression": cron_expression,
            "format": format_type.value, "output_path": output_path,
            "query_params": query_params or {}, "enabled": True,
            "last_run": 0, "created_at": time.time(),
        }
        return schedule_id
    def remove_schedule(self, schedule_id: str) -> bool:
        return self._schedules.pop(schedule_id, None) is not None
    def list_schedules(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._schedules.items()]
    def get_due_schedules(self) -> List[Dict[str, Any]]:
        return [s for s in self._schedules.values() if s["enabled"]]

class AuditExporter:
    def __init__(self):
        self.csv_exporter = CSVExporter()
        self.json_exporter = JSONExporter()
        self.elk_exporter = ELKExporter()
        self.scheduler = ExportScheduler()
    def export(self, entries: List[Dict[str, Any]], format_type: ExportFormat = ExportFormat.CSV,
               fields: Optional[List[str]] = None, exclude_fields: Optional[List[str]] = None,
               date_format: Optional[str] = None, anonymize: bool = False) -> str:
        selector = FieldSelector(included_fields=fields or [], excluded_fields=exclude_fields or [])
        date_fmt = DateFormatter(format_str=date_format) if date_format else None
        anonymizer = AnonymizationFilter() if anonymize else None
        if format_type == ExportFormat.CSV:
            exp = CSVExporter(selector, date_fmt, anonymizer)
        elif format_type == ExportFormat.JSON:
            exp = JSONExporter(selector, date_fmt, anonymizer)
        elif format_type == ExportFormat.ELK:
            exp = ELKExporter(field_selector=selector, date_formatter=date_fmt)
        else:
            exp = JSONExporter(selector, date_fmt, anonymizer)
        return exp.export(entries)
    def export_compressed(self, entries: List[Dict[str, Any]], format_type: ExportFormat = ExportFormat.JSON) -> bytes:
        data = self.export(entries, format_type)
        return gzip.compress(data.encode("utf-8"))
    def export_to_file(self, entries: List[Dict[str, Any]], filepath: str,
                       format_type: ExportFormat = ExportFormat.CSV, **kwargs) -> int:
        data = self.export(entries, format_type, **kwargs)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
        return len(entries)
