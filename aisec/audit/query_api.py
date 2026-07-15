"""Audit Query API Module - Time-range queries, filtering, full-text search, aggregation, pagination, export."""

from __future__ import annotations
import re, time, uuid, math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class QueryResult:
    results: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    total_pages: int = 0
    query_time_ms: float = 0.0
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {"total_count": self.total_count, "page": self.page, "page_size": self.page_size,
                "total_pages": self.total_pages, "query_time_ms": self.query_time_ms,
                "results": self.results, "facets": self.facets}

class TimeRangeQuery:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries
    def query(self, start_time: Optional[float] = None, end_time: Optional[float] = None,
              field: str = "timestamp") -> List[Dict[str, Any]]:
        results = self._entries
        if start_time is not None:
            results = [e for e in results if e.get(field, 0) >= start_time]
        if end_time is not None:
            results = [e for e in results if e.get(field, 0) <= end_time]
        return sorted(results, key=lambda e: e.get(field, 0), reverse=True)

class UserFilter:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries
    def filter(self, user_id: Optional[str] = None, users: Optional[List[str]] = None,
               field: str = "actor") -> List[Dict[str, Any]]:
        results = self._entries
        if user_id:
            results = [e for e in results if e.get(field) == user_id]
        if users:
            results = [e for e in results if e.get(field) in users]
        return results

class OperationFilter:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries
    def filter(self, operation: Optional[str] = None, operations: Optional[List[str]] = None,
               field: str = "action") -> List[Dict[str, Any]]:
        results = self._entries
        if operation:
            results = [e for e in results if e.get(field) == operation]
        if operations:
            results = [e for e in results if e.get(field) in operations]
        return results

class FullTextSearch:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries
        self._index: Dict[str, List[int]] = defaultdict(list)
        self._build_index()
    def _build_index(self) -> None:
        for i, entry in enumerate(self._entries):
            text = json.dumps(entry, default=str).lower()
            tokens = set(re.findall(r'\b\w+\b', text))
            for token in tokens:
                self._index[token].append(i)
    def search(self, query: str, fields: Optional[List[str]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        if not query_tokens:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for token in query_tokens:
            for idx in self._index.get(token, []):
                scores[idx] += 1.0
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for idx, score in ranked[:limit]:
            entry = self._entries[idx]
            entry_copy = dict(entry)
            entry_copy["_search_score"] = score
            results.append(entry_copy)
        return results
    def search_exact(self, phrase: str, limit: int = 100) -> List[Dict[str, Any]]:
        phrase_lower = phrase.lower()
        results = []
        for entry in self._entries:
            text = json.dumps(entry, default=str).lower()
            if phrase_lower in text:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

class Aggregator:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries
    def count_by(self, field: str) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in self._entries:
            val = str(e.get(field, "unknown"))
            counts[val] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
    def sum_by(self, field: str, value_field: str) -> Dict[str, float]:
        sums: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for e in self._entries:
            key = str(e.get(field, "unknown"))
            try:
                sums[key] += float(e.get(value_field, 0))
                counts[key] += 1
            except (TypeError, ValueError):
                pass
        return {k: v for k, v in sums.items()}
    def avg_by(self, field: str, value_field: str) -> Dict[str, float]:
        sums = self.sum_by(field, value_field)
        counts = self.count_by(field)
        return {k: sums.get(k, 0) / counts.get(k, 1) for k in counts}
    def time_series(self, field: str, interval_seconds: float = 3600) -> List[Dict[str, Any]]:
        if not self._entries:
            return []
        timestamps = [e.get("timestamp", 0) for e in self._entries]
        if not timestamps:
            return []
        min_t = min(timestamps)
        max_t = max(timestamps)
        buckets: Dict[int, int] = defaultdict(int)
        for e in self._entries:
            t = e.get("timestamp", 0)
            bucket = int((t - min_t) / interval_seconds)
            buckets[bucket] += 1
        series = []
        for bucket_idx in range(max(buckets.keys()) + 1):
            series.append({
                "time": min_t + bucket_idx * interval_seconds,
                "count": buckets.get(bucket_idx, 0),
            })
        return series
    def top_n(self, field: str, n: int = 10) -> List[Tuple[str, int]]:
        counts = self.count_by(field)
        return list(counts.items())[:n]

class Paginator:
    def __init__(self, entries: List[Dict[str, Any]], default_page_size: int = 50):
        self._entries = entries
        self.default_page_size = default_page_size
    def paginate(self, page: int = 1, page_size: Optional[int] = None) -> QueryResult:
        ps = page_size or self.default_page_size
        total = len(self._entries)
        total_pages = math.ceil(total / ps) if total > 0 else 1
        page = max(1, min(page, total_pages))
        start = (page - 1) * ps
        end = start + ps
        return QueryResult(
            results=self._entries[start:end], total_count=total,
            page=page, page_size=ps, total_pages=total_pages,
        )

class AuditQueryAPI:
    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None):
        self._entries: List[Dict[str, Any]] = entries or []
        self._time_query = TimeRangeQuery(self._entries)
        self._user_filter = UserFilter(self._entries)
        self._op_filter = OperationFilter(self._entries)
        self._search = FullTextSearch(self._entries)
        self._aggregator = Aggregator(self._entries)
    def load_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = entries
        self._time_query = TimeRangeQuery(self._entries)
        self._user_filter = UserFilter(self._entries)
        self._op_filter = OperationFilter(self._entries)
        self._search = FullTextSearch(self._entries)
        self._aggregator = Aggregator(self._entries)
    def query(self, start_time: Optional[float] = None, end_time: Optional[float] = None,
              user_id: Optional[str] = None, operation: Optional[str] = None,
              search_query: Optional[str] = None, severity: Optional[str] = None,
              page: int = 1, page_size: int = 50) -> QueryResult:
        start = time.time()
        results = self._entries
        if start_time is not None or end_time is not None:
            results = self._time_query.query(start_time, end_time)
        if user_id:
            results = self._user_filter.filter(user_id=user_id)
        if operation:
            results = self._op_filter.filter(operation=operation)
        if severity:
            results = [e for e in results if e.get("severity") == severity]
        if search_query:
            results = self._search.search(search_query)
        paginator = Paginator(results)
        result = paginator.paginate(page, page_size)
        result.facets = {
            "actors": dict(self._aggregator.count_by("actor").most_common(20)),
            "actions": dict(self._aggregator.count_by("action").most_common(20)),
            "severities": self._aggregator.count_by("severity"),
        }
        result.query_time_ms = (time.time() - start) * 1000
        return result
    def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        return self._search.search(query, limit=limit)
    def aggregate(self, group_by: str, metric: str = "count") -> Dict[str, Any]:
        if metric == "count":
            return self._aggregator.count_by(group_by)
        return self._aggregator.sum_by(group_by, metric)
    def time_series(self, interval_seconds: float = 3600) -> List[Dict[str, Any]]:
        return self._aggregator.time_series("timestamp", interval_seconds)
    def get_user_activity(self, user_id: str) -> Dict[str, Any]:
        user_entries = self._user_filter.filter(user_id=user_id)
        return {
            "user_id": user_id,
            "total_actions": len(user_entries),
            "actions_by_type": self._aggregator.__class__(user_entries).count_by("action"),
            "first_activity": min((e.get("timestamp", 0) for e in user_entries), default=0),
            "last_activity": max((e.get("timestamp", 0) for e in user_entries), default=0),
        }
