"""
ProfileHotspots - 性能热点分析

模块路径: testing/performance/profile_hotspots.py

提供代码性能分析功能，包括函数级耗时统计、调用频率分析、
内存使用监控和热点检测。
"""

import os
import sys
import json
import time
import tracemalloc
import cProfile
import pstats
import io
import threading
import functools
import inspect
import linecache
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import pytest


@dataclass
class FunctionProfile:
    """函数性能档案"""
    function_name: str
    module_name: str
    file_path: str
    line_number: int
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    own_time: float = 0.0
    cumulative_time: float = 0.0
    memory_allocated: int = 0

    @property
    def time_per_call(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count


@dataclass
class Hotspot:
    """性能热点"""
    function_name: str
    file_path: str
    line_number: int
    severity: str = "low"
    total_time: float = 0.0
    call_count: int = 0
    suggestion: str = ""


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    total_allocated: int = 0
    total_freed: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    allocation_count: int = 0
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProfileReport:
    """性能分析报告"""
    total_time: float = 0.0
    total_calls: int = 0
    function_profiles: Dict[str, FunctionProfile] = field(default_factory=dict)
    hotspots: List[Hotspot] = field(default_factory=list)
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_time": round(self.total_time, 4),
            "total_calls": self.total_calls,
            "hotspots": [
                {
                    "function": h.function_name,
                    "file": h.file_path,
                    "line": h.line_number,
                    "severity": h.severity,
                    "total_time": round(h.total_time, 4),
                    "call_count": h.call_count,
                    "suggestion": h.suggestion,
                }
                for h in self.hotspots
            ],
            "functions": {
                name: {
                    "calls": fp.call_count,
                    "total_time": round(fp.total_time, 4),
                    "avg_time": round(fp.avg_time, 4),
                    "own_time": round(fp.own_time, 4),
                }
                for name, fp in self.function_profiles.items()
            },
        }


class ProfileHotspots:
    """性能热点分析工具

    提供全面的代码性能分析功能:
        - 函数级耗时统计和排序
        - 调用频率分析
        - 内存分配追踪
        - 性能热点自动检测和分级
        - 调用图构建
        - 优化建议生成
        - 支持装饰器模式的按需分析
        - cProfile集成分析
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._profiles: Dict[str, FunctionProfile] = {}
        self._hotspots: List[Hotspot] = []
        self._memory_snapshots: List[MemorySnapshot] = []
        self._call_graph: Dict[str, List[str]] = defaultdict(list)
        self._call_stack: List[str] = []
        self._lock = threading.Lock()
        self._profiler: Optional[cProfile.Profile] = None
        self._tracemalloc_started = False
        self._hotspot_threshold: float = self.config.get("hotspot_threshold", 0.1)
        self._max_hotspots: int = self.config.get("max_hotspots", 20)
        self._track_memory: bool = self.config.get("track_memory", True)

    def initialize(self) -> None:
        """初始化性能分析器"""
        self._initialized = True

    def profile_function(self, func: Optional[Callable] = None,
                          name: str = "") -> Any:
        """函数性能分析装饰器

        Args:
            func: 被分析的函数
            name: 分析名称

        Returns:
            装饰器或包装函数
        """
        def decorator(fn: Callable) -> Callable:
            profile_name = name or fn.__name__

            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                return self._profiled_call(fn, profile_name, *args, **kwargs)

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return self._profiled_call(fn, profile_name, *args, **kwargs)

            if inspect.iscoroutinefunction(fn):
                return async_wrapper
            return sync_wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def _profiled_call(self, func: Callable, name: str, *args, **kwargs) -> Any:
        """执行被分析的函数调用

        Args:
            func: 被分析的函数
            name: 分析名称
            *args, **kwargs: 函数参数

        Returns:
            函数返回值
        """
        with self._lock:
            self._call_stack.append(name)

        start_time = time.perf_counter()
        mem_before = 0
        if self._track_memory and self._tracemalloc_started:
            mem_before = tracemalloc.get_traced_memory()[0]

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start_time
            mem_after = 0
            if self._track_memory and self._tracemalloc_started:
                mem_after = tracemalloc.get_traced_memory()[0]

            with self._lock:
                self._call_stack.pop()
                caller = self._call_stack[-1] if self._call_stack else "<root>"
                if caller not in self._call_graph:
                    self._call_graph[caller] = []
                if name not in self._call_graph[caller]:
                    self._call_graph[caller].append(name)

                if name not in self._profiles:
                    module = getattr(func, "__module__", "")
                    file_path = inspect.getfile(func) if hasattr(func, "__code__") else ""
                    line_no = func.__code__.co_firstlineno if hasattr(func, "__code__") else 0
                    self._profiles[name] = FunctionProfile(
                        function_name=name,
                        module_name=module,
                        file_path=file_path,
                        line_number=line_no,
                    )
                profile = self._profiles[name]
                profile.call_count += 1
                profile.total_time += elapsed
                profile.avg_time = profile.total_time / profile.call_count
                profile.min_time = min(profile.min_time, elapsed)
                profile.max_time = max(profile.max_time, elapsed)
                profile.own_time += elapsed
                profile.memory_allocated += max(0, mem_after - mem_before)

    def start_profiling(self) -> None:
        """启动全局性能分析"""
        if not self._initialized:
            self.initialize()
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        if self._track_memory:
            tracemalloc.start()
            self._tracemalloc_started = True

    def stop_profiling(self) -> ProfileReport:
        """停止全局性能分析并生成报告

        Returns:
            ProfileReport分析报告
        """
        if self._profiler:
            self._profiler.disable()
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
        self._detect_hotspots()
        return self.generate_report()

    def take_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """获取当前内存使用快照

        Args:
            label: 快照标签

        Returns:
            MemorySnapshot内存快照
        """
        snapshot = MemorySnapshot(timestamp=time.time())
        if self._tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            snapshot.current_usage = current
            snapshot.peak_usage = peak
            tracemalloc_snapshot = tracemalloc.take_snapshot()
            snapshot.allocation_count = len(tracemalloc_snapshot.statistics("lineno"))
            top_stats = tracemalloc_snapshot.statistics("lineno")[:10]
            snapshot.top_allocations = [
                {
                    "file": str(stat.traceback[0].filename) if stat.traceback else "",
                    "line": stat.traceback[0].lineno if stat.traceback else 0,
                    "size": stat.size,
                    "count": stat.count,
                }
                for stat in top_stats
            ]
        self._memory_snapshots.append(snapshot)
        return snapshot

    def _detect_hotspots(self) -> None:
        """自动检测性能热点"""
        if not self._profiles:
            return
        total_time = sum(p.total_time for p in self._profiles.values())
        if total_time == 0:
            return

        sorted_profiles = sorted(
            self._profiles.values(),
            key=lambda p: p.total_time,
            reverse=True,
        )

        self._hotspots = []
        for profile in sorted_profiles[:self._max_hotspots]:
            time_pct = (profile.total_time / total_time) * 100.0
            if time_pct >= self._hotspot_threshold * 100:
                severity = "critical"
            elif time_pct >= self._hotspot_threshold * 50:
                severity = "high"
            elif time_pct >= self._hotspot_threshold * 20:
                severity = "medium"
            else:
                severity = "low"

            suggestion = self._generate_suggestion(profile, severity)
            self._hotspots.append(Hotspot(
                function_name=profile.function_name,
                file_path=profile.file_path,
                line_number=profile.line_number,
                severity=severity,
                total_time=profile.total_time,
                call_count=profile.call_count,
                suggestion=suggestion,
            ))

    def _generate_suggestion(self, profile: FunctionProfile, severity: str) -> str:
        """为热点函数生成优化建议

        Args:
            profile: 函数性能档案
            severity: 严重程度

        Returns:
            优化建议字符串
        """
        suggestions = []
        if profile.avg_time > 0.1:
            suggestions.append("Consider caching results to reduce repeated computation")
        if profile.call_count > 1000 and profile.avg_time > 0.001:
            suggestions.append("High call frequency detected; consider batching or debouncing")
        if profile.memory_allocated > 1024 * 1024:
            suggestions.append(f"High memory usage ({profile.memory_allocated / 1024 / 1024:.1f}MB); "
                             "consider using generators or streaming")
        if severity == "critical":
            suggestions.append("Critical hotspot: prioritize optimization")
        if not suggestions:
            suggestions.append("Monitor for performance changes")
        return "; ".join(suggestions)

    def generate_report(self) -> ProfileReport:
        """生成完整的性能分析报告

        Returns:
            ProfileReport报告
        """
        total_time = sum(p.total_time for p in self._profiles.values())
        total_calls = sum(p.call_count for p in self._profiles.values())
        return ProfileReport(
            total_time=total_time,
            total_calls=total_calls,
            function_profiles=dict(self._profiles),
            hotspots=list(self._hotspots),
            memory_snapshots=list(self._memory_snapshots),
            call_graph=dict(self._call_graph),
        )

    def get_cprofile_stats(self, sort_by: str = "cumulative") -> str:
        """获取cProfile统计信息

        Args:
            sort_by: 排序字段

        Returns:
            格式化的统计字符串
        """
        if not self._profiler:
            return "No profiler data available. Call start_profiling() first."
        sio = io.StringIO()
        ps = pstats.Stats(self._profiler, stream=sio).sort_stats(sort_by)
        ps.print_stats(30)
        return sio.getvalue()

    def get_top_functions(self, n: int = 10,
                           sort_by: str = "total_time") -> List[FunctionProfile]:
        """获取耗时最多的N个函数

        Args:
            n: 返回数量
            sort_by: 排序字段

        Returns:
            排序后的函数档案列表
        """
        key_map = {
            "total_time": lambda p: p.total_time,
            "call_count": lambda p: p.call_count,
            "avg_time": lambda p: p.avg_time,
            "own_time": lambda p: p.own_time,
            "memory": lambda p: p.memory_allocated,
        }
        sort_key = key_map.get(sort_by, lambda p: p.total_time)
        return sorted(self._profiles.values(), key=sort_key, reverse=True)[:n]

    def get_call_graph(self) -> Dict[str, List[str]]:
        """获取函数调用图

        Returns:
            调用图字典
        """
        return dict(self._call_graph)

    def compare_profiles(self, other: "ProfileHotspots") -> Dict[str, Any]:
        """比较两个性能分析结果

        Args:
            other: 另一个分析器实例

        Returns:
            比较结果字典
        """
        common_functions = set(self._profiles.keys()) & set(other._profiles.keys())
        only_in_self = set(self._profiles.keys()) - set(other._profiles.keys())
        only_in_other = set(other._profiles.keys()) - set(self._profiles.keys())
        regressions = []
        improvements = []
        for name in common_functions:
            self_time = self._profiles[name].avg_time
            other_time = other._profiles[name].avg_time
            if self_time == 0:
                continue
            change_pct = ((other_time - self_time) / self_time) * 100.0
            entry = {
                "function": name,
                "before": round(self_time, 6),
                "after": round(other_time, 6),
                "change_pct": round(change_pct, 1),
            }
            if change_pct > 10:
                regressions.append(entry)
            elif change_pct < -10:
                improvements.append(entry)
        return {
            "common_functions": len(common_functions),
            "only_in_baseline": list(only_in_self),
            "only_in_comparison": list(only_in_other),
            "regressions": regressions,
            "improvements": improvements,
        }

    def format_report(self, report: Optional[ProfileReport] = None) -> str:
        """格式化终端输出的性能报告

        Args:
            report: 可选的报告对象

        Returns:
            格式化的报告字符串
        """
        if report is None:
            report = self.generate_report()
        lines = [
            "=" * 70,
            "PERFORMANCE PROFILE REPORT",
            "=" * 70,
            f"  Total Time:    {report.total_time:.4f}s",
            f"  Total Calls:   {report.total_calls}",
            f"  Functions:     {len(report.function_profiles)}",
            f"  Hotspots:      {len(report.hotspots)}",
            "-" * 70,
            "  TOP FUNCTIONS BY TOTAL TIME:",
        ]
        top_funcs = sorted(
            report.function_profiles.values(),
            key=lambda p: p.total_time,
            reverse=True,
        )[:10]
        for fp in top_funcs:
            pct = (fp.total_time / report.total_time * 100.0) if report.total_time > 0 else 0.0
            lines.append(f"    {fp.function_name:40s} {fp.total_time:8.4f}s  "
                        f"({pct:5.1f}%)  calls:{fp.call_count}")
        if report.hotspots:
            lines.append("-" * 70)
            lines.append("  HOTSPOTS:")
            for h in report.hotspots[:5]:
                lines.append(f"    [{h.severity.upper():8s}] {h.function_name} "
                            f"({h.file_path}:{h.line_number})")
                if h.suggestion:
                    lines.append(f"             -> {h.suggestion}")
        if report.memory_snapshots:
            lines.append("-" * 70)
            lines.append("  MEMORY SNAPSHOTS:")
            for snap in report.memory_snapshots:
                lines.append(f"    [{time.strftime('%H:%M:%S', time.localtime(snap.timestamp))}] "
                            f"Current: {snap.current_usage / 1024:.1f}KB  "
                            f"Peak: {snap.peak_usage / 1024:.1f}KB")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save_json_report(self, filepath: str) -> None:
        """保存JSON格式的性能分析报告

        Args:
            filepath: 输出文件路径
        """
        report = self.generate_report()
        data = report.to_dict()
        data["timestamp"] = time.time()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def reset(self) -> None:
        """重置所有分析数据"""
        with self._lock:
            self._profiles.clear()
            self._hotspots.clear()
            self._memory_snapshots.clear()
            self._call_graph.clear()
            self._call_stack.clear()
            self._profiler = None
            if self._tracemalloc_started:
                tracemalloc.stop()
                self._tracemalloc_started = False

    def set_hotspot_threshold(self, threshold: float) -> None:
        """设置热点检测阈值

        Args:
            threshold: 时间占比阈值 (0.0 - 1.0)
        """
        self._hotspot_threshold = max(0.01, min(threshold, 1.0))
