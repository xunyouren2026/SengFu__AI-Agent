"""
CoveragePlugin - Pytest代码覆盖率收集插件

模块路径: testing/pytest_config/plugins/coverage_plugin.py

提供pytest测试运行期间的代码覆盖率收集、分析和报告功能。
"""

import os
import sys
import json
import time
import random
import tempfile
import shutil
import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

import pytest


@dataclass
class CoverageRecord:
    """单条覆盖率记录"""
    file_path: str
    line_number: int
    hit_count: int = 0
    function_name: str = ""
    is_branch: bool = False


@dataclass
class CoverageReport:
    """覆盖率报告"""
    total_files: int = 0
    covered_files: int = 0
    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    file_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def line_coverage_pct(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100.0

    @property
    def branch_coverage_pct(self) -> float:
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100.0

    @property
    def file_coverage_pct(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.covered_files / self.total_files) * 100.0


class CoveragePlugin:
    """Pytest代码覆盖率收集插件

    在pytest测试运行期间收集代码覆盖率数据，支持行覆盖率和分支覆盖率，
    并生成详细的覆盖率报告。

    功能:
        - 自动追踪被测代码的执行路径
        - 支持行级别和分支级别的覆盖率统计
        - 生成JSON/HTML/终端格式的覆盖率报告
        - 支持覆盖率阈值检查，低于阈值时使测试失败
        - 支持排除指定目录或文件
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._coverage_data: Dict[str, Set[int]] = defaultdict(set)
        self._branch_data: Dict[str, List[Tuple[int, int, bool]]] = defaultdict(list)
        self._source_lines: Dict[str, List[str]] = {}
        self._excluded_paths: Set[str] = set()
        self._included_paths: Set[str] = set()
        self._min_coverage: float = self.config.get("min_coverage", 0.0)
        self._report_dir: Optional[str] = self.config.get("report_dir", None)
        self._report_formats: List[str] = self.config.get("report_formats", ["terminal"])
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """初始化覆盖率插件，解析配置并准备追踪环境"""
        excluded = self.config.get("exclude", [])
        for pattern in excluded:
            self._excluded_paths.add(pattern)
        included = self.config.get("include", [])
        for pattern in included:
            self._included_paths.add(pattern)
        if self._report_dir:
            os.makedirs(self._report_dir, exist_ok=True)
        self._initialized = True

    def start_coverage(self) -> None:
        """开始覆盖率收集，安装trace函数"""
        if not self._initialized:
            self.initialize()
        sys.settrace(self._trace_callback)

    def stop_coverage(self) -> None:
        """停止覆盖率收集，移除trace函数"""
        sys.settrace(None)

    def _trace_callback(self, frame, event, arg) -> Optional[Any]:
        """Python trace回调函数，记录每一行的执行"""
        if not self._initialized:
            return None
        filename = frame.f_code.co_filename
        if not self._should_track(filename):
            return None
        lineno = frame.f_lineno
        with self._lock:
            self._coverage_data[filename].add(lineno)
        return self._trace_callback

    def _should_track(self, filepath: str) -> bool:
        """判断文件是否应该被追踪"""
        abs_path = os.path.abspath(filepath)
        for excluded in self._excluded_paths:
            if excluded in abs_path:
                return False
        if self._included_paths:
            return any(inc in abs_path for inc in self._included_paths)
        return abs_path.endswith(".py") and "site-packages" not in abs_path

    def record_branch(self, file_path: str, line: int, branch_id: int, taken: bool) -> None:
        """记录分支覆盖数据

        Args:
            file_path: 源文件路径
            line: 分支所在行号
            branch_id: 分支标识符
            taken: 分支是否被执行
        """
        with self._lock:
            self._branch_data[file_path].append((line, branch_id, taken))

    def get_coverage_data(self) -> Dict[str, Set[int]]:
        """获取当前收集的覆盖率数据

        Returns:
            字典，键为文件路径，值为被执行的行号集合
        """
        return dict(self._coverage_data)

    def get_branch_data(self) -> Dict[str, List[Tuple[int, int, bool]]]:
        """获取分支覆盖数据

        Returns:
            字典，键为文件路径，值为分支记录列表
        """
        return dict(self._branch_data)

    def compute_file_coverage(self, file_path: str) -> Dict[str, Any]:
        """计算单个文件的覆盖率

        Args:
            file_path: 源文件路径

        Returns:
            包含行覆盖率、分支覆盖率等信息的字典
        """
        if file_path not in self._source_lines:
            self._load_source(file_path)
        lines = self._source_lines.get(file_path, [])
        executable_lines = set()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith('"""') \
                    and not stripped.startswith("'''") and stripped not in ("", "pass",):
                executable_lines.add(i)
        covered = self._coverage_data.get(file_path, set())
        covered_executable = executable_lines & covered
        branch_entries = self._branch_data.get(file_path, [])
        total_branches = len(set((ln, bid) for ln, bid, _ in branch_entries))
        covered_branches = len(set((ln, bid) for ln, bid, taken in branch_entries if taken))
        return {
            "file_path": file_path,
            "total_lines": len(executable_lines),
            "covered_lines": len(covered_executable),
            "coverage_pct": (len(covered_executable) / len(executable_lines) * 100.0) if executable_lines else 100.0,
            "total_branches": total_branches,
            "covered_branches": covered_branches,
            "branch_coverage_pct": (covered_branches / total_branches * 100.0) if total_branches else 100.0,
            "uncovered_lines": sorted(executable_lines - covered),
        }

    def generate_report(self) -> CoverageReport:
        """生成完整的覆盖率报告

        Returns:
            CoverageReport对象，包含所有文件的覆盖率统计
        """
        report = CoverageReport()
        all_files = set(self._coverage_data.keys()) | set(self._source_lines.keys())
        for file_path in all_files:
            file_report = self.compute_file_coverage(file_path)
            report.file_reports[file_path] = file_report
            report.total_files += 1
            report.total_lines += file_report["total_lines"]
            report.covered_lines += file_report["covered_lines"]
            report.total_branches += file_report["total_branches"]
            report.covered_branches += file_report["covered_branches"]
            if file_report["coverage_pct"] > 0:
                report.covered_files += 1
        return report

    def check_threshold(self, report: Optional[CoverageReport] = None) -> Tuple[bool, str]:
        """检查覆盖率是否达到配置的阈值

        Args:
            report: 可选的覆盖率报告，如未提供则自动生成

        Returns:
            (是否通过, 消息) 元组
        """
        if report is None:
            report = self.generate_report()
        if report.line_coverage_pct >= self._min_coverage:
            return True, f"Coverage {report.line_coverage_pct:.1f}% meets threshold {self._min_coverage:.1f}%"
        return False, f"Coverage {report.line_coverage_pct:.1f}% below threshold {self._min_coverage:.1f}%"

    def save_json_report(self, filepath: str) -> None:
        """将覆盖率报告保存为JSON格式

        Args:
            filepath: 输出文件路径
        """
        report = self.generate_report()
        data = {
            "summary": {
                "total_files": report.total_files,
                "covered_files": report.covered_files,
                "total_lines": report.total_lines,
                "covered_lines": report.covered_lines,
                "line_coverage_pct": round(report.line_coverage_pct, 2),
                "branch_coverage_pct": round(report.branch_coverage_pct, 2),
            },
            "files": report.file_reports,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_html_report(self, filepath: str) -> None:
        """将覆盖率报告保存为HTML格式

        Args:
            filepath: 输出HTML文件路径
        """
        report = self.generate_report()
        html_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>Coverage Report</title>",
            "<style>",
            "body{font-family:monospace;margin:20px;background:#1e1e1e;color:#d4d4d4}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #444}",
            "th{background:#2d2d2d;color:#569cd6}",
            ".pass{color:#4ec9b0}.fail{color:#f44747}.warn{color:#dcdcaa}",
            ".bar{height:16px;border-radius:3px}",
            "</style></head><body>",
            f"<h1>Coverage Report</h1>",
            f"<p>Line Coverage: <b>{report.line_coverage_pct:.1f}%</b> "
            f"({report.covered_lines}/{report.total_lines})</p>",
            f"<p>Branch Coverage: <b>{report.branch_coverage_pct:.1f}%</b> "
            f"({report.covered_branches}/{report.total_branches})</p>",
            "<table><tr><th>File</th><th>Lines</th><th>Covered</th><th>Coverage</th></tr>",
        ]
        for fp, fr in sorted(report.file_reports.items()):
            cls = "pass" if fr["coverage_pct"] >= 80 else ("warn" if fr["coverage_pct"] >= 50 else "fail")
            short = os.path.basename(fp)
            html_parts.append(
                f"<tr><td>{short}</td><td>{fr['total_lines']}</td>"
                f"<td>{fr['covered_lines']}</td>"
                f"<td class='{cls}'>{fr['coverage_pct']:.1f}%</td></tr>"
            )
        html_parts.append("</table></body></html>")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

    def format_terminal_report(self, report: Optional[CoverageReport] = None) -> str:
        """格式化终端输出的覆盖率报告

        Args:
            report: 可选的覆盖率报告

        Returns:
            格式化的字符串报告
        """
        if report is None:
            report = self.generate_report()
        lines = [
            "=" * 70,
            "COVERAGE REPORT",
            "=" * 70,
            f"  Total Files:  {report.total_files}  (covered: {report.covered_files})",
            f"  Total Lines:  {report.total_lines}  (covered: {report.covered_lines})",
            f"  Line Coverage: {report.line_coverage_pct:.1f}%",
            f"  Branch Coverage: {report.branch_coverage_pct:.1f}%",
            "-" * 70,
        ]
        for fp, fr in sorted(report.file_reports.items()):
            short = os.path.basename(fp)
            bar_len = 30
            filled = int(bar_len * fr["coverage_pct"] / 100.0)
            bar = "#" * filled + "-" * (bar_len - filled)
            lines.append(f"  {short:40s} [{bar}] {fr['coverage_pct']:5.1f}%")
        lines.append("=" * 70)
        return "\n".join(lines)

    def _load_source(self, file_path: str) -> None:
        """加载源文件内容用于分析可执行行"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                self._source_lines[file_path] = f.readlines()
        except (IOError, OSError):
            self._source_lines[file_path] = []

    def reset(self) -> None:
        """重置所有覆盖率数据"""
        with self._lock:
            self._coverage_data.clear()
            self._branch_data.clear()
            self._source_lines.clear()

    def merge(self, other: "CoveragePlugin") -> None:
        """合并另一个CoveragePlugin实例的覆盖率数据

        Args:
            other: 另一个覆盖率插件实例
        """
        with self._lock:
            for fp, lines in other._coverage_data.items():
                self._coverage_data[fp].update(lines)
            for fp, branches in other._branch_data.items():
                self._branch_data[fp].extend(branches)

    def add_file_filter(self, include_pattern: str = "", exclude_pattern: str = "") -> None:
        """动态添加文件过滤规则

        Args:
            include_pattern: 包含的路径模式
            exclude_pattern: 排除的路径模式
        """
        if include_pattern:
            self._included_paths.add(include_pattern)
        if exclude_pattern:
            self._excluded_paths.add(exclude_pattern)

    def get_uncovered_files(self) -> List[str]:
        """获取所有未被覆盖的文件列表

        Returns:
            未被任何测试覆盖的文件路径列表
        """
        uncovered = []
        for fp in self._source_lines:
            if fp not in self._coverage_data or len(self._coverage_data[fp]) == 0:
                uncovered.append(fp)
        return sorted(uncovered)

    def get_file_hash(self, file_path: str) -> str:
        """计算源文件的MD5哈希，用于检测文件变更

        Args:
            file_path: 源文件路径

        Returns:
            文件的MD5哈希值
        """
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except (IOError, OSError):
            return ""
        return hasher.hexdigest()
