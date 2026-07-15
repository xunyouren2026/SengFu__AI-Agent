"""
工作流执行流可视化模块

提供工作流执行过程的实时可视化功能，包括执行流渲染、
节点状态着色、进度覆盖层、错误高亮、交互式节点选择和统计仪表盘。
此模块与 visualization.py（图布局引擎）互补，专注于执行时可视化。

Classes:
    WorkflowVisualizer: 工作流可视化器主类
    ExecutionFlowRenderer: 执行流渲染器
    NodeStatusColorizer: 节点状态着色器
    ProgressOverlay: 进度覆盖层
    ErrorHighlighter: 错误高亮器
    NodeSelector: 交互式节点选择器
    StatsDashboard: 统计仪表盘
"""

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================
# 数据模型
# ============================================================

class NodeVisualStatus(Enum):
    """节点可视化状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    WARNING = "warning"


@dataclass
class NodeVisualInfo:
    """节点可视化信息"""
    node_id: str
    name: str
    status: NodeVisualStatus
    x: float = 0.0
    y: float = 0.0
    width: float = 120.0
    height: float = 50.0
    duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class EdgeVisualInfo:
    """边可视化信息"""
    source_id: str
    target_id: str
    label: str = ""
    active: bool = False
    data_flow: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionSnapshot:
    """执行快照"""
    workflow_id: str
    timestamp: float
    nodes: Dict[str, NodeVisualInfo] = dataclass_field(default_factory=dict)
    edges: List[EdgeVisualInfo] = dataclass_field(default_factory=list)
    global_start_time: float = 0.0
    global_progress: float = 0.0


# ============================================================
# 节点状态着色器
# ============================================================

class NodeStatusColorizer:
    """
    节点状态着色器

    根据节点执行状态分配颜色方案，支持自定义主题。

    Usage:
        colorizer = NodeStatusColorizer()
        colors = colorizer.get_colors(NodeVisualStatus.RUNNING)
    """

    # 默认颜色方案 (foreground, background, border)
    DEFAULT_THEME: Dict[NodeVisualStatus, Tuple[str, str, str]] = {
        NodeVisualStatus.PENDING: ("#888888", "#f5f5f5", "#cccccc"),
        NodeVisualStatus.RUNNING: ("#1a1a2e", "#4fc3f7", "#0288d1"),
        NodeVisualStatus.SUCCESS: ("#ffffff", "#66bb6a", "#388e3c"),
        NodeVisualStatus.FAILED: ("#ffffff", "#ef5350", "#c62828"),
        NodeVisualStatus.SKIPPED: ("#888888", "#e0e0e0", "#bdbdbd"),
        NodeVisualStatus.TIMEOUT: ("#ffffff", "#ffa726", "#e65100"),
        NodeVisualStatus.CANCELLED: ("#ffffff", "#ab47bc", "#6a1b9a"),
        NodeVisualStatus.WARNING: ("#333333", "#ffee58", "#f9a825"),
    }

    def __init__(
        self,
        custom_theme: Optional[Dict[NodeVisualStatus, Tuple[str, str, str]]] = None,
    ) -> None:
        self._theme: Dict[NodeVisualStatus, Tuple[str, str, str]] = {}
        self._theme.update(self.DEFAULT_THEME)
        if custom_theme:
            self._theme.update(custom_theme)

    def get_colors(
        self, status: NodeVisualStatus
    ) -> Tuple[str, str, str]:
        """
        获取状态对应的颜色

        Args:
            status: 节点状态

        Returns:
            (foreground, background, border) 颜色元组
        """
        return self._theme.get(status, self._theme[NodeVisualStatus.PENDING])

    def get_background(self, status: NodeVisualStatus) -> str:
        """获取背景色"""
        return self.get_colors(status)[1]

    def get_border(self, status: NodeVisualStatus) -> str:
        """获取边框色"""
        return self.get_colors(status)[2]

    def get_foreground(self, status: NodeVisualStatus) -> str:
        """获取前景色"""
        return self.get_colors(status)[0]

    def apply_to_svg_style(self, status: NodeVisualStatus) -> Dict[str, str]:
        """
        生成 SVG 样式属性

        Returns:
            SVG 样式字典
        """
        fg, bg, border = self.get_colors(status)
        return {
            "fill": bg,
            "stroke": border,
            "stroke-width": "2",
            "color": fg,
        }

    def set_custom_color(
        self,
        status: NodeVisualStatus,
        foreground: str,
        background: str,
        border: str,
    ) -> None:
        """设置自定义颜色"""
        self._theme[status] = (foreground, background, border)


# ============================================================
# 执行流渲染器
# ============================================================

class ExecutionFlowRenderer:
    """
    执行流渲染器

    将工作流执行状态渲染为 ASCII 文本或结构化数据。

    Usage:
        renderer = ExecutionFlowRenderer(colorizer=NodeStatusColorizer())
        ascii_output = renderer.render_ascii(snapshot)
        json_output = renderer.render_json(snapshot)
    """

    def __init__(
        self,
        colorizer: Optional[NodeStatusColorizer] = None,
    ) -> None:
        self.colorizer = colorizer or NodeStatusColorizer()

    def render_ascii(
        self,
        snapshot: ExecutionSnapshot,
        width: int = 80,
    ) -> str:
        """
        渲染 ASCII 执行流图

        Args:
            snapshot: 执行快照
            width: 输出宽度

        Returns:
            ASCII 字符串
        """
        lines: List[str] = []
        separator = "=" * width
        thin_sep = "-" * width

        lines.append(separator)
        lines.append(f"  工作流: {snapshot.workflow_id}")
        lines.append(
            f"  进度: {snapshot.global_progress:.1%}  "
            f"  时间: {time.strftime('%H:%M:%S', time.localtime(snapshot.timestamp))}"
        )
        lines.append(separator)

        # 按状态分组节点
        status_groups: Dict[NodeVisualStatus, List[NodeVisualInfo]] = defaultdict(list)
        for node_info in snapshot.nodes.values():
            status_groups[node_info.status].append(node_info)

        # 渲染各状态节点
        status_order = [
            NodeVisualStatus.RUNNING,
            NodeVisualStatus.FAILED,
            NodeVisualStatus.WARNING,
            NodeVisualStatus.SUCCESS,
            NodeVisualStatus.SKIPPED,
            NodeVisualStatus.CANCELLED,
            NodeVisualStatus.TIMEOUT,
            NodeVisualStatus.PENDING,
        ]

        status_symbols: Dict[NodeVisualStatus, str] = {
            NodeVisualStatus.PENDING: "[ ]",
            NodeVisualStatus.RUNNING: "[>]",
            NodeVisualStatus.SUCCESS: "[OK]",
            NodeVisualStatus.FAILED: "[XX]",
            NodeVisualStatus.SKIPPED: "[--]",
            NodeVisualStatus.TIMEOUT: "[!!]",
            NodeVisualStatus.CANCELLED: "[~~]",
            NodeVisualStatus.WARNING: "[??]",
        }

        for status in status_order:
            nodes = status_groups.get(status, [])
            if not nodes:
                continue

            lines.append("")
            status_label = status.value.upper()
            lines.append(f"  {status_label} ({len(nodes)})")
            lines.append(thin_sep)

            for node in nodes:
                symbol = status_symbols.get(status, "[ ]")
                duration_str = f"{node.duration:.2f}s" if node.duration > 0 else "-"
                name_display = node.name[:40]
                line = f"  {symbol} {name_display:<40} {duration_str:>8}"

                if node.error:
                    error_display = node.error[:30]
                    line += f"  ERR: {error_display}"

                lines.append(line)

                if node.retry_count > 0:
                    lines.append(f"       (重试 {node.retry_count} 次)")

        lines.append(separator)
        return "\n".join(lines)

    def render_json(self, snapshot: ExecutionSnapshot) -> str:
        """
        渲染为 JSON 格式

        Args:
            snapshot: 执行快照

        Returns:
            JSON 字符串
        """
        data: Dict[str, Any] = {
            "workflow_id": snapshot.workflow_id,
            "timestamp": snapshot.timestamp,
            "progress": round(snapshot.global_progress, 4),
            "nodes": {},
            "edges": [],
        }

        for node_id, info in snapshot.nodes.items():
            data["nodes"][node_id] = {
                "name": info.name,
                "status": info.status.value,
                "duration": round(info.duration, 4),
                "error": info.error,
                "retry_count": info.retry_count,
                "position": {"x": info.x, "y": info.y},
            }

        for edge in snapshot.edges:
            edge_data: Dict[str, Any] = {
                "source": edge.source_id,
                "target": edge.target_id,
                "active": edge.active,
            }
            if edge.label:
                edge_data["label"] = edge.label
            data["edges"].append(edge_data)

        return json.dumps(data, indent=2, ensure_ascii=False)


# ============================================================
# 进度覆盖层
# ============================================================

class ProgressOverlay:
    """
    进度覆盖层

    计算和显示工作流执行进度，支持分层进度跟踪。

    Usage:
        overlay = ProgressOverlay(total_nodes=10)
        overlay.update_node("node1", NodeVisualStatus.SUCCESS)
        overlay.update_node("node2", NodeVisualStatus.RUNNING)
        progress = overlay.get_progress()
    """

    def __init__(self, total_nodes: int = 0) -> None:
        self.total_nodes = max(0, total_nodes)
        self._node_statuses: Dict[str, NodeVisualStatus] = {}
        self._start_time: float = time.time()
        self._estimated_duration: float = 0.0

    def set_total_nodes(self, total: int) -> None:
        """设置总节点数"""
        self.total_nodes = max(0, total)

    def update_node(
        self,
        node_id: str,
        status: NodeVisualStatus,
    ) -> None:
        """更新节点状态"""
        self._node_statuses[node_id] = status

    def get_progress(self) -> float:
        """
        计算整体进度

        Returns:
            0.0 到 1.0 之间的进度值
        """
        if self.total_nodes == 0:
            return 0.0

        completed = sum(
            1 for s in self._node_statuses.values()
            if s in (
                NodeVisualStatus.SUCCESS,
                NodeVisualStatus.SKIPPED,
                NodeVisualStatus.FAILED,
                NodeVisualStatus.TIMEOUT,
                NodeVisualStatus.CANCELLED,
            )
        )

        running = sum(
            1 for s in self._node_statuses.values()
            if s == NodeVisualStatus.RUNNING
        )

        # 运行中的节点按 50% 计算进度
        effective = completed + running * 0.5
        return min(1.0, effective / self.total_nodes)

    def get_status_counts(self) -> Dict[str, int]:
        """获取各状态节点计数"""
        counts: Dict[str, int] = defaultdict(int)
        for status in self._node_statuses.values():
            counts[status.value] += 1
        return dict(counts)

    def estimate_remaining_time(self) -> float:
        """
        估算剩余时间

        Returns:
            预计剩余秒数
        """
        progress = self.get_progress()
        if progress <= 0:
            return float("inf")

        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return float("inf")

        total_estimated = elapsed / progress
        return max(0.0, total_estimated - elapsed)

    def render_progress_bar(
        self,
        width: int = 40,
        filled_char: str = "#",
        empty_char: str = "-",
    ) -> str:
        """
        渲染文本进度条

        Args:
            width: 进度条宽度
            filled_char: 已填充字符
            empty_char: 空字符

        Returns:
            进度条字符串
        """
        progress = self.get_progress()
        filled = int(width * progress)
        empty = width - filled
        bar = filled_char * filled + empty_char * empty
        percentage = f"{progress:.1%}"

        return f"[{bar}] {percentage}"

    def reset(self) -> None:
        """重置进度"""
        self._node_statuses.clear()
        self._start_time = time.time()


# ============================================================
# 错误高亮器
# ============================================================

class ErrorHighlighter:
    """
    错误高亮器

    收集和分析工作流执行中的错误信息，生成高亮报告。

    Usage:
        highlighter = ErrorHighlighter()
        highlighter.add_error("node1", "连接超时", "TimeoutError")
        highlighter.add_error("node2", "参数无效", "ValueError")
        report = highlighter.generate_report()
    """

    def __init__(self) -> None:
        self._errors: List[Dict[str, Any]] = []

    def add_error(
        self,
        node_id: str,
        message: str,
        error_type: str = "",
        timestamp: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加错误记录"""
        self._errors.append({
            "node_id": node_id,
            "message": message,
            "error_type": error_type,
            "timestamp": timestamp or time.time(),
            "details": details or {},
        })

    def clear(self) -> None:
        """清除所有错误"""
        self._errors.clear()

    @property
    def error_count(self) -> int:
        """错误数量"""
        return len(self._errors)

    @property
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self._errors) > 0

    def get_errors_by_node(self) -> Dict[str, List[Dict[str, Any]]]:
        """按节点分组错误"""
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for error in self._errors:
            groups[error["node_id"]].append(error)
        return dict(groups)

    def get_error_types(self) -> Dict[str, int]:
        """按错误类型统计"""
        counts: Dict[str, int] = defaultdict(int)
        for error in self._errors:
            etype = error.get("error_type", "Unknown")
            counts[etype] += 1
        return dict(counts)

    def get_most_recent(self, count: int = 5) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        return sorted(
            self._errors,
            key=lambda e: e.get("timestamp", 0),
            reverse=True,
        )[:count]

    def generate_report(self) -> str:
        """
        生成错误高亮报告

        Returns:
            文本报告
        """
        if not self._errors:
            return "无错误"

        lines: List[str] = []
        lines.append(f"=== 错误报告 ({len(self._errors)} 个错误) ===")
        lines.append("")

        # 按节点分组
        by_node = self.get_errors_by_node()
        for node_id, errors in by_node.items():
            lines.append(f"[节点: {node_id}]")
            for err in errors:
                ts = err.get("timestamp", 0)
                time_str = time.strftime(
                    "%H:%M:%S", time.localtime(ts)
                )
                etype = err.get("error_type", "")
                msg = err.get("message", "")
                lines.append(
                    f"  [{time_str}] {etype}: {msg}"
                )
                if err.get("details"):
                    for k, v in err["details"].items():
                        lines.append(f"    {k}: {v}")
            lines.append("")

        # 错误类型统计
        type_counts = self.get_error_types()
        if type_counts:
            lines.append("错误类型统计:")
            for etype, count in sorted(
                type_counts.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {etype}: {count}")

        return "\n".join(lines)


# ============================================================
# 交互式节点选择器
# ============================================================

class NodeSelector:
    """
    交互式节点选择器

    支持按条件筛选和选择节点，用于调试和监控。

    Usage:
        selector = NodeSelector(snapshot)
        selector.add_filter("status", NodeVisualStatus.FAILED)
        failed_nodes = selector.get_selected()
        selector.add_filter("duration_gt", 5.0)
        slow_failed = selector.get_selected()
    """

    def __init__(
        self,
        snapshot: Optional[ExecutionSnapshot] = None,
    ) -> None:
        self._snapshot = snapshot
        self._filters: List[Callable[[NodeVisualInfo], bool]] = []
        self._selected_ids: Set[str] = set()

    def set_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        """设置执行快照"""
        self._snapshot = snapshot
        self._selected_ids.clear()

    def add_filter(
        self,
        filter_type: str,
        value: Any,
    ) -> None:
        """
        添加筛选条件

        Args:
            filter_type: 筛选类型 (status, name_contains, duration_gt, duration_lt, has_error)
            value: 筛选值
        """
        if filter_type == "status":
            if isinstance(value, str):
                value = NodeVisualStatus(value)
            self._filters.append(
                lambda n, v=value: n.status == v
            )

        elif filter_type == "name_contains":
            self._filters.append(
                lambda n, v=str(value): v.lower() in n.name.lower()
            )

        elif filter_type == "duration_gt":
            self._filters.append(
                lambda n, v=float(value): n.duration > v
            )

        elif filter_type == "duration_lt":
            self._filters.append(
                lambda n, v=float(value): n.duration < v
            )

        elif filter_type == "has_error":
            self._filters.append(
                lambda n: n.error is not None
            )

        elif filter_type == "retry_count_gt":
            self._filters.append(
                lambda n, v=int(value): n.retry_count > v
            )

    def clear_filters(self) -> None:
        """清除所有筛选条件"""
        self._filters.clear()

    def get_selected(self) -> List[NodeVisualInfo]:
        """获取筛选后的节点列表"""
        if self._snapshot is None:
            return []

        results: List[NodeVisualInfo] = []
        for node_info in self._snapshot.nodes.values():
            if all(f(node_info) for f in self._filters):
                results.append(node_info)

        return sorted(results, key=lambda n: n.start_time)

    def select_by_id(self, node_id: str) -> Optional[NodeVisualInfo]:
        """按 ID 选择节点"""
        if self._snapshot is None:
            return None
        return self._snapshot.nodes.get(node_id)

    def get_node_chain(self, node_id: str) -> List[NodeVisualInfo]:
        """
        获取节点的执行链（前驱节点）

        Args:
            node_id: 起始节点 ID

        Returns:
            从起始节点到根节点的路径
        """
        if self._snapshot is None:
            return []

        # 构建邻接表
        children_to_parents: Dict[str, List[str]] = defaultdict(list)
        for edge in self._snapshot.edges:
            children_to_parents[edge.target_id].append(edge.source_id)

        chain: List[NodeVisualInfo] = []
        visited: Set[str] = set()
        current = node_id

        while current and current not in visited:
            visited.add(current)
            node_info = self._snapshot.nodes.get(current)
            if node_info:
                chain.append(node_info)
            parents = children_to_parents.get(current, [])
            current = parents[0] if parents else ""

        return chain

    @property
    def filter_count(self) -> int:
        """当前筛选条件数量"""
        return len(self._filters)


# ============================================================
# 统计仪表盘
# ============================================================

class StatsDashboard:
    """
    统计仪表盘

    汇总工作流执行统计信息，生成多维度的分析报告。

    Usage:
        dashboard = StatsDashboard()
        dashboard.record_execution(snapshot1)
        dashboard.record_execution(snapshot2)
        report = dashboard.generate_summary()
    """

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self._snapshots: List[ExecutionSnapshot] = []
        self._execution_times: List[float] = []
        self._success_counts: List[int] = []
        self._failure_counts: List[int] = []

    def record_execution(self, snapshot: ExecutionSnapshot) -> None:
        """记录一次执行快照"""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_history:
            self._snapshots = self._snapshots[-self.max_history:]

        # 提取统计
        success = sum(
            1 for n in snapshot.nodes.values()
            if n.status == NodeVisualStatus.SUCCESS
        )
        failed = sum(
            1 for n in snapshot.nodes.values()
            if n.status == NodeVisualStatus.FAILED
        )
        total_duration = max(
            (n.end_time for n in snapshot.nodes.values() if n.end_time > 0),
            default=0.0,
        )

        self._execution_times.append(total_duration)
        self._success_counts.append(success)
        self._failure_counts.append(failed)

    def get_average_duration(self) -> float:
        """获取平均执行时间"""
        if not self._execution_times:
            return 0.0
        return sum(self._execution_times) / len(self._execution_times)

    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self._success_counts and not self._failure_counts:
            return 0.0
        total_success = sum(self._success_counts)
        total_all = total_success + sum(self._failure_counts)
        if total_all == 0:
            return 0.0
        return total_success / total_all

    def get_slowest_nodes(
        self,
        count: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        获取最慢的节点

        Returns:
            (node_id, name, duration) 列表
        """
        all_nodes: List[Tuple[str, str, float]] = []
        for snapshot in self._snapshots:
            for node_id, info in snapshot.nodes.items():
                if info.duration > 0:
                    all_nodes.append(
                        (node_id, info.name, info.duration)
                    )

        all_nodes.sort(key=lambda x: -x[2])
        return all_nodes[:count]

    def get_most_failed_nodes(
        self,
        count: int = 5,
    ) -> List[Tuple[str, str, int]]:
        """
        获取最常失败的节点

        Returns:
            (node_id, name, failure_count) 列表
        """
        failure_counts: Dict[str, Tuple[str, int]] = {}
        for snapshot in self._snapshots:
            for node_id, info in snapshot.nodes.items():
                if info.status == NodeVisualStatus.FAILED:
                    if node_id in failure_counts:
                        failure_counts[node_id] = (
                            failure_counts[node_id][0],
                            failure_counts[node_id][1] + 1,
                        )
                    else:
                        failure_counts[node_id] = (info.name, 1)

        sorted_failures = sorted(
            failure_counts.items(),
            key=lambda x: -x[1][1],
        )
        return [
            (nid, name, count) for nid, (name, count) in sorted_failures
        ][:count]

    def generate_summary(self) -> str:
        """
        生成统计摘要

        Returns:
            文本摘要
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  工作流执行统计仪表盘")
        lines.append("=" * 60)

        total_runs = len(self._snapshots)
        lines.append(f"  总执行次数: {total_runs}")

        if total_runs > 0:
            avg_duration = self.get_average_duration()
            success_rate = self.get_success_rate()
            lines.append(f"  平均执行时间: {avg_duration:.2f}s")
            lines.append(f"  成功率: {success_rate:.1%}")

            # 最近一次执行
            latest = self._snapshots[-1]
            lines.append(f"  最近执行进度: {latest.global_progress:.1%}")

        lines.append("")

        # 最慢节点
        slowest = self.get_slowest_nodes(3)
        if slowest:
            lines.append("  最慢节点:")
            for nid, name, duration in slowest:
                lines.append(f"    {name}: {duration:.2f}s")

        lines.append("")

        # 最常失败节点
        most_failed = self.get_most_failed_nodes(3)
        if most_failed:
            lines.append("  最常失败节点:")
            for nid, name, count in most_failed:
                lines.append(f"    {name}: {count} 次失败")

        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_stats_dict(self) -> Dict[str, Any]:
        """生成统计字典"""
        return {
            "total_runs": len(self._snapshots),
            "average_duration": round(self.get_average_duration(), 4),
            "success_rate": round(self.get_success_rate(), 4),
            "slowest_nodes": self.get_slowest_nodes(5),
            "most_failed_nodes": self.get_most_failed_nodes(5),
        }

    def clear(self) -> None:
        """清除所有历史记录"""
        self._snapshots.clear()
        self._execution_times.clear()
        self._success_counts.clear()
        self._failure_counts.clear()


# ============================================================
# 工作流可视化器
# ============================================================

class WorkflowVisualizer:
    """
    工作流可视化器

    整合所有可视化组件，提供统一的工作流执行可视化接口。

    Usage:
        viz = WorkflowVisualizer(workflow_id="my_workflow")
        viz.update_node_status("node1", NodeVisualStatus.SUCCESS, duration=2.5)
        viz.update_node_status("node2", NodeVisualStatus.RUNNING)
        viz.add_error("node2", "处理超时")
        print(viz.render_ascii())
        print(viz.render_stats())
    """

    def __init__(
        self,
        workflow_id: str = "",
        total_nodes: int = 0,
    ) -> None:
        self.workflow_id = workflow_id
        self._colorizer = NodeStatusColorizer()
        self._renderer = ExecutionFlowRenderer(self._colorizer)
        self._progress = ProgressOverlay(total_nodes=total_nodes)
        self._error_highlighter = ErrorHighlighter()
        self._node_selector = NodeSelector()
        self._dashboard = StatsDashboard()
        self._nodes: Dict[str, NodeVisualInfo] = {}
        self._edges: List[EdgeVisualInfo] = []
        self._start_time: float = time.time()

    def update_node_status(
        self,
        node_id: str,
        status: NodeVisualStatus,
        name: str = "",
        duration: float = 0.0,
        error: Optional[str] = None,
        retry_count: int = 0,
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        """更新节点状态"""
        now = time.time()
        info = NodeVisualInfo(
            node_id=node_id,
            name=name or node_id,
            status=status,
            x=x,
            y=y,
            duration=duration,
            start_time=now - duration if duration > 0 else now,
            end_time=now if duration > 0 else 0.0,
            error=error,
            retry_count=retry_count,
        )
        self._nodes[node_id] = info
        self._progress.update_node(node_id, status)

        if error and status == NodeVisualStatus.FAILED:
            self._error_highlighter.add_error(
                node_id=node_id,
                message=error,
            )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        label: str = "",
        active: bool = False,
    ) -> None:
        """添加边"""
        self._edges.append(EdgeVisualInfo(
            source_id=source_id,
            target_id=target_id,
            label=label,
            active=active,
        ))

    def add_error(
        self,
        node_id: str,
        message: str,
        error_type: str = "",
    ) -> None:
        """添加错误"""
        self._error_highlighter.add_error(
            node_id=node_id,
            message=message,
            error_type=error_type,
        )

    def _build_snapshot(self) -> ExecutionSnapshot:
        """构建当前执行快照"""
        return ExecutionSnapshot(
            workflow_id=self.workflow_id,
            timestamp=time.time(),
            nodes=dict(self._nodes),
            edges=list(self._edges),
            global_start_time=self._start_time,
            global_progress=self._progress.get_progress(),
        )

    def render_ascii(self, width: int = 80) -> str:
        """渲染 ASCII 可视化"""
        snapshot = self._build_snapshot()
        return self._renderer.render_ascii(snapshot, width)

    def render_json(self) -> str:
        """渲染 JSON 可视化"""
        snapshot = self._build_snapshot()
        return self._renderer.render_json(snapshot)

    def render_progress_bar(self, width: int = 40) -> str:
        """渲染进度条"""
        return self._progress.render_progress_bar(width)

    def render_error_report(self) -> str:
        """渲染错误报告"""
        return self._error_highlighter.generate_report()

    def render_stats(self) -> str:
        """渲染统计摘要"""
        snapshot = self._build_snapshot()
        self._dashboard.record_execution(snapshot)
        return self._dashboard.generate_summary()

    def get_node_selector(self) -> NodeSelector:
        """获取节点选择器"""
        self._node_selector.set_snapshot(self._build_snapshot())
        return self._node_selector

    def get_progress(self) -> float:
        """获取当前进度"""
        return self._progress.get_progress()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        snapshot = self._build_snapshot()
        self._dashboard.record_execution(snapshot)
        return self._dashboard.generate_stats_dict()

    def reset(self) -> None:
        """重置可视化器"""
        self._nodes.clear()
        self._edges.clear()
        self._progress.reset()
        self._error_highlighter.clear()
        self._start_time = time.time()
