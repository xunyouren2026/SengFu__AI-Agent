"""
插件依赖解析模块

提供依赖图构建、循环依赖检测、拓扑排序和版本冲突检测功能。
使用 Kahn 算法进行拓扑排序。
仅使用 Python 标准库。
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .spec import PluginSpec, _check_version_compatible


# ---------------------------------------------------------------------------
# DependencyGraph - 依赖图
# ---------------------------------------------------------------------------
class DependencyGraph:
    """依赖图（邻接表表示）

    使用有向图表示插件间的依赖关系。
    边 (A, B) 表示 A 依赖 B（A -> B）。
    """

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependencies
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependents
        self._edge_versions: Dict[Tuple[str, str], str] = {}  # (node, dep) -> version constraint

    def add_node(self, name: str) -> None:
        """添加节点"""
        self._nodes.add(name)

    def remove_node(self, name: str) -> None:
        """移除节点及其所有边"""
        self._nodes.discard(name)
        # 移除出边
        for dep in self._edges.get(name, set()):
            self._reverse_edges[dep].discard(name)
            self._edge_versions.pop((name, dep), None)
        self._edges.pop(name, None)
        # 移除入边
        for dependent in self._reverse_edges.get(name, set()):
            self._edges[dependent].discard(name)
            self._edge_versions.pop((dependent, name), None)
        self._reverse_edges.pop(name, None)

    def add_edge(
        self, from_node: str, to_node: str, version: str = ""
    ) -> None:
        """添加边（from_node 依赖 to_node）"""
        self._nodes.add(from_node)
        self._nodes.add(to_node)
        self._edges[from_node].add(to_node)
        self._reverse_edges[to_node].add(from_node)
        if version:
            self._edge_versions[(from_node, to_node)] = version

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """移除边"""
        self._edges[from_node].discard(to_node)
        self._reverse_edges[to_node].discard(from_node)
        self._edge_versions.pop((from_node, to_node), None)

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def get_nodes(self) -> Set[str]:
        return set(self._nodes)

    def get_dependencies(self, name: str) -> Set[str]:
        """获取节点的依赖（出边）"""
        return set(self._edges.get(name, set()))

    def get_dependents(self, name: str) -> Set[str]:
        """获取依赖此节点的节点（入边）"""
        return set(self._reverse_edges.get(name, set()))

    def get_edge_version(self, from_node: str, to_node: str) -> str:
        """获取边的版本约束"""
        return self._edge_versions.get((from_node, to_node), "")

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """获取所有边"""
        result = []
        for node, deps in self._edges.items():
            for dep in deps:
                result.append((node, dep))
        return result

    def copy(self) -> "DependencyGraph":
        """创建图的副本"""
        new_graph = DependencyGraph()
        new_graph._nodes = set(self._nodes)
        new_graph._edges = defaultdict(set, {k: set(v) for k, v in self._edges.items()})
        new_graph._reverse_edges = defaultdict(set, {k: set(v) for k, v in self._reverse_edges.items()})
        new_graph._edge_versions = dict(self._edge_versions)
        return new_graph

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        return name in self._nodes


# ---------------------------------------------------------------------------
# ResolveResult - 解析结果
# ---------------------------------------------------------------------------
@dataclass
class ResolveResult:
    """依赖解析结果"""
    success: bool
    sorted: List[str] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    missing: List[Tuple[str, str]] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DependencyResolver - 依赖解析器
# ---------------------------------------------------------------------------
class DependencyResolver:
    """依赖解析器

    提供:
    - 循环依赖检测（DFS）
    - 拓扑排序（Kahn 算法）
    - 缺失依赖检测
    - 版本冲突检测
    """

    def resolve(
        self,
        graph: DependencyGraph,
        available_versions: Optional[Dict[str, str]] = None,
    ) -> ResolveResult:
        """解析依赖关系

        执行完整的依赖分析，包括循环检测、缺失检测和版本冲突检测。

        Args:
            graph: 依赖图
            available_versions: 可用插件的版本号 {name: version}

        Returns:
            ResolveResult
        """
        result = ResolveResult(success=True)
        available_versions = available_versions or {}

        # 1. 检测循环依赖
        cycles = self.detect_cycles(graph)
        if cycles:
            result.success = False
            result.cycles = cycles
            for cycle in cycles:
                result.errors.append(
                    f"检测到循环依赖: {' -> '.join(cycle)}"
                )

        # 2. 检测缺失依赖
        missing = self.missing_dependencies(graph)
        if missing:
            result.success = False
            result.missing = missing
            for node, dep in missing:
                result.errors.append(
                    f"插件 '{node}' 缺少依赖: '{dep}'"
                )

        # 3. 检测版本冲突
        if available_versions:
            conflicts = self.version_conflicts(graph, available_versions)
            if conflicts:
                result.success = False
                result.conflicts = conflicts
                for conflict in conflicts:
                    result.errors.append(
                        f"版本冲突: {conflict['description']}"
                    )

        # 4. 拓扑排序
        if result.success:
            sorted_list = self.topological_sort(graph)
            if sorted_list is None:
                result.success = False
                result.errors.append("拓扑排序失败（存在循环依赖）")
            else:
                result.sorted = sorted_list

        return result

    # ----- 循环依赖检测 -----

    def detect_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """检测循环依赖（DFS）

        Returns:
            循环路径列表，每个循环是一个节点列表
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {node: WHITE for node in graph.get_nodes()}
        parent: Dict[str, Optional[str]] = {node: None for node in graph.get_nodes()}
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            color[node] = GRAY
            for dep in graph.get_dependencies(node):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # 找到环，回溯路径
                    cycle = [dep, node]
                    current = node
                    while parent.get(current) is not None and parent[current] != dep:
                        current = parent[current]  # type: ignore[assignment]
                        cycle.append(current)
                    cycle.append(dep)
                    cycle.reverse()
                    cycles.append(cycle)
                elif color[dep] == WHITE:
                    parent[dep] = node
                    dfs(dep)
            color[node] = BLACK

        for node in graph.get_nodes():
            if color[node] == WHITE:
                dfs(node)

        # 去重
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            # 规范化循环表示
            min_idx = cycle.index(min(cycle[:-1]))
            normalized = tuple(cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]])
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized))

        return unique_cycles

    # ----- 拓扑排序（Kahn 算法）-----

    def topological_sort(self, graph: DependencyGraph) -> Optional[List[str]]:
        """拓扑排序（Kahn 算法）

        Returns:
            排序后的节点列表，如果存在循环则返回 None
        """
        # 计算入度
        in_degree: Dict[str, int] = {node: 0 for node in graph.get_nodes()}
        for from_node, deps in graph._edges.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] = in_degree.get(dep, 0)

        # 重新计算：入度 = 被依赖的次数
        in_degree = {node: 0 for node in graph.get_nodes()}
        for from_node in graph.get_nodes():
            for dep in graph.get_dependencies(from_node):
                if dep in in_degree:
                    pass  # dep 被 from_node 依赖

        # 正确的入度计算：对于每个节点，入度 = 依赖它的节点数
        # 在我们的图中，边 (A, B) 表示 A 依赖 B
        # 拓扑排序应该先处理被依赖的节点
        # 所以入度应该 = 有多少节点依赖它（即 reverse_edges 的大小）
        in_degree = {node: len(graph.get_dependents(node)) for node in graph.get_nodes()}

        # 不对，让我重新思考。在依赖图中：
        # A -> B 表示 A 依赖 B
        # 拓扑排序应该让 B 在 A 前面（先加载被依赖的）
        # Kahn 算法中，入度为 0 的节点先处理
        # 对于 A -> B，B 的入度应该 +1（因为 A 指向 B）
        # 所以 in_degree[B] = 有多少节点指向 B = len(graph.get_dependents(B))

        # 重新计算
        in_degree = {node: 0 for node in graph.get_nodes()}
        for node in graph.get_nodes():
            for dep in graph.get_dependencies(node):
                if dep in in_degree:
                    in_degree[dep] += 1

        # 初始化队列：入度为 0 的节点（没有被任何节点依赖的叶子节点）
        # 等等，这不对。入度为 0 意味着没有节点依赖它，但它可能依赖别人。
        # 我们需要的是：没有依赖的节点先处理。

        # 让我重新定义：
        # 在 Kahn 算法中，对于有向边 u -> v:
        # - u 必须在 v 之前处理
        # - in_degree[v] += 1
        # 在我们的图中，边 A -> B 表示 A 依赖 B:
        # - B 必须在 A 之前处理
        # - 所以 in_degree[A] += 1（A 有一个依赖）

        in_degree = {node: 0 for node in graph.get_nodes()}
        for node in graph.get_nodes():
            in_degree[node] = len(graph.get_dependencies(node))

        queue = deque()
        for node in graph.get_nodes():
            if in_degree[node] == 0:
                queue.append(node)

        sorted_list: List[str] = []
        while queue:
            # 优先按名称排序以确保确定性
            queue_list = sorted(queue)
            queue.clear()
            for node in queue_list:
                sorted_list.append(node)

            for node in queue_list:
                # 对于依赖此节点的所有节点，减少入度
                for dependent in graph.get_dependents(node):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(sorted_list) != len(graph.get_nodes()):
            return None

        return sorted_list

    # ----- 缺失依赖检测 -----

    def missing_dependencies(
        self, graph: DependencyGraph
    ) -> List[Tuple[str, str]]:
        """找出缺失的依赖

        Returns:
            [(插件名, 缺失的依赖名), ...]
        """
        missing = []
        for node in graph.get_nodes():
            for dep in graph.get_dependencies(node):
                if dep not in graph.get_nodes():
                    missing.append((node, dep))
        return missing

    # ----- 版本冲突检测 -----

    def version_conflicts(
        self,
        graph: DependencyGraph,
        available_versions: Dict[str, str],
    ) -> List[dict]:
        """检测版本冲突

        检查每个依赖的版本约束是否与可用版本兼容。

        Args:
            graph: 依赖图
            available_versions: {插件名: 实际版本}

        Returns:
            冲突列表
        """
        conflicts = []
        seen = set()

        for node in graph.get_nodes():
            for dep in graph.get_dependencies(node):
                version_constraint = graph.get_edge_version(node, dep)
                if not version_constraint:
                    continue

                actual_version = available_versions.get(dep, "")
                if not actual_version:
                    continue

                conflict_key = (dep, version_constraint, actual_version)
                if conflict_key in seen:
                    continue
                seen.add(conflict_key)

                if not _check_version_compatible(actual_version, version_constraint):
                    conflicts.append({
                        "plugin": node,
                        "dependency": dep,
                        "required": version_constraint,
                        "actual": actual_version,
                        "description": (
                            f"插件 '{node}' 要求 '{dep}' 版本 {version_constraint}，"
                            f"但实际版本为 {actual_version}"
                        ),
                    })

        return conflicts

    # ----- 从 PluginSpec 列表构建依赖图 -----

    @staticmethod
    def build_graph(plugins: List[PluginSpec]) -> DependencyGraph:
        """从插件规范列表构建依赖图"""
        graph = DependencyGraph()
        for spec in plugins:
            graph.add_node(spec.name)
            for dep in spec.dependencies:
                graph.add_edge(spec.name, dep.name, dep.version)
        return graph
