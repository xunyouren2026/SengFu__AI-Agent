"""
DAG任务图引擎

提供有向无环图（DAG）的定义、验证和执行顺序计算。
支持拓扑排序（Kahn算法）、环检测、上下游节点查询等功能。
"""

import copy
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class NodeState(Enum):
    """节点状态枚举"""
    PENDING = "pending"         # 等待中
    READY = "ready"             # 就绪（依赖已满足）
    RUNNING = "running"         # 运行中
    SUCCESS = "success"         # 成功
    FAILED = "failed"           # 失败
    SKIPPED = "skipped"         # 跳过
    CANCELLED = "cancelled"     # 已取消


class DAGError(Exception):
    """DAG操作异常"""
    pass


class CycleDetectedError(DAGError):
    """检测到环"""
    pass


class NodeNotFoundError(DAGError):
    """节点不存在"""
    pass


class DuplicateNodeError(DAGError):
    """节点ID重复"""
    pass


@dataclass
class DAGNode:
    """
    DAG节点

    Attributes:
        id: 节点唯一标识
        name: 节点名称
        node_type: 节点类型（task/llm/tool/condition/loop/parallel等）
        config: 节点配置
        dependencies: 依赖的节点ID列表
        state: 当前状态
        metadata: 元数据
        retry_count: 重试次数
        max_retries: 最大重试次数
        timeout: 超时时间（秒），None表示无超时
    """
    id: str
    name: str = ""
    node_type: str = "task"
    config: Dict[str, Any] = dataclass_field(default_factory=dict)
    dependencies: List[str] = dataclass_field(default_factory=list)
    state: NodeState = NodeState.PENDING
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 0
    timeout: Optional[float] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.id

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "config": copy.deepcopy(self.config),
            "dependencies": list(self.dependencies),
            "state": self.state.value,
            "metadata": copy.deepcopy(self.metadata),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGNode":
        """从字典创建"""
        data = copy.deepcopy(data)
        state_str = data.pop("state", "pending")
        state = NodeState(state_str) if isinstance(state_str, str) else state_str
        return cls(state=state, **data)

    def __repr__(self) -> str:
        return f"<DAGNode id={self.id!r} type={self.node_type!r} state={self.state.value}>"


@dataclass
class DAGEdge:
    """
    DAG边

    Attributes:
        from_node: 源节点ID
        to_node: 目标节点ID
        condition: 边条件（可选），条件为True时边才生效
        data_mapping: 数据映射（源节点输出 -> 目标节点输入）
    """
    from_node: str
    to_node: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    condition_expr: Optional[str] = None
    data_mapping: Dict[str, str] = dataclass_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "from": self.from_node,
            "to": self.to_node,
        }
        if self.condition_expr:
            result["condition"] = self.condition_expr
        if self.data_mapping:
            result["data_mapping"] = copy.deepcopy(self.data_mapping)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGEdge":
        """从字典创建"""
        data = copy.deepcopy(data)
        from_node = data.pop("from", data.pop("from_node", ""))
        to_node = data.pop("to", data.pop("to_node", ""))
        condition_expr = data.pop("condition", None)
        data_mapping = data.pop("data_mapping", {})
        return cls(
            from_node=from_node,
            to_node=to_node,
            condition_expr=condition_expr,
            data_mapping=data_mapping,
        )

    def __repr__(self) -> str:
        return f"<DAGEdge {self.from_node!r} -> {self.to_node!r}>"


class DAGEngine:
    """
    DAG任务图引擎

    管理工作流的有向无环图结构，提供：
    - 节点和边的增删改查
    - 拓扑排序（Kahn算法）
    - 环检测
    - 执行顺序计算
    - 上下游节点查询
    - DAG合法性验证
    - 序列化/反序列化

    Usage:
        dag = DAGEngine()
        dag.add_node(DAGNode(id="start", node_type="task"))
        dag.add_node(DAGNode(id="process", node_type="task"))
        dag.add_edge("start", "process")
        order = dag.get_execution_order()
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._nodes: Dict[str, DAGNode] = {}
        self._edges: List[DAGEdge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # 出边
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # 入边

    # ============================================================
    # 节点管理
    # ============================================================

    def add_node(self, node: DAGNode) -> "DAGEngine":
        """
        添加节点

        Args:
            node: DAG节点

        Returns:
            self（支持链式调用）

        Raises:
            DuplicateNodeError: 节点ID已存在
        """
        if node.id in self._nodes:
            raise DuplicateNodeError(f"节点ID已存在: {node.id}")

        self._nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = set()
        if node.id not in self._reverse_adjacency:
            self._reverse_adjacency[node.id] = set()
        return self

    def remove_node(self, node_id: str) -> "DAGEngine":
        """
        删除节点及其相关的所有边

        Args:
            node_id: 节点ID

        Returns:
            self

        Raises:
            NodeNotFoundError: 节点不存在
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"节点不存在: {node_id}")

        del self._nodes[node_id]

        # 删除相关边
        self._edges = [
            e for e in self._edges
            if e.from_node != node_id and e.to_node != node_id
        ]

        # 更新邻接表
        if node_id in self._adjacency:
            for target in self._adjacency[node_id]:
                self._reverse_adjacency[target].discard(node_id)
            del self._adjacency[node_id]

        if node_id in self._reverse_adjacency:
            for source in self._reverse_adjacency[node_id]:
                self._adjacency[source].discard(node_id)
            del self._reverse_adjacency[node_id]

        return self

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        """获取节点"""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """检查节点是否存在"""
        return node_id in self._nodes

    def get_all_nodes(self) -> List[DAGNode]:
        """获取所有节点"""
        return list(self._nodes.values())

    def node_count(self) -> int:
        """节点数量"""
        return len(self._nodes)

    # ============================================================
    # 边管理
    # ============================================================

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        condition_expr: Optional[str] = None,
        data_mapping: Optional[Dict[str, str]] = None,
    ) -> "DAGEngine":
        """
        添加边

        Args:
            from_node: 源节点ID
            to_node: 目标节点ID
            condition: 边条件函数
            condition_expr: 条件表达式（用于序列化）
            data_mapping: 数据映射

        Returns:
            self

        Raises:
            NodeNotFoundError: 源或目标节点不存在
            CycleDetectedError: 添加边会形成环
        """
        if from_node not in self._nodes:
            raise NodeNotFoundError(f"源节点不存在: {from_node}")
        if to_node not in self._nodes:
            raise NodeNotFoundError(f"目标节点不存在: {to_node}")

        edge = DAGEdge(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            condition_expr=condition_expr,
            data_mapping=data_mapping or {},
        )

        # 检查是否已存在
        for existing in self._edges:
            if existing.from_node == from_node and existing.to_node == to_node:
                return self

        self._edges.append(edge)
        self._adjacency[from_node].add(to_node)
        self._reverse_adjacency[to_node].add(from_node)

        # 检查是否形成环
        if self.detect_cycles():
            # 回滚
            self._edges.remove(edge)
            self._adjacency[from_node].discard(to_node)
            self._reverse_adjacency[to_node].discard(from_node)
            raise CycleDetectedError(
                f"添加边 {from_node} -> {to_node} 会形成环"
            )

        # 更新目标节点的依赖列表
        if from_node not in self._nodes[to_node].dependencies:
            self._nodes[to_node].dependencies.append(from_node)

        return self

    def remove_edge(self, from_node: str, to_node: str) -> "DAGEngine":
        """
        删除边

        Args:
            from_node: 源节点ID
            to_node: 目标节点ID

        Returns:
            self
        """
        self._edges = [
            e for e in self._edges
            if not (e.from_node == from_node and e.to_node == to_node)
        ]
        self._adjacency[from_node].discard(to_node)
        self._reverse_adjacency[to_node].discard(from_node)

        # 更新依赖列表
        if to_node in self._nodes:
            node = self._nodes[to_node]
            if from_node in node.dependencies:
                node.dependencies.remove(from_node)

        return self

    def get_edges(self, from_node: Optional[str] = None) -> List[DAGEdge]:
        """
        获取边

        Args:
            from_node: 如果指定，只返回从该节点出发的边

        Returns:
            边列表
        """
        if from_node:
            return [e for e in self._edges if e.from_node == from_node]
        return list(self._edges)

    def get_incoming_edges(self, to_node: str) -> List[DAGEdge]:
        """获取指向某节点的所有边"""
        return [e for e in self._edges if e.to_node == to_node]

    def get_outgoing_edges(self, from_node: str) -> List[DAGEdge]:
        """获取从某节点出发的所有边"""
        return [e for e in self._edges if e.from_node == from_node]

    def edge_count(self) -> int:
        """边数量"""
        return len(self._edges)

    # ============================================================
    # 图算法
    # ============================================================

    def topological_sort(self) -> List[str]:
        """
        拓扑排序（Kahn算法）

        Returns:
            拓扑排序后的节点ID列表

        Raises:
            CycleDetectedError: 图中存在环
        """
        # 计算入度
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for edge in self._edges:
            in_degree[edge.to_node] = in_degree.get(edge.to_node, 0) + 1

        # 初始化队列（入度为0的节点）
        queue = deque()
        for nid, degree in in_degree.items():
            if degree == 0:
                queue.append(nid)

        result = []
        while queue:
            # 按字母序处理以保证确定性
            queue = deque(sorted(queue))
            node_id = queue.popleft()
            result.append(node_id)

            for neighbor in self._adjacency.get(node_id, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._nodes):
            raise CycleDetectedError("图中存在环，无法完成拓扑排序")

        return result

    def get_execution_order(self) -> List[List[str]]:
        """
        获取分层执行顺序

        同一层的节点可以并行执行。

        Returns:
            分层节点ID列表，每层是一个列表
        """
        try:
            sorted_ids = self.topological_sort()
        except CycleDetectedError:
            raise

        # 计算层级
        levels: Dict[str, int] = {}
        for node_id in sorted_ids:
            node = self._nodes[node_id]
            if not node.dependencies:
                levels[node_id] = 0
            else:
                max_dep_level = -1
                for dep_id in node.dependencies:
                    if dep_id in levels:
                        max_dep_level = max(max_dep_level, levels[dep_id])
                levels[node_id] = max_dep_level + 1

        # 按层级分组
        max_level = max(levels.values()) if levels else 0
        result: List[List[str]] = [[] for _ in range(max_level + 1)]
        for node_id, level in levels.items():
            result[level].append(node_id)

        # 每层内排序以保证确定性
        for layer in result:
            layer.sort()

        return result

    def detect_cycles(self) -> bool:
        """
        检测图中是否存在环（DFS算法）

        Returns:
            True表示存在环
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in self._nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            for neighbor in self._adjacency.get(node_id, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node_id] = BLACK
            return False

        for node_id in self._nodes:
            if color[node_id] == WHITE:
                if dfs(node_id):
                    return True
        return False

    def get_root_nodes(self) -> List[DAGNode]:
        """
        获取根节点（没有入边的节点）

        Returns:
            根节点列表
        """
        roots = []
        for node_id, node in self._nodes.items():
            if not self._reverse_adjacency.get(node_id):
                roots.append(node)
        return roots

    def get_leaf_nodes(self) -> List[DAGNode]:
        """
        获取叶节点（没有出边的节点）

        Returns:
            叶节点列表
        """
        leaves = []
        for node_id, node in self._nodes.items():
            if not self._adjacency.get(node_id):
                leaves.append(node)
        return leaves

    def get_downstream(self, node_id: str) -> Set[str]:
        """
        获取下游节点（BFS）

        Args:
            node_id: 起始节点ID

        Returns:
            所有下游节点ID集合
        """
        visited: Set[str] = set()
        queue = deque([node_id])

        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def get_upstream(self, node_id: str) -> Set[str]:
        """
        获取上游节点（BFS）

        Args:
            node_id: 起始节点ID

        Returns:
            所有上游节点ID集合
        """
        visited: Set[str] = set()
        queue = deque([node_id])

        while queue:
            current = queue.popleft()
            for neighbor in self._reverse_adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def get_reachable_nodes(self, from_node: str) -> Set[str]:
        """获取从指定节点可达的所有节点"""
        return self.get_downstream(from_node)

    # ============================================================
    # 验证
    # ============================================================

    def validate(self) -> List[str]:
        """
        验证DAG合法性

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors: List[str] = []

        # 检查是否有节点
        if not self._nodes:
            errors.append("DAG中没有节点")
            return errors

        # 检查环
        if self.detect_cycles():
            errors.append("DAG中存在环")

        # 检查边的引用
        for edge in self._edges:
            if edge.from_node not in self._nodes:
                errors.append(f"边引用了不存在的源节点: {edge.from_node}")
            if edge.to_node not in self._nodes:
                errors.append(f"边引用了不存在的目标节点: {edge.to_node}")

        # 检查孤立节点（没有入边也没有出边，且不是唯一节点）
        if len(self._nodes) > 1:
            for node_id in self._nodes:
                has_in = bool(self._reverse_adjacency.get(node_id))
                has_out = bool(self._adjacency.get(node_id))
                if not has_in and not has_out:
                    errors.append(f"节点 {node_id} 是孤立的（无入边也无出边）")

        # 检查节点ID格式
        for node_id in self._nodes:
            if not node_id or not isinstance(node_id, str):
                errors.append(f"节点ID无效: {node_id!r}")

        # 检查自环
        for edge in self._edges:
            if edge.from_node == edge.to_node:
                errors.append(f"节点 {edge.from_node} 存在自环")

        return errors

    def is_valid(self) -> bool:
        """检查DAG是否合法"""
        return len(self.validate()) == 0

    # ============================================================
    # 序列化
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为字典

        Returns:
            DAG的字典表示
        """
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGEngine":
        """
        从字典导入

        Args:
            data: DAG字典表示

        Returns:
            DAGEngine实例
        """
        data = copy.deepcopy(data)
        name = data.get("name", "")
        engine = cls(name=name)

        # 添加节点
        for node_data in data.get("nodes", []):
            node = DAGNode.from_dict(node_data)
            engine.add_node(node)

        # 添加边
        for edge_data in data.get("edges", []):
            edge = DAGEdge.from_dict(edge_data)
            engine.add_edge(edge.from_node, edge.to_node)

        return engine

    # ============================================================
    # 其他
    # ============================================================

    def reset_states(self) -> None:
        """重置所有节点状态为PENDING"""
        for node in self._nodes.values():
            node.state = NodeState.PENDING
            node.retry_count = 0

    def get_nodes_by_type(self, node_type: str) -> List[DAGNode]:
        """获取指定类型的所有节点"""
        return [
            n for n in self._nodes.values()
            if n.node_type == node_type
        ]

    def get_subgraph(self, root_id: str) -> "DAGEngine":
        """
        获取以指定节点为根的子图

        Args:
            root_id: 根节点ID

        Returns:
            新的DAGEngine实例
        """
        reachable = self.get_downstream(root_id)
        reachable.add(root_id)

        sub_dag = DAGEngine(name=f"{self.name}_sub_{root_id}")
        for node_id in reachable:
            if node_id in self._nodes:
                sub_dag.add_node(copy.deepcopy(self._nodes[node_id]))

        for edge in self._edges:
            if edge.from_node in reachable and edge.to_node in reachable:
                sub_dag.add_edge(
                    edge.from_node, edge.to_node,
                    condition=edge.condition,
                    condition_expr=edge.condition_expr,
                    data_mapping=edge.data_mapping,
                )

        return sub_dag

    def __len__(self) -> int:
        return self.node_count()

    def __contains__(self, node_id: str) -> bool:
        return self.has_node(node_id)

    def __repr__(self) -> str:
        return (
            f"<DAGEngine name={self.name!r} "
            f"nodes={self.node_count()} edges={self.edge_count()}>"
        )
