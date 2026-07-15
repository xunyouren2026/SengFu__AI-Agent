"""
工作流构建器

将分配结果转换为执行工作流图，支持多种工作流模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class WorkflowNodeType(Enum):
    """工作流节点类型"""
    TASK = auto()
    DECISION = auto()
    PARALLEL = auto()
    JOIN = auto()
    START = auto()
    END = auto()


@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    node_type: WorkflowNodeType
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    """工作流边"""
    from_node: str
    to_node: str
    condition: Optional[str] = None
    weight: float = 1.0


@dataclass
class Workflow:
    """工作流"""
    workflow_id: str
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: List[WorkflowEdge] = field(default_factory=list)
    
    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: WorkflowEdge) -> None:
        self.edges.append(edge)
    
    def get_start_nodes(self) -> List[WorkflowNode]:
        """获取开始节点"""
        return [n for n in self.nodes.values() if n.node_type == WorkflowNodeType.START]
    
    def get_end_nodes(self) -> List[WorkflowNode]:
        """获取结束节点"""
        return [n for n in self.nodes.values() if n.node_type == WorkflowNodeType.END]
    
    def get_successors(self, node_id: str) -> List[WorkflowNode]:
        """获取后继节点"""
        successors = []
        for edge in self.edges:
            if edge.from_node == node_id and edge.to_node in self.nodes:
                successors.append(self.nodes[edge.to_node])
        return successors
    
    def get_predecessors(self, node_id: str) -> List[WorkflowNode]:
        """获取前驱节点"""
        predecessors = []
        for edge in self.edges:
            if edge.to_node == node_id and edge.from_node in self.nodes:
                predecessors.append(self.nodes[edge.from_node])
        return predecessors


class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self):
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_dependencies: Dict[str, Set[str]] = {}  # task_id -> dependencies
    
    def set_assignments(self, assignments: Dict[str, str]) -> None:
        """设置任务分配"""
        self.task_assignments = assignments
    
    def set_dependencies(self, dependencies: Dict[str, Set[str]]) -> None:
        """设置任务依赖"""
        self.task_dependencies = dependencies
    
    def build_workflow(self, workflow_id: str) -> Workflow:
        """构建工作流"""
        workflow = Workflow(workflow_id=workflow_id)
        
        # 添加开始节点
        start_node = WorkflowNode(
            node_id="start",
            node_type=WorkflowNodeType.START,
            name="Start"
        )
        workflow.add_node(start_node)
        
        # 添加任务节点
        for task_id, agent_id in self.task_assignments.items():
            node = WorkflowNode(
                node_id=f"task_{task_id}",
                node_type=WorkflowNodeType.TASK,
                task_id=task_id,
                agent_id=agent_id,
                name=task_id
            )
            workflow.add_node(node)
        
        # 添加边
        # 从开始节点连接到没有依赖的任务
        for task_id in self.task_assignments:
            deps = self.task_dependencies.get(task_id, set())
            if not deps:
                edge = WorkflowEdge(from_node="start", to_node=f"task_{task_id}")
                workflow.add_edge(edge)
        
        # 添加任务间的依赖边
        for task_id, deps in self.task_dependencies.items():
            for dep in deps:
                if dep in self.task_assignments and task_id in self.task_assignments:
                    edge = WorkflowEdge(
                        from_node=f"task_{dep}",
                        to_node=f"task_{task_id}"
                    )
                    workflow.add_edge(edge)
        
        # 添加结束节点
        end_node = WorkflowNode(
            node_id="end",
            node_type=WorkflowNodeType.END,
            name="End"
        )
        workflow.add_node(end_node)
        
        # 连接到结束节点
        all_tasks = set(self.task_assignments.keys())
        for task_id in self.task_assignments:
            # 检查是否是最后一个任务（没有其他任务依赖它）
            is_last = True
            for other_task, other_deps in self.task_dependencies.items():
                if task_id in other_deps:
                    is_last = False
                    break
            
            if is_last:
                edge = WorkflowEdge(from_node=f"task_{task_id}", to_node="end")
                workflow.add_edge(edge)
        
        return workflow
    
    def build_parallel_workflow(self, workflow_id: str) -> Workflow:
        """构建并行工作流"""
        workflow = Workflow(workflow_id=workflow_id)
        
        # 开始节点
        workflow.add_node(WorkflowNode("start", WorkflowNodeType.START, name="Start"))
        
        # 并行分支节点
        workflow.add_node(WorkflowNode("parallel", WorkflowNodeType.PARALLEL, name="Parallel"))
        workflow.add_edge(WorkflowEdge("start", "parallel"))
        
        # 任务节点
        for task_id, agent_id in self.task_assignments.items():
            node = WorkflowNode(
                node_id=f"task_{task_id}",
                node_type=WorkflowNodeType.TASK,
                task_id=task_id,
                agent_id=agent_id,
                name=task_id
            )
            workflow.add_node(node)
            workflow.add_edge(WorkflowEdge("parallel", f"task_{task_id}"))
        
        # 汇聚节点
        workflow.add_node(WorkflowNode("join", WorkflowNodeType.JOIN, name="Join"))
        
        for task_id in self.task_assignments:
            workflow.add_edge(WorkflowEdge(f"task_{task_id}", "join"))
        
        # 结束节点
        workflow.add_node(WorkflowNode("end", WorkflowNodeType.END, name="End"))
        workflow.add_edge(WorkflowEdge("join", "end"))
        
        return workflow


class WorkflowOptimizer:
    """工作流优化器"""
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
    
    def optimize_critical_path(self) -> List[str]:
        """识别关键路径"""
        # 简化的关键路径分析
        # 返回节点ID列表
        
        # 计算每个节点的最早开始时间
        earliest_start: Dict[str, float] = {}
        
        for node_id in self._topological_sort():
            node = self.workflow.nodes[node_id]
            if node.node_type == WorkflowNodeType.START:
                earliest_start[node_id] = 0.0
            else:
                preds = self.workflow.get_predecessors(node_id)
                max_end = 0.0
                for pred in preds:
                    duration = pred.metadata.get("duration", 1.0)
                    pred_start = earliest_start.get(pred.node_id, 0.0)
                    max_end = max(max_end, pred_start + duration)
                earliest_start[node_id] = max_end
        
        # 找到结束时间最大的路径
        critical_path: List[str] = []
        current = self._find_end_node()
        
        while current and current.node_type != WorkflowNodeType.START:
            critical_path.append(current.node_id)
            preds = self.workflow.get_predecessors(current.node_id)
            if preds:
                current = max(preds, key=lambda n: earliest_start.get(n.node_id, 0.0))
            else:
                break
        
        critical_path.reverse()
        return critical_path
    
    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.workflow.nodes}
        
        for edge in self.workflow.edges:
            in_degree[edge.to_node] += 1
        
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: List[str] = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for edge in self.workflow.edges:
                if edge.from_node == current:
                    in_degree[edge.to_node] -= 1
                    if in_degree[edge.to_node] == 0:
                        queue.append(edge.to_node)
        
        return result
    
    def _find_end_node(self) -> Optional[WorkflowNode]:
        """找到结束节点"""
        for node in self.workflow.nodes.values():
            if node.node_type == WorkflowNodeType.END:
                return node
        return None
