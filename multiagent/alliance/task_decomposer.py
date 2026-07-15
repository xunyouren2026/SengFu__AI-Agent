"""
任务分解器

使用规则将大任务递归分解为DAG（有向无环图），支持多种分解策略。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class DecompositionStrategy(Enum):
    """分解策略"""
    SEQUENTIAL = auto()      # 顺序分解
    PARALLEL = auto()        # 并行分解
    HIERARCHICAL = auto()    # 层次分解
    FUNCTIONAL = auto()      # 功能分解
    DATA_DRIVEN = auto()     # 数据驱动分解


@dataclass
class SubTask:
    """子任务"""
    task_id: str
    name: str = ""
    description: str = ""
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_effort: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class TaskGraph:
    """任务图"""
    root_task_id: str
    subtasks: Dict[str, SubTask] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> dependent task_ids
    
    def add_subtask(self, subtask: SubTask) -> None:
        """添加子任务"""
        self.subtasks[subtask.task_id] = subtask
        if subtask.task_id not in self.edges:
            self.edges[subtask.task_id] = set()
    
    def add_dependency(self, from_task: str, to_task: str) -> None:
        """添加依赖关系"""
        if to_task not in self.edges:
            self.edges[to_task] = set()
        self.edges[to_task].add(from_task)
        if to_task in self.subtasks:
            self.subtasks[to_task].dependencies.add(from_task)
    
    def get_topological_order(self) -> List[str]:
        """获取拓扑排序"""
        in_degree: Dict[str, int] = {tid: 0 for tid in self.subtasks}
        for deps in self.edges.values():
            for dep in deps:
                in_degree[dep] += 1
        
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: List[str] = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for tid, deps in self.edges.items():
                if current in deps:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
        
        return result
    
    def get_parallel_groups(self) -> List[Set[str]]:
        """获取可并行执行的组"""
        order = self.get_topological_order()
        groups: List[Set[str]] = []
        completed: Set[str] = set()
        
        while len(completed) < len(order):
            group: Set[str] = set()
            for tid in order:
                if tid in completed:
                    continue
                task = self.subtasks[tid]
                if task.dependencies.issubset(completed):
                    group.add(tid)
            
            if group:
                groups.append(group)
                completed.update(group)
            else:
                break
        
        return groups


@dataclass
class DecompositionRule:
    """分解规则"""
    rule_id: str
    name: str
    pattern: str  # 正则表达式模式
    strategy: DecompositionStrategy
    decomposer: Callable[[str, str], List[SubTask]]
    priority: int = 1


class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self):
        self.rules: List[DecompositionRule] = []
        self.decomposition_history: List[TaskGraph] = []
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """注册默认分解规则"""
        # 分析-规划-执行-验证 规则
        self.add_rule(DecompositionRule(
            rule_id="analyze_plan_execute",
            name="分析-规划-执行-验证",
            pattern=r".*",
            strategy=DecompositionStrategy.SEQUENTIAL,
            decomposer=self._decompose_analyze_plan_execute,
            priority=1
        ))
        
        # 数据收集-处理-分析 规则
        self.add_rule(DecompositionRule(
            rule_id="data_pipeline",
            name="数据处理流水线",
            pattern=r"(data|collect|gather|process).*",
            strategy=DecompositionStrategy.PIPELINE,
            decomposer=self._decompose_data_pipeline,
            priority=2
        ))
        
        # 搜索-评估-决策 规则
        self.add_rule(DecompositionRule(
            rule_id="search_evaluate_decide",
            name="搜索-评估-决策",
            pattern=r"(search|find|research|investigate).*",
            strategy=DecompositionStrategy.FUNCTIONAL,
            decomposer=self._decompose_search_evaluate,
            priority=2
        ))
    
    def add_rule(self, rule: DecompositionRule) -> None:
        """添加分解规则"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)
    
    def decompose(
        self,
        task_id: str,
        task_description: str,
        strategy: Optional[DecompositionStrategy] = None
    ) -> TaskGraph:
        """
        分解任务
        
        Args:
            task_id: 任务ID
            task_description: 任务描述
            strategy: 指定分解策略，None则自动选择
        """
        graph = TaskGraph(root_task_id=task_id)
        
        # 查找匹配的规则
        matching_rule: Optional[DecompositionRule] = None
        
        if strategy:
            for rule in self.rules:
                if rule.strategy == strategy:
                    matching_rule = rule
                    break
        else:
            for rule in self.rules:
                if re.match(rule.pattern, task_description, re.IGNORECASE):
                    matching_rule = rule
                    break
        
        if matching_rule:
            subtasks = matching_rule.decomposer(task_id, task_description)
            
            # 添加子任务
            for subtask in subtasks:
                graph.add_subtask(subtask)
            
            # 添加依赖关系
            self._add_dependencies(graph, subtasks, matching_rule.strategy)
        else:
            # 没有匹配规则，创建单节点
            graph.add_subtask(SubTask(
                task_id=f"{task_id}_atomic",
                name=task_description,
                description=task_description
            ))
        
        self.decomposition_history.append(graph)
        return graph
    
    def _add_dependencies(
        self,
        graph: TaskGraph,
        subtasks: List[SubTask],
        strategy: DecompositionStrategy
    ) -> None:
        """根据策略添加依赖关系"""
        if strategy == DecompositionStrategy.SEQUENTIAL:
            # 顺序依赖：每个任务依赖前一个
            for i in range(1, len(subtasks)):
                graph.add_dependency(subtasks[i-1].task_id, subtasks[i].task_id)
        
        elif strategy == DecompositionStrategy.PIPELINE:
            # 流水线依赖
            for i in range(1, len(subtasks)):
                graph.add_dependency(subtasks[i-1].task_id, subtasks[i].task_id)
        
        elif strategy == DecompositionStrategy.FUNCTIONAL:
            # 功能分解：搜索和评估并行，然后决策
            # 假设最后一个任务是决策
            if len(subtasks) >= 2:
                decision_task = subtasks[-1]
                for subtask in subtasks[:-1]:
                    graph.add_dependency(subtask.task_id, decision_task.task_id)
        
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            # 层次分解：子任务之间可能有复杂依赖
            pass  # 依赖由分解器显式设置
    
    def _decompose_analyze_plan_execute(
        self,
        task_id: str,
        description: str
    ) -> List[SubTask]:
        """分析-规划-执行-验证分解"""
        return [
            SubTask(
                task_id=f"{task_id}_analyze",
                name="分析",
                description=f"分析任务: {description}",
                required_capabilities={"analysis"},
                estimated_effort=1.0
            ),
            SubTask(
                task_id=f"{task_id}_plan",
                name="规划",
                description=f"制定执行计划",
                required_capabilities={"planning"},
                estimated_effort=0.5,
                dependencies={f"{task_id}_analyze"}
            ),
            SubTask(
                task_id=f"{task_id}_execute",
                name="执行",
                description=f"执行任务",
                required_capabilities={"execution"},
                estimated_effort=2.0,
                dependencies={f"{task_id}_plan"}
            ),
            SubTask(
                task_id=f"{task_id}_verify",
                name="验证",
                description=f"验证结果",
                required_capabilities={"verification"},
                estimated_effort=0.5,
                dependencies={f"{task_id}_execute"}
            )
        ]
    
    def _decompose_data_pipeline(
        self,
        task_id: str,
        description: str
    ) -> List[SubTask]:
        """数据处理流水线分解"""
        return [
            SubTask(
                task_id=f"{task_id}_collect",
                name="数据收集",
                description="收集原始数据",
                required_capabilities={"data_collection"},
                estimated_effort=1.0
            ),
            SubTask(
                task_id=f"{task_id}_clean",
                name="数据清洗",
                description="清洗和预处理数据",
                required_capabilities={"data_cleaning"},
                estimated_effort=1.0,
                dependencies={f"{task_id}_collect"}
            ),
            SubTask(
                task_id=f"{task_id}_process",
                name="数据处理",
                description="处理数据",
                required_capabilities={"data_processing"},
                estimated_effort=2.0,
                dependencies={f"{task_id}_clean"}
            ),
            SubTask(
                task_id=f"{task_id}_analyze",
                name="数据分析",
                description="分析处理后的数据",
                required_capabilities={"data_analysis"},
                estimated_effort=1.5,
                dependencies={f"{task_id}_process"}
            )
        ]
    
    def _decompose_search_evaluate(
        self,
        task_id: str,
        description: str
    ) -> List[SubTask]:
        """搜索-评估-决策分解"""
        return [
            SubTask(
                task_id=f"{task_id}_search",
                name="信息搜索",
                description="搜索相关信息",
                required_capabilities={"search"},
                estimated_effort=1.5
            ),
            SubTask(
                task_id=f"{task_id}_gather",
                name="信息收集",
                description="收集补充信息",
                required_capabilities={"collection"},
                estimated_effort=1.0
            ),
            SubTask(
                task_id=f"{task_id}_evaluate",
                name="评估分析",
                description="评估收集的信息",
                required_capabilities={"evaluation"},
                estimated_effort=1.0,
                dependencies={f"{task_id}_search", f"{task_id}_gather"}
            ),
            SubTask(
                task_id=f"{task_id}_decide",
                name="决策",
                description="做出决策",
                required_capabilities={"decision_making"},
                estimated_effort=0.5,
                dependencies={f"{task_id}_evaluate"}
            )
        ]
    
    def recursive_decompose(
        self,
        task_id: str,
        task_description: str,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> TaskGraph:
        """递归分解任务"""
        graph = self.decompose(task_id, task_description)
        
        if current_depth >= max_depth:
            return graph
        
        # 递归分解复杂子任务
        for subtask_id in list(graph.subtasks.keys()):
            subtask = graph.subtasks[subtask_id]
            
            # 判断是否需要进一步分解
            if subtask.estimated_effort > 2.0:
                child_graph = self.recursive_decompose(
                    subtask_id,
                    subtask.description,
                    max_depth,
                    current_depth + 1
                )
                
                # 合并子图
                self._merge_graphs(graph, child_graph, subtask_id)
        
        return graph
    
    def _merge_graphs(
        self,
        parent_graph: TaskGraph,
        child_graph: TaskGraph,
        parent_task_id: str
    ) -> None:
        """合并子图到父图"""
        # 添加子图的所有子任务
        for subtask_id, subtask in child_graph.subtasks.items():
            if subtask_id != parent_task_id:
                parent_graph.add_subtask(subtask)
        
        # 更新依赖关系
        # 原来依赖parent_task_id的任务现在依赖child_graph的最后一个任务
        child_order = child_graph.get_topological_order()
        if child_order:
            last_child = child_order[-1]
            
            for tid, deps in list(parent_graph.edges.items()):
                if parent_task_id in deps:
                    parent_graph.edges[tid].remove(parent_task_id)
                    parent_graph.edges[tid].add(last_child)
                    if tid in parent_graph.subtasks:
                        parent_graph.subtasks[tid].dependencies.discard(parent_task_id)
                        parent_graph.subtasks[tid].dependencies.add(last_child)


class AdaptiveTaskDecomposer(TaskDecomposer):
    """自适应任务分解器"""
    
    def __init__(self):
        super().__init__()
        self.decomposition_stats: Dict[str, Dict[str, Any]] = {}
    
    def decompose(
        self,
        task_id: str,
        task_description: str,
        strategy: Optional[DecompositionStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskGraph:
        """自适应分解任务"""
        context = context or {}
        
        # 根据上下文选择策略
        if strategy is None:
            strategy = self._select_strategy(task_description, context)
        
        graph = super().decompose(task_id, task_description, strategy)
        
        # 记录统计
        self.decomposition_stats[task_id] = {
            "description": task_description,
            "strategy": strategy,
            "num_subtasks": len(graph.subtasks),
            "context": context
        }
        
        return graph
    
    def _select_strategy(
        self,
        description: str,
        context: Dict[str, Any]
    ) -> DecompositionStrategy:
        """根据上下文选择分解策略"""
        # 检查关键词
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["data", "process", "pipeline"]):
            return DecompositionStrategy.PIPELINE
        
        if any(word in desc_lower for word in ["search", "research", "investigate"]):
            return DecompositionStrategy.FUNCTIONAL
        
        if any(word in desc_lower for word in ["design", "architect", "structure"]):
            return DecompositionStrategy.HIERARCHICAL
        
        # 检查时间约束
        if context.get("time_constraint") == "tight":
            return DecompositionStrategy.PARALLEL
        
        return DecompositionStrategy.SEQUENTIAL
