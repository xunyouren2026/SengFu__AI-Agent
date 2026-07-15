"""
联盟形成核心引擎

根据任务图构建最优Agent联盟，实现基于能力匹配和成本优化的联盟形成算法。
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar
from enum import Enum, auto


T = TypeVar('T')


class CoalitionStrategy(Enum):
    """联盟形成策略"""
    GREEDY = auto()           # 贪心策略
    OPTIMAL = auto()          # 最优策略（穷举）
    HEURISTIC = auto()        # 启发式策略
    DYNAMIC = auto()          # 动态规划策略


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_effort: float = 1.0
    priority: int = 1
    deadline: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    
    def __hash__(self) -> int:
        return hash(self.task_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return False
        return self.task_id == other.task_id


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    cost_per_unit: float = 1.0
    reliability: float = 1.0
    current_load: float = 0.0
    max_load: float = 10.0
    
    def can_perform(self, task: Task) -> bool:
        """检查Agent是否能执行任务"""
        return task.required_capabilities.issubset(self.capabilities)
    
    def available_capacity(self) -> float:
        """获取可用容量"""
        return max(0.0, self.max_load - self.current_load)
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False
        return self.agent_id == other.agent_id


@dataclass
class Coalition:
    """联盟表示"""
    coalition_id: str
    agents: Set[Agent] = field(default_factory=set)
    assigned_tasks: Set[Task] = field(default_factory=set)
    total_cost: float = 0.0
    total_value: float = 0.0
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent到联盟"""
        self.agents.add(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        """从联盟移除Agent"""
        self.agents.discard(agent)
    
    def has_capability(self, capability: str) -> bool:
        """检查联盟是否有指定能力"""
        return any(capability in agent.capabilities for agent in self.agents)
    
    def get_combined_capabilities(self) -> Set[str]:
        """获取联盟的所有能力"""
        capabilities: Set[str] = set()
        for agent in self.agents:
            capabilities.update(agent.capabilities)
        return capabilities
    
    def can_perform_task(self, task: Task) -> bool:
        """检查联盟是否能执行任务"""
        return task.required_capabilities.issubset(self.get_combined_capabilities())
    
    def calculate_cost(self) -> float:
        """计算联盟总成本"""
        return sum(agent.cost_per_unit for agent in self.agents)
    
    def __hash__(self) -> int:
        return hash(self.coalition_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coalition):
            return False
        return self.coalition_id == other.coalition_id


@dataclass
class TaskGraph:
    """任务图表示"""
    tasks: Dict[str, Task] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # task_id -> dependent task_ids
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
        if task.task_id not in self.edges:
            self.edges[task.task_id] = set()
    
    def add_dependency(self, from_task: str, to_task: str) -> None:
        """添加依赖关系: from_task 必须在 to_task 之前完成"""
        if from_task in self.tasks and to_task in self.tasks:
            if to_task not in self.edges:
                self.edges[to_task] = set()
            self.edges[to_task].add(from_task)
            self.tasks[to_task].dependencies.add(from_task)
    
    def get_topological_order(self) -> List[str]:
        """获取拓扑排序"""
        in_degree: Dict[str, int] = {tid: 0 for tid in self.tasks}
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
        """获取可并行执行的任务组"""
        order = self.get_topological_order()
        groups: List[Set[str]] = []
        completed: Set[str] = set()
        
        while len(completed) < len(order):
            group: Set[str] = set()
            for tid in order:
                if tid in completed:
                    continue
                task = self.tasks[tid]
                if task.dependencies.issubset(completed):
                    group.add(tid)
            
            if group:
                groups.append(group)
                completed.update(group)
            else:
                break
        
        return groups


@dataclass
class FormationResult:
    """联盟形成结果"""
    coalitions: List[Coalition] = field(default_factory=list)
    task_assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> coalition_id
    total_cost: float = 0.0
    total_value: float = 0.0
    formation_time_ms: float = 0.0
    strategy_used: CoalitionStrategy = CoalitionStrategy.GREEDY
    
    def get_coalition_for_task(self, task_id: str) -> Optional[Coalition]:
        """获取任务所属的联盟"""
        coalition_id = self.task_assignments.get(task_id)
        if coalition_id:
            for coalition in self.coalitions:
                if coalition.coalition_id == coalition_id:
                    return coalition
        return None


class CoalitionFormationEngine:
    """联盟形成引擎"""
    
    def __init__(
        self,
        strategy: CoalitionStrategy = CoalitionStrategy.GREEDY,
        value_function: Optional[Callable[[Coalition, Task], float]] = None
    ):
        self.strategy = strategy
        self.value_function = value_function or self._default_value_function
        self.agents: Dict[str, Agent] = {}
        self.task_graph: Optional[TaskGraph] = None
    
    def register_agent(self, agent: Agent) -> None:
        """注册Agent"""
        self.agents[agent.agent_id] = agent
    
    def set_task_graph(self, task_graph: TaskGraph) -> None:
        """设置任务图"""
        self.task_graph = task_graph
    
    def _default_value_function(self, coalition: Coalition, task: Task) -> float:
        """默认价值函数"""
        if not coalition.can_perform_task(task):
            return 0.0
        
        # 价值 = 任务优先级 * 可靠性 - 成本
        reliability = min(agent.reliability for agent in coalition.agents) if coalition.agents else 0.0
        cost = coalition.calculate_cost()
        value = task.priority * reliability * 10 - cost
        return max(0.0, value)
    
    def form_coalitions(self) -> FormationResult:
        """形成联盟"""
        start_time = time.time()
        
        if self.task_graph is None or not self.task_graph.tasks:
            return FormationResult(strategy_used=self.strategy)
        
        if self.strategy == CoalitionStrategy.GREEDY:
            result = self._greedy_formation()
        elif self.strategy == CoalitionStrategy.OPTIMAL:
            result = self._optimal_formation()
        elif self.strategy == CoalitionStrategy.HEURISTIC:
            result = self._heuristic_formation()
        elif self.strategy == CoalitionStrategy.DYNAMIC:
            result = self._dynamic_formation()
        else:
            result = self._greedy_formation()
        
        result.formation_time_ms = (time.time() - start_time) * 1000
        result.strategy_used = self.strategy
        return result
    
    def _greedy_formation(self) -> FormationResult:
        """贪心联盟形成"""
        result = FormationResult()
        coalitions: Dict[str, Coalition] = {}
        
        # 按优先级和拓扑顺序处理任务
        task_order = self.task_graph.get_topological_order()
        
        for task_id in task_order:
            task = self.task_graph.tasks[task_id]
            best_coalition: Optional[Coalition] = None
            best_value = -float('inf')
            
            # 尝试所有可能的Agent组合
            agents_list = list(self.agents.values())
            n = len(agents_list)
            
            # 限制搜索空间：最多考虑3个Agent的组合
            for i in range(n):
                # 单个Agent
                c1 = Coalition(f"coal_{task_id}_1")
                c1.add_agent(agents_list[i])
                v1 = self.value_function(c1, task)
                if v1 > best_value:
                    best_value = v1
                    best_coalition = c1
                
                # 两个Agent
                for j in range(i + 1, n):
                    c2 = Coalition(f"coal_{task_id}_2")
                    c2.add_agent(agents_list[i])
                    c2.add_agent(agents_list[j])
                    v2 = self.value_function(c2, task)
                    if v2 > best_value:
                        best_value = v2
                        best_coalition = c2
                    
                    # 三个Agent
                    for k in range(j + 1, n):
                        c3 = Coalition(f"coal_{task_id}_3")
                        c3.add_agent(agents_list[i])
                        c3.add_agent(agents_list[j])
                        c3.add_agent(agents_list[k])
                        v3 = self.value_function(c3, task)
                        if v3 > best_value:
                            best_value = v3
                            best_coalition = c3
            
            if best_coalition and best_value > 0:
                coalitions[task_id] = best_coalition
                result.task_assignments[task_id] = best_coalition.coalition_id
        
        result.coalitions = list(coalitions.values())
        result.total_cost = sum(c.calculate_cost() for c in result.coalitions)
        result.total_value = sum(
            self.value_function(c, self.task_graph.tasks[tid])
            for tid, c in coalitions.items()
        )
        
        return result
    
    def _optimal_formation(self) -> FormationResult:
        """最优联盟形成（穷举）"""
        result = FormationResult()
        
        if not self.task_graph or not self.task_graph.tasks:
            return result
        
        tasks = list(self.task_graph.tasks.values())
        agents_list = list(self.agents.values())
        
        best_assignment: Dict[str, Set[int]] = {}
        best_total_value = -float('inf')
        
        # 对每个任务，尝试所有可能的Agent子集
        def enumerate_assignments(task_idx: int, current_assignment: Dict[str, Set[int]]):
            nonlocal best_assignment, best_total_value
            
            if task_idx == len(tasks):
                # 计算总价值
                total_value = 0.0
                for tid, agent_indices in current_assignment.items():
                    coalition = Coalition(f"coal_{tid}")
                    for idx in agent_indices:
                        coalition.add_agent(agents_list[idx])
                    task = self.task_graph.tasks[tid]
                    total_value += self.value_function(coalition, task)
                
                if total_value > best_total_value:
                    best_total_value = total_value
                    best_assignment = {k: set(v) for k, v in current_assignment.items()}
                return
            
            task = tasks[task_idx]
            n = len(agents_list)
            
            # 枚举所有可能的Agent子集
            for mask in range(1, 1 << n):
                coalition = Coalition(f"temp")
                agent_indices: Set[int] = set()
                for i in range(n):
                    if mask & (1 << i):
                        coalition.add_agent(agents_list[i])
                        agent_indices.add(i)
                
                if coalition.can_perform_task(task):
                    current_assignment[task.task_id] = agent_indices
                    enumerate_assignments(task_idx + 1, current_assignment)
                    del current_assignment[task.task_id]
        
        enumerate_assignments(0, {})
        
        # 构建结果
        coalitions: Dict[str, Coalition] = {}
        for tid, agent_indices in best_assignment.items():
            coalition = Coalition(f"coal_{tid}")
            for idx in agent_indices:
                coalition.add_agent(agents_list[idx])
            coalitions[tid] = coalition
            result.task_assignments[tid] = coalition.coalition_id
        
        result.coalitions = list(coalitions.values())
        result.total_value = best_total_value
        result.total_cost = sum(c.calculate_cost() for c in result.coalitions)
        
        return result
    
    def _heuristic_formation(self) -> FormationResult:
        """启发式联盟形成"""
        result = FormationResult()
        
        # 基于能力匹配度进行分组
        capability_groups: Dict[frozenset, List[Agent]] = {}
        for agent in self.agents.values():
            cap_key = frozenset(agent.capabilities)
            if cap_key not in capability_groups:
                capability_groups[cap_key] = []
            capability_groups[cap_key].append(agent)
        
        # 为每个任务找到最匹配的Agent组
        task_order = self.task_graph.get_topological_order()
        coalitions: Dict[str, Coalition] = {}
        
        for task_id in task_order:
            task = self.task_graph.tasks[task_id]
            req_caps = frozenset(task.required_capabilities)
            
            best_group: Optional[List[Agent]] = None
            best_match = 0
            
            for cap_key, group in capability_groups.items():
                if req_caps.issubset(cap_key):
                    match_score = len(req_caps) / len(cap_key) if cap_key else 0
                    if match_score > best_match:
                        best_match = match_score
                        best_group = group
            
            if best_group:
                coalition = Coalition(f"coal_{task_id}")
                # 选择成本最低的Agent
                sorted_agents = sorted(best_group, key=lambda a: a.cost_per_unit)
                coalition.add_agent(sorted_agents[0])
                coalitions[task_id] = coalition
                result.task_assignments[task_id] = coalition.coalition_id
        
        result.coalitions = list(coalitions.values())
        result.total_cost = sum(c.calculate_cost() for c in result.coalitions)
        result.total_value = sum(
            self.value_function(c, self.task_graph.tasks[tid])
            for tid, c in coalitions.items()
        )
        
        return result
    
    def _dynamic_formation(self) -> FormationResult:
        """动态规划联盟形成"""
        result = FormationResult()
        
        # 简化的DP实现：按任务顺序逐个分配
        task_order = self.task_graph.get_topological_order()
        agents_list = list(self.agents.values())
        
        coalitions: Dict[str, Coalition] = {}
        assigned_agents: Set[str] = set()
        
        for task_id in task_order:
            task = self.task_graph.tasks[task_id]
            best_coalition: Optional[Coalition] = None
            best_value = -float('inf')
            
            # 考虑未分配的Agent
            available_agents = [a for a in agents_list if a.agent_id not in assigned_agents]
            
            for agent in available_agents:
                if agent.can_perform(task):
                    coalition = Coalition(f"coal_{task_id}")
                    coalition.add_agent(agent)
                    value = self.value_function(coalition, task)
                    if value > best_value:
                        best_value = value
                        best_coalition = coalition
            
            if best_coalition:
                coalitions[task_id] = best_coalition
                result.task_assignments[task_id] = best_coalition.coalition_id
                for agent in best_coalition.agents:
                    assigned_agents.add(agent.agent_id)
        
        result.coalitions = list(coalitions.values())
        result.total_cost = sum(c.calculate_cost() for c in result.coalitions)
        result.total_value = sum(
            self.value_function(c, self.task_graph.tasks[tid])
            for tid, c in coalitions.items()
        )
        
        return result
