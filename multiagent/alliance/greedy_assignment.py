"""
贪心任务分配

每次选择最适合当前子任务的Agent，基于局部最优实现快速分配。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    priority: int = 1
    estimated_effort: float = 1.0
    deadline: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    
    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    cost_per_unit: float = 1.0
    reliability: float = 1.0
    current_load: float = 0.0
    max_load: float = 10.0
    performance_history: List[float] = field(default_factory=list)
    
    def can_perform(self, task: Task) -> bool:
        """检查是否能执行任务"""
        return task.required_capabilities.issubset(self.capabilities)
    
    def available_capacity(self) -> float:
        """获取可用容量"""
        return max(0.0, self.max_load - self.current_load)
    
    def average_performance(self) -> float:
        """获取平均性能"""
        if not self.performance_history:
            return 1.0
        return sum(self.performance_history) / len(self.performance_history)
    
    def __hash__(self) -> int:
        return hash(self.agent_id)


class GreedyStrategy(Enum):
    """贪心策略类型"""
    BEST_FIT = auto()       # 最佳适配
    FIRST_FIT = auto()      # 首次适配
    WORST_FIT = auto()      # 最差适配
    CAPABILITY_MATCH = auto()  # 能力匹配度优先
    COST_MINIMIZATION = auto()  # 成本最小化
    RELIABILITY_MAXIMIZATION = auto()  # 可靠性最大化


@dataclass
class Assignment:
    """分配结果"""
    task_id: str
    agent_id: str
    score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssignmentResult:
    """分配结果集合"""
    assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    unassigned_tasks: List[str] = field(default_factory=list)
    assignment_scores: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_score: float = 0.0
    assignment_time_ms: float = 0.0
    strategy_used: GreedyStrategy = GreedyStrategy.BEST_FIT
    
    def get_agent_tasks(self, agent_id: str) -> List[str]:
        """获取分配给Agent的所有任务"""
        return [tid for tid, aid in self.assignments.items() if aid == agent_id]


class GreedyAssignmentSolver:
    """贪心任务分配求解器"""
    
    def __init__(
        self,
        strategy: GreedyStrategy = GreedyStrategy.BEST_FIT,
        scoring_function: Optional[Callable[[Agent, Task], float]] = None
    ):
        self.strategy = strategy
        self.scoring_function = scoring_function or self._default_scoring_function
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent"""
        self.agents[agent.agent_id] = agent
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def _default_scoring_function(self, agent: Agent, task: Task) -> float:
        """默认评分函数"""
        if not agent.can_perform(task):
            return -float('inf')
        
        if agent.available_capacity() < task.estimated_effort:
            return -float('inf')
        
        # 综合评分：能力匹配度、可靠性、成本
        capability_match = len(
            task.required_capabilities & agent.capabilities
        ) / len(task.required_capabilities) if task.required_capabilities else 1.0
        
        reliability_score = agent.reliability
        cost_score = 1.0 / (1.0 + agent.cost_per_unit * task.estimated_effort)
        capacity_score = agent.available_capacity() / agent.max_load
        performance_score = agent.average_performance()
        
        # 加权综合
        score = (
            0.3 * capability_match +
            0.2 * reliability_score +
            0.2 * cost_score +
            0.1 * capacity_score +
            0.2 * performance_score
        )
        
        return score
    
    def solve(self) -> AssignmentResult:
        """求解分配问题"""
        start_time = time.time()
        result = AssignmentResult(strategy_used=self.strategy)
        
        # 按优先级和依赖关系排序任务
        sorted_tasks = self._sort_tasks()
        
        # 复制Agent状态（避免修改原始状态）
        agent_states = {
            aid: Agent(
                agent_id=agent.agent_id,
                capabilities=set(agent.capabilities),
                cost_per_unit=agent.cost_per_unit,
                reliability=agent.reliability,
                current_load=agent.current_load,
                max_load=agent.max_load,
                performance_history=list(agent.performance_history)
            )
            for aid, agent in self.agents.items()
        }
        
        for task in sorted_tasks:
            best_agent: Optional[str] = None
            best_score = -float('inf')
            
            if self.strategy == GreedyStrategy.BEST_FIT:
                best_agent, best_score = self._best_fit(task, agent_states)
            elif self.strategy == GreedyStrategy.FIRST_FIT:
                best_agent, best_score = self._first_fit(task, agent_states)
            elif self.strategy == GreedyStrategy.WORST_FIT:
                best_agent, best_score = self._worst_fit(task, agent_states)
            elif self.strategy == GreedyStrategy.CAPABILITY_MATCH:
                best_agent, best_score = self._capability_match(task, agent_states)
            elif self.strategy == GreedyStrategy.COST_MINIMIZATION:
                best_agent, best_score = self._cost_minimization(task, agent_states)
            elif self.strategy == GreedyStrategy.RELIABILITY_MAXIMIZATION:
                best_agent, best_score = self._reliability_maximization(task, agent_states)
            
            if best_agent and best_score > -float('inf'):
                result.assignments[task.task_id] = best_agent
                result.assignment_scores[task.task_id] = best_score
                result.total_score += best_score
                
                # 更新Agent负载
                agent_states[best_agent].current_load += task.estimated_effort
                result.total_cost += (
                    agent_states[best_agent].cost_per_unit * task.estimated_effort
                )
            else:
                result.unassigned_tasks.append(task.task_id)
        
        result.assignment_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _sort_tasks(self) -> List[Task]:
        """排序任务"""
        # 拓扑排序 + 优先级
        tasks_list = list(self.tasks.values())
        
        # 计算任务深度（依赖链长度）
        depth: Dict[str, int] = {}
        
        def get_depth(task_id: str, visited: Set[str]) -> int:
            if task_id in depth:
                return depth[task_id]
            if task_id in visited:
                return 0
            
            visited.add(task_id)
            task = self.tasks.get(task_id)
            if not task or not task.dependencies:
                depth[task_id] = 0
                return 0
            
            max_dep_depth = max(
                get_depth(dep, visited) for dep in task.dependencies
            )
            depth[task_id] = max_dep_depth + 1
            return depth[task_id]
        
        for task in tasks_list:
            get_depth(task.task_id, set())
        
        # 按深度和优先级排序
        return sorted(
            tasks_list,
            key=lambda t: (-t.priority, depth.get(t.task_id, 0))
        )
    
    def _best_fit(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """最佳适配：选择评分最高的Agent"""
        best_agent: Optional[str] = None
        best_score = -float('inf')
        
        for agent_id, agent in agent_states.items():
            score = self.scoring_function(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent, best_score
    
    def _first_fit(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """首次适配：选择第一个能执行任务的Agent"""
        for agent_id, agent in agent_states.items():
            score = self.scoring_function(agent, task)
            if score > -float('inf'):
                return agent_id, score
        return None, -float('inf')
    
    def _worst_fit(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """最差适配：选择剩余容量最大的Agent"""
        best_agent: Optional[str] = None
        max_capacity = -float('inf')
        score = 0.0
        
        for agent_id, agent in agent_states.items():
            s = self.scoring_function(agent, task)
            if s > -float('inf'):
                capacity = agent.available_capacity()
                if capacity > max_capacity:
                    max_capacity = capacity
                    best_agent = agent_id
                    score = s
        
        return best_agent, score
    
    def _capability_match(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """能力匹配度优先"""
        best_agent: Optional[str] = None
        best_match = -float('inf')
        
        for agent_id, agent in agent_states.items():
            if not agent.can_perform(task):
                continue
            if agent.available_capacity() < task.estimated_effort:
                continue
            
            # 计算能力匹配度
            match = len(task.required_capabilities & agent.capabilities)
            if match > best_match:
                best_match = match
                best_agent = agent_id
        
        score = self.scoring_function(agent_states[best_agent], task) if best_agent else -float('inf')
        return best_agent, score
    
    def _cost_minimization(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """成本最小化"""
        best_agent: Optional[str] = None
        min_cost = float('inf')
        
        for agent_id, agent in agent_states.items():
            if not agent.can_perform(task):
                continue
            if agent.available_capacity() < task.estimated_effort:
                continue
            
            cost = agent.cost_per_unit * task.estimated_effort
            if cost < min_cost:
                min_cost = cost
                best_agent = agent_id
        
        score = self.scoring_function(agent_states[best_agent], task) if best_agent else -float('inf')
        return best_agent, score
    
    def _reliability_maximization(self, task: Task, agent_states: Dict[str, Agent]) -> Tuple[Optional[str], float]:
        """可靠性最大化"""
        best_agent: Optional[str] = None
        max_reliability = -float('inf')
        
        for agent_id, agent in agent_states.items():
            if not agent.can_perform(task):
                continue
            if agent.available_capacity() < task.estimated_effort:
                continue
            
            if agent.reliability > max_reliability:
                max_reliability = agent.reliability
                best_agent = agent_id
        
        score = self.scoring_function(agent_states[best_agent], task) if best_agent else -float('inf')
        return best_agent, score


class MultiRoundGreedySolver:
    """多轮贪心求解器"""
    
    def __init__(self, rounds: int = 3):
        self.rounds = rounds
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.assignment_history: List[AssignmentResult] = []
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent"""
        self.agents[agent.agent_id] = agent
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def solve(self) -> AssignmentResult:
        """多轮求解，尝试改进结果"""
        best_result: Optional[AssignmentResult] = None
        
        strategies = [
            GreedyStrategy.BEST_FIT,
            GreedyStrategy.CAPABILITY_MATCH,
            GreedyStrategy.COST_MINIMIZATION,
            GreedyStrategy.RELIABILITY_MAXIMIZATION
        ]
        
        for strategy in strategies[:self.rounds]:
            solver = GreedyAssignmentSolver(strategy)
            
            for agent in self.agents.values():
                solver.add_agent(agent)
            for task in self.tasks.values():
                solver.add_task(task)
            
            result = solver.solve()
            self.assignment_history.append(result)
            
            if best_result is None or result.total_score > best_result.total_score:
                best_result = result
        
        return best_result or AssignmentResult()
