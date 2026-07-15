"""
匈牙利算法优化分配

实现匈牙利算法（Kuhn-Munkres算法），用于解决二分图最小权匹配问题，
最小化总体执行成本。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_effort: float = 1.0
    priority: int = 1
    
    def __hash__(self) -> int:
        return hash(self.task_id)


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    cost_per_unit: float = 1.0
    reliability: float = 1.0
    
    def can_perform(self, task: Task) -> bool:
        """检查是否能执行任务"""
        return task.required_capabilities.issubset(self.capabilities)
    
    def __hash__(self) -> int:
        return hash(self.agent_id)


@dataclass
class AssignmentResult:
    """分配结果"""
    assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    total_cost: float = 0.0
    assignment_time_ms: float = 0.0
    optimal: bool = True
    unassigned_tasks: List[str] = field(default_factory=list)
    
    def get_agent_tasks(self, agent_id: str) -> List[str]:
        """获取分配给Agent的所有任务"""
        return [tid for tid, aid in self.assignments.items() if aid == agent_id]


class HungarianOptimizer:
    """匈牙利算法优化器"""
    
    def __init__(
        self,
        cost_matrix: Optional[List[List[float]]] = None,
        maximize: bool = False
    ):
        """
        初始化匈牙利算法优化器
        
        Args:
            cost_matrix: 成本矩阵，cost_matrix[i][j]表示第i个任务分配给第j个Agent的成本
            maximize: 是否最大化（默认为最小化）
        """
        self.cost_matrix = cost_matrix or []
        self.maximize = maximize
        self.n = len(self.cost_matrix) if self.cost_matrix else 0
        self.m = len(self.cost_matrix[0]) if self.cost_matrix and self.cost_matrix[0] else 0
    
    def set_cost_matrix(self, cost_matrix: List[List[float]]) -> None:
        """设置成本矩阵"""
        self.cost_matrix = cost_matrix
        self.n = len(cost_matrix)
        self.m = len(cost_matrix[0]) if cost_matrix else 0
    
    def solve(self) -> Tuple[List[Tuple[int, int]], float]:
        """
        求解分配问题
        
        Returns:
            (assignments, total_cost): 分配列表和总成本
            assignments是[(task_idx, agent_idx), ...]的列表
        """
        if not self.cost_matrix or self.n == 0 or self.m == 0:
            return [], 0.0
        
        # 确保是方阵（添加虚拟行/列）
        size = max(self.n, self.m)
        matrix = self._pad_matrix(self.cost_matrix, size)
        
        # 如果是最大化问题，转换为最小化
        if self.maximize:
            max_val = max(max(row) for row in matrix)
            matrix = [[max_val - val for val in row] for row in matrix]
        
        # 执行匈牙利算法
        assignment = self._hungarian_algorithm(matrix)
        
        # 计算总成本（使用原始成本矩阵）
        original_matrix = self._pad_matrix(self.cost_matrix, size)
        total_cost = sum(
            original_matrix[i][j] for i, j in assignment
        )
        
        # 过滤掉虚拟的行/列
        real_assignment = [
            (i, j) for i, j in assignment
            if i < self.n and j < self.m
        ]
        
        return real_assignment, total_cost
    
    def _pad_matrix(
        self,
        matrix: List[List[float]],
        size: int
    ) -> List[List[float]]:
        """将矩阵填充为方阵"""
        # 创建新矩阵
        padded = []
        
        for i in range(size):
            row = []
            for j in range(size):
                if i < len(matrix) and j < len(matrix[i]):
                    row.append(matrix[i][j])
                else:
                    # 填充大数（对于最小化问题）
                    row.append(float('inf') if not self.maximize else 0.0)
            padded.append(row)
        
        return padded
    
    def _hungarian_algorithm(self, matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        匈牙利算法核心实现
        
        基于Kuhn-Munkres算法，时间复杂度O(n^3)
        """
        n = len(matrix)
        
        # 初始化标签
        u = [0.0] * (n + 1)
        v = [0.0] * (n + 1)
        p = [0] * (n + 1)  # 匹配
        way = [0] * (n + 1)
        
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n + 1)
            used = [False] * (n + 1)
            
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                
                for j in range(n + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                
                j0 = j1
                if p[j0] == 0:
                    break
            
            # 增广路径
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        
        # 构建结果
        assignment = []
        for j in range(1, n + 1):
            if p[j] != 0:
                assignment.append((p[j] - 1, j - 1))
        
        return assignment


class TaskAgentAssignmentOptimizer:
    """任务-Agent分配优化器"""
    
    def __init__(
        self,
        cost_function: Optional[
            callable[[Agent, Task], float]
        ] = None
    ):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.cost_function = cost_function or self._default_cost_function
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent"""
        self.agents[agent.agent_id] = agent
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def _default_cost_function(self, agent: Agent, task: Task) -> float:
        """默认成本函数"""
        if not agent.can_perform(task):
            return float('inf')
        
        # 成本 = 单位成本 * 工作量 / 可靠性
        base_cost = agent.cost_per_unit * task.estimated_effort
        reliability_factor = 1.0 / max(0.1, agent.reliability)
        priority_factor = 1.0 / task.priority
        
        return base_cost * reliability_factor * priority_factor
    
    def optimize(self) -> AssignmentResult:
        """优化分配"""
        start_time = time.time()
        result = AssignmentResult()
        
        if not self.tasks or not self.agents:
            result.assignment_time_ms = (time.time() - start_time) * 1000
            return result
        
        task_list = list(self.tasks.values())
        agent_list = list(self.agents.values())
        
        # 构建成本矩阵
        cost_matrix: List[List[float]] = []
        
        for task in task_list:
            row: List[float] = []
            for agent in agent_list:
                cost = self.cost_function(agent, task)
                row.append(cost)
            cost_matrix.append(row)
        
        # 使用匈牙利算法求解
        optimizer = HungarianOptimizer(cost_matrix, maximize=False)
        assignment_indices, total_cost = optimizer.solve()
        
        # 构建结果
        for task_idx, agent_idx in assignment_indices:
            if task_idx < len(task_list) and agent_idx < len(agent_list):
                task = task_list[task_idx]
                agent = agent_list[agent_idx]
                
                # 检查是否是有效分配（成本不是无穷大）
                if cost_matrix[task_idx][agent_idx] < float('inf'):
                    result.assignments[task.task_id] = agent.agent_id
        
        # 找出未分配的任务
        assigned_tasks = set(result.assignments.keys())
        result.unassigned_tasks = [
            t.task_id for t in task_list
            if t.task_id not in assigned_tasks
        ]
        
        result.total_cost = sum(
            cost_matrix[task_list.index(self.tasks[tid])][
                agent_list.index(self.agents[aid])
            ]
            for tid, aid in result.assignments.items()
        )
        
        result.assignment_time_ms = (time.time() - start_time) * 1000
        result.optimal = len(result.unassigned_tasks) == 0
        
        return result


class MultiTaskHungarianOptimizer:
    """多任务匈牙利优化器（支持一个Agent执行多个任务）"""
    
    def __init__(self, max_tasks_per_agent: int = 3):
        self.max_tasks_per_agent = max_tasks_per_agent
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent"""
        self.agents[agent.agent_id] = agent
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def optimize(self) -> AssignmentResult:
        """优化分配（允许Agent执行多个任务）"""
        start_time = time.time()
        result = AssignmentResult()
        
        if not self.tasks or not self.agents:
            result.assignment_time_ms = (time.time() - start_time) * 1000
            return result
        
        task_list = list(self.tasks.values())
        agent_list = list(self.agents.values())
        
        n_tasks = len(task_list)
        n_agents = len(agent_list)
        
        # 扩展Agent列表（每个Agent可以执行多个任务）
        expanded_agents: List[Tuple[Agent, int]] = []  # (agent, copy_index)
        for agent in agent_list:
            for i in range(self.max_tasks_per_agent):
                expanded_agents.append((agent, i))
        
        n_expanded = len(expanded_agents)
        
        # 构建成本矩阵
        cost_matrix: List[List[float]] = []
        
        for task in task_list:
            row: List[float] = []
            for agent, copy_idx in expanded_agents:
                cost = self._calculate_cost(agent, task, copy_idx)
                row.append(cost)
            cost_matrix.append(row)
        
        # 使用匈牙利算法
        optimizer = HungarianOptimizer(cost_matrix, maximize=False)
        assignment_indices, total_cost = optimizer.solve()
        
        # 构建结果（合并扩展的Agent）
        for task_idx, agent_idx in assignment_indices:
            if task_idx < n_tasks and agent_idx < n_expanded:
                task = task_list[task_idx]
                agent, _ = expanded_agents[agent_idx]
                
                if cost_matrix[task_idx][agent_idx] < float('inf'):
                    result.assignments[task.task_id] = agent.agent_id
        
        # 找出未分配的任务
        assigned_tasks = set(result.assignments.keys())
        result.unassigned_tasks = [
            t.task_id for t in task_list
            if t.task_id not in assigned_tasks
        ]
        
        result.total_cost = total_cost
        result.assignment_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _calculate_cost(self, agent: Agent, task: Task, copy_idx: int) -> float:
        """计算成本（考虑任务复制索引）"""
        if not agent.can_perform(task):
            return float('inf')
        
        base_cost = agent.cost_per_unit * task.estimated_effort
        # 同一Agent的后续任务成本递增
        copy_penalty = 1.0 + 0.1 * copy_idx
        reliability_factor = 1.0 / max(0.1, agent.reliability)
        
        return base_cost * copy_penalty * reliability_factor


class BalancedAssignmentOptimizer:
    """平衡分配优化器（考虑负载均衡）"""
    
    def __init__(self, balance_weight: float = 0.3):
        self.balance_weight = balance_weight
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
    
    def add_agent(self, agent: Agent) -> None:
        """添加Agent"""
        self.agents[agent.agent_id] = agent
    
    def add_task(self, task: Task) -> None:
        """添加任务"""
        self.tasks[task.task_id] = task
    
    def optimize(self) -> AssignmentResult:
        """优化分配（考虑负载均衡）"""
        start_time = time.time()
        result = AssignmentResult()
        
        if not self.tasks or not self.agents:
            result.assignment_time_ms = (time.time() - start_time) * 1000
            return result
        
        task_list = list(self.tasks.values())
        agent_list = list(self.agents.values())
        
        # 跟踪每个Agent的当前负载
        agent_loads: Dict[str, float] = {a.agent_id: 0.0 for a in agent_list}
        
        # 迭代分配
        remaining_tasks = list(task_list)
        
        while remaining_tasks:
            # 构建当前成本矩阵
            cost_matrix: List[List[float]] = []
            
            for task in remaining_tasks:
                row: List[float] = []
                for agent in agent_list:
                    cost = self._calculate_balanced_cost(
                        agent, task, agent_loads[agent.agent_id]
                    )
                    row.append(cost)
                cost_matrix.append(row)
            
            # 使用匈牙利算法进行一次分配
            optimizer = HungarianOptimizer(cost_matrix, maximize=False)
            assignment_indices, _ = optimizer.solve()
            
            if not assignment_indices:
                break
            
            # 应用分配
            assigned_in_round: Set[int] = set()
            
            for task_idx, agent_idx in assignment_indices:
                if task_idx < len(remaining_tasks) and agent_idx < len(agent_list):
                    task = remaining_tasks[task_idx]
                    agent = agent_list[agent_idx]
                    
                    if cost_matrix[task_idx][agent_idx] < float('inf'):
                        result.assignments[task.task_id] = agent.agent_id
                        agent_loads[agent.agent_id] += task.estimated_effort
                        assigned_in_round.add(task_idx)
            
            # 移除已分配的任务
            remaining_tasks = [
                t for i, t in enumerate(remaining_tasks)
                if i not in assigned_in_round
            ]
            
            if not assigned_in_round:
                break
        
        # 找出未分配的任务
        assigned_tasks = set(result.assignments.keys())
        result.unassigned_tasks = [
            t.task_id for t in task_list
            if t.task_id not in assigned_tasks
        ]
        
        # 计算总成本
        result.total_cost = sum(
            self._calculate_balanced_cost(
                self.agents[aid], self.tasks[tid], 0
            )
            for tid, aid in result.assignments.items()
        )
        
        result.assignment_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _calculate_balanced_cost(
        self,
        agent: Agent,
        task: Task,
        current_load: float
    ) -> float:
        """计算平衡成本"""
        if not agent.can_perform(task):
            return float('inf')
        
        # 基础成本
        base_cost = agent.cost_per_unit * task.estimated_effort
        
        # 负载均衡惩罚
        load_ratio = current_load / max(1.0, agent.max_load) if hasattr(agent, 'max_load') else 0
        balance_penalty = 1.0 + self.balance_weight * load_ratio
        
        return base_cost * balance_penalty
