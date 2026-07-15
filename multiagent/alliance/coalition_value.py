"""
联盟价值计算

实现联盟价值计算和Shapley值计算，用于评估不同Agent组合的总收益和公平分配。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, FrozenSet
from functools import lru_cache


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    cost: float = 0.0
    
    def __hash__(self) -> int:
        return hash(self.agent_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Agent):
            return False
        return self.agent_id == other.agent_id


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    value: float = 1.0
    
    def __hash__(self) -> int:
        return hash(self.task_id)


class CoalitionValueCalculator:
    """联盟价值计算器"""
    
    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        value_function: Optional[Callable[[FrozenSet[str]], float]] = None
    ):
        self.agents = {a.agent_id: a for a in agents}
        self.tasks = tasks
        self.value_function = value_function or self._default_value_function
        self._value_cache: Dict[FrozenSet[str], float] = {}
    
    def _default_value_function(self, coalition: FrozenSet[str]) -> float:
        """默认价值函数：计算联盟能完成的任务总价值"""
        if not coalition:
            return 0.0
        
        # 获取联盟的所有能力
        coalition_capabilities: Set[str] = set()
        coalition_cost = 0.0
        
        for agent_id in coalition:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                coalition_capabilities.update(agent.capabilities)
                coalition_cost += agent.cost
        
        # 计算能完成的任务价值
        total_value = 0.0
        for task in self.tasks:
            if task.required_capabilities.issubset(coalition_capabilities):
                total_value += task.value
        
        # 净价值 = 任务价值 - 成本
        return max(0.0, total_value - coalition_cost)
    
    def calculate_value(self, coalition: Set[str]) -> float:
        """计算联盟价值"""
        coalition_key = frozenset(coalition)
        
        if coalition_key not in self._value_cache:
            self._value_cache[coalition_key] = self.value_function(coalition_key)
        
        return self._value_cache[coalition_key]
    
    def calculate_marginal_contribution(
        self,
        agent_id: str,
        coalition: Set[str]
    ) -> float:
        """
        计算Agent对联盟的边际贡献
        
        v(S ∪ {i}) - v(S)
        """
        if agent_id in coalition:
            return 0.0
        
        value_without = self.calculate_value(coalition)
        
        coalition_with = coalition | {agent_id}
        value_with = self.calculate_value(coalition_with)
        
        return value_with - value_without
    
    def calculate_shapley_value(self, agent_id: str) -> float:
        """
        计算Shapley值
        
        Shapley值表示Agent对所有可能联盟的边际贡献的加权平均。
        公式: φ(i) = Σ [|S|!(n-|S|-1)! / n!] * [v(S∪{i}) - v(S)]
        """
        if agent_id not in self.agents:
            return 0.0
        
        n = len(self.agents)
        other_agents = set(self.agents.keys()) - {agent_id}
        
        shapley_value = 0.0
        
        # 遍历所有可能的联盟（不包含当前Agent）
        for r in range(len(other_agents) + 1):
            for coalition_subset in itertools.combinations(other_agents, r):
                coalition = set(coalition_subset)
                
                # 计算权重: |S|!(n-|S|-1)! / n!
                s = len(coalition)
                weight = self._factorial(s) * self._factorial(n - s - 1) / self._factorial(n)
                
                # 计算边际贡献
                marginal = self.calculate_marginal_contribution(agent_id, coalition)
                
                shapley_value += weight * marginal
        
        return shapley_value
    
    def calculate_all_shapley_values(self) -> Dict[str, float]:
        """计算所有Agent的Shapley值"""
        return {
            agent_id: self.calculate_shapley_value(agent_id)
            for agent_id in self.agents
        }
    
    def calculate_banzhaf_index(self, agent_id: str) -> float:
        """
        计算Banzhaf权力指数
        
        与Shapley值类似，但使用不同的权重。
        """
        if agent_id not in self.agents:
            return 0.0
        
        other_agents = set(self.agents.keys()) - {agent_id}
        total_contribution = 0.0
        count = 0
        
        for r in range(len(other_agents) + 1):
            for coalition_subset in itertools.combinations(other_agents, r):
                coalition = set(coalition_subset)
                marginal = self.calculate_marginal_contribution(agent_id, coalition)
                total_contribution += marginal
                count += 1
        
        return total_contribution / count if count > 0 else 0.0
    
    def calculate_core(self) -> Optional[Dict[str, float]]:
        """
        计算核心（Core）
        
        核心是满足以下条件的分配方案：
        1. 效率性: Σ x_i = v(N)
        2. 个体理性: x_i >= v({i})
        3. 联盟理性: Σ_{i∈S} x_i >= v(S) 对所有 S ⊆ N
        
        返回一个满足核心条件的分配方案，如果不存在则返回None。
        """
        n = len(self.agents)
        agent_ids = list(self.agents.keys())
        
        # 计算大联盟的价值
        grand_coalition = set(agent_ids)
        grand_value = self.calculate_value(grand_coalition)
        
        # 简化的核心计算：使用Shapley值作为近似
        # 注意：Shapley值不一定在核心中，但在凸博弈中一定在
        shapley_values = self.calculate_all_shapley_values()
        
        # 验证是否满足核心条件
        total_shapley = sum(shapley_values.values())
        
        # 检查效率性
        if abs(total_shapley - grand_value) > 1e-6:
            # 归一化
            factor = grand_value / total_shapley if total_shapley > 0 else 1.0
            shapley_values = {k: v * factor for k, v in shapley_values.items()}
        
        # 检查联盟理性（简化检查）
        for r in range(1, n):
            for coalition_subset in itertools.combinations(agent_ids, r):
                coalition = set(coalition_subset)
                coalition_value = self.calculate_value(coalition)
                allocation_sum = sum(shapley_values[a] for a in coalition)
                
                if allocation_sum < coalition_value - 1e-6:
                    # 不满足核心条件
                    return None
        
        return shapley_values
    
    def is_convex_game(self) -> bool:
        """
        检查是否为凸博弈
        
        凸博弈：v(S ∪ T) + v(S ∩ T) >= v(S) + v(T) 对所有 S, T ⊆ N
        """
        agent_ids = list(self.agents.keys())
        n = len(agent_ids)
        
        # 检查所有子集对
        for i in range(1 << n):
            for j in range(1 << n):
                S = {agent_ids[k] for k in range(n) if i & (1 << k)}
                T = {agent_ids[k] for k in range(n) if j & (1 << k)}
                
                union = S | T
                intersection = S & T
                
                left = self.calculate_value(union) + self.calculate_value(intersection)
                right = self.calculate_value(S) + self.calculate_value(T)
                
                if left + 1e-9 < right:
                    return False
        
        return True
    
    def calculate_nucleolus(self, max_iterations: int = 1000) -> Dict[str, float]:
        """
        计算核仁（Nucleolus）
        
        核仁是使最大不满最小的分配方案。
        使用迭代方法近似计算。
        """
        agent_ids = list(self.agents.keys())
        n = len(agent_ids)
        
        if n == 0:
            return {}
        
        grand_coalition = set(agent_ids)
        grand_value = self.calculate_value(grand_coalition)
        
        # 初始分配：平均分配
        allocation = {agent_id: grand_value / n for agent_id in agent_ids}
        
        for _ in range(max_iterations):
            # 计算所有联盟的超出量（excess）
            excesses: List[Tuple[Set[str], float]] = []
            
            for r in range(n + 1):
                for coalition_subset in itertools.combinations(agent_ids, r):
                    coalition = set(coalition_subset)
                    coalition_value = self.calculate_value(coalition)
                    allocated = sum(allocation[a] for a in coalition)
                    excess = coalition_value - allocated
                    excesses.append((coalition, excess))
            
            # 找到最大超出量
            max_excess = max(excesses, key=lambda x: x[1])
            
            if max_excess[1] <= 1e-6:
                break
            
            # 调整分配（简化策略）
            # 增加对最大超出量联盟中Agent的分配
            for agent_id in max_excess[0]:
                allocation[agent_id] += max_excess[1] / len(max_excess[0])
            
            # 归一化
            total = sum(allocation.values())
            if total > 0:
                factor = grand_value / total
                allocation = {k: v * factor for k, v in allocation.items()}
        
        return allocation
    
    def _factorial(self, n: int) -> float:
        """计算阶乘"""
        if n <= 1:
            return 1.0
        result = 1.0
        for i in range(2, n + 1):
            result *= i
        return result
    
    def get_characteristic_function(self) -> Dict[FrozenSet[str], float]:
        """获取特征函数（所有联盟的价值）"""
        agent_ids = list(self.agents.keys())
        n = len(agent_ids)
        
        characteristic: Dict[FrozenSet[str], float] = {}
        
        for i in range(1 << n):
            coalition = frozenset(agent_ids[k] for k in range(n) if i & (1 << k))
            characteristic[coalition] = self.calculate_value(set(coalition))
        
        return characteristic


class CoalitionStabilityAnalyzer:
    """联盟稳定性分析器"""
    
    def __init__(self, calculator: CoalitionValueCalculator):
        self.calculator = calculator
    
    def check_individual_rationality(self, allocation: Dict[str, float]) -> Dict[str, bool]:
        """检查个体理性"""
        results = {}
        for agent_id, payoff in allocation.items():
            singleton_value = self.calculator.calculate_value({agent_id})
            results[agent_id] = payoff >= singleton_value - 1e-6
        return results
    
    def check_coalitional_rationality(self, allocation: Dict[str, float]) -> bool:
        """检查联盟理性"""
        agent_ids = list(self.calculator.agents.keys())
        n = len(agent_ids)
        
        for r in range(1, n + 1):
            for coalition_subset in itertools.combinations(agent_ids, r):
                coalition = set(coalition_subset)
                coalition_value = self.calculator.calculate_value(coalition)
                allocated = sum(allocation.get(a, 0) for a in coalition)
                
                if allocated < coalition_value - 1e-6:
                    return False
        
        return True
    
    def find_blocking_coalition(self, allocation: Dict[str, float]) -> Optional[Set[str]]:
        """寻找阻止联盟"""
        agent_ids = list(self.calculator.agents.keys())
        n = len(agent_ids)
        
        for r in range(1, n + 1):
            for coalition_subset in itertools.combinations(agent_ids, r):
                coalition = set(coalition_subset)
                coalition_value = self.calculator.calculate_value(coalition)
                allocated = sum(allocation.get(a, 0) for a in coalition)
                
                if coalition_value > allocated + 1e-6:
                    return coalition
        
        return None
    
    def analyze_stability(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """全面分析稳定性"""
        individual_rational = self.check_individual_rationality(allocation)
        coalitional_rational = self.check_coalitional_rationality(allocation)
        blocking_coalition = self.find_blocking_coalition(allocation)
        
        return {
            "individual_rationality": individual_rational,
            "all_individually_rational": all(individual_rational.values()),
            "coalitional_rationality": coalitional_rational,
            "in_core": coalitional_rational and all(individual_rational.values()),
            "blocking_coalition": blocking_coalition,
            "is_stable": blocking_coalition is None
        }
