"""
约束求解器

处理Agent间的互斥或依赖约束，实现约束满足问题求解。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class ConstraintType(Enum):
    """约束类型"""
    MUTEX = auto()           # 互斥：两个Agent不能同时被选中
    DEPENDENCY = auto()      # 依赖：选中A必须先选中B
    COLOCATION = auto()      # 同位置：两个Agent必须在同一联盟
    EXCLUSION = auto()       # 排除：Agent不能被选中
    REQUIREMENT = auto()     # 要求：必须选中的Agent


@dataclass
class Constraint:
    """约束定义"""
    constraint_id: str
    constraint_type: ConstraintType
    agents: Set[str]  # 受约束的Agent
    weight: float = 1.0  # 约束权重（软约束时使用）
    is_hard: bool = True  # 是否为硬约束
    description: str = ""


@dataclass
class ConstraintViolation:
    """约束违反"""
    constraint: Constraint
    violated_by: Set[str]
    severity: float = 1.0


@dataclass
class Assignment:
    """分配方案"""
    task_id: str
    agent_assignments: Dict[str, str]  # task_id -> agent_id


class ConstraintSolver:
    """约束求解器"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.agents: Set[str] = set()
        self.tasks: Set[str] = set()
    
    def add_constraint(self, constraint: Constraint) -> None:
        """添加约束"""
        self.constraints.append(constraint)
        self.agents.update(constraint.agents)
    
    def validate_assignment(
        self,
        assignment: Assignment
    ) -> List[ConstraintViolation]:
        """
        验证分配方案是否满足约束
        
        Returns:
            违反的约束列表
        """
        violations = []
        selected_agents = set(assignment.agent_assignments.values())
        
        for constraint in self.constraints:
            violation = self._check_constraint(constraint, selected_agents)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_constraint(
        self,
        constraint: Constraint,
        selected_agents: Set[str]
    ) -> Optional[ConstraintViolation]:
        """检查单个约束"""
        if constraint.constraint_type == ConstraintType.MUTEX:
            # 互斥约束：不能同时选中
            violated = constraint.agents & selected_agents
            if len(violated) > 1:
                return ConstraintViolation(
                    constraint=constraint,
                    violated_by=violated,
                    severity=len(violated) / len(constraint.agents)
                )
        
        elif constraint.constraint_type == ConstraintType.DEPENDENCY:
            # 依赖约束：如果选中A，必须也选中B
            # 假设agents[0]依赖agents[1]
            if len(constraint.agents) >= 2:
                dependent = list(constraint.agents)[0]
                prerequisite = list(constraint.agents)[1]
                
                if dependent in selected_agents and prerequisite not in selected_agents:
                    return ConstraintViolation(
                        constraint=constraint,
                        violated_by={dependent},
                        severity=1.0
                    )
        
        elif constraint.constraint_type == ConstraintType.COLOCATION:
            # 同位置约束：必须同时选中或同时不选中
            selected_count = len(constraint.agents & selected_agents)
            if 0 < selected_count < len(constraint.agents):
                return ConstraintViolation(
                    constraint=constraint,
                    violated_by=constraint.agents & selected_agents,
                    severity=selected_count / len(constraint.agents)
                )
        
        elif constraint.constraint_type == ConstraintType.EXCLUSION:
            # 排除约束：不能选中
            violated = constraint.agents & selected_agents
            if violated:
                return ConstraintViolation(
                    constraint=constraint,
                    violated_by=violated,
                    severity=1.0
                )
        
        elif constraint.constraint_type == ConstraintType.REQUIREMENT:
            # 要求约束：必须选中
            missing = constraint.agents - selected_agents
            if missing:
                return ConstraintViolation(
                    constraint=constraint,
                    violated_by=missing,
                    severity=len(missing) / len(constraint.agents)
                )
        
        return None
    
    def find_valid_assignment(
        self,
        tasks: List[str],
        agent_candidates: Dict[str, List[str]]
    ) -> Optional[Assignment]:
        """
        寻找满足约束的分配方案
        
        使用回溯法求解
        """
        if not tasks:
            return Assignment(task_id="", agent_assignments={})
        
        def backtrack(
            task_idx: int,
            current_assignment: Dict[str, str]
        ) -> Optional[Dict[str, str]]:
            if task_idx == len(tasks):
                return current_assignment
            
            task = tasks[task_idx]
            candidates = agent_candidates.get(task, [])
            
            for agent in candidates:
                # 尝试分配
                current_assignment[task] = agent
                
                # 检查约束
                test_assignment = Assignment(
                    task_id="",
                    agent_assignments=current_assignment.copy()
                )
                violations = self.validate_assignment(test_assignment)
                
                # 只考虑硬约束
                hard_violations = [v for v in violations if v.constraint.is_hard]
                
                if not hard_violations:
                    result = backtrack(task_idx + 1, current_assignment)
                    if result is not None:
                        return result
                
                # 回溯
                del current_assignment[task]
            
            return None
        
        result = backtrack(0, {})
        
        if result:
            return Assignment(task_id="", agent_assignments=result)
        
        return None
    
    def optimize_assignment(
        self,
        tasks: List[str],
        agent_candidates: Dict[str, List[str]],
        agent_costs: Dict[str, float]
    ) -> Optional[Assignment]:
        """
        优化分配方案
        
        在满足约束的前提下最小化成本
        """
        # 首先找到有效分配
        valid_assignment = self.find_valid_assignment(tasks, agent_candidates)
        
        if not valid_assignment:
            return None
        
        # 简化优化：贪心选择最低成本的Agent
        optimized = {}
        
        for task in tasks:
            candidates = agent_candidates.get(task, [])
            # 过滤掉会导致约束违反的Agent
            valid_candidates = []
            
            for agent in candidates:
                test_assign = optimized.copy()
                test_assign[task] = agent
                
                test_assignment = Assignment(task_id="", agent_assignments=test_assign)
                violations = self.validate_assignment(test_assignment)
                hard_violations = [v for v in violations if v.constraint.is_hard]
                
                if not hard_violations:
                    valid_candidates.append(agent)
            
            if valid_candidates:
                # 选择成本最低的
                best_agent = min(valid_candidates, key=lambda a: agent_costs.get(a, float('inf')))
                optimized[task] = best_agent
        
        return Assignment(task_id="", agent_assignments=optimized)
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """获取约束摘要"""
        type_counts = {}
        for c in self.constraints:
            type_counts[c.constraint_type.name] = type_counts.get(c.constraint_type.name, 0) + 1
        
        return {
            "total_constraints": len(self.constraints),
            "hard_constraints": sum(1 for c in self.constraints if c.is_hard),
            "soft_constraints": sum(1 for c in self.constraints if not c.is_hard),
            "by_type": type_counts,
            "affected_agents": len(self.agents)
        }


class MutualExclusionSolver(ConstraintSolver):
    """互斥约束专用求解器"""
    
    def __init__(self):
        super().__init__()
        self.mutex_groups: List[Set[str]] = []
    
    def add_mutex_group(self, agents: Set[str]) -> None:
        """添加互斥组"""
        self.mutex_groups.append(agents)
        self.add_constraint(Constraint(
            constraint_id=f"mutex_{len(self.mutex_groups)}",
            constraint_type=ConstraintType.MUTEX,
            agents=agents,
            is_hard=True,
            description=f"Mutual exclusion for {agents}"
        ))
    
    def select_one_per_group(
        self,
        preferences: Dict[str, float]
    ) -> Set[str]:
        """
        从每个互斥组中选择一个Agent
        
        基于偏好分数选择
        """
        selected = set()
        
        for group in self.mutex_groups:
            # 在组内选择偏好最高的
            best_agent = None
            best_score = -float('inf')
            
            for agent in group:
                score = preferences.get(agent, 0)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                selected.add(best_agent)
        
        return selected
