"""
信誉加权分配

信誉高的Agent更易中标，实现基于信誉的任务分配。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    base_reputation: float = 1.0
    
    def __hash__(self) -> int:
        return hash(self.agent_id)


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    priority: int = 1


@dataclass
class ReputationRecord:
    """信誉记录"""
    agent_id: str
    score: float = 1.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    def update(self, success: bool, task_id: str) -> None:
        """更新信誉"""
        self.total_tasks += 1
        
        if success:
            self.successful_tasks += 1
            # 成功增加信誉
            self.score = min(5.0, self.score + 0.1)
        else:
            self.failed_tasks += 1
            # 失败降低信誉
            self.score = max(0.1, self.score - 0.2)
        
        self.history.append({
            "task_id": task_id,
            "success": success,
            "timestamp": time.time()
        })
        
        self.last_updated = time.time()
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_tasks == 0:
            return 1.0
        return self.successful_tasks / self.total_tasks


@dataclass
class AssignmentResult:
    """分配结果"""
    task_id: str
    agent_id: str
    reputation_score: float
    weighted_score: float


class ReputationManager:
    """信誉管理器"""
    
    def __init__(
        self,
        initial_reputation: float = 1.0,
        decay_factor: float = 0.99
    ):
        self.initial_reputation = initial_reputation
        self.decay_factor = decay_factor
        self.reputations: Dict[str, ReputationRecord] = {}
    
    def get_reputation(self, agent_id: str) -> ReputationRecord:
        """获取Agent信誉"""
        if agent_id not in self.reputations:
            self.reputations[agent_id] = ReputationRecord(
                agent_id=agent_id,
                score=self.initial_reputation
            )
        return self.reputations[agent_id]
    
    def update_reputation(self, agent_id: str, success: bool, task_id: str) -> None:
        """更新Agent信誉"""
        record = self.get_reputation(agent_id)
        record.update(success, task_id)
    
    def apply_time_decay(self) -> None:
        """应用时间衰减"""
        current_time = time.time()
        
        for record in self.reputations.values():
            time_diff = current_time - record.last_updated
            # 每天衰减
            days = time_diff / 86400
            record.score *= (self.decay_factor ** days)
            record.score = max(0.1, record.score)
    
    def get_ranking(self) -> List[Tuple[str, float]]:
        """获取信誉排名"""
        rankings = [
            (agent_id, record.score)
            for agent_id, record in self.reputations.items()
        ]
        rankings.sort(key=lambda x: -x[1])
        return rankings


class ReputationWeightedAssigner:
    """信誉加权分配器"""
    
    def __init__(
        self,
        reputation_weight: float = 0.5,
        capability_weight: float = 0.5
    ):
        self.reputation_weight = reputation_weight
        self.capability_weight = capability_weight
        self.reputation_manager = ReputationManager()
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
    
    def register_agent(self, agent: Agent) -> None:
        """注册Agent"""
        self.agents[agent.agent_id] = agent
    
    def register_task(self, task: Task) -> None:
        """注册任务"""
        self.tasks[task.task_id] = task
    
    def assign_task(self, task_id: str) -> Optional[AssignmentResult]:
        """
        分配任务
        
        综合考虑信誉和能力进行分配
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        best_agent: Optional[str] = None
        best_score = -float('inf')
        
        for agent_id, agent in self.agents.items():
            # 检查能力匹配
            if not task.required_capabilities.issubset(agent.capabilities):
                continue
            
            # 计算能力得分
            capability_score = 1.0  # 能执行就是1.0
            
            # 获取信誉得分
            reputation_record = self.reputation_manager.get_reputation(agent_id)
            reputation_score = reputation_record.score
            
            # 加权综合得分
            weighted_score = (
                self.reputation_weight * reputation_score +
                self.capability_weight * capability_score
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_agent = agent_id
        
        if best_agent:
            return AssignmentResult(
                task_id=task_id,
                agent_id=best_agent,
                reputation_score=self.reputation_manager.get_reputation(best_agent).score,
                weighted_score=best_score
            )
        
        return None
    
    def assign_all_tasks(self) -> List[AssignmentResult]:
        """分配所有任务"""
        results = []
        
        for task_id in self.tasks:
            result = self.assign_task(task_id)
            if result:
                results.append(result)
        
        return results
    
    def report_task_completion(
        self,
        agent_id: str,
        task_id: str,
        success: bool
    ) -> None:
        """报告任务完成情况"""
        self.reputation_manager.update_reputation(agent_id, success, task_id)
    
    def get_reputation_report(self) -> Dict[str, Any]:
        """获取信誉报告"""
        rankings = self.reputation_manager.get_ranking()
        
        return {
            "total_agents": len(self.reputation_manager.reputations),
            "rankings": [
                {
                    "agent_id": agent_id,
                    "score": score,
                    "record": {
                        "total_tasks": self.reputation_manager.reputations[agent_id].total_tasks,
                        "success_rate": self.reputation_manager.reputations[agent_id].get_success_rate()
                    }
                }
                for agent_id, score in rankings
            ]
        }
