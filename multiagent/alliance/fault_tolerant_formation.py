"""
容错形成

为关键任务预分配备用Agent，实现故障转移和高可用性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto


class FaultToleranceLevel(Enum):
    """容错级别"""
    NONE = auto()           # 无容错
    PRIMARY_BACKUP = auto()  # 主备模式
    ACTIVE_REPLICATION = auto()  # 主动复制
    BYZANTINE = auto()      # 拜占庭容错


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    reliability: float = 1.0
    is_active: bool = True
    failure_probability: float = 0.0


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    is_critical: bool = False
    priority: int = 1


@dataclass
class BackupAssignment:
    """备用分配"""
    task_id: str
    primary_agent: str
    backup_agents: List[str] = field(default_factory=list)
    failover_order: List[str] = field(default_factory=list)


@dataclass
class FaultTolerantCoalition:
    """容错联盟"""
    coalition_id: str
    primary_assignments: Dict[str, str] = field(default_factory=dict)
    backup_assignments: Dict[str, BackupAssignment] = field(default_factory=dict)
    fault_tolerance_level: FaultToleranceLevel = FaultToleranceLevel.PRIMARY_BACKUP
    
    def get_active_agent(self, task_id: str) -> Optional[str]:
        """获取任务的当前活跃Agent"""
        if task_id in self.primary_assignments:
            return self.primary_assignments[task_id]
        return None
    
    def get_backup_agents(self, task_id: str) -> List[str]:
        """获取任务的备用Agent列表"""
        if task_id in self.backup_assignments:
            return self.backup_assignments[task_id].backup_agents
        return []


class FaultTolerantFormation:
    """容错形成器"""
    
    def __init__(
        self,
        fault_tolerance_level: FaultToleranceLevel = FaultToleranceLevel.PRIMARY_BACKUP,
        backup_ratio: float = 0.5
    ):
        self.fault_tolerance_level = fault_tolerance_level
        self.backup_ratio = backup_ratio  # 备用Agent比例
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
    
    def register_agent(self, agent: Agent) -> None:
        """注册Agent"""
        self.agents[agent.agent_id] = agent
    
    def register_task(self, task: Task) -> None:
        """注册任务"""
        self.tasks[task.task_id] = task
    
    def form_fault_tolerant_coalition(
        self,
        coalition_id: str,
        primary_assignments: Dict[str, str]
    ) -> FaultTolerantCoalition:
        """
        形成容错联盟
        
        Args:
            coalition_id: 联盟ID
            primary_assignments: 主分配方案 {task_id: agent_id}
        """
        coalition = FaultTolerantCoalition(
            coalition_id=coalition_id,
            primary_assignments=primary_assignments,
            fault_tolerance_level=self.fault_tolerance_level
        )
        
        # 为每个任务分配备用Agent
        for task_id, primary_agent_id in primary_assignments.items():
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            primary_agent = self.agents.get(primary_agent_id)
            
            if not primary_agent:
                continue
            
            # 计算需要的备用数量
            num_backups = self._calculate_backup_count(task)
            
            # 选择备用Agent
            backup_agents = self._select_backup_agents(
                task, primary_agent_id, num_backups
            )
            
            backup_assignment = BackupAssignment(
                task_id=task_id,
                primary_agent=primary_agent_id,
                backup_agents=backup_agents,
                failover_order=backup_agents.copy()
            )
            
            coalition.backup_assignments[task_id] = backup_assignment
        
        return coalition
    
    def _calculate_backup_count(self, task: Task) -> int:
        """计算需要的备用Agent数量"""
        if not task.is_critical:
            return 0
        
        if self.fault_tolerance_level == FaultToleranceLevel.PRIMARY_BACKUP:
            return 1
        elif self.fault_tolerance_level == FaultToleranceLevel.ACTIVE_REPLICATION:
            return max(1, int(len(self.agents) * self.backup_ratio))
        elif self.fault_tolerance_level == FaultToleranceLevel.BYZANTINE:
            # 拜占庭容错需要 3f+1 个节点
            return 3
        
        return 0
    
    def _select_backup_agents(
        self,
        task: Task,
        primary_agent_id: str,
        num_backups: int
    ) -> List[str]:
        """选择备用Agent"""
        candidates = []
        
        for agent_id, agent in self.agents.items():
            if agent_id == primary_agent_id:
                continue
            
            # 检查能力匹配
            if not task.required_capabilities.issubset(agent.capabilities):
                continue
            
            # 计算候选得分（可靠性越高越好）
            score = agent.reliability * (1 - agent.failure_probability)
            candidates.append((agent_id, score))
        
        # 按得分排序
        candidates.sort(key=lambda x: -x[1])
        
        # 选择前N个
        return [aid for aid, _ in candidates[:num_backups]]
    
    def handle_agent_failure(
        self,
        coalition: FaultTolerantCoalition,
        failed_agent_id: str
    ) -> Dict[str, str]:
        """
        处理Agent故障
        
        Returns:
            需要重新分配的任务 {task_id: new_agent_id}
        """
        reassignments: Dict[str, str] = {}
        
        # 找到所有受影响的任务
        for task_id, primary_id in coalition.primary_assignments.items():
            if primary_id == failed_agent_id:
                # 主Agent故障，切换到备用
                backup_assignment = coalition.backup_assignments.get(task_id)
                if backup_assignment and backup_assignment.failover_order:
                    # 选择第一个可用的备用
                    for backup_id in backup_assignment.failover_order:
                        agent = self.agents.get(backup_id)
                        if agent and agent.is_active:
                            reassignments[task_id] = backup_id
                            coalition.primary_assignments[task_id] = backup_id
                            backup_assignment.failover_order.remove(backup_id)
                            break
        
        return reassignments
    
    def calculate_system_reliability(
        self,
        coalition: FaultTolerantCoalition
    ) -> float:
        """计算系统整体可靠性"""
        if not coalition.primary_assignments:
            return 1.0
        
        total_reliability = 0.0
        
        for task_id in coalition.primary_assignments:
            task_reliability = self._calculate_task_reliability(coalition, task_id)
            total_reliability += task_reliability
        
        return total_reliability / len(coalition.primary_assignments)
    
    def _calculate_task_reliability(
        self,
        coalition: FaultTolerantCoalition,
        task_id: str
    ) -> float:
        """计算单个任务的可靠性"""
        primary_id = coalition.primary_assignments.get(task_id)
        if not primary_id:
            return 0.0
        
        primary_agent = self.agents.get(primary_id)
        if not primary_agent:
            return 0.0
        
        # 主Agent可靠性
        primary_reliability = primary_agent.reliability
        
        # 如果有备用，计算备用带来的额外可靠性
        backup_assignment = coalition.backup_assignments.get(task_id)
        if backup_assignment and backup_assignment.backup_agents:
            backup_reliability = 1.0
            for backup_id in backup_assignment.backup_agents:
                backup_agent = self.agents.get(backup_id)
                if backup_agent:
                    backup_reliability *= (1 - backup_agent.reliability)
            
            # 总可靠性 = 主可靠 + (1-主可靠) * (1 - 所有备用都失败)
            total_reliability = primary_reliability + (1 - primary_reliability) * (1 - backup_reliability)
            return min(1.0, total_reliability)
        
        return primary_reliability


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self):
        self.failure_history: List[Tuple[str, float]] = []  # (agent_id, timestamp)
        self.recovery_strategies: Dict[str, Any] = {}
    
    def record_failure(self, agent_id: str, timestamp: float) -> None:
        """记录故障"""
        self.failure_history.append((agent_id, timestamp))
    
    def detect_failure_pattern(self) -> Dict[str, Any]:
        """检测故障模式"""
        # 简化的故障模式检测
        agent_failures: Dict[str, int] = {}
        
        for agent_id, _ in self.failure_history:
            agent_failures[agent_id] = agent_failures.get(agent_id, 0) + 1
        
        # 找出频繁故障的Agent
        frequent_failures = {
            aid: count for aid, count in agent_failures.items()
            if count > 2
        }
        
        return {
            "frequent_failures": frequent_failures,
            "total_failures": len(self.failure_history),
            "unique_agents": len(agent_failures)
        }
    
    def get_recommendations(self) -> List[str]:
        """获取改进建议"""
        pattern = self.detect_failure_pattern()
        recommendations = []
        
        if pattern["frequent_failures"]:
            recommendations.append(
                f"Consider removing frequently failing agents: {list(pattern['frequent_failures'].keys())}"
            )
        
        if pattern["total_failures"] > 5:
            recommendations.append(
                "High failure rate detected. Consider increasing fault tolerance level."
            )
        
        return recommendations
