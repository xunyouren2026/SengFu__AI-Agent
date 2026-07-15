"""
动态重规划

任务执行中Agent掉线时重新分配，支持在线重规划。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time


class ReplanningTrigger(Enum):
    """重规划触发条件"""
    AGENT_FAILURE = auto()
    TASK_FAILURE = auto()
    NEW_TASK_ARRIVAL = auto()
    PERFORMANCE_DEGRADATION = auto()
    TIMEOUT = auto()


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    is_online: bool = True
    current_load: float = 0.0
    max_load: float = 10.0


@dataclass
class Task:
    """任务表示"""
    task_id: str
    required_capabilities: Set[str] = field(default_factory=set)
    status: str = "pending"
    assigned_agent: Optional[str] = None
    priority: int = 1


@dataclass
class ExecutionState:
    """执行状态"""
    task_id: str
    agent_id: str
    start_time: float
    progress: float = 0.0
    status: str = "running"


@dataclass
class ReplanningResult:
    """重规划结果"""
    trigger: ReplanningTrigger
    affected_tasks: List[str]
    new_assignments: Dict[str, str]
    removed_agents: List[str]
    replanning_time_ms: float = 0.0


class DynamicReplanningEngine:
    """动态重规划引擎"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.execution_states: Dict[str, ExecutionState] = {}
        self.assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.replanning_history: List[ReplanningResult] = []
    
    def register_agent(self, agent: Agent) -> None:
        """注册Agent"""
        self.agents[agent.agent_id] = agent
    
    def register_task(self, task: Task) -> None:
        """注册任务"""
        self.tasks[task.task_id] = task
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """分配任务"""
        if task_id not in self.tasks or agent_id not in self.agents:
            return False
        
        self.assignments[task_id] = agent_id
        self.tasks[task_id].assigned_agent = agent_id
        self.tasks[task_id].status = "assigned"
        
        return True
    
    def start_execution(self, task_id: str) -> bool:
        """开始执行任务"""
        if task_id not in self.assignments:
            return False
        
        agent_id = self.assignments[task_id]
        state = ExecutionState(
            task_id=task_id,
            agent_id=agent_id,
            start_time=time.time()
        )
        self.execution_states[task_id] = state
        self.tasks[task_id].status = "running"
        
        return True
    
    def report_agent_failure(self, agent_id: str) -> ReplanningResult:
        """报告Agent故障，触发重规划"""
        start_time = time.time()
        
        if agent_id not in self.agents:
            return ReplanningResult(
                trigger=ReplanningTrigger.AGENT_FAILURE,
                affected_tasks=[],
                new_assignments={},
                removed_agents=[]
            )
        
        # 标记Agent离线
        self.agents[agent_id].is_online = False
        
        # 找到所有受影响的任务
        affected_tasks = [
            task_id for task_id, aid in self.assignments.items()
            if aid == agent_id
        ]
        
        # 重新分配任务
        new_assignments = self._reassign_tasks(affected_tasks)
        
        result = ReplanningResult(
            trigger=ReplanningTrigger.AGENT_FAILURE,
            affected_tasks=affected_tasks,
            new_assignments=new_assignments,
            removed_agents=[agent_id],
            replanning_time_ms=(time.time() - start_time) * 1000
        )
        
        self.replanning_history.append(result)
        return result
    
    def _reassign_tasks(self, task_ids: List[str]) -> Dict[str, str]:
        """重新分配任务"""
        new_assignments = {}
        
        for task_id in task_ids:
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            # 找到可用的Agent
            best_agent = self._find_best_available_agent(task)
            
            if best_agent:
                # 更新分配
                old_agent = self.assignments.get(task_id)
                if old_agent:
                    del self.assignments[task_id]
                
                self.assignments[task_id] = best_agent
                task.assigned_agent = best_agent
                new_assignments[task_id] = best_agent
            else:
                # 没有可用Agent，标记为未分配
                task.status = "unassigned"
                task.assigned_agent = None
                if task_id in self.assignments:
                    del self.assignments[task_id]
        
        return new_assignments
    
    def _find_best_available_agent(self, task: Task) -> Optional[str]:
        """找到最佳可用Agent"""
        candidates = []
        
        for agent_id, agent in self.agents.items():
            if not agent.is_online:
                continue
            
            if not task.required_capabilities.issubset(agent.capabilities):
                continue
            
            if agent.current_load >= agent.max_load:
                continue
            
            # 计算得分（负载越低越好）
            load_ratio = agent.current_load / agent.max_load
            score = 1.0 - load_ratio
            
            candidates.append((agent_id, score))
        
        if not candidates:
            return None
        
        # 选择负载最低的
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]
    
    def check_and_replan(self) -> Optional[ReplanningResult]:
        """检查并执行重规划"""
        # 检查是否有Agent掉线
        offline_agents = [
            aid for aid, agent in self.agents.items()
            if not agent.is_online
        ]
        
        if offline_agents:
            # 合并所有受影响的任务
            all_affected = []
            for agent_id in offline_agents:
                affected = [
                    tid for tid, aid in self.assignments.items()
                    if aid == agent_id
                ]
                all_affected.extend(affected)
            
            if all_affected:
                start_time = time.time()
                new_assignments = self._reassign_tasks(list(set(all_affected)))
                
                result = ReplanningResult(
                    trigger=ReplanningTrigger.AGENT_FAILURE,
                    affected_tasks=list(set(all_affected)),
                    new_assignments=new_assignments,
                    removed_agents=offline_agents,
                    replanning_time_ms=(time.time() - start_time) * 1000
                )
                
                self.replanning_history.append(result)
                return result
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "total_agents": len(self.agents),
            "online_agents": sum(1 for a in self.agents.values() if a.is_online),
            "total_tasks": len(self.tasks),
            "assigned_tasks": len(self.assignments),
            "running_tasks": sum(1 for t in self.tasks.values() if t.status == "running"),
            "replanning_count": len(self.replanning_history)
        }


class AdaptiveReplanningStrategy:
    """自适应重规划策略"""
    
    def __init__(self):
        self.failure_threshold = 3
        self.replanning_cooldown = 60  # 秒
        self.last_replanning_time: Optional[float] = None
    
    def should_replan(
        self,
        trigger: ReplanningTrigger,
        context: Dict[str, Any]
    ) -> bool:
        """判断是否应该重规划"""
        current_time = time.time()
        
        # 检查冷却时间
        if self.last_replanning_time:
            if current_time - self.last_replanning_time < self.replanning_cooldown:
                return False
        
        if trigger == ReplanningTrigger.AGENT_FAILURE:
            return True
        
        if trigger == ReplanningTrigger.PERFORMANCE_DEGRADATION:
            degradation = context.get("performance_degradation", 0)
            return degradation > 0.3
        
        if trigger == ReplanningTrigger.TIMEOUT:
            return True
        
        return False
    
    def record_replanning(self) -> None:
        """记录重规划事件"""
        self.last_replanning_time = time.time()
