"""
自组织规则系统 - Self-Organization Rules

无需中央调度，Agent基于规则自主组队。
实现了基于任务特征匹配、能力互补和动态聚类的自组织算法。
"""

from __future__ import annotations

import heapq
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar


class OrganizationStrategy(Enum):
    """自组织策略类型"""
    CAPABILITY_MATCHING = auto()      # 能力匹配
    LOAD_BALANCING = auto()           # 负载均衡
    PROXIMITY_BASED = auto()          # 邻近性
    HIERARCHICAL = auto()             # 层次化
    ADHOC = auto()                    # 临时组队


@dataclass
class Capability:
    """Agent能力描述"""
    name: str
    level: float  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Capability):
            return NotImplemented
        return self.name == other.name


@dataclass
class TaskRequirement:
    """任务需求描述"""
    task_id: str
    required_capabilities: Set[str]
    min_capability_level: float = 0.5
    priority: int = 5  # 1-10, 1最高
    max_team_size: int = 5
    deadline: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentProfile:
    """Agent档案"""
    agent_id: str
    capabilities: Set[Capability]
    current_load: float = 0.0  # 0.0 - 1.0
    reputation: float = 1.0    # 0.0 - 1.0
    last_active: float = field(default_factory=time.time)
    team_affinity: Dict[str, float] = field(default_factory=dict)  # 与其他Agent的亲和度
    
    def has_capability(self, cap_name: str, min_level: float = 0.5) -> bool:
        """检查是否具备某能力"""
        for cap in self.capabilities:
            if cap.name == cap_name and cap.level >= min_level:
                return True
        return False
    
    def get_capability_level(self, cap_name: str) -> float:
        """获取某能力的等级"""
        for cap in self.capabilities:
            if cap.name == cap_name:
                return cap.level
        return 0.0
    
    def calculate_match_score(self, requirement: TaskRequirement) -> float:
        """计算与任务需求的匹配分数"""
        if not requirement.required_capabilities:
            return 0.0
        
        total_score = 0.0
        matched_caps = 0
        
        for req_cap in requirement.required_capabilities:
            level = self.get_capability_level(req_cap)
            if level >= requirement.min_capability_level:
                total_score += level
                matched_caps += 1
        
        if matched_caps == 0:
            return 0.0
        
        # 基础匹配分
        match_ratio = matched_caps / len(requirement.required_capabilities)
        avg_level = total_score / matched_caps
        
        # 负载惩罚
        load_penalty = self.current_load * 0.3
        
        # 声誉加成
        reputation_bonus = self.reputation * 0.1
        
        return (match_ratio * 0.4 + avg_level * 0.4 - load_penalty + reputation_bonus)


T = TypeVar('T')


class TeamFormationRule(ABC, Generic[T]):
    """团队组建规则基类"""
    
    @abstractmethod
    def evaluate(self, agent: AgentProfile, team: List[AgentProfile], 
                 requirement: TaskRequirement) -> float:
        """评估Agent加入团队的适合度，返回分数"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取规则名称"""
        pass


class CapabilityCoverageRule(TeamFormationRule):
    """能力覆盖规则 - 优先补充团队缺失的能力"""
    
    def get_name(self) -> str:
        return "CapabilityCoverage"
    
    def evaluate(self, agent: AgentProfile, team: List[AgentProfile],
                 requirement: TaskRequirement) -> float:
        if not team:
            return agent.calculate_match_score(requirement)
        
        # 计算团队已有能力
        team_capabilities: Dict[str, float] = {}
        for member in team:
            for cap in member.capabilities:
                team_capabilities[cap.name] = max(
                    team_capabilities.get(cap.name, 0.0), cap.level
                )
        
        # 计算Agent能补充的能力
        contribution = 0.0
        for req_cap in requirement.required_capabilities:
            agent_level = agent.get_capability_level(req_cap)
            team_level = team_capabilities.get(req_cap, 0.0)
            if agent_level > team_level:
                contribution += agent_level - team_level
        
        return contribution / max(len(requirement.required_capabilities), 1)


class LoadBalancingRule(TeamFormationRule):
    """负载均衡规则 - 优先选择负载低的Agent"""
    
    def get_name(self) -> str:
        return "LoadBalancing"
    
    def evaluate(self, agent: AgentProfile, team: List[AgentProfile],
                 requirement: TaskRequirement) -> float:
        # 负载越低分数越高
        return 1.0 - agent.current_load


class TeamAffinityRule(TeamFormationRule):
    """团队亲和度规则 - 优先选择有合作历史的Agent"""
    
    def get_name(self) -> str:
        return "TeamAffinity"
    
    def evaluate(self, agent: AgentProfile, team: List[AgentProfile],
                 requirement: TaskRequirement) -> float:
        if not team:
            return 0.5
        
        total_affinity = 0.0
        for member in team:
            affinity = agent.team_affinity.get(member.agent_id, 0.5)
            total_affinity += affinity
        
        return total_affinity / len(team)


class DiversityRule(TeamFormationRule):
    """多样性规则 - 避免能力过度重叠"""
    
    def get_name(self) -> str:
        return "Diversity"
    
    def evaluate(self, agent: AgentProfile, team: List[AgentProfile],
                 requirement: TaskRequirement) -> float:
        if not team:
            return 1.0
        
        # 计算Agent能力与团队的重叠度
        agent_caps = {cap.name for cap in agent.capabilities}
        team_caps: Set[str] = set()
        for member in team:
            team_caps.update(cap.name for cap in member.capabilities)
        
        if not agent_caps:
            return 0.0
        
        overlap = len(agent_caps & team_caps) / len(agent_caps)
        # 重叠度越低分数越高
        return 1.0 - overlap


@dataclass
class Team:
    """团队"""
    team_id: str
    members: List[AgentProfile] = field(default_factory=list)
    requirement: Optional[TaskRequirement] = None
    formation_time: float = field(default_factory=time.time)
    status: str = "forming"  # forming, active, disbanded
    performance_score: float = 0.0
    
    def get_collective_capabilities(self) -> Dict[str, float]:
        """获取团队的集体能力"""
        caps: Dict[str, float] = {}
        for member in self.members:
            for cap in member.capabilities:
                caps[cap.name] = max(caps.get(cap.name, 0.0), cap.level)
        return caps
    
    def can_fulfill_requirement(self, requirement: TaskRequirement) -> bool:
        """检查团队是否能满足需求"""
        collective = self.get_collective_capabilities()
        for req_cap in requirement.required_capabilities:
            if collective.get(req_cap, 0.0) < requirement.min_capability_level:
                return False
        return True
    
    def calculate_cohesion(self) -> float:
        """计算团队凝聚力"""
        if len(self.members) < 2:
            return 1.0
        
        total_affinity = 0.0
        count = 0
        for i, m1 in enumerate(self.members):
            for m2 in self.members[i+1:]:
                affinity = m1.team_affinity.get(m2.agent_id, 0.5)
                total_affinity += affinity
                count += 1
        
        return total_affinity / max(count, 1)


class SelfOrganizationEngine:
    """自组织引擎 - 核心调度器"""
    
    def __init__(self, strategy: OrganizationStrategy = OrganizationStrategy.CAPABILITY_MATCHING):
        self.strategy = strategy
        self.agents: Dict[str, AgentProfile] = {}
        self.teams: Dict[str, Team] = {}
        self.pending_requirements: List[TaskRequirement] = []
        self.rules: List[TeamFormationRule] = [
            CapabilityCoverageRule(),
            LoadBalancingRule(),
            TeamAffinityRule(),
            DiversityRule()
        ]
        self.rule_weights: Dict[str, float] = {
            "CapabilityCoverage": 0.4,
            "LoadBalancing": 0.2,
            "TeamAffinity": 0.2,
            "Diversity": 0.2
        }
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
    def register_agent(self, agent: AgentProfile) -> None:
        """注册Agent"""
        with self._lock:
            self.agents[agent.agent_id] = agent
            
    def unregister_agent(self, agent_id: str) -> None:
        """注销Agent"""
        with self._lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                # 从团队中移除
                for team in self.teams.values():
                    team.members = [m for m in team.members if m.agent_id != agent_id]
                    
    def submit_requirement(self, requirement: TaskRequirement) -> Optional[str]:
        """提交任务需求，返回团队ID或None"""
        with self._lock:
            team = self._form_team(requirement)
            if team:
                self.teams[team.team_id] = team
                # 更新Agent负载
                for member in team.members:
                    member.current_load = min(1.0, member.current_load + 0.2)
                return team.team_id
            else:
                self.pending_requirements.append(requirement)
                return None
    
    def _form_team(self, requirement: TaskRequirement) -> Optional[Team]:
        """组建团队"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.current_load < 0.8  # 负载不能太高
        ]
        
        if not available_agents:
            return None
        
        # 根据策略选择组建算法
        if self.strategy == OrganizationStrategy.CAPABILITY_MATCHING:
            return self._form_by_capability_matching(available_agents, requirement)
        elif self.strategy == OrganizationStrategy.LOAD_BALANCING:
            return self._form_by_load_balancing(available_agents, requirement)
        elif self.strategy == OrganizationStrategy.HIERARCHICAL:
            return self._form_hierarchical(available_agents, requirement)
        else:
            return self._form_by_capability_matching(available_agents, requirement)
    
    def _form_by_capability_matching(self, agents: List[AgentProfile],
                                      requirement: TaskRequirement) -> Optional[Team]:
        """基于能力匹配组建团队"""
        # 计算每个Agent的匹配分数
        scored_agents: List[Tuple[float, AgentProfile]] = []
        for agent in agents:
            score = self._calculate_weighted_score(agent, [], requirement)
            scored_agents.append((score, agent))
        
        # 按分数排序
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        
        # 贪心选择，直到满足所有能力需求
        team_members: List[AgentProfile] = []
        covered_capabilities: Set[str] = set()
        
        for score, agent in scored_agents:
            if len(team_members) >= requirement.max_team_size:
                break
            
            # 检查Agent是否能提供新能力
            new_capabilities = False
            for cap in agent.capabilities:
                if cap.name in requirement.required_capabilities:
                    if cap.name not in covered_capabilities:
                        new_capabilities = True
                        break
                    elif cap.level > requirement.min_capability_level:
                        # 即使能力已覆盖，更高等级也有价值
                        pass
            
            if new_capabilities or len(team_members) == 0:
                team_members.append(agent)
                for cap in agent.capabilities:
                    if cap.level >= requirement.min_capability_level:
                        covered_capabilities.add(cap.name)
            
            # 检查是否满足所有需求
            if requirement.required_capabilities <= covered_capabilities:
                break
        
        if requirement.required_capabilities <= covered_capabilities:
            team = Team(
                team_id=f"team_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                members=team_members,
                requirement=requirement,
                status="active"
            )
            return team
        
        return None
    
    def _form_by_load_balancing(self, agents: List[AgentProfile],
                                 requirement: TaskRequirement) -> Optional[Team]:
        """基于负载均衡组建团队"""
        # 优先选择负载低的Agent
        sorted_agents = sorted(agents, key=lambda a: a.current_load)
        return self._form_by_capability_matching(sorted_agents, requirement)
    
    def _form_hierarchical(self, agents: List[AgentProfile],
                           requirement: TaskRequirement) -> Optional[Team]:
        """层次化组建 - 先选leader，再选成员"""
        if not agents:
            return None
        
        # 选择leader（能力最强且负载适中）
        leader_candidates = [
            (a.calculate_match_score(requirement) * (1 - a.current_load * 0.5), a)
            for a in agents
        ]
        leader_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if not leader_candidates:
            return None
        
        leader = leader_candidates[0][1]
        team_members = [leader]
        
        # 选择补充成员
        remaining_agents = [a for a in agents if a.agent_id != leader.agent_id]
        team = self._form_by_capability_matching(remaining_agents, requirement)
        
        if team:
            team.members = [leader] + team.members
            team.team_id = f"team_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            return team
        
        return None
    
    def _calculate_weighted_score(self, agent: AgentProfile, team: List[AgentProfile],
                                   requirement: TaskRequirement) -> float:
        """计算加权分数"""
        total_score = 0.0
        for rule in self.rules:
            score = rule.evaluate(agent, team, requirement)
            weight = self.rule_weights.get(rule.get_name(), 0.25)
            total_score += score * weight
        return total_score
    
    def disband_team(self, team_id: str) -> None:
        """解散团队"""
        with self._lock:
            if team_id in self.teams:
                team = self.teams[team_id]
                team.status = "disbanded"
                # 减少成员负载
                for member in team.members:
                    member.current_load = max(0.0, member.current_load - 0.2)
                del self.teams[team_id]
    
    def update_team_performance(self, team_id: str, performance: float) -> None:
        """更新团队表现"""
        with self._lock:
            if team_id in self.teams:
                team = self.teams[team_id]
                team.performance_score = performance
                # 更新成员声誉
                for member in team.members:
                    member.reputation = 0.9 * member.reputation + 0.1 * performance
                    member.last_active = time.time()
    
    def update_affinity(self, agent_id1: str, agent_id2: str, delta: float) -> None:
        """更新两个Agent之间的亲和度"""
        with self._lock:
            if agent_id1 in self.agents and agent_id2 in self.agents:
                a1 = self.agents[agent_id1]
                a2 = self.agents[agent_id2]
                current = a1.team_affinity.get(agent_id2, 0.5)
                a1.team_affinity[agent_id2] = max(0.0, min(1.0, current + delta))
                a2.team_affinity[agent_id1] = a1.team_affinity[agent_id2]
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """获取团队信息"""
        with self._lock:
            return self.teams.get(team_id)
    
    def get_agent_teams(self, agent_id: str) -> List[Team]:
        """获取Agent所属的所有团队"""
        with self._lock:
            return [
                team for team in self.teams.values()
                if any(m.agent_id == agent_id for m in team.members)
            ]
    
    def set_rule_weights(self, weights: Dict[str, float]) -> None:
        """设置规则权重"""
        with self._lock:
            self.rule_weights.update(weights)
    
    def start(self) -> None:
        """启动自组织引擎"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._worker_thread.start()
    
    def stop(self) -> None:
        """停止自组织引擎"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
    
    def _main_loop(self) -> None:
        """主循环 - 处理待处理的需求"""
        while self._running:
            with self._lock:
                # 处理待处理的需求
                still_pending: List[TaskRequirement] = []
                for req in self.pending_requirements:
                    team = self._form_team(req)
                    if team:
                        self.teams[team.team_id] = team
                        for member in team.members:
                            member.current_load = min(1.0, member.current_load + 0.2)
                    else:
                        still_pending.append(req)
                self.pending_requirements = still_pending
                
                # 检查超时团队
                current_time = time.time()
                for team_id, team in list(self.teams.items()):
                    if team.requirement and team.requirement.deadline:
                        if current_time > team.requirement.deadline:
                            self.disband_team(team_id)
            
            time.sleep(1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "registered_agents": len(self.agents),
                "active_teams": len(self.teams),
                "pending_requirements": len(self.pending_requirements),
                "avg_agent_load": sum(a.current_load for a in self.agents.values()) / max(len(self.agents), 1),
                "avg_team_size": sum(len(t.members) for t in self.teams.values()) / max(len(self.teams), 1),
                "strategy": self.strategy.name
            }


class EmergentBehaviorDetector:
    """涌现行为检测器 - 检测自组织过程中出现的涌现模式"""
    
    def __init__(self, engine: SelfOrganizationEngine):
        self.engine = engine
        self.pattern_history: List[Dict[str, Any]] = []
        self._detection_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def register_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """注册模式检测回调"""
        self._detection_callbacks.append(callback)
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """检测涌现模式"""
        patterns = []
        
        # 检测核心-边缘结构
        core_periphery = self._detect_core_periphery()
        if core_periphery:
            patterns.append({
                "type": "core_periphery",
                "data": core_periphery
            })
        
        # 检测小团体
        cliques = self._detect_cliques()
        if cliques:
            patterns.append({
                "type": "cliques",
                "data": cliques
            })
        
        # 检测层级结构
        hierarchy = self._detect_hierarchy()
        if hierarchy:
            patterns.append({
                "type": "hierarchy",
                "data": hierarchy
            })
        
        self.pattern_history.extend(patterns)
        
        # 触发回调
        for pattern in patterns:
            for callback in self._detection_callbacks:
                callback(pattern["type"], pattern["data"])
        
        return patterns
    
    def _detect_core_periphery(self) -> Optional[Dict[str, Any]]:
        """检测核心-边缘结构"""
        with self.engine._lock:
            if len(self.engine.agents) < 5:
                return None
            
            # 计算每个Agent的连接度（参与的团队数）
            degrees: Dict[str, int] = {}
            for agent_id in self.engine.agents:
                degrees[agent_id] = len(self.engine.get_agent_teams(agent_id))
            
            if not degrees:
                return None
            
            avg_degree = sum(degrees.values()) / len(degrees)
            core = [aid for aid, deg in degrees.items() if deg > avg_degree * 1.5]
            periphery = [aid for aid, deg in degrees.items() if deg <= avg_degree * 0.5]
            
            if len(core) > 0 and len(periphery) > 0:
                return {
                    "core_agents": core,
                    "periphery_agents": periphery,
                    "core_ratio": len(core) / len(degrees)
                }
            return None
    
    def _detect_cliques(self) -> Optional[List[Set[str]]]:
        """检测小团体（经常一起组队的Agent组）"""
        with self.engine._lock:
            if len(self.engine.teams) < 3:
                return None
            
            # 构建共现矩阵
            cooccurrence: Dict[Tuple[str, str], int] = {}
            for team in self.engine.teams.values():
                members = [m.agent_id for m in team.members]
                for i, m1 in enumerate(members):
                    for m2 in members[i+1:]:
                        key = tuple(sorted([m1, m2]))
                        cooccurrence[key] = cooccurrence.get(key, 0) + 1
            
            # 寻找频繁共现的组
            threshold = max(2, len(self.engine.teams) * 0.3)
            cliques: List[Set[str]] = []
            
            for (m1, m2), count in cooccurrence.items():
                if count >= threshold:
                    # 检查是否属于已有小团体
                    added = False
                    for clique in cliques:
                        if m1 in clique or m2 in clique:
                            clique.add(m1)
                            clique.add(m2)
                            added = True
                            break
                    if not added:
                        cliques.append({m1, m2})
            
            return cliques if cliques else None
    
    def _detect_hierarchy(self) -> Optional[Dict[str, Any]]:
        """检测层级结构"""
        with self.engine._lock:
            if len(self.engine.teams) < 3:
                return None
            
            # 基于团队规模和成员重叠度推断层级
            team_sizes = [len(t.members) for t in self.engine.teams.values()]
            avg_size = sum(team_sizes) / len(team_sizes) if team_sizes else 0
            
            large_teams = [t for t in self.engine.teams.values() if len(t.members) > avg_size]
            small_teams = [t for t in self.engine.teams.values() if len(t.members) <= avg_size]
            
            if large_teams and small_teams:
                return {
                    "coordinators": [t.team_id for t in large_teams],
                    "workers": [t.team_id for t in small_teams],
                    "levels": 2
                }
            return None
