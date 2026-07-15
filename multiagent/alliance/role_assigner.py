"""
角色分配器

动态角色分配，选举Leader、Executor、Verifier等角色。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class RoleType(Enum):
    """角色类型"""
    LEADER = auto()
    EXECUTOR = auto()
    VERIFIER = auto()
    COORDINATOR = auto()
    MONITOR = auto()
    SPECIALIST = auto()


class ElectionStrategy(Enum):
    """选举策略"""
    HIGHEST_CAPABILITY = auto()
    ROUND_ROBIN = auto()
    RANDOM = auto()
    REPUTATION_BASED = auto()
    CONSENSUS = auto()


@dataclass
class Agent:
    """Agent表示"""
    agent_id: str
    capabilities: Set[str] = field(default_factory=set)
    reputation: float = 1.0
    current_load: float = 0.0
    max_load: float = 10.0
    role_history: List[RoleType] = field(default_factory=list)
    performance_scores: Dict[RoleType, float] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.agent_id)


@dataclass
class Role:
    """角色定义"""
    role_type: RoleType
    required_capabilities: Set[str] = field(default_factory=set)
    max_agents: int = 1
    min_agents: int = 1
    priority: int = 1
    description: str = ""


@dataclass
class CoalitionRoles:
    """联盟角色配置"""
    coalition_id: str
    assignments: Dict[RoleType, List[str]] = field(default_factory=dict)
    role_map: Dict[str, RoleType] = field(default_factory=dict)
    
    def get_agents_by_role(self, role_type: RoleType) -> List[str]:
        return self.assignments.get(role_type, [])
    
    def get_role_of_agent(self, agent_id: str) -> Optional[RoleType]:
        return self.role_map.get(agent_id)


class RoleAssigner(ABC):
    """角色分配器抽象基类"""
    
    def __init__(self, strategy: ElectionStrategy):
        self.strategy = strategy
        self.agents: Dict[str, Agent] = {}
        self.roles: Dict[RoleType, Role] = {}
    
    def register_agent(self, agent: Agent) -> None:
        self.agents[agent.agent_id] = agent
    
    def define_role(self, role: Role) -> None:
        self.roles[role.role_type] = role
    
    @abstractmethod
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        pass
    
    def _calculate_role_aptitude(self, agent: Agent, role: Role) -> float:
        if not role.required_capabilities:
            return 1.0
        matched = len(agent.capabilities & role.required_capabilities)
        total = len(role.required_capabilities)
        capability_score = matched / total if total > 0 else 1.0
        performance_score = agent.performance_scores.get(role.role_type, 1.0)
        load_factor = 1.0 - (agent.current_load / agent.max_load)
        reputation_factor = agent.reputation
        return 0.4 * capability_score + 0.3 * performance_score + 0.2 * load_factor + 0.1 * reputation_factor


class CapabilityBasedAssigner(RoleAssigner):
    """基于能力的角色分配器"""
    
    def __init__(self):
        super().__init__(ElectionStrategy.HIGHEST_CAPABILITY)
    
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        result = CoalitionRoles(coalition_id=coalition_id)
        sorted_roles = sorted(self.roles.values(), key=lambda r: -r.priority)
        available_agents = set(self.agents.keys())
        
        for role in sorted_roles:
            aptitudes = [(aid, self._calculate_role_aptitude(self.agents[aid], role)) for aid in available_agents]
            aptitudes.sort(key=lambda x: -x[1])
            num_to_assign = min(role.max_agents, len(aptitudes))
            selected = aptitudes[:num_to_assign]
            
            if len(selected) >= role.min_agents:
                agent_ids = [aid for aid, _ in selected]
                result.assignments[role.role_type] = agent_ids
                for aid in agent_ids:
                    result.role_map[aid] = role.role_type
                    available_agents.discard(aid)
        return result


class RoundRobinAssigner(RoleAssigner):
    """轮询角色分配器"""
    
    def __init__(self):
        super().__init__(ElectionStrategy.ROUND_ROBIN)
        self.last_index: Dict[RoleType, int] = {}
    
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        result = CoalitionRoles(coalition_id=coalition_id)
        agent_list = list(self.agents.keys())
        if not agent_list:
            return result
        
        for role_type, role in self.roles.items():
            last_idx = self.last_index.get(role_type, -1)
            agent_ids = [agent_list[(last_idx + 1 + i) % len(agent_list)] for i in range(role.max_agents)]
            result.assignments[role_type] = agent_ids
            for aid in agent_ids:
                result.role_map[aid] = role_type
            self.last_index[role_type] = (last_idx + role.max_agents) % len(agent_list)
        return result


class RandomAssigner(RoleAssigner):
    """随机角色分配器"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(ElectionStrategy.RANDOM)
        self.rng = random.Random(seed)
    
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        result = CoalitionRoles(coalition_id=coalition_id)
        agent_list = list(self.agents.keys())
        self.rng.shuffle(agent_list)
        
        idx = 0
        for role_type, role in self.roles.items():
            num = min(role.max_agents, len(agent_list) - idx)
            if num >= role.min_agents:
                selected = agent_list[idx:idx + num]
                result.assignments[role_type] = selected
                for aid in selected:
                    result.role_map[aid] = role_type
                idx += num
        return result


class ReputationBasedAssigner(RoleAssigner):
    """基于信誉的角色分配器"""
    
    def __init__(self):
        super().__init__(ElectionStrategy.REPUTATION_BASED)
    
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        result = CoalitionRoles(coalition_id=coalition_id)
        sorted_roles = sorted(self.roles.values(), key=lambda r: -r.priority)
        available = set(self.agents.keys())
        
        for role in sorted_roles:
            candidates = [(aid, self.agents[aid].reputation) for aid in available]
            candidates.sort(key=lambda x: -x[1])
            num = min(role.max_agents, len(candidates))
            selected = [aid for aid, _ in candidates[:num]]
            
            if len(selected) >= role.min_agents:
                result.assignments[role.role_type] = selected
                for aid in selected:
                    result.role_map[aid] = role.role_type
                    available.discard(aid)
        return result


class ConsensusAssigner(RoleAssigner):
    """共识选举分配器"""
    
    def __init__(self):
        super().__init__(ElectionStrategy.CONSENSUS)
        self.votes: Dict[str, Dict[str, int]] = {}
    
    def cast_vote(self, voter_id: str, candidate_id: str, role_type: RoleType) -> None:
        if voter_id not in self.votes:
            self.votes[voter_id] = {}
        self.votes[voter_id][f"{candidate_id}:{role_type.name}"] = self.votes[voter_id].get(f"{candidate_id}:{role_type.name}", 0) + 1
    
    def assign_roles(self, coalition_id: str) -> CoalitionRoles:
        result = CoalitionRoles(coalition_id=coalition_id)
        
        for role_type, role in self.roles.items():
            vote_counts: Dict[str, int] = {}
            for voter_votes in self.votes.values():
                for key, count in voter_votes.items():
                    if key.endswith(f":{role_type.name}"):
                        candidate = key.split(":")[0]
                        vote_counts[candidate] = vote_counts.get(candidate, 0) + count
            
            if vote_counts:
                winner = max(vote_counts.items(), key=lambda x: x[1])[0]
                result.assignments[role_type] = [winner]
                result.role_map[winner] = role_type
        
        self.votes.clear()
        return result
