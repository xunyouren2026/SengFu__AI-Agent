"""
文化演化模拟 - Agent间信念传播与突变
"""
from __future__ import annotations
import random
import copy
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from .world import Agent, Position, World


class BeliefType(Enum):
    """信念类型"""
    COOPERATIVE = auto()      # 合作倾向
    INDIVIDUALIST = auto()    # 个人主义
    TRADITIONAL = auto()      # 传统保守
    INNOVATIVE = auto()       # 创新开放
    RISK_AVERSE = auto()      # 风险厌恶
    RISK_SEEKING = auto()     # 风险偏好


@dataclass
class Belief:
    """信念"""
    belief_type: BeliefType
    strength: float  # 0-1，信念强度
    certainty: float  # 0-1，确定性
    source: Optional[str] = None  # 来源Agent
    timestamp: float = 0.0

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.certainty = max(0.0, min(1.0, self.certainty))


@dataclass
class CulturalTrait:
    """文化特征"""
    trait_id: str
    name: str
    value: float
    mutability: float = 0.1  # 突变概率
    transmission_rate: float = 0.5  # 传播率


@dataclass
class AgentCulture:
    """Agent文化状态"""
    agent_id: str
    beliefs: Dict[BeliefType, Belief] = field(default_factory=dict)
    traits: Dict[str, CulturalTrait] = field(default_factory=dict)
    cultural_group: Optional[str] = None
    influence_score: float = 1.0  # 影响力
    openness: float = 0.5  # 开放程度

    def get_belief_strength(self, belief_type: BeliefType) -> float:
        """获取特定信念强度"""
        belief = self.beliefs.get(belief_type)
        return belief.strength if belief else 0.0

    def update_belief(self, belief: Belief, learning_rate: float = 0.1) -> None:
        """更新信念"""
        existing = self.beliefs.get(belief.belief_type)
        if existing:
            # 贝叶斯更新
            combined_strength = (existing.strength * existing.certainty +
                               belief.strength * belief.certainty * learning_rate) / \
                              (existing.certainty + belief.certainty * learning_rate + 0.01)
            existing.strength = max(0.0, min(1.0, combined_strength))
            existing.certainty = min(1.0, existing.certainty + 0.05)
        else:
            self.beliefs[belief.belief_type] = copy.deepcopy(belief)


class BeliefPropagationModel:
    """信念传播模型"""

    def __init__(self, world: World):
        self.world = world
        self.agent_cultures: Dict[str, AgentCulture] = {}
        self.propagation_history: List[Dict[str, Any]] = []
        self.mutation_rate: float = 0.05
        self.conformity_pressure: float = 0.3

    def register_agent(self, agent_id: str, openness: float = 0.5,
                       initial_beliefs: Optional[Dict[BeliefType, float]] = None) -> AgentCulture:
        """注册Agent到文化系统"""
        culture = AgentCulture(agent_id=agent_id, openness=openness)

        if initial_beliefs:
            for btype, strength in initial_beliefs.items():
                belief = Belief(
                    belief_type=btype,
                    strength=strength,
                    certainty=random.uniform(0.3, 0.7),
                    timestamp=self.world.current_time
                )
                culture.beliefs[btype] = belief

        self.agent_cultures[agent_id] = culture
        return culture

    def propagate_beliefs(self, interaction_radius: float = 10.0) -> int:
        """
        传播信念
        返回: 传播事件数量
        """
        propagation_count = 0

        for agent_id, culture in self.agent_cultures.items():
            agent = self.world.get_agent(agent_id)
            if not agent:
                continue

            # 获取邻近Agent
            neighbors = self.world.get_agents_near(agent.position, interaction_radius)

            for neighbor in neighbors:
                if neighbor.agent_id == agent_id:
                    continue

                neighbor_culture = self.agent_cultures.get(neighbor.agent_id)
                if not neighbor_culture:
                    continue

                # 尝试传播信念
                if self._attempt_propagation(culture, neighbor_culture):
                    propagation_count += 1

        return propagation_count

    def _attempt_propagation(self, source: AgentCulture, target: AgentCulture) -> bool:
        """尝试从源向目标传播信念"""
        # 检查目标开放性
        if random.random() > target.openness:
            return False

        # 选择要传播的信念
        if not source.beliefs:
            return False

        belief_to_spread = random.choice(list(source.beliefs.values()))

        # 计算传播成功概率
        influence = source.influence_score * belief_to_spread.certainty
        resistance = 1 - target.openness
        success_prob = influence * (1 - resistance)

        if random.random() < success_prob:
            # 传播成功
            new_belief = copy.deepcopy(belief_to_spread)
            new_belief.source = source.agent_id
            new_belief.timestamp = self.world.current_time

            # 可能产生突变
            if random.random() < self.mutation_rate:
                new_belief = self._mutate_belief(new_belief)

            target.update_belief(new_belief)

            self.propagation_history.append({
                "time": self.world.current_time,
                "from": source.agent_id,
                "to": target.agent_id,
                "belief": belief_to_spread.belief_type.name,
                "mutated": new_belief != belief_to_spread
            })

            return True

        return False

    def _mutate_belief(self, belief: Belief) -> Belief:
        """突变信念"""
        mutated = copy.deepcopy(belief)
        mutation = random.gauss(0, 0.1)
        mutated.strength = max(0.0, min(1.0, belief.strength + mutation))
        mutated.certainty *= random.uniform(0.8, 1.0)
        return mutated

    def apply_conformity_pressure(self, group_radius: float = 15.0) -> None:
        """应用从众压力"""
        for agent_id, culture in self.agent_cultures.items():
            agent = self.world.get_agent(agent_id)
            if not agent:
                continue

            neighbors = self.world.get_agents_near(agent.position, group_radius)
            neighbor_cultures = [
                self.agent_cultures[n.agent_id]
                for n in neighbors
                if n.agent_id in self.agent_cultures and n.agent_id != agent_id
            ]

            if len(neighbor_cultures) < 2:
                continue

            # 计算群体平均信念
            group_beliefs: Dict[BeliefType, List[float]] = defaultdict(list)
            for nc in neighbor_cultures:
                for btype, belief in nc.beliefs.items():
                    group_beliefs[btype].append(belief.strength)

            # 向群体平均靠拢
            for btype, strengths in group_beliefs.items():
                if not strengths:
                    continue
                avg_strength = sum(strengths) / len(strengths)

                if btype in culture.beliefs:
                    current = culture.beliefs[btype].strength
                    new_strength = current + (avg_strength - current) * self.conformity_pressure
                    culture.beliefs[btype].strength = new_strength

    def identify_cultural_groups(self, threshold: float = 0.7) -> Dict[str, Set[str]]:
        """识别文化群体"""
        groups: Dict[str, Set[str]] = {}
        assigned = set()

        for agent_id, culture in self.agent_cultures.items():
            if agent_id in assigned:
                continue

            # 创建新群体
            group_id = f"group_{len(groups)}"
            groups[group_id] = {agent_id}
            assigned.add(agent_id)
            culture.cultural_group = group_id

            # 寻找相似Agent
            for other_id, other_culture in self.agent_cultures.items():
                if other_id in assigned:
                    continue

                similarity = self._calculate_similarity(culture, other_culture)
                if similarity >= threshold:
                    groups[group_id].add(other_id)
                    assigned.add(other_id)
                    other_culture.cultural_group = group_id

        return groups

    def _calculate_similarity(self, c1: AgentCulture, c2: AgentCulture) -> float:
        """计算两个Agent文化的相似度"""
        all_beliefs = set(c1.beliefs.keys()) | set(c2.beliefs.keys())
        if not all_beliefs:
            return 0.0

        total_diff = 0.0
        for btype in all_beliefs:
            s1 = c1.get_belief_strength(btype)
            s2 = c2.get_belief_strength(btype)
            total_diff += abs(s1 - s2)

        avg_diff = total_diff / len(all_beliefs)
        return 1 - avg_diff

    def get_cultural_diversity(self) -> float:
        """计算文化多样性"""
        if len(self.agent_cultures) < 2:
            return 0.0

        similarities = []
        cultures = list(self.agent_cultures.values())

        for i, c1 in enumerate(cultures):
            for c2 in cultures[i+1:]:
                similarities.append(self._calculate_similarity(c1, c2))

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1 - avg_similarity


class CulturalEvolutionSimulator:
    """文化演化模拟器"""

    def __init__(self, world: World):
        self.world = world
        self.propagation_model = BeliefPropagationModel(world)
        self.evolution_history: List[Dict[str, Any]] = []
        self.selection_pressure: float = 0.1

    def step(self) -> Dict[str, Any]:
        """执行文化演化一步"""
        # 信念传播
        propagations = self.propagation_model.propagate_beliefs()

        # 从众压力
        self.propagation_model.apply_conformity_pressure()

        # 文化选择（成功的信念增强）
        self._apply_cultural_selection()

        # 识别文化群体
        groups = self.propagation_model.identify_cultural_groups()

        # 记录历史
        diversity = self.propagation_model.get_cultural_diversity()
        self.evolution_history.append({
            "time": self.world.current_time,
            "propagations": propagations,
            "num_groups": len(groups),
            "diversity": diversity
        })

        return {
            "propagations": propagations,
            "cultural_groups": len(groups),
            "diversity": diversity
        }

    def _apply_cultural_selection(self) -> None:
        """应用文化选择"""
        # 这里可以基于Agent的适应度来调整信念
        pass

    def get_cultural_statistics(self) -> Dict[str, Any]:
        """获取文化统计"""
        total_beliefs = sum(len(c.beliefs) for c in self.propagation_model.agent_cultures.values())

        belief_distribution: Dict[str, List[float]] = defaultdict(list)
        for culture in self.propagation_model.agent_cultures.values():
            for btype, belief in culture.beliefs.items():
                belief_distribution[btype.name].append(belief.strength)

        avg_beliefs = {
            k: sum(v) / len(v) if v else 0
            for k, v in belief_distribution.items()
        }

        return {
            "total_agents": len(self.propagation_model.agent_cultures),
            "total_beliefs": total_beliefs,
            "avg_beliefs": avg_beliefs,
            "cultural_diversity": self.propagation_model.get_cultural_diversity()
        }
