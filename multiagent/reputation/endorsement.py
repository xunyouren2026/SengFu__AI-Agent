"""
背书机制模块

高信誉Agent可为其他Agent背书，建立信任传递网络
背书关系具有风险共担特性，背书者需对被背书者负责
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math


class EndorsementStatus(Enum):
    """背书状态"""
    PENDING = "pending"       # 待确认
    ACTIVE = "active"         # 有效
    REVOKED = "revoked"       # 已撤销
    EXPIRED = "expired"       # 已过期


class EndorsementType(Enum):
    """背书类型"""
    SKILL = "skill"           # 技能背书
    TRUST = "trust"           # 信任背书
    EXPERTISE = "expertise"   # 专业背书
    GENERAL = "general"       # 综合背书


@dataclass
class Endorsement:
    """背书记录"""
    endorsement_id: str
    endorser_id: str          # 背书者
    endorsee_id: str          # 被背书者
    endorsement_type: EndorsementType
    strength: float           # 背书强度 0-1
    message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    status: EndorsementStatus = EndorsementStatus.ACTIVE
    revoked_at: Optional[float] = None
    revoke_reason: Optional[str] = None


@dataclass
class EndorsementMetrics:
    """背书指标"""
    agent_id: str
    endorsements_given: int
    endorsements_received: int
    active_endorsements: int
    total_strength_received: float
    network_reach: int        # 网络影响力范围
    trust_score: float        # 基于背书的信任分


class EndorsementManager:
    """
    背书管理器

    实现信任传递机制:
    1. 高信誉Agent可为其他Agent背书
    2. 背书具有风险共担 - 被背书者作恶影响背书者
    3. 建立信任网络，支持信任传递计算
    4. 背书可撤销，但保留历史记录
    """

    # 背书配置
    MIN_ENDORSER_REPUTATION = 60.0  # 最低背书者信誉
    MAX_ENDORSEMENTS_PER_AGENT = 20  # 每个Agent最多背书数
    DEFAULT_ENDORSEMENT_DURATION = 180 * 24 * 3600  # 默认180天有效期
    ENDORSEMENT_DECAY_RATE = 0.01  # 每日衰减率

    def __init__(self):
        self._endorsements: Dict[str, Endorsement] = {}
        self._agent_endorsements_given: Dict[str, Set[str]] = {}
        self._agent_endorsements_received: Dict[str, Set[str]] = {}
        self._agent_reputations: Dict[str, float] = {}
        self._trust_network: Dict[str, Dict[str, float]] = {}  # 信任网络邻接表

    def can_endorse(self, endorser_id: str, endorsee_id: str) -> Tuple[bool, str]:
        """
        检查是否可以背书

        Returns:
            (是否可以, 原因)
        """
        # 不能自背书
        if endorser_id == endorsee_id:
            return False, "不能为自己背书"

        # 检查背书者信誉
        endorser_rep = self._agent_reputations.get(endorser_id, 0)
        if endorser_rep < self.MIN_ENDORSER_REPUTATION:
            return False, f"背书者信誉不足，需要至少 {self.MIN_ENDORSER_REPUTATION}"

        # 检查背书数量限制
        given = self._agent_endorsements_given.get(endorser_id, set())
        if len(given) >= self.MAX_ENDORSEMENTS_PER_AGENT:
            return False, f"已达到最大背书数量限制 {self.MAX_ENDORSEMENTS_PER_AGENT}"

        # 检查是否已存在有效背书
        existing = self._get_active_endorsement(endorser_id, endorsee_id)
        if existing:
            return False, "已存在有效背书"

        return True, "可以背书"

    def create_endorsement(
        self,
        endorsement_id: str,
        endorser_id: str,
        endorsee_id: str,
        endorsement_type: EndorsementType = EndorsementType.GENERAL,
        strength: float = 0.5,
        message: Optional[str] = None,
        duration_days: Optional[int] = None
    ) -> Optional[Endorsement]:
        """
        创建背书

        Args:
            endorsement_id: 背书ID
            endorser_id: 背书者ID
            endorsee_id: 被背书者ID
            endorsement_type: 背书类型
            strength: 背书强度 (0-1)
            message: 背书信息
            duration_days: 有效期天数

        Returns:
            Endorsement对象或None
        """
        can_endorse, reason = self.can_endorse(endorser_id, endorsee_id)
        if not can_endorse:
            return None

        # 限制强度范围
        strength = max(0.0, min(1.0, strength))

        # 计算过期时间
        duration = (duration_days * 24 * 3600) if duration_days else self.DEFAULT_ENDORSEMENT_DURATION
        expires_at = time.time() + duration

        endorsement = Endorsement(
            endorsement_id=endorsement_id,
            endorser_id=endorser_id,
            endorsee_id=endorsee_id,
            endorsement_type=endorsement_type,
            strength=strength,
            message=message,
            expires_at=expires_at
        )

        # 存储背书
        self._endorsements[endorsement_id] = endorsement

        # 更新索引
        if endorser_id not in self._agent_endorsements_given:
            self._agent_endorsements_given[endorser_id] = set()
        self._agent_endorsements_given[endorser_id].add(endorsement_id)

        if endorsee_id not in self._agent_endorsements_received:
            self._agent_endorsements_received[endorsee_id] = set()
        self._agent_endorsements_received[endorsee_id].add(endorsement_id)

        # 更新信任网络
        self._update_trust_network(endorser_id, endorsee_id, strength)

        return endorsement

    def _get_active_endorsement(
        self,
        endorser_id: str,
        endorsee_id: str
    ) -> Optional[Endorsement]:
        """获取两个Agent之间的有效背书"""
        given = self._agent_endorsements_given.get(endorser_id, set())

        for end_id in given:
            end = self._endorsements.get(end_id)
            if end and end.endorsee_id == endorsee_id and end.status == EndorsementStatus.ACTIVE:
                return end

        return None

    def _update_trust_network(
        self,
        from_id: str,
        to_id: str,
        weight: float
    ) -> None:
        """更新信任网络"""
        if from_id not in self._trust_network:
            self._trust_network[from_id] = {}

        self._trust_network[from_id][to_id] = weight

    def revoke_endorsement(
        self,
        endorsement_id: str,
        reason: str = ""
    ) -> bool:
        """撤销背书"""
        if endorsement_id not in self._endorsements:
            return False

        endorsement = self._endorsements[endorsement_id]

        if endorsement.status != EndorsementStatus.ACTIVE:
            return False

        endorsement.status = EndorsementStatus.REVOKED
        endorsement.revoked_at = time.time()
        endorsement.revoke_reason = reason

        # 更新信任网络
        endorser_id = endorsement.endorser_id
        endorsee_id = endorsement.endorsee_id

        if endorser_id in self._trust_network and endorsee_id in self._trust_network[endorser_id]:
            del self._trust_network[endorser_id][endorsee_id]

        return True

    def get_endorsement(self, endorsement_id: str) -> Optional[Endorsement]:
        """获取背书详情"""
        return self._endorsements.get(endorsement_id)

    def get_agent_endorsements_given(
        self,
        agent_id: str,
        status: Optional[EndorsementStatus] = None
    ) -> List[Endorsement]:
        """获取Agent给出的背书"""
        endorsement_ids = self._agent_endorsements_given.get(agent_id, set())
        endorsements = [self._endorsements[eid] for eid in endorsement_ids if eid in self._endorsements]

        if status:
            endorsements = [e for e in endorsements if e.status == status]

        return sorted(endorsements, key=lambda x: x.created_at, reverse=True)

    def get_agent_endorsements_received(
        self,
        agent_id: str,
        status: Optional[EndorsementStatus] = None
    ) -> List[Endorsement]:
        """获取Agent收到的背书"""
        endorsement_ids = self._agent_endorsements_received.get(agent_id, set())
        endorsements = [self._endorsements[eid] for eid in endorsement_ids if eid in self._endorsements]

        if status:
            endorsements = [e for e in endorsements if e.status == status]

        return sorted(endorsements, key=lambda x: x.created_at, reverse=True)

    def calculate_trust_score(
        self,
        agent_id: str,
        max_hops: int = 3
    ) -> float:
        """
        计算基于背书的信任分

        使用信任传递算法，考虑多跳信任关系
        """
        # 直接收到的背书
        received = self.get_agent_endorsements_received(agent_id, EndorsementStatus.ACTIVE)

        if not received:
            return 0.0

        # 计算直接信任
        direct_trust = 0.0
        total_weight = 0.0

        for end in received:
            endorser_rep = self._agent_reputations.get(end.endorser_id, 50)
            weight = end.strength * (endorser_rep / 100)
            direct_trust += weight
            total_weight += 1

        if total_weight > 0:
            direct_trust = direct_trust / total_weight * 100

        # 计算网络信任 (信任传递)
        network_trust = self._calculate_network_trust(agent_id, max_hops)

        # 综合信任分
        final_score = 0.7 * direct_trust + 0.3 * network_trust

        return min(100.0, final_score)

    def _calculate_network_trust(self, agent_id: str, max_hops: int) -> float:
        """计算网络信任分 (PageRank-like算法)"""
        if max_hops <= 0:
            return 0.0

        # 找到所有背书该Agent的节点
        received = self.get_agent_endorsements_received(agent_id, EndorsementStatus.ACTIVE)

        if not received:
            return 0.0

        network_score = 0.0

        for end in received:
            endorser_id = end.endorser_id
            endorser_rep = self._agent_reputations.get(endorser_id, 50)

            # 递归计算背书者的网络信任
            endorser_network = self._calculate_network_trust(endorser_id, max_hops - 1)

            # 综合背书者的信誉和网络影响力
            endorser_influence = 0.6 * endorser_rep + 0.4 * endorser_network

            network_score += end.strength * endorser_influence * 0.5  # 每跳衰减50%

        return min(100.0, network_score / len(received) if received else 0)

    def get_endorsement_metrics(self, agent_id: str) -> EndorsementMetrics:
        """获取背书指标"""
        given = self._agent_endorsements_given.get(agent_id, set())
        received = self._agent_endorsements_received.get(agent_id, set())

        active_given = [
            self._endorsements[eid] for eid in given
            if eid in self._endorsements and self._endorsements[eid].status == EndorsementStatus.ACTIVE
        ]
        active_received = [
            self._endorsements[eid] for eid in received
            if eid in self._endorsements and self._endorsements[eid].status == EndorsementStatus.ACTIVE
        ]

        total_strength = sum(e.strength for e in active_received)

        # 计算网络影响力范围
        network_reach = self._calculate_network_reach(agent_id)

        # 计算信任分
        trust_score = self.calculate_trust_score(agent_id)

        return EndorsementMetrics(
            agent_id=agent_id,
            endorsements_given=len(given),
            endorsements_received=len(received),
            active_endorsements=len(active_received),
            total_strength_received=round(total_strength, 2),
            network_reach=network_reach,
            trust_score=round(trust_score, 2)
        )

    def _calculate_network_reach(self, agent_id: str) -> int:
        """计算网络影响力范围 (BFS遍历)"""
        visited = set()
        queue = [(agent_id, 0)]
        max_depth = 3

        while queue:
            current_id, depth = queue.pop(0)

            if current_id in visited or depth >= max_depth:
                continue

            visited.add(current_id)

            # 找到当前节点背书的节点
            if current_id in self._trust_network:
                for neighbor_id in self._trust_network[current_id]:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, depth + 1))

        return len(visited) - 1  # 排除自己

    def find_trust_path(
        self,
        from_id: str,
        to_id: str,
        max_length: int = 5
    ) -> Optional[List[str]]:
        """
        查找信任路径

        Returns:
            路径上的Agent ID列表，或None
        """
        if from_id == to_id:
            return [from_id]

        # BFS查找最短路径
        visited = {from_id}
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_length:
                continue

            if current in self._trust_network:
                for neighbor in self._trust_network[current]:
                    if neighbor == to_id:
                        return path + [neighbor]

                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        return None

    def update_agent_reputation(self, agent_id: str, reputation: float) -> None:
        """更新Agent信誉 (用于背书计算)"""
        self._agent_reputations[agent_id] = reputation

    def get_mutual_endorsements(self, agent_id1: str, agent_id2: str) -> List[Endorsement]:
        """获取双向背书"""
        mutual = []

        # 1->2
        end1 = self._get_active_endorsement(agent_id1, agent_id2)
        if end1:
            mutual.append(end1)

        # 2->1
        end2 = self._get_active_endorsement(agent_id2, agent_id1)
        if end2:
            mutual.append(end2)

        return mutual

    def cleanup_expired(self) -> int:
        """清理过期背书，返回清理数量"""
        current_time = time.time()
        expired_count = 0

        for endorsement in self._endorsements.values():
            if (endorsement.status == EndorsementStatus.ACTIVE and
                endorsement.expires_at and
                endorsement.expires_at < current_time):

                endorsement.status = EndorsementStatus.EXPIRED
                expired_count += 1

        return expired_count

    def get_recommendations(self, agent_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        获取推荐背书对象

        基于共同连接推荐
        """
        # 获取Agent已背书的对象
        given = self._agent_endorsements_given.get(agent_id, set())
        endorsed = set()
        for eid in given:
            if eid in self._endorsements:
                endorsed.add(self._endorsements[eid].endorsee_id)

        # 获取Agent已被谁背书
        received = self._agent_endorsements_received.get(agent_id, set())
        endorsers = set()
        for eid in received:
            if eid in self._endorsements:
                endorsers.add(self._endorsements[eid].endorser_id)

        # 找到共同连接推荐的Agent
        recommendations = {}

        for endorser_id in endorsers:
            if endorser_id in self._agent_endorsements_given:
                for eid in self._agent_endorsements_given[endorser_id]:
                    if eid in self._endorsements:
                        candidate = self._endorsements[eid].endorsee_id
                        if candidate != agent_id and candidate not in endorsed:
                            recommendations[candidate] = recommendations.get(candidate, 0) + 1

        # 排序返回
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:limit]
