"""
女巫攻击防御模块

基于工作量证明(PoW)和身份绑定限制虚假评分
防止恶意用户创建大量虚假身份操纵评分系统
"""

from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time
import secrets
import struct


class IdentityType(Enum):
    """身份类型"""
    BASIC = "basic"           # 基础身份 (低信任)
    VERIFIED = "verified"     # 验证身份 (中等信任)
    PREMIUM = "premium"       # 高级身份 (高信任)
    ORGANIZATION = "org"      # 组织身份 (最高信任)


@dataclass
class Identity:
    """身份信息"""
    identity_id: str
    identity_type: IdentityType
    created_at: float
    reputation_score: float
    pow_nonce: Optional[int] = None
    pow_difficulty: int = 0
    linked_identities: Set[str] = field(default_factory=set)
    review_count: int = 0
    last_review_time: float = 0.0
    is_banned: bool = False


@dataclass
class PoWProof:
    """工作量证明"""
    identity_id: str
    nonce: int
    difficulty: int
    timestamp: float
    hash_value: str
    valid: bool = False


class SybilResistantManager:
    """
    女巫攻击防御管理器

    实现多种防御机制:
    1. 工作量证明 (PoW) - 创建身份需要计算成本
    2. 身份绑定 - 限制同一实体的多个身份
    3. 评分频率限制 - 防止刷分
    4. 信誉门槛 - 只有足够信誉才能评分
    """

    # 难度等级配置
    DIFFICULTY_LEVELS: Dict[IdentityType, int] = {
        IdentityType.BASIC: 4,        # 4个前导零
        IdentityType.VERIFIED: 6,     # 6个前导零
        IdentityType.PREMIUM: 8,      # 8个前导零
        IdentityType.ORGANIZATION: 10, # 10个前导零
    }

    # 评分频率限制 (秒)
    REVIEW_RATE_LIMIT: Dict[IdentityType, float] = {
        IdentityType.BASIC: 3600,     # 每小时1次
        IdentityType.VERIFIED: 600,   # 每10分钟1次
        IdentityType.PREMIUM: 60,     # 每分钟1次
        IdentityType.ORGANIZATION: 10, # 每10秒1次
    }

    # 评分信誉门槛
    REPUTATION_THRESHOLD: Dict[IdentityType, float] = {
        IdentityType.BASIC: 0.0,
        IdentityType.VERIFIED: 20.0,
        IdentityType.PREMIUM: 50.0,
        IdentityType.ORGANIZATION: 70.0,
    }

    def __init__(self):
        self._identities: Dict[str, Identity] = {}
        self._banned_identities: Set[str] = set()
        self._suspicious_patterns: Dict[str, List[float]] = {}  # 可疑行为记录
        self._detection_callbacks: List[Callable[[str, str], None]] = []

    def register_detection_callback(self, callback: Callable[[str, str], None]) -> None:
        """注册女巫攻击检测回调"""
        self._detection_callbacks.append(callback)

    def _generate_identity_id(self) -> str:
        """生成唯一身份ID"""
        return hashlib.sha256(
            f"{time.time()}{secrets.token_hex(16)}".encode()
        ).hexdigest()[:32]

    def create_identity_challenge(
        self,
        identity_type: IdentityType = IdentityType.BASIC
    ) -> Tuple[str, int]:
        """
        创建身份注册挑战

        Returns:
            (challenge_string, difficulty)
        """
        difficulty = self.DIFFICULTY_LEVELS.get(identity_type, 4)
        challenge = secrets.token_hex(16)
        return challenge, difficulty

    def verify_pow(self, challenge: str, nonce: int, difficulty: int) -> bool:
        """
        验证工作量证明

        检查hash(challenge + nonce)是否有足够多的前导零
        """
        data = f"{challenge}{nonce}".encode()
        hash_value = hashlib.sha256(data).hexdigest()
        prefix = "0" * difficulty
        return hash_value.startswith(prefix)

    def mine_identity(
        self,
        identity_type: IdentityType = IdentityType.BASIC,
        max_attempts: int = 10000000
    ) -> Optional[PoWProof]:
        """
        挖矿创建身份 (PoW)

        通过计算找到满足难度要求的nonce
        """
        challenge, difficulty = self.create_identity_challenge(identity_type)
        identity_id = self._generate_identity_id()
        challenge = f"{identity_id}:{challenge}"

        start_time = time.time()

        for nonce in range(max_attempts):
            if self.verify_pow(challenge, nonce, difficulty):
                data = f"{challenge}{nonce}".encode()
                hash_value = hashlib.sha256(data).hexdigest()

                proof = PoWProof(
                    identity_id=identity_id,
                    nonce=nonce,
                    difficulty=difficulty,
                    timestamp=start_time,
                    hash_value=hash_value,
                    valid=True
                )

                # 创建身份
                identity = Identity(
                    identity_id=identity_id,
                    identity_type=identity_type,
                    created_at=start_time,
                    reputation_score=0.0,
                    pow_nonce=nonce,
                    pow_difficulty=difficulty
                )
                self._identities[identity_id] = identity

                return proof

        return None

    def verify_identity(self, identity_id: str) -> bool:
        """验证身份有效性"""
        if identity_id in self._banned_identities:
            return False

        identity = self._identities.get(identity_id)
        if not identity:
            return False

        if identity.is_banned:
            return False

        return True

    def can_submit_review(self, identity_id: str) -> Tuple[bool, str]:
        """
        检查是否可以提交评分

        Returns:
            (是否可以评分, 原因)
        """
        if not self.verify_identity(identity_id):
            return False, "身份无效或已被封禁"

        identity = self._identities[identity_id]
        current_time = time.time()

        # 检查信誉门槛
        min_reputation = self.REPUTATION_THRESHOLD.get(identity.identity_type, 0.0)
        if identity.reputation_score < min_reputation:
            return False, f"信誉分不足，需要至少 {min_reputation}"

        # 检查频率限制
        rate_limit = self.REVIEW_RATE_LIMIT.get(identity.identity_type, 3600)
        time_since_last = current_time - identity.last_review_time

        if identity.review_count > 0 and time_since_last < rate_limit:
            wait_time = int(rate_limit - time_since_last)
            return False, f"评分过于频繁，请等待 {wait_time} 秒"

        return True, "可以评分"

    def record_review(self, identity_id: str) -> bool:
        """记录评分行为"""
        if identity_id not in self._identities:
            return False

        identity = self._identities[identity_id]
        identity.review_count += 1
        identity.last_review_time = time.time()

        # 检测可疑行为
        self._detect_suspicious_behavior(identity_id)

        return True

    def _detect_suspicious_behavior(self, identity_id: str) -> None:
        """检测可疑的女巫攻击行为"""
        identity = self._identities[identity_id]
        current_time = time.time()

        # 记录行为时间
        if identity_id not in self._suspicious_patterns:
            self._suspicious_patterns[identity_id] = []

        self._suspicious_patterns[identity_id].append(current_time)

        # 只保留最近1小时的行为记录
        cutoff = current_time - 3600
        self._suspicious_patterns[identity_id] = [
            t for t in self._suspicious_patterns[identity_id] if t > cutoff
        ]

        # 检测异常模式
        recent_actions = self._suspicious_patterns[identity_id]

        # 模式1: 短时间内大量行为
        if len(recent_actions) > 50:
            self._flag_suspicious(identity_id, "短时间内行为过多")
            return

        # 模式2: 行为间隔过于规律 (可能是自动化脚本)
        if len(recent_actions) >= 5:
            intervals = [
                recent_actions[i] - recent_actions[i-1]
                for i in range(1, len(recent_actions))
            ]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)

                # 如果方差很小，可能是自动化
                if variance < 1.0 and avg_interval < 5.0:
                    self._flag_suspicious(identity_id, "行为模式过于规律")

    def _flag_suspicious(self, identity_id: str, reason: str) -> None:
        """标记可疑身份"""
        for callback in self._detection_callbacks:
            callback(identity_id, reason)

    def link_identities(self, identity_id1: str, identity_id2: str) -> bool:
        """
        绑定两个身份 (表明属于同一实体)

        绑定后，两个身份共享评分限制
        """
        if identity_id1 not in self._identities or identity_id2 not in self._identities:
            return False

        id1 = self._identities[identity_id1]
        id2 = self._identities[identity_id2]

        # 双向绑定
        id1.linked_identities.add(identity_id2)
        id2.linked_identities.add(identity_id1)

        # 合并关联集合
        all_linked = id1.linked_identities.union(id2.linked_identities)
        for linked_id in all_linked:
            if linked_id in self._identities:
                self._identities[linked_id].linked_identities = all_linked

        return True

    def get_linked_review_count(self, identity_id: str) -> int:
        """获取身份及其关联身份的总评分数"""
        if identity_id not in self._identities:
            return 0

        identity = self._identities[identity_id]
        total = identity.review_count

        for linked_id in identity.linked_identities:
            if linked_id in self._identities:
                total += self._identities[linked_id].review_count

        return total

    def ban_identity(self, identity_id: str, reason: str = "") -> bool:
        """封禁身份"""
        if identity_id not in self._identities:
            return False

        identity = self._identities[identity_id]
        identity.is_banned = True
        self._banned_identities.add(identity_id)

        # 同时封禁关联身份
        for linked_id in identity.linked_identities:
            if linked_id in self._identities:
                self._identities[linked_id].is_banned = True
                self._banned_identities.add(linked_id)

        return True

    def upgrade_identity(
        self,
        identity_id: str,
        new_type: IdentityType
    ) -> Optional[PoWProof]:
        """升级身份类型 (需要更高难度的PoW)"""
        if identity_id not in self._identities:
            return None

        current_type = self._identities[identity_id].identity_type

        # 只能向上升级
        type_order = [
            IdentityType.BASIC,
            IdentityType.VERIFIED,
            IdentityType.PREMIUM,
            IdentityType.ORGANIZATION
        ]

        if type_order.index(new_type) <= type_order.index(current_type):
            return None

        # 重新挖矿
        proof = self.mine_identity(new_type)

        if proof and proof.valid:
            # 更新身份类型
            self._identities[identity_id].identity_type = new_type
            self._identities[identity_id].pow_nonce = proof.nonce
            self._identities[identity_id].pow_difficulty = proof.difficulty

        return proof

    def update_reputation(self, identity_id: str, delta: float) -> float:
        """更新身份信誉分"""
        if identity_id not in self._identities:
            return 0.0

        identity = self._identities[identity_id]
        identity.reputation_score = max(0.0, identity.reputation_score + delta)

        return identity.reputation_score

    def get_identity_info(self, identity_id: str) -> Optional[Dict]:
        """获取身份信息"""
        if identity_id not in self._identities:
            return None

        identity = self._identities[identity_id]

        return {
            "identity_id": identity.identity_id,
            "type": identity.identity_type.value,
            "reputation_score": round(identity.reputation_score, 2),
            "review_count": identity.review_count,
            "linked_count": len(identity.linked_identities),
            "pow_difficulty": identity.pow_difficulty,
            "created_at": identity.created_at,
            "is_banned": identity.is_banned
        }

    def get_system_stats(self) -> Dict:
        """获取系统统计"""
        total = len(self._identities)
        banned = len(self._banned_identities)

        type_counts = {}
        for identity in self._identities.values():
            t = identity.identity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_identities": total,
            "banned_identities": banned,
            "active_identities": total - banned,
            "type_distribution": type_counts,
            "average_reputation": round(
                sum(i.reputation_score for i in self._identities.values()) / total, 2
            ) if total > 0 else 0.0
        }
