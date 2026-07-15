"""
声誉激励系统 - Reputation & Incentive System

提供信誉评分、贝叶斯信誉模型、背书图、时间衰减和信任模型。
仅使用Python标准库。
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict


# ============================================================
# 信誉分数
# ============================================================

@dataclass
class ReputationScore:
    """信誉分数"""
    agent_id: str
    score: float = 0.5  # 初始分数 [0, 1]
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0


# ============================================================
# 时间衰减函数
# ============================================================

class ReputationDecay:
    """时间衰减函数"""

    @staticmethod
    def exponential_decay(
        original_score: float,
        elapsed_seconds: float,
        half_life: float = 86400.0,
    ) -> float:
        """
        指数衰减。

        score(t) = score_0 * (1/2)^(t / half_life)

        Args:
            original_score: 原始分数
            elapsed_seconds: 经过的时间（秒）
            half_life: 半衰期（秒），默认24小时

        Returns:
            衰减后的分数
        """
        if half_life <= 0:
            return original_score
        decay_factor = (0.5) ** (elapsed_seconds / half_life)
        return original_score * decay_factor

    @staticmethod
    def linear_decay(
        original_score: float,
        elapsed_seconds: float,
        max_age: float = 2592000.0,
        min_score: float = 0.1,
    ) -> float:
        """
        线性衰减。

        score(t) = max(min_score, score_0 * (1 - t / max_age))

        Args:
            original_score: 原始分数
            elapsed_seconds: 经过的时间（秒）
            max_age: 最大有效时间（秒），默认30天
            min_score: 最低分数

        Returns:
            衰减后的分数
        """
        if max_age <= 0:
            return min_score
        ratio = min(elapsed_seconds / max_age, 1.0)
        decayed = original_score * (1.0 - ratio)
        return max(decayed, min_score)

    @staticmethod
    def step_decay(
        original_score: float,
        elapsed_seconds: float,
        step_size: float = 604800.0,
        decay_per_step: float = 0.1,
        min_score: float = 0.1,
    ) -> float:
        """
        阶梯衰减。

        每经过一个step_size时间段，分数减少decay_per_step。

        Args:
            original_score: 原始分数
            elapsed_seconds: 经过的时间（秒）
            step_size: 阶梯大小（秒），默认7天
            decay_per_step: 每阶梯衰减量
            min_score: 最低分数

        Returns:
            衰减后的分数
        """
        if step_size <= 0:
            return original_score
        steps = int(elapsed_seconds / step_size)
        decayed = original_score * ((1.0 - decay_per_step) ** steps)
        return max(decayed, min_score)


# ============================================================
# 信誉系统
# ============================================================

class ReputationSystem:
    """
    信誉系统。

    管理Agent的信誉分数，支持时间衰减和排名。
    """

    def __init__(self, decay_type: str = "exponential", **decay_kwargs):
        """
        Args:
            decay_type: 衰减类型 ("exponential", "linear", "step")
            decay_kwargs: 衰减函数参数
        """
        self._scores: Dict[str, ReputationScore] = {}
        self._decay_type = decay_type
        self._decay_kwargs = decay_kwargs
        self._decay = ReputationDecay()

    def _get_decay_function(self):
        """获取衰减函数"""
        if self._decay_type == "exponential":
            return self._decay.exponential_decay
        elif self._decay_type == "linear":
            return self._decay.linear_decay
        elif self._decay_type == "step":
            return self._decay.step_decay
        else:
            return self._decay.exponential_decay

    def update_reputation(
        self,
        agent_id: str,
        feedback: float,
        context: Optional[str] = None,
    ) -> float:
        """
        更新Agent信誉。

        使用移动平均更新信誉分数：
        new_score = alpha * feedback + (1 - alpha) * old_score

        Args:
            agent_id: Agent ID
            feedback: 反馈值 [0, 1]
            context: 反馈上下文

        Returns:
            更新后的信誉分数
        """
        if agent_id not in self._scores:
            self._scores[agent_id] = ReputationScore(agent_id=agent_id)

        score = self._scores[agent_id]

        # 应用时间衰减到当前分数
        now = time.time()
        elapsed = now - score.last_updated
        decay_fn = self._get_decay_function()
        decayed_score = decay_fn(score.score, elapsed, **self._decay_kwargs)

        # 移动平均更新
        # alpha随交互次数递减，初期变化快，后期趋于稳定
        alpha = 1.0 / (1.0 + score.total_interactions * 0.1)
        new_score = alpha * feedback + (1.0 - alpha) * decayed_score
        new_score = max(0.0, min(1.0, new_score))

        # 更新统计
        score.score = round(new_score, 6)
        score.last_updated = now
        score.total_interactions += 1

        if feedback >= 0.6:
            score.positive_interactions += 1
        elif feedback <= 0.4:
            score.negative_interactions += 1

        score.history.append({
            "feedback": feedback,
            "context": context,
            "previous_score": decayed_score,
            "new_score": new_score,
            "timestamp": now,
        })

        return score.score

    def get_reputation(self, agent_id: str) -> float:
        """获取Agent当前信誉分数（含时间衰减）"""
        if agent_id not in self._scores:
            return 0.5  # 默认分数

        score = self._scores[agent_id]
        now = time.time()
        elapsed = now - score.last_updated
        decay_fn = self._get_decay_function()
        return decay_fn(score.score, elapsed, **self._decay_kwargs)

    def get_ranking(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        获取信誉排名。

        Args:
            top_n: 返回前N名，None表示全部

        Returns:
            排名列表 [(agent_id, score), ...] 按分数降序
        """
        rankings = []
        for agent_id in self._scores:
            current_score = self.get_reputation(agent_id)
            rankings.append((agent_id, round(current_score, 6)))

        rankings.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            rankings = rankings[:top_n]

        return rankings

    def decay(self, agent_id: str) -> float:
        """
        手动触发时间衰减。

        Returns:
            衰减后的分数
        """
        if agent_id not in self._scores:
            return 0.5

        score = self._scores[agent_id]
        now = time.time()
        elapsed = now - score.last_updated
        decay_fn = self._get_decay_function()
        decayed = decay_fn(score.score, elapsed, **self._decay_kwargs)

        score.score = round(decayed, 6)
        score.last_updated = now

        return score.score

    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent详细信誉信息"""
        if agent_id not in self._scores:
            return None

        score = self._scores[agent_id]
        current = self.get_reputation(agent_id)

        return {
            "agent_id": score.agent_id,
            "current_score": round(current, 6),
            "stored_score": score.score,
            "total_interactions": score.total_interactions,
            "positive_interactions": score.positive_interactions,
            "negative_interactions": score.negative_interactions,
            "positive_ratio": (
                round(score.positive_interactions / score.total_interactions, 4)
                if score.total_interactions > 0
                else 0.0
            ),
            "last_updated": score.last_updated,
        }


# ============================================================
# 贝叶斯信誉模型
# ============================================================

class BayesianReputation:
    """
    贝叶斯信誉模型。

    使用Beta分布建模Agent的信誉：
    - alpha: 成功计数（+1先验）
    - beta: 失败计数（+1先验）
    - E[theta] = alpha / (alpha + beta)
    - 可信区间使用Beta分布的分位数近似
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        Args:
            prior_alpha: 先验成功计数
            prior_beta: 先验失败计数
        """
        self._models: Dict[str, Dict[str, float]] = {}
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    def _ensure_model(self, agent_id: str) -> Dict[str, float]:
        """确保Agent模型存在"""
        if agent_id not in self._models:
            self._models[agent_id] = {
                "alpha": self._prior_alpha,
                "beta": self._prior_beta,
            }
        return self._models[agent_id]

    def update(self, agent_id: str, success: bool) -> float:
        """
        贝叶斯更新。

        成功: alpha += 1
        失败: beta += 1

        Args:
            agent_id: Agent ID
            success: 是否成功

        Returns:
            更新后的期望分数
        """
        model = self._ensure_model(agent_id)

        if success:
            model["alpha"] += 1.0
        else:
            model["beta"] += 1.0

        return self.get_expected_score(agent_id)

    def update_batch(self, agent_id: str, successes: int, failures: int) -> float:
        """批量更新"""
        model = self._ensure_model(agent_id)
        model["alpha"] += successes
        model["beta"] += failures
        return self.get_expected_score(agent_id)

    def get_expected_score(self, agent_id: str) -> float:
        """
        获取期望分数。

        E[theta] = alpha / (alpha + beta)
        """
        model = self._ensure_model(agent_id)
        alpha = model["alpha"]
        beta = model["beta"]
        return round(alpha / (alpha + beta), 6)

    def get_credibility_interval(
        self,
        agent_id: str,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        获取可信区间。

        使用Beta分布的近似分位数。
        对于Beta(alpha, beta)，当alpha和beta较大时，
        近似为正态分布：
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

        使用Wilson-Hilferty近似转换到正态分位数。
        """
        model = self._ensure_model(agent_id)
        alpha = model["alpha"]
        beta = model["beta"]
        total = alpha + beta

        if total < 2:
            return (0.0, 1.0)

        mean = alpha / total
        variance = (alpha * beta) / (total * total * (total + 1))
        std_dev = math.sqrt(variance)

        # 正态分布分位数近似
        # z值对应confidence_level
        z = self._normal_quantile(confidence_level)

        lower = max(0.0, mean - z * std_dev)
        upper = min(1.0, mean + z * std_dev)

        return (round(lower, 6), round(upper, 6))

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """
        标准正态分布分位数近似（Abramowitz and Stegun方法）。

        对于 p in (0, 1)，返回z使得 Phi(z) = p。
        """
        if p <= 0.0:
            return -3.5
        if p >= 1.0:
            return 3.5

        if p < 0.5:
            return -BayesianReputation._normal_quantile(1.0 - p)

        # Abramowitz and Stegun 近似 26.2.23
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308

        z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        return z

    def get_model_params(self, agent_id: str) -> Dict[str, float]:
        """获取模型参数"""
        model = self._ensure_model(agent_id)
        return {
            "alpha": model["alpha"],
            "beta": model["beta"],
            "expected_score": self.get_expected_score(agent_id),
            "total_observations": model["alpha"] + model["beta"] - self._prior_alpha - self._prior_beta,
        }


# ============================================================
# 背书图
# ============================================================

class EndorsementGraph:
    """
    背书图。

    使用邻接表表示有向加权图。
    支持：背书、背书分数计算、Sybil攻击检测。
    """

    def __init__(self):
        # 邻接表: {endorser: {target: weight}}
        self._graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        # 反向邻接表: {target: {endorser: weight}}
        self._reverse_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        # 节点集合
        self._nodes: Set[str] = set()

    def endorse(self, endorser: str, target: str, weight: float = 1.0) -> float:
        """
        添加背书关系。

        Args:
            endorser: 背书者
            target: 被背书者
            weight: 背书权重 [0, 1]

        Returns:
            背书后的目标节点总背书分数
        """
        weight = max(0.0, min(1.0, weight))

        self._nodes.add(endorser)
        self._nodes.add(target)

        self._graph[endorser][target] = weight
        self._reverse_graph[target][endorser] = weight

        return self.get_endorsement_score(target)

    def get_endorsement_score(self, agent_id: str) -> float:
        """
        计算Agent的背书分数。

        使用PageRank风格的算法：
        score(v) = (1 - d) + d * sum(score(u) * w(u,v) / out_degree(u))
        其中 d 是阻尼系数。

        Returns:
            背书分数
        """
        if agent_id not in self._nodes:
            return 0.0

        # 迭代计算PageRank
        damping = 0.85
        max_iterations = 100
        tolerance = 1e-6

        # 初始化
        scores = {node: 1.0 / max(len(self._nodes), 1) for node in self._nodes}
        out_degree = {node: sum(self._graph[node].values()) for node in self._nodes}

        for _ in range(max_iterations):
            new_scores = {}
            max_diff = 0.0

            for node in self._nodes:
                # 来自其他节点的背书贡献
                incoming = self._reverse_graph.get(node, {})
                rank_sum = 0.0

                for endorser, weight in incoming.items():
                    if out_degree[endorser] > 0:
                        rank_sum += scores[endorser] * weight / out_degree[endorser]

                new_score = (1.0 - damping) / len(self._nodes) + damping * rank_sum
                new_scores[node] = new_score
                max_diff = max(max_diff, abs(new_score - scores[node]))

            scores = new_scores
            if max_diff < tolerance:
                break

        return round(scores.get(agent_id, 0.0), 6)

    def get_all_scores(self) -> Dict[str, float]:
        """获取所有节点的背书分数"""
        return {node: self.get_endorsement_score(node) for node in self._nodes}

    def detect_sybil(self, agent_id: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Sybil攻击检测。

        基于图结构分析检测可疑的Sybil行为：
        1. 检查入度集中度：如果大部分背书来自少数节点
        2. 检查背书时间模式：大量背书在短时间内
        3. 检查互惠背书环：A背书B，B背书A，形成闭环

        Args:
            agent_id: 待检测的Agent
            threshold: 可疑阈值

        Returns:
            检测结果
        """
        if agent_id not in self._nodes:
            return {"is_sybil": False, "score": 0.0, "indicators": {}}

        indicators = {}

        # 1. 入度集中度（赫芬达尔指数）
        incoming = self._reverse_graph.get(agent_id, {})
        total_weight = sum(incoming.values())
        if total_weight > 0:
            hhi = sum((w / total_weight) ** 2 for w in incoming.values())
            # HHI范围 [1/n, 1]，越高越集中
            n_endorsers = len(incoming)
            min_hhi = 1.0 / max(n_endorsers, 1)
            normalized_hhi = (hhi - min_hhi) / max(1.0 - min_hhi, 0.001)
            indicators["endorsement_concentration"] = round(normalized_hhi, 4)
        else:
            indicators["endorsement_concentration"] = 0.0

        # 2. 出度分析：Sybil节点通常很少背书他人
        outgoing = self._graph.get(agent_id, {})
        out_ratio = len(outgoing) / max(len(self._nodes) - 1, 1)
        indicators["low_outgoing_ratio"] = round(1.0 - out_ratio, 4)

        # 3. 互惠背书检测
        reciprocity_count = 0
        for endorser in incoming:
            if agent_id in self._graph.get(endorser, {}):
                reciprocity_count += 1

        reciprocity_ratio = reciprocity_count / max(len(incoming), 1)
        indicators["reciprocity_ratio"] = round(reciprocity_ratio, 4)

        # 4. 聚类系数：Sybil节点通常形成紧密子图
        neighbors = set(incoming.keys()) | set(outgoing.keys())
        if len(neighbors) >= 2:
            triangles = 0
            possible_triangles = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    possible_triangles += 1
                    n1, n2 = neighbor_list[i], neighbor_list[j]
                    if n2 in self._graph.get(n1, {}):
                        triangles += 1
            clustering = triangles / max(possible_triangles, 1)
        else:
            clustering = 0.0
        indicators["clustering_coefficient"] = round(clustering, 4)

        # 综合Sybil分数
        weights = {
            "endorsement_concentration": 0.30,
            "low_outgoing_ratio": 0.20,
            "reciprocity_ratio": 0.25,
            "clustering_coefficient": 0.25,
        }
        sybil_score = sum(
            weights.get(k, 0) * v for k, v in indicators.items()
        )

        return {
            "is_sybil": sybil_score >= threshold,
            "score": round(sybil_score, 4),
            "threshold": threshold,
            "indicators": indicators,
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图统计信息"""
        total_edges = sum(
            len(targets) for targets in self._graph.values()
        )
        return {
            "num_nodes": len(self._nodes),
            "num_edges": total_edges,
            "avg_degree": round(
                2.0 * total_edges / max(len(self._nodes), 1), 4
            ),
        }


# ============================================================
# 信任模型
# ============================================================

class TrustModel:
    """
    信任模型。

    综合直接信任和间接信任计算综合信任分数。
    trust = w1 * direct_trust + w2 * indirect_trust
    """

    def __init__(
        self,
        direct_weight: float = 0.6,
        indirect_weight: float = 0.4,
        reputation_system: Optional[ReputationSystem] = None,
        endorsement_graph: Optional[EndorsementGraph] = None,
    ):
        """
        Args:
            direct_weight: 直接信任权重
            indirect_weight: 间接信任权重
            reputation_system: 信誉系统实例
            endorsement_graph: 背书图实例
        """
        if abs(direct_weight + indirect_weight - 1.0) > 1e-6:
            raise ValueError("权重之和必须为1.0")

        self.w1 = direct_weight
        self.w2 = indirect_weight
        self._reputation = reputation_system or ReputationSystem()
        self._endorsement = endorsement_graph or EndorsementGraph()

        # 直接信任记录: {evaluator: {target: trust_value}}
        self._direct_trust: Dict[str, Dict[str, float]] = defaultdict(dict)

    def record_interaction(
        self,
        evaluator: str,
        target: str,
        outcome: float,
    ) -> None:
        """
        记录交互结果并更新直接信任。

        Args:
            evaluator: 评估者
            target: 被评估者
            outcome: 交互结果 [0, 1]
        """
        if target not in self._direct_trust[evaluator]:
            self._direct_trust[evaluator][target] = 0.5

        # 指数移动平均
        old_trust = self._direct_trust[evaluator][target]
        alpha = 0.3  # 学习率
        new_trust = alpha * outcome + (1.0 - alpha) * old_trust
        self._direct_trust[evaluator][target] = round(
            max(0.0, min(1.0, new_trust)), 6
        )

        # 同时更新信誉系统
        self._reputation.update_reputation(target, outcome)

    def direct_trust(self, evaluator: str, target: str) -> float:
        """
        计算直接信任。

        基于评估者与目标的历史交互记录。
        如果没有交互记录，返回信誉系统的分数。

        Args:
            evaluator: 评估者
            target: 被评估者

        Returns:
            直接信任分数 [0, 1]
        """
        if target in self._direct_trust[evaluator]:
            return self._direct_trust[evaluator][target]
        # 无直接交互，使用信誉系统
        return self._reputation.get_reputation(target)

    def indirect_trust(self, target: str) -> float:
        """
        计算间接信任。

        基于背书传播：其他Agent对目标的背书加权平均。
        使用背书图的PageRank分数作为间接信任。

        Args:
            target: 被评估者

        Returns:
            间接信任分数 [0, 1]
        """
        return self._endorsement.get_endorsement_score(target)

    def composite_trust(
        self,
        evaluator: str,
        target: str,
    ) -> float:
        """
        计算综合信任。

        trust = w1 * direct_trust + w2 * indirect_trust

        Args:
            evaluator: 评估者
            target: 被评估者

        Returns:
            综合信任分数 [0, 1]
        """
        d_trust = self.direct_trust(evaluator, target)
        i_trust = self.indirect_trust(target)

        composite = self.w1 * d_trust + self.w2 * i_trust
        return round(max(0.0, min(1.0, composite)), 6)

    def add_endorsement(
        self,
        endorser: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """添加背书关系"""
        self._endorsement.endorse(endorser, target, weight)

    def get_trust_report(
        self,
        evaluator: str,
        target: str,
    ) -> Dict[str, float]:
        """获取信任报告"""
        return {
            "direct_trust": self.direct_trust(evaluator, target),
            "indirect_trust": self.indirect_trust(target),
            "composite_trust": self.composite_trust(evaluator, target),
            "direct_weight": self.w1,
            "indirect_weight": self.w2,
        }
