"""
代币经济与激励系统 - Token Economy & Incentive System

提供代币经济、贡献追踪、质押管理和排行榜功能。
仅使用Python标准库。
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
from itertools import permutations


# ============================================================
# 交易记录
# ============================================================

@dataclass
class Transaction:
    """交易记录"""
    id: str = ""
    from_id: str = ""
    to_id: str = ""
    amount: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    tx_type: str = "transfer"  # transfer / reward / penalize / stake / unstake / slash


# ============================================================
# 代币经济系统
# ============================================================

class TokenEconomy:
    """
    代币经济系统。

    管理Agent的代币余额、转账、奖励和惩罚。
    """

    def __init__(self, initial_supply: float = 1000000.0):
        self._balances: Dict[str, float] = defaultdict(float)
        self._transactions: List[Transaction] = []
        self._total_supply = initial_supply
        self._circulating = 0.0
        self._tx_counter = 0

    def _next_tx_id(self) -> str:
        """生成交易ID"""
        self._tx_counter += 1
        return f"tx_{self._tx_counter}_{int(time.time() * 1000)}"

    def _record_transaction(
        self,
        from_id: str,
        to_id: str,
        amount: float,
        reason: str,
        tx_type: str,
    ) -> Transaction:
        """记录交易"""
        tx = Transaction(
            id=self._next_tx_id(),
            from_id=from_id,
            to_id=to_id,
            amount=amount,
            reason=reason,
            tx_type=tx_type,
        )
        self._transactions.append(tx)
        return tx

    def balance(self, agent_id: str) -> float:
        """查询Agent余额"""
        return round(self._balances[agent_id], 6)

    def transfer(self, from_id: str, to_id: str, amount: float) -> bool:
        """
        转账。

        从from_id转amount到to_id。
        需要from_id有足够余额。

        Returns:
            是否转账成功
        """
        if amount <= 0:
            return False

        if self._balances[from_id] < amount:
            return False

        self._balances[from_id] -= amount
        self._balances[to_id] += amount

        self._record_transaction(from_id, to_id, amount, "transfer", "transfer")
        return True

    def reward(self, agent_id: str, amount: float, reason: str = "") -> float:
        """
        奖励Agent代币。

        从系统储备中发放。

        Args:
            agent_id: Agent ID
            amount: 奖励金额
            reason: 奖励原因

        Returns:
            奖励后的余额
        """
        if amount <= 0:
            return self.balance(agent_id)

        self._balances[agent_id] += amount
        self._circulating += amount

        self._record_transaction("system", agent_id, amount, reason, "reward")
        return self.balance(agent_id)

    def penalize(self, agent_id: str, amount: float, reason: str = "") -> float:
        """
        惩罚Agent，扣除代币。

        扣除的代币返回系统储备。

        Args:
            agent_id: Agent ID
            amount: 惩罚金额
            reason: 惩罚原因

        Returns:
            惩罚后的余额
        """
        if amount <= 0:
            return self.balance(agent_id)

        actual_penalty = min(amount, self._balances[agent_id])
        self._balances[agent_id] -= actual_penalty
        self._circulating -= actual_penalty

        self._record_transaction(agent_id, "system", actual_penalty, reason, "penalize")
        return self.balance(agent_id)

    def get_transaction_history(
        self,
        agent_id: Optional[str] = None,
        tx_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        获取交易历史。

        Args:
            agent_id: 筛选特定Agent的交易
            tx_type: 筛选交易类型
            limit: 返回数量限制

        Returns:
            交易记录列表
        """
        result = []
        for tx in reversed(self._transactions):
            if agent_id and tx.from_id != agent_id and tx.to_id != agent_id:
                continue
            if tx_type and tx.tx_type != tx_type:
                continue
            result.append({
                "id": tx.id,
                "from": tx.from_id,
                "to": tx.to_id,
                "amount": tx.amount,
                "reason": tx.reason,
                "type": tx.tx_type,
                "timestamp": tx.timestamp,
            })
            if len(result) >= limit:
                break
        return result

    def get_economy_stats(self) -> Dict[str, Any]:
        """获取经济统计"""
        return {
            "total_supply": self._total_supply,
            "circulating": round(self._circulating, 6),
            "reserved": round(self._total_supply - self._circulating, 6),
            "total_accounts": len(self._balances),
            "total_transactions": len(self._transactions),
        }


# ============================================================
# 贡献追踪器
# ============================================================

@dataclass
class Contribution:
    """贡献记录"""
    agent_id: str
    task_id: str
    value: float
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0


class ContributionTracker:
    """
    贡献追踪器。

    记录和评估Agent的贡献，支持Shapley值分配奖励。
    """

    def __init__(self):
        self._contributions: Dict[str, List[Contribution]] = defaultdict(list)
        self._task_agents: Dict[str, Set[str]] = defaultdict(set)

    def record_contribution(
        self,
        agent_id: str,
        task_id: str,
        value: float,
        quality_score: float = 1.0,
    ) -> float:
        """
        记录贡献。

        Args:
            agent_id: Agent ID
            task_id: 任务ID
            value: 贡献值
            quality_score: 质量评分 [0, 1]

        Returns:
            加权贡献分数
        """
        contribution = Contribution(
            agent_id=agent_id,
            task_id=task_id,
            value=value,
            quality_score=quality_score,
        )
        self._contributions[agent_id].append(contribution)
        self._task_agents[task_id].add(agent_id)

        return value * quality_score

    def get_contribution_score(self, agent_id: str) -> Dict[str, float]:
        """
        获取Agent的贡献分数。

        Returns:
            {"total": 总分, "average": 平均分, "count": 贡献次数}
        """
        contributions = self._contributions.get(agent_id, [])
        if not contributions:
            return {"total": 0.0, "average": 0.0, "count": 0}

        total = sum(c.value * c.quality_score for c in contributions)
        average = total / len(contributions)

        return {
            "total": round(total, 6),
            "average": round(average, 6),
            "count": len(contributions),
        }

    def shapley_allocation(
        self,
        task_id: str,
        total_reward: float,
    ) -> Dict[str, float]:
        """
        Shapley值分配奖励。

        Shapley值公式：
        phi_i(v) = sum_{S subset N\\{i}} (|S|! * (|N|-|S|-1)! / |N|!) * (v(S U {i}) - v(S))

        简化实现：基于贡献值的边际贡献计算。

        Args:
            task_id: 任务ID
            total_reward: 待分配的总奖励

        Returns:
            各Agent的Shapley值分配
        """
        agents = list(self._task_agents.get(task_id, set()))
        if not agents:
            return {}

        n = len(agents)

        # 计算每个Agent的贡献值
        agent_values: Dict[str, float] = {}
        for agent_id in agents:
            contributions = [
                c for c in self._contributions.get(agent_id, [])
                if c.task_id == task_id
            ]
            agent_values[agent_id] = sum(c.value * c.quality_score for c in contributions)

        # 计算联盟价值函数 v(S)
        def coalition_value(coalition: Set[str]) -> float:
            if not coalition:
                return 0.0
            # 联盟价值 = 成员贡献之和 + 协同bonus
            base = sum(agent_values.get(a, 0.0) for a in coalition)
            # 协同效应：成员越多，bonus越大（对数增长）
            synergy = math.log(1.0 + len(coalition)) * 0.1 * base
            return base + synergy

        # 计算Shapley值
        shapley_values: Dict[str, float] = {}
        all_agents = set(agents)

        for agent in agents:
            others = all_agents - {agent}
            shapley = 0.0

            # 遍历所有不包含agent的子集
            for size in range(len(others) + 1):
                for subset in self._subsets(list(others), size):
                    s = set(subset)
                    # 边际贡献
                    marginal = coalition_value(s | {agent}) - coalition_value(s)
                    # 权重: |S|! * (n - |S| - 1)! / n!
                    weight = (
                        math.factorial(len(s))
                        * math.factorial(n - len(s) - 1)
                        / math.factorial(n)
                    )
                    shapley += weight * marginal

            shapley_values[agent] = shapley

        # 归一化到total_reward
        total_shapley = sum(shapley_values.values())
        if total_shapley > 0:
            allocation = {
                agent: round(sv / total_shapley * total_reward, 6)
                for agent, sv in shapley_values.items()
            }
        else:
            # 均分
            share = total_reward / n
            allocation = {agent: round(share, 6) for agent in agents}

        return allocation

    @staticmethod
    def _subsets(lst: List[str], size: int) -> List[List[str]]:
        """生成列表的所有指定大小的子集"""
        if size == 0:
            return [[]]
        if size > len(lst):
            return []
        if size == len(lst):
            return [list(lst)]

        result = []
        # 包含第一个元素
        for sub in ContributionTracker._subsets(lst[1:], size - 1):
            result.append([lst[0]] + sub)
        # 不包含第一个元素
        result.extend(ContributionTracker._subsets(lst[1:], size))

        return result

    def get_task_summary(self, task_id: str) -> Dict[str, Any]:
        """获取任务贡献摘要"""
        agents = list(self._task_agents.get(task_id, set()))
        contributions = []
        for agent_id in agents:
            agent_contribs = [
                c for c in self._contributions.get(agent_id, [])
                if c.task_id == task_id
            ]
            contributions.extend(agent_contribs)

        total_value = sum(c.value * c.quality_score for c in contributions)

        return {
            "task_id": task_id,
            "num_contributors": len(agents),
            "total_contributions": len(contributions),
            "total_value": round(total_value, 6),
            "contributors": agents,
        }


# ============================================================
# 质押管理器
# ============================================================

@dataclass
class StakeInfo:
    """质押信息"""
    agent_id: str
    staked_amount: float = 0.0
    pending_unstake: float = 0.0
    unstake_time: float = 0.0
    slash_count: int = 0
    stake_history: List[Dict[str, Any]] = field(default_factory=list)


class StakeManager:
    """
    质押管理器。

    管理Agent的质押、解除质押和惩罚削减。
    """

    def __init__(
        self,
        min_stake: float = 10.0,
        unstake_delay: float = 86400.0,
        slash_rate: float = 0.1,
    ):
        """
        Args:
            min_stake: 最小质押量
            unstake_delay: 解除质押延迟（秒）
            slash_rate: 惩罚削减比例
        """
        self._stakes: Dict[str, StakeInfo] = {}
        self._min_stake = min_stake
        self._unstake_delay = unstake_delay
        self._slash_rate = slash_rate
        self._total_staked = 0.0

    def _ensure_stake(self, agent_id: str) -> StakeInfo:
        """确保质押信息存在"""
        if agent_id not in self._stakes:
            self._stakes[agent_id] = StakeInfo(agent_id=agent_id)
        return self._stakes[agent_id]

    def stake(self, agent_id: str, amount: float) -> float:
        """
        质押代币。

        Args:
            agent_id: Agent ID
            amount: 质押金额

        Returns:
            质押后的总质押量
        """
        if amount < self._min_stake:
            raise ValueError(
                f"最小质押量为 {self._min_stake}，当前 {amount}"
            )

        info = self._ensure_stake(agent_id)
        info.staked_amount += amount
        self._total_staked += amount

        info.stake_history.append({
            "action": "stake",
            "amount": amount,
            "timestamp": time.time(),
        })

        return info.staked_amount

    def unstake(self, agent_id: str, amount: float) -> Dict[str, Any]:
        """
        申请解除质押。

        有延迟生效期，在延迟期内可以取消。

        Args:
            agent_id: Agent ID
            amount: 解除金额

        Returns:
            操作结果
        """
        info = self._ensure_stake(agent_id)

        if amount > info.staked_amount:
            return {
                "success": False,
                "reason": "insufficient_stake",
                "available": info.staked_amount,
            }

        info.pending_unstake = amount
        info.unstake_time = time.time() + self._unstake_delay

        info.stake_history.append({
            "action": "unstake_request",
            "amount": amount,
            "timestamp": time.time(),
            "available_at": info.unstake_time,
        })

        return {
            "success": True,
            "pending_amount": amount,
            "available_at": info.unstake_time,
        }

    def process_pending_unstakes(self) -> List[Dict[str, Any]]:
        """处理待解除的质押"""
        now = time.time()
        processed = []

        for agent_id, info in self._stakes.items():
            if info.pending_unstake > 0 and now >= info.unstake_time:
                amount = info.pending_unstake
                info.staked_amount -= amount
                self._total_staked -= amount

                info.stake_history.append({
                    "action": "unstake_complete",
                    "amount": amount,
                    "timestamp": now,
                })

                processed.append({
                    "agent_id": agent_id,
                    "amount": amount,
                    "remaining_stake": info.staked_amount,
                })

                info.pending_unstake = 0.0
                info.unstake_time = 0.0

        return processed

    def slash(self, agent_id: str, amount: Optional[float] = None) -> float:
        """
        惩罚削减质押。

        如果未指定amount，则按slash_rate比例削减。

        Args:
            agent_id: Agent ID
            amount: 削减金额（可选）

        Returns:
            实际削减金额
        """
        info = self._ensure_stake(agent_id)

        if amount is None:
            amount = info.staked_amount * self._slash_rate

        actual_slash = min(amount, info.staked_amount)
        info.staked_amount -= actual_slash
        self._total_staked -= actual_slash
        info.slash_count += 1

        info.stake_history.append({
            "action": "slash",
            "amount": actual_slash,
            "timestamp": time.time(),
            "slash_count": info.slash_count,
        })

        return actual_slash

    def get_stake_info(self, agent_id: str) -> Dict[str, Any]:
        """获取质押信息"""
        if agent_id not in self._stakes:
            return {
                "agent_id": agent_id,
                "staked_amount": 0.0,
                "pending_unstake": 0.0,
                "slash_count": 0,
            }

        info = self._stakes[agent_id]
        return {
            "agent_id": info.agent_id,
            "staked_amount": round(info.staked_amount, 6),
            "pending_unstake": round(info.pending_unstake, 6),
            "unstake_time": info.unstake_time,
            "slash_count": info.slash_count,
            "history_count": len(info.stake_history),
        }

    def get_total_staked(self) -> float:
        """获取总质押量"""
        return round(self._total_staked, 6)


# ============================================================
# 排行榜
# ============================================================

class Leaderboard:
    """
    排行榜。

    支持时间加权排名，近期表现权重更高。
    """

    def __init__(self, time_decay_rate: float = 0.01):
        """
        Args:
            time_decay_rate: 时间衰减率，越大衰减越快
        """
        self._scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._time_decay_rate = time_decay_rate

    def update_score(self, agent_id: str, score: float) -> None:
        """
        更新Agent分数。

        记录带时间戳的分数。

        Args:
            agent_id: Agent ID
            score: 分数
        """
        self._scores[agent_id].append((score, time.time()))

    def _time_weighted_score(self, score_history: List[Tuple[float, float]]) -> float:
        """
        计算时间加权分数。

        使用指数加权移动平均（EWMA）：
        weighted_score = sum(score_i * w_i) / sum(w_i)
        其中 w_i = exp(-decay_rate * age_i)
        """
        if not score_history:
            return 0.0

        now = time.time()
        weighted_sum = 0.0
        weight_sum = 0.0

        for score, timestamp in score_history:
            age = now - timestamp
            weight = math.exp(-self._time_decay_rate * age)
            weighted_sum += score * weight
            weight_sum += weight

        if weight_sum == 0:
            return 0.0

        return weighted_sum / weight_sum

    def get_ranking(self) -> List[Tuple[str, float]]:
        """
        获取完整排名。

        Returns:
            [(agent_id, time_weighted_score), ...] 按分数降序
        """
        rankings = []
        for agent_id, history in self._scores.items():
            tw_score = self._time_weighted_score(history)
            rankings.append((agent_id, round(tw_score, 6)))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_top_n(self, n: int) -> List[Tuple[str, float]]:
        """
        获取前N名。

        Args:
            n: 返回数量

        Returns:
            前N名列表
        """
        return self.get_ranking()[:n]

    def get_agent_rank(self, agent_id: str) -> Optional[int]:
        """获取Agent的排名（从1开始），未找到返回None"""
        ranking = self.get_ranking()
        for i, (aid, _) in enumerate(ranking):
            if aid == agent_id:
                return i + 1
        return None

    def get_leaderboard_summary(self, top_n: int = 10) -> Dict[str, Any]:
        """获取排行榜摘要"""
        top = self.get_top_n(top_n)
        return {
            "total_participants": len(self._scores),
            "top_n": [
                {"rank": i + 1, "agent_id": aid, "score": score}
                for i, (aid, score) in enumerate(top)
            ],
        }
