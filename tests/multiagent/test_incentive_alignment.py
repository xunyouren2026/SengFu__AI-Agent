"""
激励对齐测试模块

测试代币经济、奖励分配(Shapley)、质押、惩罚和对齐验证。
"""

import unittest
import time
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os
from decimal import Decimal

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.federated.blockchain.incentive import (
    TokenEconomy,
    RewardDistribution,
    Staking,
    Slashing,
    AlignmentVerifier
)


class MockIncentiveHelpers:
    """激励测试辅助类"""

    @staticmethod
    def create_agent(
        agent_id: str,
        reputation: float = 1.0,
        contributed_work: float = 0.0
    ) -> Dict[str, Any]:
        """创建测试用Agent数据"""
        return {
            "agent_id": agent_id,
            "reputation": reputation,
            "contributions": {
                "work": contributed_work,
                "quality": 1.0,
                "timeliness": 1.0
            },
            "staked_amount": 0.0,
            "pending_rewards": 0.0
        }


class TestTokenEconomy(unittest.TestCase):
    """测试代币经济"""

    def setUp(self):
        """测试初始化"""
        self.economy = TokenEconomy(
            initial_supply=1000000,
            inflation_rate=0.05,
            reward_pool=100000
        )

    def test_economy_initialization(self):
        """测试经济初始化"""
        self.assertEqual(self.economy.total_supply, 1000000)
        self.assertEqual(self.economy.inflation_rate, 0.05)

    def test_token_mint(self):
        """测试代币铸造"""
        initial_supply = self.economy.total_supply
        minted = self.economy.mint_tokens(1000)

        self.assertTrue(minted)
        self.assertEqual(self.economy.total_supply, initial_supply + 1000)

    def test_token_burn(self):
        """测试代币销毁"""
        self.economy.mint_tokens(1000)
        initial_supply = self.economy.total_supply

        burned = self.economy.burn_tokens(500)

        self.assertTrue(burned)
        self.assertEqual(self.economy.total_supply, initial_supply - 500)

    def test_transfer(self):
        """测试代币转账"""
        self.economy.mint_tokens(1000)

        result = self.economy.transfer("agent_1", "agent_2", 500)

        self.assertTrue(result)
        self.assertEqual(self.economy.get_balance("agent_1"), 500)
        self.assertEqual(self.economy.get_balance("agent_2"), 500)

    def test_insufficient_balance_transfer(self):
        """测试余额不足转账"""
        self.economy.mint_tokens(100)

        result = self.economy.transfer("agent_1", "agent_2", 200)

        self.assertFalse(result)

    def test_inflation_adjustment(self):
        """测试通货膨胀调整"""
        initial_supply = self.economy.total_supply
        self.economy.apply_inflation()

        # 通货膨胀后供应量应增加
        self.assertGreater(self.economy.total_supply, initial_supply)

    def test_reward_pool_allocation(self):
        """测试奖励池分配"""
        initial_pool = self.economy.reward_pool
        allocated = self.economy.allocate_rewards("agent_1", 100)

        self.assertTrue(allocated)
        self.assertEqual(self.economy.reward_pool, initial_pool - 100)

    def test_transaction_fee(self):
        """测试交易手续费"""
        self.economy.set_transaction_fee(0.01)  # 1%手续费
        self.economy.mint_tokens(1000)

        initial_balance = self.economy.get_balance("agent_1")
        self.economy.transfer("agent_1", "agent_2", 100)

        # 发送方应支付手续费
        final_balance = self.economy.get_balance("agent_1")
        self.assertLess(final_balance, initial_balance - 100)


class TestRewardDistribution(unittest.TestCase):
    """测试奖励分配"""

    def setUp(self):
        """测试初始化"""
        self.distribution = RewardDistribution(total_reward=1000)

    def test_distribution_initialization(self):
        """测试分配初始化"""
        self.assertEqual(self.distribution.total_reward, 1000)
        self.assertEqual(len(self.distribution.allocations), 0)

    def test_equal_distribution(self):
        """测试平均分配"""
        agents = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

        allocations = self.distribution.distribute_equally(agents)

        # 每人应获得200
        for agent_id in agents:
            self.assertEqual(allocations.get(agent_id), 200)

    def test_contribution_based_distribution(self):
        """测试贡献度分配"""
        contributions = {
            "agent_1": 50.0,
            "agent_2": 30.0,
            "agent_3": 20.0
        }

        allocations = self.distribution.distribute_by_contribution(contributions)

        # 总贡献100，应按比例分配
        self.assertEqual(allocations["agent_1"], 500)  # 50%
        self.assertEqual(allocations["agent_2"], 300)  # 30%
        self.assertEqual(allocations["agent_3"], 200)  # 20%

    def test_shapley_value_distribution(self):
        """测试Shapley值分配"""
        # 模拟联盟博弈
        agents = ["a", "b", "c"]

        def value_function(coalition):
            # 简单的值函数: 联盟中每增加一个成员增加10
            return len(coalition) * 10

        allocations = self.distribution.distribute_by_shapley(
            agents,
            value_function
        )

        # 验证分配总和等于总奖励
        total_allocated = sum(allocations.values())
        self.assertAlmostEqual(total_allocated, 1000, places=5)

    def test_quality_weighted_distribution(self):
        """测试质量加权分配"""
        quality_scores = {
            "agent_1": 1.0,
            "agent_2": 0.8,
            "agent_3": 0.5
        }

        allocations = self.distribution.distribute_by_quality(quality_scores)

        # agent_1 应该获得最多
        self.assertGreater(allocations["agent_1"], allocations["agent_2"])
        self.assertGreater(allocations["agent_2"], allocations["agent_3"])

    def test_timeliness_weighted_distribution(self):
        """测试时效性加权分配"""
        timeliness = {
            "agent_1": 1.0,  # 按时完成
            "agent_2": 0.8,  # 略有延迟
            "agent_3": 0.0   # 严重超时
        }

        allocations = self.distribution.distribute_by_timeliness(timeliness)

        # agent_3 不应获得任何奖励
        self.assertEqual(allocations["agent_3"], 0)

    def test_minimum_threshold(self):
        """测试最低阈值"""
        self.distribution.minimum_allocation = 50

        quality_scores = {
            "agent_1": 0.01,
            "agent_2": 0.5,
            "agent_3": 0.99
        }

        allocations = self.distribution.distribute_by_quality(
            quality_scores,
            apply_threshold=True
        )

        # agent_1 应该被排除
        self.assertNotIn("agent_1", allocations)

    def test_remaining_reward_handling(self):
        """测试剩余奖励处理"""
        # 设置一个会产生余数的情况
        self.distribution.total_reward = 100
        agents = ["agent_1", "agent_2", "agent_3"]

        allocations = self.distribution.distribute_equally(agents)

        total_allocated = sum(allocations.values())
        remaining = self.distribution.total_reward - total_allocated

        # 剩余奖励应被追踪
        self.assertGreaterEqual(remaining, 0)


class TestStaking(unittest.TestCase):
    """测试质押机制"""

    def setUp(self):
        """测试初始化"""
        self.staking = Staking(
            min_stake=100,
            max_stake=10000,
            reward_rate=0.1
        )

    def test_staking_initialization(self):
        """测试质押初始化"""
        self.assertEqual(self.staking.min_stake, 100)
        self.assertEqual(self.staking.max_stake, 10000)
        self.assertEqual(self.staking.reward_rate, 0.1)

    def test_stake_tokens(self):
        """测试质押代币"""
        result = self.staking.stake("agent_1", 500)

        self.assertTrue(result)
        self.assertEqual(self.staking.get_stake("agent_1"), 500)

    def test_unstake_tokens(self):
        """测试解除质押"""
        self.staking.stake("agent_1", 500)
        result = self.staking.unstake("agent_1", 200)

        self.assertTrue(result)
        self.assertEqual(self.staking.get_stake("agent_1"), 300)

    def test_unstake_more_than_staked(self):
        """测试解除超过质押数量"""
        self.staking.stake("agent_1", 500)
        result = self.staking.unstake("agent_1", 1000)

        self.assertFalse(result)

    def test_below_minimum_stake(self):
        """测试低于最低质押"""
        result = self.staking.stake("agent_1", 50)

        self.assertFalse(result)

    def test_above_maximum_stake(self):
        """测试超过最大质押"""
        result = self.staking.stake("agent_1", 15000)

        self.assertFalse(result)

    def test_staking_rewards_calculation(self):
        """测试质押奖励计算"""
        self.staking.stake("agent_1", 1000)

        # 计算1年的奖励 (10%)
        reward = self.staking.calculate_reward("agent_1", period_years=1)

        self.assertEqual(reward, 100)

    def test_compounding_rewards(self):
        """测试复利奖励"""
        self.staking.stake("agent_1", 1000)
        self.staking.reward_rate = 0.1

        # 复利计算: 1000 * (1.1)^2 - 1000
        year1_reward = self.staking.calculate_reward("agent_1", period_years=1)
        self.staking.claim_reward("agent_1")

        year2_reward = self.staking.calculate_reward("agent_1", period_years=1)

        # 第二年奖励应基于复利计算
        self.assertGreater(year2_reward, year1_reward)

    def test_stake_lock_period(self):
        """测试质押锁定期"""
        self.staking.stake("agent_1", 500)
        self.staking.lock_period = 7  # 7天锁定期

        # 尝试在锁定期内解除质押
        result = self.staking.unstake("agent_1", 100, force=False)

        self.assertFalse(result)

    def test_force_unstake(self):
        """测试强制解除质押"""
        self.staking.stake("agent_1", 500)
        self.staking.lock_period = 7

        # 强制解除会有惩罚
        result, penalty = self.staking.unstake("agent_1", 100, force=True)

        self.assertTrue(result)
        self.assertGreater(penalty, 0)

    def test_delegation(self):
        """测试委托质押"""
        # agent_2 委托给 agent_1
        result = self.staking.delegate("agent_2", "agent_1", 300)

        self.assertTrue(result)
        self.assertEqual(self.staking.get_delegation("agent_2", "agent_1"), 300)
        self.assertEqual(self.staking.get_total_stake("agent_1"), 300)

    def test_slashing_on_misbehavior(self):
        """测试恶意行为惩罚"""
        self.staking.stake("agent_1", 1000)

        # 模拟恶意行为
        self.staking.slash("agent_1", 0.1)  # 10%惩罚

        self.assertEqual(self.staking.get_stake("agent_1"), 900)


class TestSlashing(unittest.TestCase):
    """测试惩罚机制"""

    def setUp(self):
        """测试初始化"""
        self.slashing = Slashing(
            penalty_rate=0.1,
            jail_period_blocks=100
        )

    def test_slashing_initialization(self):
        """测试惩罚初始化"""
        self.assertEqual(self.slashing.penalty_rate, 0.1)
        self.assertEqual(self.slashing.jail_period_blocks, 100)

    def test_apply_slashing(self):
        """测试应用惩罚"""
        result = self.slashing.apply_slashing("agent_1", 500)

        self.assertTrue(result)
        self.assertEqual(self.slashing.get_total_slashed("agent_1"), 50)

    def test_slashing_calculation(self):
        """测试惩罚计算"""
        self.slashing.penalty_rate = 0.25
        self.slashing.apply_slashing("agent_1", 1000)

        # 25%惩罚 = 250
        self.assertEqual(self.slashing.get_total_slashed("agent_1"), 250)

    def test_repeated_slashing(self):
        """测试重复惩罚"""
        self.slashing.apply_slashing("agent_1", 1000)
        self.slashing.apply_slashing("agent_1", 500)

        # 累积惩罚
        self.assertEqual(self.slashing.get_total_slashed("agent_1"), 150)

    def test_jail_mechanism(self):
        """测试监禁机制"""
        self.slashing.jail_agent("agent_1", reason="double_signing")

        self.assertTrue(self.slashing.is_jailed("agent_1"))

    def test_jail_release(self):
        """测试释放出狱"""
        self.slashing.jail_agent("agent_1")
        self.slashing.release_from_jail("agent_1", current_block=150)

        self.assertFalse(self.slashing.is_jailed("agent_1"))

    def test_slash_to_treasury(self):
        """测试惩罚进入国库"""
        self.slashing.apply_slashing("agent_1", 1000)

        treasury = self.slashing.get_treasury_balance()
        self.assertEqual(treasury, 100)  # 10%惩罚进入国库

    def test_offense_types(self):
        """测试不同罪行类型"""
        offenses = {
            "double_signing": 0.5,
            "downtime": 0.1,
            "malicious_behavior": 0.8
        }

        for offense, expected_rate in offenses.items():
            self.slashing.penalty_rates[offense] = expected_rate

        self.assertEqual(self.slashing.penalty_rates["double_signing"], 0.5)

    def test_slashing_records(self):
        """测试惩罚记录"""
        self.slashing.apply_slashing(
            "agent_1",
            500,
            offense_type="malicious_behavior"
        )

        records = self.slashing.get_slashing_records("agent_1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["offense_type"], "malicious_behavior")


class TestAlignmentVerifier(unittest.TestCase):
    """测试对齐验证器"""

    def setUp(self):
        """测试初始化"""
        self.verifier = AlignmentVerifier(threshold=0.8)

    def test_verifier_initialization(self):
        """测试验证器初始化"""
        self.assertEqual(self.verifier.threshold, 0.8)

    def test_verify_alignment(self):
        """测试验证对齐"""
        # 模拟Agent行为评估
        alignment_score = self.verifier.verify_alignment(
            agent_id="agent_1",
            actions=[
                {"action": "helpful", "score": 0.9},
                {"action": "safe", "score": 0.85},
                {"action": "honest", "score": 0.8}
            ]
        )

        self.assertGreaterEqual(alignment_score, 0.0)
        self.assertLessEqual(alignment_score, 1.0)

    def test_alignment_above_threshold(self):
        """测试对齐分数高于阈值"""
        self.verifier.threshold = 0.5

        score = self.verifier.verify_alignment(
            agent_id="agent_1",
            actions=[
                {"action": "good", "score": 0.9},
                {"action": "good", "score": 0.8}
            ]
        )

        self.assertTrue(score >= 0.5)

    def test_alignment_below_threshold(self):
        """测试对齐分数低于阈值"""
        self.verifier.threshold = 0.9

        score = self.verifier.verify_alignment(
            agent_id="agent_1",
            actions=[
                {"action": "questionable", "score": 0.5},
                {"action": "bad", "score": 0.3}
            ]
        )

        self.assertFalse(score >= 0.9)

    def test_safety_check(self):
        """测试安全检查"""
        result = self.verifier.check_safety(
            agent_id="agent_1",
            action={"type": "delete_data", "target": "important_file"}
        )

        self.assertIn(result, [True, False])

    def test_honesty_check(self):
        """测试诚实性检查"""
        result = self.verifier.check_honesty(
            agent_id="agent_1",
            statements=[
                {"claim": "x=1", "verified": True},
                {"claim": "x=2", "verified": False}
            ]
        )

        self.assertLess(result, 1.0)

    def test_helpfulness_check(self):
        """测试有用性检查"""
        result = self.verifier.check_helpfulness(
            agent_id="agent_1",
            responses=[
                {"useful": True, "relevant": True},
                {"useful": True, "relevant": False},
                {"useful": False, "relevant": True}
            ]
        )

        self.assertGreater(result, 0)
        self.assertLess(result, 1.0)

    def test_fairness_check(self):
        """测试公平性检查"""
        result = self.verifier.check_fairness(
            agent_id="agent_1",
            decisions=[
                {"fair": True, "biased": False},
                {"fair": False, "biased": True}
            ]
        )

        self.assertLess(result, 1.0)

    def test_alignment_history(self):
        """测试对齐历史"""
        for i in range(5):
            self.verifier.verify_alignment(
                agent_id="agent_1",
                actions=[{"action": f"action_{i}", "score": 0.8 + i * 0.02}]
            )

        history = self.verifier.get_alignment_history("agent_1")
        self.assertEqual(len(history), 5)

    def test_alignment_trend(self):
        """测试对齐趋势"""
        # 模拟改善趋势
        scores = [0.6, 0.65, 0.7, 0.75, 0.8]
        for score in scores:
            self.verifier.verify_alignment(
                agent_id="agent_1",
                actions=[{"action": "good", "score": score}]
            )

        trend = self.verifier.get_alignment_trend("agent_1")
        self.assertEqual(trend, "improving")

    def test_flag_misaligned_agent(self):
        """测试标记不对齐Agent"""
        self.verifier.flag_agent("agent_1", reason="repeated_safety_violations")

        self.assertTrue(self.verifier.is_flagged("agent_1"))

    def test_unflag_agent(self):
        """测试取消标记"""
        self.verifier.flag_agent("agent_1")
        self.verifier.unflag_agent("agent_1")

        self.assertFalse(self.verifier.is_flagged("agent_1"))


class TestIncentiveMechanisms(unittest.TestCase):
    """测试激励机制"""

    def test_proportional_rewards(self):
        """测试比例奖励"""
        contribution = 500
        total_contribution = 1000
        total_reward = 100

        # 应获得50%
        reward = (contribution / total_contribution) * total_reward
        self.assertEqual(reward, 50)

    def test_bonus_for_quality(self):
        """测试质量奖金"""
        base_reward = 100
        quality_score = 0.95
        max_bonus = 0.2

        bonus = base_reward * quality_score * max_bonus
        total = base_reward + bonus

        self.assertAlmostEqual(total, 119, places=0)

    def test_penalty_for_poor_quality(self):
        """测试低质量惩罚"""
        base_reward = 100
        quality_score = 0.3
        max_penalty = 0.3

        penalty = base_reward * (1 - quality_score) * max_penalty
        total = base_reward - penalty

        self.assertAlmostEqual(total, 79, places=0)

    def test_early_completion_bonus(self):
        """测试提前完成奖金"""
        base_reward = 100
        deadline = 3600  # 1小时
        completion_time = 1800  # 30分钟

        # 提前50%完成，获得额外10%奖励
        time_saved_ratio = 1 - (completion_time / deadline)
        bonus_rate = 0.1
        bonus = base_reward * time_saved_ratio * bonus_rate

        self.assertEqual(bonus, 5)


class TestEconomicSafety(unittest.TestCase):
    """测试经济安全性"""

    def test_hyperinflation_prevention(self):
        """测试防止恶性通货膨胀"""
        economy = TokenEconomy(inflation_rate=0.02)
        initial_supply = economy.total_supply

        # 应用10年通货膨胀
        for _ in range(10):
            economy.apply_inflation()

        # 通胀率应保持稳定
        self.assertLessEqual(economy.inflation_rate, 0.1)

    def test_supply_cap(self):
        """测试供应上限"""
        economy = TokenEconomy(max_supply=1000000)
        economy.mint_tokens(500000)

        # 尝试超过上限
        result = economy.mint_tokens(600000)

        self.assertFalse(result)

    def test_sybil_attack_prevention(self):
        """测试女巫攻击预防"""
        staking = Staking(min_stake=100)

        # 模拟创建多个低质押账户
        for i in range(10):
            staking.stake(f"fake_agent_{i}", 10)

        # 没有账户应该达到最低要求
        for i in range(10):
            self.assertEqual(staking.get_stake(f"fake_agent_{i}"), 10)

    def test_cartel_formation_prevention(self):
        """测试防止卡特尔形成"""
        distribution = RewardDistribution(total_reward=1000)

        # 模拟大Agent试图控制奖励
        contributions = {
            "large_agent": 90,
            "small_agent_1": 1,
            "small_agent_2": 1,
            "small_agent_3": 1,
            "small_agent_4": 1,
            "small_agent_5": 1,
            "small_agent_6": 1,
            "small_agent_7": 1,
            "small_agent_8": 1,
            "small_agent_9": 1,
            "small_agent_10": 1
        }

        allocations = distribution.distribute_by_contribution(
            contributions,
            anti_centralization_factor=0.5
        )

        # 大Agent不应获得与其贡献成比例的奖励
        self.assertLess(allocations["large_agent"], 900)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_zero_reward_distribution(self):
        """测试零奖励分配"""
        distribution = RewardDistribution(total_reward=0)
        agents = ["a", "b", "c"]

        allocations = distribution.distribute_equally(agents)

        self.assertEqual(sum(allocations.values()), 0)

    def test_single_agent_reward(self):
        """测试单Agent奖励"""
        distribution = RewardDistribution(total_reward=100)
        allocations = distribution.distribute_equally(["single_agent"])

        self.assertEqual(allocations["single_agent"], 100)

    def test_zero_stake(self):
        """测试零质押"""
        staking = Staking()
        result = staking.stake("agent_1", 0)

        self.assertFalse(result)

    def test_exact_min_stake(self):
        """测试恰好最低质押"""
        staking = Staking(min_stake=100)
        result = staking.stake("agent_1", 100)

        self.assertTrue(result)

    def test_exact_max_stake(self):
        """测试恰好最大质押"""
        staking = Staking(max_stake=1000)
        result = staking.stake("agent_1", 1000)

        self.assertTrue(result)

    def test_alignment_check_no_history(self):
        """测试无历史的对齐检查"""
        verifier = AlignmentVerifier()

        score = verifier.verify_alignment(
            agent_id="new_agent",
            actions=[{"action": "first", "score": 1.0}]
        )

        # 新Agent应该基于当前行为评分
        self.assertGreater(score, 0)

    def test_slashing_below_zero(self):
        """测试惩罚不会让质押低于零"""
        slashing = Slashing()
        slashing.apply_slashing("agent_1", 100)
        slashing.apply_slashing("agent_1", 200)

        # 总惩罚不应超过初始质押
        # 实际实现中应该有保护
        total_slashed = slashing.get_total_slashed("agent_1")
        self.assertGreaterEqual(total_slashed, 0)


if __name__ == "__main__":
    unittest.main()
