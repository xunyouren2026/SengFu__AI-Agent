"""
市场效率测试模块

测试订单撮合、定价预言机、托管、SLA监控和争议解决。
"""

import unittest
import time
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.market.order_matching import (
    MatchingEngine,
    TaskRequest,
    TaskRequirement,
    TaskPriority,
    TaskStatus,
    MatchingStrategy,
    MatchResult
)
from multiagent.market.escrow import (
    EscrowManager,
    EscrowAccount,
    EscrowStatus,
    ReleaseCondition
)


class MockMarketHelpers:
    """市场测试辅助类"""

    @staticmethod
    def create_task_request(
        request_id: str = "req_001",
        requester_id: str = "requester_1",
        capability: str = "text_processing"
    ) -> TaskRequest:
        """创建测试用任务请求"""
        return TaskRequest(
            request_id=request_id,
            requester_id=requester_id,
            requirements=[
                TaskRequirement(
                    capability=capability,
                    min_rating=4.0,
                    max_price=100.0
                )
            ],
            priority=TaskPriority.NORMAL,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            expires_at=time.time() + 300
        )

    @staticmethod
    def create_escrow_manager() -> EscrowManager:
        """创建测试用托管管理器"""
        manager = EscrowManager()
        # 添加一些初始余额
        manager.deposit("payer_1", 10000)
        manager.deposit("payee_1", 0)
        return manager


class TestOrderMatching(unittest.TestCase):
    """测试订单撮合"""

    def setUp(self):
        """测试初始化"""
        self.engine = MatchingEngine()

    def test_engine_initialization(self):
        """测试引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine._pending_tasks), 0)

    def test_submit_task(self):
        """测试提交任务"""
        task = MockMarketHelpers.create_task_request()

        result = self.engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements,
            priority=task.priority
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.requester_id, "requester_1")

    def test_submit_multiple_tasks(self):
        """测试提交多个任务"""
        for i in range(5):
            task = MockMarketHelpers.create_task_request(request_id=f"req_{i}")
            self.engine.submit_task(
                requester_id=task.requester_id,
                requirements=task.requirements
            )

        self.assertEqual(len(self.engine._pending_tasks), 5)

    def test_task_priority_queue(self):
        """测试任务优先级队列"""
        # 提交不同优先级的任务
        self.engine.submit_task(
            requester_id="low",
            requirements=[],
            priority=TaskPriority.LOW
        )
        self.engine.submit_task(
            requester_id="high",
            requirements=[],
            priority=TaskPriority.HIGH
        )
        self.engine.submit_task(
            requester_id="critical",
            requirements=[],
            priority=TaskPriority.CRITICAL
        )

        # 高优先级任务应该在队列前面
        _, _, first_id = self.engine._task_queue[0]
        self.assertEqual(first_id, "requester_2")

    def test_find_matches_no_candidates(self):
        """测试无候选者的匹配查找"""
        task = MockMarketHelpers.create_task_request()
        submitted = self.engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements
        )

        result = self.engine.find_matches(submitted.request_id)

        self.assertFalse(result.success)

    def test_find_matches_price_first(self):
        """测试价格优先匹配"""
        task = MockMarketHelpers.create_task_request()
        submitted = self.engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements
        )

        result = self.engine.find_matches(
            submitted.request_id,
            strategy=MatchingStrategy.PRICE_FIRST
        )

        # 没有候选者时应该失败
        self.assertFalse(result.success)

    def test_task_expiration(self):
        """测试任务过期"""
        task = MockMarketHelpers.create_task_request()
        submitted = self.engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements,
            ttl_seconds=1
        )

        # 等待过期
        time.sleep(1.1)

        result = self.engine.find_matches(submitted.request_id)

        self.assertFalse(result.success)
        self.assertIn("expired", result.error_message.lower())

    def test_cancel_task(self):
        """测试取消任务"""
        task = MockMarketHelpers.create_task_request()
        submitted = self.engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements
        )

        result = self.engine.cancel_task(submitted.request_id, "User cancelled")

        self.assertTrue(result)
        self.assertEqual(
            self.engine.get_task_status(submitted.request_id),
            TaskStatus.CANCELLED
        )

    def test_task_lifecycle(self):
        """测试任务生命周期"""
        # 提交任务
        submitted = self.engine.submit_task(
            requester_id="lifecycle_test",
            requirements=[],
            priority=TaskPriority.NORMAL
        )

        # 状态应该为PENDING
        self.assertEqual(
            self.engine.get_task_status(submitted.request_id),
            TaskStatus.PENDING
        )

        # 取消任务
        self.engine.cancel_task(submitted.request_id)

        # 状态应该为CANCELLED
        self.assertEqual(
            self.engine.get_task_status(submitted.request_id),
            TaskStatus.CANCELLED
        )


class TestPricingOracle(unittest.TestCase):
    """测试定价预言机"""

    def setUp(self):
        """测试初始化"""
        from multiagent.market.pricing_oracle import PricingOracle, PriceQuote
        self.oracle_module = __import__('multiagent.market.pricing_oracle', fromlist=['PricingOracle'])
        self.PricingOracle = self.oracle_module.PricingOracle

    def test_oracle_initialization(self):
        """测试预言机初始化"""
        oracle = self.PricingOracle()
        self.assertIsNotNone(oracle)

    def test_get_quote(self):
        """测试获取报价"""
        oracle = self.PricingOracle()

        quote = oracle.get_quote(
            service_type="text_processing",
            quantity=10
        )

        self.assertIsNotNone(quote)
        self.assertGreater(quote.price, 0)

    def test_price_history(self):
        """测试价格历史"""
        oracle = self.PricingOracle()

        # 获取多个报价
        for i in range(10):
            oracle.get_quote("service", 1)

        history = oracle.get_price_history("service")
        self.assertGreater(len(history), 0)

    def test_volatility_calculation(self):
        """测试波动性计算"""
        oracle = self.PricingOracle()

        # 添加历史数据
        for i in range(10):
            oracle.record_price("volatile_service", 100 + i * 10)

        volatility = oracle.calculate_volatility("volatile_service")
        self.assertGreater(volatility, 0)

    def test_moving_average(self):
        """测试移动平均"""
        oracle = self.PricingOracle()

        prices = [100, 105, 110, 115, 120]
        for price in prices:
            oracle.record_price("ma_service", price)

        ma = oracle.get_moving_average("ma_service", window=3)
        self.assertGreater(ma, 0)


class TestEscrow(unittest.TestCase):
    """测试托管系统"""

    def setUp(self):
        """测试初始化"""
        self.manager = MockMarketHelpers.create_escrow_manager()

    def test_escrow_initialization(self):
        """测试托管初始化"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(len(self.manager._escrows), 0)

    def test_create_escrow(self):
        """测试创建托管"""
        escrow = self.manager.create_escrow(
            task_id="task_001",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=1000
        )

        self.assertIsNotNone(escrow)
        self.assertEqual(escrow.task_id, "task_001")
        self.assertEqual(escrow.total_amount, 1000)
        self.assertEqual(escrow.status, EscrowStatus.PENDING)

    def test_create_escrow_with_stages(self):
        """测试分阶段托管"""
        escrow = self.manager.create_escrow(
            task_id="task_002",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=1000,
            stages=[
                ("Milestone 1", 300, ReleaseCondition.MILESTONE, {"milestone_id": "m1"}),
                ("Milestone 2", 700, ReleaseCondition.TASK_COMPLETION, {}),
            ]
        )

        self.assertEqual(len(escrow.stages), 2)
        self.assertEqual(escrow.stages[0].amount, 300)

    def test_fund_escrow(self):
        """测试充值托管"""
        escrow = self.manager.create_escrow(
            task_id="task_003",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        receipt = self.manager.fund_escrow(escrow.escrow_id)

        self.assertIsNotNone(receipt)
        self.assertTrue(receipt.confirmed)

    def test_release_single_stage(self):
        """测试释放单个阶段"""
        escrow = self.manager.create_escrow(
            task_id="task_004",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500,
            stages=[
                ("Completion", 500, ReleaseCondition.MANUAL, {}),
            ]
        )

        self.manager.fund_escrow(escrow.escrow_id)

        result = self.manager.release_stage(
            escrow.escrow_id,
            escrow.stages[0].stage_id,
            released_by="payer_1",
            proof={}
        )

        self.assertTrue(result)

    def test_release_all_stages(self):
        """测试释放所有阶段"""
        escrow = self.manager.create_escrow(
            task_id="task_005",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=1000,
            stages=[
                ("Stage 1", 500, ReleaseCondition.MANUAL, {}),
                ("Stage 2", 500, ReleaseCondition.MANUAL, {}),
            ]
        )

        self.manager.fund_escrow(escrow.escrow_id)

        result = self.manager.release_all(
            escrow.escrow_id,
            released_by="payer_1",
            proof={}
        )

        self.assertTrue(result)

    def test_escrow_cancellation(self):
        """测试托管取消"""
        escrow = self.manager.create_escrow(
            task_id="task_006",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        result = self.manager.cancel_escrow(escrow.escrow_id, "payer_1")

        self.assertTrue(result)
        self.assertEqual(
            self.manager.get_escrow(escrow.escrow_id).status,
            EscrowStatus.CANCELLED
        )

    def test_refund_on_cancellation(self):
        """测试取消时退款"""
        escrow = self.manager.create_escrow(
            task_id="task_007",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        self.manager.fund_escrow(escrow.escrow_id)
        initial_balance = self.manager.get_balance("payer_1")

        self.manager.cancel_escrow(escrow.escrow_id, "payer_1")

        final_balance = self.manager.get_balance("payer_1")
        self.assertGreater(final_balance, initial_balance - 500)

    def test_initiate_dispute(self):
        """测试发起争议"""
        escrow = self.manager.create_escrow(
            task_id="task_008",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        self.manager.fund_escrow(escrow.escrow_id)

        result = self.manager.initiate_dispute(
            escrow.escrow_id,
            "payer_1",
            "Quality issues"
        )

        self.assertTrue(result)
        self.assertEqual(
            self.manager.get_escrow(escrow.escrow_id).status,
            EscrowStatus.DISPUTED
        )

    def test_resolve_dispute(self):
        """测试解决争议"""
        escrow = self.manager.create_escrow(
            task_id="task_009",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        self.manager.fund_escrow(escrow.escrow_id)
        self.manager.initiate_dispute(escrow.escrow_id, "payer_1", "Disagreement")

        result = self.manager.resolve_dispute(
            escrow.escrow_id,
            resolution="Partial refund",
            refund_amount=250
        )

        self.assertTrue(result)

    def test_get_escrow_by_task(self):
        """测试通过任务ID获取托管"""
        escrow = self.manager.create_escrow(
            task_id="task_unique",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500
        )

        retrieved = self.manager.get_escrow_by_task("task_unique")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.escrow_id, escrow.escrow_id)


class TestSLAMonitoring(unittest.TestCase):
    """测试SLA监控"""

    def setUp(self):
        """测试初始化"""
        from multiagent.market.sla_monitor import SLAMonitor, SLAViolation, SLAThreshold
        self.sla_module = __import__('multiagent.market.sla_monitor', fromlist=['SLAMonitor'])
        self.SLAMonitor = self.sla_module.SLAMonitor
        self.SLAViolation = self.sla_module.SLAViolation

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = self.SLAMonitor()
        self.assertIsNotNone(monitor)

    def test_define_sla(self):
        """测试定义SLA"""
        monitor = self.SLAMonitor()

        monitor.define_sla(
            service_type="premium",
            response_time_ms=1000,
            availability=0.99,
            error_rate=0.01
        )

        self.assertIn("premium", monitor.sla_definitions)

    def test_record_response_time(self):
        """测试记录响应时间"""
        monitor = self.SLAMonitor()

        monitor.define_sla("standard", response_time_ms=2000)
        monitor.record_response_time("standard", 1500)

        self.assertGreater(len(monitor.response_times["standard"]), 0)

    def test_check_availability(self):
        """测试检查可用性"""
        monitor = self.SLAMonitor()

        monitor.define_sla("test_service", availability=0.95)

        # 模拟一些请求
        for i in range(20):
            monitor.record_request("test_service", success=(i % 20 < 18))

        availability = monitor.calculate_availability("test_service")
        self.assertGreaterEqual(availability, 0)

    def test_detect_violation(self):
        """测试检测违规"""
        monitor = self.SLAMonitor()

        monitor.define_sla("slow_service", response_time_ms=500)

        # 记录超时响应
        monitor.record_response_time("slow_service", 1000)
        monitor.record_response_time("slow_service", 1000)
        monitor.record_response_time("slow_service", 1000)

        violations = monitor.detect_violations("slow_service")
        self.assertGreater(len(violations), 0)

    def test_sla_breach_notification(self):
        """测试SLA违约通知"""
        monitor = self.SLAMonitor()

        notifications = []
        def notification_handler(violation):
            notifications.append(violation)

        monitor.set_notification_handler(notification_handler)
        monitor.define_sla("notify_service", response_time_ms=100)

        monitor.record_response_time("notify_service", 500)

        self.assertGreaterEqual(len(notifications), 0)


class TestDisputeResolution(unittest.TestCase):
    """测试争议解决"""

    def setUp(self):
        """测试初始化"""
        from multiagent.market.dispute_resolution import DisputeResolver, Dispute, DisputeStatus
        self.dispute_module = __import__('multiagent.market.dispute_resolution', fromlist=['DisputeResolver'])
        self.DisputeResolver = self.dispute_module.DisputeResolver
        self.Dispute = self.dispute_module.Dispute

    def test_resolver_initialization(self):
        """测试解决器初始化"""
        resolver = self.DisputeResolver()
        self.assertIsNotNone(resolver)

    def test_create_dispute(self):
        """测试创建争议"""
        resolver = self.DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="escrow_123",
            initiator_id="user_1",
            reason="Service not delivered"
        )

        self.assertIsNotNone(dispute)
        self.assertEqual(dispute.initiator_id, "user_1")

    def test_add_evidence(self):
        """测试添加证据"""
        resolver = self.DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="escrow_456",
            initiator_id="user_1",
            reason="Issue"
        )

        resolver.add_evidence(
            dispute.dispute_id,
            submitted_by="user_1",
            evidence_type="screenshot",
            content="data:image/png;base64,..."
        )

        self.assertGreater(len(dispute.evidence), 0)

    def test_arbitration_vote(self):
        """测试仲裁投票"""
        resolver = self.DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="escrow_789",
            initiator_id="user_1",
            reason="Disagreement"
        )

        # 模拟仲裁投票
        resolver.cast_vote(dispute.dispute_id, "arbitrator_1", "refund")
        resolver.cast_vote(dispute.dispute_id, "arbitrator_2", "partial")
        resolver.cast_vote(dispute.dispute_id, "arbitrator_3", "release")

        votes = resolver.get_votes(dispute.dispute_id)
        self.assertEqual(len(votes), 3)

    def test_finalize_dispute(self):
        """测试最终化争议"""
        resolver = self.DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="escrow_final",
            initiator_id="user_1",
            reason="Final test"
        )

        # 添加投票
        resolver.cast_vote(dispute.dispute_id, "arb_1", "refund")

        result = resolver.finalize(dispute.dispute_id)

        self.assertIsNotNone(result)

    def test_appeal_process(self):
        """测试申诉流程"""
        resolver = self.DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="escrow_appeal",
            initiator_id="user_1",
            reason="Appeal test"
        )

        # 初始解决
        resolver.finalize(dispute.dispute_id)

        # 发起申诉
        appeal = resolver.appeal(dispute.dispute_id, "user_1", "Unfair decision")

        self.assertIsNotNone(appeal)

    def test_dispute_timeout(self):
        """测试争议超时"""
        resolver = self.DisputeResolver(timeout_hours=1)

        dispute = resolver.create_dispute(
            escrow_id="escrow_timeout",
            initiator_id="user_1",
            reason="Timeout test"
        )

        # 检查超时
        is_timed_out = resolver.check_timeout(dispute.dispute_id)
        self.assertFalse(is_timed_out)


class TestMarketStatistics(unittest.TestCase):
    """测试市场统计"""

    def test_matching_statistics(self):
        """测试撮合统计"""
        engine = MatchingEngine()

        for i in range(10):
            engine.submit_task(
                requester_id=f"user_{i}",
                requirements=[],
                priority=TaskPriority.NORMAL
            )

        stats = engine.get_statistics()

        self.assertEqual(stats["total_tasks"], 10)

    def test_escrow_statistics(self):
        """测试托管统计"""
        manager = MockMarketHelpers.create_escrow_manager()

        for i in range(5):
            manager.create_escrow(
                task_id=f"task_{i}",
                payer_id="payer_1",
                payee_id="payee_1",
                total_amount=100 * (i + 1)
            )

        stats = manager.get_statistics()

        self.assertEqual(stats["total_escrows"], 5)


class TestServiceListings(unittest.TestCase):
    """测试服务列表"""

    def setUp(self):
        """测试初始化"""
        from multiagent.market.listings import ListingManager, ServiceListing, Capability
        self.listings_module = __import__('multiagent.market.listings', fromlist=['ListingManager'])
        self.ListingManager = self.listings_module.ListingManager

    def test_listing_manager_initialization(self):
        """测试列表管理器初始化"""
        manager = self.ListingManager()
        self.assertIsNotNone(manager)

    def test_register_service(self):
        """测试注册服务"""
        manager = self.ListingManager()

        manager.register_service(
            agent_id="agent_001",
            name="Text Processing Service",
            capabilities=["text_processing", "nlp"],
            base_price=50.0
        )

        self.assertEqual(len(manager.listings), 1)

    def test_search_by_capability(self):
        """测试按能力搜索"""
        manager = self.ListingManager()

        manager.register_service(
            agent_id="nlp_agent",
            name="NLP Service",
            capabilities=["nlp", "sentiment_analysis"],
            base_price=100.0
        )
        manager.register_service(
            agent_id="cv_agent",
            name="CV Service",
            capabilities=["computer_vision", "object_detection"],
            base_price=150.0
        )

        results = manager.search_by_capability("nlp")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_id, "nlp_agent")

    def test_update_rating(self):
        """测试更新评分"""
        manager = self.ListingManager()

        manager.register_service(
            agent_id="rating_test",
            name="Test Service",
            capabilities=["test"],
            base_price=10.0
        )

        manager.update_rating("rating_test", 4.5)
        listing = manager.get_listing("rating_test")

        self.assertEqual(listing.average_rating, 4.5)


class TestAgentLoad(unittest.TestCase):
    """测试Agent负载"""

    def test_load_factor_calculation(self):
        """测试负载因子计算"""
        from multiagent.market.order_matching import AgentLoad

        load = AgentLoad(
            agent_id="load_test",
            current_tasks=5,
            max_concurrent=10,
            queue_depth=3
        )

        self.assertAlmostEqual(load.load_factor, 0.8, places=1)

    def test_agent_availability(self):
        """测试Agent可用性"""
        from multiagent.market.order_matching import AgentLoad

        load = AgentLoad(
            agent_id="availability_test",
            current_tasks=5,
            max_concurrent=10
        )

        self.assertTrue(load.is_available)

        load.current_tasks = 10
        self.assertFalse(load.is_available)


class TestMatchingStrategies(unittest.TestCase):
    """测试匹配策略"""

    def test_strategy_selection(self):
        """测试策略选择"""
        strategies = [
            MatchingStrategy.PRICE_FIRST,
            MatchingStrategy.QUALITY_FIRST,
            MatchingStrategy.SPEED_FIRST,
            MatchingStrategy.BALANCED,
            MatchingStrategy.LOAD_BALANCED
        ]

        self.assertEqual(len(strategies), 5)

    def test_strategy_weights(self):
        """测试策略权重"""
        engine = MatchingEngine()

        weights = engine._strategy_weights.get(MatchingStrategy.PRICE_FIRST)

        self.assertIn("price", weights)
        self.assertGreater(weights["price"], 0)


class TestMarketEdgeCases(unittest.TestCase):
    """测试市场边界情况"""

    def test_zero_budget_task(self):
        """测试零预算任务"""
        engine = MatchingEngine()

        task = MockMarketHelpers.create_task_request()
        submitted = engine.submit_task(
            requester_id=task.requester_id,
            requirements=task.requirements,
            budget=0
        )

        self.assertIsNotNone(submitted)

    def test_expired_escrow(self):
        """测试过期托管"""
        manager = MockMarketHelpers.create_escrow_manager()

        escrow = manager.create_escrow(
            task_id="expiring_task",
            payer_id="payer_1",
            payee_id="payee_1",
            total_amount=500,
            ttl_hours=0.001  # 非常短的TTL
        )

        # 等待过期
        time.sleep(0.01)

        expired = manager.check_expired_escrows()
        self.assertGreater(len(expired), 0)

    def test_dispute_with_no_evidence(self):
        """测试无证据的争议"""
        from multiagent.market.dispute_resolution import DisputeResolver

        resolver = DisputeResolver()

        dispute = resolver.create_dispute(
            escrow_id="no_evidence",
            initiator_id="user_1",
            reason="No evidence case"
        )

        # 没有证据也应该能解决
        result = resolver.finalize(dispute.dispute_id)
        self.assertIsNotNone(result)

    def test_duplicate_task_submission(self):
        """测试重复任务提交"""
        engine = MatchingEngine()

        # 多次提交相同任务
        for _ in range(3):
            engine.submit_task(
                requester_id="duplicate_user",
                requirements=[],
                priority=TaskPriority.NORMAL
            )

        stats = engine.get_statistics()
        self.assertGreaterEqual(stats["total_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
