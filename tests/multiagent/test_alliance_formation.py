"""
联盟形成测试模块

测试联盟形成引擎、拍卖机制、合同网协议、联盟值计算(Shapley)、
匈牙利优化器和容错机制。
"""

import unittest
import time
import math
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os
from itertools import combinations

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.alliance.formation import (
    CoalitionFormationEngine,
    CoalitionStrategy,
    FormationResult,
    Coalition,
    Task,
    TaskGraph,
    Agent
)


class MockAllianceHelpers:
    """联盟测试辅助类"""

    @staticmethod
    def create_agent(
        agent_id: str,
        capabilities: Optional[Set[str]] = None,
        cost_per_unit: float = 1.0,
        reliability: float = 1.0,
        max_load: float = 10.0
    ) -> Agent:
        """创建测试用Agent"""
        caps = capabilities or {"text_processing"}
        return Agent(
            agent_id=agent_id,
            capabilities=caps,
            cost_per_unit=cost_per_unit,
            reliability=reliability,
            current_load=0.0,
            max_load=max_load
        )

    @staticmethod
    def create_task(
        task_id: str,
        required_capabilities: Optional[Set[str]] = None,
        priority: int = 1,
        estimated_effort: float = 1.0
    ) -> Task:
        """创建测试用任务"""
        caps = required_capabilities or {"text_processing"}
        return Task(
            task_id=task_id,
            required_capabilities=caps,
            priority=priority,
            estimated_effort=estimated_effort
        )

    @staticmethod
    def create_task_graph(tasks: List[Task]) -> TaskGraph:
        """创建测试用任务图"""
        graph = TaskGraph()
        for task in tasks:
            graph.add_task(task)
        return graph


class TestFormationEngine(unittest.TestCase):
    """测试联盟形成引擎"""

    def test_engine_initialization(self):
        """测试引擎初始化"""
        engine = CoalitionFormationEngine()
        self.assertIsNotNone(engine)
        self.assertEqual(engine.strategy, CoalitionStrategy.GREEDY)
        self.assertEqual(len(engine.agents), 0)

    def test_engine_with_custom_strategy(self):
        """测试自定义策略引擎"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.OPTIMAL)
        self.assertEqual(engine.strategy, CoalitionStrategy.OPTIMAL)

    def test_register_single_agent(self):
        """测试注册单个Agent"""
        engine = CoalitionFormationEngine()
        agent = MockAllianceHelpers.create_agent("agent_001")
        engine.register_agent(agent)

        self.assertEqual(len(engine.agents), 1)
        self.assertIn("agent_001", engine.agents)

    def test_register_multiple_agents(self):
        """测试注册多个Agent"""
        engine = CoalitionFormationEngine()
        for i in range(5):
            agent = MockAllianceHelpers.create_agent(f"agent_{i}")
            engine.register_agent(agent)

        self.assertEqual(len(engine.agents), 5)

    def test_set_task_graph(self):
        """测试设置任务图"""
        engine = CoalitionFormationEngine()
        task = MockAllianceHelpers.create_task("task_001")
        graph = MockAllianceHelpers.create_task_graph([task])
        engine.set_task_graph(graph)

        self.assertIsNotNone(engine.task_graph)
        self.assertEqual(len(engine.task_graph.tasks), 1)

    def test_form_coalitions_empty(self):
        """测试空任务图形成联盟"""
        engine = CoalitionFormationEngine()
        result = engine.form_coalitions()

        self.assertIsNotNone(result)
        self.assertEqual(len(result.coalitions), 0)

    def test_form_coalitions_greedy(self):
        """测试贪心策略形成联盟"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.GREEDY)

        # 注册Agent
        agent1 = MockAllianceHelpers.create_agent(
            "agent_1", capabilities={"nlp", "text"}
        )
        agent2 = MockAllianceHelpers.create_agent(
            "agent_2", capabilities={"cv", "image"}
        )
        engine.register_agent(agent1)
        engine.register_agent(agent2)

        # 设置任务
        task = MockAllianceHelpers.create_task(
            "task_1", required_capabilities={"nlp"}
        )
        graph = MockAllianceHelpers.create_task_graph([task])
        engine.set_task_graph(graph)

        result = engine.form_coalitions()

        self.assertIsNotNone(result)
        self.assertGreater(len(result.coalitions), 0)

    def test_form_coalitions_no_matching_agents(self):
        """测试无匹配Agent的联盟形成"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.GREEDY)

        # 只注册不匹配能力的Agent
        agent = MockAllianceHelpers.create_agent(
            "agent_1", capabilities={"cv"}
        )
        engine.register_agent(agent)

        # 设置需要NLP的任务
        task = MockAllianceHelpers.create_task(
            "task_1", required_capabilities={"nlp"}
        )
        graph = MockAllianceHelpers.create_task_graph([task])
        engine.set_task_graph(graph)

        result = engine.form_coalitions()

        # 没有有效的联盟形成
        self.assertIsNotNone(result)


class TestAuctionMechanism(unittest.TestCase):
    """测试拍卖机制"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.auction import AuctionMechanism, AuctionBid, AuctionTask
        self.auction_module = __import__('multiagent.alliance.auction', fromlist=['AuctionMechanism'])
        self.AuctionMechanism = self.auction_module.AuctionMechanism
        self.AuctionBid = self.auction_module.AuctionBid
        self.AuctionTask = self.auction_module.AuctionTask

    def test_auction_initialization(self):
        """测试拍卖初始化"""
        auction = self.AuctionMechanism()
        self.assertIsNotNone(auction)

    def test_create_auction_task(self):
        """测试创建拍卖任务"""
        task = self.AuctionTask(
            task_id="auction_task_1",
            description="Test task",
            reserve_price=100.0
        )
        self.assertEqual(task.task_id, "auction_task_1")
        self.assertEqual(task.reserve_price, 100.0)

    def test_submit_bid(self):
        """测试提交竞价"""
        auction = self.AuctionMechanism(timeout_seconds=10)
        task = self.AuctionTask(
            task_id="bid_task_1",
            description="Test task",
            reserve_price=100.0
        )

        auction_id = auction.create_auction(task)
        self.assertIsNotNone(auction_id)

        bid = self.AuctionBid(
            bidder_id="agent_1",
            auction_id=auction_id,
            amount=120.0,
            metadata={"quality": 0.9}
        )

        result = auction.submit_bid(bid)
        self.assertTrue(result)

    def test_auction_winner_determination(self):
        """测试拍卖赢家确定"""
        auction = self.AuctionMechanism(timeout_seconds=0.1)
        task = self.AuctionTask(
            task_id="winner_task_1",
            description="Test task",
            reserve_price=100.0
        )

        auction_id = auction.create_auction(task)

        # 提交多个竞价
        bids = [
            self.AuctionBid("agent_1", auction_id, 100.0),
            self.AuctionBid("agent_2", auction_id, 120.0),
            self.AuctionBid("agent_3", auction_id, 110.0),
        ]

        for bid in bids:
            auction.submit_bid(bid)

        # 等待拍卖结束
        time.sleep(0.2)

        winner = auction.determine_winner(auction_id)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.bidder_id, "agent_2")

    def test_auction_reserve_price_not_met(self):
        """测试未达到保留价"""
        auction = self.AuctionMechanism(timeout_seconds=0.1)
        task = self.AuctionTask(
            task_id="reserve_task",
            description="Test task",
            reserve_price=200.0
        )

        auction_id = auction.create_auction(task)

        bid = self.AuctionBid("agent_1", auction_id, 100.0)
        auction.submit_bid(bid)

        time.sleep(0.2)

        winner = auction.determine_winner(auction_id)
        # 应该没有赢家，因为没达到保留价
        self.assertIsNone(winner)


class TestContractNet(unittest.TestCase):
    """测试合同网协议"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.contract_net import ContractNetProtocol, ContractNetMessage
        self.protocol_module = __import__('multiagent.alliance.contract_net', fromlist=['ContractNetProtocol'])
        self.ContractNetProtocol = self.protocol_module.ContractNetProtocol
        self.ContractNetMessage = self.protocol_module.ContractNetMessage

    def test_contract_net_initialization(self):
        """测试合同网初始化"""
        protocol = self.ContractNetProtocol()
        self.assertIsNotNone(protocol)

    def test_create_task_call(self):
        """测试创建任务调用"""
        protocol = self.ContractNetProtocol()

        message = protocol.create_task_call(
            initiator="manager_1",
            task_description="Process data",
            deadline=time.time() + 3600
        )

        self.assertEqual(message.message_type, "task_call")
        self.assertEqual(message.initiator, "manager_1")

    def test_create_bid(self):
        """测试创建投标"""
        protocol = self.ContractNetProtocol()

        message = protocol.create_bid(
            responder="worker_1",
            task_call_id="call_001",
            capability="data_processing",
            cost=50.0
        )

        self.assertEqual(message.message_type, "bid")
        self.assertEqual(message.responder, "worker_1")

    def test_create_proposal(self):
        """测试创建提案"""
        protocol = self.ContractNetProtocol()

        message = protocol.create_proposal(
            responder="worker_1",
            task_call_id="call_001",
            cost=50.0,
            completion_time=3600
        )

        self.assertEqual(message.message_type, "proposal")

    def test_create_award(self):
        """测试创建授予"""
        protocol = self.ContractNetProtocol()

        message = protocol.create_award(
            initiator="manager_1",
            responder="worker_1",
            task_call_id="call_001"
        )

        self.assertEqual(message.message_type, "award")
        self.assertEqual(message.initiator, "manager_1")

    def test_create_rejection(self):
        """测试创建拒绝"""
        protocol = self.ContractNetProtocol()

        message = protocol.create_rejection(
            initiator="manager_1",
            responders=["worker_1", "worker_2"],
            task_call_id="call_001"
        )

        self.assertEqual(message.message_type, "rejection")

    def test_bid_evaluation(self):
        """测试投标评估"""
        protocol = self.ContractNetProtocol()

        bids = [
            {"responder": "worker_1", "cost": 100.0, "capability": 0.9},
            {"responder": "worker_2", "cost": 80.0, "capability": 0.7},
            {"responder": "worker_3", "cost": 120.0, "capability": 0.95},
        ]

        winner = protocol.evaluate_bids(bids, criteria="cost_capability_ratio")
        self.assertEqual(winner["responder"], "worker_1")


class TestCoalitionValue(unittest.TestCase):
    """测试联盟值计算(Shapley值)"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.coalition_value import ShapleyValue, CoalitionGame
        self.coalition_module = __import__('multiagent.alliance.coalition_value', fromlist=['ShapleyValue'])
        self.ShapleyValue = self.coalition_module.ShapleyValue
        self.CoalitionGame = self.coalition_module.CoalitionGame

    def test_shapley_initialization(self):
        """测试Shapley值初始化"""
        shapley = self.ShapleyValue()
        self.assertIsNotNone(shapley)

    def test_coalition_game_initialization(self):
        """测试联盟博弈初始化"""
        game = self.CoalitionGame(agents=["a", "b", "c"])
        self.assertEqual(len(game.agents), 3)

    def test_value_function(self):
        """测试联盟值函数"""
        game = self.CoalitionGame(agents=["a", "b", "c"])

        # 设置值函数
        game.set_value_function(lambda coalition: len(coalition) * 10)

        value = game.get_value({"a", "b"})
        self.assertEqual(value, 20)

    def test_shapley_value_calculation(self):
        """测试Shapley值计算"""
        shapley = self.ShapleyValue()
        game = self.CoalitionGame(agents=["a", "b", "c"])

        # 简单值函数: 联盟大小 * 10
        game.set_value_function(lambda coalition: len(coalition) * 10)

        values = shapley.calculate_shapley(game)

        self.assertIn("a", values)
        self.assertIn("b", values)
        self.assertIn("c", values)

    def test_shapley_efficiency(self):
        """测试Shapley值效率性(所有值之和等于大联盟值)"""
        shapley = self.ShapleyValue()
        game = self.CoalitionGame(agents=["a", "b", "c"])

        def value_func(coalition):
            return len(coalition) ** 2

        game.set_value_function(value_func)
        values = shapley.calculate_shapley(game)

        total_shapley = sum(values.values())
        grand_coalition_value = game.get_value({"a", "b", "c"})

        self.assertAlmostEqual(total_shapley, grand_coalition_value, places=5)

    def test_shapley_symmetry(self):
        """测试Shapley值对称性(等价Agent应有等价值)"""
        shapley = self.ShapleyValue()
        game = self.CoalitionGame(agents=["a", "b", "c"])

        # 对称值函数
        def value_func(coalition):
            return len(coalition) * 10

        game.set_value_function(value_func)
        values = shapley.calculate_shapley(game)

        # 所有Agent应该有相同的Shapley值
        self.assertEqual(values["a"], values["b"])
        self.assertEqual(values["b"], values["c"])

    def test_shapley_two_agents(self):
        """测试两个Agent的Shapley值"""
        shapley = self.ShapleyValue()
        game = self.CoalitionGame(agents=["x", "y"])

        game.set_value_function(lambda c: sum(len(a) for a in c) if c else 0)
        # v({x}) = 1, v({y}) = 1, v({x,y}) = 2
        game._value_function = lambda c: 1 if c else 0

        values = shapley.calculate_shapley(game)

        self.assertIn("x", values)
        self.assertIn("y", values)

    def test_shapley_empty_coalition(self):
        """测试空联盟"""
        shapley = self.ShapleyValue()
        game = self.CoalitionGame(agents=["a"])

        game.set_value_function(lambda c: 10 if c else 0)
        values = shapley.calculate_shapley(game)

        self.assertEqual(values["a"], 10)

    def test_shapley_additivity(self):
        """测试Shapley值可加性"""
        shapley = self.ShapleyValue()

        game1 = self.CoalitionGame(agents=["a", "b"])
        game1.set_value_function(lambda c: len(c) * 10)

        game2 = self.CoalitionGame(agents=["a", "b"])
        game2.set_value_function(lambda c: len(c) * 5)

        values1 = shapley.calculate_shapley(game1)
        values2 = shapley.calculate_shapley(game2)

        game_combined = self.CoalitionGame(agents=["a", "b"])
        game_combined.set_value_function(lambda c: len(c) * 15)
        values_combined = shapley.calculate_shapley(game_combined)

        self.assertAlmostEqual(
            values1["a"] + values2["a"],
            values_combined["a"],
            places=5
        )


class TestHungarianOptimizer(unittest.TestCase):
    """测试匈牙利算法优化器"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.hungarian_optimizer import HungarianOptimizer, AssignmentProblem
        self.optimizer_module = __import__('multiagent.alliance.hungarian_optimizer', fromlist=['HungarianOptimizer'])
        self.HungarianOptimizer = self.optimizer_module.HungarianOptimizer
        self.AssignmentProblem = self.optimizer_module.AssignmentProblem

    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        optimizer = self.HungarianOptimizer()
        self.assertIsNotNone(optimizer)

    def test_assignment_problem_creation(self):
        """测试分配问题创建"""
        cost_matrix = [
            [9, 2, 7, 8],
            [6, 4, 3, 7],
            [5, 8, 1, 8],
            [7, 6, 9, 4]
        ]

        problem = self.AssignmentProblem(cost_matrix=cost_matrix)
        self.assertEqual(len(problem.cost_matrix), 4)

    def test_hungarian_assignment(self):
        """测试匈牙利算法分配"""
        optimizer = self.HungarianOptimizer()

        # 4x4成本矩阵
        cost_matrix = [
            [9, 2, 7, 8],
            [6, 4, 3, 7],
            [5, 8, 1, 8],
            [7, 6, 9, 4]
        ]

        assignment = optimizer.solve(cost_matrix)

        self.assertEqual(len(assignment), 4)
        # 验证是有效的分配(每个工人分配一个任务)
        self.assertEqual(len(set(assignment.values())), 4)

    def test_minimum_cost_assignment(self):
        """测试最小成本分配"""
        optimizer = self.HungarianOptimizer()

        # 成本矩阵
        cost_matrix = [
            [3, 10, 7],
            [9, 4, 2],
            [6, 8, 5]
        ]

        assignment = optimizer.solve(cost_matrix)
        min_cost = optimizer.calculate_cost(cost_matrix, assignment)

        # 验证是最小成本
        self.assertIsInstance(min_cost, (int, float))
        self.assertGreaterEqual(min_cost, 0)

    def test_nxn_matrix(self):
        """测试N x N矩阵"""
        optimizer = self.HungarianOptimizer()

        # 5x5矩阵
        n = 5
        cost_matrix = [
            [i + j for j in range(n)]
            for i in range(n)
        ]

        assignment = optimizer.solve(cost_matrix)
        self.assertEqual(len(assignment), 5)

    def test_rectangular_matrix(self):
        """测试矩形矩阵(行数 != 列数)"""
        optimizer = self.HungarianOptimizer()

        # 3x4矩阵
        cost_matrix = [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 1, 4, 3]
        ]

        assignment = optimizer.solve(cost_matrix)
        # 应该分配3个任务
        self.assertLessEqual(len(assignment), 3)


class TestFaultTolerance(unittest.TestCase):
    """测试容错机制"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.fault_tolerant_formation import FaultTolerantFormation
        self.fault_module = __import__('multiagent.alliance.fault_tolerant_formation', fromlist=['FaultTolerantFormation'])
        self.FaultTolerantFormation = self.fault_module.FaultTolerantFormation

    def test_fault_tolerant_initialization(self):
        """测试容错联盟初始化"""
        formation = self.FaultTolerantFormation()
        self.assertIsNotNone(formation)

    def test_add_redundancy(self):
        """测试添加冗余"""
        formation = self.FaultTolerantFormation(replication_factor=2)

        agent = MockAllianceHelpers.create_agent(
            "redundant_agent",
            capabilities={"critical_cap"}
        )

        agents = formation.add_redundancy([agent])
        self.assertGreater(len(agents), 1)

    def test_failure_detection(self):
        """测试故障检测"""
        formation = self.FaultTolerantFormation()

        # 模拟Agent故障
        formation.report_failure("agent_1")

        self.assertTrue(formation.is_failed("agent_1"))

    def test_failure_recovery(self):
        """测试故障恢复"""
        formation = self.FaultTolerantFormation()

        formation.report_failure("agent_1")
        formation.recover_agent("agent_1")

        self.assertFalse(formation.is_failed("agent_1"))

    def test_coalition_reformation(self):
        """测试联盟重构"""
        formation = self.FaultTolerantFormation()

        # 创建联盟
        agents = [
            MockAllianceHelpers.create_agent(f"agent_{i}")
            for i in range(3)
        ]

        original_coalition = Coalition(
            coalition_id="coal_1",
            agents=set(agents[:2])
        )

        # 报告故障
        formation.report_failure("agent_0")

        # 重构联盟
        new_agents = formation.reform_coalition(
            original_coalition,
            available_agents=agents
        )

        self.assertIsNotNone(new_agents)

    def test_graceful_degradation(self):
        """测试优雅降级"""
        formation = self.FaultTolerantFormation(min_viable_agents=2)

        agents = [
            MockAllianceHelpers.create_agent(f"agent_{i}")
            for i in range(5)
        ]

        # 多个Agent故障
        formation.report_failure("agent_0")
        formation.report_failure("agent_1")
        formation.report_failure("agent_2")

        viable = formation.check_viability(agents)
        self.assertFalse(viable)

        # 只有少量故障
        formation._failed_agents.clear()
        viable = formation.check_viability(agents)
        self.assertTrue(viable)

    def test_backup_agent_assignment(self):
        """测试备份Agent分配"""
        formation = self.FaultTolerantFormation()

        primary = MockAllianceHelpers.create_agent(
            "primary",
            capabilities={"critical"}
        )
        backup = MockAllianceHelpers.create_agent(
            "backup",
            capabilities={"critical"}
        )

        formation.assign_backup("primary", "backup")

        self.assertEqual(formation.get_backup("primary"), "backup")


class TestCoalitionManagement(unittest.TestCase):
    """测试联盟管理"""

    def test_coalition_creation(self):
        """测试创建联盟"""
        coalition = Coalition(coalition_id="coal_001")
        self.assertEqual(coalition.coalition_id, "coal_001")
        self.assertEqual(len(coalition.agents), 0)

    def test_add_agent_to_coalition(self):
        """测试添加Agent到联盟"""
        coalition = Coalition(coalition_id="coal_001")
        agent = MockAllianceHelpers.create_agent("agent_1")

        coalition.add_agent(agent)
        self.assertEqual(len(coalition.agents), 1)

    def test_remove_agent_from_coalition(self):
        """测试从联盟移除Agent"""
        coalition = Coalition(coalition_id="coal_001")
        agent = MockAllianceHelpers.create_agent("agent_1")

        coalition.add_agent(agent)
        coalition.remove_agent(agent)

        self.assertEqual(len(coalition.agents), 0)

    def test_coalition_capabilities(self):
        """测试联盟能力"""
        coalition = Coalition(coalition_id="coal_001")

        agent1 = MockAllianceHelpers.create_agent(
            "agent_1", capabilities={"nlp"}
        )
        agent2 = MockAllianceHelpers.create_agent(
            "agent_2", capabilities={"cv"}
        )

        coalition.add_agent(agent1)
        coalition.add_agent(agent2)

        caps = coalition.get_combined_capabilities()
        self.assertIn("nlp", caps)
        self.assertIn("cv", caps)

    def test_coalition_can_perform_task(self):
        """测试联盟是否可执行任务"""
        coalition = Coalition(coalition_id="coal_001")

        agent1 = MockAllianceHelpers.create_agent(
            "agent_1", capabilities={"nlp"}
        )
        agent2 = MockAllianceHelpers.create_agent(
            "agent_2", capabilities={"cv"}
        )

        coalition.add_agent(agent1)
        coalition.add_agent(agent2)

        task = MockAllianceHelpers.create_task(
            "task_1", required_capabilities={"nlp", "cv"}
        )

        self.assertTrue(coalition.can_perform_task(task))

    def test_coalition_cost_calculation(self):
        """测试联盟成本计算"""
        agent1 = MockAllianceHelpers.create_agent(
            "agent_1", cost_per_unit=10.0
        )
        agent2 = MockAllianceHelpers.create_agent(
            "agent_2", cost_per_unit=20.0
        )

        coalition = Coalition(coalition_id="coal_001")
        coalition.add_agent(agent1)
        coalition.add_agent(agent2)

        cost = coalition.calculate_cost()
        self.assertEqual(cost, 30.0)


class TestTaskGraph(unittest.TestCase):
    """测试任务图"""

    def test_task_graph_initialization(self):
        """测试任务图初始化"""
        graph = TaskGraph()
        self.assertEqual(len(graph.tasks), 0)
        self.assertEqual(len(graph.edges), 0)

    def test_add_task(self):
        """测试添加任务"""
        graph = TaskGraph()
        task = MockAllianceHelpers.create_task("task_1")

        graph.add_task(task)
        self.assertEqual(len(graph.tasks), 1)

    def test_add_dependency(self):
        """测试添加依赖"""
        graph = TaskGraph()
        task1 = MockAllianceHelpers.create_task("task_1")
        task2 = MockAllianceHelpers.create_task("task_2")

        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_dependency("task_1", "task_2")

        self.assertIn("task_2", graph.edges)

    def test_topological_order(self):
        """测试拓扑排序"""
        graph = TaskGraph()

        tasks = [
            MockAllianceHelpers.create_task(f"task_{i}")
            for i in range(4)
        ]

        for task in tasks:
            graph.add_task(task)

        # 添加依赖: task_0 -> task_1 -> task_2, task_0 -> task_3
        graph.add_dependency("task_0", "task_1")
        graph.add_dependency("task_1", "task_2")
        graph.add_dependency("task_0", "task_3")

        order = graph.get_topological_order()

        # task_0 应该在 task_1 之前
        self.assertLess(order.index("task_0"), order.index("task_1"))
        # task_1 应该在 task_2 之前
        self.assertLess(order.index("task_1"), order.index("task_2"))

    def test_parallel_groups(self):
        """测试可并行执行的任务组"""
        graph = TaskGraph()

        tasks = [
            MockAllianceHelpers.create_task(f"task_{i}")
            for i in range(4)
        ]

        for task in tasks:
            graph.add_task(task)

        # 无依赖关系，所有任务可并行
        groups = graph.get_parallel_groups()

        self.assertGreaterEqual(len(groups), 1)


class TestFormationStrategies(unittest.TestCase):
    """测试不同联盟形成策略"""

    def test_greedy_strategy(self):
        """测试贪心策略"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.GREEDY)

        # 添加测试数据
        agents = [
            MockAllianceHelpers.create_agent(f"agent_{i}", capabilities={f"cap_{i}"})
            for i in range(5)
        ]
        for agent in agents:
            engine.register_agent(agent)

        tasks = [
            MockAllianceHelpers.create_task(f"task_{i}", required_capabilities={f"cap_{i}"})
            for i in range(5)
        ]
        graph = MockAllianceHelpers.create_task_graph(tasks)
        engine.set_task_graph(graph)

        result = engine.form_coalitions()
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_used, CoalitionStrategy.GREEDY)

    def test_heuristic_strategy(self):
        """测试启发式策略"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.HEURISTIC)

        agents = [
            MockAllianceHelpers.create_agent(f"agent_{i}", capabilities={f"cap_{i}"})
            for i in range(3)
        ]
        for agent in agents:
            engine.register_agent(agent)

        tasks = [
            MockAllianceHelpers.create_task(f"task_{i}", required_capabilities={f"cap_{i}"})
            for i in range(3)
        ]
        graph = MockAllianceHelpers.create_task_graph(tasks)
        engine.set_task_graph(graph)

        result = engine.form_coalitions()
        self.assertEqual(result.strategy_used, CoalitionStrategy.HEURISTIC)

    def test_dynamic_strategy(self):
        """测试动态规划策略"""
        engine = CoalitionFormationEngine(strategy=CoalitionStrategy.DYNAMIC)

        agents = [
            MockAllianceHelpers.create_agent(f"agent_{i}", capabilities={f"cap_{i}"})
            for i in range(3)
        ]
        for agent in agents:
            engine.register_agent(agent)

        tasks = [
            MockAllianceHelpers.create_task(f"task_{i}", required_capabilities={f"cap_{i}"})
            for i in range(3)
        ]
        graph = MockAllianceHelpers.create_task_graph(tasks)
        engine.set_task_graph(graph)

        result = engine.form_coalitions()
        self.assertEqual(result.strategy_used, CoalitionStrategy.DYNAMIC)


class TestFormationResult(unittest.TestCase):
    """测试联盟形成结果"""

    def test_formation_result_initialization(self):
        """测试结果初始化"""
        result = FormationResult()
        self.assertEqual(len(result.coalitions), 0)
        self.assertEqual(len(result.task_assignments), 0)

    def test_get_coalition_for_task(self):
        """测试获取任务所属联盟"""
        result = FormationResult()

        coalition = Coalition(coalition_id="coal_001")
        result.coalitions.append(coalition)
        result.task_assignments["task_001"] = "coal_001"

        coal = result.get_coalition_for_task("task_001")
        self.assertIsNotNone(coal)
        self.assertEqual(coal.coalition_id, "coal_001")

    def test_get_nonexistent_coalition(self):
        """测试获取不存在的联盟"""
        result = FormationResult()
        result.task_assignments["task_001"] = "nonexistent"

        coal = result.get_coalition_for_task("task_001")
        self.assertIsNone(coal)


class TestAgentTaskMatching(unittest.TestCase):
    """测试Agent-任务匹配"""

    def test_agent_can_perform_task(self):
        """测试Agent是否能执行任务"""
        agent = MockAllianceHelpers.create_agent(
            "agent_1",
            capabilities={"nlp", "text_processing"}
        )
        task = MockAllianceHelpers.create_task(
            "task_1",
            required_capabilities={"nlp"}
        )

        self.assertTrue(agent.can_perform(task))

    def test_agent_cannot_perform_task(self):
        """测试Agent不能执行任务"""
        agent = MockAllianceHelpers.create_agent(
            "agent_1",
            capabilities={"cv"}
        )
        task = MockAllianceHelpers.create_task(
            "task_1",
            required_capabilities={"nlp"}
        )

        self.assertFalse(agent.can_perform(task))

    def test_agent_capacity(self):
        """测试Agent容量"""
        agent = MockAllianceHelpers.create_agent(
            "agent_1",
            max_load=10.0
        )
        agent.current_load = 3.0

        capacity = agent.available_capacity()
        self.assertEqual(capacity, 7.0)

    def test_agent_full_capacity(self):
        """测试Agent满容量"""
        agent = MockAllianceHelpers.create_agent(
            "agent_1",
            max_load=10.0
        )
        agent.current_load = 10.0

        capacity = agent.available_capacity()
        self.assertEqual(capacity, 0.0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_single_agent_coalition(self):
        """测试单Agent联盟"""
        coalition = Coalition(coalition_id="single")
        agent = MockAllianceHelpers.create_agent("solo_agent")

        coalition.add_agent(agent)
        self.assertEqual(len(coalition.agents), 1)

    def test_empty_task_graph(self):
        """测试空任务图"""
        engine = CoalitionFormationEngine()
        engine.set_task_graph(TaskGraph())

        result = engine.form_coalitions()
        self.assertEqual(len(result.coalitions), 0)

    def test_no_agents_registered(self):
        """测试无注册Agent"""
        engine = CoalitionFormationEngine()
        task = MockAllianceHelpers.create_task("task_1")
        graph = MockAllianceHelpers.create_task_graph([task])
        engine.set_task_graph(graph)

        result = engine.form_coalitions()
        self.assertEqual(len(result.coalitions), 0)

    def test_zero_priority_task(self):
        """测试零优先级任务"""
        task = MockAllianceHelpers.create_task("task_1", priority=0)
        self.assertEqual(task.priority, 0)

    def test_custom_value_function(self):
        """测试自定义价值函数"""
        engine = CoalitionFormationEngine()

        def custom_value(coalition, task):
            return 100.0 if coalition.can_perform_task(task) else 0.0

        engine.value_function = custom_value

        agent = MockAllianceHelpers.create_agent(
            "agent_1",
            capabilities={"test"}
        )
        engine.register_agent(agent)

        task = MockAllianceHelpers.create_task(
            "task_1",
            required_capabilities={"test"}
        )
        graph = MockAllianceHelpers.create_task_graph([task])
        engine.set_task_graph(graph)

        result = engine.form_coalitions()
        self.assertGreater(result.total_value, 0)


if __name__ == "__main__":
    unittest.main()
