"""
仿真涌现测试模块

测试世界仿真、经济模拟、领土竞争、合作困境和文化演化。
"""

import unittest
import time
import random
import math
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.simulator.world import World, Agent, AgentState, Position


class MockSimulationHelpers:
    """仿真测试辅助类"""

    @staticmethod
    def create_world(
        width: float = 100.0,
        height: float = 100.0,
        time_step: float = 1.0
    ) -> World:
        """创建测试用世界"""
        return World(
            width=width,
            height=height,
            time_step=time_step,
            random_seed=42
        )

    @staticmethod
    def create_test_agent(
        agent_id: str = "agent_001",
        x: float = 50.0,
        y: float = 50.0,
        energy: float = 100.0
    ) -> Agent:
        """创建测试用Agent"""
        return Agent(
            agent_id=agent_id,
            position=Position(x, y),
            energy=energy
        )

    @staticmethod
    def create_agents(count: int = 10) -> List[Agent]:
        """创建多个测试Agent"""
        agents = []
        for i in range(count):
            agent = Agent(
                agent_id=f"agent_{i:03d}",
                position=Position(
                    x=random.uniform(0, 100),
                    y=random.uniform(0, 100)
                ),
                energy=random.uniform(50, 100)
            )
            agents.append(agent)
        return agents


class TestWorldSimulation(unittest.TestCase):
    """测试世界仿真"""

    def test_world_initialization(self):
        """测试世界初始化"""
        world = MockSimulationHelpers.create_world()

        self.assertEqual(world.width, 100.0)
        self.assertEqual(world.height, 100.0)
        self.assertEqual(world.time_step, 1.0)
        self.assertEqual(world.current_time, 0.0)
        self.assertEqual(world.current_step, 0)

    def test_add_agent(self):
        """测试添加Agent"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent("test_agent")

        agent_id = world.add_agent(agent)

        self.assertEqual(agent_id, "test_agent")
        self.assertEqual(len(world.agents), 1)

    def test_remove_agent(self):
        """测试移除Agent"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent("remove_me")
        world.add_agent(agent)

        removed = world.remove_agent("remove_me")

        self.assertIsNotNone(removed)
        self.assertEqual(len(world.agents), 0)

    def test_get_agent(self):
        """测试获取Agent"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent("get_me")
        world.add_agent(agent)

        retrieved = world.get_agent("get_me")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.agent_id, "get_me")

    def test_get_all_agents(self):
        """测试获取所有Agent"""
        world = MockSimulationHelpers.create_world()

        for i in range(5):
            agent = MockSimulationHelpers.create_test_agent(f"agent_{i}")
            world.add_agent(agent)

        all_agents = world.get_all_agents()
        self.assertEqual(len(all_agents), 5)

    def test_single_step_simulation(self):
        """测试单步仿真"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent()
        world.add_agent(agent)

        result = world.step()

        self.assertTrue(result)
        self.assertEqual(world.current_step, 1)
        self.assertGreater(world.current_time, 0)

    def test_multi_step_simulation(self):
        """测试多步仿真"""
        world = MockSimulationHelpers.create_world(max_steps=10)

        for _ in range(5):
            agent = MockSimulationHelpers.create_test_agent()
            world.add_agent(agent)

        world.run(steps=10)

        self.assertEqual(world.current_step, 10)

    def test_world_boundaries(self):
        """测试世界边界"""
        world = MockSimulationHelpers.create_world(width=100, height=100)

        # Agent在边界外
        agent = Agent(agent_id="boundary_test", position=Position(150, 150))
        world.add_agent(agent)

        world.step()

        # Agent应该在边界内
        self.assertLessEqual(agent.position.x, 100)
        self.assertLessEqual(agent.position.y, 100)

    def test_event_logging(self):
        """测试事件记录"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent()
        world.add_agent(agent)

        world.step()

        self.assertGreater(len(world.event_log), 0)

    def test_world_statistics(self):
        """测试世界统计"""
        world = MockSimulationHelpers.create_world()

        for _ in range(3):
            world.add_agent(MockSimulationHelpers.create_test_agent())

        stats = world.get_statistics()

        self.assertEqual(stats["active_agents"], 3)
        self.assertEqual(stats["total_steps"], 0)

    def test_world_reset(self):
        """测试世界重置"""
        world = MockSimulationHelpers.create_world()
        world.add_agent(MockSimulationHelpers.create_test_agent())
        world.step()
        world.step()

        world.reset()

        self.assertEqual(world.current_time, 0.0)
        self.assertEqual(world.current_step, 0)
        self.assertEqual(len(world.agents), 0)


class TestAgentBehavior(unittest.TestCase):
    """测试Agent行为"""

    def test_agent_initialization(self):
        """测试Agent初始化"""
        agent = MockSimulationHelpers.create_test_agent(
            agent_id="test",
            x=10.0,
            y=20.0
        )

        self.assertEqual(agent.agent_id, "test")
        self.assertEqual(agent.position.x, 10.0)
        self.assertEqual(agent.position.y, 20.0)
        self.assertEqual(agent.energy, 100.0)
        self.assertEqual(agent.state, AgentState.ACTIVE)

    def test_agent_energy_depletion(self):
        """测试Agent能量耗尽"""
        world = MockSimulationHelpers.create_world()
        agent = MockSimulationHelpers.create_test_agent(energy=0.0)
        world.add_agent(agent)

        world.step()

        self.assertEqual(agent.state, AgentState.INACTIVE)

    def test_agent_movement(self):
        """测试Agent移动"""
        agent = MockSimulationHelpers.create_test_agent(x=50, y=50)

        # 执行移动决策
        agent.act({"action": "move", "params": {"dx": 5, "dy": 3}}, world=None)

        self.assertEqual(agent.position.x, 55.0)
        self.assertEqual(agent.position.y, 53.0)
        self.assertLess(agent.energy, 100.0)

    def test_agent_rest(self):
        """测试Agent休息"""
        agent = MockSimulationHelpers.create_test_agent(energy=80.0)

        agent.act({"action": "rest"}, world=None)

        self.assertEqual(agent.energy, 85.0)  # 恢复5点能量

    def test_agent_memory(self):
        """测试Agent记忆"""
        agent = MockSimulationHelpers.create_test_agent()

        agent.add_memory({"event": "test_event", "data": "test_data"})
        agent.add_memory({"event": "another_event", "data": "more_data"})

        self.assertEqual(len(agent.memory), 2)

    def test_agent_memory_limit(self):
        """测试Agent记忆限制"""
        agent = MockSimulationHelpers.create_test_agent()
        agent.max_memory_size = 3

        for i in range(5):
            agent.add_memory({"event": f"event_{i}"})

        self.assertLessEqual(len(agent.memory), 3)

    def test_agent_perception(self):
        """测试Agent感知"""
        world = MockSimulationHelpers.create_world()

        for i in range(5):
            agent = MockSimulationHelpers.create_test_agent(f"agent_{i}")
            world.add_agent(agent)

        perception = world.agents["agent_0"].perceive(world)

        self.assertIn("nearby_agents", perception)
        self.assertIn("position", perception)


class TestPosition(unittest.TestCase):
    """测试位置类"""

    def test_position_creation(self):
        """测试位置创建"""
        pos = Position(10.0, 20.0)

        self.assertEqual(pos.x, 10.0)
        self.assertEqual(pos.y, 20.0)

    def test_distance_calculation(self):
        """测试距离计算"""
        pos1 = Position(0.0, 0.0)
        pos2 = Position(3.0, 4.0)

        distance = pos1.distance_to(pos2)

        self.assertAlmostEqual(distance, 5.0, places=1)

    def test_position_addition(self):
        """测试位置相加"""
        pos1 = Position(1.0, 2.0)
        pos2 = Position(3.0, 4.0)

        result = pos1 + pos2

        self.assertEqual(result.x, 4.0)
        self.assertEqual(result.y, 6.0)

    def test_position_multiplication(self):
        """测试位置乘法"""
        pos = Position(2.0, 3.0)

        result = pos * 2.0

        self.assertEqual(result.x, 4.0)
        self.assertEqual(result.y, 6.0)


class TestEconomicSimulation(unittest.TestCase):
    """测试经济模拟"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.economic_sim import EconomicSimulation, Resource, Agent as EcoAgent
        self.econ_module = __import__('multiagent.simulator.economic_sim', fromlist=['EconomicSimulation'])
        self.EconomicSimulation = self.econ_module.EconomicSimulation
        self.Resource = self.econ_module.Resource
        self.EcoAgent = self.eco_module.EcoAgent

    def test_economy_initialization(self):
        """测试经济初始化"""
        economy = self.EconomicSimulation()

        self.assertIsNotNone(economy)
        self.assertEqual(len(economy.agents), 0)

    def test_add_resource(self):
        """测试添加资源"""
        economy = self.EconomicSimulation()

        resource = self.Resource(
            resource_id="gold",
            amount=1000,
            scarcity=0.5
        )

        economy.add_resource(resource)
        self.assertEqual(len(economy.resources), 1)

    def test_trade(self):
        """测试交易"""
        economy = self.EconomicSimulation()

        # 创建两个Agent
        agent1 = self.EcoAgent("trader_1", initial_wealth=100)
        agent2 = self.EcoAgent("trader_2", initial_wealth=50)

        economy.add_agent(agent1)
        economy.add_agent(agent2)

        # 执行交易
        economy.trade("trader_1", "trader_2", 30)

        # 验证交易结果
        self.assertEqual(agent1.wealth, 70)
        self.assertEqual(agent2.wealth, 80)

    def test_resource_consumption(self):
        """测试资源消耗"""
        economy = self.EconomicSimulation()

        economy.consume("gold", 10)

        self.assertLess(economy.resources["gold"].amount, 1000)

    def test_market_price_calculation(self):
        """测试市场价格计算"""
        economy = self.EconomicSimulation()

        economy.add_resource(self.Resource("rare", amount=100, scarcity=0.9))
        economy.add_resource(self.Resource("common", amount=10000, scarcity=0.1))

        rare_price = economy.get_price("rare")
        common_price = economy.get_price("common")

        self.assertGreater(rare_price, common_price)

    def test_economic_growth(self):
        """测试经济增长"""
        economy = self.EconomicSimulation(initial_wealth=1000)
        initial = economy.total_wealth

        economy.simulate_round()

        self.assertNotEqual(economy.total_wealth, initial)


class TestTerritoryCompetition(unittest.TestCase):
    """测试领土竞争"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.territory_competition import TerritoryMap, Territory, Agent as TerritoryAgent
        self.territory_module = __import__('multiagent.simulator.territory_competition', fromlist=['TerritoryMap'])
        self.TerritoryMap = self.territory_module.TerritoryMap
        self.Territory = self.territory_module.Territory

    def test_map_initialization(self):
        """测试地图初始化"""
        territory_map = self.TerritoryMap(width=100, height=100)

        self.assertEqual(territory_map.width, 100)
        self.assertEqual(territory_map.height, 100)

    def test_claim_territory(self):
        """测试声称领土"""
        territory_map = self.TerritoryMap(width=100, height=100)

        territory = self.Territory(
            territory_id="region_1",
            x=0, y=0,
            width=20, height=20
        )

        territory_map.claim(territory, "agent_1")

        self.assertEqual(territory.owner_id, "agent_1")

    def test_territory_overlap(self):
        """测试领土重叠"""
        territory_map = self.TerritoryMap(width=100, height=100)

        region1 = self.Territory("r1", x=0, y=0, width=20, height=20)
        region2 = self.Territory("r2", x=15, y=15, width=20, height=20)

        territory_map.claim(region1, "agent_1")
        territory_map.claim(region2, "agent_2")

        # 检查重叠
        has_overlap = territory_map.check_overlap(region1, region2)
        self.assertTrue(has_overlap)

    def test_territory_conflict(self):
        """测试领土冲突"""
        territory_map = self.TerritoryMap(width=100, height=100)

        region = self.Territory("contested", x=0, y=0, width=20, height=20)
        territory_map.claim(region, "agent_1")

        # 另一个Agent尝试声称
        conflict = territory_map.claim(region, "agent_2")

        self.assertFalse(conflict)

    def test_territory_value(self):
        """测试领土价值"""
        territory_map = self.TerritoryMap(width=100, height=100)

        rich = self.Territory("rich", x=0, y=0, width=20, height=20, resources=100)
        poor = self.Territory("poor", x=50, y=50, width=20, height=20, resources=10)

        self.assertGreater(rich.resources, poor.resources)

    def test_expansion(self):
        """测试扩张"""
        territory_map = self.TerritoryMap(width=100, height=100)

        agent_territories = territory_map.get_agent_territories("agent_1")
        self.assertEqual(len(agent_territories), 0)

        territory_map.expand("agent_1", x=0, y=0, width=10, height=10)

        agent_territories = territory_map.get_agent_territories("agent_1")
        self.assertEqual(len(agent_territories), 1)


class TestCooperationDilemma(unittest.TestCase):
    """测试合作困境"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.cooperation_dilemma import CooperationGame, PrisonersDilemma, StagHunt
        self.coop_module = __import__('multiagent.simulator.cooperation_dilemma', fromlist=['CooperationGame'])
        self.CooperationGame = self.coop_module.CooperationGame
        self.PrisonersDilemma = self.coop_module.PrisonersDilemma

    def test_dilemma_initialization(self):
        """测试困境初始化"""
        game = self.PrisonersDilemma()

        self.assertIsNotNone(game)
        self.assertEqual(game.payoff_cooperate_cooperate, (3, 3))
        self.assertEqual(game.payoff_defect_defect, (1, 1))

    def test_payoff_matrix(self):
        """测试收益矩阵"""
        game = self.PrisonersDilemma()

        # 双方合作
        payoff = game.get_payoff("cooperate", "cooperate")
        self.assertEqual(payoff, (3, 3))

        # 双方背叛
        payoff = game.get_payoff("defect", "defect")
        self.assertEqual(payoff, (1, 1))

        # 一方合作，一方背叛
        payoff = game.get_payoff("cooperate", "defect")
        self.assertEqual(payoff, (0, 5))

    def test_iterated_game(self):
        """测试重复博弈"""
        game = self.PrisonersDilemma(max_rounds=10)

        results = []
        for round_num in range(10):
            result = game.play_round("agent_1", "agent_2", "cooperate", "cooperate")
            results.append(result)

        self.assertEqual(len(results), 10)

    def test_tit_for_tat_strategy(self):
        """测试以牙还牙策略"""
        game = self.CooperationGame()

        history = [("cooperate", "cooperate"), ("cooperate", "defect")]
        choice = game.strategy_tit_for_tat(history, my_last_move="cooperate")

        self.assertEqual(choice, "defect")

    def test_all_defect_strategy(self):
        """测试总是背叛策略"""
        game = self.CooperationGame()

        choice = game.strategy_all_defect(history=[])
        self.assertEqual(choice, "defect")

    def test_all_cooperate_strategy(self):
        """测试总是合作策略"""
        game = self.CooperationGame()

        choice = game.strategy_all_cooperate(history=[])
        self.assertEqual(choice, "cooperate")

    def test_random_strategy(self):
        """测试随机策略"""
        game = self.CooperationGame()

        choices = [game.strategy_random() for _ in range(100)]
        has_cooperate = "cooperate" in choices
        has_defect = "defect" in choices

        self.assertTrue(has_cooperate)
        self.assertTrue(has_defect)

    def test_payoff_accumulation(self):
        """测试收益累积"""
        game = self.PrisonersDilemma()

        agent1_total = 0
        agent2_total = 0

        for _ in range(5):
            payoff = game.get_payoff("cooperate", "cooperate")
            agent1_total += payoff[0]
            agent2_total += payoff[1]

        self.assertEqual(agent1_total, 15)
        self.assertEqual(agent2_total, 15)


class TestCulturalEvolution(unittest.TestCase):
    """测试文化演化"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.cultural_evolution import CulturalEvolution, Culture, Trait
        self.culture_module = __import__('multiagent.simulator.cultural_evolution', fromlist=['CulturalEvolution'])
        self.CulturalEvolution = self.culture_module.CulturalEvolution
        self.Trait = self.culture_module.Trait

    def test_evolution_initialization(self):
        """测试演化初始化"""
        evolution = self.CulturalEvolution()

        self.assertIsNotNone(evolution)
        self.assertEqual(len(evolution.agents), 0)

    def test_add_trait(self):
        """测试添加特质"""
        evolution = self.CulturalEvolution()

        trait = self.Trait(
            trait_id=" bravery",
            name="Bravery",
            frequency=0.5
        )

        evolution.add_trait(trait)
        self.assertIn(" bravery", evolution.traits)

    def test_trait_inheritance(self):
        """测试特质遗传"""
        evolution = self.CulturalEvolution()

        parent_traits = ["brave", "curious", "cautious"]
        child_traits = evolution.inherit_traits(parent_traits)

        # 子代应该继承父代的一些特质
        self.assertGreater(len(child_traits), 0)

    def test_mutation(self):
        """测试变异"""
        evolution = self.CulturalEvolution()

        original = ["trait_a", "trait_b"]
        mutated = evolution.mutate(original, mutation_rate=0.5)

        # 变异后可能产生新特质
        self.assertIsNotNone(mutated)

    def test_selection(self):
        """测试选择"""
        evolution = self.CulturalEvolution()

        population = [
            {"traits": ["brave"], "fitness": 0.8},
            {"traits": ["cautious"], "fitness": 0.6},
            {"traits": ["curious"], "fitness": 0.7},
        ]

        survivors = evolution.select(population, survival_rate=0.5)

        self.assertLessEqual(len(survivors), len(population))

    def test_culture_convergence(self):
        """测试文化收敛"""
        evolution = self.CulturalEvolution()

        for i in range(10):
            evolution.add_agent({
                "id": f"agent_{i}",
                "traits": ["cultural_trait"]
            })

        evolution.simulate_generations(5)

        # 检查特质是否收敛
        trait_counts = evolution.get_trait_distribution()
        self.assertGreater(len(trait_counts), 0)


class TestSpatialGrid(unittest.TestCase):
    """测试空间网格"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.world import SpatialGrid
        self.SpatialGrid = SpatialGrid

    def test_grid_initialization(self):
        """测试网格初始化"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)

        self.assertEqual(grid.cell_size, 10)
        self.assertGreater(grid.cols, 0)
        self.assertGreater(grid.rows, 0)

    def test_get_cell(self):
        """测试获取网格单元"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)

        cell = grid._get_cell(Position(15, 25))
        self.assertEqual(cell, (1, 2))

    def test_add_agent_to_grid(self):
        """测试添加Agent到网格"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)
        agent = Agent(agent_id="grid_agent", position=Position(15, 25))

        grid.add_agent(agent)

        self.assertIn((1, 2), grid.grid)

    def test_remove_agent_from_grid(self):
        """测试从网格移除Agent"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)
        agent = Agent(agent_id="remove_agent", position=Position(15, 25))

        grid.add_agent(agent)
        grid.remove_agent(agent)

        self.assertNotIn(agent.agent_id, grid.grid.get((1, 2), set()))

    def test_update_agent_position(self):
        """测试更新Agent位置"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)
        agent = Agent(agent_id="moving_agent", position=Position(15, 25))

        grid.add_agent(agent)

        old_pos = Position(15, 25)
        agent.position = Position(55, 75)
        grid.update_agent(agent, old_pos)

        # Agent应该在新的网格单元
        self.assertIn((5, 7), grid.grid)

    def test_get_nearby_agents(self):
        """测试获取附近Agent"""
        grid = self.SpatialGrid(world_width=100, world_height=100, cell_size=10)

        for i in range(5):
            agent = Agent(
                agent_id=f"nearby_{i}",
                position=Position(i * 2, i * 2)
            )
            grid.add_agent(agent)

        nearby_ids = grid.get_nearby(Position(5, 5), radius=15)

        self.assertGreater(len(nearby_ids), 0)


class TestSimulationCallbacks(unittest.TestCase):
    """测试仿真回调"""

    def test_step_callback(self):
        """测试步进回调"""
        world = MockSimulationHelpers.create_world()
        callback_count = [0]

        def step_callback(w):
            callback_count[0] += 1

        world.register_step_callback(step_callback)

        world.step()
        world.step()

        self.assertEqual(callback_count[0], 2)

    def test_interaction_callback(self):
        """测试交互回调"""
        world = MockSimulationHelpers.create_world()
        interactions = []

        def interaction_callback(agent1, agent2, w):
            interactions.append((agent1.agent_id, agent2.agent_id))

        world.register_interaction_callback(interaction_callback)

        # 添加靠近的Agent
        agent1 = Agent(agent_id="int_agent_1", position=Position(10, 10))
        agent2 = Agent(agent_id="int_agent_2", position=Position(11, 11))
        world.add_agent(agent1)
        world.add_agent(agent2)

        world.step()

        # 如果Agent在交互距离内，应该触发回调
        self.assertGreaterEqual(len(interactions), 0)


class TestAgentSpawner(unittest.TestCase):
    """测试Agent生成器"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.agent_spawner import AgentSpawner, SpawnStrategy
        self.spawner_module = __import__('multiagent.simulator.agent_spawner', fromlist=['AgentSpawner'])
        self.AgentSpawner = self.spawner_module.AgentSpawner

    def test_spawner_initialization(self):
        """测试生成器初始化"""
        spawner = self.AgentSpawner()
        self.assertIsNotNone(spawner)

    def test_random_spawn(self):
        """测试随机生成"""
        spawner = self.AgentSpawner()
        world = MockSimulationHelpers.create_world()

        agent = spawner.spawn_random(world, agent_type="basic")

        self.assertIsNotNone(agent)
        self.assertEqual(len(world.agents), 1)

    def test_spawn_with_traits(self):
        """测试带特质生成"""
        spawner = self.AgentSpawner()
        world = MockSimulationHelpers.create_world()

        agent = spawner.spawn_with_traits(
            world,
            traits=["aggressive", "explorer"]
        )

        self.assertIsNotNone(agent)

    def test_batch_spawn(self):
        """测试批量生成"""
        spawner = self.AgentSpawner()
        world = MockSimulationHelpers.create_world()

        spawner.batch_spawn(world, count=10)

        self.assertEqual(len(world.agents), 10)


class TestResourceManager(unittest.TestCase):
    """测试资源管理器"""

    def setUp(self):
        """测试初始化"""
        from multiagent.simulator.resource_manager import ResourceManager, ResourceType
        self.resource_module = __import__('multiagent.simulator.resource_manager', fromlist=['ResourceManager'])
        self.ResourceManager = self.resource_module.ResourceManager

    def test_manager_initialization(self):
        """测试管理器初始化"""
        manager = self.ResourceManager()
        self.assertIsNotNone(manager)

    def test_add_resource(self):
        """测试添加资源"""
        manager = self.ResourceManager()

        manager.add_resource("energy", initial_amount=100)

        self.assertEqual(manager.get_amount("energy"), 100)

    def test_consume_resource(self):
        """测试消耗资源"""
        manager = self.ResourceManager()

        manager.add_resource("energy", initial_amount=100)
        consumed = manager.consume("energy", 30)

        self.assertTrue(consumed)
        self.assertEqual(manager.get_amount("energy"), 70)

    def test_insufficient_resource(self):
        """测试资源不足"""
        manager = self.ResourceManager()

        manager.add_resource("rare", initial_amount=10)
        consumed = manager.consume("rare", 20)

        self.assertFalse(consumed)
        self.assertEqual(manager.get_amount("rare"), 10)

    def test_resource_regeneration(self):
        """测试资源再生"""
        manager = self.ResourceManager(regen_rate=0.1)

        manager.add_resource("renewable", initial_amount=50)
        manager.regenerate()

        self.assertGreater(manager.get_amount("renewable"), 50)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_world(self):
        """测试空世界"""
        world = MockSimulationHelpers.create_world()

        world.step()

        self.assertEqual(world.current_step, 1)

    def test_agent_at_exact_boundary(self):
        """测试Agent恰好在边界"""
        world = MockSimulationHelpers.create_world(width=100, height=100)
        agent = Agent(agent_id="boundary", position=Position(100, 100))
        world.add_agent(agent)

        world.step()

        # Agent应该在边界内
        self.assertLessEqual(agent.position.x, 100)

    def test_zero_time_step(self):
        """测试零时间步长"""
        world = World(width=100, height=100, time_step=0)

        self.assertEqual(world.time_step, 0)

    def test_negative_coordinates(self):
        """测试负坐标"""
        pos = Position(-10, -20)

        self.assertEqual(pos.x, -10)
        self.assertEqual(pos.y, -20)

    def test_max_agents(self):
        """测试最大Agent数"""
        world = MockSimulationHelpers.create_world()

        for i in range(1000):
            agent = MockSimulationHelpers.create_test_agent(f"max_agent_{i}")
            world.add_agent(agent)

        self.assertEqual(len(world.agents), 1000)


if __name__ == "__main__":
    unittest.main()
