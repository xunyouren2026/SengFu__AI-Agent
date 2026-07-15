"""
自组织测试模块

测试Agent工厂、领导选举、退休、克隆、合并、Agent教练和动态重规划。
"""

import unittest
import time
import random
import threading
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.meta.agent_factory import (
    DynamicAgentFactory,
    AgentType,
    AgentConfiguration,
    TaskFeature,
    AgentTemplate,
    CapabilityLevel
)


class MockSelfOrgHelpers:
    """自组织测试辅助类"""

    @staticmethod
    def create_factory() -> DynamicAgentFactory:
        """创建测试用Agent工厂"""
        return DynamicAgentFactory()

    @staticmethod
    def create_task_feature(
        domain: str = "software",
        complexity: float = 0.5,
        skills: Optional[Set[str]] = None
    ) -> TaskFeature:
        """创建测试用任务特征"""
        return TaskFeature(
            domain=domain,
            complexity=complexity,
            required_skills=skills or {"python"},
            estimated_duration=3600,
            data_intensity=0.5,
            creativity_required=0.3,
            precision_required=0.7,
            collaboration_required=0.4
        )


class TestAgentFactory(unittest.TestCase):
    """测试Agent工厂"""

    def test_factory_initialization(self):
        """测试工厂初始化"""
        factory = MockSelfOrgHelpers.create_factory()

        self.assertIsNotNone(factory)
        self.assertGreater(len(factory.templates), 0)

    def test_register_template(self):
        """测试注册模板"""
        factory = MockSelfOrgHelpers.create_factory()

        template = AgentTemplate(
            template_id="custom_template",
            name="CustomAgent",
            agent_type=AgentType.SPECIALIZED,
            base_capabilities={"custom_skill": CapabilityLevel.INTERMEDIATE},
            knowledge_domains={"custom_domain"},
            description_pattern="A custom agent",
            feature_match_weights={},
            configuration_generator=lambda f: {}
        )

        factory.register_template(template)

        self.assertIn("custom_template", factory.templates)

    def test_generate_agent(self):
        """测试生成Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        config = factory.generate_agent(
            "Create a Python web scraper with error handling"
        )

        self.assertIsNotNone(config)
        self.assertIsNotNone(config.agent_id)

    def test_generate_agent_with_custom_id(self):
        """测试使用自定义ID生成Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        config = factory.generate_agent(
            "Test task",
            custom_id="my_custom_agent"
        )

        self.assertEqual(config.agent_id, "my_custom_agent")

    def test_generate_multiple_agents(self):
        """测试生成多个Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        configs = []
        for i in range(5):
            config = factory.generate_agent(f"Task {i}")
            configs.append(config)

        self.assertEqual(len(configs), 5)

    def test_generate_hybrid_agent(self):
        """测试生成混合型Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        config = factory.generate_hybrid_agent([
            "Write Python code",
            "Create data visualizations"
        ])

        self.assertIsNotNone(config)
        self.assertEqual(config.agent_type, AgentType.HYBRID)

    def test_get_agent_config(self):
        """测试获取Agent配置"""
        factory = MockSelfOrgHelpers.create_factory()

        generated = factory.generate_agent("Test task")
        retrieved = factory.get_agent_config(generated.agent_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.agent_id, generated.agent_id)

    def test_list_generated_agents(self):
        """测试列出生成的Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        for i in range(3):
            factory.generate_agent(f"Task {i}")

        agents = factory.list_generated_agents()
        self.assertEqual(len(agents), 3)

    def test_update_agent_config(self):
        """测试更新Agent配置"""
        factory = MockSelfOrgHelpers.create_factory()

        config = factory.generate_agent("Original task")
        original_version = config.version

        updated = factory.update_agent_config(
            config.agent_id,
            {"description": "Updated description"}
        )

        self.assertIsNotNone(updated)
        self.assertGreater(updated.version, original_version)

    def test_clone_agent_config(self):
        """测试克隆Agent配置"""
        factory = MockSelfOrgHelpers.create_factory()

        original = factory.generate_agent("Original task")

        cloned = factory.clone_agent_config(original.agent_id)

        self.assertIsNotNone(cloned)
        self.assertNotEqual(cloned.agent_id, original.agent_id)
        self.assertEqual(cloned.agent_type, original.agent_type)

    def test_get_agent_by_type(self):
        """测试按类型获取Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        for i in range(3):
            factory.generate_agent(f"Task {i}")

        # 检查是否有任何类型的Agent
        agents = factory.get_agent_by_type(AgentType.CODE)
        self.assertIsInstance(agents, list)


class TestLeaderElection(unittest.TestCase):
    """测试领导选举"""

    def setUp(self):
        """测试初始化"""
        from multiagent.meta.agent_factory import AgentFactory
        self.leader_module = __import__('multiagent.alliance.role_assigner', fromlist=['LeaderElection'])
        self.LeaderElection = self.leader_module.LeaderElection

    def test_election_initialization(self):
        """测试选举初始化"""
        election = self.LeaderElection()
        self.assertIsNotNone(election)

    def test_register_candidate(self):
        """测试注册候选人"""
        election = self.LeaderElection()

        election.register_candidate("agent_1", 100)
        election.register_candidate("agent_2", 200)

        self.assertEqual(len(election.candidates), 2)

    def test_start_election(self):
        """测试开始选举"""
        election = self.LeaderElection()

        election.register_candidate("agent_1", 100)
        election.start_election()

        self.assertTrue(election.is_election_active)

    def test_determine_winner(self):
        """测试确定获胜者"""
        election = self.LeaderElection()

        election.register_candidate("low_priority", 50)
        election.register_candidate("high_priority", 150)
        election.register_candidate("medium_priority", 100)

        winner = election.determine_winner()

        self.assertEqual(winner, "high_priority")

    def test_election_with_tie(self):
        """测试平局选举"""
        election = self.LeaderElection()

        election.register_candidate("candidate_a", 100)
        election.register_candidate("candidate_b", 100)

        winner = election.determine_winner()
        self.assertIn(winner, ["candidate_a", "candidate_b"])

    def test_election_timeout(self):
        """测试选举超时"""
        election = self.LeaderElection(timeout_seconds=1)

        election.start_election()
        time.sleep(1.1)

        self.assertTrue(election.has_timed_out)

    def test_leader_heartbeat(self):
        """测试领导者心跳"""
        election = self.LeaderElection()

        election.register_candidate("leader_candidate", 100)
        leader = election.determine_winner()

        election.send_heartbeat(leader)
        self.assertTrue(election.is_leader_active(leader))


class TestAgentRetirement(unittest.TestCase):
    """测试Agent退休"""

    def setUp(self):
        """测试初始化"""
        from multiagent.meta.agent_factory import AgentFactory
        self.retirement_module = __import__('multiagent.alliance.role_assigner', fromlist=['AgentRetirement'])
        self.AgentRetirement = self.retirement_module.AgentRetirement

    def test_retirement_initialization(self):
        """测试退休初始化"""
        retirement = self.AgentRetirement()
        self.assertIsNotNone(retirement)

    def test_retire_agent(self):
        """测试退休Agent"""
        retirement = self.AgentRetirement()

        result = retirement.retire("agent_to_retire", reason="task_complete")

        self.assertTrue(result)
        self.assertTrue(retirement.is_retired("agent_to_retire"))

    def test_retirement_reason_tracking(self):
        """测试退休原因追踪"""
        retirement = self.AgentRetirement()

        retirement.retire("agent_1", reason="performance_decline")
        retirement.retire("agent_2", reason="task_complete")

        reason = retirement.get_retirement_reason("agent_1")
        self.assertEqual(reason, "performance_decline")

    def test_retirement_archive(self):
        """测试退休存档"""
        retirement = self.AgentRetirement()

        retirement.retire("archive_agent", reason="normal_retirement")
        archive = retirement.get_archive("archive_agent")

        self.assertIsNotNone(archive)

    def test_cannot_retire_nonexistent(self):
        """测试不能退休不存在的Agent"""
        retirement = self.AgentRetirement()

        result = retirement.retire("nonexistent_agent")
        self.assertFalse(result)


class TestAgentCloning(unittest.TestCase):
    """测试Agent克隆"""

    def test_clone_creation(self):
        """测试克隆创建"""
        factory = MockSelfOrgHelpers.create_factory()

        original = factory.generate_agent("Original task")
        cloned = factory.clone_agent_config(original.agent_id)

        self.assertIsNotNone(cloned)
        self.assertNotEqual(original.agent_id, cloned.agent_id)

    def test_clone_inherits_capabilities(self):
        """测试克隆继承能力"""
        factory = MockSelfOrgHelpers.create_factory()

        original = factory.generate_agent("Complex Python task")
        cloned = factory.clone_agent_config(original.agent_id)

        # 克隆应该继承原始Agent的一些特性
        self.assertEqual(cloned.agent_type, original.agent_type)

    def test_clone_mutation(self):
        """测试克隆变异"""
        factory = MockSelfOrgHelpers.create_factory()

        original = factory.generate_agent("Task")
        cloned = factory.clone_agent_config(
            original.agent_id,
            mutations={"specialization_score": 0.5}
        )

        self.assertIsNotNone(cloned)

    def test_batch_cloning(self):
        """测试批量克隆"""
        factory = MockSelfOrgHelpers.create_factory()

        original = factory.generate_agent("Batch source")
        clones = []

        for i in range(5):
            clone = factory.clone_agent_config(original.agent_id)
            clones.append(clone)

        self.assertEqual(len(clones), 5)


class TestAgentMerge(unittest.TestCase):
    """测试Agent合并"""

    def setUp(self):
        """测试初始化"""
        from multiagent.meta.agent_factory import AgentFactory
        self.merge_module = __import__('multiagent.alliance.role_assigner', fromlist=['AgentMerger'])
        self.AgentMerger = self.merge_module.AgentMerger

    def test_merge_initialization(self):
        """测试合并初始化"""
        merger = self.AgentMerger()
        self.assertIsNotNone(merger)

    def test_merge_agents(self):
        """测试合并Agent"""
        merger = self.AgentMerger()

        agent1 = AgentConfiguration(
            agent_id="merge_1",
            agent_type=AgentType.CODE,
            name="Code Agent",
            description="Code specialized",
            capabilities={"python": CapabilityLevel.EXPERT},
            knowledge_domains={"software"},
            processing_style="sequential",
            memory_config={},
            reasoning_depth=3,
            communication_style="formal",
            specialization_score=0.8
        )

        agent2 = AgentConfiguration(
            agent_id="merge_2",
            agent_type=AgentType.CREATIVE,
            name="Creative Agent",
            description="Creative specialized",
            capabilities={"design": CapabilityLevel.ADVANCED},
            knowledge_domains={"arts"},
            processing_style="divergent",
            memory_config={},
            reasoning_depth=2,
            communication_style="inspirational",
            specialization_score=0.7
        )

        merged = merger.merge([agent1, agent2])

        self.assertIsNotNone(merged)
        self.assertEqual(merged.agent_type, AgentType.HYBRID)

    def test_merge_capabilities(self):
        """测试合并能力"""
        merger = self.AgentMerger()

        agent1_capabilities = {"python": CapabilityLevel.EXPERT}
        agent2_capabilities = {"design": CapabilityLevel.ADVANCED}

        merged_caps = merger.merge_capabilities(agent1_capabilities, agent2_capabilities)

        self.assertIn("python", merged_caps)
        self.assertIn("design", merged_caps)

    def test_merge_conflict_resolution(self):
        """测试合并冲突解决"""
        merger = self.AgentMerger()

        agent1 = {"style": "formal", "depth": 3}
        agent2 = {"style": "casual", "depth": 5}

        merged = merger.merge_attributes(agent1, agent2)

        self.assertIn("style", merged)
        self.assertIn("depth", merged)


class TestAgentCoach(unittest.TestCase):
    """测试Agent教练"""

    def setUp(self):
        """测试初始化"""
        from multiagent.meta.agent_factory import AgentFactory
        self.coach_module = __import__('multiagent.alliance.role_assigner', fromlist=['AgentCoach'])
        self.AgentCoach = self.coach_module.AgentCoach

    def test_coach_initialization(self):
        """测试教练初始化"""
        coach = self.AgentCoach()
        self.assertIsNotNone(coach)

    def test_identify_weaknesses(self):
        """测试识别弱点"""
        coach = self.AgentCoach()

        performance_data = {
            "accuracy": 0.7,
            "speed": 0.6,
            "reliability": 0.8
        }

        weaknesses = coach.identify_weaknesses(performance_data)

        self.assertIn("accuracy", weaknesses)
        self.assertIn("speed", weaknesses)

    def test_create_training_plan(self):
        """测试创建训练计划"""
        coach = self.AgentCoach()

        weaknesses = ["accuracy", "speed"]
        plan = coach.create_training_plan(
            agent_id="student_agent",
            weaknesses=weaknesses
        )

        self.assertIsNotNone(plan)
        self.assertIn("exercises", plan)

    def test_monitor_progress(self):
        """测试监控进度"""
        coach = self.AgentCoach()

        coach.record_performance("agent_1", {"accuracy": 0.7})
        coach.record_performance("agent_1", {"accuracy": 0.75})
        coach.record_performance("agent_1", {"accuracy": 0.8})

        progress = coach.get_progress("agent_1")
        self.assertGreater(progress["improvement"], 0)

    def test_coaching_session(self):
        """测试辅导会话"""
        coach = self.AgentCoach()

        session = coach.start_session(
            coach_id="coach_1",
            agent_id="student_1",
            topic="improvement"
        )

        self.assertIsNotNone(session)
        self.assertEqual(session.coach_id, "coach_1")


class TestDynamicReplanning(unittest.TestCase):
    """测试动态重规划"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.dynamic_replanning import Replanner, Plan, Task as PlanTask
        self.replan_module = __import__('multiagent.alliance.dynamic_replanning', fromlist=['Replanner'])
        self.Replanner = self.replan_module.Replanner

    def test_replanner_initialization(self):
        """测试重规划器初始化"""
        replanner = self.Replanner()
        self.assertIsNotNone(replanner)

    def test_create_initial_plan(self):
        """测试创建初始计划"""
        replanner = self.Replanner()

        plan = replanner.create_plan(
            task_id="main_task",
            steps=[
                {"step": 1, "description": "Analyze requirements"},
                {"step": 2, "description": "Implement solution"},
                {"step": 3, "description": "Test and verify"}
            ]
        )

        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)

    def test_detect_plan_failure(self):
        """测试检测计划失败"""
        replanner = self.Replanner()

        plan = replanner.create_plan(
            task_id="failure_task",
            steps=[{"step": 1}]
        )

        # 模拟步骤失败
        failure_detected = replanner.detect_failure(plan, step=1)

        self.assertTrue(failure_detected)

    def test_replan_on_failure(self):
        """测试失败时重规划"""
        replanner = self.Replanner()

        original_plan = replanner.create_plan(
            task_id="replan_task",
            steps=[
                {"step": 1, "approach": "original"},
                {"step": 2, "approach": "original"}
            ]
        )

        new_plan = replanner.replan(
            original_plan,
            reason="Step 1 failed",
            constraints={"time_limit": 3600}
        )

        self.assertIsNotNone(new_plan)

    def test_adaptive_replanning(self):
        """测试自适应重规划"""
        replanner = self.Replanner()

        # 模拟环境变化
        env_change = {"resource_availability": "low"}

        new_plan = replanner.adapt_plan(
            current_plan=None,
            environment_change=env_change
        )

        self.assertIsNotNone(new_plan)

    def test_rollback_capability(self):
        """测试回滚能力"""
        replanner = self.Replanner()

        plan = replanner.create_plan(
            task_id="rollback_test",
            steps=[
                {"step": 1, "id": "step_1"},
                {"step": 2, "id": "step_2"},
                {"step": 3, "id": "step_3"}
            ]
        )

        replanner.mark_completed("rollback_test", "step_2")
        rollback_plan = replanner.create_rollback_plan("rollback_test")

        self.assertIsNotNone(rollback_plan)


class TestTaskFeatureExtractor(unittest.TestCase):
    """测试任务特征提取"""

    def test_extract_from_code_task(self):
        """测试从代码任务提取"""
        factory = MockSelfOrgHelpers.create_factory()

        features = factory.feature_extractor.extract_features(
            "Write a Python function to process data with error handling"
        )

        self.assertEqual(features.domain, "software")
        self.assertIn("python", features.required_skills)

    def test_extract_from_analysis_task(self):
        """测试从分析任务提取"""
        factory = MockSelfOrgHelpers.create_factory()

        features = factory.feature_extractor.extract_features(
            "Analyze large datasets using machine learning models"
        )

        self.assertIsNotNone(features.complexity)
        self.assertGreater(features.data_intensity, 0)

    def test_extract_from_creative_task(self):
        """测试从创意任务提取"""
        factory = MockSelfOrgHelpers.create_factory()

        features = factory.feature_extractor.extract_features(
            "Create an innovative design for a mobile app interface"
        )

        self.assertGreater(features.creativity_required, 0)

    def test_extract_complexity(self):
        """测试复杂度提取"""
        factory = MockSelfOrgHelpers.create_factory()

        complex_features = factory.feature_extractor.extract_features(
            "This is a very complex and challenging task requiring sophisticated architecture design"
        )

        self.assertGreater(complex_features.complexity, 0.5)

    def test_feature_caching(self):
        """测试特征缓存"""
        factory = MockSelfOrgHelpers.create_factory()

        task_desc = "A simple Python task"
        features1 = factory.feature_extractor.extract_features(task_desc)
        features2 = factory.feature_extractor.extract_features(task_desc)

        # 应该使用缓存
        self.assertEqual(features1.domain, features2.domain)


class TestCapabilityLevel(unittest.TestCase):
    """测试能力等级"""

    def test_level_values(self):
        """测试等级值"""
        self.assertEqual(CapabilityLevel.NOVICE.value, 1)
        self.assertEqual(CapabilityLevel.INTERMEDIATE.value, 2)
        self.assertEqual(CapabilityLevel.ADVANCED.value, 3)
        self.assertEqual(CapabilityLevel.EXPERT.value, 4)
        self.assertEqual(CapabilityLevel.MASTER.value, 5)

    def test_level_comparison(self):
        """测试等级比较"""
        self.assertLess(CapabilityLevel.NOVICE, CapabilityLevel.INTERMEDIATE)
        self.assertGreater(CapabilityLevel.MASTER, CapabilityLevel.EXPERT)


class TestAgentType(unittest.TestCase):
    """测试Agent类型"""

    def test_type_values(self):
        """测试类型值"""
        self.assertIsNotNone(AgentType.RESEARCH)
        self.assertIsNotNone(AgentType.CODE)
        self.assertIsNotNone(AgentType.CREATIVE)
        self.assertIsNotNone(AgentType.ANALYSIS)
        self.assertIsNotNone(AgentType.COORDINATION)


class TestAgentConfiguration(unittest.TestCase):
    """测试Agent配置"""

    def test_config_creation(self):
        """测试配置创建"""
        config = AgentConfiguration(
            agent_id="config_001",
            agent_type=AgentType.CODE,
            name="TestAgent",
            description="A test configuration",
            capabilities={"python": CapabilityLevel.EXPERT},
            knowledge_domains={"software"},
            processing_style="sequential",
            memory_config={"type": "standard"},
            reasoning_depth=3,
            communication_style="formal",
            specialization_score=0.8
        )

        self.assertEqual(config.agent_id, "config_001")
        self.assertEqual(config.agent_type, AgentType.CODE)

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = AgentConfiguration(
            agent_id="dict_test",
            agent_type=AgentType.RESEARCH,
            name="Researcher",
            description="Test",
            capabilities={},
            knowledge_domains=set(),
            processing_style="adaptive",
            memory_config={},
            reasoning_depth=2,
            communication_style="formal",
            specialization_score=0.5
        )

        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["agent_id"], "dict_test")


class TestAgentTemplateMatching(unittest.TestCase):
    """测试Agent模板匹配"""

    def test_template_match_score(self):
        """测试模板匹配分数"""
        factory = MockSelfOrgHelpers.create_factory()

        # 获取第一个模板
        template = list(factory.templates.values())[0]

        features = MockSelfOrgHelpers.create_task_feature(
            domain="research",
            complexity=0.7
        )

        score = template.calculate_match_score(features)

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_best_template_selection(self):
        """测试最佳模板选择"""
        factory = MockSelfOrgHelpers.create_factory()

        features = MockSelfOrgHelpers.create_task_feature(
            domain="software",
            skills={"python"}
        )

        # 工厂应该选择最匹配的模板
        self.assertIsNotNone(factory._select_best_template(features))


class TestSelfOrganizationEdgeCases(unittest.TestCase):
    """测试自组织边界情况"""

    def test_empty_factory(self):
        """测试空工厂"""
        factory = DynamicAgentFactory()
        # 清除所有默认模板
        factory.templates.clear()

        # 没有模板时应该抛出异常
        with self.assertRaises(ValueError):
            factory.generate_agent("Any task")

    def test_invalid_task_description(self):
        """测试无效任务描述"""
        factory = MockSelfOrgHelpers.create_factory()

        # 空描述应该仍然能处理
        features = factory.feature_extractor.extract_features("")
        self.assertIsNotNone(features)

    def test_concurrent_agent_generation(self):
        """测试并发Agent生成"""
        factory = MockSelfOrgHelpers.create_factory()
        results = []

        def generate():
            config = factory.generate_agent("Concurrent task")
            results.append(config)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=generate)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 10)

    def test_clone_nonexistent_agent(self):
        """测试克隆不存在的Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        result = factory.clone_agent_config("nonexistent_id")
        self.assertIsNone(result)

    def test_update_nonexistent_agent(self):
        """测试更新不存在的Agent"""
        factory = MockSelfOrgHelpers.create_factory()

        result = factory.update_agent_config("nonexistent_id", {})
        self.assertIsNone(result)

    def test_election_no_candidates(self):
        """测试无候选人的选举"""
        election = self.LeaderElection()

        winner = election.determine_winner()
        self.assertIsNone(winner)

    def test_merge_single_agent(self):
        """测试合并单个Agent"""
        merger = self.AgentMerger()

        agent = AgentConfiguration(
            agent_id="single",
            agent_type=AgentType.GENERALIST,
            name="Single",
            description="Single agent",
            capabilities={},
            knowledge_domains=set(),
            processing_style="",
            memory_config={},
            reasoning_depth=1,
            communication_style="",
            specialization_score=0.5
        )

        merged = merger.merge([agent])
        self.assertIsNotNone(merged)


class TestRoleAssignment(unittest.TestCase):
    """测试角色分配"""

    def setUp(self):
        """测试初始化"""
        from multiagent.alliance.role_assigner import RoleAssigner, AgentRole
        self.role_module = __import__('multiagent.alliance.role_assigner', fromlist=['RoleAssigner'])
        self.RoleAssigner = self.role_module.RoleAssigner

    def test_assigner_initialization(self):
        """测试分配器初始化"""
        assigner = self.RoleAssigner()
        self.assertIsNotNone(assigner)

    def test_assign_role(self):
        """测试分配角色"""
        assigner = self.RoleAssigner()

        result = assigner.assign_role(
            agent_id="test_agent",
            role="coordinator"
        )

        self.assertTrue(result)

    def test_get_agent_role(self):
        """测试获取Agent角色"""
        assigner = self.RoleAssigner()

        assigner.assign_role("agent_1", "worker")
        role = assigner.get_role("agent_1")

        self.assertEqual(role, "worker")

    def test_role_swap(self):
        """测试角色交换"""
        assigner = self.RoleAssigner()

        assigner.assign_role("agent_1", "coordinator")
        assigner.assign_role("agent_2", "worker")

        assigner.swap_roles("agent_1", "agent_2")

        self.assertEqual(assigner.get_role("agent_1"), "worker")
        self.assertEqual(assigner.get_role("agent_2"), "coordinator")


if __name__ == "__main__":
    unittest.main()
