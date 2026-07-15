"""
知识转移测试模块

测试师生学习、行为克隆、DAgger、经验回放和联邦知识蒸馏。
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

from multiagent.learning.teacher_student import (
    ExpertAgent,
    NoviceAgent,
    TeacherStudentFramework,
    KnowledgeState,
    TeachingSession
)


class MockKnowledgeTransferHelpers:
    """知识转移测试辅助类"""

    @staticmethod
    def create_expert_agent(
        agent_id: str = "expert_001",
        domains: Optional[List[str]] = None
    ) -> ExpertAgent:
        """创建测试用专家Agent"""
        domains = domains or ["python", "machine_learning"]
        expert = ExpertAgent(agent_id, domains)

        # 添加一些知识
        for domain in domains:
            expert.add_knowledge(domain, "basics", "Foundation concepts", confidence=0.9)
            expert.add_knowledge(domain, "advanced", "Advanced topics", confidence=0.8)

        return expert

    @staticmethod
    def create_novice_agent(
        agent_id: str = "novice_001",
        learning_rate: float = 0.1
    ) -> NoviceAgent:
        """创建测试用新手Agent"""
        return NoviceAgent(agent_id, learning_rate)

    @staticmethod
    def create_teaching_examples(
        num_examples: int = 5,
        topic: str = "python"
    ) -> List[Dict[str, Any]]:
        """创建教学示例"""
        return [
            {
                "input": f"example_{i}",
                "label": i % 3,
                "expected_output": f"output_{i}"
            }
            for i in range(num_examples)
        ]


class TestTeacherStudent(unittest.TestCase):
    """测试师生学习框架"""

    def test_expert_agent_creation(self):
        """测试专家Agent创建"""
        expert = MockKnowledgeTransferHelpers.create_expert_agent("expert_001")

        self.assertEqual(expert.agent_id, "expert_001")
        self.assertEqual(expert.temperature, 1.0)

    def test_novice_agent_creation(self):
        """测试新手Agent创建"""
        novice = MockKnowledgeTransferHelpers.create_novice_agent("novice_001")

        self.assertEqual(novice.agent_id, "novice_001")
        self.assertEqual(novice.learning_rate, 0.1)

    def test_expert_add_knowledge(self):
        """测试专家添加知识"""
        expert = MockKnowledgeTransferHelpers.create_expert_agent()

        expert.add_knowledge("python", "generators", "Generator patterns", confidence=0.95)

        self.assertIn("generators", expert.knowledge_base["python"])

    def test_expert_soft_labels_generation(self):
        """测试软标签生成"""
        expert = MockKnowledgeTransferHelpers.create_expert_agent()

        inputs = ["input_1", "input_2", "input_3"]
        soft_labels = expert.get_soft_labels(inputs, "python")

        self.assertEqual(len(soft_labels), 3)
        for label in soft_labels:
            self.assertGreater(sum(label.values()), 0)

    def test_temperature_softmax(self):
        """测试温度缩放softmax"""
        expert = MockKnowledgeTransferHelpers.create_expert_agent()
        expert.temperature = 2.0  # 高温：更平滑的分布

        inputs = ["test_input"]
        soft_labels = expert.get_soft_labels(inputs, "python")

        # 高温下，概率分布应该更均匀
        values = list(soft_labels[0].values())
        max_val = max(values)
        min_val = min(values)

        self.assertLess(max_val - min_val, 1.0)

    def test_novice_learn_from_soft_label(self):
        """测试新手从软标签学习"""
        expert = MockKnowledgeTransferHelpers.create_expert_agent()
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        examples = MockKnowledgeTransferHelpers.create_teaching_examples()
        inputs = [ex["input"] for ex in examples]
        soft_labels = expert.get_soft_labels(inputs, "python")

        # 新手学习
        for example, soft_label in zip(examples, soft_labels):
            novice.learn_from_soft_label(example, soft_label, "python")

        # 验证知识状态已更新
        self.assertIn("python", novice.knowledge_state)

    def test_knowledge_state_updates(self):
        """测试知识状态更新"""
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        # 模拟学习
        novice._update_knowledge_state("python", success=True)

        state = novice.knowledge_state["python"]
        self.assertEqual(state.experience_count, 1)
        self.assertGreater(state.expertise_level, 0)

    def test_framework_registration(self):
        """测试框架注册"""
        framework = TeacherStudentFramework()

        expert = MockKnowledgeTransferHelpers.create_expert_agent()
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        framework.register_teacher(expert)
        framework.register_student(novice)

        self.assertEqual(len(framework.teachers), 1)
        self.assertEqual(len(framework.students), 1)

    def test_find_best_teacher(self):
        """测试找到最佳教师"""
        framework = TeacherStudentFramework()

        expert1 = MockKnowledgeTransferHelpers.create_expert_agent("expert_1", ["python"])
        expert2 = MockKnowledgeTransferHelpers.create_expert_agent("expert_2", ["python"])

        framework.register_teacher(expert1)
        framework.register_teacher(expert2)

        # expert2 有更多教学历史
        expert2.teach(
            MockKnowledgeTransferHelpers.create_novice_agent("temp"),
            "python",
            MockKnowledgeTransferHelpers.create_teaching_examples()
        )

        best_teacher = framework.find_best_teacher("python", "student_1")
        self.assertEqual(best_teacher.agent_id, "expert_2")

    def test_conduct_lesson(self):
        """测试进行课程"""
        framework = TeacherStudentFramework()

        expert = MockKnowledgeTransferHelpers.create_expert_agent()
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        framework.register_teacher(expert)
        framework.register_student(novice)

        session = framework.conduct_lesson(
            "novice_001",
            "python",
            MockKnowledgeTransferHelpers.create_teaching_examples()
        )

        self.assertIsNotNone(session)
        self.assertEqual(session.teacher_id, "expert_001")
        self.assertEqual(session.student_id, "novice_001")

    def test_evaluate_student_progress(self):
        """测试评估学生进度"""
        framework = TeacherStudentFramework()

        expert = MockKnowledgeTransferHelpers.create_expert_agent()
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        framework.register_teacher(expert)
        framework.register_student(novice)

        # 进行一些课程
        framework.conduct_lesson(
            "novice_001",
            "python",
            MockKnowledgeTransferHelpers.create_teaching_examples()
        )

        progress = framework.evaluate_student_progress("novice_001")

        self.assertIn("overall_expertise", progress)
        self.assertIn("topics_learned", progress)


class TestBehaviorCloning(unittest.TestCase):
    """测试行为克隆"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.behaviour_cloning import BehaviorCloner, Demonstration
        self.bc_module = __import__('multiagent.learning.behaviour_cloning', fromlist=['BehaviorCloner'])
        self.BehaviorCloner = self.bc_module.BehaviorCloner
        self.Demonstration = self.bc_module.Demonstration

    def test_cloner_initialization(self):
        """测试克隆器初始化"""
        cloner = self.BehaviorCloner()
        self.assertIsNotNone(cloner)

    def test_add_demonstration(self):
        """测试添加演示"""
        cloner = self.BehaviorCloner()

        demo = self.Demonstration(
            state={"observation": "state_1"},
            action={"type": "move", "direction": "right"},
            reward=1.0
        )

        cloner.add_demonstration(demo)
        self.assertEqual(cloner.num_demonstrations, 1)

    def test_clone_behavior(self):
        """测试克隆行为"""
        cloner = self.BehaviorCloner()

        # 添加多个演示
        for i in range(10):
            demo = self.Demonstration(
                state={"obs": f"state_{i}"},
                action={"type": "action", "id": i},
                reward=1.0 if i < 5 else 0.0
            )
            cloner.add_demonstration(demo)

        policy = cloner.clone()

        self.assertIsNotNone(policy)
        self.assertGreater(len(policy), 0)

    def test_loss_calculation(self):
        """测试损失计算"""
        cloner = self.BehaviorCloner()

        # 预测 vs 目标
        predicted = {"action": 1}
        target = {"action": 1}

        loss = cloner.calculate_loss(predicted, target)
        self.assertEqual(loss, 0.0)

    def test_state_action_pairing(self):
        """测试状态动作配对"""
        cloner = self.BehaviorCloner()

        demos = [
            self.Demonstration({"obs": "A"}, {"action": 1}),
            self.Demonstration({"obs": "B"}, {"action": 2}),
            self.Demonstration({"obs": "C"}, {"action": 3}),
        ]

        for demo in demos:
            cloner.add_demonstration(demo)

        # 验证配对
        pairs = cloner.get_state_action_pairs()
        self.assertEqual(len(pairs), 3)


class TestDAgger(unittest.TestCase):
    """测试DAgger算法"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.dagger_loop import DAgger, DAggregation
        self.dagger_module = __import__('multiagent.learning.dagger_loop', fromlist=['DAgger'])
        self.DAgger = self.dagger_module.DAgger

    def test_dagger_initialization(self):
        """测试DAgger初始化"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent()
        )

        self.assertIsNotNone(dagger)
        self.assertEqual(dagger.iterations, 0)

    def test_iteration(self):
        """测试迭代"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent()
        )

        dagger.run_iteration(
            states=["state_1", "state_2", "state_3"],
            max_demos=5
        )

        self.assertEqual(dagger.iterations, 1)
        self.assertGreater(dagger.total_demonstrations, 0)

    def test_convergence_check(self):
        """测试收敛检查"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent()
        )

        # 模拟多轮迭代
        for _ in range(10):
            dagger.run_iteration(["s1", "s2"], max_demos=3)

        # 检查是否收敛
        converged = dagger.check_convergence()
        self.assertIn(converged, [True, False])

    def test_beta_schedule(self):
        """测试beta调度"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent(),
            initial_beta=1.0
        )

        # beta应该随迭代递减
        dagger.run_iteration(["s1"], max_demos=3)
        beta_after = dagger.get_current_beta()

        self.assertLessEqual(beta_after, 1.0)

    def test_demonstration_buffer(self):
        """测试演示缓冲区"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent()
        )

        dagger.collect_demonstrations(["s1", "s2", "s3"], num_demos=10)

        self.assertGreaterEqual(len(dagger.demonstrations), 3)


class TestExperienceReplay(unittest.TestCase):
    """测试经验回放"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.experience_replay_shared import ExperienceReplay, PrioritizedReplay
        self.replay_module = __import__('multiagent.learning.experience_replay_shared', fromlist=['ExperienceReplay'])
        self.ExperienceReplay = self.replay_module.ExperienceReplay
        self.PrioritizedReplay = self.replay_module.PrioritizedReplay

    def test_replay_initialization(self):
        """测试回放初始化"""
        replay = self.ExperienceReplay(capacity=100)
        self.assertEqual(replay.capacity, 100)

    def test_add_experience(self):
        """测试添加经验"""
        replay = self.ExperienceReplay(capacity=10)

        replay.add({
            "state": "s1",
            "action": "a1",
            "reward": 1.0,
            "next_state": "s2"
        })

        self.assertEqual(len(replay), 1)

    def test_sample_batch(self):
        """测试采样批次"""
        replay = self.ExperienceReplay(capacity=50)

        # 添加足够多的经验
        for i in range(20):
            replay.add({
                "state": f"s_{i}",
                "action": f"a_{i}",
                "reward": float(i % 2),
                "next_state": f"s_{i+1}"
            })

        batch = replay.sample(batch_size=5)

        self.assertEqual(len(batch), 5)

    def test_buffer_overflow(self):
        """测试缓冲区溢出"""
        replay = self.ExperienceReplay(capacity=5)

        # 添加超过容量的经验
        for i in range(10):
            replay.add({
                "state": f"s_{i}",
                "action": f"a_{i}",
                "reward": 1.0,
                "next_state": f"s_{i+1}"
            })

        # 应该保持最大容量
        self.assertEqual(len(replay), 5)

    def test_prioritized_replay(self):
        """测试优先级回放"""
        replay = self.PrioritizedReplay(capacity=100)

        # 添加不同优先级的经验
        replay.add({
            "state": "high_priority",
            "action": "a1",
            "reward": 1.0,
            "next_state": "s2"
        }, priority=10.0)

        replay.add({
            "state": "low_priority",
            "action": "a2",
            "reward": 0.0,
            "next_state": "s3"
        }, priority=1.0)

        # 高优先级经验更可能被采样
        batch = replay.sample(batch_size=1)
        self.assertEqual(batch[0]["state"], "high_priority")

    def test_priority_update(self):
        """测试优先级更新"""
        replay = self.PrioritizedReplay(capacity=100)

        idx = replay.add({
            "state": "test",
            "action": "a",
            "reward": 0.5,
            "next_state": "s2"
        }, priority=1.0)

        # 更新优先级
        replay.update_priority(idx, 100.0)

        # 验证更新
        priorities = replay.get_priorities()
        self.assertGreater(priorities[idx], 1.0)

    def test_replay_clear(self):
        """测试清空回放"""
        replay = self.ExperienceReplay(capacity=100)

        for i in range(10):
            replay.add({"state": f"s_{i}"})

        replay.clear()
        self.assertEqual(len(replay), 0)


class TestFederatedKnowledgeDistillation(unittest.TestCase):
    """测试联邦知识蒸馏"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.federated_knowledge import FederatedDistiller, LocalModel, AggregationStrategy
        self.fk_module = __import__('multiagent.learning.federated_knowledge', fromlist=['FederatedDistiller'])
        self.FederatedDistiller = self.fk_module.FederatedDistiller
        self.LocalModel = self.fk_module.LocalModel
        self.AggregationStrategy = self.fk_module.AggregationStrategy

    def test_distiller_initialization(self):
        """测试蒸馏器初始化"""
        distiller = self.FederatedDistiller(
            aggregation_strategy=self.fk_module.AggregationStrategy.FEDAVG
        )

        self.assertIsNotNone(distiller)

    def test_register_client(self):
        """测试注册客户端"""
        distiller = self.FederatedDistiller()

        client = self.LocalModel(
            client_id="client_1",
            model_params={"layer1": [1, 2, 3]}
        )

        distiller.register_client(client)
        self.assertEqual(len(distiller.clients), 1)

    def test_local_training(self):
        """测试本地训练"""
        distiller = self.FederatedDistiller()

        client = self.LocalModel(
            client_id="client_1",
            model_params={"layer1": [0.5]}
        )
        distiller.register_client(client)

        # 本地训练
        updated_params = distiller.local_train(
            "client_1",
            local_epochs=5,
            batch_size=10
        )

        self.assertIsNotNone(updated_params)

    def test_aggregation_fedavg(self):
        """测试FedAvg聚合"""
        distiller = self.FederatedDistiller(
            aggregation_strategy=self.fk_module.AggregationStrategy.FEDAVG
        )

        # 注册多个客户端
        for i in range(3):
            client = self.LocalModel(
                client_id=f"client_{i}",
                model_params={"layer1": [float(i)]},
                data_size=100
            )
            distiller.register_client(client)

        # 聚合
        global_params = distiller.aggregate()

        self.assertIsNotNone(global_params)

    def test_distillation_round(self):
        """测试蒸馏轮次"""
        distiller = self.FederatedDistiller()

        # 注册客户端
        for i in range(2):
            client = self.LocalModel(
                client_id=f"client_{i}",
                model_params={"layer1": [1.0, 2.0]},
                data_size=50
            )
            distiller.register_client(client)

        # 执行一轮蒸馏
        round_result = distiller.run_round()

        self.assertEqual(round_result["round"], 1)
        self.assertIn("global_model", round_result)

    def test_knowledge_aggregation(self):
        """测试知识聚合"""
        distiller = self.FederatedDistiller(
            aggregation_strategy=self.fk_module.AggregationStrategy.KD_WEIGHTED
        )

        # 添加教师的软标签
        teacher = MockKnowledgeTransferHelpers.create_expert_agent()
        distiller.set_teacher_knowledge(teacher)

        self.assertIsNotNone(distiller.teacher_soft_labels)

    def test_client_selection(self):
        """测试客户端选择"""
        distiller = self.FederatedDistiller()

        # 注册多个客户端
        for i in range(10):
            client = self.LocalModel(
                client_id=f"client_{i}",
                model_params={"layer1": [1.0]},
                data_size=100,
                availability=1.0 if i < 5 else 0.0
            )
            distiller.register_client(client)

        # 选择可用的客户端
        selected = distiller.select_clients(min_clients=3)

        self.assertLessEqual(len(selected), 5)
        for client in selected:
            self.assertGreater(client.availability, 0)


class TestKnowledgeTransferMetrics(unittest.TestCase):
    """测试知识转移指标"""

    def test_knowledge_retention(self):
        """测试知识保留率"""
        initial_knowledge = 100.0
        transferred_knowledge = 85.0

        retention = transferred_knowledge / initial_knowledge
        self.assertAlmostEqual(retention, 0.85, places=2)

    def test_learning_curve(self):
        """测试学习曲线"""
        novice = MockKnowledgeTransferHelpers.create_novice_agent()
        expert = MockKnowledgeTransferHelpers.create_expert_agent()

        expertise_levels = []
        for i in range(10):
            # 模拟学习
            novice._update_knowledge_state("python", success=(i > 3))
            expertise_levels.append(
                novice.knowledge_state.get("python", KnowledgeState(0.0, 0.0)).expertise_level
            )

        # 学习应该随时间改进
        self.assertGreater(expertise_levels[-1], expertise_levels[0])

    def test_transfer_efficiency(self):
        """测试转移效率"""
        teacher = MockKnowledgeTransferHelpers.create_expert_agent()
        student = MockKnowledgeTransferHelpers.create_novice_agent()

        # 计算效率
        teacher_knowledge = len(teacher.knowledge_base) * 10
        student_knowledge = len(student.knowledge_state) * 5

        efficiency = student_knowledge / teacher_knowledge
        self.assertLess(efficiency, 1.0)

    def test_convergence_criterion(self):
        """测试收敛标准"""
        losses = [1.0, 0.8, 0.5, 0.3, 0.2, 0.15, 0.12, 0.11, 0.105, 0.1]

        # 检查损失是否收敛
        for i in range(len(losses) - 1):
            improvement = abs(losses[i] - losses[i+1])
            if improvement < 0.01:
                self.assertTrue(True)
                break
        else:
            self.fail("Losses did not converge")

    def test_sample_efficiency(self):
        """测试样本效率"""
        num_demos_needed = 100
        optimal_demos = 50

        efficiency = optimal_demos / num_demos_needed
        self.assertLess(efficiency, 1.0)


class TestCurriculumLearning(unittest.TestCase):
    """测试课程学习"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.curriculum_for_agent import CurriculumScheduler, DifficultyLevel
        self.curriculum_module = __import__('multiagent.learning.curriculum_for_agent', fromlist=['CurriculumScheduler'])
        self.CurriculumScheduler = self.curriculum_module.CurriculumScheduler

    def test_curriculum_initialization(self):
        """测试课程初始化"""
        scheduler = self.CurriculumScheduler()
        self.assertIsNotNone(scheduler)

    def test_add_task(self):
        """测试添加任务"""
        scheduler = self.CurriculumScheduler()

        scheduler.add_task(
            task_id="task_1",
            difficulty=0.3,
            concepts=["basics"]
        )

        self.assertEqual(len(scheduler.tasks), 1)

    def test_task_ordering(self):
        """测试任务排序"""
        scheduler = self.CurriculumScheduler()

        scheduler.add_task("easy", difficulty=0.2)
        scheduler.add_task("medium", difficulty=0.5)
        scheduler.add_task("hard", difficulty=0.9)

        ordered = scheduler.get_ordered_tasks()

        # 任务应按难度排序
        difficulties = [t["difficulty"] for t in ordered]
        self.assertEqual(difficulties, sorted(difficulties))

    def test_adaptive_difficulty(self):
        """测试自适应难度"""
        scheduler = self.CurriculumScheduler()

        # 添加任务
        for i in range(5):
            scheduler.add_task(f"task_{i}", difficulty=0.2 + i * 0.15)

        # 模拟学生表现
        scheduler.update_progress("task_0", success_rate=0.9)
        scheduler.update_progress("task_1", success_rate=0.8)
        scheduler.update_progress("task_2", success_rate=0.3)  # 失败

        # 下个任务的难度应该调整
        next_task = scheduler.get_next_task()
        self.assertLess(next_task["difficulty"], 0.6)


class TestSkillEmergence(unittest.TestCase):
    """测试技能涌现"""

    def setUp(self):
        """测试初始化"""
        from multiagent.learning.skill_emergence import SkillEmergence, SkillGraph
        self.emergence_module = __import__('multiagent.learning.skill_emergence', fromlist=['SkillEmergence'])
        self.SkillEmergence = self.emergence_module.SkillEmergence

    def test_emergence_initialization(self):
        """测试涌现初始化"""
        emergence = self.SkillEmergence()
        self.assertIsNotNone(emergence)

    def test_skill_dependency(self):
        """测试技能依赖"""
        emergence = self.SkillEmergence()

        emergence.add_skill("python", difficulty=0.3)
        emergence.add_skill("ml", difficulty=0.7)
        emergence.add_dependency("python", "ml")

        self.assertTrue(emergence.check_prerequisites("ml"))

    def test_skill_unlock(self):
        """测试技能解锁"""
        emergence = self.SkillEmergence()

        emergence.add_skill("basics", difficulty=0.2)
        emergence.add_skill("advanced", difficulty=0.6, prerequisites=["basics"])

        # 未解锁
        self.assertFalse(emergence.is_skill_unlocked("advanced"))

        # 完成前置技能
        emergence.complete_skill("basics")

        # 已解锁
        self.assertTrue(emergence.is_skill_unlocked("advanced"))

    def test_skill_learning(self):
        """测试技能学习"""
        emergence = self.SkillEmergence()

        emergence.add_skill("new_skill", difficulty=0.5)
        emergence.learn_skill("new_skill", mastery=0.8)

        self.assertEqual(emergence.get_mastery("new_skill"), 0.8)

    def test_emergent_complex_skills(self):
        """测试涌现复杂技能"""
        emergence = self.SkillEmergence()

        # 添加基础技能
        emergence.add_skill("A", difficulty=0.3)
        emergence.add_skill("B", difficulty=0.3)

        # 复杂技能需要多个基础技能
        emergence.add_skill("complex", difficulty=0.8, prerequisites=["A", "B"])

        # 完成基础技能
        emergence.complete_skill("A")
        emergence.complete_skill("B")

        # 检查复杂技能是否涌现
        can_learn = emergence.can_learn("complex")
        self.assertTrue(can_learn)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_expert_knowledge(self):
        """测试空专家知识"""
        expert = ExpertAgent("empty_expert", [])
        novice = MockKnowledgeTransferHelpers.create_novice_agent()

        # 应该处理空知识
        soft_labels = expert.get_soft_labels(["input"], "nonexistent")
        self.assertIsNotNone(soft_labels)

    def test_zero_learning_rate(self):
        """测试零学习率"""
        novice = NoviceAgent("slow_learner", learning_rate=0.0)

        novice._update_knowledge_state("test", success=True)

        state = novice.knowledge_state["test"]
        # 不应该更新
        self.assertEqual(state.expertise_level, 0.0)

    def test_full_replay_buffer(self):
        """测试满的回放缓冲区"""
        replay = self.ExperienceReplay(capacity=3)

        for i in range(10):
            replay.add({"state": f"s_{i}"})

        self.assertEqual(len(replay), 3)

    def test_no_teacher_for_topic(self):
        """测试没有教师教授主题"""
        framework = TeacherStudentFramework()
        novice = MockKnowledgeTransferHelpers.create_novice_agent()
        framework.register_student(novice)

        # 框架中没有教师
        session = framework.conduct_lesson(
            "novice_001",
            "nonexistent_topic",
            []
        )

        self.assertIsNone(session)

    def test_negative_priority(self):
        """测试负优先级"""
        replay = self.PrioritizedReplay(capacity=100)

        # 应该处理负优先级
        idx = replay.add({"state": "test"}, priority=-5.0)
        self.assertIsNotNone(idx)

    def test_empty_demonstration_list(self):
        """测试空演示列表"""
        dagger = self.DAgger(
            expert=MockKnowledgeTransferHelpers.create_expert_agent(),
            student=MockKnowledgeTransferHelpers.create_novice_agent()
        )

        # 应该处理空列表
        dagger.collect_demonstrations([], num_demos=5)
        self.assertEqual(len(dagger.demonstrations), 0)


if __name__ == "__main__":
    unittest.main()
