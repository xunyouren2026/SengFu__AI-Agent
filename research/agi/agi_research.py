"""
AGI研究模块
===========
实现通用人工智能研究相关的评估与分析算法，包括：
- 通用智能指标
- 迁移学习评估
- 少样本学习能力
- 元学习研究
- 灾难性遗忘研究
- 持续学习

重构说明：
- 内部使用core/unified_algorithms/统一核心
- 使用UnifiedMoE进行专家路由
- 使用UnifiedConstraintSystem进行约束管理

作者: AGI研究框架
版本: 2.0.0 (Unified)
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod
import json
import copy

# 导入统一核心
from ...core.unified_algorithms.unified_moe import (
    UnifiedMoE,
    UnifiedExpert,
    UnifiedRouter,
    ExpertType,
    RoutingStrategy,
)
from ...core.unified_algorithms.unified_constraints import (
    UnifiedConstraintSystem,
    UnifiedConstraint,
    ConstraintType,
    ConstraintPriority,
)
from ...core.unified_algorithms.unified_config import (
    UnifiedAlgorithmConfig,
)


# ============================================================================
# 基础数据结构与枚举
# ============================================================================

class LearningType(Enum):
    """学习类型"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    FEW_SHOT = "few_shot"
    META = "meta"
    CONTINUAL = "continual"


class TaskDomain(Enum):
    """任务领域"""
    VISION = "vision"
    LANGUAGE = "language"
    REASONING = "reasoning"
    MOTOR = "motor"
    MEMORY = "memory"
    SOCIAL = "social"


@dataclass
class Task:
    """任务定义"""
    task_id: str
    domain: TaskDomain
    difficulty: float
    train_data: List[Tuple[Any, Any]]
    test_data: List[Tuple[Any, Any]]
    eval_metric: str = "accuracy"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Performance:
    """性能记录"""
    task_id: str
    score: float
    learning_curve: List[float]
    samples_used: int
    training_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGIReport:
    """AGI评估报告"""
    overall_score: float
    domain_scores: Dict[TaskDomain, float]
    transfer_efficiency: float
    few_shot_capability: float
    meta_learning_score: float
    forgetting_rate: float
    continual_learning_score: float
    timestamp: int
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 通用智能指标
# ============================================================================

class GeneralIntelligenceMetrics:
    """
    通用智能指标评估
    
    基于多任务性能评估系统的通用智能水平，
    包括领域广度、任务难度适应性和学习效率。
    """
    
    def __init__(self, domains: List[TaskDomain] = None):
        self.domains = domains or list(TaskDomain)
        
        # 任务性能记录
        self.task_performances: Dict[str, Performance] = {}
        
        # 领域性能汇总
        self.domain_performances: Dict[TaskDomain, List[float]] = {
            domain: [] for domain in self.domains
        }
        
        # 难度-性能映射
        self.difficulty_performance: Dict[float, List[float]] = defaultdict(list)
        
        # 学习曲线历史
        self.learning_curves: Dict[str, List[float]] = {}
        
    def evaluate_task(self, 
                     task: Task,
                     model: Callable[[Any], Any],
                     training_fn: Callable[[Any, Any], None]) -> Performance:
        """
        评估模型在特定任务上的性能
        
        Args:
            task: 任务定义
            model: 模型推理函数
            training_fn: 训练函数
            
        Returns:
            Performance: 性能记录
        """
        learning_curve = []
        
        # 逐步训练并记录学习曲线
        train_subset_size = max(1, len(task.train_data) // 10)
        
        for i in range(1, 11):
            # 使用部分训练数据
            subset = task.train_data[:i * train_subset_size]
            
            # 训练
            for x, y in subset:
                training_fn(model, (x, y))
                
            # 评估
            correct = 0
            for x, y in task.test_data:
                pred = model(x)
                if self._check_correctness(pred, y, task.eval_metric):
                    correct += 1
                    
            accuracy = correct / len(task.test_data)
            learning_curve.append(accuracy)
            
        final_score = learning_curve[-1]
        
        performance = Performance(
            task_id=task.task_id,
            score=final_score,
            learning_curve=learning_curve,
            samples_used=len(task.train_data),
            training_time=0.0,  # 应由外部计时
            metadata={'difficulty': task.difficulty, 'domain': task.domain}
        )
        
        self.task_performances[task.task_id] = performance
        self.domain_performances[task.domain].append(final_score)
        self.difficulty_performance[task.difficulty].append(final_score)
        self.learning_curves[task.task_id] = learning_curve
        
        return performance
    
    def _check_correctness(self, 
                          prediction: Any, 
                          target: Any, 
                          metric: str) -> bool:
        """检查预测是否正确"""
        if metric == "accuracy":
            return prediction == target
        elif metric == "mse":
            return abs(prediction - target) < 0.1
        elif metric == "cosine":
            # 简化的余弦相似度检查
            return np.dot(prediction, target) > 0.9
        return False
    
    def compute_general_intelligence_score(self) -> float:
        """
        计算通用智能分数
        
        综合考虑：
        - 跨领域平均性能
        - 领域覆盖度
        - 难度适应性
        - 学习效率
        """
        if not self.task_performances:
            return 0.0
            
        # 1. 跨领域平均性能
        domain_averages = []
        for domain in self.domains:
            scores = self.domain_performances[domain]
            if scores:
                domain_averages.append(np.mean(scores))
            else:
                domain_averages.append(0.0)
                
        cross_domain_performance = np.mean(domain_averages)
        
        # 2. 领域覆盖度（有多少领域被覆盖）
        covered_domains = sum(1 for scores in self.domain_performances.values() if scores)
        domain_coverage = covered_domains / len(self.domains)
        
        # 3. 难度适应性
        difficulty_scores = []
        for diff, scores in self.difficulty_performance.items():
            if scores:
                # 加权：高难度任务的权重更高
                weight = 1 + diff
                difficulty_scores.append(np.mean(scores) * weight)
                
        difficulty_adaptation = np.mean(difficulty_scores) if difficulty_scores else 0
        difficulty_adaptation = min(1.0, difficulty_adaptation / 2)  # 归一化
        
        # 4. 学习效率（平均达到80%性能所需的样本比例）
        learning_efficiencies = []
        for curve in self.learning_curves.values():
            if curve:
                target = 0.8 * curve[-1]
                for i, score in enumerate(curve):
                    if score >= target:
                        learning_efficiencies.append(1 - i / len(curve))
                        break
                else:
                    learning_efficiencies.append(0.0)
                    
        learning_efficiency = np.mean(learning_efficiencies) if learning_efficiencies else 0
        
        # 综合分数
        gi_score = (
            0.35 * cross_domain_performance +
            0.20 * domain_coverage +
            0.25 * difficulty_adaptation +
            0.20 * learning_efficiency
        )
        
        return gi_score
    
    def compute_domain_balance(self) -> Dict[str, float]:
        """计算领域平衡度"""
        domain_scores = {}
        for domain in self.domains:
            scores = self.domain_performances[domain]
            domain_scores[domain.value] = np.mean(scores) if scores else 0.0
            
        # 计算平衡度（分数的方差倒数）
        values = list(domain_scores.values())
        variance = np.var(values)
        balance = np.exp(-variance * 2)
        
        domain_scores['balance'] = balance
        return domain_scores
    
    def get_capability_profile(self) -> Dict[str, Any]:
        """获取能力画像"""
        return {
            'general_intelligence': self.compute_general_intelligence_score(),
            'domain_scores': self.compute_domain_balance(),
            'n_tasks_evaluated': len(self.task_performances),
            'average_performance': np.mean([p.score for p in self.task_performances.values()]),
            'performance_variance': np.var([p.score for p in self.task_performances.values()]),
            'difficulty_range': {
                'min': min(self.difficulty_performance.keys()) if self.difficulty_performance else 0,
                'max': max(self.difficulty_performance.keys()) if self.difficulty_performance else 0
            }
        }


# ============================================================================
# 迁移学习评估
# ============================================================================

class TransferLearningEvaluator:
    """
    迁移学习评估器
    
    评估系统在不同任务间迁移知识的能力，
    包括正向迁移、负向迁移和零样本迁移。
    """
    
    def __init__(self):
        # 源任务性能
        self.source_performances: Dict[str, float] = {}
        
        # 目标任务性能（无迁移）
        self.target_baseline: Dict[str, float] = {}
        
        # 目标任务性能（有迁移）
        self.target_with_transfer: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 迁移矩阵
        self.transfer_matrix: Dict[Tuple[str, str], float] = {}
        
        # 迁移历史
        self.transfer_history: List[Dict] = []
        
    def record_source_performance(self, task_id: str, performance: float):
        """记录源任务性能"""
        self.source_performances[task_id] = performance
        
    def record_target_baseline(self, task_id: str, performance: float):
        """记录目标任务基线性能（无迁移）"""
        self.target_baseline[task_id] = performance
        
    def record_transfer_performance(self,
                                    source_task: str,
                                    target_task: str,
                                    performance: float):
        """记录迁移后的性能"""
        self.target_with_transfer[source_task][target_task] = performance
        
        # 计算迁移效果
        baseline = self.target_baseline.get(target_task, 0)
        if baseline > 0:
            transfer_effect = (performance - baseline) / baseline
            self.transfer_matrix[(source_task, target_task)] = transfer_effect
            
        self.transfer_history.append({
            'source': source_task,
            'target': target_task,
            'performance': performance,
            'baseline': baseline,
            'transfer_effect': self.transfer_matrix.get((source_task, target_task), 0)
        })
    
    def compute_transfer_efficiency(self, 
                                   source_task: str,
                                   target_task: str) -> float:
        """
        计算迁移效率
        
        Returns:
            迁移效率分数（正表示正向迁移，负表示负向迁移）
        """
        key = (source_task, target_task)
        
        if key not in self.transfer_matrix:
            return 0.0
            
        return self.transfer_matrix[key]
    
    def compute_average_transfer(self) -> Dict[str, float]:
        """计算平均迁移效果"""
        if not self.transfer_matrix:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'average': 0}
            
        effects = list(self.transfer_matrix.values())
        
        positive = [e for e in effects if e > 0.05]
        negative = [e for e in effects if e < -0.05]
        neutral = [e for e in effects if -0.05 <= e <= 0.05]
        
        return {
            'positive_transfer_rate': len(positive) / len(effects),
            'negative_transfer_rate': len(negative) / len(effects),
            'neutral_rate': len(neutral) / len(effects),
            'average_transfer_effect': np.mean(effects),
            'positive_transfer_magnitude': np.mean(positive) if positive else 0,
            'negative_transfer_magnitude': np.mean(negative) if negative else 0
        }
    
    def compute_transfer_matrix_stats(self) -> Dict[str, Any]:
        """计算迁移矩阵统计信息"""
        if not self.transfer_matrix:
            return {}
            
        matrix_values = list(self.transfer_matrix.values())
        
        return {
            'matrix_size': len(matrix_values),
            'mean': np.mean(matrix_values),
            'std': np.std(matrix_values),
            'min': np.min(matrix_values),
            'max': np.max(matrix_values),
            'diagonal_dominance': self._check_diagonal_dominance(),
            'symmetry': self._check_symmetry()
        }
    
    def _check_diagonal_dominance(self) -> float:
        """检查迁移矩阵的对角线优势（任务自我迁移）"""
        # 在实际应用中，对角线元素表示任务自身的性能
        # 这里简化为检查源任务和目标任务相同的配对
        diagonal_effects = []
        
        for (src, tgt), effect in self.transfer_matrix.items():
            if src == tgt:
                diagonal_effects.append(effect)
                
        return np.mean(diagonal_effects) if diagonal_effects else 0
    
    def _check_symmetry(self) -> float:
        """检查迁移矩阵的对称性"""
        symmetric_pairs = []
        total_pairs = 0
        
        for (src1, tgt1), effect1 in self.transfer_matrix.items():
            if src1 != tgt1:
                total_pairs += 1
                key2 = (tgt1, src1)
                if key2 in self.transfer_matrix:
                    effect2 = self.transfer_matrix[key2]
                    symmetric_pairs.append(abs(effect1 - effect2) < 0.1)
                    
        return np.mean(symmetric_pairs) if symmetric_pairs else 0
    
    def identify_transfer_clusters(self) -> List[Set[str]]:
        """识别迁移聚类（哪些任务组之间有强迁移）"""
        # 基于迁移效果构建图
        from collections import defaultdict
        
        graph = defaultdict(set)
        
        for (src, tgt), effect in self.transfer_matrix.items():
            if effect > 0.3:  # 强迁移阈值
                graph[src].add(tgt)
                graph[tgt].add(src)
                
        # 查找连通分量
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            visited.add(node)
            cluster.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
                    
        for node in graph:
            if node not in visited:
                cluster = set()
                dfs(node, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)
                    
        return clusters
    
    def get_transfer_report(self) -> Dict[str, Any]:
        """获取迁移学习评估报告"""
        return {
            'average_transfer': self.compute_average_transfer(),
            'matrix_stats': self.compute_transfer_matrix_stats(),
            'transfer_clusters': [list(c) for c in self.identify_transfer_clusters()],
            'n_source_tasks': len(self.source_performances),
            'n_target_tasks': len(self.target_baseline),
            'n_transfer_evaluations': len(self.transfer_matrix)
        }


# ============================================================================
# 少样本学习能力评估
# ============================================================================

class FewShotEvaluator:
    """
    少样本学习能力评估器
    
    评估系统从少量示例中快速学习的能力，
    包括N-way K-shot分类和快速适应。
    """
    
    def __init__(self, 
                 n_way_options: List[int] = None,
                 k_shot_options: List[int] = None):
        self.n_way_options = n_way_options or [5, 10, 20]
        self.k_shot_options = k_shot_options or [1, 5, 10]
        
        # 少样本实验结果
        self.experiment_results: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        
        # 学习速度曲线
        self.learning_speed_curves: Dict[str, List[float]] = {}
        
        # 元学习性能
        self.meta_learning_scores: Dict[str, float] = {}
        
    def evaluate_episode(self,
                        n_way: int,
                        k_shot: int,
                        model: Callable,
                        support_set: List[Tuple[Any, Any]],
                        query_set: List[Tuple[Any, Any]]) -> float:
        """
        评估单个少样本学习episode
        
        Args:
            n_way: N个类别
            k_shot: 每类K个样本
            model: 学习模型
            support_set: 支持集（训练样本）
            query_set: 查询集（测试样本）
            
        Returns:
            准确率
        """
        # 在支持集上快速学习
        model.adapt(support_set)
        
        # 在查询集上评估
        correct = 0
        for x, y in query_set:
            pred = model.predict(x)
            if pred == y:
                correct += 1
                
        accuracy = correct / len(query_set)
        
        self.experiment_results[(n_way, k_shot)].append(accuracy)
        
        return accuracy
    
    def run_benchmark(self,
                     model: Callable,
                     task_generator: Callable[[int, int], Tuple[List, List]],
                     n_episodes: int = 100) -> Dict[str, Any]:
        """
        运行少样本学习基准测试
        
        Args:
            model: 待评估模型
            task_generator: 任务生成器函数
            n_episodes: 每个配置的episode数量
            
        Returns:
            基准测试结果
        """
        results = {}
        
        for n_way in self.n_way_options:
            for k_shot in self.k_shot_options:
                episode_accuracies = []
                
                for episode in range(n_episodes):
                    support_set, query_set = task_generator(n_way, k_shot)
                    
                    # 重置模型
                    model.reset()
                    
                    # 评估episode
                    accuracy = self.evaluate_episode(
                        n_way, k_shot, model, support_set, query_set
                    )
                    episode_accuracies.append(accuracy)
                    
                key = f"{n_way}-way_{k_shot}-shot"
                results[key] = {
                    'mean': np.mean(episode_accuracies),
                    'std': np.std(episode_accuracies),
                    'min': np.min(episode_accuracies),
                    'max': np.max(episode_accuracies)
                }
                
        return results
    
    def compute_learning_speed(self, 
                              learning_curve: List[float]) -> Dict[str, float]:
        """
        计算学习速度指标
        
        Args:
            learning_curve: 学习曲线（准确率序列）
            
        Returns:
            学习速度指标
        """
        if not learning_curve or len(learning_curve) < 2:
            return {'initial_slope': 0, 'time_to_threshold': -1, 'final_performance': 0}
            
        # 初始学习斜率
        initial_slope = learning_curve[1] - learning_curve[0]
        
        # 达到阈值所需时间
        threshold = 0.8 * learning_curve[-1]
        time_to_threshold = -1
        for i, acc in enumerate(learning_curve):
            if acc >= threshold:
                time_to_threshold = i
                break
                
        return {
            'initial_slope': initial_slope,
            'time_to_threshold': time_to_threshold,
            'final_performance': learning_curve[-1],
            'total_improvement': learning_curve[-1] - learning_curve[0]
        }
    
    def compute_few_shot_capability_score(self) -> float:
        """
        计算少样本能力综合分数
        """
        if not self.experiment_results:
            return 0.0
            
        scores = []
        
        for (n_way, k_shot), accuracies in self.experiment_results.items():
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                
                # 难度加权：更多类别、更少样本 = 更难
                difficulty = (n_way / 20) * (1 / max(1, k_shot))
                weighted_score = avg_accuracy * (1 + difficulty)
                
                scores.append(weighted_score)
                
        return np.mean(scores) if scores else 0.0
    
    def analyze_shot_scaling(self) -> Dict[int, List[float]]:
        """分析样本数量对性能的影响"""
        shot_scaling = defaultdict(list)
        
        for (n_way, k_shot), accuracies in self.experiment_results.items():
            if accuracies:
                shot_scaling[k_shot].append(np.mean(accuracies))
                
        # 计算每个shot数量的平均性能
        return {k: np.mean(v) for k, v in shot_scaling.items()}
    
    def get_few_shot_report(self) -> Dict[str, Any]:
        """获取少样本学习评估报告"""
        return {
            'capability_score': self.compute_few_shot_capability_score(),
            'experiment_results': {
                f"{n}-way_{k}-shot": {
                    'mean': np.mean(accs),
                    'std': np.std(accs)
                }
                for (n, k), accs in self.experiment_results.items()
            },
            'shot_scaling': self.analyze_shot_scaling(),
            'n_experiments': sum(len(accs) for accs in self.experiment_results.values())
        }


# ============================================================================
# 元学习研究
# ============================================================================

class MetaLearningResearch:
    """
    元学习研究模块
    
    研究系统"学习如何学习"的能力，
    包括MAML、原型网络等元学习算法的评估。
    """
    
    def __init__(self):
        # 元训练历史
        self.meta_train_history: List[Dict] = []
        
        # 元测试性能
        self.meta_test_performances: List[float] = []
        
        # 任务分布学习进度
        self.task_distribution_learning: Dict[str, List[float]] = defaultdict(list)
        
        # 元学习参数演化
        self.meta_parameter_evolution: List[Dict[str, Any]] = []
        
    def evaluate_meta_learning_algorithm(self,
                                        algorithm: str,
                                        model: Any,
                                        meta_train_tasks: List[Task],
                                        meta_test_tasks: List[Task],
                                        n_meta_iterations: int = 100) -> Dict[str, Any]:
        """
        评估元学习算法
        
        Args:
            algorithm: 算法名称（'maml', 'protonet', 'relationnet'等）
            model: 元学习模型
            meta_train_tasks: 元训练任务集
            meta_test_tasks: 元测试任务集
            n_meta_iterations: 元迭代次数
            
        Returns:
            评估结果
        """
        results = {
            'algorithm': algorithm,
            'meta_train_curve': [],
            'meta_test_performances': [],
            'adaptation_speed': []
        }
        
        for iteration in range(n_meta_iterations):
            # 元训练步骤
            batch_tasks = random.sample(meta_train_tasks, 
                                       min(4, len(meta_train_tasks)))
            
            meta_loss = self._meta_train_step(model, batch_tasks, algorithm)
            results['meta_train_curve'].append(meta_loss)
            
            # 定期元测试
            if iteration % 10 == 0:
                test_perf = self._meta_test(model, meta_test_tasks, algorithm)
                results['meta_test_performances'].append(test_perf)
                
        # 计算适应速度
        results['adaptation_speed'] = self._compute_adaptation_speed(
            model, meta_test_tasks, algorithm
        )
        
        # 计算元泛化能力
        results['meta_generalization'] = self._compute_meta_generalization(
            results['meta_train_curve'],
            results['meta_test_performances']
        )
        
        self.meta_train_history.append({
            'algorithm': algorithm,
            'results': results,
            'timestamp': len(self.meta_train_history)
        })
        
        return results
    
    def _meta_train_step(self, 
                        model: Any, 
                        tasks: List[Task],
                        algorithm: str) -> float:
        """执行元训练步骤"""
        if algorithm == 'maml':
            # MAML风格的元训练
            meta_loss = 0
            for task in tasks:
                # 内循环适应
                adapted_params = self._inner_loop_adaptation(model, task)
                # 外循环损失
                task_loss = self._compute_task_loss(model, task, adapted_params)
                meta_loss += task_loss
            return meta_loss / len(tasks)
            
        elif algorithm == 'protonet':
            # 原型网络训练
            return self._train_protonet(model, tasks)
            
        else:
            return random.uniform(0.1, 1.0)
    
    def _inner_loop_adaptation(self, 
                              model: Any, 
                              task: Task,
                              n_steps: int = 5) -> Dict:
        """执行内循环适应（MAML）"""
        # 简化的适应过程
        params = {'learning_rate': 0.01}
        
        for step in range(n_steps):
            # 在支持集上计算梯度并更新
            batch = random.sample(task.train_data, 
                                 min(5, len(task.train_data)))
            # 模拟梯度更新
            params['learning_rate'] *= 0.99
            
        return params
    
    def _compute_task_loss(self, 
                          model: Any, 
                          task: Task,
                          params: Dict) -> float:
        """计算任务损失"""
        # 模拟损失计算
        return random.uniform(0.1, 0.5)
    
    def _train_protonet(self, model: Any, tasks: List[Task]) -> float:
        """训练原型网络"""
        # 计算每个任务的原型
        prototypes = {}
        for task in tasks:
            # 计算类别原型
            task_prototypes = self._compute_prototypes(task)
            prototypes[task.task_id] = task_prototypes
            
        # 计算分类损失
        loss = self._compute_protonet_loss(prototypes, tasks)
        return loss
    
    def _compute_prototypes(self, task: Task) -> Dict[Any, np.ndarray]:
        """计算任务的原型"""
        prototypes = defaultdict(list)
        
        for x, y in task.train_data:
            prototypes[y].append(np.array(x) if isinstance(x, list) else x)
            
        # 计算平均原型
        return {
            y: np.mean(embeddings, axis=0) 
            for y, embeddings in prototypes.items()
        }
    
    def _compute_protonet_loss(self, 
                              prototypes: Dict, 
                              tasks: List[Task]) -> float:
        """计算原型网络损失"""
        # 简化的损失计算
        return random.uniform(0.1, 0.5)
    
    def _meta_test(self, 
                  model: Any, 
                  test_tasks: List[Task],
                  algorithm: str) -> float:
        """执行元测试"""
        performances = []
        
        for task in test_tasks:
            # 快速适应
            if algorithm == 'maml':
                adapted_params = self._inner_loop_adaptation(model, task)
                perf = 1 - self._compute_task_loss(model, task, adapted_params)
            else:
                perf = random.uniform(0.4, 0.9)
                
            performances.append(perf)
            
        return np.mean(performances)
    
    def _compute_adaptation_speed(self,
                                  model: Any,
                                  test_tasks: List[Task],
                                  algorithm: str) -> Dict[str, float]:
        """计算适应速度"""
        adaptation_curves = []
        
        for task in test_tasks[:5]:  # 抽样测试
            curve = []
            for n_steps in [1, 3, 5, 10]:
                if algorithm == 'maml':
                    adapted_params = self._inner_loop_adaptation(model, task, n_steps)
                    perf = 1 - self._compute_task_loss(model, task, adapted_params)
                else:
                    perf = min(0.9, 0.5 + n_steps * 0.05)
                curve.append(perf)
            adaptation_curves.append(curve)
            
        avg_curve = np.mean(adaptation_curves, axis=0)
        
        return {
            'steps': [1, 3, 5, 10],
            'performances': avg_curve.tolist(),
            'initial_performance': avg_curve[0],
            'final_performance': avg_curve[-1],
            'improvement_rate': (avg_curve[-1] - avg_curve[0]) / 9
        }
    
    def _compute_meta_generalization(self,
                                    train_curve: List[float],
                                    test_performances: List[float]) -> Dict[str, float]:
        """计算元泛化能力"""
        if not train_curve or not test_performances:
            return {'generalization_gap': 0, 'stability': 0}
            
        final_train = 1 - np.mean(train_curve[-10:])  # 转换为准确率
        final_test = test_performances[-1]
        
        return {
            'generalization_gap': final_train - final_test,
            'stability': 1 - np.std(test_performances),
            'final_train_performance': final_train,
            'final_test_performance': final_test
        }
    
    def compare_algorithms(self, 
                          results: Dict[str, Dict]) -> Dict[str, Any]:
        """比较不同元学习算法"""
        comparison = {
            'algorithms': list(results.keys()),
            'final_performance': {
                alg: res['meta_test_performances'][-1] 
                if res['meta_test_performances'] else 0
                for alg, res in results.items()
            },
            'adaptation_speed': {
                alg: res['adaptation_speed']['improvement_rate']
                for alg, res in results.items()
            },
            'generalization': {
                alg: res['meta_generalization']['generalization_gap']
                for alg, res in results.items()
            }
        }
        
        # 找出最佳算法
        best_perf = max(comparison['final_performance'].items(), 
                       key=lambda x: x[1])
        comparison['best_algorithm'] = best_perf[0]
        
        return comparison
    
    def get_meta_learning_report(self) -> Dict[str, Any]:
        """获取元学习研究报告"""
        if not self.meta_train_history:
            return {'status': 'no_data'}
            
        algorithms = [h['algorithm'] for h in self.meta_train_history]
        
        return {
            'n_experiments': len(self.meta_train_history),
            'algorithms_tested': list(set(algorithms)),
            'average_meta_test_performance': np.mean([
                h['results']['meta_test_performances'][-1]
                for h in self.meta_train_history
                if h['results']['meta_test_performances']
            ]) if self.meta_train_history else 0
        }


# ============================================================================
# 灾难性遗忘研究
# ============================================================================

class CatastrophicForgettingStudy:
    """
    灾难性遗忘研究模块
    
    研究神经网络在学习新任务时遗忘旧任务的现象，
    包括遗忘度量、原因分析和缓解策略评估。
    """
    
    def __init__(self):
        # 任务序列性能
        self.task_sequence_performances: Dict[str, List[float]] = defaultdict(list)
        
        # 遗忘矩阵
        self.forgetting_matrix: Dict[Tuple[str, str], float] = {}
        
        # 参数变化追踪
        self.parameter_changes: List[Dict] = []
        
        # 缓解策略效果
        self.mitigation_results: Dict[str, Dict] = {}
        
    def evaluate_forgetting(self,
                           task_sequence: List[str],
                           performance_matrix: np.ndarray) -> Dict[str, Any]:
        """
        评估灾难性遗忘
        
        Args:
            task_sequence: 任务学习序列
            performance_matrix: 性能矩阵 (n_tasks x n_checkpoints)
            
        Returns:
            遗忘评估结果
        """
        n_tasks = len(task_sequence)
        
        results = {
            'task_sequence': task_sequence,
            'forgetting_per_task': {},
            'average_forgetting': 0,
            'backward_transfer': {},
            'forward_transfer': {}
        }
        
        total_forgetting = 0
        
        for i, task in enumerate(task_sequence):
            # 计算该任务的遗忘量
            # 遗忘 = 学习后的峰值性能 - 学习后续任务后的性能
            peak_performance = performance_matrix[i, i]
            final_performance = performance_matrix[i, -1]
            
            forgetting = max(0, peak_performance - final_performance)
            results['forgetting_per_task'][task] = {
                'peak': peak_performance,
                'final': final_performance,
                'forgetting': forgetting,
                'forgetting_rate': forgetting / peak_performance if peak_performance > 0 else 0
            }
            
            total_forgetting += forgetting
            
            # 计算前向迁移（学习当前任务对后续任务的影响）
            if i < n_tasks - 1:
                forward_transfer = []
                for j in range(i + 1, n_tasks):
                    if i > 0:
                        improvement = performance_matrix[j, i] - performance_matrix[j, i-1]
                    else:
                        improvement = performance_matrix[j, i]
                    forward_transfer.append(improvement)
                results['forward_transfer'][task] = np.mean(forward_transfer) if forward_transfer else 0
                
            # 计算后向迁移（学习后续任务对当前任务的影响）
            if i > 0:
                backward_transfer = []
                for j in range(i):
                    change = performance_matrix[j, i] - performance_matrix[j, i-1]
                    backward_transfer.append(change)
                results['backward_transfer'][task] = np.mean(backward_transfer) if backward_transfer else 0
                
        results['average_forgetting'] = total_forgetting / n_tasks if n_tasks > 0 else 0
        results['max_forgetting'] = max(
            r['forgetting'] for r in results['forgetting_per_task'].values()
        ) if results['forgetting_per_task'] else 0
        
        return results
    
    def analyze_forgetting_patterns(self, 
                                   forgetting_results: Dict) -> Dict[str, Any]:
        """分析遗忘模式"""
        patterns = {
            'severity': 'none',
            'pattern_type': 'unknown',
            'recovery_potential': 'unknown'
        }
        
        avg_forgetting = forgetting_results['average_forgetting']
        max_forgetting = forgetting_results['max_forgetting']
        
        # 判断严重程度
        if avg_forgetting > 0.5:
            patterns['severity'] = 'severe'
        elif avg_forgetting > 0.2:
            patterns['severity'] = 'moderate'
        elif avg_forgetting > 0.05:
            patterns['severity'] = 'mild'
        else:
            patterns['severity'] = 'minimal'
            
        # 分析模式类型
        task_forgettings = [
            r['forgetting_rate'] 
            for r in forgetting_results['forgetting_per_task'].values()
        ]
        
        if task_forgettings:
            variance = np.var(task_forgettings)
            if variance > 0.1:
                patterns['pattern_type'] = 'selective'
            else:
                patterns['pattern_type'] = 'uniform'
                
        return patterns
    
    def evaluate_mitigation_strategy(self,
                                    strategy: str,
                                    baseline_results: Dict,
                                    mitigated_results: Dict) -> Dict[str, Any]:
        """
        评估遗忘缓解策略
        
        Args:
            strategy: 策略名称
            baseline_results: 基线结果
            mitigated_results: 应用策略后的结果
            
        Returns:
            策略评估结果
        """
        baseline_forgetting = baseline_results['average_forgetting']
        mitigated_forgetting = mitigated_results['average_forgetting']
        
        improvement = baseline_forgetting - mitigated_forgetting
        improvement_rate = improvement / baseline_forgetting if baseline_forgetting > 0 else 0
        
        evaluation = {
            'strategy': strategy,
            'baseline_forgetting': baseline_forgetting,
            'mitigated_forgetting': mitigated_forgetting,
            'absolute_improvement': improvement,
            'improvement_rate': improvement_rate,
            'effectiveness': 'high' if improvement_rate > 0.5 else 
                           'medium' if improvement_rate > 0.2 else 'low'
        }
        
        self.mitigation_results[strategy] = evaluation
        
        return evaluation
    
    def compare_mitigation_strategies(self) -> Dict[str, Any]:
        """比较不同缓解策略"""
        if not self.mitigation_results:
            return {'status': 'no_data'}
            
        strategies = list(self.mitigation_results.keys())
        
        comparison = {
            'strategies': strategies,
            'ranking': sorted(
                strategies,
                key=lambda s: self.mitigation_results[s]['improvement_rate'],
                reverse=True
            ),
            'effectiveness': {
                s: self.mitigation_results[s]['effectiveness']
                for s in strategies
            }
        }
        
        return comparison
    
    def get_forgetting_report(self) -> Dict[str, Any]:
        """获取遗忘研究报告"""
        return {
            'n_tasks_evaluated': len(self.task_sequence_performances),
            'mitigation_strategies_tested': list(self.mitigation_results.keys()),
            'best_mitigation_strategy': (
                self.compare_mitigation_strategies().get('ranking', [None])[0]
                if self.mitigation_results else None
            )
        }


# ============================================================================
# 持续学习
# ============================================================================

class ContinualLearning:
    """
    持续学习模块
    
    实现持续学习算法和评估，包括：
    - 正则化方法（EWC, SI）
    - 回放方法
    - 动态架构方法
    - 持续学习指标
    """
    
    def __init__(self, 
                 n_tasks: int = 10,
                 memory_size: int = 100):
        self.n_tasks = n_tasks
        self.memory_size = memory_size
        
        # 记忆库
        self.memory: List[Tuple[Any, Any]] = []
        
        # 任务重要性权重（用于EWC）
        self.ewc_params: Dict[str, Dict] = {}
        
        # 参数重要性（用于SI）
        self.si_importance: Dict[str, float] = {}
        
        # 学习历史
        self.learning_history: List[Dict] = []
        
        # 当前任务ID
        self.current_task_id: int = 0
        
    def initialize_ewc(self, param_names: List[str]):
        """初始化EWC参数"""
        for name in param_names:
            self.ewc_params[name] = {
                'fisher': 0,
                'optimal_value': 0
            }
            
    def update_ewc(self, 
                  task_id: int,
                  fisher_info: Dict[str, float],
                  optimal_params: Dict[str, float]):
        """更新EWC统计"""
        for param_name, fisher in fisher_info.items():
            if param_name in self.ewc_params:
                self.ewc_params[param_name]['fisher'] = fisher
                self.ewc_params[param_name]['optimal_value'] = optimal_params.get(param_name, 0)
                
    def compute_ewc_loss(self, 
                        current_params: Dict[str, float],
                        lambda_ewc: float = 1000) -> float:
        """
        计算EWC正则化损失
        
        L_ewc = λ/2 * Σ F_i * (θ_i - θ*_i)^2
        """
        loss = 0
        for param_name, ewc_data in self.ewc_params.items():
            if param_name in current_params:
                fisher = ewc_data['fisher']
                optimal = ewc_data['optimal_value']
                current = current_params[param_name]
                
                loss += fisher * (current - optimal) ** 2
                
        return (lambda_ewc / 2) * loss
    
    def update_si_importance(self, 
                            param_changes: Dict[str, float],
                            losses: Dict[str, float]):
        """更新Synaptic Intelligence重要性"""
        for param_name, change in param_changes.items():
            if param_name not in self.si_importance:
                self.si_importance[param_name] = 0
                
            # 重要性 = 损失变化 / 参数变化
            loss_change = losses.get(param_name, 0)
            if abs(change) > 1e-8:
                importance = abs(loss_change / change)
                self.si_importance[param_name] += importance
                
    def compute_si_loss(self,
                       current_params: Dict[str, float],
                       old_params: Dict[str, float],
                       lambda_si: float = 1) -> float:
        """
        计算SI正则化损失
        
        L_si = λ * Σ Ω_i * (θ_i - θ_old_i)^2
        """
        loss = 0
        for param_name, importance in self.si_importance.items():
            if param_name in current_params and param_name in old_params:
                diff = current_params[param_name] - old_params[param_name]
                loss += importance * (diff ** 2)
                
        return lambda_si * loss
    
    def add_to_memory(self, 
                     samples: List[Tuple[Any, Any]],
                     strategy: str = 'random'):
        """
        添加样本到记忆库
        
        Args:
            samples: 样本列表
            strategy: 选择策略 ('random', 'herding', 'prototype')
        """
        if strategy == 'random':
            # 随机选择
            for sample in samples:
                if len(self.memory) < self.memory_size:
                    self.memory.append(sample)
                else:
                    # 随机替换
                    idx = random.randint(0, self.memory_size - 1)
                    self.memory[idx] = sample
                    
        elif strategy == 'herding':
            # 基于herding的选择（简化版）
            self.memory.extend(samples[:self.memory_size - len(self.memory)])
            self.memory = self.memory[:self.memory_size]
            
    def sample_from_memory(self, n_samples: int) -> List[Tuple[Any, Any]]:
        """从记忆库中采样"""
        if not self.memory:
            return []
            
        n = min(n_samples, len(self.memory))
        return random.sample(self.memory, n)
    
    def compute_replay_loss(self,
                           model: Any,
                           current_batch: List[Tuple[Any, Any]],
                           replay_ratio: float = 0.3) -> float:
        """
        计算回放损失
        
        结合当前任务样本和记忆库样本
        """
        # 当前任务损失
        current_loss = self._compute_batch_loss(model, current_batch)
        
        # 回放损失
        n_replay = int(len(current_batch) * replay_ratio)
        replay_samples = self.sample_from_memory(n_replay)
        
        if replay_samples:
            replay_loss = self._compute_batch_loss(model, replay_samples)
            return (1 - replay_ratio) * current_loss + replay_ratio * replay_loss
        else:
            return current_loss
    
    def _compute_batch_loss(self, 
                           model: Any, 
                           batch: List[Tuple[Any, Any]]) -> float:
        """计算批次损失"""
        if not batch:
            return 0.0
            
        # 模拟损失计算
        return random.uniform(0.1, 0.5)
    
    def evaluate_continual_learning(self,
                                   task_performances: List[List[float]]) -> Dict[str, Any]:
        """
        评估持续学习性能
        
        Args:
            task_performances: 每个任务在每个时间点的性能
            
        Returns:
            评估结果
        """
        n_tasks = len(task_performances)
        
        metrics = {
            'average_accuracy': 0,
            'backward_transfer': 0,
            'forward_transfer': 0,
            'forgetting': 0,
            'learning_curve': []
        }
        
        # 计算平均准确率（最终时刻所有任务的平均）
        final_accuracies = [perf[-1] for perf in task_performances]
        metrics['average_accuracy'] = np.mean(final_accuracies)
        
        # 计算遗忘
        total_forgetting = 0
        for i in range(n_tasks):
            peak = max(task_performances[i])
            final = task_performances[i][-1]
            forgetting = max(0, peak - final)
            total_forgetting += forgetting
            
        metrics['forgetting'] = total_forgetting / n_tasks if n_tasks > 0 else 0
        
        # 计算迁移
        backward_transfers = []
        forward_transfers = []
        
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                # 前向迁移：任务i对任务j的影响
                if i < len(task_performances[j]):
                    ft = task_performances[j][i] - task_performances[j][0]
                    forward_transfers.append(ft)
                    
        for i in range(1, n_tasks):
            for j in range(i):
                # 后向迁移：任务i对任务j的影响
                if i < len(task_performances[j]):
                    bt = task_performances[j][i] - task_performances[j][i-1]
                    backward_transfers.append(bt)
                    
        metrics['forward_transfer'] = np.mean(forward_transfers) if forward_transfers else 0
        metrics['backward_transfer'] = np.mean(backward_transfers) if backward_transfers else 0
        
        return metrics
    
    def get_continual_learning_report(self) -> Dict[str, Any]:
        """获取持续学习报告"""
        return {
            'memory_size': self.memory_size,
            'memory_utilization': len(self.memory) / self.memory_size,
            'n_tasks_learned': self.current_task_id,
            'ewc_params_tracked': len(self.ewc_params),
            'si_params_tracked': len(self.si_importance)
        }


# ============================================================================
# 综合AGI评估系统
# ============================================================================

class AGIEvaluator:
    """
    综合AGI评估系统

    整合所有AGI研究模块，提供统一的评估接口。

    重构说明：
    - 使用UnifiedMoE进行多专家路由
    - 使用UnifiedConstraintSystem管理评估约束
    """

    def __init__(self):
        self.gi_metrics = GeneralIntelligenceMetrics()
        self.transfer_evaluator = TransferLearningEvaluator()
        self.few_shot_evaluator = FewShotEvaluator()
        self.meta_learning = MetaLearningResearch()
        self.forgetting_study = CatastrophicForgettingStudy()
        self.continual_learning = ContinualLearning()

        self.evaluation_history: List[AGIReport] = []

        # 初始化统一MoE系统
        self._init_unified_moe()

        # 初始化统一约束系统
        self._init_constraint_system()

    def _init_unified_moe(self):
        """初始化统一MoE系统用于专家路由"""
        config = UnifiedAlgorithmConfig.default_config()

        # 创建专家
        experts = [
            UnifiedExpert(
                expert_id="gi_expert",
                expert_type=ExpertType.STANDARD,
                capacity=1.0,
                specialization_score=0.9
            ),
            UnifiedExpert(
                expert_id="transfer_expert",
                expert_type=ExpertType.STANDARD,
                capacity=1.0,
                specialization_score=0.85
            ),
            UnifiedExpert(
                expert_id="few_shot_expert",
                expert_type=ExpertType.STANDARD,
                capacity=1.0,
                specialization_score=0.9
            ),
            UnifiedExpert(
                expert_id="meta_expert",
                expert_type=ExpertType.ADAPTIVE,
                capacity=1.0,
                specialization_score=0.8
            ),
            UnifiedExpert(
                expert_id="forgetting_expert",
                expert_type=ExpertType.STANDARD,
                capacity=1.0,
                specialization_score=0.85
            ),
            UnifiedExpert(
                expert_id="continual_expert",
                expert_type=ExpertType.STANDARD,
                capacity=1.0,
                specialization_score=0.85
            ),
        ]

        # 创建路由器
        router = UnifiedRouter(
            strategy=RoutingStrategy.CAPACITY_AWARE,
            top_k=3,
            config=config
        )

        # 创建MoE系统
        self.moe = UnifiedMoE(
            experts=experts,
            router=router,
            config=config
        )

    def _init_constraint_system(self):
        """初始化统一约束系统"""
        self.constraint_system = UnifiedConstraintSystem()

        # 添加评估约束
        constraints = [
            UnifiedConstraint(
                constraint_id="min_samples",
                constraint_type=ConstraintType.RESOURCE,
                priority=ConstraintPriority.HIGH,
                condition=lambda ctx: ctx.get('n_samples', 0) >= 10,
                violation_penalty=0.5
            ),
            UnifiedConstraint(
                constraint_id="max_eval_time",
                constraint_type=ConstraintType.TEMPORAL,
                priority=ConstraintPriority.MEDIUM,
                condition=lambda ctx: ctx.get('eval_time', 0) < 3600,
                violation_penalty=0.3
            ),
            UnifiedConstraint(
                constraint_id="domain_coverage",
                constraint_type=ConstraintType.SEMANTIC,
                priority=ConstraintPriority.HIGH,
                condition=lambda ctx: len(ctx.get('domains', [])) >= 3,
                violation_penalty=0.4
            ),
        ]

        for constraint in constraints:
            self.constraint_system.add_constraint(constraint)
        
    def comprehensive_evaluation(self,
                                model: Any,
                                task_suite: List[Task]) -> AGIReport:
        """
        执行综合AGI评估

        使用统一MoE系统和约束系统进行智能评估。

        Args:
            model: 待评估模型
            task_suite: 任务套件

        Returns:
            AGI评估报告
        """
        # 准备评估上下文
        eval_context = {
            'n_samples': sum(len(t.train_data) for t in task_suite),
            'eval_time': 0,
            'domains': list(set(t.domain for t in task_suite)),
            'n_tasks': len(task_suite)
        }

        # 检查约束
        constraint_result = self.constraint_system.check_all(eval_context)
        if not constraint_result.is_valid:
            print(f"警告: 约束违反 - {constraint_result.violations}")

        # 使用MoE路由评估任务
        expert_outputs = []

        # 1. 通用智能评估（路由到gi_expert）
        gi_expert = self.moe.get_expert("gi_expert")
        if gi_expert:
            for task in task_suite:
                self.gi_metrics.evaluate_task(
                    task,
                    model.predict if hasattr(model, 'predict') else model,
                    model.train if hasattr(model, 'train') else lambda m, d: None
                )
            gi_score = self.gi_metrics.compute_general_intelligence_score()
            expert_outputs.append(("gi_expert", gi_score))

        domain_scores = {
            domain: np.mean(scores) if scores else 0
            for domain, scores in self.gi_metrics.domain_performances.items()
        }

        # 2. 迁移学习评估（路由到transfer_expert）
        transfer_expert = self.moe.get_expert("transfer_expert")
        if transfer_expert:
            transfer_report = self.transfer_evaluator.get_transfer_report()
            transfer_efficiency = transfer_report.get('average_transfer', {}).get('average_transfer_effect', 0)
            expert_outputs.append(("transfer_expert", max(0, transfer_efficiency)))
        else:
            transfer_efficiency = 0
            transfer_report = {}

        # 3. 少样本学习评估（路由到few_shot_expert）
        few_shot_expert = self.moe.get_expert("few_shot_expert")
        if few_shot_expert:
            few_shot_report = self.few_shot_evaluator.get_few_shot_report()
            few_shot_capability = few_shot_report.get('capability_score', 0)
            expert_outputs.append(("few_shot_expert", few_shot_capability))
        else:
            few_shot_capability = 0
            few_shot_report = {}

        # 4. 元学习评估（路由到meta_expert）
        meta_expert = self.moe.get_expert("meta_expert")
        if meta_expert:
            meta_report = self.meta_learning.get_meta_learning_report()
            meta_score = meta_report.get('average_meta_test_performance', 0)
            expert_outputs.append(("meta_expert", meta_score))
        else:
            meta_score = 0
            meta_report = {}

        # 5. 灾难性遗忘评估（路由到forgetting_expert）
        forgetting_expert = self.moe.get_expert("forgetting_expert")
        if forgetting_expert:
            forgetting_report = self.forgetting_study.get_forgetting_report()
            forgetting_rate = 0.1  # 默认值
            expert_outputs.append(("forgetting_expert", 1 - forgetting_rate))
        else:
            forgetting_rate = 0.1
            forgetting_report = {}

        # 6. 持续学习评估（路由到continual_expert）
        continual_expert = self.moe.get_expert("continual_expert")
        if continual_expert:
            continual_report = self.continual_learning.get_continual_learning_report()
            continual_score = continual_report.get('memory_utilization', 0)
            expert_outputs.append(("continual_expert", continual_score))
        else:
            continual_score = 0
            continual_report = {}

        # 使用MoE融合专家输出
        if expert_outputs:
            # 计算加权平均（基于专家容量）
            total_weight = 0
            weighted_sum = 0
            weights = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]  # 对应各组件权重

            for i, (expert_id, output) in enumerate(expert_outputs):
                expert = self.moe.get_expert(expert_id)
                if expert:
                    weight = weights[i] * expert.capacity
                    weighted_sum += output * weight
                    total_weight += weight

            overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            # 回退到传统计算
            overall_score = (
                0.25 * gi_score +
                0.15 * max(0, transfer_efficiency) +
                0.20 * few_shot_capability +
                0.15 * meta_score +
                0.15 * (1 - forgetting_rate) +
                0.10 * continual_score
            )

        report = AGIReport(
            overall_score=overall_score,
            domain_scores=domain_scores,
            transfer_efficiency=transfer_efficiency,
            few_shot_capability=few_shot_capability,
            meta_learning_score=meta_score,
            forgetting_rate=forgetting_rate,
            continual_learning_score=continual_score,
            timestamp=len(self.evaluation_history),
            details={
                'general_intelligence': self.gi_metrics.get_capability_profile(),
                'transfer_learning': transfer_report,
                'few_shot': few_shot_report,
                'meta_learning': meta_report,
                'forgetting': forgetting_report,
                'continual_learning': continual_report,
                'expert_outputs': {k: v for k, v in expert_outputs},
                'constraint_violations': constraint_result.violations if not constraint_result.is_valid else []
            }
        )

        self.evaluation_history.append(report)

        return report
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {'status': 'no_evaluations'}
            
        recent_reports = self.evaluation_history[-10:]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_overall_score': np.mean([r.overall_score for r in recent_reports]),
            'score_trend': 'improving' if len(recent_reports) > 1 and 
                          recent_reports[-1].overall_score > recent_reports[0].overall_score 
                          else 'stable',
            'best_domain': max(self.evaluation_history[-1].domain_scores.items(),
                             key=lambda x: x[1])[0].value,
            'weakest_domain': min(self.evaluation_history[-1].domain_scores.items(),
                                key=lambda x: x[1])[0].value
        }


# ============================================================================
# 演示与测试
# ============================================================================

def run_agi_demo():
    """运行AGI研究演示"""
    print("=" * 70)
    print("AGI研究模块演示")
    print("=" * 70)
    
    # 创建模拟模型
    class MockModel:
        def __init__(self):
            self.params = {'weight': 0.5}
            
        def predict(self, x):
            return random.choice([0, 1])
            
        def train(self, model, data):
            self.params['weight'] += 0.01
            
        def adapt(self, support_set):
            pass
            
        def reset(self):
            self.params['weight'] = 0.5
    
    model = MockModel()
    
    # 1. 通用智能指标演示
    print("\n[1. 通用智能指标评估]")
    gi_metrics = GeneralIntelligenceMetrics()
    
    # 创建模拟任务
    tasks = []
    for i, domain in enumerate(TaskDomain):
        task = Task(
            task_id=f'task_{domain.value}_{i}',
            domain=domain,
            difficulty=random.uniform(0.3, 0.9),
            train_data=[(j, j % 2) for j in range(50)],
            test_data=[(j, j % 2) for j in range(50, 70)],
            eval_metric='accuracy'
        )
        tasks.append(task)
        
        # 评估任务
        perf = gi_metrics.evaluate_task(task, model.predict, model.train)
        print(f"  {task.task_id}: 分数={perf.score:.3f}, 难度={task.difficulty:.2f}")
        
    gi_score = gi_metrics.compute_general_intelligence_score()
    print(f"\n  通用智能分数: {gi_score:.3f}")
    print(f"  领域平衡度: {gi_metrics.compute_domain_balance()}")
    
    # 2. 迁移学习演示
    print("\n[2. 迁移学习评估]")
    transfer_eval = TransferLearningEvaluator()
    
    # 模拟迁移学习实验
    source_tasks = ['task_a', 'task_b', 'task_c']
    target_tasks = ['task_x', 'task_y']
    
    for src in source_tasks:
        transfer_eval.record_source_performance(src, random.uniform(0.7, 0.9))
        
    for tgt in target_tasks:
        transfer_eval.record_target_baseline(tgt, random.uniform(0.4, 0.6))
        
    for src in source_tasks:
        for tgt in target_tasks:
            transfer_eval.record_transfer_performance(
                src, tgt, random.uniform(0.5, 0.8)
            )
            
    transfer_report = transfer_eval.get_transfer_report()
    print(f"  平均迁移效果: {transfer_report['average_transfer']['average_transfer_effect']:.3f}")
    print(f"  正向迁移率: {transfer_report['average_transfer']['positive_transfer_rate']:.2%}")
    
    # 3. 少样本学习演示
    print("\n[3. 少样本学习能力评估]")
    few_shot_eval = FewShotEvaluator()
    
    # 模拟少样本实验
    for n_way in [5, 10]:
        for k_shot in [1, 5]:
            for _ in range(20):
                accuracy = random.uniform(0.3 + k_shot * 0.05, 0.9)
                few_shot_eval.experiment_results[(n_way, k_shot)].append(accuracy)
                
    few_shot_report = few_shot_eval.get_few_shot_report()
    print(f"  少样本能力分数: {few_shot_report['capability_score']:.3f}")
    print(f"  实验结果: {few_shot_report['experiment_results']}")
    
    # 4. 元学习演示
    print("\n[4. 元学习研究]")
    meta_learning = MetaLearningResearch()
    
    # 模拟元学习实验
    meta_tasks = [Task(f'meta_{i}', TaskDomain.VISION, 0.5, [], []) 
                  for i in range(10)]
    
    results = meta_learning.evaluate_meta_learning_algorithm(
        'maml', model, meta_tasks[:7], meta_tasks[7:], n_meta_iterations=50
    )
    print(f"  算法: {results['algorithm']}")
    print(f"  最终元测试性能: {results['meta_test_performances'][-1]:.3f}")
    print(f"  适应速度: {results['adaptation_speed']['improvement_rate']:.4f}")
    
    # 5. 灾难性遗忘演示
    print("\n[5. 灾难性遗忘研究]")
    forgetting_study = CatastrophicForgettingStudy()
    
    # 模拟遗忘实验
    task_sequence = ['task_1', 'task_2', 'task_3', 'task_4']
    performance_matrix = np.array([
        [0.9, 0.85, 0.70, 0.65],  # task_1
        [0.0, 0.88, 0.82, 0.75],  # task_2
        [0.0, 0.0, 0.85, 0.78],   # task_3
        [0.0, 0.0, 0.0, 0.87]     # task_4
    ])
    
    forgetting_results = forgetting_study.evaluate_forgetting(
        task_sequence, performance_matrix
    )
    print(f"  平均遗忘: {forgetting_results['average_forgetting']:.3f}")
    print(f"  最大遗忘: {forgetting_results['max_forgetting']:.3f}")
    
    for task, data in forgetting_results['forgetting_per_task'].items():
        print(f"    {task}: 遗忘率={data['forgetting_rate']:.2%}")
        
    # 6. 持续学习演示
    print("\n[6. 持续学习]")
    continual = ContinualLearning(n_tasks=5, memory_size=50)
    
    # 模拟持续学习
    for i in range(5):
        samples = [(j, j % 2) for j in range(10)]
        continual.add_to_memory(samples, strategy='random')
        continual.current_task_id += 1
        
    continual_report = continual.get_continual_learning_report()
    print(f"  记忆库利用率: {continual_report['memory_utilization']:.1%}")
    print(f"  学习任务数: {continual_report['n_tasks_learned']}")
    
    # 7. 综合评估
    print("\n[7. 综合AGI评估]")
    agi_evaluator = AGIEvaluator()
    
    # 使用之前创建的任务
    report = agi_evaluator.comprehensive_evaluation(model, tasks)
    
    print(f"  总体分数: {report.overall_score:.3f}")
    print(f"  迁移效率: {report.transfer_efficiency:.3f}")
    print(f"  少样本能力: {report.few_shot_capability:.3f}")
    print(f"  元学习分数: {report.meta_learning_score:.3f}")
    print(f"  遗忘率: {report.forgetting_rate:.3f}")
    print(f"  持续学习分数: {report.continual_learning_score:.3f}")
    
    print("\n  领域分数:")
    for domain, score in report.domain_scores.items():
        print(f"    {domain.value}: {score:.3f}")
        
    # 打印评估摘要
    print("\n" + "=" * 70)
    print("评估摘要")
    print("=" * 70)
    summary = agi_evaluator.get_evaluation_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run_agi_demo()
