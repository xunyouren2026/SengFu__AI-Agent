"""
跨Agent微调 - 用其他Agent的数据微调本地模型

实现跨Agent的模型微调机制，允许Agent利用其他Agent收集的
数据来改进自己的模型，同时处理数据分布差异问题。
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random
import math


@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    input_data: Any
    target: Any
    source_agent: str
    timestamp: float
    
    # 质量指标
    confidence: float = 1.0
    difficulty: float = 0.5
    
    # 元数据
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAdapter:
    """模型适配器"""
    agent_id: str
    adapter_params: Dict[str, float] = field(default_factory=dict)
    feature_transform: Optional[Callable[[Any], Any]] = None


class DomainAdaptation:
    """域适应 - 处理不同Agent间的数据分布差异"""
    
    def __init__(self):
        self.source_stats: Dict[str, Dict[str, Any]] = {}
        self.target_stats: Dict[str, Dict[str, Any]] = {}
        self.adaptation_weights: Dict[Tuple[str, str], float] = {}
        
    def compute_feature_statistics(self, agent_id: str, 
                                   samples: List[TrainingSample]) -> Dict[str, Any]:
        """计算特征统计"""
        if not samples:
            return {}
        
        # 提取数值特征
        features = []
        for sample in samples:
            if isinstance(sample.input_data, (list, tuple)):
                features.append([float(x) for x in sample.input_data])
            elif isinstance(sample.input_data, dict):
                features.append([float(v) for v in sample.input_data.values() 
                               if isinstance(v, (int, float))])
        
        if not features:
            return {}
        
        # 计算均值和方差
        num_dims = len(features[0])
        means = [sum(f[i] for f in features) / len(features) for i in range(num_dims)]
        variances = [
            sum((f[i] - means[i]) ** 2 for f in features) / len(features)
            for i in range(num_dims)
        ]
        
        stats = {
            'mean': means,
            'variance': variances,
            'count': len(samples)
        }
        
        self.source_stats[agent_id] = stats
        return stats
    
    def normalize_sample(self, sample: TrainingSample, 
                        source_agent: str, target_agent: str) -> TrainingSample:
        """归一化样本以适配目标域"""
        if source_agent not in self.source_stats or target_agent not in self.target_stats:
            return sample
        
        source_stats = self.source_stats[source_agent]
        target_stats = self.target_stats[target_agent]
        
        # 标准化变换
        if isinstance(sample.input_data, (list, tuple)):
            normalized = []
            for i, val in enumerate(sample.input_data):
                if i < len(source_stats['mean']):
                    # Z-score标准化后再根据目标域调整
                    z_score = (val - source_stats['mean'][i]) / (math.sqrt(source_stats['variance'][i]) + 1e-8)
                    adapted = target_stats['mean'][i] + z_score * math.sqrt(target_stats['variance'][i])
                    normalized.append(adapted)
                else:
                    normalized.append(val)
            
            # 创建新样本
            new_sample = TrainingSample(
                sample_id=sample.sample_id + "_adapted",
                input_data=normalized,
                target=sample.target,
                source_agent=source_agent,
                timestamp=sample.timestamp,
                confidence=sample.confidence,
                difficulty=sample.difficulty,
                tags=sample.tags | {'domain_adapted'},
                metadata={**sample.metadata, 'adaptation': 'z_score'}
            )
            return new_sample
        
        return sample
    
    def compute_domain_similarity(self, agent1: str, agent2: str) -> float:
        """计算域相似度"""
        if agent1 not in self.source_stats or agent2 not in self.source_stats:
            return 0.5  # 默认中等相似度
        
        stats1 = self.source_stats[agent1]
        stats2 = self.source_stats[agent2]
        
        # 使用均值差异计算相似度
        mean_diff = sum((m1 - m2) ** 2 
                       for m1, m2 in zip(stats1['mean'], stats2['mean']))
        
        similarity = 1.0 / (1.0 + math.sqrt(mean_diff))
        
        self.adaptation_weights[(agent1, agent2)] = similarity
        self.adaptation_weights[(agent2, agent1)] = similarity
        
        return similarity


class CrossAgentFineTuner:
    """跨Agent微调器"""
    
    def __init__(self, local_agent_id: str):
        self.local_agent_id = local_agent_id
        self.local_samples: List[TrainingSample] = []
        self.external_samples: Dict[str, List[TrainingSample]] = defaultdict(list)
        
        # 域适应
        self.domain_adaptation = DomainAdaptation()
        
        # 样本权重
        self.sample_weights: Dict[str, float] = {}
        
        # 模型参数
        self.model_params: Dict[str, float] = {}
        self.learning_rate = 0.01
        
        # 信任度
        self.agent_trust: Dict[str, float] = defaultdict(lambda: 0.5)
        
    def add_local_sample(self, sample: TrainingSample):
        """添加本地样本"""
        sample.source_agent = self.local_agent_id
        self.local_samples.append(sample)
        self.sample_weights[sample.sample_id] = 1.0
    
    def add_external_samples(self, agent_id: str, samples: List[TrainingSample]):
        """添加外部Agent样本"""
        # 计算域统计
        self.domain_adaptation.compute_feature_statistics(agent_id, samples)
        self.domain_adaptation.target_stats[self.local_agent_id] = \
            self.domain_adaptation.compute_feature_statistics(
                self.local_agent_id, self.local_samples
            )
        
        # 域适应
        adapted_samples = []
        for sample in samples:
            adapted = self.domain_adaptation.normalize_sample(
                sample, agent_id, self.local_agent_id
            )
            adapted_samples.append(adapted)
            
            # 根据域相似度设置权重
            similarity = self.domain_adaptation.compute_domain_similarity(
                agent_id, self.local_agent_id
            )
            self.sample_weights[adapted.sample_id] = similarity * self.agent_trust[agent_id]
        
        self.external_samples[agent_id].extend(adapted_samples)
    
    def update_trust(self, agent_id: str, performance_delta: float):
        """更新对其他Agent的信任度"""
        # 基于性能变化更新信任
        current_trust = self.agent_trust[agent_id]
        new_trust = current_trust + 0.1 * performance_delta
        self.agent_trust[agent_id] = max(0.0, min(1.0, new_trust))
    
    def compute_sample_importance(self, sample: TrainingSample) -> float:
        """计算样本重要性"""
        base_weight = self.sample_weights.get(sample.sample_id, 1.0)
        
        # 考虑样本难度
        difficulty_factor = 1.0 + sample.difficulty
        
        # 考虑置信度
        confidence_factor = sample.confidence
        
        # 考虑时效性
        # (简化处理，假设timestamp是数值)
        time_factor = 1.0
        
        return base_weight * difficulty_factor * confidence_factor * time_factor
    
    def fine_tune(self, epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        """
        执行微调
        
        结合本地和外部数据进行微调
        """
        # 合并所有样本
        all_samples = self.local_samples.copy()
        for samples in self.external_samples.values():
            all_samples.extend(samples)
        
        if not all_samples:
            return {'status': 'no_data', 'loss': float('inf')}
        
        # 按重要性加权采样
        weights = [self.compute_sample_importance(s) for s in all_samples]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # 训练
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # 随机打乱并按概率采样
            indices = list(range(len(all_samples)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_samples = [all_samples[j] for j in batch_indices]
                batch_weights = [weights[j] for j in batch_indices]
                
                # 训练步骤
                loss = self._train_step(batch_samples, batch_weights)
                epoch_loss += loss
            
            losses.append(epoch_loss / max(1, len(indices) // batch_size))
        
        return {
            'status': 'success',
            'final_loss': losses[-1],
            'loss_history': losses,
            'total_samples': len(all_samples),
            'local_samples': len(self.local_samples),
            'external_samples': sum(len(s) for s in self.external_samples.values())
        }
    
    def _train_step(self, samples: List[TrainingSample], weights: List[float]) -> float:
        """训练步骤"""
        loss = 0.0
        
        for sample, weight in zip(samples, weights):
            # 预测
            prediction = self._predict(sample.input_data)
            
            # 计算加权损失
            sample_loss = self._compute_loss(prediction, sample.target)
            weighted_loss = sample_loss * weight
            
            # 更新参数
            self._update_params(sample.input_data, sample.target, weight)
            
            loss += weighted_loss
        
        return loss / len(samples) if samples else 0.0
    
    def _predict(self, input_data: Any) -> Any:
        """模型预测"""
        # 简化的线性模型
        if isinstance(input_data, (list, tuple)):
            return sum(self.model_params.get(f'w{i}', 0.1) * x 
                      for i, x in enumerate(input_data))
        return 0.0
    
    def _compute_loss(self, prediction: Any, target: Any) -> float:
        """计算损失"""
        if isinstance(target, (int, float)) and isinstance(prediction, (int, float)):
            return (prediction - float(target)) ** 2
        return 0.0 if prediction == target else 1.0
    
    def _update_params(self, input_data: Any, target: Any, weight: float):
        """更新参数"""
        if isinstance(input_data, (list, tuple)) and isinstance(target, (int, float)):
            prediction = self._predict(input_data)
            error = prediction - target
            
            for i, x in enumerate(input_data):
                key = f'w{i}'
                current = self.model_params.get(key, 0.1)
                gradient = 2 * error * x * weight
                self.model_params[key] = current - self.learning_rate * gradient
    
    def evaluate_sample_contribution(self, sample: TrainingSample) -> float:
        """评估单个样本对模型的贡献"""
        # 计算移除该样本前后的性能差异
        # (简化实现)
        return self.compute_sample_importance(sample)
    
    def select_valuable_external_samples(self, agent_id: str, 
                                         max_samples: int = 100) -> List[TrainingSample]:
        """选择有价值的外部样本"""
        if agent_id not in self.external_samples:
            return []
        
        samples = self.external_samples[agent_id]
        
        # 计算每个样本的贡献
        scored_samples = [
            (self.evaluate_sample_contribution(s), s) 
            for s in samples
        ]
        
        # 按贡献排序并选择
        scored_samples.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored_samples[:max_samples]]


class FederatedDataAggregator:
    """联邦数据聚合器 - 聚合多个Agent的数据"""
    
    def __init__(self):
        self.agent_data: Dict[str, List[TrainingSample]] = {}
        self.aggregation_weights: Dict[str, float] = {}
        self.quality_scores: Dict[str, float] = {}
        
    def register_agent_data(self, agent_id: str, 
                           samples: List[TrainingSample],
                           quality_score: float = 1.0):
        """注册Agent数据"""
        self.agent_data[agent_id] = samples
        self.quality_scores[agent_id] = quality_score
        
        # 初始均匀权重
        self._update_aggregation_weights()
    
    def _update_aggregation_weights(self):
        """更新聚合权重"""
        total_quality = sum(self.quality_scores.values())
        if total_quality > 0:
            for agent_id in self.agent_data:
                self.aggregation_weights[agent_id] = \
                    self.quality_scores[agent_id] / total_quality
    
    def aggregate_gradient(self, gradients: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        聚合梯度
        
        使用加权平均聚合来自不同Agent的梯度
        """
        if not gradients:
            return {}
        
        # 获取所有参数
        all_params = set()
        for grad in gradients.values():
            all_params.update(grad.keys())
        
        # 加权聚合
        aggregated = {}
        for param in all_params:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for agent_id, grad in gradients.items():
                if param in grad:
                    weight = self.aggregation_weights.get(agent_id, 1.0)
                    weighted_sum += grad[param] * weight
                    total_weight += weight
            
            if total_weight > 0:
                aggregated[param] = weighted_sum / total_weight
        
        return aggregated
    
    def aggregate_data(self, max_samples_per_agent: int = 1000) -> List[TrainingSample]:
        """聚合数据样本"""
        aggregated = []
        
        for agent_id, samples in self.agent_data.items():
            weight = self.aggregation_weights.get(agent_id, 1.0)
            
            # 根据权重决定采样数量
            n_samples = int(max_samples_per_agent * weight)
            n_samples = min(n_samples, len(samples))
            
            # 随机采样
            selected = random.sample(samples, n_samples)
            
            # 标记来源
            for sample in selected:
                sample.tags.add(f'aggregated_from_{agent_id}')
                sample.metadata['aggregation_weight'] = weight
            
            aggregated.extend(selected)
        
        return aggregated
    
    def update_quality_scores(self, agent_id: str, validation_performance: float):
        """更新数据质量分数"""
        # 基于验证性能更新质量分数
        current = self.quality_scores.get(agent_id, 0.5)
        new_score = 0.7 * current + 0.3 * validation_performance
        self.quality_scores[agent_id] = max(0.0, min(1.0, new_score))
        
        self._update_aggregation_weights()


class TransferLearningManager:
    """迁移学习管理器"""
    
    def __init__(self):
        self.source_models: Dict[str, Dict[str, float]] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self.transfer_effectiveness: Dict[Tuple[str, str], float] = {}
        
    def register_source_model(self, agent_id: str, model_params: Dict[str, float]):
        """注册源模型"""
        self.source_models[agent_id] = model_params.copy()
    
    def compute_transferability(self, source_agent: str, 
                                target_samples: List[TrainingSample]) -> float:
        """计算可迁移性分数"""
        if source_agent not in self.source_models:
            return 0.0
        
        # 基于历史效果
        history_key = (source_agent, 'current_target')
        if history_key in self.transfer_effectiveness:
            return self.transfer_effectiveness[history_key]
        
        # 基于数据相似度估计
        source_model = self.source_models[source_agent]
        
        # 简化的相似度: 测试源模型在目标数据上的性能
        predictions = []
        targets = []
        
        for sample in target_samples[:100]:  # 使用子集
            if isinstance(sample.input_data, (list, tuple)):
                pred = sum(source_model.get(f'w{i}', 0.0) * x 
                          for i, x in enumerate(sample.input_data))
                predictions.append(pred)
                if isinstance(sample.target, (int, float)):
                    targets.append(float(sample.target))
        
        if predictions and targets:
            # 计算性能作为可迁移性指标
            mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
            transferability = 1.0 / (1.0 + mse)
        else:
            transferability = 0.5
        
        return transferability
    
    def transfer_parameters(self, source_agent: str, 
                           target_params: Dict[str, float],
                           transfer_ratio: float = 0.3) -> Dict[str, float]:
        """迁移参数"""
        if source_agent not in self.source_models:
            return target_params
        
        source_params = self.source_models[source_agent]
        
        # 参数融合
        merged = target_params.copy()
        for key, source_value in source_params.items():
            if key in merged:
                # 加权平均
                merged[key] = (1 - transfer_ratio) * merged[key] + transfer_ratio * source_value
            else:
                merged[key] = source_value * transfer_ratio
        
        return merged
    
    def record_transfer_result(self, source_agent: str, target_agent: str,
                               performance_before: float, performance_after: float):
        """记录迁移结果"""
        improvement = performance_after - performance_before
        
        self.transfer_history.append({
            'source_agent': source_agent,
            'target_agent': target_agent,
            'performance_before': performance_before,
            'performance_after': performance_after,
            'improvement': improvement
        })
        
        # 更新效果记录
        key = (source_agent, target_agent)
        current_effectiveness = self.transfer_effectiveness.get(key, 0.5)
        self.transfer_effectiveness[key] = 0.8 * current_effectiveness + 0.2 * max(0, improvement)


class DataPrivacyFilter:
    """数据隐私过滤器"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # 差分隐私参数
        
    def add_noise(self, sample: TrainingSample) -> TrainingSample:
        """添加差分隐私噪声"""
        if not isinstance(sample.input_data, (list, tuple)):
            return sample
        
        # 拉普拉斯噪声
        noisy_data = []
        for val in sample.input_data:
            if isinstance(val, (int, float)):
                noise = random.expovariate(self.epsilon) * (1 if random.random() > 0.5 else -1)
                noisy_data.append(val + noise)
            else:
                noisy_data.append(val)
        
        return TrainingSample(
            sample_id=sample.sample_id + "_noisy",
            input_data=noisy_data,
            target=sample.target,
            source_agent=sample.source_agent,
            timestamp=sample.timestamp,
            confidence=sample.confidence * 0.9,  # 降低置信度
            difficulty=sample.difficulty,
            tags=sample.tags | {'privacy_noisy'},
            metadata={**sample.metadata, 'privacy_epsilon': self.epsilon}
        )
    
    def filter_sensitive_features(self, sample: TrainingSample, 
                                  sensitive_indices: Set[int]) -> TrainingSample:
        """过滤敏感特征"""
        if not isinstance(sample.input_data, (list, tuple)):
            return sample
        
        filtered_data = []
        for i, val in enumerate(sample.input_data):
            if i in sensitive_indices:
                # 用平均值或噪声替换
                filtered_data.append(0.0)
            else:
                filtered_data.append(val)
        
        return TrainingSample(
            sample_id=sample.sample_id + "_filtered",
            input_data=filtered_data,
            target=sample.target,
            source_agent=sample.source_agent,
            timestamp=sample.timestamp,
            tags=sample.tags | {'privacy_filtered'},
            metadata={**sample.metadata, 'filtered_indices': list(sensitive_indices)}
        )
