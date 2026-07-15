"""
TTT层 - Test-Time Training Layer

测试时训练，通过"惊喜指标"识别重要信息，动态更新长期记忆

作者: UFO Framework Team
"""

import math
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class TTTConfig:
    """TTT配置"""
    hidden_size: int = 768
    memory_size: int = 512
    learning_rate: float = 0.01
    surprise_threshold: float = 0.5
    update_steps: int = 3
    momentum: float = 0.9


class SurpriseMetric:
    """
    惊喜指标计算器
    
    衡量输入的"惊讶程度"
    """
    
    def __init__(
        self,
        history_size: int = 100,
        threshold: float = 0.5
    ):
        self.history_size = history_size
        self.threshold = threshold
        
        # 预测历史
        self.predictions: List[np.ndarray] = []
        self.actuals: List[np.ndarray] = []
        
        # 统计
        self.stats = {
            'total_measurements': 0,
            'avg_surprise': 0.0,
            'surprising_events': 0
        }
    
    def measure(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        计算惊喜度
        
        Args:
            predicted: 预测值
            actual: 实际值
            
        Returns:
            惊喜度分数 [0, 1]
        """
        self.stats['total_measurements'] += 1
        
        # 更新历史
        self.predictions.append(predicted.flatten())
        self.actuals.append(actual.flatten())
        
        if len(self.predictions) > self.history_size:
            self.predictions.pop(0)
            self.actuals.pop(0)
        
        # 计算预测误差
        error = np.mean((predicted - actual) ** 2)
        
        # 归一化到 [0, 1]
        surprise = min(1.0, error)
        
        # 更新统计
        n = self.stats['total_measurements']
        self.stats['avg_surprise'] = (
            (n - 1) * self.stats['avg_surprise'] + surprise
        ) / n
        
        if surprise > self.threshold:
            self.stats['surprising_events'] += 1
        
        return surprise
    
    def is_surprising(self, surprise: float) -> bool:
        """判断是否令人惊讶"""
        return surprise > self.threshold


class LongTermMemory:
    """
    长期记忆模块
    
    存储重要信息，支持动态更新
    """
    
    def __init__(
        self,
        memory_size: int,
        hidden_size: int
    ):
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        
        # 记忆矩阵
        self.memory = np.zeros((memory_size, hidden_size), dtype=np.float32)
        
        # 记忆状态
        self.memory_age = np.zeros(memory_size, dtype=np.float32)
        self.memory_importance = np.zeros(memory_size, dtype=np.float32)
        
        # 更新动量
        self.momentum = np.zeros_like(self.memory)
    
    def store(
        self,
        hidden_state: np.ndarray,
        importance: float
    ) -> int:
        """
        存储到记忆
        
        Args:
            hidden_state: 隐藏状态 [hidden_size]
            importance: 重要性分数
            
        Returns:
            存储位置
        """
        # 找到最不重要的位置
        min_idx = np.argmin(self.memory_importance)
        
        # 更新记忆
        self.memory[min_idx] = hidden_state
        self.memory_age[min_idx] = 0
        self.memory_importance[min_idx] = importance
        
        return min_idx
    
    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 10
    ) -> np.ndarray:
        """
        检索记忆
        
        Args:
            query: 查询向量
            top_k: 检索数量
            
        Returns:
            检索结果
        """
        # 计算相似度
        similarities = np.dot(self.memory, query)
        
        # 考虑重要性
        scores = similarities * (1 + self.memory_importance)
        
        # 获取top_k
        top_indices = np.argsort(scores)[-top_k:]
        
        # 加权求和
        weights = scores[top_indices]
        weights = weights / (np.sum(np.abs(weights)) + 1e-8)
        
        result = np.sum(
            self.memory[top_indices] * weights[:, np.newaxis],
            axis=0
        )
        
        return result
    
    def update(
        self,
        position: int,
        gradient: np.ndarray,
        learning_rate: float,
        momentum: float
    ) -> None:
        """
        更新记忆（梯度下降）
        
        Args:
            position: 位置
            gradient: 梯度
            learning_rate: 学习率
            momentum: 动量系数
        """
        # 动量更新
        self.momentum[position] = (
            momentum * self.momentum[position] +
            learning_rate * gradient
        )
        
        self.memory[position] += self.momentum[position]
        self.memory_age[position] += 1


class TTTLayer:
    """
    TTT层 - 测试时训练层
    
    核心思想：
    1. 通过惊喜指标识别重要信息
    2. 在推理时动态更新长期记忆
    3. 实现持续学习能力
    """
    
    def __init__(self, config: TTTConfig):
        self.config = config
        
        # 组件
        self.surprise_metric = SurpriseMetric(
            threshold=config.surprise_threshold
        )
        self.long_term_memory = LongTermMemory(
            memory_size=config.memory_size,
            hidden_size=config.hidden_size
        )
        
        # 预测器权重
        scale = 1.0 / math.sqrt(config.hidden_size)
        self.W_predict = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        self.W_update = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        
        # 统计
        self.stats = {
            'total_forward': 0,
            'total_updates': 0,
            'avg_surprise': 0.0,
            'memory_utilization': 0.0
        }
    
    def forward(
        self,
        hidden_states: np.ndarray,
        training: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            training: 是否训练模式
            
        Returns:
            (输出, 统计信息)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 预测
        predictions = np.dot(hidden_states, self.W_predict)
        
        # 计算惊喜度
        surprises = []
        for i in range(seq_len):
            surprise = self.surprise_metric.measure(
                predictions[0, i],  # 简化：只处理第一个batch
                hidden_states[0, i]
            )
            surprises.append(surprise)
        
        avg_surprise = np.mean(surprises)
        
        # 检索长期记忆
        memory_output = np.zeros_like(hidden_states)
        for i in range(seq_len):
            memory_output[0, i] = self.long_term_memory.retrieve(hidden_states[0, i])
        
        # 合并
        output = hidden_states + 0.1 * memory_output
        
        # 测试时训练
        if not training:
            for i, surprise in enumerate(surprises):
                if self.surprise_metric.is_surprising(surprise):
                    # 存储到长期记忆
                    self.long_term_memory.store(
                        hidden_states[0, i],
                        importance=surprise
                    )
                    
                    # 更新预测器
                    self._update_predictor(
                        hidden_states[0, i],
                        predictions[0, i]
                    )
                    
                    self.stats['total_updates'] += 1
        
        # 更新统计
        self.stats['total_forward'] += 1
        n = self.stats['total_forward']
        self.stats['avg_surprise'] = (
            (n - 1) * self.stats['avg_surprise'] + avg_surprise
        ) / n
        
        utilized = np.sum(self.long_term_memory.memory_importance > 0)
        self.stats['memory_utilization'] = utilized / self.config.memory_size
        
        return output, {
            'surprises': surprises,
            'avg_surprise': avg_surprise,
            'num_updates': self.stats['total_updates']
        }
    
    def _update_predictor(
        self,
        target: np.ndarray,
        prediction: np.ndarray
    ) -> None:
        """
        更新预测器
        
        Args:
            target: 目标值
            prediction: 预测值
        """
        # 计算梯度
        error = target - prediction
        gradient = np.outer(error, target)
        
        # 梯度下降
        self.W_predict += self.config.learning_rate * gradient
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            'memory_size': self.config.memory_size,
            'utilized': np.sum(self.long_term_memory.memory_importance > 0),
            'avg_importance': np.mean(self.long_term_memory.memory_importance[
                self.long_term_memory.memory_importance > 0
            ]) if np.any(self.long_term_memory.memory_importance > 0) else 0.0,
            'avg_age': np.mean(self.long_term_memory.memory_age[
                self.long_term_memory.memory_age > 0
            ]) if np.any(self.long_term_memory.memory_age > 0) else 0.0
        }
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'surprise_stats': self.surprise_metric.stats,
            'memory_stats': self.get_memory_stats()
        }
    
    def reset_memory(self) -> None:
        """重置记忆"""
        self.long_term_memory = LongTermMemory(
            memory_size=self.config.memory_size,
            hidden_size=self.config.hidden_size
        )


class TTTModel:
    """
    TTT模型（包含多层TTT）
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        hidden_size: int = 768,
        memory_size: int = 512
    ):
        self.num_layers = num_layers
        
        # 创建多层TTT
        self.layers = [
            TTTLayer(TTTConfig(
                hidden_size=hidden_size,
                memory_size=memory_size
            ))
            for _ in range(num_layers)
        ]
    
    def forward(
        self,
        hidden_states: np.ndarray,
        training: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            training: 是否训练模式
            
        Returns:
            (输出, 统计信息)
        """
        all_stats = []
        
        for layer in self.layers:
            hidden_states, stats = layer.forward(hidden_states, training)
            all_stats.append(stats)
        
        # 汇总统计
        total_updates = sum(s['num_updates'] for s in all_stats)
        avg_surprise = np.mean([s['avg_surprise'] for s in all_stats])
        
        return hidden_states, {
            'total_updates': total_updates,
            'avg_surprise': avg_surprise,
            'layer_stats': all_stats
        }


# 便捷函数
def create_ttt_layer(
    hidden_size: int = 768,
    memory_size: int = 512
) -> TTTLayer:
    """创建TTT层"""
    config = TTTConfig(
        hidden_size=hidden_size,
        memory_size=memory_size
    )
    return TTTLayer(config)


if __name__ == "__main__":
    # 测试
    layer = create_ttt_layer(hidden_size=256, memory_size=128)
    
    print("=" * 60)
    print("TTT层测试")
    print("=" * 60)
    
    # 模拟输入
    batch_size, seq_len = 2, 32
    hidden_states = np.random.randn(batch_size, seq_len, 256).astype(np.float32)
    
    # 多次前向传播
    for i in range(5):
        output, stats = layer.forward(hidden_states)
        print(f"\n步骤 {i+1}:")
        print(f"  平均惊喜度: {stats['avg_surprise']:.3f}")
        print(f"  更新次数: {stats['num_updates']}")
    
    # 统计
    print(f"\n最终统计: {layer.get_stats()}")
