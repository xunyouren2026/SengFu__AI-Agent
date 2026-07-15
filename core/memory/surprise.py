"""
Surprise-Based Memory - 基于惊喜度的智能记忆选择系统

核心思想:
- 只存储"令人惊讶"的信息（预测误差大）
- 自动遗忘不再重要的记忆
- 基于信息价值动态管理记忆库

作者: UFO Framework Team
"""

import math
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class MemoryEntry:
    """记忆条目"""
    content: str
    embedding: Optional[np.ndarray] = None
    surprise_score: float = 0.0
    importance_score: float = 0.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.99
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_importance(self) -> float:
        """计算当前重要性（考虑时间衰减）"""
        time_decay = self.decay_rate ** (time.time() - self.created_at)
        access_boost = 1.0 + 0.1 * math.log(1 + self.access_count)
        return self.importance_score * time_decay * access_boost
    
    def access(self) -> None:
        """访问记忆"""
        self.access_count += 1
        self.last_accessed = time.time()


class SurpriseCalculator:
    """惊喜度计算器"""
    
    def __init__(
        self,
        prediction_window: int = 10,
        surprise_threshold: float = 0.3,
        use_entropy: bool = True
    ):
        self.prediction_window = prediction_window
        self.surprise_threshold = surprise_threshold
        self.use_entropy = use_entropy
        
        # 预测历史
        self.prediction_history: deque = deque(maxlen=1000)
        self.actual_history: deque = deque(maxlen=1000)
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'surprising_events': 0,
            'avg_surprise': 0.0
        }
    
    def calculate_surprise(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """
        计算惊喜度
        
        惊喜度 = 预测误差 + 熵差异
        
        Args:
            predicted: 预测值
            actual: 实际值
            
        Returns:
            惊喜度分数 [0, 1]
        """
        self.stats['total_calculations'] += 1
        
        # 方法1: 欧氏距离
        if predicted is not None and actual is not None:
            mse = np.mean((predicted - actual) ** 2)
            distance_score = min(1.0, mse)
        else:
            distance_score = 0.5
        
        # 方法2: 熵差异
        if self.use_entropy:
            entropy_diff = self._calculate_entropy_difference(predicted, actual)
            surprise = 0.6 * distance_score + 0.4 * entropy_diff
        else:
            surprise = distance_score
        
        # 更新统计
        if surprise > self.surprise_threshold:
            self.stats['surprising_events'] += 1
        
        # 滑动平均
        n = self.stats['total_calculations']
        self.stats['avg_surprise'] = (
            (n - 1) * self.stats['avg_surprise'] + surprise
        ) / n
        
        return surprise
    
    def _calculate_entropy_difference(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray
    ) -> float:
        """计算两个分布的熵差异"""
        try:
            # 确保是概率分布
            p1 = np.clip(dist1, 1e-10, 1.0)
            p1 = p1 / np.sum(p1)
            p2 = np.clip(dist2, 1e-10, 1.0)
            p2 = p2 / np.sum(p2)
            
            # 计算KL散度
            kl_div = np.sum(p1 * np.log(p1 / p2))
            return min(1.0, abs(kl_div))
        except:
            return 0.0
    
    def is_surprising(self, surprise_score: float) -> bool:
        """判断是否足够令人惊讶"""
        return surprise_score > self.surprise_threshold
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'surprise_rate': (
                self.stats['surprising_events'] / 
                max(1, self.stats['total_calculations'])
            )
        }


class ForgettingMechanism:
    """遗忘机制"""
    
    def __init__(
        self,
        base_decay_rate: float = 0.99,
        min_importance_threshold: float = 0.01,
        max_memory_size: int = 10000
    ):
        self.base_decay_rate = base_decay_rate
        self.min_importance_threshold = min_importance_threshold
        self.max_memory_size = max_memory_size
    
    def apply_forgetting(
        self,
        memories: List[MemoryEntry]
    ) -> List[MemoryEntry]:
        """
        应用遗忘机制
        
        Args:
            memories: 当前记忆列表
            
        Returns:
            过滤后的记忆列表
        """
        # 计算当前重要性
        for mem in memories:
            current_importance = mem.get_current_importance()
            mem.importance_score = current_importance
        
        # 过滤低重要性记忆
        filtered = [
            mem for mem in memories
            if mem.get_current_importance() >= self.min_importance_threshold
        ]
        
        # 如果超过最大容量，按重要性排序后截断
        if len(filtered) > self.max_memory_size:
            filtered.sort(
                key=lambda x: x.get_current_importance(),
                reverse=True
            )
            filtered = filtered[:self.max_memory_size]
        
        return filtered
    
    def calculate_forgetting_curve(
        self,
        time_elapsed: float,
        initial_importance: float = 1.0
    ) -> float:
        """
        计算遗忘曲线（艾宾浩斯遗忘曲线）
        
        R = e^(-t/S)
        R: 记忆保持率
        t: 时间
        S: 记忆强度
        """
        memory_strength = 1.0 / (1.0 - self.base_decay_rate)
        retention = math.exp(-time_elapsed / memory_strength)
        return initial_importance * retention


class SurpriseBasedMemory:
    """
    基于惊喜度的智能记忆系统
    
    核心功能:
    1. 只存储令人惊讶的信息
    2. 自动遗忘不再重要的记忆
    3. 基于重要性检索
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        surprise_threshold: float = 0.3,
        decay_rate: float = 0.99,
        enable_forgetting: bool = True
    ):
        self.max_size = max_size
        self.surprise_threshold = surprise_threshold
        
        # 核心组件
        self.surprise_calculator = SurpriseCalculator(
            surprise_threshold=surprise_threshold
        )
        self.forgetting_mechanism = ForgettingMechanism(
            base_decay_rate=decay_rate
        )
        
        # 记忆存储
        self.memories: List[MemoryEntry] = []
        self.enable_forgetting = enable_forgetting
        
        # 索引（用于快速检索）
        self._content_index: Dict[str, int] = {}
        
        # 统计信息
        self.stats = {
            'total_stored': 0,
            'total_rejected': 0,
            'total_forgotten': 0,
            'total_retrieved': 0
        }
    
    def store(
        self,
        content: str,
        predicted: Optional[np.ndarray] = None,
        actual: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        存储记忆（基于惊喜度决定是否存储）
        
        Args:
            content: 记忆内容
            predicted: 预测值（用于计算惊喜度）
            actual: 实际值
            metadata: 额外元数据
            
        Returns:
            是否成功存储
        """
        # 计算惊喜度
        if predicted is not None and actual is not None:
            surprise_score = self.surprise_calculator.calculate_surprise(
                predicted, actual
            )
        else:
            # 无预测时，使用默认惊喜度
            surprise_score = 0.5
        
        # 决定是否存储
        if not self.surprise_calculator.is_surprising(surprise_score):
            self.stats['total_rejected'] += 1
            return False
        
        # 创建记忆条目
        entry = MemoryEntry(
            content=content,
            embedding=actual,
            surprise_score=surprise_score,
            importance_score=surprise_score,
            metadata=metadata or {}
        )
        
        # 添加到存储
        self.memories.append(entry)
        self._content_index[content] = len(self.memories) - 1
        self.stats['total_stored'] += 1
        
        # 应用遗忘机制
        if self.enable_forgetting:
            self._apply_forgetting()
        
        return True
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        min_importance: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        检索记忆
        
        Args:
            query: 查询内容
            top_k: 返回数量
            min_importance: 最小重要性阈值
            
        Returns:
            (记忆条目, 相关性分数) 列表
        """
        # 过滤低重要性记忆
        candidates = [
            mem for mem in self.memories
            if mem.get_current_importance() >= min_importance
        ]
        
        # 计算相关性（简化：基于文本相似度）
        scored = []
        for mem in candidates:
            score = self._calculate_relevance(query, mem.content)
            scored.append((mem, score))
        
        # 排序并返回top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]
        
        # 更新访问计数
        for mem, _ in results:
            mem.access()
        
        self.stats['total_retrieved'] += len(results)
        
        return results
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """计算相关性分数（简化版）"""
        # 词重叠率
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def _apply_forgetting(self) -> None:
        """应用遗忘机制"""
        original_count = len(self.memories)
        self.memories = self.forgetting_mechanism.apply_forgetting(self.memories)
        forgotten = original_count - len(self.memories)
        self.stats['total_forgotten'] += forgotten
        
        # 重建索引
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """重建内容索引"""
        self._content_index = {
            mem.content: i
            for i, mem in enumerate(self.memories)
        }
    
    def get_memory_stats(self) -> Dict:
        """获取记忆系统统计"""
        return {
            **self.stats,
            'current_size': len(self.memories),
            'max_size': self.max_size,
            'surprise_stats': self.surprise_calculator.get_stats(),
            'avg_importance': (
                sum(m.get_current_importance() for m in self.memories) /
                max(1, len(self.memories))
            )
        }
    
    def clear(self) -> None:
        """清空记忆"""
        self.memories.clear()
        self._content_index.clear()
    
    def export_memories(self) -> List[Dict]:
        """导出记忆为字典列表"""
        return [
            {
                'content': mem.content,
                'surprise_score': mem.surprise_score,
                'importance_score': mem.get_current_importance(),
                'access_count': mem.access_count,
                'created_at': mem.created_at,
                'metadata': mem.metadata
            }
            for mem in self.memories
        ]


# 便捷函数
def create_surprise_memory(
    max_size: int = 10000,
    surprise_threshold: float = 0.3
) -> SurpriseBasedMemory:
    """创建惊喜度记忆系统"""
    return SurpriseBasedMemory(
        max_size=max_size,
        surprise_threshold=surprise_threshold
    )


if __name__ == "__main__":
    # 测试
    memory = SurpriseBasedMemory(surprise_threshold=0.2)
    
    # 模拟存储
    for i in range(10):
        predicted = np.random.rand(10)
        actual = predicted + np.random.randn(10) * 0.5  # 添加噪声
        content = f"记忆内容 {i}"
        stored = memory.store(content, predicted, actual)
        print(f"存储 '{content}': {'成功' if stored else '被拒绝'}")
    
    # 检索
    results = memory.retrieve("记忆", top_k=5)
    print(f"\n检索结果: {len(results)} 条")
    
    # 统计
    print(f"\n统计: {memory.get_memory_stats()}")
