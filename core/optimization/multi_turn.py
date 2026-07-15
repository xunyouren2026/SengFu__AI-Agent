"""
多轮对话优化器 - Multi-Turn Optimizer

识别对话转折点，智能重置上下文

作者: UFO Framework Team
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class TurnAnalysis:
    """轮次分析结果"""
    turn_index: int
    is_topic_shift: bool
    shift_score: float
    topic_keywords: List[str]
    coherence_with_previous: float


class TopicShiftDetector:
    """话题转换检测器"""
    
    def __init__(
        self,
        shift_threshold: float = 0.4,
        window_size: int = 3
    ):
        self.shift_threshold = shift_threshold
        self.window_size = window_size
        
        # 转折指示词
        self.shift_indicators = {
            '但是', '不过', '然而', '另外', '换个话题', '对了',
            'by the way', 'however', 'anyway', 'speaking of',
            '其实', '话说', '说起来', '对了'
        }
    
    def detect(
        self,
        current_msg: str,
        previous_msgs: List[str]
    ) -> Tuple[bool, float]:
        """
        检测话题转换
        
        Args:
            current_msg: 当前消息
            previous_msgs: 之前消息列表
            
        Returns:
            (是否转换, 转换分数)
        """
        if not previous_msgs:
            return False, 0.0
        
        # 1. 检查转折指示词
        has_indicator = any(
            ind in current_msg.lower()
            for ind in self.shift_indicators
        )
        indicator_score = 0.5 if has_indicator else 0.0
        
        # 2. 计算词汇重叠
        current_words = set(current_msg.lower().split())
        
        recent_msgs = previous_msgs[-self.window_size:]
        recent_words = set()
        for msg in recent_msgs:
            recent_words.update(msg.lower().split())
        
        if current_words and recent_words:
            overlap = len(current_words & recent_words)
            overlap_ratio = overlap / len(current_words)
            overlap_score = 1 - overlap_ratio  # 重叠少=可能转换
        else:
            overlap_score = 0.5
        
        # 3. 综合分数
        shift_score = 0.4 * indicator_score + 0.6 * overlap_score
        
        is_shift = shift_score > self.shift_threshold
        
        return is_shift, shift_score


class CoherenceAnalyzer:
    """连贯性分析器"""
    
    def __init__(self):
        # 连接词
        self.connectors = {
            '所以', '因此', '那么', '然后', '接着', '于是',
            'therefore', 'so', 'then', 'thus', 'hence',
            '因为', '由于', '既然', 'because', 'since'
        }
    
    def analyze(
        self,
        current_msg: str,
        previous_msg: str
    ) -> float:
        """
        分析连贯性
        
        Args:
            current_msg: 当前消息
            previous_msg: 前一条消息
            
        Returns:
            连贯性分数 [0, 1]
        """
        if not previous_msg:
            return 1.0
        
        # 1. 连接词使用
        has_connector = any(
            conn in current_msg.lower()
            for conn in self.connectors
        )
        connector_score = 0.3 if has_connector else 0.0
        
        # 2. 代词引用（简化检测）
        pronouns = {'它', '他', '她', '这个', '那个', '这', '那', 'it', 'this', 'that'}
        has_reference = any(p in current_msg.lower() for p in pronouns)
        reference_score = 0.2 if has_reference else 0.0
        
        # 3. 词汇延续
        prev_words = set(previous_msg.lower().split())
        curr_words = set(current_msg.lower().split())
        
        if curr_words:
            continuation = len(prev_words & curr_words) / len(curr_words)
        else:
            continuation = 0.0
        
        # 综合分数
        coherence = connector_score + reference_score + 0.5 * continuation
        
        return min(1.0, coherence)


class MultiTurnOptimizer:
    """
    多轮对话优化器
    
    功能:
    1. 检测话题转换
    2. 分析对话连贯性
    3. 智能重置上下文
    """
    
    def __init__(
        self,
        shift_threshold: float = 0.4,
        coherence_threshold: float = 0.3,
        auto_reset: bool = True
    ):
        self.shift_threshold = shift_threshold
        self.coherence_threshold = coherence_threshold
        self.auto_reset = auto_reset
        
        # 组件
        self.shift_detector = TopicShiftDetector(shift_threshold)
        self.coherence_analyzer = CoherenceAnalyzer()
        
        # 历史
        self.turn_history: List[TurnAnalysis] = []
        self.message_history: List[str] = []
        
        # 统计
        self.stats = {
            'total_turns': 0,
            'topic_shifts': 0,
            'resets': 0,
            'avg_coherence': 0.0
        }
    
    def analyze_turn(
        self,
        user_msg: str,
        assistant_msg: Optional[str] = None
    ) -> TurnAnalysis:
        """
        分析对话轮次
        
        Args:
            user_msg: 用户消息
            assistant_msg: 助手消息（可选）
            
        Returns:
            TurnAnalysis
        """
        self.stats['total_turns'] += 1
        turn_index = len(self.turn_history)
        
        # 检测话题转换
        is_shift, shift_score = self.shift_detector.detect(
            user_msg,
            self.message_history
        )
        
        if is_shift:
            self.stats['topic_shifts'] += 1
        
        # 分析连贯性
        if self.message_history:
            coherence = self.coherence_analyzer.analyze(
                user_msg,
                self.message_history[-1]
            )
        else:
            coherence = 1.0
        
        # 更新平均连贯性
        n = self.stats['total_turns']
        self.stats['avg_coherence'] = (
            (n - 1) * self.stats['avg_coherence'] + coherence
        ) / n
        
        # 提取关键词
        topic_keywords = self._extract_keywords(user_msg)
        
        # 创建分析结果
        analysis = TurnAnalysis(
            turn_index=turn_index,
            is_topic_shift=is_shift,
            shift_score=shift_score,
            topic_keywords=topic_keywords,
            coherence_with_previous=coherence
        )
        
        # 记录历史
        self.turn_history.append(analysis)
        self.message_history.append(user_msg)
        
        return analysis
    
    def should_reset_context(self) -> bool:
        """
        判断是否应该重置上下文
        
        Returns:
            是否重置
        """
        if not self.turn_history:
            return False
        
        recent = self.turn_history[-1]
        
        # 话题转换且连贯性低
        if recent.is_topic_shift and recent.coherence_with_previous < self.coherence_threshold:
            if self.auto_reset:
                self.stats['resets'] += 1
            return True
        
        # 连续低连贯性
        if len(self.turn_history) >= 3:
            recent_coherence = [
                t.coherence_with_previous
                for t in self.turn_history[-3:]
            ]
            if all(c < self.coherence_threshold for c in recent_coherence):
                if self.auto_reset:
                    self.stats['resets'] += 1
                return True
        
        return False
    
    def get_context_boundary(self) -> int:
        """
        获取上下文边界（应该保留的最近轮次数）
        
        Returns:
            轮次数
        """
        # 从后向前找最后一个话题转换点
        for i in range(len(self.turn_history) - 1, -1, -1):
            if self.turn_history[i].is_topic_shift:
                return len(self.turn_history) - i
        
        return len(self.turn_history)
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        # 简单提取：非停用词的高频词
        stopwords = {'的', '了', '是', '在', '有', '和', 'the', 'a', 'an', 'is', 'are'}
        
        words = re.findall(r'\w+', text.lower())
        words = [w for w in words if w not in stopwords and len(w) > 1]
        
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [w for w, _ in sorted_words[:top_n]]
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'shift_rate': (
                self.stats['topic_shifts'] / 
                max(1, self.stats['total_turns'])
            ),
            'reset_rate': (
                self.stats['resets'] / 
                max(1, self.stats['total_turns'])
            )
        }
    
    def reset(self) -> None:
        """重置状态"""
        self.turn_history.clear()
        self.message_history.clear()


# 便捷函数
def analyze_conversation(messages: List[Dict[str, str]]) -> Dict:
    """分析对话"""
    optimizer = MultiTurnOptimizer()
    
    for msg in messages:
        if msg['role'] == 'user':
            optimizer.analyze_turn(msg['content'])
    
    return optimizer.get_stats()


if __name__ == "__main__":
    # 测试
    optimizer = MultiTurnOptimizer()
    
    # 模拟对话
    test_dialog = [
        ("你好，我想了解Python编程", False),
        ("Python有什么优点？", False),
        ("对了，你知道机器学习吗？", True),  # 话题转换
        ("机器学习怎么入门？", False),
        ("换个话题，推荐一些好书", True),  # 话题转换
    ]
    
    print("=" * 60)
    print("多轮对话优化测试")
    print("=" * 60)
    
    for msg, expected_shift in test_dialog:
        analysis = optimizer.analyze_turn(msg)
        print(f"\n消息: {msg}")
        print(f"  话题转换: {analysis.is_topic_shift} (预期: {expected_shift})")
        print(f"  转换分数: {analysis.shift_score:.2f}")
        print(f"  连贯性: {analysis.coherence_with_previous:.2f}")
        print(f"  关键词: {analysis.topic_keywords}")
    
    print(f"\n统计: {optimizer.get_stats()}")
    print(f"上下文边界: {optimizer.get_context_boundary()}")
