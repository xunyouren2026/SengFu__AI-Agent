"""
质量评分机制 - Quality Scorer

评估每条历史消息的价值，优先保留高质量内容

作者: UFO Framework Team
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class QualityMetrics:
    """质量指标"""
    clarity: float = 0.0      # 清晰度
    relevance: float = 0.0    # 相关性
    informativeness: float = 0.0  # 信息量
    coherence: float = 0.0    # 连贯性
    novelty: float = 0.0      # 新颖性
    overall: float = 0.0      # 综合评分


class TextQualityAnalyzer:
    """文本质量分析器"""
    
    def __init__(self):
        # 停用词
        self.stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            '的', '了', '是', '在', '有', '和', '与', '或', '这', '那'
        }
        
        # 信息词权重
        self.information_words = {
            'important', 'key', 'critical', 'essential', 'significant',
            '重要', '关键', '核心', '必要', '显著'
        }
    
    def analyze(self, text: str) -> QualityMetrics:
        """
        分析文本质量
        
        Args:
            text: 待分析文本
            
        Returns:
            QualityMetrics 质量指标
        """
        metrics = QualityMetrics()
        
        if not text or not text.strip():
            return metrics
        
        # 1. 清晰度：句子完整度、标点使用
        metrics.clarity = self._calculate_clarity(text)
        
        # 2. 相关性：关键词密度
        metrics.relevance = self._calculate_relevance(text)
        
        # 3. 信息量：词汇丰富度
        metrics.informativeness = self._calculate_informativeness(text)
        
        # 4. 连贯性：句子连接
        metrics.coherence = self._calculate_coherence(text)
        
        # 5. 新颖性：独特词汇比例
        metrics.novelty = self._calculate_novelty(text)
        
        # 综合评分（加权平均）
        metrics.overall = (
            0.25 * metrics.clarity +
            0.25 * metrics.relevance +
            0.20 * metrics.informativeness +
            0.15 * metrics.coherence +
            0.15 * metrics.novelty
        )
        
        return metrics
    
    def _calculate_clarity(self, text: str) -> float:
        """计算清晰度"""
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # 平均句子长度（适中为佳）
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        length_score = 1.0 - abs(avg_len - 20) / 50  # 20字左右最佳
        length_score = max(0, min(1, length_score))
        
        # 标点使用率
        punct_count = len(re.findall(r'[.!?，。！？、；：]', text))
        punct_ratio = punct_count / max(1, len(text))
        punct_score = min(1, punct_ratio * 10)  # 10%标点最佳
        
        return 0.6 * length_score + 0.4 * punct_score
    
    def _calculate_relevance(self, text: str) -> float:
        """计算相关性"""
        words = text.lower().split()
        
        if not words:
            return 0.0
        
        # 信息词比例
        info_count = sum(1 for w in words if w in self.information_words)
        info_ratio = info_count / len(words)
        
        # 非停用词比例
        content_words = [w for w in words if w not in self.stopwords]
        content_ratio = len(content_words) / len(words)
        
        return 0.4 * info_ratio * 5 + 0.6 * content_ratio
    
    def _calculate_informativeness(self, text: str) -> float:
        """计算信息量"""
        words = text.split()
        
        if not words:
            return 0.0
        
        # 词汇多样性（唯一词比例）
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # 数字和专有名词比例
        numbers = len(re.findall(r'\d+', text))
        proper_nouns = len(re.findall(r'[A-Z][a-z]+', text))
        specific_ratio = (numbers + proper_nouns) / len(words)
        
        return 0.7 * diversity + 0.3 * min(1, specific_ratio * 5)
    
    def _calculate_coherence(self, text: str) -> float:
        """计算连贯性"""
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0
        
        # 连接词使用
        connectors = {
            'however', 'therefore', 'moreover', 'furthermore',
            '但是', '因此', '而且', '此外', '所以', '然后'
        }
        
        connector_count = 0
        for sent in sentences:
            words = sent.lower().split()
            connector_count += sum(1 for w in words if w in connectors)
        
        connector_ratio = connector_count / len(sentences)
        
        # 句子长度一致性
        lengths = [len(s) for s in sentences]
        if len(lengths) > 1:
            variance = np.var(lengths)
            consistency = 1.0 / (1.0 + variance / 100)
        else:
            consistency = 1.0
        
        return 0.5 * min(1, connector_ratio) + 0.5 * consistency
    
    def _calculate_novelty(self, text: str) -> float:
        """计算新颖性"""
        words = text.split()
        
        if not words:
            return 0.0
        
        # 独特词汇（出现一次的词）
        word_counts = {}
        for w in words:
            w = w.lower()
            word_counts[w] = word_counts.get(w, 0) + 1
        
        unique_once = sum(1 for c in word_counts.values() if c == 1)
        novelty_ratio = unique_once / len(words)
        
        return min(1, novelty_ratio * 2)


class QualityScorer:
    """
    质量评分器
    
    评估消息质量，用于筛选优质上下文
    """
    
    def __init__(
        self,
        min_quality_threshold: float = 0.3,
        boost_recent: bool = True,
        recent_weight: float = 0.1
    ):
        self.min_quality_threshold = min_quality_threshold
        self.boost_recent = boost_recent
        self.recent_weight = recent_weight
        
        self.analyzer = TextQualityAnalyzer()
        
        # 历史评分缓存
        self._score_cache: Dict[str, QualityMetrics] = {}
        
        # 统计
        self.stats = {
            'total_scored': 0,
            'high_quality': 0,
            'low_quality': 0,
            'avg_quality': 0.0
        }
    
    def score(
        self,
        text: str,
        context: Optional[List[str]] = None,
        position: int = 0
    ) -> QualityMetrics:
        """
        评分
        
        Args:
            text: 待评分文本
            context: 上下文（用于计算相关性）
            position: 位置（用于时间衰减）
            
        Returns:
            QualityMetrics
        """
        self.stats['total_scored'] += 1
        
        # 检查缓存
        cache_key = text[:100]  # 使用前100字符作为key
        if cache_key in self._score_cache:
            metrics = self._score_cache[cache_key]
        else:
            metrics = self.analyzer.analyze(text)
            self._score_cache[cache_key] = metrics
        
        # 上下文相关性调整
        if context:
            relevance_boost = self._calculate_context_relevance(text, context)
            metrics.relevance = min(1, metrics.relevance + relevance_boost * 0.3)
        
        # 时间衰减
        if self.boost_recent and position > 0:
            decay = math.exp(-self.recent_weight * position)
            metrics.overall *= decay
        
        # 更新统计
        if metrics.overall >= 0.7:
            self.stats['high_quality'] += 1
        elif metrics.overall < self.min_quality_threshold:
            self.stats['low_quality'] += 1
        
        n = self.stats['total_scored']
        self.stats['avg_quality'] = (
            (n - 1) * self.stats['avg_quality'] + metrics.overall
        ) / n
        
        return metrics
    
    def _calculate_context_relevance(
        self,
        text: str,
        context: List[str]
    ) -> float:
        """计算与上下文的相关性"""
        if not context:
            return 0.0
        
        text_words = set(text.lower().split())
        
        # 计算与每条上下文的重叠
        overlaps = []
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            overlap = len(text_words & ctx_words) / max(1, len(text_words))
            overlaps.append(overlap)
        
        return max(overlaps) if overlaps else 0.0
    
    def score_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Tuple[Dict, QualityMetrics]]:
        """
        批量评分消息
        
        Args:
            messages: 消息列表
            
        Returns:
            (消息, 评分) 列表
        """
        results = []
        context = []
        
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            metrics = self.score(content, context, i)
            results.append((msg, metrics))
            context.append(content)
        
        return results
    
    def filter_by_quality(
        self,
        messages: List[Dict[str, str]],
        min_quality: Optional[float] = None
    ) -> List[Dict[str, str]]:
        """
        按质量过滤消息
        
        Args:
            messages: 消息列表
            min_quality: 最小质量阈值
            
        Returns:
            过滤后的消息列表
        """
        threshold = min_quality or self.min_quality_threshold
        
        scored = self.score_messages(messages)
        filtered = [
            msg for msg, metrics in scored
            if metrics.overall >= threshold
        ]
        
        return filtered
    
    def get_top_k(
        self,
        messages: List[Dict[str, str]],
        k: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取质量最高的k条消息
        
        Args:
            messages: 消息列表
            k: 数量
            
        Returns:
            top k消息
        """
        scored = self.score_messages(messages)
        scored.sort(key=lambda x: x[1].overall, reverse=True)
        
        return [msg for msg, _ in scored[:k]]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_size': len(self._score_cache),
            'quality_rate': (
                self.stats['high_quality'] / 
                max(1, self.stats['total_scored'])
            )
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._score_cache.clear()


# 便捷函数
def score_text(text: str) -> float:
    """快速评分文本"""
    scorer = QualityScorer()
    metrics = scorer.score(text)
    return metrics.overall


if __name__ == "__main__":
    # 测试
    scorer = QualityScorer()
    
    test_texts = [
        "这是一个重要的消息，包含了关键信息。",
        "好的",
        "根据分析结果，我们可以得出以下结论：第一，数据质量良好；第二，模型性能优秀。",
        "嗯嗯",
        "核心问题在于系统架构设计不合理，需要重构。",
    ]
    
    print("=" * 60)
    print("质量评分测试")
    print("=" * 60)
    
    for text in test_texts:
        metrics = scorer.score(text)
        print(f"\n文本: {text[:30]}...")
        print(f"  清晰度: {metrics.clarity:.2f}")
        print(f"  相关性: {metrics.relevance:.2f}")
        print(f"  信息量: {metrics.informativeness:.2f}")
        print(f"  连贯性: {metrics.coherence:.2f}")
        print(f"  新颖性: {metrics.novelty:.2f}")
        print(f"  综合评分: {metrics.overall:.2f}")
    
    print(f"\n统计: {scorer.get_stats()}")
