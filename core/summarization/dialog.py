"""
对话摘要器 - Dialog Summarizer

自动总结长对话，减少重复Token

作者: UFO Framework Team
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import time


@dataclass
class SummaryResult:
    """摘要结果"""
    summary: str
    key_points: List[str]
    entities: List[str]
    compression_ratio: float
    original_length: int
    summary_length: int


class KeyPointExtractor:
    """关键点提取器"""
    
    def __init__(self, min_importance: float = 0.3):
        self.min_importance = min_importance
        
        # 关键词指示词
        self.importance_indicators = {
            'important', 'key', 'critical', 'essential', 'main',
            '重要', '关键', '核心', '主要', '首先', '其次', '最后'
        }
        
        # 问题词
        self.question_words = {
            'what', 'why', 'how', 'when', 'where', 'who',
            '什么', '为什么', '怎么', '何时', '哪里', '谁'
        }
    
    def extract(self, text: str) -> List[str]:
        """
        提取关键点
        
        Args:
            text: 输入文本
            
        Returns:
            关键点列表
        """
        sentences = self._split_sentences(text)
        key_points = []
        
        for sent in sentences:
            score = self._calculate_importance(sent)
            if score >= self.min_importance:
                key_points.append(sent.strip())
        
        return key_points
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = re.split(r'[.!?。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_importance(self, sentence: str) -> float:
        """计算句子重要性"""
        words = set(sentence.lower().split())
        
        # 关键词指示词
        indicator_count = len(words & self.importance_indicators)
        indicator_score = min(1.0, indicator_count * 0.3)
        
        # 问题词
        has_question = len(words & self.question_words) > 0
        question_score = 0.3 if has_question else 0
        
        # 句子长度（适中为佳）
        length = len(sentence)
        if 20 <= length <= 100:
            length_score = 0.3
        else:
            length_score = 0.1
        
        # 数字和专有名词
        has_numbers = bool(re.search(r'\d+', sentence))
        has_proper = bool(re.search(r'[A-Z][a-z]+', sentence))
        specific_score = 0.2 if (has_numbers or has_proper) else 0
        
        return indicator_score + question_score + length_score + specific_score


class EntityRecognizer:
    """实体识别器"""
    
    def __init__(self):
        # 实体模式
        self.patterns = {
            'email': r'[\w.-]+@[\w.-]+\.\w+',
            'url': r'https?://[\w./-]+',
            'phone': r'\d{3,4}[-.]?\d{3,4}[-.]?\d{4}',
            'date': r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?',
            'time': r'\d{1,2}:\d{2}',
            'number': r'\d+(?:\.\d+)?(?:%|万|亿)?',
        }
    
    def recognize(self, text: str) -> List[str]:
        """
        识别实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        for pattern in self.patterns.values():
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 提取大写开头的词（可能是专有名词）
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        entities.extend(proper_nouns)
        
        # 去重
        return list(set(entities))


class DialogSummarizer:
    """
    对话摘要器
    
    功能:
    1. 提取关键点
    2. 识别实体
    3. 生成压缩摘要
    """
    
    def __init__(
        self,
        max_summary_length: int = 200,
        preserve_entities: bool = True,
        preserve_questions: bool = True
    ):
        self.max_summary_length = max_summary_length
        self.preserve_entities = preserve_entities
        self.preserve_questions = preserve_questions
        
        # 组件
        self.key_point_extractor = KeyPointExtractor()
        self.entity_recognizer = EntityRecognizer()
        
        # 统计
        self.stats = {
            'total_summarized': 0,
            'total_chars_saved': 0,
            'avg_compression_ratio': 0.0
        }
    
    def summarize(
        self,
        messages: List[Dict[str, str]],
        style: str = "concise"
    ) -> SummaryResult:
        """
        摘要对话
        
        Args:
            messages: 消息列表
            style: 摘要风格 (concise/detailed/bulleted)
            
        Returns:
            SummaryResult
        """
        self.stats['total_summarized'] += 1
        
        # 合并所有内容
        full_text = self._merge_messages(messages)
        original_length = len(full_text)
        
        # 提取关键点
        key_points = self.key_point_extractor.extract(full_text)
        
        # 识别实体
        entities = self.entity_recognizer.recognize(full_text)
        
        # 生成摘要
        if style == "bulleted":
            summary = self._generate_bulleted_summary(key_points, entities)
        elif style == "detailed":
            summary = self._generate_detailed_summary(key_points, entities)
        else:  # concise
            summary = self._generate_concise_summary(key_points, entities)
        
        # 截断到最大长度
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length] + "..."
        
        summary_length = len(summary)
        compression_ratio = 1 - (summary_length / max(1, original_length))
        
        # 更新统计
        self.stats['total_chars_saved'] += original_length - summary_length
        n = self.stats['total_summarized']
        self.stats['avg_compression_ratio'] = (
            (n - 1) * self.stats['avg_compression_ratio'] + compression_ratio
        ) / n
        
        return SummaryResult(
            summary=summary,
            key_points=key_points,
            entities=entities,
            compression_ratio=compression_ratio,
            original_length=original_length,
            summary_length=summary_length
        )
    
    def _merge_messages(self, messages: List[Dict[str, str]]) -> str:
        """合并消息"""
        parts = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)
    
    def _generate_concise_summary(
        self,
        key_points: List[str],
        entities: List[str]
    ) -> str:
        """生成简洁摘要"""
        if not key_points:
            return ""
        
        # 取最重要的3-5个关键点
        top_points = key_points[:5]
        
        summary = "对话要点: " + "; ".join(top_points)
        
        # 添加关键实体
        if self.preserve_entities and entities:
            summary += f"\n关键信息: {', '.join(entities[:5])}"
        
        return summary
    
    def _generate_detailed_summary(
        self,
        key_points: List[str],
        entities: List[str]
    ) -> str:
        """生成详细摘要"""
        parts = []
        
        if key_points:
            parts.append("主要讨论:")
            for i, point in enumerate(key_points[:10], 1):
                parts.append(f"  {i}. {point}")
        
        if self.preserve_entities and entities:
            parts.append(f"\n涉及内容: {', '.join(entities)}")
        
        return "\n".join(parts)
    
    def _generate_bulleted_summary(
        self,
        key_points: List[str],
        entities: List[str]
    ) -> str:
        """生成要点摘要"""
        lines = []
        
        for point in key_points[:8]:
            lines.append(f"• {point}")
        
        if self.preserve_entities and entities:
            lines.append(f"\n📌 关键信息: {', '.join(entities[:5])}")
        
        return "\n".join(lines)
    
    def summarize_turn(
        self,
        user_msg: str,
        assistant_msg: str
    ) -> str:
        """
        摘要单轮对话
        
        Args:
            user_msg: 用户消息
            assistant_msg: 助手消息
            
        Returns:
            摘要
        """
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
        
        result = self.summarize(messages, style="concise")
        return result.summary
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return self.stats.copy()


class ConversationCompressor:
    """
    对话压缩器
    
    将长对话压缩为摘要形式
    """
    
    def __init__(
        self,
        max_turns_before_summary: int = 10,
        keep_recent_turns: int = 3
    ):
        self.max_turns_before_summary = max_turns_before_summary
        self.keep_recent_turns = keep_recent_turns
        
        self.summarizer = DialogSummarizer()
        
        # 存储摘要历史
        self.summaries: List[str] = []
    
    def compress(
        self,
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        压缩对话
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表
        """
        if len(messages) <= self.max_turns_before_summary:
            return messages
        
        # 分离需要摘要的部分和保留的部分
        to_summarize = messages[:-self.keep_recent_turns]
        to_keep = messages[-self.keep_recent_turns:]
        
        # 生成摘要
        result = self.summarizer.summarize(to_summarize)
        
        # 创建摘要消息
        summary_msg = {
            "role": "system",
            "content": f"[对话摘要] {result.summary}"
        }
        
        self.summaries.append(result.summary)
        
        return [summary_msg] + to_keep
    
    def get_all_summaries(self) -> List[str]:
        """获取所有摘要"""
        return self.summaries.copy()


# 便捷函数
def summarize_dialog(messages: List[Dict[str, str]]) -> str:
    """快速摘要对话"""
    summarizer = DialogSummarizer()
    result = summarizer.summarize(messages)
    return result.summary


if __name__ == "__main__":
    # 测试
    summarizer = DialogSummarizer()
    
    test_messages = [
        {"role": "user", "content": "你好，我想了解Python编程语言。"},
        {"role": "assistant", "content": "Python是一种流行的编程语言，广泛用于数据科学、机器学习、Web开发等领域。"},
        {"role": "user", "content": "Python有什么优点？"},
        {"role": "assistant", "content": "Python的优点包括：语法简洁、学习曲线平缓、丰富的第三方库、跨平台支持等。"},
        {"role": "user", "content": "如何开始学习Python？"},
        {"role": "assistant", "content": "建议从基础语法开始，然后学习数据结构和算法，最后选择一个方向深入学习。"},
    ]
    
    result = summarizer.summarize(test_messages, style="concise")
    
    print("=" * 60)
    print("对话摘要测试")
    print("=" * 60)
    print(f"\n摘要: {result.summary}")
    print(f"\n关键点: {result.key_points}")
    print(f"\n实体: {result.entities}")
    print(f"\n压缩率: {result.compression_ratio:.1%}")
    print(f"\n统计: {summarizer.get_stats()}")
