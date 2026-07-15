"""
AGI Unified Framework - LLMLingua 提示压缩模块
=================================================

本模块实现了LLMLingua风格的提示压缩算法。

LLMLingua原理：
- 使用小型语言模型（如TinyBERT）评估每个token的重要性
- 保留高信息量的token，删除低信息量token
- 在保持语义完整性的同时显著减少token数量

主要功能：
1. 基于信息熵的token重要性评估
2. 提示压缩
3. 对话历史压缩
4. 压缩效果评估

使用示例：
    from core.prompt_compression import LLMLinguaCompressor, PromptCompressor
    
    compressor = PromptCompressor()
    compressed = compressor.compress_prompt(long_prompt, target_ratio=0.5)
"""

from __future__ import annotations

import re
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import Counter
import heapq

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Token分析工具
# =============================================================================

class Tokenizer:
    """简单分词器"""
    
    def __init__(self, method: str = "whitespace"):
        """
        Args:
            method: 分词方法 ("whitespace", "word", "char")
        """
        self.method = method
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.method == "whitespace":
            return text.split()
        elif self.method == "word":
            # 简单单词分词
            return re.findall(r'\b\w+\b|[^\w\s]|\s+', text)
        elif self.method == "char":
            return list(text)
        else:
            return text.split()
    
    def count_tokens(self, text: str) -> int:
        """估算token数量（简单方法）"""
        # 粗略估算：平均每4个字符约1个token
        return len(text) // 4


class EntropyCalculator:
    """信息熵计算器"""
    
    @staticmethod
    def word_frequency_entropy(text: str) -> Dict[str, float]:
        """
        计算单词频率熵
        
        Returns:
            {word: entropy} 字典
        """
        words = text.lower().split()
        if not words:
            return {}
        
        # 词频统计
        word_counts = Counter(words)
        total = len(words)
        
        # 计算每个词的熵贡献
        entropies = {}
        for word, count in word_counts.items():
            # 使用词频作为概率
            p = count / total
            entropy = -p * math.log2(p + 1e-10)
            entropies[word] = entropy
        
        return entropies
    
    @staticmethod
    def position_entropy(seq_len: int) -> np.ndarray:
        """
        计算位置重要性（基于Transformer的attention模式）
        
        早期token和最近token通常更重要。
        """
        positions = np.arange(seq_len)
        
        # 使用指数衰减计算重要性
        # 早期（sink）和最近更重要
        sink_weight = 0.3
        recency_weight = 0.5
        
        # Sink区域（前几个）
        sink_size = min(4, seq_len // 10)
        importance = np.ones(seq_len)
        
        # 早期sink
        importance[:sink_size] = sink_weight
        
        # 最近窗口
        window_size = seq_len // 4
        importance[-window_size:] = recency_weight
        
        # 中间部分权重较低
        if seq_len > sink_size + window_size:
            middle_weight = 0.1
            importance[sink_size:-window_size] = middle_weight
        
        return importance
    
    @staticmethod
    def conditional_entropy(context: str, target: str) -> float:
        """
        计算条件熵
        
        H(target | context) - 衡量在给定上下文后target的信息量
        """
        # 简化的实现
        # 实际需要使用n-gram统计
        
        # 使用长度作为代理
        if not target:
            return 0.0
        
        # 短文本通常信息密度高
        return 1.0 / (len(target) + 1)


# =============================================================================
# 压缩算法
# =============================================================================

@dataclass
class CompressedPrompt:
    """压缩后的提示"""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    removed_segments: List[str]
    preserved_segments: List[str]
    metadata: Dict[str, Any]


@dataclass
class CompressionResult:
    """压缩结果"""
    text: str
    removed: List[str]
    preserved: List[str]
    token_count_before: int
    token_count_after: int
    compression_ratio: float


class PromptCompressor:
    """
    提示压缩器
    
    实现多种压缩策略：
    1. 基于重要性的过滤
    2. 冗余消除
    3. 结构化压缩
    """
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()
        self.stats = {
            'total_compressions': 0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
        }
    
    def compress_prompt(self, prompt: str, 
                        target_ratio: float = 0.5,
                        method: str = "importance") -> CompressedPrompt:
        """
        压缩提示
        
        Args:
            prompt: 原始提示
            target_ratio: 目标压缩比例 (0.0 - 1.0)
            method: 压缩方法 ("importance", "redundancy", "structured")
            
        Returns:
            CompressedPrompt: 压缩结果
        """
        tokens = self.tokenizer.tokenize(prompt)
        original_count = len(tokens)
        
        if method == "importance":
            result = self._compress_by_importance(tokens, target_ratio)
        elif method == "redundancy":
            result = self._compress_by_redundancy(tokens, target_ratio)
        elif method == "structured":
            result = self._compress_structured(prompt, target_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        compressed_text = ' '.join(result.preserved)
        
        self.stats['total_compressions'] += 1
        self.stats['total_tokens_before'] += original_count
        self.stats['total_tokens_after'] += result.token_count_after
        
        return CompressedPrompt(
            original_text=prompt,
            compressed_text=compressed_text,
            original_tokens=original_count,
            compressed_tokens=result.token_count_after,
            compression_ratio=result.compression_ratio,
            removed_segments=result.removed,
            preserved_segments=result.preserved,
            metadata={
                'method': method,
                'target_ratio': target_ratio,
            }
        )
    
    def _compress_by_importance(self, tokens: List[str], 
                               target_ratio: float) -> CompressionResult:
        """基于重要性压缩"""
        seq_len = len(tokens)
        target_len = int(seq_len * target_ratio)
        
        # 计算每个token的重要性分数
        importances = self._calculate_importance(tokens)
        
        # 保留最重要的token
        if target_len >= seq_len:
            return CompressionResult(
                text=' '.join(tokens),
                removed=[],
                preserved=tokens,
                token_count_before=seq_len,
                token_count_after=seq_len,
                compression_ratio=1.0
            )
        
        # 使用最大堆找到最重要的token
        indexed_importances = [(imp, i, tok) for i, (tok, imp) in enumerate(zip(tokens, importances))]
        top_k = heapq.nlargest(target_len, indexed_importances)
        
        # 按原始顺序排列
        top_k.sort(key=lambda x: x[1])
        
        preserved = [item[2] for item in top_k]
        removed_indices = set(range(seq_len)) - {item[1] for item in top_k}
        removed = [tokens[i] for i in sorted(removed_indices)]
        
        return CompressionResult(
            text=' '.join(preserved),
            removed=removed,
            preserved=preserved,
            token_count_before=seq_len,
            token_count_after=len(preserved),
            compression_ratio=len(preserved) / seq_len
        )
    
    def _calculate_importance(self, tokens: List[str]) -> np.ndarray:
        """计算每个token的重要性"""
        seq_len = len(tokens)
        
        # 位置重要性
        position_imp = EntropyCalculator.position_entropy(seq_len)
        
        # 词频熵
        text = ' '.join(tokens)
        freq_entropy = EntropyCalculator.word_frequency_entropy(text)
        
        # 组合重要性
        importances = np.zeros(seq_len)
        
        for i, token in enumerate(tokens):
            # 位置分数
            pos_score = position_imp[i]
            
            # 词频分数（稀有词更重要）
            freq_score = 1.0 - freq_entropy.get(token.lower(), 0.5)
            
            # 组合
            importances[i] = 0.6 * pos_score + 0.4 * freq_score
        
        # 归一化
        if importances.max() > 0:
            importances = importances / importances.max()
        
        return importances
    
    def _compress_by_redundancy(self, tokens: List[str],
                               target_ratio: float) -> CompressionResult:
        """基于冗余消除的压缩"""
        seq_len = len(tokens)
        target_len = int(seq_len * target_ratio)
        
        if target_len >= seq_len:
            return CompressionResult(
                text=' '.join(tokens),
                removed=[],
                preserved=tokens,
                token_count_before=seq_len,
                token_count_after=seq_len,
                compression_ratio=1.0
            )
        
        # 找到连续的重复token
        i = 0
        unique_tokens = []
        remove_indices = set()
        
        while i < seq_len:
            token = tokens[i]
            unique_tokens.append((i, token))
            
            # 检查后续是否有相同token
            j = i + 1
            while j < seq_len and tokens[j] == token:
                remove_indices.add(j)
                j += 1
            
            i = j
        
        # 如果还不够，移除低频词
        if len(unique_tokens) > target_len:
            # 统计词频
            freq = Counter([t for _, t in unique_tokens])
            
            # 按频率排序
            unique_tokens.sort(key=lambda x: freq[x[1]])
            
            # 移除最低频的
            while len(unique_tokens) > target_len:
                idx, token = unique_tokens.pop(0)
                remove_indices.add(idx)
                freq[token] -= 1
        
        # 构建结果
        preserved = []
        removed = []
        
        for i, token in enumerate(tokens):
            if i in remove_indices:
                removed.append(token)
            else:
                preserved.append(token)
        
        return CompressionResult(
            text=' '.join(preserved),
            removed=removed,
            preserved=preserved,
            token_count_before=seq_len,
            token_count_after=len(preserved),
            compression_ratio=len(preserved) / seq_len
        )
    
    def _compress_structured(self, text: str, target_ratio: float) -> CompressionResult:
        """结构化压缩"""
        original_len = len(text)
        target_len = int(original_len * target_ratio)
        
        # 识别结构
        sentences = re.split(r'[.!?。！？]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return CompressionResult(
                text=text,
                removed=[],
                preserved=[text],
                token_count_before=original_len,
                token_count_after=original_len,
                compression_ratio=1.0
            )
        
        # 计算每个句子的重要性（长度 + 关键词）
        sentence_importance = []
        keywords = self._extract_keywords(text)
        
        for i, sent in enumerate(sentences):
            # 长度分数
            len_score = len(sent) / max(len(text), 1)
            
            # 关键词匹配
            keyword_score = sum(1 for kw in keywords if kw in sent.lower()) / max(1, len(keywords))
            
            # 位置分数（开头和结尾更重要）
            pos_score = 1.0
            if i < 2:
                pos_score = 1.2
            elif i >= len(sentences) - 2:
                pos_score = 1.1
            
            importance = 0.5 * len_score + 0.3 * keyword_score + 0.2 * pos_score
            sentence_importance.append((i, sent, importance))
        
        # 按重要性排序，保留最重要的
        sentence_importance.sort(key=lambda x: x[2], reverse=True)
        
        # 计算目标字符数
        accumulated = 0
        preserved_sentences = []
        removed_sentences = []
        
        for idx, sent, imp in sentence_importance:
            if accumulated + len(sent) <= target_len * 1.2:  # 允许10%的overhead
                preserved_sentences.append((idx, sent))
                accumulated += len(sent)
            else:
                removed_sentences.append(sent)
        
        # 按原始顺序排列
        preserved_sentences.sort(key=lambda x: x[0])
        final_text = '. '.join([s for _, s in preserved_sentences])
        
        return CompressionResult(
            text=final_text,
            removed=removed_sentences,
            preserved=[s for _, s in preserved_sentences],
            token_count_before=self.tokenizer.count_tokens(text),
            token_count_after=self.tokenizer.count_tokens(final_text),
            compression_ratio=len(final_text) / max(1, original_len)
        )
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        # 简单实现：使用词频
        words = text.lower().split()
        word_counts = Counter(words)
        
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as',
                    'of', 'at', 'by', 'for', 'with', 'about', 'into', 'through',
                    'to', 'from', 'in', 'on', 'this', 'that', 'these', 'those'}
        
        filtered = {w: c for w, c in word_counts.items() 
                   if w not in stopwords and len(w) > 2}
        
        # 返回top_k
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_k]]
    
    def compress_conversation(self, messages: List[Dict[str, str]],
                           target_ratio: float = 0.5) -> List[Dict[str, str]]:
        """
        压缩对话历史
        
        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            target_ratio: 目标压缩比例
            
        Returns:
            压缩后的对话
        """
        compressed = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if not content:
                continue
            
            # 特别处理：保留system和最近的user消息
            if role == 'system':
                # System消息通常重要，压缩少一些
                result = self.compress_prompt(content, target_ratio=0.8)
            elif role == 'user':
                # 用户消息正常压缩
                result = self.compress_prompt(content, target_ratio=target_ratio)
            else:
                # Assistant消息，适度压缩
                result = self.compress_prompt(content, target_ratio=target_ratio * 0.7)
            
            compressed.append({
                'role': role,
                'content': result.compressed_text,
                'compressed': result.compression_ratio < 0.95,
                'original_length': result.original_tokens,
                'compressed_length': result.compressed_tokens,
            })
        
        return compressed
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.stats['total_compressions'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'avg_compression_ratio': (
                self.stats['total_tokens_after'] / 
                max(1, self.stats['total_tokens_before'])
            ),
            'tokens_saved': (
                self.stats['total_tokens_before'] - 
                self.stats['total_tokens_after']
            ),
        }


# =============================================================================
# 自适应压缩控制器
# =============================================================================

class AdaptiveCompressor:
    """
    自适应压缩控制器
    
    根据对话上下文动态调整压缩策略。
    """
    
    def __init__(self, base_compressor: Optional[PromptCompressor] = None):
        self.compressor = base_compressor or PromptCompressor()
        self.context_window = 10  # 考虑的最近消息数
        self.adaptive_ratio = 0.5  # 默认压缩比例
    
    def compress_with_context(self, prompt: str,
                           conversation_history: List[Dict[str, str]] = None) -> str:
        """
        带上下文的压缩
        
        考虑对话历史来调整压缩策略。
        """
        # 分析上下文复杂度
        complexity = self._analyze_complexity(prompt, conversation_history)
        
        # 根据复杂度调整压缩比例
        if complexity == "high":
            self.adaptive_ratio = 0.7  # 保留更多内容
        elif complexity == "medium":
            self.adaptive_ratio = 0.5
        else:
            self.adaptive_ratio = 0.3  # 可以压缩更多
        
        # 执行压缩
        result = self.compressor.compress_prompt(prompt, target_ratio=self.adaptive_ratio)
        return result.compressed_text
    
    def _analyze_complexity(self, prompt: str,
                          history: List[Dict[str, str]] = None) -> str:
        """分析任务复杂度"""
        # 基于多个指标评估
        
        # 长度指标
        length = len(prompt)
        if length > 2000:
            length_score = "high"
        elif length > 500:
            length_score = "medium"
        else:
            length_score = "low"
        
        # 关键词检测
        complex_keywords = ['analyze', 'compare', 'explain', 'solve', 'calculate',
                          'implement', 'design', 'create', 'optimize', 'debug']
        has_complex = any(kw in prompt.lower() for kw in complex_keywords)
        
        # 代码检测
        has_code = '```' in prompt or 'def ' in prompt or 'class ' in prompt
        
        # 综合评估
        if has_code or has_complex:
            return "high"
        elif length_score == "high":
            return "medium"
        else:
            return "low"
    
    def estimate_tokens_saved(self, prompt: str, target_ratio: float) -> int:
        """估算节省的token数量"""
        tokenizer = self.compressor.tokenizer
        current_tokens = tokenizer.count_tokens(prompt)
        saved_tokens = int(current_tokens * (1 - target_ratio))
        return saved_tokens


# =============================================================================
# 压缩效果评估
# =============================================================================

class CompressionEvaluator:
    """压缩效果评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate(self, original: str, compressed: str,
                task_accuracy_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        评估压缩效果
        
        Args:
            original: 原始提示
            compressed: 压缩后提示
            task_accuracy_fn: 任务准确率评估函数
            
        Returns:
            评估结果
        """
        orig_tokens = len(original.split())
        comp_tokens = len(compressed.split())
        
        result = {
            'compression_ratio': comp_tokens / max(1, orig_tokens),
            'tokens_saved': orig_tokens - comp_tokens,
            'length_reduction': 1 - len(compressed) / max(1, len(original)),
            'task_accuracy_degradation': None,
        }
        
        # 如果有准确率函数，评估任务性能
        if task_accuracy_fn:
            orig_acc = task_accuracy_fn(original)
            comp_acc = task_accuracy_fn(compressed)
            result['original_accuracy'] = orig_acc
            result['compressed_accuracy'] = comp_acc
            result['task_accuracy_degradation'] = orig_acc - comp_acc
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """获取汇总统计"""
        if not self.results:
            return {}
        
        ratios = [r['compression_ratio'] for r in self.results]
        lengths = [r['length_reduction'] for r in self.results]
        
        summary = {
            'total_evaluations': len(self.results),
            'avg_compression_ratio': np.mean(ratios),
            'avg_length_reduction': np.mean(lengths),
            'min_compression_ratio': np.min(ratios),
            'max_compression_ratio': np.max(ratios),
        }
        
        if 'task_accuracy_degradation' in self.results[0]:
            degradations = [r['task_accuracy_degradation'] for r in self.results if r['task_accuracy_degradation'] is not None]
            if degradations:
                summary['avg_accuracy_degradation'] = np.mean(degradations)
        
        return summary


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'Tokenizer',
    'EntropyCalculator',
    'CompressedPrompt',
    'CompressionResult',
    'PromptCompressor',
    'AdaptiveCompressor',
    'CompressionEvaluator',
]
