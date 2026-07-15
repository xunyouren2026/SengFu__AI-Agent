"""
Context Pruner - 上下文修剪器
智能修剪上下文，保留关键信息，优化Token使用
"""

import re
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class PruningStrategy(Enum):
    """修剪策略"""
    FIFO = "fifo"                     # 先进先出
    IMPORTANCE = "importance"         # 重要性优先
    RECENCY = "recency"               # 最近性优先
    RELEVANCE = "relevance"           # 相关性优先
    SEMANTIC = "semantic"             # 语义修剪
    HYBRID = "hybrid"                 # 混合策略


class ContentType(Enum):
    """内容类型"""
    SYSTEM = "system"       # 系统消息
    USER = "user"           # 用户消息
    ASSISTANT = "assistant" # 助手消息
    FUNCTION = "function"   # 函数调用
    CONTEXT = "context"     # 上下文信息


@dataclass
class Message:
    """消息"""
    id: str
    role: str
    content: str
    token_count: int
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    content_type: ContentType = ContentType.USER
    entities: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content[:200],
            "token_count": self.token_count,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "content_type": self.content_type.value
        }


@dataclass
class PruningResult:
    """修剪结果"""
    original_messages: List[Message]
    pruned_messages: List[Message]
    removed_messages: List[Message]
    original_tokens: int
    pruned_tokens: int
    tokens_saved: int
    strategy: PruningStrategy
    pruning_ratio: float
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": len(self.original_messages),
            "pruned_count": len(self.pruned_messages),
            "removed_count": len(self.removed_messages),
            "original_tokens": self.original_tokens,
            "pruned_tokens": self.pruned_tokens,
            "tokens_saved": self.tokens_saved,
            "pruning_ratio": self.pruning_ratio,
            "strategy": self.strategy.value,
            "processing_time": self.processing_time
        }


@dataclass
class PruningConfig:
    """修剪配置"""
    # Token限制
    max_tokens: int = 4096
    target_tokens: int = 3000
    min_tokens: int = 500
    
    # 保留配置
    preserve_system: bool = True
    preserve_first_n: int = 2
    preserve_last_n: int = 2
    
    # 策略配置
    strategy: PruningStrategy = PruningStrategy.HYBRID
    
    # 重要性阈值
    importance_threshold: float = 0.3
    
    # 语义配置
    enable_semantic_pruning: bool = True
    semantic_threshold: float = 0.7
    
    # 压缩配置
    enable_compression: bool = True
    compression_target_ratio: float = 0.5


class ImportanceScorer:
    """重要性评分器"""
    
    def __init__(self):
        self._important_keywords = {
            'important', 'key', 'critical', 'essential', 'must',
            '重要', '关键', '必须', '核心', '注意'
        }
        
        self._question_words = {'what', 'why', 'how', 'when', 'where', 'who',
                                '什么', '为什么', '怎么', '如何', '哪'}
    
    def score(self, message: Message, context: List[Message]) -> float:
        """计算消息重要性"""
        score = 0.5  # 基础分
        
        # 角色因素
        if message.role == "system":
            score += 0.3
        elif message.role == "user":
            score += 0.1
        
        # 内容类型因素
        if message.content_type == ContentType.SYSTEM:
            score += 0.2
        
        # 关键词因素
        content_lower = message.content.lower()
        for kw in self._important_keywords:
            if kw in content_lower:
                score += 0.1
        
        # 问题因素
        for qw in self._question_words:
            if qw in content_lower:
                score += 0.05
        
        # 实体因素
        if message.entities:
            score += min(0.2, len(message.entities) * 0.05)
        
        # 长度因素（适中长度更重要）
        length_factor = 1 - abs(message.token_count - 100) / 500
        score += max(0, length_factor * 0.1)
        
        # 上下文相关性
        if len(context) > 1:
            relevance = self._calculate_relevance(message, context)
            score += relevance * 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_relevance(self, message: Message, context: List[Message]) -> float:
        """计算与上下文的相关性"""
        if not message.keywords:
            return 0.5
        
        # 收集上下文关键词
        context_keywords = set()
        for msg in context:
            if msg.id != message.id:
                context_keywords.update(msg.keywords)
        
        if not context_keywords:
            return 0.5
        
        # 计算重叠
        overlap = len(message.keywords & context_keywords)
        return min(1.0, overlap / len(message.keywords))


class RecencyCalculator:
    """最近性计算器"""
    
    def calculate(self, message: Message, current_time: float) -> float:
        """计算最近性分数"""
        age = current_time - message.timestamp
        
        # 指数衰减
        decay_rate = 0.1  # 每小时衰减10%
        recency = math.exp(-decay_rate * age / 3600)
        
        return recency


class RelevanceCalculator:
    """相关性计算器"""
    
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
    
    def calculate(
        self,
        message: Message,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """计算与查询的相关性"""
        if not self.embedding_model:
            # 使用关键词匹配
            return self._keyword_relevance(message, query)
        
        # 使用嵌入相似度
        return self._embedding_relevance(message, query, query_embedding)
    
    def _keyword_relevance(self, message: Message, query: str) -> float:
        """关键词相关性"""
        query_words = set(query.lower().split())
        message_words = set(message.content.lower().split())
        
        if not query_words or not message_words:
            return 0.0
        
        overlap = len(query_words & message_words)
        return overlap / len(query_words)
    
    def _embedding_relevance(
        self,
        message: Message,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> float:
        """嵌入相关性"""
        try:
            if message.embedding is None:
                message.embedding = self.embedding_model.encode(message.content)
            
            if query_embedding is None:
                query_embedding = self.embedding_model.encode(query)
            
            return self._cosine_similarity(query_embedding, message.embedding)
        except Exception as e:
            logger.warning(f"嵌入计算失败: {e}")
            return self._keyword_relevance(message, query)
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class SemanticPruner:
    """语义修剪器"""
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        threshold: float = 0.7
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
    
    def find_redundant(
        self,
        messages: List[Message]
    ) -> List[Tuple[str, str, float]]:
        """
        找出冗余消息对
        
        Returns:
            [(msg_id1, msg_id2, similarity), ...]
        """
        if not self.embedding_model:
            return []
        
        # 计算嵌入
        embeddings = {}
        for msg in messages:
            if msg.embedding is None:
                try:
                    msg.embedding = self.embedding_model.encode(msg.content)
                    embeddings[msg.id] = msg.embedding
                except Exception:
                    continue
            else:
                embeddings[msg.id] = msg.embedding
        
        # 计算相似度
        redundant = []
        msg_ids = list(embeddings.keys())
        
        for i in range(len(msg_ids)):
            for j in range(i + 1, len(msg_ids)):
                sim = self._cosine_similarity(
                    embeddings[msg_ids[i]],
                    embeddings[msg_ids[j]]
                )
                if sim >= self.threshold:
                    redundant.append((msg_ids[i], msg_ids[j], sim))
        
        return redundant
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class ContextPruner:
    """上下文修剪器主类"""
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        embedding_model: Optional[Any] = None
    ):
        self.config = config or PruningConfig()
        self.embedding_model = embedding_model
        
        # 组件
        self.importance_scorer = ImportanceScorer()
        self.recency_calculator = RecencyCalculator()
        self.relevance_calculator = RelevanceCalculator(embedding_model)
        self.semantic_pruner = SemanticPruner(
            embedding_model,
            self.config.semantic_threshold
        ) if self.config.enable_semantic_pruning else None
        
        self._message_id_counter = 0
        self._lock = threading.Lock()
    
    def _generate_message_id(self) -> str:
        """生成消息ID"""
        self._message_id_counter += 1
        return f"msg_{self._message_id_counter}"
    
    def prune(
        self,
        messages: List[Message],
        target_tokens: Optional[int] = None,
        query: Optional[str] = None
    ) -> PruningResult:
        """
        修剪上下文
        
        Args:
            messages: 消息列表
            target_tokens: 目标Token数
            query: 当前查询（用于相关性计算）
        
        Returns:
            PruningResult
        """
        start_time = time.time()
        
        if not messages:
            return PruningResult(
                original_messages=[],
                pruned_messages=[],
                removed_messages=[],
                original_tokens=0,
                pruned_tokens=0,
                tokens_saved=0,
                strategy=self.config.strategy,
                pruning_ratio=0.0
            )
        
        target_tokens = target_tokens or self.config.target_tokens
        
        # 计算原始Token数
        original_tokens = sum(m.token_count for m in messages)
        
        # 如果已经在限制内，直接返回
        if original_tokens <= target_tokens:
            return PruningResult(
                original_messages=messages,
                pruned_messages=messages,
                removed_messages=[],
                original_tokens=original_tokens,
                pruned_tokens=original_tokens,
                tokens_saved=0,
                strategy=self.config.strategy,
                pruning_ratio=0.0
            )
        
        # 计算消息分数
        scored_messages = self._score_messages(messages, query)
        
        # 选择修剪策略
        if self.config.strategy == PruningStrategy.HYBRID:
            pruned, removed = self._hybrid_prune(scored_messages, target_tokens)
        elif self.config.strategy == PruningStrategy.IMPORTANCE:
            pruned, removed = self._importance_prune(scored_messages, target_tokens)
        elif self.config.strategy == PruningStrategy.RECENCY:
            pruned, removed = self._recency_prune(scored_messages, target_tokens)
        elif self.config.strategy == PruningStrategy.RELEVANCE:
            pruned, removed = self._relevance_prune(scored_messages, target_tokens, query)
        elif self.config.strategy == PruningStrategy.SEMANTIC:
            pruned, removed = self._semantic_prune(scored_messages, target_tokens)
        else:
            pruned, removed = self._fifo_prune(scored_messages, target_tokens)
        
        # 计算结果
        pruned_tokens = sum(m.token_count for m in pruned)
        
        return PruningResult(
            original_messages=messages,
            pruned_messages=pruned,
            removed_messages=removed,
            original_tokens=original_tokens,
            pruned_tokens=pruned_tokens,
            tokens_saved=original_tokens - pruned_tokens,
            strategy=self.config.strategy,
            pruning_ratio=1 - (pruned_tokens / original_tokens),
            processing_time=time.time() - start_time
        )
    
    def _score_messages(
        self,
        messages: List[Message],
        query: Optional[str] = None
    ) -> List[Tuple[Message, Dict[str, float]]]:
        """计算消息分数"""
        current_time = time.time()
        scored = []
        
        for msg in messages:
            scores = {
                "importance": self.importance_scorer.score(msg, messages),
                "recency": self.recency_calculator.calculate(msg, current_time),
                "position": self._position_score(msg, messages)
            }
            
            if query:
                scores["relevance"] = self.relevance_calculator.calculate(msg, query)
            
            scored.append((msg, scores))
        
        return scored
    
    def _position_score(self, message: Message, messages: List[Message]) -> float:
        """计算位置分数"""
        try:
            idx = messages.index(message)
            total = len(messages)
            
            # 首尾消息更重要
            if idx < self.config.preserve_first_n:
                return 1.0
            elif idx >= total - self.config.preserve_last_n:
                return 0.9
            else:
                # 中间消息按位置递减
                return 0.5 - (idx / total) * 0.3
        except ValueError:
            return 0.5
    
    def _hybrid_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """混合修剪"""
        # 计算综合分数
        def combined_score(item):
            msg, scores = item
            return (
                scores["importance"] * 0.35 +
                scores["recency"] * 0.25 +
                scores["position"] * 0.2 +
                scores.get("relevance", 0.5) * 0.2
            )
        
        # 排序
        sorted_messages = sorted(scored_messages, key=combined_score, reverse=True)
        
        # 选择消息
        selected = []
        current_tokens = 0
        preserved_ids = set()
        
        # 先保留必须保留的消息
        for msg, scores in scored_messages:
            if self._must_preserve(msg, len(scored_messages)):
                selected.append(msg)
                current_tokens += msg.token_count
                preserved_ids.add(msg.id)
        
        # 按分数添加其他消息
        for msg, scores in sorted_messages:
            if msg.id in preserved_ids:
                continue
            
            if current_tokens + msg.token_count <= target_tokens:
                selected.append(msg)
                current_tokens += msg.token_count
        
        # 按原始顺序排序
        original_order = {msg.id: i for i, (msg, _) in enumerate(scored_messages)}
        selected.sort(key=lambda m: original_order[m.id])
        
        # 确定移除的消息
        selected_ids = {m.id for m in selected}
        removed = [m for m, _ in scored_messages if m.id not in selected_ids]
        
        return selected, removed
    
    def _must_preserve(self, message: Message, total_messages: int) -> bool:
        """检查是否必须保留"""
        if self.config.preserve_system and message.role == "system":
            return True
        if self.config.preserve_system and message.content_type == ContentType.SYSTEM:
            return True
        return False
    
    def _importance_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """重要性修剪"""
        sorted_messages = sorted(
            scored_messages,
            key=lambda x: x[1]["importance"],
            reverse=True
        )
        
        selected = []
        current_tokens = 0
        
        for msg, scores in sorted_messages:
            if current_tokens + msg.token_count <= target_tokens:
                selected.append(msg)
                current_tokens += msg.token_count
        
        selected_ids = {m.id for m in selected}
        removed = [m for m, _ in scored_messages if m.id not in selected_ids]
        
        return selected, removed
    
    def _recency_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """最近性修剪"""
        sorted_messages = sorted(
            scored_messages,
            key=lambda x: x[1]["recency"],
            reverse=True
        )
        
        selected = []
        current_tokens = 0
        
        for msg, scores in sorted_messages:
            if current_tokens + msg.token_count <= target_tokens:
                selected.append(msg)
                current_tokens += msg.token_count
        
        selected_ids = {m.id for m in selected}
        removed = [m for m, _ in scored_messages if m.id not in selected_ids]
        
        return selected, removed
    
    def _relevance_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int,
        query: Optional[str]
    ) -> Tuple[List[Message], List[Message]]:
        """相关性修剪"""
        if not query:
            return self._importance_prune(scored_messages, target_tokens)
        
        sorted_messages = sorted(
            scored_messages,
            key=lambda x: x[1].get("relevance", 0),
            reverse=True
        )
        
        selected = []
        current_tokens = 0
        
        for msg, scores in sorted_messages:
            if current_tokens + msg.token_count <= target_tokens:
                selected.append(msg)
                current_tokens += msg.token_count
        
        selected_ids = {m.id for m in selected}
        removed = [m for m, _ in scored_messages if m.id not in selected_ids]
        
        return selected, removed
    
    def _semantic_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """语义修剪"""
        messages = [m for m, _ in scored_messages]
        
        # 找出冗余消息
        redundant = self.semantic_pruner.find_redundant(messages) if self.semantic_pruner else []
        
        # 标记要移除的消息
        to_remove = set()
        for id1, id2, sim in redundant:
            # 移除较短的消息
            msg1 = next((m for m in messages if m.id == id1), None)
            msg2 = next((m for m in messages if m.id == id2), None)
            
            if msg1 and msg2:
                if msg1.token_count < msg2.token_count:
                    to_remove.add(id1)
                else:
                    to_remove.add(id2)
        
        # 检查Token是否足够
        selected = [m for m in messages if m.id not in to_remove]
        current_tokens = sum(m.token_count for m in selected)
        
        # 如果仍然超过目标，使用重要性修剪
        if current_tokens > target_tokens:
            remaining_scored = [(m, s) for m, s in scored_messages if m.id not in to_remove]
            return self._importance_prune(remaining_scored, target_tokens)
        
        removed = [m for m in messages if m.id in to_remove]
        return selected, removed
    
    def _fifo_prune(
        self,
        scored_messages: List[Tuple[Message, Dict[str, float]]],
        target_tokens: int
    ) -> Tuple[List[Message], List[Message]]:
        """先进先出修剪"""
        selected = []
        current_tokens = 0
        
        # 从后向前选择
        for msg, scores in reversed(scored_messages):
            if current_tokens + msg.token_count <= target_tokens:
                selected.insert(0, msg)
                current_tokens += msg.token_count
        
        selected_ids = {m.id for m in selected}
        removed = [m for m, _ in scored_messages if m.id not in selected_ids]
        
        return selected, removed
    
    def create_message(
        self,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        content_type: ContentType = ContentType.USER,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """创建消息"""
        if token_count is None:
            # 简单估算
            token_count = len(content) // 4
        
        return Message(
            id=self._generate_message_id(),
            role=role,
            content=content,
            token_count=token_count,
            content_type=content_type,
            keywords=self._extract_keywords(content),
            metadata=metadata or {}
        )
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        return {w for w in words if len(w) > 1}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "config": {
                "max_tokens": self.config.max_tokens,
                "target_tokens": self.config.target_tokens,
                "strategy": self.config.strategy.value,
                "preserve_system": self.config.preserve_system,
                "enable_semantic_pruning": self.config.enable_semantic_pruning
            }
        }


# 工厂函数
def create_context_pruner(
    config: Optional[PruningConfig] = None,
    embedding_model: Optional[Any] = None
) -> ContextPruner:
    """创建上下文修剪器"""
    return ContextPruner(config, embedding_model)
