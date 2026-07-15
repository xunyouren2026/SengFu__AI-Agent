"""
Memory Consolidator - 记忆整合器
整合和优化记忆存储，模拟人类记忆巩固过程
"""

import re
import time
import logging
import threading
import random
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"     # 情景记忆
    SEMANTIC = "semantic"     # 语义记忆
    PROCEDURAL = "procedural" # 程序记忆
    WORKING = "working"       # 工作记忆
    LONG_TERM = "long_term"   # 长期记忆


class ConsolidationPhase(Enum):
    """整合阶段"""
    ENCODING = "encoding"       # 编码阶段
    STABILIZATION = "stabilization"  # 稳定化阶段
    INTEGRATION = "integration" # 整合阶段
    MAINTENANCE = "maintenance" # 维护阶段


@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    memory_type: MemoryType
    importance: float
    strength: float = 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_consolidated: float = field(default_factory=time.time)
    
    # 关联信息
    entities: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    related_memories: Set[str] = field(default_factory=set)
    
    # 嵌入向量
    embedding: Optional[List[float]] = None
    
    # 元数据
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """记忆年龄（秒）"""
        return time.time() - self.created_at
    
    @property
    def decay_factor(self) -> float:
        """衰减因子"""
        # 基于艾宾浩斯遗忘曲线
        hours = self.age / 3600
        return math.exp(-hours / 24)  # 24小时半衰期
    
    @property
    def retrieval_strength(self) -> float:
        """检索强度"""
        # 综合强度 = 基础强度 * 衰减因子 * 访问加成
        access_bonus = min(0.5, self.access_count * 0.1)
        return self.strength * self.decay_factor * (1 + access_bonus)
    
    def access(self):
        """记录访问"""
        self.last_accessed = time.time()
        self.access_count += 1
        # 访问增强记忆
        self.strength = min(1.0, self.strength + 0.05)
    
    def consolidate(self):
        """整合记忆"""
        self.last_consolidated = time.time()
        # 整合增强记忆
        self.strength = min(1.0, self.strength + 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:100],
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "strength": self.strength,
            "retrieval_strength": self.retrieval_strength,
            "access_count": self.access_count,
            "age": self.age,
            "entities": list(self.entities),
            "keywords": list(self.keywords)
        }


@dataclass
class ConsolidationConfig:
    """整合配置"""
    # 容量限制
    max_memories: int = 10000
    max_working_memory: int = 100
    
    # 强度阈值
    consolidation_threshold: float = 0.5
    forgetting_threshold: float = 0.1
    
    # 整合周期
    consolidation_interval: float = 3600.0  # 1小时
    maintenance_interval: float = 86400.0   # 24小时
    
    # 模拟参数
    enable_dream_consolidation: bool = True
    dream_iterations: int = 3
    
    # 关联参数
    max_relations: int = 10
    relation_threshold: float = 0.7
    
    # 压缩参数
    enable_compression: bool = True
    compression_ratio: float = 0.3


class MemoryEncoder:
    """记忆编码器"""
    
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
    
    def encode(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5
    ) -> Memory:
        """编码新记忆"""
        memory = Memory(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            importance=importance,
            keywords=self._extract_keywords(content),
            entities=self._extract_entities(content)
        )
        
        # 计算嵌入
        if self.embedding_model:
            try:
                memory.embedding = self.embedding_model.encode(content)
            except Exception as e:
                logger.warning(f"嵌入计算失败: {e}")
        
        return memory
    
    def _generate_id(self) -> str:
        """生成记忆ID"""
        return f"mem_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        words = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        return {w for w in words if len(w) > 1}
    
    def _extract_entities(self, text: str) -> Set[str]:
        """提取实体"""
        # 简单的实体提取
        entities = set()
        
        # 大写开头的词（英文人名、地名等）
        entities.update(re.findall(r'\b[A-Z][a-z]+\b', text))
        
        # 中文实体（简化处理）
        # 实际应用中应使用NER模型
        entities.update(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
        
        return entities


class MemoryStabilizer:
    """记忆稳定器"""
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
    
    def stabilize(self, memory: Memory) -> Memory:
        """稳定记忆"""
        # 基于重要性和访问频率调整强度
        stability_factor = (
            memory.importance * 0.4 +
            min(1.0, memory.access_count / 10) * 0.3 +
            (1 - memory.decay_factor) * 0.3
        )
        
        memory.strength = min(1.0, memory.strength * (1 + stability_factor * 0.1))
        
        return memory
    
    def should_consolidate(self, memory: Memory) -> bool:
        """判断是否需要整合"""
        return (
            memory.retrieval_strength >= self.config.consolidation_threshold or
            memory.importance >= 0.8
        )
    
    def should_forget(self, memory: Memory) -> bool:
        """判断是否应该遗忘"""
        return (
            memory.retrieval_strength < self.config.forgetting_threshold and
            memory.importance < 0.3
        )


class MemoryIntegrator:
    """记忆整合器"""
    
    def __init__(
        self,
        config: ConsolidationConfig,
        embedding_model: Optional[Any] = None
    ):
        self.config = config
        self.embedding_model = embedding_model
    
    def integrate(
        self,
        memory: Memory,
        existing_memories: List[Memory]
    ) -> Tuple[Memory, List[str]]:
        """
        整合记忆到现有记忆网络
        
        Returns:
            (更新后的记忆, 关联的记忆ID列表)
        """
        related_ids = []
        
        # 找到相关记忆
        related = self._find_related_memories(memory, existing_memories)
        
        for related_memory, similarity in related[:self.config.max_relations]:
            if similarity >= self.config.relation_threshold:
                # 建立双向关联
                memory.related_memories.add(related_memory.id)
                related_memory.related_memories.add(memory.id)
                related_ids.append(related_memory.id)
        
        # 如果有相似记忆，考虑合并
        if related and related[0][1] >= 0.95:
            memory = self._merge_memories(memory, related[0][0])
        
        return memory, related_ids
    
    def _find_related_memories(
        self,
        memory: Memory,
        existing_memories: List[Memory]
    ) -> List[Tuple[Memory, float]]:
        """找到相关记忆"""
        related = []
        
        for existing in existing_memories:
            if existing.id == memory.id:
                continue
            
            similarity = self._calculate_similarity(memory, existing)
            if similarity > 0.3:
                related.append((existing, similarity))
        
        # 按相似度排序
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def _calculate_similarity(self, m1: Memory, m2: Memory) -> float:
        """计算记忆相似度"""
        score = 0.0
        
        # 关键词重叠
        if m1.keywords and m2.keywords:
            keyword_overlap = len(m1.keywords & m2.keywords)
            keyword_union = len(m1.keywords | m2.keywords)
            score += (keyword_overlap / keyword_union) * 0.4 if keyword_union > 0 else 0
        
        # 实体重叠
        if m1.entities and m2.entities:
            entity_overlap = len(m1.entities & m2.entities)
            entity_union = len(m1.entities | m2.entities)
            score += (entity_overlap / entity_union) * 0.3 if entity_union > 0 else 0
        
        # 嵌入相似度
        if m1.embedding and m2.embedding:
            emb_sim = self._cosine_similarity(m1.embedding, m2.embedding)
            score += emb_sim * 0.3
        
        return score
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _merge_memories(self, m1: Memory, m2: Memory) -> Memory:
        """合并相似记忆"""
        # 合并关键词和实体
        m1.keywords.update(m2.keywords)
        m1.entities.update(m2.entities)
        m1.related_memories.update(m2.related_memories)
        
        # 合并上下文
        m1.context.update(m2.context)
        
        # 更新强度
        m1.strength = max(m1.strength, m2.strength)
        m1.importance = max(m1.importance, m2.importance)
        
        return m1


class DreamConsolidator:
    """梦境整合器（模拟睡眠时的记忆巩固）"""
    
    def __init__(
        self,
        config: ConsolidationConfig,
        embedding_model: Optional[Any] = None
    ):
        self.config = config
        self.embedding_model = embedding_model
    
    def consolidate(
        self,
        memories: List[Memory]
    ) -> Tuple[List[Memory], List[Memory]]:
        """
        执行梦境整合
        
        Returns:
            (整合后的记忆, 遗忘的记忆)
        """
        if not memories:
            return [], []
        
        consolidated = []
        forgotten = []
        
        # 模拟多次REM睡眠周期
        for iteration in range(self.config.dream_iterations):
            memories, newly_forgotten = self._dream_cycle(memories)
            forgotten.extend(newly_forgotten)
        
        # 最终整合
        for memory in memories:
            if memory.retrieval_strength >= self.config.forgetting_threshold:
                memory.consolidate()
                consolidated.append(memory)
            else:
                forgotten.append(memory)
        
        return consolidated, forgotten
    
    def _dream_cycle(
        self,
        memories: List[Memory]
    ) -> Tuple[List[Memory], List[Memory]]:
        """单次梦境周期"""
        surviving = []
        forgotten = []
        
        for memory in memories:
            # 随机重放
            if random.random() < memory.importance:
                memory.access()
            
            # 模拟神经重塑
            if random.random() < 0.3:
                # 随机增强或减弱
                delta = random.gauss(0, 0.1)
                memory.strength = max(0, min(1.0, memory.strength + delta))
            
            # 检查是否遗忘
            if memory.retrieval_strength < self.config.forgetting_threshold:
                forgotten.append(memory)
            else:
                surviving.append(memory)
        
        return surviving, forgotten


class MemoryConsolidator:
    """记忆整合器主类"""
    
    def __init__(
        self,
        config: Optional[ConsolidationConfig] = None,
        embedding_model: Optional[Any] = None
    ):
        self.config = config or ConsolidationConfig()
        self.embedding_model = embedding_model
        
        # 组件
        self.encoder = MemoryEncoder(embedding_model)
        self.stabilizer = MemoryStabilizer(self.config)
        self.integrator = MemoryIntegrator(self.config, embedding_model)
        self.dream_consolidator = DreamConsolidator(
            self.config, embedding_model
        ) if self.config.enable_dream_consolidation else None
        
        # 存储
        self._memories: Dict[str, Memory] = {}
        self._working_memory: Dict[str, Memory] = {}
        self._type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 状态
        self._last_consolidation = time.time()
        self._last_maintenance = time.time()
        
        self._lock = threading.Lock()
    
    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        source: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """添加新记忆"""
        # 编码记忆
        memory = self.encoder.encode(content, memory_type, importance)
        memory.source = source
        memory.context = context or {}
        
        with self._lock:
            # 添加到工作记忆
            self._working_memory[memory.id] = memory
            
            # 如果工作记忆满了，转移到长期记忆
            if len(self._working_memory) > self.config.max_working_memory:
                self._transfer_to_long_term()
        
        return memory
    
    def _transfer_to_long_term(self):
        """将工作记忆转移到长期记忆"""
        # 按强度排序
        sorted_memories = sorted(
            self._working_memory.values(),
            key=lambda m: m.retrieval_strength
        )
        
        # 转移最弱的记忆
        transfer_count = len(self._working_memory) - self.config.max_working_memory // 2
        
        for memory in sorted_memories[:transfer_count]:
            # 整合到长期记忆
            memory.memory_type = MemoryType.LONG_TERM
            memory, related_ids = self.integrator.integrate(
                memory, 
                list(self._memories.values())
            )
            
            # 存储
            self._memories[memory.id] = memory
            self._type_index[MemoryType.LONG_TERM].add(memory.id)
            
            # 更新实体索引
            for entity in memory.entities:
                self._entity_index[entity].add(memory.id)
            
            # 从工作记忆移除
            del self._working_memory[memory.id]
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """获取记忆"""
        memory = self._memories.get(memory_id) or self._working_memory.get(memory_id)
        if memory:
            memory.access()
        return memory
    
    def recall(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[Tuple[Memory, float]]:
        """
        回忆记忆
        
        Returns:
            [(memory, relevance), ...]
        """
        # 提取查询关键词
        query_keywords = self.encoder._extract_keywords(query)
        
        candidates = []
        
        with self._lock:
            # 确定搜索范围
            if memory_type:
                search_ids = self._type_index.get(memory_type, set())
                memories = [self._memories[mid] for mid in search_ids if mid in self._memories]
            else:
                memories = list(self._memories.values()) + list(self._working_memory.values())
            
            # 计算相关性
            for memory in memories:
                relevance = self._calculate_relevance(memory, query_keywords, query)
                if relevance > 0.1:
                    candidates.append((memory, relevance))
        
        # 按相关性和检索强度排序
        candidates.sort(
            key=lambda x: x[1] * 0.7 + x[0].retrieval_strength * 0.3,
            reverse=True
        )
        
        # 记录访问
        for memory, _ in candidates[:limit]:
            memory.access()
        
        return candidates[:limit]
    
    def _calculate_relevance(
        self,
        memory: Memory,
        query_keywords: Set[str],
        query: str
    ) -> float:
        """计算相关性"""
        score = 0.0
        
        # 关键词匹配
        if query_keywords and memory.keywords:
            overlap = len(query_keywords & memory.keywords)
            score += (overlap / len(query_keywords)) * 0.5
        
        # 实体匹配
        query_entities = self.encoder._extract_entities(query)
        if query_entities and memory.entities:
            overlap = len(query_entities & memory.entities)
            score += (overlap / len(query_entities)) * 0.3
        
        # 嵌入相似度
        if memory.embedding and self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode(query)
                emb_sim = self.integrator._cosine_similarity(query_embedding, memory.embedding)
                score += emb_sim * 0.2
            except Exception:
                pass
        
        return score
    
    def consolidate(self) -> Dict[str, Any]:
        """
        执行记忆整合
        
        Returns:
            整合统计信息
        """
        start_time = time.time()
        stats = {
            "memories_before": len(self._memories),
            "working_before": len(self._working_memory),
            "consolidated": 0,
            "forgotten": 0,
            "relations_created": 0
        }
        
        with self._lock:
            # 1. 稳定化
            for memory in list(self._memories.values()):
                self.stabilizer.stabilize(memory)
            
            # 2. 梦境整合
            if self.dream_consolidator:
                consolidated, forgotten = self.dream_consolidator.consolidate(
                    list(self._memories.values())
                )
                
                # 更新存储
                self._memories = {m.id: m for m in consolidated}
                
                # 清理索引
                for memory in forgotten:
                    self._remove_from_indexes(memory)
                
                stats["consolidated"] = len(consolidated)
                stats["forgotten"] = len(forgotten)
            
            # 3. 建立关联
            for memory in list(self._memories.values()):
                _, related_ids = self.integrator.integrate(
                    memory,
                    [m for m in self._memories.values() if m.id != memory.id]
                )
                stats["relations_created"] += len(related_ids)
        
        self._last_consolidation = time.time()
        stats["processing_time"] = time.time() - start_time
        
        return stats
    
    def _remove_from_indexes(self, memory: Memory):
        """从索引中移除记忆"""
        self._type_index[memory.memory_type].discard(memory.id)
        for entity in memory.entities:
            self._entity_index[entity].discard(memory.id)
    
    def forget(self, memory_id: str) -> bool:
        """主动遗忘记忆"""
        with self._lock:
            memory = self._memories.pop(memory_id, None)
            if memory:
                self._remove_from_indexes(memory)
                return True
            
            if memory_id in self._working_memory:
                del self._working_memory[memory_id]
                return True
            
            return False
    
    def get_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2
    ) -> List[Memory]:
        """获取相关记忆"""
        memory = self.get_memory(memory_id)
        if not memory:
            return []
        
        related = []
        visited = {memory_id}
        queue = [(memory_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            current = self._memories.get(current_id)
            if not current:
                continue
            
            for related_id in current.related_memories:
                if related_id not in visited:
                    visited.add(related_id)
                    related_memory = self._memories.get(related_id)
                    if related_memory:
                        related.append(related_memory)
                        queue.append((related_id, depth + 1))
        
        return related
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            type_counts = {
                t.value: len(ids) 
                for t, ids in self._type_index.items()
            }
            
            avg_strength = 0.0
            avg_access = 0.0
            
            if self._memories:
                avg_strength = sum(m.strength for m in self._memories.values()) / len(self._memories)
                avg_access = sum(m.access_count for m in self._memories.values()) / len(self._memories)
            
            return {
                "total_memories": len(self._memories),
                "working_memories": len(self._working_memory),
                "memory_types": type_counts,
                "avg_strength": avg_strength,
                "avg_access_count": avg_access,
                "last_consolidation": self._last_consolidation,
                "config": {
                    "max_memories": self.config.max_memories,
                    "consolidation_threshold": self.config.consolidation_threshold,
                    "forgetting_threshold": self.config.forgetting_threshold
                }
            }
    
    def clear(self):
        """清空所有记忆"""
        with self._lock:
            self._memories.clear()
            self._working_memory.clear()
            self._type_index.clear()
            self._entity_index.clear()


# 工厂函数
def create_memory_consolidator(
    config: Optional[ConsolidationConfig] = None,
    embedding_model: Optional[Any] = None
) -> MemoryConsolidator:
    """创建记忆整合器"""
    return MemoryConsolidator(config, embedding_model)
