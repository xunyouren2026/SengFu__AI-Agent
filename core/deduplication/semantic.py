"""
Semantic Deduplication - 语义去重系统
智能检测和去除语义重复内容
"""

import re
import time
import logging
import threading
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class DuplicateType(Enum):
    """重复类型"""
    EXACT = "exact"           # 完全相同
    NEAR = "near"             # 近似重复
    SEMANTIC = "semantic"     # 语义重复
    PARAPHRASE = "paraphrase" # 改写重复
    FUZZY = "fuzzy"          # 模糊重复


class DeduplicationLevel(Enum):
    """去重级别"""
    STRICT = "strict"        # 严格模式 - 只保留完全不同
    NORMAL = "normal"        # 正常模式 - 平衡去重
    LENIENT = "lenient"      # 宽松模式 - 只去除明显重复


@dataclass
class DuplicateRecord:
    """重复记录"""
    id: str
    original_id: str
    duplicate_type: DuplicateType
    similarity_score: float
    content_preview: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_id": self.original_id,
            "duplicate_type": self.duplicate_type.value,
            "similarity_score": self.similarity_score,
            "content_preview": self.content_preview,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


@dataclass
class ContentItem:
    """内容项"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    hash_exact: str = ""
    hash_simhash: int = 0
    hash_minhash: List[int] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.hash_exact:
            self.hash_exact = self._compute_exact_hash()
    
    def _compute_exact_hash(self) -> str:
        """计算精确哈希"""
        normalized = self._normalize_text(self.content)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """标准化文本"""
        # 转小写
        text = text.lower()
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除标点
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()


@dataclass
class DeduplicationConfig:
    """去重配置"""
    # 相似度阈值
    exact_threshold: float = 1.0
    near_threshold: float = 0.95
    semantic_threshold: float = 0.85
    paraphrase_threshold: float = 0.75
    fuzzy_threshold: float = 0.65
    
    # 性能配置
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = True
    enable_semantic_dedup: bool = True
    enable_fuzzy_dedup: bool = False
    
    # 索引配置
    use_simhash: bool = True
    use_minhash: bool = True
    minhash_permutations: int = 128
    simhash_bits: int = 64
    
    # LSH配置
    lsh_bands: int = 16
    lsh_rows: int = 8
    
    # 缓存配置
    max_cache_size: int = 10000
    dedup_level: DeduplicationLevel = DeduplicationLevel.NORMAL


class SimHash:
    """SimHash算法实现"""
    
    def __init__(self, bits: int = 64):
        self.bits = bits
    
    def compute(self, text: str) -> int:
        """计算SimHash值"""
        # 分词
        tokens = self._tokenize(text)
        
        # 初始化向量
        v = [0] * self.bits
        
        for token in tokens:
            # 计算token的hash值
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # 更新向量
            for i in range(self.bits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # 生成最终hash
        fingerprint = 0
        for i in range(self.bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单分词：按空格和标点分割
        tokens = re.findall(r'\w+', text.lower())
        # 添加n-grams
        ngrams = []
        for n in [2, 3]:
            for i in range(len(tokens) - n + 1):
                ngrams.append(' '.join(tokens[i:i + n]))
        return tokens + ngrams
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """计算汉明距离"""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1
        return distance
    
    def similarity(self, hash1: int, hash2: int) -> float:
        """计算相似度"""
        distance = self.hamming_distance(hash1, hash2)
        return 1.0 - (distance / self.bits)


class MinHash:
    """MinHash算法实现"""
    
    def __init__(self, num_permutations: int = 128, seed: int = 42):
        self.num_permutations = num_permutations
        self.seed = seed
        self._coeff_a = self._generate_coefficients()
        self._coeff_b = self._generate_coefficients()
        self._max_hash = (1 << 32) - 1
    
    def _generate_coefficients(self) -> List[int]:
        """生成随机系数"""
        import random
        random.seed(self.seed)
        return [random.randint(1, self._max_hash) for _ in range(self.num_permutations)]
    
    def compute(self, text: str) -> List[int]:
        """计算MinHash签名"""
        tokens = set(self._tokenize(text))
        
        if not tokens:
            return [self._max_hash] * self.num_permutations
        
        signature = []
        for i in range(self.num_permutations):
            min_hash = self._max_hash
            for token in tokens:
                token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._max_hash
                perm_hash = (self._coeff_a[i] * token_hash + self._coeff_b[i]) % self._max_hash
                min_hash = min(min_hash, perm_hash)
            signature.append(min_hash)
        
        return signature
    
    def _tokenize(self, text: str) -> List[str]:
        """分词（shingle）"""
        text = text.lower()
        # 生成3-shingles
        shingles = []
        for i in range(len(text) - 2):
            shingles.append(text[i:i + 3])
        return shingles
    
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """计算Jaccard相似度估计"""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LSHIndex:
    """局部敏感哈希索引"""
    
    def __init__(self, bands: int = 16, rows: int = 8):
        self.bands = bands
        self.rows = rows
        self.tables: List[Dict[str, Set[str]]] = [defaultdict(set) for _ in range(bands)]
        self._lock = threading.Lock()
    
    def insert(self, item_id: str, minhash: List[int]):
        """插入项"""
        with self._lock:
            for band_idx in range(self.bands):
                start = band_idx * self.rows
                end = start + self.rows
                band_hash = tuple(minhash[start:end])
                self.tables[band_idx][band_hash].add(item_id)
    
    def query(self, minhash: List[int]) -> Set[str]:
        """查询候选重复项"""
        candidates = set()
        with self._lock:
            for band_idx in range(self.bands):
                start = band_idx * self.rows
                end = start + self.rows
                band_hash = tuple(minhash[start:end])
                candidates.update(self.tables[band_idx].get(band_hash, set()))
        return candidates
    
    def remove(self, item_id: str, minhash: List[int]):
        """移除项"""
        with self._lock:
            for band_idx in range(self.bands):
                start = band_idx * self.rows
                end = start + self.rows
                band_hash = tuple(minhash[start:end])
                self.tables[band_idx][band_hash].discard(item_id)


class SemanticDeduplicator:
    """语义去重器主类"""
    
    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
        embedding_model: Optional[Any] = None
    ):
        self.config = config or DeduplicationConfig()
        self.embedding_model = embedding_model
        
        # 初始化哈希算法
        self.simhash = SimHash(self.config.simhash_bits) if self.config.use_simhash else None
        self.minhash = MinHash(self.config.minhash_permutations) if self.config.use_minhash else None
        self.lsh_index = LSHIndex(self.config.lsh_bands, self.config.lsh_rows) if self.config.use_minhash else None
        
        # 存储结构
        self._items: Dict[str, ContentItem] = {}
        self._exact_index: Dict[str, str] = {}  # exact_hash -> item_id
        self._simhash_index: Dict[int, List[str]] = defaultdict(list)
        
        # 重复记录
        self._duplicates: Dict[str, DuplicateRecord] = {}
        self._duplicate_id_counter = 0
        
        self._lock = threading.Lock()
    
    def _generate_duplicate_id(self) -> str:
        """生成重复记录ID"""
        self._duplicate_id_counter += 1
        return f"dup_{self._duplicate_id_counter}"
    
    def add_item(
        self, 
        item_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[DuplicateRecord]]:
        """
        添加内容项
        
        Returns:
            (是否为重复, 重复记录)
        """
        # 创建内容项
        item = ContentItem(
            id=item_id,
            content=content,
            metadata=metadata or {}
        )
        
        # 计算哈希
        if self.simhash:
            item.hash_simhash = self.simhash.compute(content)
        if self.minhash:
            item.hash_minhash = self.minhash.compute(content)
        
        # 分词
        item.tokens = self._tokenize(content)
        
        # 检查重复
        duplicate = self._check_duplicate(item)
        
        if duplicate:
            # 记录重复
            with self._lock:
                self._duplicates[duplicate.id] = duplicate
            return True, duplicate
        
        # 添加到索引
        with self._lock:
            self._items[item_id] = item
            self._exact_index[item.hash_exact] = item_id
            
            if self.simhash:
                self._simhash_index[item.hash_simhash].append(item_id)
            
            if self.minhash and self.lsh_index:
                self.lsh_index.insert(item_id, item.hash_minhash)
        
        return False, None
    
    def _check_duplicate(self, item: ContentItem) -> Optional[DuplicateRecord]:
        """检查是否重复"""
        # 1. 精确重复检查
        if self.config.enable_exact_dedup:
            if item.hash_exact in self._exact_index:
                original_id = self._exact_index[item.hash_exact]
                return DuplicateRecord(
                    id=self._generate_duplicate_id(),
                    original_id=original_id,
                    duplicate_type=DuplicateType.EXACT,
                    similarity_score=1.0,
                    content_preview=item.content[:100]
                )
        
        # 2. 近似重复检查（SimHash）
        if self.config.enable_near_dedup and self.simhash:
            for simhash, item_ids in self._simhash_index.items():
                similarity = self.simhash.similarity(item.hash_simhash, simhash)
                if similarity >= self.config.near_threshold:
                    original_id = item_ids[0]
                    return DuplicateRecord(
                        id=self._generate_duplicate_id(),
                        original_id=original_id,
                        duplicate_type=DuplicateType.NEAR,
                        similarity_score=similarity,
                        content_preview=item.content[:100]
                    )
        
        # 3. 语义重复检查（MinHash + LSH）
        if self.config.enable_semantic_dedup and self.minhash and self.lsh_index:
            candidates = self.lsh_index.query(item.hash_minhash)
            for candidate_id in candidates:
                if candidate_id not in self._items:
                    continue
                candidate = self._items[candidate_id]
                similarity = self.minhash.jaccard_similarity(
                    item.hash_minhash, 
                    candidate.hash_minhash
                )
                if similarity >= self.config.semantic_threshold:
                    return DuplicateRecord(
                        id=self._generate_duplicate_id(),
                        original_id=candidate_id,
                        duplicate_type=DuplicateType.SEMANTIC,
                        similarity_score=similarity,
                        content_preview=item.content[:100]
                    )
        
        # 4. 深度语义检查（使用嵌入模型）
        if self.config.enable_semantic_dedup and self.embedding_model:
            duplicate = self._check_semantic_embedding(item)
            if duplicate:
                return duplicate
        
        return None
    
    def _check_semantic_embedding(self, item: ContentItem) -> Optional[DuplicateRecord]:
        """使用嵌入模型检查语义重复"""
        try:
            item.embedding = self.embedding_model.encode(item.content)
        except Exception as e:
            logger.warning(f"嵌入计算失败: {e}")
            return None
        
        # 与现有项比较
        for existing_id, existing_item in self._items.items():
            if existing_item.embedding is None:
                try:
                    existing_item.embedding = self.embedding_model.encode(existing_item.content)
                except Exception:
                    continue
            
            similarity = self._cosine_similarity(item.embedding, existing_item.embedding)
            
            threshold = self.config.paraphrase_threshold
            if similarity >= threshold:
                dup_type = DuplicateType.PARAPHRASE if similarity < 0.9 else DuplicateType.SEMANTIC
                return DuplicateRecord(
                    id=self._generate_duplicate_id(),
                    original_id=existing_id,
                    duplicate_type=dup_type,
                    similarity_score=similarity,
                    content_preview=item.content[:100]
                )
        
        return None
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def batch_deduplicate(
        self, 
        items: List[Tuple[str, str]]
    ) -> Tuple[List[str], List[DuplicateRecord]]:
        """
        批量去重
        
        Args:
            items: [(item_id, content), ...]
        
        Returns:
            (唯一项ID列表, 重复记录列表)
        """
        unique_ids = []
        duplicates = []
        
        for item_id, content in items:
            is_dup, dup_record = self.add_item(item_id, content)
            if is_dup and dup_record:
                duplicates.append(dup_record)
            else:
                unique_ids.append(item_id)
        
        return unique_ids, duplicates
    
    def find_near_duplicates(
        self, 
        content: str, 
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float, DuplicateType]]:
        """
        查找近似重复项
        
        Returns:
            [(item_id, similarity, duplicate_type), ...]
        """
        threshold = threshold or self.config.semantic_threshold
        results = []
        
        # 计算查询哈希
        query_simhash = self.simhash.compute(content) if self.simhash else 0
        query_minhash = self.minhash.compute(content) if self.minhash else []
        
        # SimHash检查
        if self.simhash:
            for simhash, item_ids in self._simhash_index.items():
                similarity = self.simhash.similarity(query_simhash, simhash)
                if similarity >= threshold:
                    for item_id in item_ids:
                        results.append((item_id, similarity, DuplicateType.NEAR))
        
        # MinHash检查
        if self.minhash and self.lsh_index:
            candidates = self.lsh_index.query(query_minhash)
            for candidate_id in candidates:
                if candidate_id not in self._items:
                    continue
                candidate = self._items[candidate_id]
                similarity = self.minhash.jaccard_similarity(query_minhash, candidate.hash_minhash)
                if similarity >= threshold:
                    # 检查是否已在结果中
                    if not any(r[0] == candidate_id for r in results):
                        results.append((candidate_id, similarity, DuplicateType.SEMANTIC))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def remove_item(self, item_id: str) -> bool:
        """移除项"""
        with self._lock:
            if item_id not in self._items:
                return False
            
            item = self._items[item_id]
            
            # 从索引中移除
            self._exact_index.pop(item.hash_exact, None)
            
            if self.simhash and item.hash_simhash in self._simhash_index:
                self._simhash_index[item.hash_simhash] = [
                    id for id in self._simhash_index[item.hash_simhash] if id != item_id
                ]
            
            if self.minhash and self.lsh_index:
                self.lsh_index.remove(item_id, item.hash_minhash)
            
            del self._items[item_id]
            return True
    
    def get_duplicate_records(self, item_id: Optional[str] = None) -> List[DuplicateRecord]:
        """获取重复记录"""
        if item_id:
            return [d for d in self._duplicates.values() if d.original_id == item_id]
        return list(self._duplicates.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_items = len(self._items)
        total_duplicates = len(self._duplicates)
        
        type_counts = defaultdict(int)
        for dup in self._duplicates.values():
            type_counts[dup.duplicate_type.value] += 1
        
        return {
            "total_items": total_items,
            "total_duplicates": total_duplicates,
            "dedup_rate": total_duplicates / (total_items + total_duplicates) if total_items + total_duplicates > 0 else 0,
            "duplicate_types": dict(type_counts),
            "index_sizes": {
                "exact": len(self._exact_index),
                "simhash": len(self._simhash_index)
            }
        }
    
    def clear(self):
        """清空所有数据"""
        with self._lock:
            self._items.clear()
            self._exact_index.clear()
            self._simhash_index.clear()
            self._duplicates.clear()
            if self.lsh_index:
                self.lsh_index = LSHIndex(self.config.lsh_bands, self.config.lsh_rows)


class IncrementalDeduplicator(SemanticDeduplicator):
    """增量去重器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_queue: List[Tuple[str, str, Dict]] = []
        self._batch_size = 100
    
    def add_pending(
        self, 
        item_id: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ):
        """添加待处理项"""
        self._pending_queue.append((item_id, content, metadata or {}))
        
        if len(self._pending_queue) >= self._batch_size:
            self._process_batch()
    
    def _process_batch(self):
        """处理批量"""
        if not self._pending_queue:
            return
        
        batch = self._pending_queue[:self._batch_size]
        self._pending_queue = self._pending_queue[self._batch_size:]
        
        items = [(id, content) for id, content, _ in batch]
        metadata_map = {id: meta for id, _, meta in batch}
        
        unique_ids, duplicates = self.batch_deduplicate(items)
        
        # 更新元数据
        for item_id in unique_ids:
            if item_id in self._items:
                self._items[item_id].metadata.update(metadata_map.get(item_id, {}))
    
    def flush(self) -> Tuple[List[str], List[DuplicateRecord]]:
        """刷新待处理队列"""
        self._process_batch()
        
        # 处理剩余
        if self._pending_queue:
            items = [(id, content) for id, content, _ in self._pending_queue]
            unique_ids, duplicates = self.batch_deduplicate(items)
            self._pending_queue = []
            return unique_ids, duplicates
        
        return [], []


# 工厂函数
def create_deduplicator(
    config: Optional[DeduplicationConfig] = None,
    embedding_model: Optional[Any] = None,
    incremental: bool = False
) -> SemanticDeduplicator:
    """创建去重器"""
    if incremental:
        return IncrementalDeduplicator(config, embedding_model)
    return SemanticDeduplicator(config, embedding_model)


# 便捷函数
def deduplicate_texts(
    texts: List[str],
    threshold: float = 0.85
) -> Tuple[List[str], List[int]]:
    """
    便捷去重函数
    
    Returns:
        (唯一文本列表, 原始索引映射)
    """
    config = DeduplicationConfig(semantic_threshold=threshold)
    dedup = SemanticDeduplicator(config)
    
    unique_texts = []
    index_mapping = []
    
    for i, text in enumerate(texts):
        is_dup, _ = dedup.add_item(f"item_{i}", text)
        if not is_dup:
            unique_texts.append(text)
            index_mapping.append(i)
    
    return unique_texts, index_mapping
