"""
嵌入缓存
实现特征嵌入的缓存机制
"""
from typing import Optional, List, Dict, Any, Tuple
import hashlib
import time


class EmbeddingCache:
    """嵌入缓存
    
    缓存计算过的嵌入向量，避免重复计算
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.max_size = config.get('max_size', 10000)
        self.ttl = config.get('ttl', 3600)  # 缓存过期时间（秒）
        self.enable_cache = config.get('enable_cache', True)
        
        # 缓存存储
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # 统计信息
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, modality: str, data: Any) -> str:
        """生成缓存键"""
        # 使用数据内容的哈希作为键
        if isinstance(data, str):
            content = data
        elif isinstance(data, (list, tuple)):
            # 递归转换为字符串
            content = str(data)[:1000]  # 限制长度
        else:
            content = str(data)[:1000]
        
        hash_val = hashlib.md5(content.encode()).hexdigest()
        return f"{modality}:{hash_val}"
    
    def get(self, modality: str, data: Any) -> Optional[List[List[float]]]:
        """
        获取缓存的嵌入
        
        Args:
            modality: 模态名称
            data: 输入数据
        
        Returns:
            缓存的嵌入向量，如果不存在则返回None
        """
        if not self.enable_cache:
            return None
        
        key = self._generate_key(modality, data)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # 检查过期
            if time.time() - entry['timestamp'] > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.hits += 1
            return entry['embedding']
        
        self.misses += 1
        return None
    
    def put(self, modality: str, data: Any, embedding: List[List[float]]):
        """
        存储嵌入到缓存
        
        Args:
            modality: 模态名称
            data: 输入数据
            embedding: 嵌入向量
        """
        if not self.enable_cache:
            return
        
        # 检查缓存大小
        if len(self.cache) >= self.max_size:
            self._evict()
        
        key = self._generate_key(modality, data)
        self.cache[key] = {
            'embedding': embedding,
            'timestamp': time.time(),
            'modality': modality
        }
    
    def _evict(self):
        """驱逐策略：删除最旧的条目"""
        if not self.cache:
            return
        
        # 找到最旧的条目
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
        del self.cache[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }
    
    def get_or_compute(self, modality: str, data: Any, 
                       compute_fn: callable) -> List[List[float]]:
        """
        获取缓存或计算嵌入
        
        Args:
            modality: 模态名称
            data: 输入数据
            compute_fn: 计算函数
        
        Returns:
            嵌入向量
        """
        cached = self.get(modality, data)
        if cached is not None:
            return cached
        
        # 计算
        embedding = compute_fn(data)
        
        # 缓存
        self.put(modality, data, embedding)
        
        return embedding


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, levels: List[Dict[str, Any]]):
        """
        Args:
            levels: 各级缓存配置
                   [{'max_size': 1000, 'ttl': 300}, ...]
        """
        self.caches = [EmbeddingCache(config) for config in levels]
    
    def get(self, modality: str, data: Any) -> Optional[List[List[float]]]:
        """从多级缓存获取"""
        for i, cache in enumerate(self.caches):
            result = cache.get(modality, data)
            if result is not None:
                # 提升到更高级缓存
                for j in range(i):
                    self.caches[j].put(modality, data, result)
                return result
        return None
    
    def put(self, modality: str, data: Any, embedding: List[List[float]]):
        """存储到最高级缓存"""
        if self.caches:
            self.caches[0].put(modality, data, embedding)
    
    def clear(self):
        """清空所有缓存"""
        for cache in self.caches:
            cache.clear()
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """获取各级缓存统计"""
        return [cache.get_stats() for cache in self.caches]


class PersistentCache:
    """持久化缓存"""
    
    def __init__(self, cache_dir: str = '/tmp/embedding_cache'):
        self.cache_dir = cache_dir
        self.memory_cache = EmbeddingCache()
        self._ensure_dir()
    
    def _ensure_dir(self):
        """确保缓存目录存在"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_path(self, key: str) -> str:
        """获取缓存文件路径"""
        import os
        return os.path.join(self.cache_dir, f"{key}.cache")
    
    def get(self, modality: str, data: Any) -> Optional[List[List[float]]]:
        """获取缓存"""
        # 先查内存缓存
        result = self.memory_cache.get(modality, data)
        if result is not None:
            return result
        
        # 查磁盘缓存
        key = self.memory_cache._generate_key(modality, data)
        file_path = self._get_file_path(key)
        
        try:
            import os
            if os.path.exists(file_path):
                # 简化：直接读取
                with open(file_path, 'r') as f:
                    content = f.read()
                    # 解析简单的列表格式
                    result = eval(content)
                    self.memory_cache.put(modality, data, result)
                    return result
        except Exception:
            pass
        
        return None
    
    def put(self, modality: str, data: Any, embedding: List[List[float]]):
        """存储缓存"""
        self.memory_cache.put(modality, data, embedding)
        
        # 持久化到磁盘
        key = self.memory_cache._generate_key(modality, data)
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'w') as f:
                f.write(str(embedding))
        except Exception:
            pass
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        
        import os
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self._ensure_dir()


class BatchCache:
    """批量缓存"""
    
    def __init__(self, cache: EmbeddingCache):
        self.cache = cache
        self.pending: Dict[str, List[Tuple[Any, int]]] = {}
        self.results: Dict[str, List[List[List[float]]]] = {}
    
    def add(self, modality: str, data: Any, idx: int):
        """添加待处理项"""
        if modality not in self.pending:
            self.pending[modality] = []
        self.pending[modality].append((data, idx))
    
    def process_batch(self, modality: str, compute_fn: callable):
        """批量处理"""
        if modality not in self.pending:
            return
        
        items = self.pending[modality]
        uncached = []
        uncached_indices = []
        
        # 检查缓存
        for data, idx in items:
            cached = self.cache.get(modality, data)
            if cached is not None:
                if modality not in self.results:
                    self.results[modality] = []
                self.results[modality].append((idx, cached))
            else:
                uncached.append(data)
                uncached_indices.append(idx)
        
        # 批量计算未缓存的
        if uncached:
            embeddings = compute_fn(uncached)
            
            for data, embedding, idx in zip(uncached, embeddings, uncached_indices):
                self.cache.put(modality, data, embedding)
                if modality not in self.results:
                    self.results[modality] = []
                self.results[modality].append((idx, embedding))
        
        # 清理
        del self.pending[modality]
    
    def get_results(self, modality: str) -> List[Tuple[int, List[List[float]]]]:
        """获取结果"""
        return self.results.get(modality, [])


def create_embedding_cache(max_size: int = 10000) -> EmbeddingCache:
    """创建嵌入缓存"""
    return EmbeddingCache({'max_size': max_size})
