"""
FAISS向量热记忆模块 (Warm Memory Module)

该模块实现了基于FAISS的高性能向量索引和检索系统，支持多种索引策略
（Flat、IVF、HNSW），适用于大规模向量相似度搜索场景。

核心功能:
- FAISS向量索引
- 相似度搜索
- 增量更新
- 多索引策略 (Flat/IVF/HNSW)
- 记忆检索与存储

作者: AGI Universal Framework Team
版本: 1.0.0
"""

import numpy as np
import json
import pickle
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import warnings

# 尝试导入FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Using fallback implementation.")


class IndexType(Enum):
    """索引类型枚举"""
    FLAT = "flat"           # 精确搜索，暴力匹配
    IVF = "ivf"             # 倒排文件索引
    HNSW = "hnsw"           # 层次导航小世界图
    IVF_HNSW = "ivf_hnsw"   # IVF + HNSW组合
    PQ = "pq"               # 乘积量化
    IVF_PQ = "ivf_pq"       # IVF + PQ组合


@dataclass
class MemoryEntry:
    """
    记忆条目数据类
    
    Attributes:
        key: 向量键
        value: 存储的值
        metadata: 元数据
        timestamp: 时间戳
        access_count: 访问次数
        last_access: 最后访问时间
    """
    key: np.ndarray
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key.tolist(),
            'value': self.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_access': self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """从字典创建"""
        return cls(
            key=np.array(data['key']),
            value=data['value'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', time.time()),
            access_count=data.get('access_count', 0),
            last_access=data.get('last_access', time.time())
        )


@dataclass
class SearchResult:
    """
    搜索结果数据类
    
    Attributes:
        distances: 距离数组
        indices: 索引数组
        values: 值列表
        entries: 完整条目列表
    """
    distances: np.ndarray
    indices: np.ndarray
    values: List[Any]
    entries: List[MemoryEntry]


class WarmMemory:
    """
    FAISS向量热记忆
    
    基于FAISS实现的高性能向量记忆系统，支持多种索引策略和增量更新。
    
    Attributes:
        dim: 向量维度
        index_type: 索引类型
        index: FAISS索引对象
        entries: 存储的条目
        id_map: ID映射表
    """
    
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        ef_search: int = 64,
        ef_construction: int = 128,
        m: int = 16,
        metric: str = "l2",
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        初始化FAISS向量热记忆
        
        Args:
            dim: 向量维度
            index_type: 索引类型 ("flat", "ivf", "hnsw", "ivf_hnsw", "pq", "ivf_pq")
            nlist: IVF的聚类中心数
            nprobe: IVF搜索时探查的聚类数
            ef_search: HNSW搜索参数
            ef_construction: HNSW构建参数
            m: HNSW每个节点的连接数
            metric: 距离度量 ("l2", "ip", "cosine")
            use_gpu: 是否使用GPU
            gpu_id: GPU ID
        """
        self.dim = dim
        self.index_type = IndexType(index_type.lower())
        self.nlist = nlist
        self.nprobe = nprobe
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.gpu_id = gpu_id
        
        # 距离度量
        self.metric = metric
        if metric == "l2":
            self.metric_type = faiss.METRIC_L2 if FAISS_AVAILABLE else None
        elif metric == "ip":
            self.metric_type = faiss.METRIC_INNER_PRODUCT if FAISS_AVAILABLE else None
        elif metric == "cosine":
            self.metric_type = faiss.METRIC_INNER_PRODUCT if FAISS_AVAILABLE else None
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # 初始化索引
        self.index = None
        self._index_trained = False
        self._create_index()
        
        # 存储的条目和ID映射
        self.entries: Dict[int, MemoryEntry] = {}
        self.id_map: Dict[int, int] = {}  # faiss_id -> entry_id
        self._next_id = 0
        
        # 统计信息
        self._add_count = 0
        self._search_count = 0
        self._total_search_time = 0.0
        
        # 线程锁
        self._lock = threading.RLock()
    
    def _create_index(self) -> None:
        """创建FAISS索引"""
        if not FAISS_AVAILABLE:
            # 使用numpy回退实现
            self.index = None
            return
        
        if self.index_type == IndexType.FLAT:
            # 精确搜索
            if self.metric == "cosine":
                # 归一化向量后使用内积
                base_index = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIDMap(base_index)
            else:
                base_index = faiss.IndexFlat(self.dim, self.metric_type)
                self.index = faiss.IndexIDMap(base_index)
        
        elif self.index_type == IndexType.IVF:
            # IVF索引
            quantizer = faiss.IndexFlat(self.dim, self.metric_type)
            base_index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, self.metric_type)
            base_index.nprobe = self.nprobe
            self.index = faiss.IndexIDMap(base_index)
        
        elif self.index_type == IndexType.HNSW:
            # HNSW索引
            if self.metric == "cosine":
                base_index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
            else:
                base_index = faiss.IndexHNSWFlat(self.dim, self.m, self.metric_type)
            base_index.hnsw.efConstruction = self.ef_construction
            base_index.hnsw.efSearch = self.ef_search
            self.index = faiss.IndexIDMap(base_index)
        
        elif self.index_type == IndexType.IVF_HNSW:
            # IVF + HNSW
            quantizer = faiss.IndexHNSWFlat(self.dim, self.m, self.metric_type)
            quantizer.hnsw.efConstruction = self.ef_construction
            base_index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, self.metric_type)
            base_index.nprobe = self.nprobe
            self.index = faiss.IndexIDMap(base_index)
        
        elif self.index_type == IndexType.PQ:
            # 乘积量化
            base_index = faiss.IndexPQ(self.dim, self.m, 8, self.metric_type)
            self.index = faiss.IndexIDMap(base_index)
        
        elif self.index_type == IndexType.IVF_PQ:
            # IVF + PQ
            quantizer = faiss.IndexFlat(self.dim, self.metric_type)
            base_index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, 8, self.metric_type)
            base_index.nprobe = self.nprobe
            self.index = faiss.IndexIDMap(base_index)
        
        # 转移到GPU
        if self.use_gpu and FAISS_AVAILABLE:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
            except Exception as e:
                warnings.warn(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
    
    def add(
        self,
        keys: Union[np.ndarray, List[np.ndarray]],
        values: Union[List[Any], Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        添加向量到索引
        
        Args:
            keys: 向量或向量列表 [n, dim] 或 List[dim]
            values: 对应的值或值列表
            metadata: 可选的元数据列表
            
        Returns:
            添加的条目ID列表
        """
        with self._lock:
            # 标准化输入
            if isinstance(keys, list):
                keys = np.array(keys)
            
            if keys.ndim == 1:
                keys = keys.reshape(1, -1)
            
            # 归一化向量（余弦相似度）
            if self.metric == "cosine":
                keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-10)
            
            # 确保values是列表
            if not isinstance(values, list):
                values = [values]
            
            # 确保metadata是列表
            if metadata is None:
                metadata = [{} for _ in range(len(values))]
            
            # 生成条目ID
            entry_ids = list(range(self._next_id, self._next_id + len(values)))
            self._next_id += len(values)
            
            # 创建条目
            for i, (key, value, meta) in enumerate(zip(keys, values, metadata)):
                entry_id = entry_ids[i]
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    metadata=meta
                )
                self.entries[entry_id] = entry
                self.id_map[entry_id] = entry_id
            
            # 添加到FAISS索引
            if FAISS_AVAILABLE and self.index is not None:
                # 训练索引（如果需要）
                if not self._index_trained and hasattr(self.index.index, 'is_trained'):
                    if not self.index.index.is_trained:
                        self.index.index.train(keys.astype(np.float32))
                        self._index_trained = True
                
                # 添加向量
                ids = np.array(entry_ids, dtype=np.int64)
                self.index.add_with_ids(keys.astype(np.float32), ids)
            
            self._add_count += len(values)
            
            return entry_ids
    
    def search(
        self,
        query: Union[np.ndarray, List[np.ndarray]],
        k: int = 10,
        filter_fn: Optional[Callable[[MemoryEntry], bool]] = None
    ) -> SearchResult:
        """
        搜索相似向量
        
        Args:
            query: 查询向量或向量列表
            k: 返回的最相似结果数
            filter_fn: 可选的过滤函数
            
        Returns:
            搜索结果
        """
        with self._lock:
            start_time = time.time()
            
            # 标准化输入
            if isinstance(query, list):
                query = np.array(query)
            
            if query.ndim == 1:
                query = query.reshape(1, -1)
            
            # 归一化查询向量（余弦相似度）
            if self.metric == "cosine":
                query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
            
            # 执行搜索
            if FAISS_AVAILABLE and self.index is not None and self._add_count > 0:
                distances, indices = self.index.search(query.astype(np.float32), k)
            else:
                # 使用暴力搜索回退
                distances, indices = self._brute_force_search(query, k)
            
            # 收集结果
            all_distances = []
            all_indices = []
            all_values = []
            all_entries = []
            
            for i in range(len(query)):
                query_distances = distances[i]
                query_indices = indices[i]
                
                valid_mask = query_indices >= 0
                query_distances = query_distances[valid_mask]
                query_indices = query_indices[valid_mask]
                
                # 过滤
                if filter_fn is not None:
                    filtered_indices = []
                    filtered_distances = []
                    for idx, dist in zip(query_indices, query_distances):
                        if idx in self.entries:
                            entry = self.entries[idx]
                            if filter_fn(entry):
                                filtered_indices.append(idx)
                                filtered_distances.append(dist)
                    query_indices = np.array(filtered_indices)
                    query_distances = np.array(filtered_distances)
                
                # 获取条目
                query_values = []
                query_entries = []
                for idx in query_indices:
                    if idx in self.entries:
                        entry = self.entries[idx]
                        entry.access_count += 1
                        entry.last_access = time.time()
                        query_values.append(entry.value)
                        query_entries.append(entry)
                
                all_distances.append(query_distances)
                all_indices.append(query_indices)
                all_values.append(query_values)
                all_entries.append(query_entries)
            
            # 更新统计
            self._search_count += len(query)
            self._total_search_time += time.time() - start_time
            
            # 如果是单查询，展平结果
            if len(query) == 1:
                return SearchResult(
                    distances=all_distances[0],
                    indices=all_indices[0],
                    values=all_values[0],
                    entries=all_entries[0]
                )
            else:
                return SearchResult(
                    distances=np.array(all_distances),
                    indices=np.array(all_indices),
                    values=all_values,
                    entries=all_entries
                )
    
    def _brute_force_search(
        self,
        query: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        暴力搜索回退实现
        
        Args:
            query: 查询向量
            k: 返回数量
            
        Returns:
            (距离, 索引)元组
        """
        if len(self.entries) == 0:
            return np.full((len(query), k), -1.0), np.full((len(query), k), -1, dtype=np.int64)
        
        all_keys = np.array([entry.key for entry in self.entries.values()])
        all_ids = np.array(list(self.entries.keys()))
        
        distances_list = []
        indices_list = []
        
        for q in query:
            if self.metric == "l2":
                dists = np.linalg.norm(all_keys - q, axis=1)
            elif self.metric in ["ip", "cosine"]:
                dists = -np.dot(all_keys, q)  # 负内积，因为FAISS返回距离
            else:
                dists = np.linalg.norm(all_keys - q, axis=1)
            
            # 获取top-k
            k_actual = min(k, len(dists))
            top_k_idx = np.argsort(dists)[:k_actual]
            
            # 填充结果
            dists_result = np.full(k, -1.0)
            idx_result = np.full(k, -1, dtype=np.int64)
            
            dists_result[:k_actual] = dists[top_k_idx]
            idx_result[:k_actual] = all_ids[top_k_idx]
            
            distances_list.append(dists_result)
            indices_list.append(idx_result)
        
        return np.array(distances_list), np.array(indices_list)
    
    def delete(self, entry_id: int) -> bool:
        """
        删除条目
        
        Args:
            entry_id: 条目ID
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if entry_id not in self.entries:
                return False
            
            del self.entries[entry_id]
            
            if entry_id in self.id_map:
                del self.id_map[entry_id]
            
            # FAISS不直接支持删除，需要重建索引
            if FAISS_AVAILABLE and self.index is not None:
                self._rebuild_index()
            
            return True
    
    def update(
        self,
        entry_id: int,
        key: Optional[np.ndarray] = None,
        value: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        更新条目
        
        Args:
            entry_id: 条目ID
            key: 新的向量（可选）
            value: 新的值（可选）
            metadata: 新的元数据（可选）
            
        Returns:
            是否成功更新
        """
        with self._lock:
            if entry_id not in self.entries:
                return False
            
            entry = self.entries[entry_id]
            
            if key is not None:
                entry.key = key
                # 需要重建索引
                if FAISS_AVAILABLE and self.index is not None:
                    self._rebuild_index()
            
            if value is not None:
                entry.value = value
            
            if metadata is not None:
                entry.metadata.update(metadata)
            
            return True
    
    def _rebuild_index(self) -> None:
        """重建FAISS索引"""
        if not FAISS_AVAILABLE or len(self.entries) == 0:
            return
        
        # 创建新索引
        self._create_index()
        self._index_trained = False
        
        # 重新添加所有条目
        if len(self.entries) > 0:
            keys = np.array([entry.key for entry in self.entries.values()])
            ids = np.array(list(self.entries.keys()), dtype=np.int64)
            
            # 归一化
            if self.metric == "cosine":
                keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-10)
            
            # 训练并添加
            if hasattr(self.index.index, 'is_trained') and not self.index.index.is_trained:
                self.index.index.train(keys.astype(np.float32))
                self._index_trained = True
            
            self.index.add_with_ids(keys.astype(np.float32), ids)
    
    def save(self, path: str) -> bool:
        """
        保存索引和条目到磁盘
        
        Args:
            path: 保存路径
            
        Returns:
            是否成功保存
        """
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # 保存FAISS索引
            if FAISS_AVAILABLE and self.index is not None:
                # 如果索引在GPU上，先转移到CPU
                if self.use_gpu:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, f"{path}.faiss")
                else:
                    faiss.write_index(self.index, f"{path}.faiss")
            
            # 保存条目
            entries_data = {
                str(k): v.to_dict() for k, v in self.entries.items()
            }
            
            config = {
                'dim': self.dim,
                'index_type': self.index_type.value,
                'nlist': self.nlist,
                'nprobe': self.nprobe,
                'ef_search': self.ef_search,
                'ef_construction': self.ef_construction,
                'm': self.m,
                'metric': self.metric,
                'use_gpu': self.use_gpu,
                'gpu_id': self.gpu_id,
                'next_id': self._next_id,
                'add_count': self._add_count,
                'search_count': self._search_count,
                'total_search_time': self._total_search_time
            }
            
            data = {
                'config': config,
                'entries': entries_data,
                'id_map': {str(k): v for k, v in self.id_map.items()}
            }
            
            with open(f"{path}.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to save warm memory: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        从磁盘加载索引和条目
        
        Args:
            path: 加载路径
            
        Returns:
            是否成功加载
        """
        try:
            # 加载配置和条目
            with open(f"{path}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = data['config']
            self.dim = config['dim']
            self.index_type = IndexType(config['index_type'])
            self.nlist = config['nlist']
            self.nprobe = config['nprobe']
            self.ef_search = config['ef_search']
            self.ef_construction = config['ef_construction']
            self.m = config['m']
            self.metric = config['metric']
            self.use_gpu = config['use_gpu']
            self.gpu_id = config['gpu_id']
            self._next_id = config['next_id']
            self._add_count = config['add_count']
            self._search_count = config['search_count']
            self._total_search_time = config['total_search_time']
            
            # 加载条目
            self.entries = {}
            for k, v in data['entries'].items():
                self.entries[int(k)] = MemoryEntry.from_dict(v)
            
            self.id_map = {int(k): v for k, v in data['id_map'].items()}
            
            # 加载FAISS索引
            if FAISS_AVAILABLE and os.path.exists(f"{path}.faiss"):
                self.index = faiss.read_index(f"{path}.faiss")
                
                # 转移到GPU
                if self.use_gpu:
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                    except Exception as e:
                        warnings.warn(f"Failed to move index to GPU: {e}")
                        self.use_gpu = False
                
                self._index_trained = True
            else:
                # 重建索引
                self._create_index()
                self._rebuild_index()
            
            return True
        except Exception as e:
            warnings.warn(f"Failed to load warm memory: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            avg_search_time = (
                self._total_search_time / self._search_count 
                if self._search_count > 0 else 0.0
            )
            
            return {
                'dim': self.dim,
                'index_type': self.index_type.value,
                'num_entries': len(self.entries),
                'add_count': self._add_count,
                'search_count': self._search_count,
                'avg_search_time': avg_search_time,
                'metric': self.metric,
                'use_gpu': self.use_gpu,
                'faiss_available': FAISS_AVAILABLE
            }
    
    def clear(self) -> None:
        """清空所有数据"""
        with self._lock:
            self.entries.clear()
            self.id_map.clear()
            self._next_id = 0
            self._add_count = 0
            self._search_count = 0
            self._total_search_time = 0.0
            self._create_index()
    
    def get_entry(self, entry_id: int) -> Optional[MemoryEntry]:
        """
        获取指定条目
        
        Args:
            entry_id: 条目ID
            
        Returns:
            条目对象，如果不存在则返回None
        """
        with self._lock:
            return self.entries.get(entry_id)
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """获取所有条目"""
        with self._lock:
            return list(self.entries.values())


# 便捷函数
def create_warm_memory(
    dim: int,
    index_type: str = "flat",
    **kwargs
) -> WarmMemory:
    """
    创建FAISS向量热记忆的便捷函数
    
    Args:
        dim: 向量维度
        index_type: 索引类型
        **kwargs: 其他参数
        
    Returns:
        WarmMemory实例
    """
    return WarmMemory(dim=dim, index_type=index_type, **kwargs)
