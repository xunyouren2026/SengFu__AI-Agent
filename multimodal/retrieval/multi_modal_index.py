"""
多模态向量索引
实现高效的向量存储和检索
"""
from typing import Optional, List, Dict, Any, Tuple
import math
import heapq


class VectorIndex:
    """向量索引基类
    
    提供默认的暴力搜索实现。子类可覆盖 add/search 以实现更高效的索引结构。
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []
    
    def _normalize(self, x: List[float]) -> List[float]:
        """L2归一化向量"""
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return sum(x * y for x, y in zip(a_norm, b_norm))
    
    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """添加向量到索引
        
        Args:
            vector: 要添加的向量
            metadata: 可选的元数据
        """
        if len(vector) != self.dim:
            raise ValueError(f"向量维度 {len(vector)} 与索引维度 {self.dim} 不匹配")
        self.vectors.append(vector.copy())
        self.metadata.append(metadata if metadata is not None else {})
    
    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """搜索最相似的向量
        
        默认实现使用暴力搜索（余弦相似度）。
        
        Args:
            query: 查询向量
            k: 返回的结果数量
            
        Returns:
            (索引, 相似度, 元数据) 列表，按相似度降序排列
        """
        if not self.vectors:
            return []
        
        k = min(k, len(self.vectors))
        similarities: List[Tuple[int, float, Dict[str, Any]]] = []
        
        for i, vector in enumerate(self.vectors):
            sim = self._cosine_similarity(query, vector)
            similarities.append((i, sim, self.metadata[i]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class FlatIndex(VectorIndex):
    """暴力搜索索引"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return sum(x * y for x, y in zip(a_norm, b_norm))
    
    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """添加向量"""
        self.vectors.append(vector.copy())
        self.metadata.append(metadata or {})
    
    def add_batch(self, vectors: List[List[float]], 
                  metadatas: Optional[List[Dict[str, Any]]] = None):
        """批量添加向量"""
        metadatas = metadatas or [{}] * len(vectors)
        for vector, meta in zip(vectors, metadatas):
            self.add(vector, meta)
    
    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """搜索最相似的向量"""
        if not self.vectors:
            return []
        
        # 计算所有相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            sim = self._cosine_similarity(query, vector)
            similarities.append((i, sim, self.metadata[i]))
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def search_batch(self, queries: List[List[float]], 
                     k: int = 10) -> List[List[Tuple[int, float, Dict[str, Any]]]]:
        """批量搜索"""
        return [self.search(q, k) for q in queries]


class IVFIndex(VectorIndex):
    """倒排文件索引 (简化版)"""
    
    def __init__(self, dim: int, n_clusters: int = 100):
        super().__init__(dim)
        self.n_clusters = n_clusters
        self.centroids: List[List[float]] = []
        self.cluster_vectors: List[List[Tuple[int, List[float]]]] = []
        self.n_probe = 10  # 搜索时探测的聚类数
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return sum(x * y for x, y in zip(a_norm, b_norm))
    
    def _kmeans(self, vectors: List[List[float]], 
                n_clusters: int, n_iters: int = 10) -> List[List[float]]:
        """简化的K-means聚类"""
        if len(vectors) < n_clusters:
            return vectors.copy()
        
        # 随机初始化质心
        indices = list(range(len(vectors)))
        import random
        random.shuffle(indices)
        centroids = [vectors[i] for i in indices[:n_clusters]]
        
        for _ in range(n_iters):
            # 分配点到最近的质心
            clusters = [[] for _ in range(n_clusters)]
            for v in vectors:
                min_dist = float('inf')
                best_cluster = 0
                for i, c in enumerate(centroids):
                    dist = sum((a - b) ** 2 for a, b in zip(v, c))
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = i
                clusters[best_cluster].append(v)
            
            # 更新质心
            for i in range(n_clusters):
                if clusters[i]:
                    dim = len(clusters[i][0])
                    new_centroid = [
                        sum(v[d] for v in clusters[i]) / len(clusters[i])
                        for d in range(dim)
                    ]
                    centroids[i] = new_centroid
        
        return centroids
    
    def build_index(self):
        """构建索引"""
        if len(self.vectors) < self.n_clusters:
            self.centroids = self.vectors.copy()
            self.cluster_vectors = [[(i, v)] for i, v in enumerate(self.vectors)]
            return
        
        # 聚类
        self.centroids = self._kmeans(self.vectors, self.n_clusters)
        
        # 分配向量到聚类
        self.cluster_vectors = [[] for _ in range(self.n_clusters)]
        for i, v in enumerate(self.vectors):
            min_dist = float('inf')
            best_cluster = 0
            for j, c in enumerate(self.centroids):
                dist = sum((a - b) ** 2 for a, b in zip(v, c))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j
            self.cluster_vectors[best_cluster].append((i, v))
    
    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """添加向量"""
        self.vectors.append(vector.copy())
        self.metadata.append(metadata or {})
    
    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """搜索"""
        if not self.vectors:
            return []
        
        if not self.centroids:
            # 使用暴力搜索
            return FlatIndex(self.dim).search(query, k)
        
        # 找到最近的聚类
        cluster_dists = []
        for i, c in enumerate(self.centroids):
            dist = sum((a - b) ** 2 for a, b in zip(query, c))
            cluster_dists.append((i, dist))
        
        cluster_dists.sort(key=lambda x: x[1])
        
        # 在最近的聚类中搜索
        candidates = []
        for i, _ in cluster_dists[:self.n_probe]:
            candidates.extend(self.cluster_vectors[i])
        
        # 计算相似度
        similarities = []
        for idx, vec in candidates:
            sim = self._cosine_similarity(query, vec)
            similarities.append((idx, sim, self.metadata[idx]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class MultiModalIndex:
    """多模态向量索引"""
    
    def __init__(self, modalities: List[str], dim: int):
        self.modalities = modalities
        self.dim = dim
        
        # 为每个模态创建索引
        self.indices: Dict[str, VectorIndex] = {}
        for mod in modalities:
            self.indices[mod] = FlatIndex(dim)
        
        # 跨模态映射
        self.cross_modal_map: Dict[str, Dict[int, Dict[str, int]]] = {}
        for mod in modalities:
            self.cross_modal_map[mod] = {}
    
    def add(self, modality: str, vector: List[float], 
            metadata: Optional[Dict[str, Any]] = None,
            cross_modal_ids: Optional[Dict[str, int]] = None):
        """
        添加向量
        
        Args:
            modality: 模态名称
            vector: 向量
            metadata: 元数据
            cross_modal_ids: 跨模态ID映射
        """
        if modality not in self.indices:
            raise ValueError(f"Unknown modality: {modality}")
        
        idx = len(self.indices[modality].vectors)
        self.indices[modality].add(vector, metadata)
        
        # 存储跨模态映射
        if cross_modal_ids:
            self.cross_modal_map[modality][idx] = cross_modal_ids
    
    def add_batch(self, modality: str, vectors: List[List[float]], 
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  cross_modal_ids_list: Optional[List[Dict[str, int]]] = None):
        """批量添加"""
        metadatas = metadatas or [{}] * len(vectors)
        cross_modal_ids_list = cross_modal_ids_list or [None] * len(vectors)
        
        for vector, meta, cm_ids in zip(vectors, metadatas, cross_modal_ids_list):
            self.add(modality, vector, meta, cm_ids)
    
    def search(self, modality: str, query: List[float], 
               k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """在指定模态中搜索"""
        if modality not in self.indices:
            return []
        return self.indices[modality].search(query, k)
    
    def search_cross_modal(self, query_modality: str, query: List[float], 
                           target_modality: str, k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        跨模态搜索
        
        Args:
            query_modality: 查询模态
            query: 查询向量
            target_modality: 目标模态
            k: 返回数量
        
        Returns:
            目标模态中的搜索结果
        """
        # 首先在查询模态中搜索
        results = self.search(query_modality, query, k)
        
        # 映射到目标模态
        cross_results = []
        seen_ids = set()
        
        for idx, score, meta in results:
            # 查找跨模态映射
            if idx in self.cross_modal_map.get(query_modality, {}):
                cm_ids = self.cross_modal_map[query_modality][idx]
                if target_modality in cm_ids:
                    target_idx = cm_ids[target_modality]
                    if target_idx not in seen_ids:
                        seen_ids.add(target_idx)
                        target_meta = self.indices[target_modality].metadata[target_idx]
                        cross_results.append((target_idx, score, target_meta))
        
        # 如果跨模态映射不足，直接在目标模态中搜索
        if len(cross_results) < k:
            direct_results = self.search(target_modality, query, k)
            for idx, score, meta in direct_results:
                if idx not in seen_ids:
                    seen_ids.add(idx)
                    cross_results.append((idx, score, meta))
        
        cross_results.sort(key=lambda x: x[1], reverse=True)
        return cross_results[:k]
    
    def get_vector(self, modality: str, idx: int) -> Optional[List[float]]:
        """获取指定向量"""
        if modality in self.indices and idx < len(self.indices[modality].vectors):
            return self.indices[modality].vectors[idx]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        for mod, index in self.indices.items():
            stats[mod] = {
                'count': len(index.vectors),
                'dim': index.dim
            }
        return stats


class ProductQuantizationIndex(VectorIndex):
    """乘积量化索引 (简化版)"""
    
    def __init__(self, dim: int, n_subvectors: int = 8, n_centroids: int = 256):
        super().__init__(dim)
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.subvector_dim = dim // n_subvectors
        
        # 码本
        self.codebooks: List[List[List[float]]] = []
        
        # 编码后的向量
        self.codes: List[List[int]] = []
    
    def _quantize_subvector(self, subvector: List[float], 
                            codebook: List[List[float]]) -> int:
        """量化子向量"""
        min_dist = float('inf')
        best_code = 0
        
        for i, centroid in enumerate(codebook):
            dist = sum((a - b) ** 2 for a, b in zip(subvector, centroid))
            if dist < min_dist:
                min_dist = dist
                best_code = i
        
        return best_code
    
    def _train_codebooks(self, vectors: List[List[float]], n_iters: int = 10):
        """训练码本"""
        import random
        
        self.codebooks = []
        
        for m in range(self.n_subvectors):
            # 提取子向量
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = [v[start:end] for v in vectors]
            
            # K-means聚类
            if len(subvectors) < self.n_centroids:
                codebook = subvectors.copy()
            else:
                # 随机初始化
                indices = list(range(len(subvectors)))
                random.shuffle(indices)
                codebook = [subvectors[i] for i in indices[:self.n_centroids]]
                
                for _ in range(n_iters):
                    clusters = [[] for _ in range(self.n_centroids)]
                    for sv in subvectors:
                        min_dist = float('inf')
                        best = 0
                        for i, c in enumerate(codebook):
                            dist = sum((a - b) ** 2 for a, b in zip(sv, c))
                            if dist < min_dist:
                                min_dist = dist
                                best = i
                        clusters[best].append(sv)
                    
                    for i in range(self.n_centroids):
                        if clusters[i]:
                            codebook[i] = [
                                sum(sv[d] for sv in clusters[i]) / len(clusters[i])
                                for d in range(len(clusters[i][0]))
                            ]
            
            self.codebooks.append(codebook)
    
    def build_index(self):
        """构建索引"""
        if not self.vectors:
            return
        
        # 训练码本
        self._train_codebooks(self.vectors)
        
        # 编码所有向量
        self.codes = []
        for v in self.vectors:
            code = []
            for m in range(self.n_subvectors):
                start = m * self.subvector_dim
                end = start + self.subvector_dim
                subvector = v[start:end]
                c = self._quantize_subvector(subvector, self.codebooks[m])
                code.append(c)
            self.codes.append(code)
    
    def add(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """添加向量"""
        self.vectors.append(vector.copy())
        self.metadata.append(metadata or {})
    
    def _decode(self, code: List[int]) -> List[float]:
        """解码向量"""
        decoded = []
        for m, c in enumerate(code):
            decoded.extend(self.codebooks[m][c])
        return decoded
    
    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """搜索"""
        if not self.codes or not self.codebooks:
            return []
        
        # 量化查询
        query_code = []
        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            subvector = query[start:end]
            c = self._quantize_subvector(subvector, self.codebooks[m])
            query_code.append(c)
        
        # 计算距离表
        distance_tables = []
        for m in range(self.n_subvectors):
            start = m * self.subvector_dim
            end = start + self.subvector_dim
            query_sub = query[start:end]
            
            table = []
            for centroid in self.codebooks[m]:
                dist = sum((a - b) ** 2 for a, b in zip(query_sub, centroid))
                table.append(dist)
            distance_tables.append(table)
        
        # 计算近似距离
        distances = []
        for i, code in enumerate(self.codes):
            dist = sum(distance_tables[m][c] for m, c in enumerate(code))
            distances.append((i, dist, self.metadata[i]))
        
        # 排序
        distances.sort(key=lambda x: x[1])
        
        # 转换为相似度
        results = []
        for idx, dist, meta in distances[:k]:
            sim = 1.0 / (1.0 + math.sqrt(dist))
            results.append((idx, sim, meta))
        
        return results


def create_vector_index(dim: int, index_type: str = 'flat') -> VectorIndex:
    """创建向量索引"""
    if index_type == 'flat':
        return FlatIndex(dim)
    elif index_type == 'ivf':
        return IVFIndex(dim)
    elif index_type == 'pq':
        return ProductQuantizationIndex(dim)
    return FlatIndex(dim)
