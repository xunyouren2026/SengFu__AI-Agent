"""
跨模态检索器
实现不同模态之间的检索功能
"""
from typing import Optional, List, Dict, Any, Tuple
import math


class CrossModalRetriever:
    """跨模态检索器基类"""
    
    def __init__(self, modalities: List[str], embed_dim: int):
        self.modalities = modalities
        self.embed_dim = embed_dim
    
    def retrieve(self, query_modality: str, query_embedding: List[float],
                 target_modality: str, k: int = 10) -> List[Tuple[int, float]]:
        """跨模态检索的默认实现
        
        使用余弦相似度在目标模态的嵌入空间中检索与查询最相似的 top-k 项。
        
        Args:
            query_modality: 查询所属的模态名称
            query_embedding: 查询的特征嵌入向量
            target_modality: 目标检索的模态名称
            k: 返回的结果数量
            
        Returns:
            (索引, 相似度分数) 的列表，按相似度降序排列
        """
        if not query_embedding:
            return []
        
        # 归一化查询向量
        query_norm = math.sqrt(sum(x ** 2 for x in query_embedding))
        if query_norm == 0:
            return []
        query_normalized = [x / query_norm for x in query_embedding]
        
        # 默认使用简单的暴力搜索：生成一些示例索引
        # 子类应覆盖此方法以使用实际的嵌入存储
        results: List[Tuple[int, float]] = []
        dim = len(query_embedding)
        
        # 如果没有嵌入存储，返回空结果
        if not hasattr(self, '_embeddings') or not self._embeddings:
            return []
        
        target_embeddings = self._embeddings.get(target_modality, [])
        for i, emb in enumerate(target_embeddings):
            emb_norm = math.sqrt(sum(x ** 2 for x in emb))
            if emb_norm == 0:
                continue
            emb_normalized = [x / emb_norm for x in emb]
            sim = sum(q * e for q, e in zip(query_normalized, emb_normalized))
            results.append((i, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class SimpleCrossModalRetriever(CrossModalRetriever):
    """简单跨模态检索器"""
    
    def __init__(self, modalities: List[str], embed_dim: int):
        super().__init__(modalities, embed_dim)
        
        # 各模态的嵌入存储
        self.embeddings: Dict[str, List[List[float]]] = {mod: [] for mod in modalities}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {mod: [] for mod in modalities}
        
        # 跨模态关联
        self.cross_links: Dict[str, Dict[int, Dict[str, int]]] = {mod: {} for mod in modalities}
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return sum(x * y for x, y in zip(a_norm, b_norm))
    
    def add_item(self, modality: str, embedding: List[float],
                 metadata: Optional[Dict[str, Any]] = None,
                 cross_links: Optional[Dict[str, int]] = None):
        """添加项目"""
        if modality not in self.embeddings:
            raise ValueError(f"Unknown modality: {modality}")
        
        idx = len(self.embeddings[modality])
        self.embeddings[modality].append(embedding.copy())
        self.metadata[modality].append(metadata or {})
        
        if cross_links:
            self.cross_links[modality][idx] = cross_links
    
    def add_batch(self, modality: str, embeddings: List[List[float]],
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  cross_links_list: Optional[List[Dict[str, int]]] = None):
        """批量添加"""
        metadatas = metadatas or [{}] * len(embeddings)
        cross_links_list = cross_links_list or [None] * len(embeddings)
        
        for emb, meta, links in zip(embeddings, metadatas, cross_links_list):
            self.add_item(modality, emb, meta, links)
    
    def retrieve(self, query_modality: str, query_embedding: List[float],
                 target_modality: str, k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        跨模态检索
        
        Args:
            query_modality: 查询模态
            query_embedding: 查询嵌入
            target_modality: 目标模态
            k: 返回数量
        
        Returns:
            (索引, 相似度, 元数据)列表
        """
        if target_modality not in self.embeddings:
            return []
        
        target_embeddings = self.embeddings[target_modality]
        if not target_embeddings:
            return []
        
        # 计算相似度
        similarities = []
        for i, emb in enumerate(target_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((i, sim, self.metadata[target_modality][i]))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def retrieve_with_filter(self, query_embedding: List[float],
                             target_modality: str,
                             filter_fn: callable,
                             k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """带过滤的检索"""
        if target_modality not in self.embeddings:
            return []
        
        similarities = []
        for i, emb in enumerate(self.embeddings[target_modality]):
            meta = self.metadata[target_modality][i]
            if filter_fn(meta):
                sim = self._cosine_similarity(query_embedding, emb)
                similarities.append((i, sim, meta))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def retrieve_cross_linked(self, query_modality: str, query_idx: int,
                              target_modality: str) -> Optional[Tuple[int, float, Dict[str, Any]]]:
        """通过跨模态链接检索"""
        if query_idx in self.cross_links.get(query_modality, {}):
            links = self.cross_links[query_modality][query_idx]
            if target_modality in links:
                target_idx = links[target_modality]
                if target_idx < len(self.embeddings[target_modality]):
                    emb = self.embeddings[target_modality][target_idx]
                    query_emb = self.embeddings[query_modality][query_idx]
                    sim = self._cosine_similarity(query_emb, emb)
                    meta = self.metadata[target_modality][target_idx]
                    return (target_idx, sim, meta)
        return None


class WeightedCrossModalRetriever(SimpleCrossModalRetriever):
    """加权跨模态检索器"""
    
    def __init__(self, modalities: List[str], embed_dim: int):
        super().__init__(modalities, embed_dim)
        
        # 各模态的权重
        self.modality_weights: Dict[str, float] = {mod: 1.0 for mod in modalities}
    
    def set_modality_weight(self, modality: str, weight: float):
        """设置模态权重"""
        self.modality_weights[modality] = weight
    
    def retrieve_multi_modal(self, query_embeddings: Dict[str, List[float]],
                             target_modality: str,
                             k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        多模态查询检索
        
        Args:
            query_embeddings: 各模态的查询嵌入
            target_modality: 目标模态
            k: 返回数量
        
        Returns:
            检索结果
        """
        if target_modality not in self.embeddings:
            return []
        
        target_embeddings = self.embeddings[target_modality]
        if not target_embeddings:
            return []
        
        # 计算加权相似度
        weighted_sims = [0.0] * len(target_embeddings)
        total_weight = 0.0
        
        for query_mod, query_emb in query_embeddings.items():
            if query_mod not in self.modality_weights:
                continue
            
            weight = self.modality_weights[query_mod]
            total_weight += weight
            
            for i, target_emb in enumerate(target_embeddings):
                sim = self._cosine_similarity(query_emb, target_emb)
                weighted_sims[i] += weight * sim
        
        # 归一化
        if total_weight > 0:
            weighted_sims = [s / total_weight for s in weighted_sims]
        
        # 排序
        results = [(i, weighted_sims[i], self.metadata[target_modality][i]) 
                   for i in range(len(target_embeddings))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]


class HybridCrossModalRetriever(SimpleCrossModalRetriever):
    """混合跨模态检索器"""
    
    def __init__(self, modalities: List[str], embed_dim: int):
        super().__init__(modalities, embed_dim)
        
        # BM25参数 (简化版)
        self.k1 = 1.5
        self.b = 0.75
    
    def _compute_bm25_score(self, query_terms: List[str], 
                            doc_terms: List[str],
                            avg_doc_len: float,
                            doc_len: int,
                            doc_freqs: Dict[str, int],
                            n_docs: int) -> float:
        """计算BM25分数 (简化版)"""
        score = 0.0
        
        for term in query_terms:
            if term in doc_freqs:
                # IDF
                idf = math.log((n_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5) + 1)
                
                # TF
                tf = doc_terms.count(term)
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
                
                score += idf * numerator / denominator
        
        return score
    
    def retrieve_hybrid(self, query_embedding: List[float],
                        query_terms: List[str],
                        target_modality: str,
                        alpha: float = 0.5,
                        k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        混合检索 (向量 + 关键词)
        
        Args:
            query_embedding: 查询嵌入
            query_terms: 查询词项
            target_modality: 目标模态
            alpha: 向量检索权重 (1-alpha为BM25权重)
            k: 返回数量
        
        Returns:
            检索结果
        """
        if target_modality not in self.embeddings:
            return []
        
        target_embeddings = self.embeddings[target_modality]
        n_docs = len(target_embeddings)
        
        if n_docs == 0:
            return []
        
        # 计算向量相似度
        vector_scores = []
        for emb in target_embeddings:
            sim = self._cosine_similarity(query_embedding, emb)
            vector_scores.append(sim)
        
        # 归一化向量分数
        max_vec = max(vector_scores) if vector_scores else 1.0
        min_vec = min(vector_scores) if vector_scores else 0.0
        vec_range = max_vec - min_vec if max_vec > min_vec else 1.0
        vector_scores = [(s - min_vec) / vec_range for s in vector_scores]
        
        # 计算BM25分数 (简化处理)
        bm25_scores = []
        avg_doc_len = 10.0  # 简化
        doc_freqs = {}  # 简化
        
        for i, meta in enumerate(self.metadata[target_modality]):
            doc_terms = meta.get('terms', [])
            doc_len = len(doc_terms)
            score = self._compute_bm25_score(query_terms, doc_terms, avg_doc_len, 
                                             doc_len, doc_freqs, n_docs)
            bm25_scores.append(score)
        
        # 归一化BM25分数
        if bm25_scores:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
            bm25_scores = [(s - min_bm25) / bm25_range for s in bm25_scores]
        
        # 混合分数
        hybrid_scores = [alpha * vec + (1 - alpha) * bm25 
                        for vec, bm25 in zip(vector_scores, bm25_scores)]
        
        # 排序
        results = [(i, hybrid_scores[i], self.metadata[target_modality][i]) 
                   for i in range(n_docs)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]


class ReRanker:
    """重排序器"""
    
    def __init__(self):
        pass
    
    def rerank(self, query_embedding: List[float],
               candidates: List[Tuple[int, float, Dict[str, Any]]],
               embeddings: List[List[float]],
               k: int = 10) -> List[Tuple[int, float, Dict[str, Any]]]:
        """重排序"""
        # 简单的精确重排序
        results = []
        
        for idx, _, meta in candidates:
            sim = self._cosine_similarity(query_embedding, embeddings[idx])
            results.append((idx, sim, meta))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _normalize(self, x: List[float]) -> List[float]:
        norm = math.sqrt(sum(xi ** 2 for xi in x))
        return [xi / norm for xi in x] if norm > 0 else x
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_norm = self._normalize(a)
        b_norm = self._normalize(b)
        return sum(x * y for x, y in zip(a_norm, b_norm))


def create_cross_modal_retriever(modalities: List[str], 
                                  embed_dim: int) -> SimpleCrossModalRetriever:
    """创建跨模态检索器"""
    return SimpleCrossModalRetriever(modalities, embed_dim)
