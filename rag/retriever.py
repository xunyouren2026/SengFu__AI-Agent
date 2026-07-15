"""
检索器模块

提供多种检索器实现，包括稠密检索、稀疏检索（BM25）、混合检索、
HyDE假设性文档嵌入检索和多查询检索。
仅使用Python标准库。
"""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag.vector_store import NaiveVectorStore, VectorStore
from rag.embedder import TextEmbedder, TFIDFEmbedder


# ============================================================
# RetrievalResult: 检索结果
# ============================================================

@dataclass
class RetrievalResult:
    """检索结果。"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # 标识来源检索器

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


# ============================================================
# Retriever: 检索器抽象基类
# ============================================================

class Retriever(ABC):
    """检索器抽象基类。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表，按分数降序排列
        """
        pass

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档。

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
        """
        pass


# ============================================================
# DenseRetriever: 稠密检索器
# ============================================================

class DenseRetriever(Retriever):
    """稠密检索器，使用向量存储进行语义搜索。

    将查询和文档都嵌入为向量，通过余弦相似度等度量进行检索。
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: Optional[VectorStore] = None,
        score_threshold: float = 0.0,
    ):
        """初始化稠密检索器。

        Args:
            embedder: 文本嵌入器
            vector_store: 向量存储（默认使用NaiveVectorStore）
            score_threshold: 分数阈值，低于此分数的结果将被过滤
        """
        self._embedder = embedder
        self._vector_store = vector_store or NaiveVectorStore(
            metric="cosine", dimension=embedder.dimension
        )
        self._score_threshold = score_threshold

        # 文档内容存储
        self._contents: Dict[str, str] = {}

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档。

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
        """
        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # 嵌入文档
            vector = self._embedder.embed(content)

            # 存储到向量存储
            self._vector_store.add(doc_id, vector, metadata)

            # 存储内容
            self._contents[doc_id] = content

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加单个文档。

        Args:
            doc_id: 文档ID
            content: 文档内容
            metadata: 元数据
        """
        vector = self._embedder.embed(content)
        self._vector_store.add(doc_id, vector, metadata or {})
        self._contents[doc_id] = content

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 嵌入查询
        query_vector = self._embedder.embed(query)

        # 搜索
        results = self._vector_store.search(query_vector, top_k=top_k * 2)

        # 转换为RetrievalResult并过滤
        retrieval_results = []
        for doc_id, score, metadata in results:
            if score >= self._score_threshold:
                content = self._contents.get(doc_id, "")
                retrieval_results.append(
                    RetrievalResult(
                        doc_id=doc_id,
                        content=content,
                        score=score,
                        metadata=metadata,
                        source="dense",
                    )
                )

        return retrieval_results[:top_k]


# ============================================================
# SparseRetriever: 稀疏检索器（BM25）
# ============================================================

class SparseRetriever(Retriever):
    """稀疏检索器，基于BM25算法。

    BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    其中:
        tf: 词频
        dl: 文档长度
        avgdl: 平均文档长度
        k1: 词频饱和参数（默认1.5）
        b: 文档长度归一化参数（默认0.75）
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """初始化BM25检索器。

        Args:
            k1: 词频饱和参数
            b: 文档长度归一化参数
            epsilon: IDF下限
        """
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon

        # 文档存储
        self._documents: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # 倒排索引: term -> {doc_id: tf}
        self._inverted_index: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # 文档频率
        self._df: Dict[str, int] = defaultdict(int)
        self._n_docs: int = 0

        # 文档长度
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_length: float = 0.0

        # IDF缓存
        self._idf_cache: Dict[str, float] = {}

        self._tokenize_pattern = re.compile(r'[a-zA-Z0-9\u4e00-\u9fff]+')

    def _tokenize(self, text: str) -> List[str]:
        """分词。

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        text = text.lower()
        tokens = self._tokenize_pattern.findall(text)
        return tokens

    def _compute_idf(self, term: str) -> float:
        """计算IDF值。

        使用 Robertson-Sparck Jones IDF 变体:
        idf = log((N - df + 0.5) / (df + 0.5)) + 1

        Args:
            term: 词项

        Returns:
            IDF值
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        df = self._df.get(term, 0)
        n = self._n_docs

        # Robertson-Sparck Jones IDF
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        # 应用下限
        idf = max(idf, self._epsilon)

        self._idf_cache[term] = idf
        return idf

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档。

        Args:
            documents: 文档列表
        """
        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            metadata = doc.get("metadata", {})
            self._add_document(doc_id, content, metadata)

    def _add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """添加单个文档到索引。"""
        self._documents[doc_id] = content
        self._metadata[doc_id] = metadata

        tokens = self._tokenize(content)
        self._doc_lengths[doc_id] = len(tokens)

        # 统计词频
        tf_counts = Counter(tokens)
        for term, tf in tf_counts.items():
            self._inverted_index[term][doc_id] = tf

        # 更新文档频率
        unique_terms = set(tokens)
        for term in unique_terms:
            self._df[term] += 1

        self._n_docs += 1

        # 更新平均文档长度
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._n_docs

        # 清除IDF缓存（因为文档数变了）
        self._idf_cache.clear()

    def _compute_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """计算BM25分数。

        Args:
            query_terms: 查询词列表
            doc_id: 文档ID

        Returns:
            BM25分数
        """
        score = 0.0
        dl = self._doc_lengths.get(doc_id, 0)
        avgdl = self._avg_doc_length if self._avg_doc_length > 0 else 1.0

        for term in query_terms:
            tf = self._inverted_index.get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue

            idf = self._compute_idf(term)

            # BM25评分公式
            numerator = tf * (self._k1 + 1.0)
            denominator = tf + self._k1 * (1.0 - self._b + self._b * dl / avgdl)

            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(doc_id, score), ...]
        """
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # 计算每个文档的BM25分数
        scores = []
        for doc_id in self._documents:
            score = self._compute_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        scores = self.search(query, top_k)

        results = []
        for doc_id, score in scores:
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=self._documents.get(doc_id, ""),
                    score=score,
                    metadata=self._metadata.get(doc_id, {}),
                    source="sparse_bm25",
                )
            )

        return results


# ============================================================
# HybridRetriever: 混合检索器
# ============================================================

class HybridRetriever(Retriever):
    """混合检索器，结合稠密和稀疏检索。

    使用倒数秩融合（Reciprocal Rank Fusion, RRF）合并结果。
    RRF分数 = sum(1 / (k + rank_i)) 对于每个检索器i

    也可以通过alpha参数控制稠密和稀疏检索的权重。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """初始化混合检索器。

        Args:
            dense_retriever: 稠密检索器
            sparse_retriever: 稀疏检索器
            alpha: 稠密检索权重（0-1），稀疏权重为 1-alpha
            rrf_k: RRF常数，用于平滑排名
        """
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._alpha = alpha
        self._rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[Tuple[str, float]]:
        """倒数秩融合。

        对每个检索器的结果按排名计算RRF分数，然后加权合并。

        Args:
            dense_results: 稠密检索结果
            sparse_results: 稀疏检索结果

        Returns:
            [(doc_id, fused_score), ...]
        """
        rrf_scores: Dict[str, float] = defaultdict(float)

        # 稠密检索的RRF分数
        for rank, result in enumerate(dense_results):
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            rrf_scores[result.doc_id] += self._alpha * rrf_score

        # 稀疏检索的RRF分数
        for rank, result in enumerate(sparse_results):
            rrf_score = 1.0 / (self._rrf_k + rank + 1)
            rrf_scores[result.doc_id] += (1.0 - self._alpha) * rrf_score

        # 排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档到两个检索器。"""
        self._dense_retriever.index_documents(documents)
        self._sparse_retriever.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。

        分别使用稠密和稀疏检索器检索，然后通过RRF合并结果。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 获取更多候选结果用于融合
        fetch_k = max(top_k * 3, 30)

        dense_results = self._dense_retriever.retrieve(query, top_k=fetch_k)
        sparse_results = self._sparse_retriever.retrieve(query, top_k=fetch_k)

        # RRF融合
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # 构建结果
        # 合并所有结果的元数据和内容
        all_results_map: Dict[str, RetrievalResult] = {}
        for r in dense_results + sparse_results:
            if r.doc_id not in all_results_map or r.score > all_results_map[r.doc_id].score:
                all_results_map[r.doc_id] = r

        results = []
        for doc_id, fused_score in fused[:top_k]:
            base = all_results_map.get(doc_id)
            if base:
                results.append(
                    RetrievalResult(
                        doc_id=base.doc_id,
                        content=base.content,
                        score=fused_score,
                        metadata=base.metadata,
                        source="hybrid",
                    )
                )

        return results


# ============================================================
# HyDERetriever: HyDE假设性文档嵌入检索
# ============================================================

class HyDERetriever(Retriever):
    """HyDE (Hypothetical Document Embedding) 检索器。

    核心思想：先生成一个假设性的答案文档，然后用这个假设文档的嵌入
    去检索，而不是直接用查询嵌入。这样可以缩小查询和文档之间的语义鸿沟。

    在本实现中，使用TF-IDF扩展来模拟假设文档生成。
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: Optional[VectorStore] = None,
        expansion_terms: int = 5,
    ):
        """初始化HyDE检索器。

        Args:
            embedder: 文本嵌入器
            vector_store: 向量存储
            expansion_terms: 查询扩展词数量
        """
        self._embedder = embedder
        self._vector_store = vector_store or NaiveVectorStore(
            metric="cosine", dimension=embedder.dimension
        )
        self._expansion_terms = expansion_terms

        self._contents: Dict[str, str] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}

        # 用于查询扩展的TF-IDF
        self._tfidf: Optional[TFIDFEmbedder] = None
        self._corpus_terms: Dict[str, float] = {}

    def _build_expansion_index(self, documents: List[Dict[str, Any]]) -> None:
        """构建查询扩展索引。

        Args:
            documents: 文档列表
        """
        corpus = [doc["content"] for doc in documents]
        self._tfidf = TFIDFEmbedder(max_features=5000, ngram_range=(1, 2))
        self._tfidf.fit(corpus)

        # 收集语料中的高频词
        term_scores: Dict[str, float] = defaultdict(float)
        for doc_content in corpus:
            top_terms = self._tfidf.get_top_terms(doc_content, top_k=20)
            for term, score in top_terms:
                term_scores[term] += score

        # 归一化
        max_score = max(term_scores.values()) if term_scores else 1.0
        self._corpus_terms = {t: s / max_score for t, s in term_scores.items()}

    def generate_hypothetical_document(self, query: str) -> str:
        """生成假设性文档。

        通过TF-IDF扩展查询词，添加相关术语来构造假设文档。

        Args:
            query: 查询文本

        Returns:
            假设性文档文本
        """
        # 基础假设文档就是查询本身
        hyp_parts = [query]

        # 使用TF-IDF扩展
        if self._tfidf is not None:
            top_terms = self._tfidf.get_top_terms(query, top_k=self._expansion_terms * 2)

            # 选择与查询相关的扩展词
            query_lower = query.lower()
            query_tokens = set(query_lower.split())

            added_terms = []
            for term, score in top_terms:
                term_lower = term.lower()
                # 跳过已在查询中的词
                if any(qt in term_lower for qt in query_tokens):
                    continue
                added_terms.append(term)
                if len(added_terms) >= self._expansion_terms:
                    break

            if added_terms:
                hyp_parts.append(" ".join(added_terms))

        # 从语料词库中添加相关词
        query_tokens = set(query.lower().split())
        related_terms = []
        for term, score in sorted(
            self._corpus_terms.items(), key=lambda x: x[1], reverse=True
        ):
            term_lower = term.lower()
            if any(qt in term_lower for qt in query_tokens):
                related_terms.append(term)
                if len(related_terms) >= self._expansion_terms:
                    break

        if related_terms:
            hyp_parts.append(" ".join(related_terms))

        return " ".join(hyp_parts)

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档。"""
        self._build_expansion_index(documents)

        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            metadata = doc.get("metadata", {})

            vector = self._embedder.embed(content)
            self._vector_store.add(doc_id, vector, metadata)
            self._contents[doc_id] = content
            self._metadata_store[doc_id] = metadata

    def embed_and_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """生成假设文档并搜索。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 生成假设文档
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 嵌入假设文档
        hyp_vector = self._embedder.embed(hypothetical_doc)

        # 搜索
        results = self._vector_store.search(hyp_vector, top_k=top_k)

        retrieval_results = []
        for doc_id, score, metadata in results:
            content = self._contents.get(doc_id, "")
            retrieval_results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                    source="hyde",
                )
            )

        return retrieval_results

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。"""
        return self.embed_and_search(query, top_k)


# ============================================================
# MultiQueryRetriever: 多查询检索
# ============================================================

class MultiQueryRetriever(Retriever):
    """多查询检索器。

    通过查询扩展（同义词、相关词）生成多个查询变体，
    分别检索后合并结果。
    """

    # 内置同义词词典
    SYNONYMS: Dict[str, List[str]] = {
        "good": ["great", "excellent", "fine", "nice", "wonderful"],
        "bad": ["poor", "terrible", "awful", "horrible"],
        "big": ["large", "huge", "enormous", "vast"],
        "small": ["little", "tiny", "miniature", "compact"],
        "fast": ["quick", "rapid", "swift", "speedy"],
        "slow": ["sluggish", "gradual", "unhurried"],
        "important": ["significant", "crucial", "vital", "essential"],
        "help": ["assist", "support", "aid", "facilitate"],
        "use": ["utilize", "employ", "apply", "operate"],
        "make": ["create", "produce", "build", "construct"],
        "find": ["discover", "locate", "identify", "detect"],
        "get": ["obtain", "acquire", "gain", "receive"],
        "show": ["display", "demonstrate", "illustrate", "reveal"],
        "tell": ["inform", "notify", "explain", "describe"],
        "how": ["what way", "method", "approach", "technique"],
        "what": ["which", "which kind", "what type"],
        "why": ["reason", "cause", "purpose", "motivation"],
        "best": ["optimal", "top", "finest", "superior"],
        "problem": ["issue", "challenge", "difficulty", "obstacle"],
        "solution": ["answer", "resolution", "remedy", "fix"],
        "data": ["information", "records", "dataset", "statistics"],
        "code": ["program", "script", "source", "implementation"],
        "algorithm": ["method", "procedure", "technique", "approach"],
        "model": ["system", "framework", "architecture", "network"],
    }

    def __init__(
        self,
        base_retriever: Retriever,
        num_queries: int = 3,
        merge_strategy: str = "rrf",
        rrf_k: int = 60,
    ):
        """初始化多查询检索器。

        Args:
            base_retriever: 基础检索器
            num_queries: 生成的查询变体数量
            merge_strategy: 合并策略 ("rrf" 或 "max")
            rrf_k: RRF常数
        """
        self._base_retriever = base_retriever
        self._num_queries = num_queries
        self._merge_strategy = merge_strategy
        self._rrf_k = rrf_k

        # 自定义同义词词典
        self._custom_synonyms: Dict[str, List[str]] = {}

    def add_synonyms(self, word: str, synonyms: List[str]) -> None:
        """添加自定义同义词。

        Args:
            word: 原词
            synonyms: 同义词列表
        """
        self._custom_synonyms[word.lower()] = [s.lower() for s in synonyms]

    def expand_query(self, query: str) -> List[str]:
        """查询扩展。

        通过同义词替换生成查询变体。

        Args:
            query: 原始查询

        Returns:
            扩展后的查询列表（包含原始查询）
        """
        queries = [query]
        query_lower = query.lower()
        query_tokens = query_lower.split()

        # 合并内置和自定义同义词
        all_synonyms = dict(self.SYNONYMS)
        all_synonyms.update(self._custom_synonyms)

        # 为每个查询词找同义词
        expansions = []
        for token in query_tokens:
            if token in all_synonyms:
                expansions.append((token, all_synonyms[token]))

        # 生成查询变体
        if expansions:
            for _ in range(self._num_queries - 1):
                variant_tokens = list(query_tokens)
                # 随机替换一些词
                import random
                num_replacements = min(len(expansions), max(1, len(expansions) // 2))
                positions = random.sample(range(len(expansions)), min(num_replacements, len(expansions)))

                for pos in positions:
                    original, syns = expansions[pos]
                    synonym = random.choice(syns)
                    for i, t in enumerate(variant_tokens):
                        if t == original:
                            variant_tokens[i] = synonym

                variant = " ".join(variant_tokens)
                if variant != query_lower:
                    queries.append(variant)

        return queries[:self._num_queries]

    def search_multiple(self, query: str, top_k: int = 10) -> List[List[RetrievalResult]]:
        """多路搜索。

        Args:
            query: 查询文本
            top_k: 每路返回数量

        Returns:
            每路检索结果列表
        """
        queries = self.expand_query(query)
        all_results = []

        for q in queries:
            results = self._base_retriever.retrieve(q, top_k=top_k)
            all_results.append(results)

        return all_results

    def merge_results(
        self,
        all_results: List[List[RetrievalResult]],
    ) -> List[RetrievalResult]:
        """合并多路检索结果。

        Args:
            all_results: 多路检索结果

        Returns:
            合并后的结果列表
        """
        if self._merge_strategy == "rrf":
            return self._merge_rrf(all_results)
        elif self._merge_strategy == "max":
            return self._merge_max(all_results)
        else:
            raise ValueError(f"未知的合并策略: {self._merge_strategy}")

    def _merge_rrf(self, all_results: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        """使用RRF合并结果。"""
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, RetrievalResult] = {}

        for results in all_results:
            for rank, result in enumerate(results):
                rrf_score = 1.0 / (self._rrf_k + rank + 1)
                rrf_scores[result.doc_id] += rrf_score

                if result.doc_id not in doc_map:
                    doc_map[result.doc_id] = result

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for doc_id, score in sorted_docs:
            base = doc_map[doc_id]
            merged.append(
                RetrievalResult(
                    doc_id=base.doc_id,
                    content=base.content,
                    score=score,
                    metadata=base.metadata,
                    source="multi_query",
                )
            )

        return merged

    def _merge_max(self, all_results: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        """使用最大分数合并结果。"""
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievalResult] = {}

        for results in all_results:
            for result in results:
                if result.doc_id not in doc_scores or result.score > doc_scores[result.doc_id]:
                    doc_scores[result.doc_id] = result.score
                    doc_map[result.doc_id] = result

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for doc_id, score in sorted_docs:
            base = doc_map[doc_id]
            merged.append(
                RetrievalResult(
                    doc_id=base.doc_id,
                    content=base.content,
                    score=score,
                    metadata=base.metadata,
                    source="multi_query",
                )
            )

        return merged

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """索引文档。"""
        self._base_retriever.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档。"""
        all_results = self.search_multiple(query, top_k=top_k)
        merged = self.merge_results(all_results)
        return merged[:top_k]


# ============================================================
# ContextWindow: 上下文窗口管理
# ============================================================

class ContextWindow:
    """上下文窗口管理。

    将检索结果适配到有限的上下文窗口中，
    支持按字符数、token数和文档数量限制。
    """

    def __init__(
        self,
        max_chars: int = 4000,
        max_tokens: int = 1000,
        max_documents: int = 10,
        separator: str = "\n\n",
    ):
        """初始化上下文窗口。

        Args:
            max_chars: 最大字符数
            max_tokens: 最大token数（粗略估计）
            max_documents: 最大文档数量
            separator: 文档间分隔符
        """
        self._max_chars = max_chars
        self._max_tokens = max_tokens
        self._max_documents = max_documents
        self._separator = separator

    def _estimate_tokens(self, text: str) -> int:
        """粗略估计token数量。

        英文约4字符/token，中文约1.5字符/token。

        Args:
            text: 输入文本

        Returns:
            估计的token数
        """
        # 统计中文字符
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars

        tokens = chinese_chars / 1.5 + other_chars / 4.0
        return int(tokens)

    def fit_results(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = False,
    ) -> str:
        """将检索结果适配到上下文窗口。

        按分数排序，依次添加文档直到达到限制。

        Args:
            results: 检索结果列表
            include_metadata: 是否包含元数据

        Returns:
            适配后的上下文文本
        """
        # 按分数排序
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        context_parts = []
        total_chars = 0
        total_tokens = 0
        doc_count = 0

        sep_len = len(self._separator)

        for result in sorted_results:
            if doc_count >= self._max_documents:
                break

            # 构建文档文本
            if include_metadata and result.metadata:
                meta_str = " | ".join(
                    f"{k}: {v}" for k, v in result.metadata.items()
                )
                doc_text = f"[{result.doc_id}] (Score: {result.score:.4f})\n{meta_str}\n{result.content}"
            else:
                doc_text = f"[{result.doc_id}] (Score: {result.score:.4f})\n{result.content}"

            # 检查是否超出限制
            doc_chars = len(doc_text) + sep_len
            doc_tokens = self._estimate_tokens(doc_text)

            if total_chars + doc_chars > self._max_chars:
                # 尝试截断
                remaining_chars = self._max_chars - total_chars - sep_len
                if remaining_chars > 50:
                    truncated = self.truncate(result.content, remaining_chars)
                    doc_text = f"[{result.doc_id}] (Score: {result.score:.4f})\n{truncated}"
                    doc_chars = len(doc_text) + sep_len
                    doc_tokens = self._estimate_tokens(doc_text)
                else:
                    break

            if total_tokens + doc_tokens > self._max_tokens:
                break

            context_parts.append(doc_text)
            total_chars += doc_chars
            total_tokens += doc_tokens
            doc_count += 1

        return self._separator.join(context_parts)

    def truncate(self, text: str, max_chars: int) -> str:
        """截断文本到指定字符数。

        尝试在句子边界处截断。

        Args:
            text: 输入文本
            max_chars: 最大字符数

        Returns:
            截断后的文本
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        # 尝试在最后一个句号处截断
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        best_pos = max(last_period, last_newline)

        if best_pos > max_chars * 0.5:
            return truncated[:best_pos + 1]

        # 在最后一个空格处截断
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.5:
            return truncated[:last_space] + "..."

        return truncated + "..."
