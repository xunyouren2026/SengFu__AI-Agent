"""
RAG检索器 - RAG Retriever

实现检索增强生成，支持向量检索和关键词检索

作者: UFO Framework Team
"""

import re
import math
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import os


@dataclass
class Document:
    """文档"""
    id: str
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'title': self.title,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    highlights: List[str] = field(default_factory=list)


class SimpleEmbedding:
    """简单嵌入（基于词频的伪嵌入）"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
    
    def fit(self, documents: List[Document]) -> None:
        """训练词表"""
        # 构建词表
        doc_freq = defaultdict(int)
        
        for doc in documents:
            words = set(self._tokenize(doc.content))
            for word in words:
                doc_freq[word] += 1
        
        # 分配索引
        self.vocab = {word: i for i, word in enumerate(doc_freq.keys())}
        
        # 计算IDF
        self.doc_count = len(documents)
        for word, freq in doc_freq.items():
            self.idf[word] = math.log(self.doc_count / (1 + freq))
    
    def encode(self, text: str) -> List[float]:
        """编码文本为向量"""
        words = self._tokenize(text)
        word_freq = defaultdict(int)
        
        for word in words:
            word_freq[word] += 1
        
        # 创建TF-IDF向量
        vector = [0.0] * self.dim
        
        for word, freq in word_freq.items():
            if word in self.vocab:
                idx = self.vocab[word] % self.dim
                tf = freq / len(words) if words else 0
                idf = self.idf.get(word, 1.0)
                vector[idx] += tf * idf
        
        # L2归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        # 简单分词
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()


class VectorStore:
    """向量存储"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self._id_index: Dict[str, int] = {}
    
    def add(self, document: Document, embedding: List[float]) -> None:
        """添加文档"""
        idx = len(self.documents)
        self.documents.append(document)
        self.embeddings.append(embedding)
        self._id_index[document.id] = idx
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """搜索相似文档"""
        if not self.embeddings:
            return []
        
        # 计算余弦相似度
        scores = []
        for i, emb in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, emb)
            scores.append((i, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            (self.documents[idx], score)
            for idx, score in scores[:top_k]
        ]
    
    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float]
    ) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        idx = self._id_index.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None
    
    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        idx = self._id_index.get(doc_id)
        if idx is None:
            return False
        
        # 标记删除（简化实现）
        self.documents[idx] = None
        del self._id_index[doc_id]
        return True
    
    def count(self) -> int:
        """获取文档数量"""
        return len([d for d in self.documents if d is not None])


class KeywordIndex:
    """关键词倒排索引"""
    
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._tokenize_pattern = re.compile(r'\w+')
    
    def add(self, doc_id: str, content: str) -> None:
        """添加文档到索引"""
        words = self._tokenize(content)
        word_positions = defaultdict(list)
        
        for pos, word in enumerate(words):
            word_positions[word].append(pos)
        
        for word, positions in word_positions.items():
            self.index[word].append((doc_id, len(positions)))
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """搜索"""
        query_words = self._tokenize(query)
        
        # 计算文档分数
        doc_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.index:
                for doc_id, freq in self.index[word]:
                    doc_scores[doc_id] += freq
        
        # 排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_docs[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        return self._tokenize_pattern.findall(text.lower())


class RAGRetriever:
    """
    RAG检索器
    
    功能:
    1. 向量检索（语义相似）
    2. 关键词检索（精确匹配）
    3. 混合检索（融合两种方式）
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        self.embedding_dim = embedding_dim
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        
        # 组件
        self.embedding = SimpleEmbedding(dim=embedding_dim)
        self.vector_store = VectorStore(dim=embedding_dim)
        self.keyword_index = KeywordIndex()
        
        # 文档存储
        self.documents: Dict[str, Document] = {}
        
        # 统计
        self.stats = {
            'total_indexed': 0,
            'total_queries': 0,
            'avg_query_time_ms': 0.0
        }
    
    def index_documents(self, documents: List[Document]) -> int:
        """
        索引文档
        
        Args:
            documents: 文档列表
            
        Returns:
            索引的文档数量
        """
        # 训练嵌入模型
        self.embedding.fit(documents)
        
        # 索引每个文档
        for doc in documents:
            # 生成嵌入
            embedding = self.embedding.encode(doc.content)
            doc.embedding = embedding
            
            # 添加到向量存储
            self.vector_store.add(doc, embedding)
            
            # 添加到关键词索引
            self.keyword_index.add(doc.id, doc.content)
            
            # 存储文档
            self.documents[doc.id] = doc
        
        self.stats['total_indexed'] += len(documents)
        
        return len(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        检索
        
        Args:
            query: 查询
            top_k: 返回数量
            method: 检索方法 (vector/keyword/hybrid)
            
        Returns:
            检索结果列表
        """
        import time
        start_time = time.time()
        
        results = []
        
        if method == "vector":
            results = self._vector_search(query, top_k)
        elif method == "keyword":
            results = self._keyword_search(query, top_k)
        else:  # hybrid
            results = self._hybrid_search(query, top_k)
        
        # 更新统计
        query_time = (time.time() - start_time) * 1000
        n = self.stats['total_queries'] + 1
        self.stats['avg_query_time_ms'] = (
            (n - 1) * self.stats['avg_query_time_ms'] + query_time
        ) / n
        self.stats['total_queries'] += 1
        
        return results
    
    def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """向量检索"""
        query_embedding = self.embedding.encode(query)
        results = self.vector_store.search(query_embedding, top_k)
        
        return [
            RetrievalResult(
                document=doc,
                score=score,
                highlights=self._extract_highlights(query, doc.content)
            )
            for doc, score in results
        ]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """关键词检索"""
        results = self.keyword_index.search(query, top_k)
        
        return [
            RetrievalResult(
                document=self.documents[doc_id],
                score=score,
                highlights=self._extract_highlights(query, self.documents[doc_id].content)
            )
            for doc_id, score in results
            if doc_id in self.documents
        ]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int
    ) -> List[RetrievalResult]:
        """混合检索"""
        # 向量检索
        vector_results = self._vector_search(query, top_k * 2)
        
        # 关键词检索
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # 融合分数
        doc_scores: Dict[str, float] = {}
        
        for result in vector_results:
            doc_scores[result.document.id] = (
                doc_scores.get(result.document.id, 0) +
                result.score * self.vector_weight
            )
        
        for result in keyword_results:
            doc_scores[result.document.id] = (
                doc_scores.get(result.document.id, 0) +
                result.score * self.keyword_weight
            )
        
        # 排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            RetrievalResult(
                document=self.documents[doc_id],
                score=score,
                highlights=self._extract_highlights(query, self.documents[doc_id].content)
            )
            for doc_id, score in sorted_docs
            if doc_id in self.documents
        ]
    
    def _extract_highlights(
        self,
        query: str,
        content: str,
        max_length: int = 100
    ) -> List[str]:
        """提取高亮片段"""
        query_words = set(query.lower().split())
        sentences = re.split(r'[.!?。！？]', content)
        
        highlights = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_words = set(sent.lower().split())
            if query_words & sent_words:
                if len(sent) > max_length:
                    sent = sent[:max_length] + "..."
                highlights.append(sent)
        
        return highlights[:3]  # 最多3个高亮
    
    def add_document(
        self,
        content: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Document:
        """添加单个文档"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:16]
        
        doc = Document(
            id=doc_id,
            content=content,
            title=title,
            source=source,
            metadata=metadata or {}
        )
        
        self.index_documents([doc])
        
        return doc
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'document_count': len(self.documents),
            'vector_store_count': self.vector_store.count()
        }


# 便捷函数
def create_rag_retriever() -> RAGRetriever:
    """创建RAG检索器"""
    return RAGRetriever()


if __name__ == "__main__":
    # 测试
    retriever = RAGRetriever()
    
    # 添加文档
    docs = [
        Document(
            id="1",
            content="Python是一种流行的编程语言，广泛用于数据科学和机器学习。",
            title="Python简介"
        ),
        Document(
            id="2",
            content="机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            title="机器学习概述"
        ),
        Document(
            id="3",
            content="深度学习是机器学习的一种方法，使用神经网络进行学习。",
            title="深度学习基础"
        ),
    ]
    
    retriever.index_documents(docs)
    
    # 检索
    results = retriever.retrieve("机器学习", top_k=2)
    
    print("=" * 60)
    print("RAG检索测试")
    print("=" * 60)
    
    for result in results:
        print(f"\n文档: {result.document.title}")
        print(f"分数: {result.score:.3f}")
        print(f"内容: {result.document.content[:50]}...")
        print(f"高亮: {result.highlights}")
    
    print(f"\n统计: {retriever.get_stats()}")
