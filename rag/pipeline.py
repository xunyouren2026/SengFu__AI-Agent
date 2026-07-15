"""
RAG管道模块

提供完整的RAG管道，包括文档摄入（分块、嵌入、存储）和查询检索。
仅使用Python标准库。
"""

import math
import re
import uuid
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag.vector_store import NaiveVectorStore, VectorStore
from rag.embedder import TextEmbedder, TFIDFEmbedder
from rag.retriever import (
    Retriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    RetrievalResult,
    ContextWindow,
)


# ============================================================
# Chunk: 文档块
# ============================================================

@dataclass
class Chunk:
    """文档块。"""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    doc_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "index": self.index,
            "doc_id": self.doc_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建。"""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            index=data.get("index", 0),
            doc_id=data.get("doc_id", ""),
        )


# ============================================================
# RAGResult: RAG结果
# ============================================================

@dataclass
class RAGResult:
    """RAG查询结果。"""

    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    context: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "context": self.context,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ============================================================
# DocumentProcessor: 文档预处理器
# ============================================================

class DocumentProcessor:
    """文档预处理器。

    提供文本清洗、段落分割等预处理功能。
    """

    # 常见噪声模式
    NOISE_PATTERNS = [
        (r'\t+', ' '),           # 制表符
        (r'\n{3,}', '\n\n'),     # 多个空行
        (r' {2,}', ' '),         # 多个空格
        (r'\u00a0', ' '),        # 不间断空格
        (r'\ufeff', ''),         # BOM
        (r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ''),  # 控制字符
    ]

    def __init__(self, min_paragraph_length: int = 10):
        """初始化文档预处理器。

        Args:
            min_paragraph_length: 段落最小长度（低于此长度的段落将被过滤）
        """
        self._min_paragraph_length = min_paragraph_length

    def clean_text(self, text: str) -> str:
        """文本清洗。

        去除噪声字符、规范化空白、去除特殊标记等。

        Args:
            text: 输入文本

        Returns:
            清洗后的文本
        """
        cleaned = text

        # 应用噪声模式
        for pattern, replacement in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)

        # 去除URL中的HTML实体
        cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)

        # 规范化引号
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")

        # 去除首尾空白
        cleaned = cleaned.strip()

        return cleaned

    def split_paragraphs(self, text: str) -> List[str]:
        """段落分割。

        按空行分割文本为段落，过滤过短的段落。

        Args:
            text: 输入文本

        Returns:
            段落列表
        """
        # 按一个或多个空行分割
        paragraphs = re.split(r'\n\s*\n', text)

        # 清洗并过滤
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) >= self._min_paragraph_length:
                cleaned_paragraphs.append(para)

        return cleaned_paragraphs

    def extract_sentences(self, text: str) -> List[str]:
        """提取句子。

        简单的句子分割，基于句号、问号、感叹号。

        Args:
            text: 输入文本

        Returns:
            句子列表
        """
        # 在句子结束符后分割，保留结束符
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)

        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def normalize_whitespace(self, text: str) -> str:
        """规范化空白字符。

        Args:
            text: 输入文本

        Returns:
            规范化后的文本
        """
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# ============================================================
# 分块策略
# ============================================================

class FixedSizeChunker:
    """固定大小分块器。

    将文档按固定字符数分割为块。
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """初始化固定大小分块器。

        Args:
            chunk_size: 块大小（字符数）
            overlap: 重叠大小（字符数）
        """
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str, doc_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """分块。

        Args:
            text: 输入文本
            doc_id: 文档ID
            metadata: 元数据

        Returns:
            块列表
        """
        if not text:
            return []

        metadata = metadata or {}
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self._chunk_size

            # 如果不是最后一块，尝试在句子边界处分割
            if end < len(text):
                # 查找最后一个句号或换行
                last_break = -1
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n。！？':
                        last_break = i + 1
                        break

                if last_break > start:
                    end = last_break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = index
                chunk_meta["start_char"] = start
                chunk_meta["end_char"] = end

                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{index}" if doc_id else f"chunk_{index}",
                        content=chunk_text,
                        metadata=chunk_meta,
                        index=index,
                        doc_id=doc_id,
                    )
                )

            start = end - self._overlap
            index += 1

        return chunks


class SentenceChunker:
    """按句子分块。

    将文档按句子分割，然后合并为指定大小的块。
    """

    def __init__(self, sentences_per_chunk: int = 5, overlap_sentences: int = 1):
        """初始化句子分块器。

        Args:
            sentences_per_chunk: 每块句子数
            overlap_sentences: 重叠句子数
        """
        self._sentences_per_chunk = sentences_per_chunk
        self._overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子。"""
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, doc_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """分块。

        Args:
            text: 输入文本
            doc_id: 文档ID
            metadata: 元数据

        Returns:
            块列表
        """
        if not text:
            return []

        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        step = self._sentences_per_chunk - self._overlap_sentences
        index = 0

        start_idx = 0
        while start_idx < len(sentences):
            end_idx = min(start_idx + self._sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = index
            chunk_meta["sentence_range"] = (start_idx, end_idx)

            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{index}" if doc_id else f"chunk_{index}",
                    content=chunk_text,
                    metadata=chunk_meta,
                    index=index,
                    doc_id=doc_id,
                )
            )

            start_idx += step
            index += 1

        return chunks


class SemanticChunker:
    """语义分块器。

    基于相邻句子的语义相似度进行分块。
    当相邻句子的相似度低于阈值时，创建新的块。
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        similarity_threshold: float = 0.3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """初始化语义分块器。

        Args:
            embedder: 文本嵌入器（用于计算语义相似度）
            similarity_threshold: 相似度阈值，低于此值时分割
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度。

        如果有嵌入器，使用余弦相似度；否则使用词重叠率。

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数
        """
        if self._embedder is not None:
            vec1 = self._embedder.embed(text1)
            vec2 = self._embedder.embed(text2)

            # 余弦相似度
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot / (norm1 * norm2)
        else:
            # 词重叠率作为后备
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            intersection = words1 & words2
            union = words1 | words2

            return len(intersection) / len(union) if union else 0.0

    def chunk(self, text: str, doc_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """分块。

        Args:
            text: 输入文本
            doc_id: 文档ID
            metadata: 元数据

        Returns:
            块列表
        """
        if not text:
            return []

        metadata = metadata or {}

        # 分割为句子
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # 计算相邻句子的相似度
        boundaries = [0]  # 分割点

        current_chunk_sentences = [sentences[0]]
        current_text = sentences[0]

        for i in range(1, len(sentences)):
            similarity = self._compute_similarity(current_text, sentences[i])

            # 判断是否需要分割
            should_split = False

            if similarity < self._similarity_threshold:
                if len(current_text) >= self._min_chunk_size:
                    should_split = True

            if len(current_text) + len(sentences[i]) > self._max_chunk_size:
                should_split = True

            if should_split:
                boundaries.append(i)
                current_chunk_sentences = [sentences[i]]
                current_text = sentences[i]
            else:
                current_chunk_sentences.append(sentences[i])
                current_text += " " + sentences[i]

        # 根据分割点创建块
        boundaries.append(len(sentences))

        chunks = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text.strip():
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = idx
                chunk_meta["sentence_range"] = (start, end)

                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{idx}" if doc_id else f"chunk_{idx}",
                        content=chunk_text.strip(),
                        metadata=chunk_meta,
                        index=idx,
                        doc_id=doc_id,
                    )
                )

        return chunks


class SlidingWindowChunker:
    """滑动窗口分块器。

    使用滑动窗口在文本上移动，生成有重叠的块。
    """

    def __init__(
        self,
        window_size: int = 500,
        step_size: int = 250,
        min_chunk_size: int = 50,
    ):
        """初始化滑动窗口分块器。

        Args:
            window_size: 窗口大小（字符数）
            step_size: 步长（字符数）
            min_chunk_size: 最小块大小
        """
        self._window_size = window_size
        self._step_size = step_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, text: str, doc_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """分块。

        Args:
            text: 输入文本
            doc_id: 文档ID
            metadata: 元数据

        Returns:
            块列表
        """
        if not text:
            return []

        metadata = metadata or {}
        text_len = len(text)

        if text_len <= self._window_size:
            return [
                Chunk(
                    id=f"{doc_id}_chunk_0" if doc_id else "chunk_0",
                    content=text,
                    metadata={**metadata, "chunk_index": 0},
                    index=0,
                    doc_id=doc_id,
                )
            ]

        chunks = []
        start = 0
        index = 0

        while start < text_len:
            end = min(start + self._window_size, text_len)

            # 尝试在词边界处分割
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space > start + self._min_chunk_size:
                    end = last_space

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self._min_chunk_size:
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = index
                chunk_meta["start_char"] = start
                chunk_meta["end_char"] = end

                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{index}" if doc_id else f"chunk_{index}",
                        content=chunk_text,
                        metadata=chunk_meta,
                        index=index,
                        doc_id=doc_id,
                    )
                )

            start += self._step_size
            index += 1

        return chunks


# ============================================================
# RAGPipeline: RAG管道
# ============================================================

class RAGPipeline:
    """RAG管道。

    完整的检索增强生成管道，包括：
    1. 文档摄入：分块 -> 嵌入 -> 存储
    2. 查询：嵌入查询 -> 检索 -> 组装上下文
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[Retriever] = None,
        chunker: Optional[Any] = None,
        context_window: Optional[ContextWindow] = None,
        document_processor: Optional[DocumentProcessor] = None,
    ):
        """初始化RAG管道。

        Args:
            embedder: 文本嵌入器
            vector_store: 向量存储
            retriever: 检索器（如果提供，将使用此检索器而非默认的）
            chunker: 分块器（默认使用FixedSizeChunker）
            context_window: 上下文窗口（默认使用ContextWindow）
            document_processor: 文档预处理器
        """
        self._embedder = embedder
        self._vector_store = vector_store or NaiveVectorStore(
            metric="cosine", dimension=embedder.dimension
        )
        self._retriever = retriever
        self._chunker = chunker or FixedSizeChunker(chunk_size=500, overlap=50)
        self._context_window = context_window or ContextWindow()
        self._document_processor = document_processor or DocumentProcessor()

        # 文档和块存储
        self._documents: Dict[str, str] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._chunk_contents: Dict[str, str] = {}

        # 统计信息
        self._stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
        }

    def ingest(
        self,
        documents: List[Dict[str, Any]],
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """文档摄入。

        流程：预处理 -> 分块 -> 嵌入 -> 存储

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
            preprocess: 是否预处理

        Returns:
            摄入统计信息
        """
        total_chunks = 0

        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            metadata = doc.get("metadata", {})

            # 预处理
            if preprocess:
                content = self._document_processor.clean_text(content)

            self._documents[doc_id] = content

            # 分块
            chunks = self._chunker.chunk(content, doc_id=doc_id, metadata=metadata)

            # 嵌入并存储
            for chunk in chunks:
                vector = self._embedder.embed(chunk.content)

                chunk_metadata = dict(chunk.metadata)
                chunk_metadata["doc_id"] = doc_id

                self._vector_store.add(chunk.id, vector, chunk_metadata)
                self._chunks[chunk.id] = chunk
                self._chunk_contents[chunk.id] = chunk.content

            total_chunks += len(chunks)

        self._stats["total_documents"] += len(documents)
        self._stats["total_chunks"] += total_chunks

        return {
            "documents_ingested": len(documents),
            "chunks_created": total_chunks,
            "total_documents": self._stats["total_documents"],
            "total_chunks": self._stats["total_chunks"],
        }

    def ingest_single(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """摄入单个文档。

        Args:
            doc_id: 文档ID
            content: 文档内容
            metadata: 元数据
            preprocess: 是否预处理

        Returns:
            摄入统计信息
        """
        return self.ingest(
            [{"id": doc_id, "content": content, "metadata": metadata or {}}],
            preprocess=preprocess,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_context: bool = True,
    ) -> RAGResult:
        """查询。

        流程：嵌入查询 -> 检索 -> 组装上下文 -> 返回结果

        Args:
            question: 查询问题
            top_k: 返回数量
            return_context: 是否返回上下文

        Returns:
            RAG结果
        """
        self._stats["total_queries"] += 1

        # 检索
        if self._retriever is not None:
            retrieval_results = self._retriever.retrieve(question, top_k=top_k)
        else:
            # 使用内置的稠密检索
            query_vector = self._embedder.embed(question)
            search_results = self._vector_store.search(query_vector, top_k=top_k)

            retrieval_results = []
            for doc_id, score, metadata in search_results:
                content = self._chunk_contents.get(doc_id, "")
                retrieval_results.append(
                    RetrievalResult(
                        doc_id=doc_id,
                        content=content,
                        score=score,
                        metadata=metadata,
                        source="rag_pipeline",
                    )
                )

        # 组装上下文
        context = ""
        if return_context and retrieval_results:
            context = self._context_window.fit_results(retrieval_results)

        # 计算置信度（基于最高分和结果数量）
        confidence = 0.0
        if retrieval_results:
            top_score = retrieval_results[0].score
            avg_score = sum(r.score for r in retrieval_results) / len(retrieval_results)
            confidence = (top_score + avg_score) / 2.0

        return RAGResult(
            answer="",  # 实际的答案生成由外部LLM完成
            sources=retrieval_results,
            context=context,
            confidence=confidence,
            metadata={
                "question": question,
                "top_k": top_k,
                "num_results": len(retrieval_results),
            },
        )

    def delete_document(self, doc_id: str) -> int:
        """删除文档及其所有块。

        Args:
            doc_id: 文档ID

        Returns:
            删除的块数量
        """
        chunks_to_delete = [
            chunk_id for chunk_id, chunk in self._chunks.items()
            if chunk.doc_id == doc_id
        ]

        for chunk_id in chunks_to_delete:
            self._vector_store.delete(chunk_id)
            del self._chunks[chunk_id]
            self._chunk_contents.pop(chunk_id, None)

        self._documents.pop(doc_id, None)

        self._stats["total_documents"] = max(0, self._stats["total_documents"] - 1)
        self._stats["total_chunks"] -= len(chunks_to_delete)

        return len(chunks_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return dict(self._stats)

    def list_documents(self) -> List[str]:
        """列出所有文档ID。"""
        return list(self._documents.keys())

    def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """获取文档的所有块。"""
        return [
            chunk for chunk in self._chunks.values()
            if chunk.doc_id == doc_id
        ]
