"""
RAG (Retrieval-Augmented Generation) 系统

提供向量存储、文本嵌入、检索器和RAG管道等组件。
仅使用Python标准库实现。
"""

from rag.vector_store import (
    VectorStore,
    NaiveVectorStore,
    HNSWIndex,
    VectorDocument,
)
from rag.embedder import (
    TextEmbedder,
    TFIDFEmbedder,
    BagOfWordsEmbedder,
    HashEmbedder,
    SentencePieceTokenizer,
)
from rag.retriever import (
    Retriever,
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    HyDERetriever,
    MultiQueryRetriever,
    RetrievalResult,
    ContextWindow,
)
from rag.pipeline import (
    RAGPipeline,
    RAGResult,
    Chunk,
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    SlidingWindowChunker,
    DocumentProcessor,
)

__all__ = [
    # 向量存储
    "VectorStore",
    "NaiveVectorStore",
    "HNSWIndex",
    "VectorDocument",
    # 嵌入器
    "TextEmbedder",
    "TFIDFEmbedder",
    "BagOfWordsEmbedder",
    "HashEmbedder",
    "SentencePieceTokenizer",
    # 检索器
    "Retriever",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "HyDERetriever",
    "MultiQueryRetriever",
    "RetrievalResult",
    "ContextWindow",
    # 管道
    "RAGPipeline",
    "RAGResult",
    "Chunk",
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
    "DocumentProcessor",
]
