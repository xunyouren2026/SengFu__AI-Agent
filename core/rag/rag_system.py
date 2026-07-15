"""
RAG System
检索增强生成系统

提供文档索引、向量检索、知识库管理等功能
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """文档类型"""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


class ChunkStrategy(Enum):
    """分块策略"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class Document:
    """文档"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    title: str = ""
    source: str = ""
    doc_type: DocumentType = DocumentType.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "title": self.title,
            "source": self.source,
            "doc_type": self.doc_type.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Chunk:
    """文档块"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    embedding: Optional[List[float]] = None
    start_index: int = 0
    end_index: int = 0
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """检索结果"""
    chunk: Chunk
    score: float
    document: Optional[Document] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "document": self.document.to_dict() if self.document else None,
        }


@dataclass
class KnowledgeBase:
    """知识库"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    documents: Dict[str, Document] = field(default_factory=dict)
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "embedding_model": self.embedding_model,
            "chunk_strategy": self.chunk_strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class EmbeddingModel:
    """嵌入模型"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._model = None
        self._model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        
    async def initialize(self):
        """初始化嵌入模型"""
        if self._model:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info(f"Embedding model loaded: {self._model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """生成嵌入向量"""
        await self.initialize()
        
        if self._model:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()
        
        # 返回模拟嵌入
        return [[0.0] * 384 for _ in texts]
    
    async def embed_single(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        embeddings = await self.embed([text])
        return embeddings[0]


class VectorStore:
    """向量存储"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._index = None
        
    async def add(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """添加向量"""
        self._vectors[id] = vector
        self._metadata[id] = metadata or {}
    
    async def add_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """批量添加向量"""
        for i, id in enumerate(ids):
            self._vectors[id] = vectors[i]
            self._metadata[id] = metadatas[i] if metadatas else {}
    
    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        import numpy as np
        
        if not self._vectors:
            return []
        
        # 计算余弦相似度
        query = np.array(query_vector)
        
        scores = []
        for id, vector in self._vectors.items():
            # 应用过滤器
            if filter:
                metadata = self._metadata.get(id, {})
                match = all(metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            
            vec = np.array(vector)
            similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            scores.append((id, float(similarity)))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:k]
    
    async def delete(self, id: str):
        """删除向量"""
        if id in self._vectors:
            del self._vectors[id]
            del self._metadata[id]
    
    async def clear(self):
        """清空向量存储"""
        self._vectors.clear()
        self._metadata.clear()


class DocumentChunker:
    """文档分块器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._chunk_size = self.config.get("chunk_size", 512)
        self._chunk_overlap = self.config.get("chunk_overlap", 50)
        self._strategy = ChunkStrategy(self.config.get("strategy", "recursive"))
        
    async def chunk(self, document: Document) -> List[Chunk]:
        """分块文档"""
        if self._strategy == ChunkStrategy.FIXED_SIZE:
            return await self._fixed_size_chunk(document)
        elif self._strategy == ChunkStrategy.SENTENCE:
            return await self._sentence_chunk(document)
        elif self._strategy == ChunkStrategy.PARAGRAPH:
            return await self._paragraph_chunk(document)
        elif self._strategy == ChunkStrategy.RECURSIVE:
            return await self._recursive_chunk(document)
        else:
            return await self._fixed_size_chunk(document)
    
    async def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """固定大小分块"""
        chunks = []
        content = document.content
        
        for i in range(0, len(content), self._chunk_size - self._chunk_overlap):
            chunk_content = content[i:i + self._chunk_size]
            
            chunk = Chunk(
                document_id=document.id,
                content=chunk_content,
                start_index=i,
                end_index=min(i + self._chunk_size, len(content)),
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _sentence_chunk(self, document: Document) -> List[Chunk]:
        """句子分块"""
        import re
        
        # 简单句子分割
        sentences = re.split(r'[。！？.!?]', document.content)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            if len(current_chunk) + len(sentence) > self._chunk_size:
                if current_chunk:
                    chunk = Chunk(
                        document_id=document.id,
                        content=current_chunk.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        chunk_index=len(chunks),
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence
                current_start = document.content.find(sentence, current_start)
            else:
                current_chunk += sentence
        
        if current_chunk:
            chunk = Chunk(
                document_id=document.id,
                content=current_chunk.strip(),
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                chunk_index=len(chunks),
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _paragraph_chunk(self, document: Document) -> List[Chunk]:
        """段落分块"""
        paragraphs = document.content.split("\n\n")
        
        chunks = []
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunk = Chunk(
                    document_id=document.id,
                    content=para.strip(),
                    chunk_index=i,
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _recursive_chunk(self, document: Document) -> List[Chunk]:
        """递归分块"""
        # 先按段落分割，再按句子，最后按固定大小
        chunks = []
        
        paragraphs = document.content.split("\n\n")
        
        for para in paragraphs:
            if len(para) <= self._chunk_size:
                chunk = Chunk(
                    document_id=document.id,
                    content=para.strip(),
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)
            else:
                # 需要进一步分割
                sub_chunks = await self._sentence_chunk(Document(content=para))
                for sub in sub_chunks:
                    sub.chunk_index = len(chunks)
                    chunks.append(sub)
        
        return chunks


class RAGSystem:
    """
    检索增强生成系统
    
    功能：
    - 文档管理
    - 向量索引
    - 语义检索
    - 知识库管理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
        self._embedding_model = EmbeddingModel(config)
        self._vector_stores: Dict[str, VectorStore] = {}
        self._chunker = DocumentChunker(config)
        self._initialized = False
        
    async def initialize(self):
        """初始化RAG系统"""
        if self._initialized:
            return
        
        await self._embedding_model.initialize()
        
        self._initialized = True
        logger.info("RAG system initialized")
    
    async def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> KnowledgeBase:
        """创建知识库"""
        await self.initialize()
        
        kb = KnowledgeBase(
            name=name,
            description=description,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self._knowledge_bases[kb.id] = kb
        self._vector_stores[kb.id] = VectorStore()
        
        logger.info(f"Created knowledge base: {name} ({kb.id})")
        
        return kb
    
    async def add_document(
        self,
        kb_id: str,
        content: str,
        title: str = "",
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """添加文档"""
        await self.initialize()
        
        kb = self._knowledge_bases.get(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")
        
        # 创建文档
        document = Document(
            content=content,
            title=title,
            source=source,
            metadata=metadata or {},
        )
        
        kb.documents[document.id] = document
        
        # 分块
        chunks = await self._chunker.chunk(document)
        
        # 生成嵌入并存储
        vector_store = self._vector_stores[kb_id]
        
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.embedding = await self._embedding_model.embed_single(chunk.content)
            
            kb.chunks[chunk.id] = chunk
            
            await vector_store.add(
                chunk.id,
                chunk.embedding,
                {"document_id": document.id, "content": chunk.content},
            )
        
        kb.updated_at = time.time()
        
        logger.info(f"Added document: {title} ({document.id}) with {len(chunks)} chunks")
        
        return document
    
    async def search(
        self,
        kb_id: str,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """检索"""
        await self.initialize()
        
        kb = self._knowledge_bases.get(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")
        
        vector_store = self._vector_stores.get(kb_id)
        if not vector_store:
            return []
        
        # 生成查询向量
        query_vector = await self._embedding_model.embed_single(query)
        
        # 搜索
        results = await vector_store.search(query_vector, k, filter)
        
        # 构建结果
        search_results = []
        for chunk_id, score in results:
            chunk = kb.chunks.get(chunk_id)
            if chunk:
                document = kb.documents.get(chunk.document_id)
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    document=document,
                ))
        
        return search_results
    
    async def generate_with_rag(
        self,
        kb_id: str,
        query: str,
        k: int = 5,
        system_prompt: str = "",
    ) -> str:
        """使用RAG生成回答"""
        await self.initialize()
        
        # 检索相关内容
        results = await self.search(kb_id, query, k)
        
        # 构建上下文
        context = "\n\n".join([r.chunk.content for r in results])
        
        # 使用LLM生成
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return f"[Mock RAG Response] Based on {len(results)} retrieved documents: {query}"
            
            client = AsyncOpenAI(api_key=api_key)
            
            prompt = f"""{system_prompt or "Based on the following context, answer the question."}

Context:
{context}

Question: {query}

Answer:"""
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return f"Error: {str(e)}"
    
    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self._knowledge_bases.get(kb_id)
    
    async def list_knowledge_bases(self) -> List[KnowledgeBase]:
        """列出知识库"""
        return list(self._knowledge_bases.values())
    
    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        if kb_id in self._knowledge_bases:
            del self._knowledge_bases[kb_id]
            if kb_id in self._vector_stores:
                del self._vector_stores[kb_id]
            return True
        return False
    
    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """删除文档"""
        kb = self._knowledge_bases.get(kb_id)
        if not kb:
            return False
        
        if document_id in kb.documents:
            del kb.documents[document_id]
            
            # 删除相关块
            chunks_to_delete = [
                chunk_id for chunk_id, chunk in kb.chunks.items()
                if chunk.document_id == document_id
            ]
            
            for chunk_id in chunks_to_delete:
                del kb.chunks[chunk_id]
                if kb_id in self._vector_stores:
                    await self._vector_stores[kb_id].delete(chunk_id)
            
            return True
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_knowledge_bases": len(self._knowledge_bases),
            "total_documents": sum(len(kb.documents) for kb in self._knowledge_bases.values()),
            "total_chunks": sum(len(kb.chunks) for kb in self._knowledge_bases.values()),
        }


# 全局实例
_rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """获取全局RAG系统"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


async def init_rag_system(config: Optional[Dict[str, Any]] = None):
    """初始化全局RAG系统"""
    global _rag_system
    _rag_system = RAGSystem(config)
    await _rag_system.initialize()
    return _rag_system
