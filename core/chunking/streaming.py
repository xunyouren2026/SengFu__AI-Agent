"""
Streaming Chunker - 流式分块器
智能语义分块，支持动态窗口和边界检测
"""

import re
import time
import heapq
import logging
from typing import List, Dict, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import json

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """分块类型"""
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    TOPIC = "topic"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    HEADER = "header"


class BoundaryType(Enum):
    """边界类型"""
    HARD = "hard"      # 强边界（段落结束）
    SOFT = "soft"      # 软边界（句子结束）
    SEMANTIC = "semantic"  # 语义边界
    TOPIC = "topic"    # 话题边界


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    content: str
    chunk_type: ChunkType
    start_pos: int
    end_pos: int
    token_count: int
    importance_score: float = 0.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "token_count": self.token_count,
            "importance_score": self.importance_score,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "created_at": self.created_at
        }


@dataclass
class ChunkingConfig:
    """分块配置"""
    max_chunk_size: int = 512  # 最大分块Token数
    min_chunk_size: int = 50   # 最小分块Token数
    overlap_size: int = 50     # 重叠Token数
    target_chunk_size: int = 256  # 目标分块大小
    enable_semantic_boundary: bool = True
    enable_topic_detection: bool = True
    preserve_structure: bool = True
    max_chunks: int = 1000
    quality_threshold: float = 0.7


@dataclass
class Boundary:
    """边界信息"""
    position: int
    boundary_type: BoundaryType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticBoundaryDetector:
    """语义边界检测器"""
    
    def __init__(self, embedding_model: Optional[Any] = None):
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.7
        self.boundary_patterns = self._init_patterns()
        
    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """初始化边界检测模式"""
        return {
            "paragraph": re.compile(r'\n\s*\n'),
            "sentence": re.compile(r'[.!?。！？]\s+'),
            "header": re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            "list_item": re.compile(r'^\s*[-*•]\s+', re.MULTILINE),
            "numbered": re.compile(r'^\s*\d+[.)]\s+', re.MULTILINE),
            "code_block": re.compile(r'```[\s\S]*?```'),
            "table_row": re.compile(r'\|.*\|'),
        }
    
    def detect_boundaries(self, text: str) -> List[Boundary]:
        """检测文本边界"""
        boundaries = []
        
        # 检测段落边界
        for match in self.boundary_patterns["paragraph"].finditer(text):
            boundaries.append(Boundary(
                position=match.start(),
                boundary_type=BoundaryType.HARD,
                confidence=0.9,
                metadata={"type": "paragraph"}
            ))
        
        # 检测句子边界
        for match in self.boundary_patterns["sentence"].finditer(text):
            boundaries.append(Boundary(
                position=match.end(),
                boundary_type=BoundaryType.SOFT,
                confidence=0.7,
                metadata={"type": "sentence"}
            ))
        
        # 检测标题边界
        for match in self.boundary_patterns["header"].finditer(text):
            boundaries.append(Boundary(
                position=match.start(),
                boundary_type=BoundaryType.TOPIC,
                confidence=0.95,
                metadata={"type": "header"}
            ))
        
        # 排序并去重
        boundaries.sort(key=lambda b: b.position)
        return self._deduplicate_boundaries(boundaries)
    
    def _deduplicate_boundaries(self, boundaries: List[Boundary]) -> List[Boundary]:
        """去除重复边界"""
        if not boundaries:
            return []
        
        unique = [boundaries[0]]
        for b in boundaries[1:]:
            if abs(b.position - unique[-1].position) > 10:
                unique.append(b)
            elif b.confidence > unique[-1].confidence:
                unique[-1] = b
        
        return unique
    
    def detect_semantic_shift(self, text1: str, text2: str) -> float:
        """检测语义偏移程度"""
        if not self.embedding_model:
            # 使用简单的词汇重叠度
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 1.0
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return 1.0 - (overlap / union if union > 0 else 0)
        
        # 使用嵌入模型计算语义相似度
        try:
            emb1 = self.embedding_model.encode(text1)
            emb2 = self.embedding_model.encode(text2)
            similarity = self._cosine_similarity(emb1, emb2)
            return 1.0 - similarity
        except Exception as e:
            logger.warning(f"语义检测失败: {e}")
            return 0.5
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        import math
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class TokenCounter:
    """Token计数器"""
    
    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer
        self._cache: Dict[str, int] = {}
        self._cache_size = 10000
    
    def count(self, text: str) -> int:
        """计算文本Token数"""
        if text in self._cache:
            return self._cache[text]
        
        if self.tokenizer:
            try:
                count = len(self.tokenizer.encode(text))
            except Exception:
                count = self._estimate_tokens(text)
        else:
            count = self._estimate_tokens(text)
        
        # 缓存管理
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = count
        
        return count
    
    def _estimate_tokens(self, text: str) -> int:
        """估算Token数（简单方法）"""
        # 英文约4字符=1token，中文约1.5字符=1token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4) + 1


class StreamingChunker:
    """流式分块器主类"""
    
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedding_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config or ChunkingConfig()
        self.boundary_detector = SemanticBoundaryDetector(embedding_model)
        self.token_counter = TokenCounter(tokenizer)
        
        self._chunk_id_counter = 0
        self._chunks: Dict[str, Chunk] = {}
        self._chunk_index: List[Tuple[int, int, str]] = []  # (start, end, chunk_id)
        self._lock = threading.Lock()
        
        # 流式处理缓冲区
        self._buffer = deque(maxlen=10000)
        self._pending_text = ""
        self._position = 0
        
    def _generate_chunk_id(self) -> str:
        """生成分块ID"""
        self._chunk_id_counter += 1
        return f"chunk_{self._chunk_id_counter}_{int(time.time() * 1000)}"
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """对文本进行分块"""
        if not text.strip():
            return []
        
        # 检测边界
        boundaries = self.boundary_detector.detect_boundaries(text)
        
        # 添加语义边界
        if self.config.enable_semantic_boundary:
            boundaries.extend(self._detect_semantic_boundaries(text, boundaries))
            boundaries.sort(key=lambda b: b.position)
            boundaries = self.boundary_detector._deduplicate_boundaries(boundaries)
        
        # 根据边界分块
        chunks = self._create_chunks_from_boundaries(text, boundaries)
        
        # 后处理：合并小分块、拆分大分块
        chunks = self._post_process_chunks(chunks)
        
        # 存储分块
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
                self._chunk_index.append((chunk.start_pos, chunk.end_pos, chunk.id))
        
        return chunks
    
    def _detect_semantic_boundaries(
        self, 
        text: str, 
        existing_boundaries: List[Boundary]
    ) -> List[Boundary]:
        """检测语义边界"""
        semantic_boundaries = []
        
        # 获取现有边界位置
        existing_positions = {b.position for b in existing_boundaries}
        
        # 滑动窗口检测语义变化
        window_size = 200
        stride = 100
        
        for i in range(0, len(text) - window_size, stride):
            window1 = text[i:i + window_size]
            window2 = text[i + stride:i + stride + window_size]
            
            if len(window2) < 50:
                continue
            
            shift = self.boundary_detector.detect_semantic_shift(window1, window2)
            
            if shift > 0.5:  # 显著语义变化
                # 找到最近的句子边界
                pos = self._find_nearest_sentence_boundary(text, i + stride)
                if pos not in existing_positions:
                    semantic_boundaries.append(Boundary(
                        position=pos,
                        boundary_type=BoundaryType.SEMANTIC,
                        confidence=shift,
                        metadata={"shift_score": shift}
                    ))
        
        return semantic_boundaries
    
    def _find_nearest_sentence_boundary(self, text: str, pos: int) -> int:
        """找到最近的句子边界"""
        # 向后搜索
        for i in range(pos, min(pos + 100, len(text))):
            if text[i] in '.!?。！？':
                return i + 1
        
        # 向前搜索
        for i in range(pos, max(0, pos - 100), -1):
            if text[i] in '.!?。！？':
                return i + 1
        
        return pos
    
    def _create_chunks_from_boundaries(
        self, 
        text: str, 
        boundaries: List[Boundary]
    ) -> List[Chunk]:
        """根据边界创建分块"""
        chunks = []
        
        if not boundaries:
            # 没有边界，创建单个分块
            chunk = self._create_chunk(text, 0, len(text), ChunkType.SEMANTIC)
            if chunk:
                chunks.append(chunk)
            return chunks
        
        # 添加起始和结束边界
        all_positions = [0] + [b.position for b in boundaries] + [len(text)]
        
        for i in range(len(all_positions) - 1):
            start = all_positions[i]
            end = all_positions[i + 1]
            
            if end <= start:
                continue
            
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            
            # 确定分块类型
            chunk_type = self._determine_chunk_type(chunk_text, boundaries, i)
            
            chunk = self._create_chunk(chunk_text, start, end, chunk_type)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _determine_chunk_type(
        self, 
        text: str, 
        boundaries: List[Boundary], 
        index: int
    ) -> ChunkType:
        """确定分块类型"""
        # 检测代码块
        if text.startswith('```') or 'def ' in text or 'class ' in text:
            return ChunkType.CODE
        
        # 检测列表
        if re.match(r'^\s*[-*•]\s+', text, re.MULTILINE):
            return ChunkType.LIST
        
        # 检测表格
        if '|' in text and text.count('|') > 2:
            return ChunkType.TABLE
        
        # 检测标题
        if re.match(r'^#{1,6}\s+', text):
            return ChunkType.HEADER
        
        # 根据边界类型
        if index < len(boundaries):
            boundary = boundaries[index]
            if boundary.boundary_type == BoundaryType.TOPIC:
                return ChunkType.TOPIC
            elif boundary.boundary_type == BoundaryType.SEMANTIC:
                return ChunkType.SEMANTIC
        
        return ChunkType.PARAGRAPH
    
    def _create_chunk(
        self, 
        text: str, 
        start: int, 
        end: int, 
        chunk_type: ChunkType
    ) -> Optional[Chunk]:
        """创建单个分块"""
        token_count = self.token_counter.count(text)
        
        if token_count < self.config.min_chunk_size:
            return None
        
        return Chunk(
            id=self._generate_chunk_id(),
            content=text,
            chunk_type=chunk_type,
            start_pos=start,
            end_pos=end,
            token_count=token_count,
            importance_score=self._calculate_importance(text)
        )
    
    def _calculate_importance(self, text: str) -> float:
        """计算分块重要性"""
        score = 0.5  # 基础分
        
        # 长度因素
        token_count = self.token_counter.count(text)
        if token_count > 200:
            score += 0.1
        elif token_count < 50:
            score -= 0.1
        
        # 关键词因素
        important_keywords = ['important', 'key', 'critical', '主要', '关键', '重要']
        for kw in important_keywords:
            if kw.lower() in text.lower():
                score += 0.1
        
        # 问题因素
        if '?' in text or '？' in text:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """后处理分块"""
        if not chunks:
            return chunks
        
        processed = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            # 处理过大的分块
            if chunk.token_count > self.config.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                processed.extend(sub_chunks)
                i += 1
                continue
            
            # 尝试合并过小的分块
            if chunk.token_count < self.config.min_chunk_size and i + 1 < len(chunks):
                merged = self._merge_chunks(chunk, chunks[i + 1])
                if merged:
                    processed.append(merged)
                    i += 2
                    continue
            
            processed.append(chunk)
            i += 1
        
        return processed
    
    def _split_large_chunk(self, chunk: Chunk) -> List[Chunk]:
        """拆分过大的分块"""
        text = chunk.content
        target_size = self.config.target_chunk_size
        
        # 按句子拆分
        sentences = re.split(r'([.!?。！？]\s+)', text)
        
        sub_chunks = []
        current_text = ""
        current_start = chunk.start_pos
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加分隔符
            
            if self.token_counter.count(current_text + sentence) > target_size:
                if current_text:
                    sub_chunk = Chunk(
                        id=self._generate_chunk_id(),
                        content=current_text.strip(),
                        chunk_type=chunk.chunk_type,
                        start_pos=current_start,
                        end_pos=current_start + len(current_text),
                        token_count=self.token_counter.count(current_text),
                        parent_id=chunk.id
                    )
                    sub_chunks.append(sub_chunk)
                current_text = sentence
                current_start = chunk.start_pos + text.find(sentence, len(current_text) if current_text else 0)
            else:
                current_text += sentence
        
        # 添加最后一个分块
        if current_text.strip():
            sub_chunk = Chunk(
                id=self._generate_chunk_id(),
                content=current_text.strip(),
                chunk_type=chunk.chunk_type,
                start_pos=current_start,
                end_pos=chunk.end_pos,
                token_count=self.token_counter.count(current_text),
                parent_id=chunk.id
            )
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _merge_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Optional[Chunk]:
        """合并两个分块"""
        merged_text = chunk1.content + "\n" + chunk2.content
        merged_tokens = self.token_counter.count(merged_text)
        
        if merged_tokens > self.config.max_chunk_size:
            return None
        
        return Chunk(
            id=self._generate_chunk_id(),
            content=merged_text,
            chunk_type=chunk1.chunk_type,
            start_pos=chunk1.start_pos,
            end_pos=chunk2.end_pos,
            token_count=merged_tokens,
            importance_score=max(chunk1.importance_score, chunk2.importance_score),
            child_ids=[chunk1.id, chunk2.id]
        )
    
    def stream_chunk(self, text: str) -> List[Chunk]:
        """流式分块处理"""
        self._pending_text += text
        chunks = []
        
        # 检测完整段落
        while '\n\n' in self._pending_text:
            idx = self._pending_text.index('\n\n')
            paragraph = self._pending_text[:idx].strip()
            self._pending_text = self._pending_text[idx + 2:]
            
            if paragraph:
                chunk = self._create_chunk(
                    paragraph, 
                    self._position, 
                    self._position + len(paragraph),
                    ChunkType.PARAGRAPH
                )
                if chunk:
                    chunks.append(chunk)
                    self._position += len(paragraph) + 2
                    with self._lock:
                        self._chunks[chunk.id] = chunk
        
        return chunks
    
    def flush(self) -> List[Chunk]:
        """刷新缓冲区，返回剩余分块"""
        chunks = []
        if self._pending_text.strip():
            chunk = self._create_chunk(
                self._pending_text.strip(),
                self._position,
                self._position + len(self._pending_text),
                ChunkType.PARAGRAPH
            )
            if chunk:
                chunks.append(chunk)
                with self._lock:
                    self._chunks[chunk.id] = chunk
        
        self._pending_text = ""
        return chunks
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """获取指定分块"""
        return self._chunks.get(chunk_id)
    
    def get_chunks_by_range(self, start: int, end: int) -> List[Chunk]:
        """获取指定范围的所有分块"""
        result = []
        with self._lock:
            for chunk_start, chunk_end, chunk_id in self._chunk_index:
                if chunk_start < end and chunk_end > start:
                    chunk = self._chunks.get(chunk_id)
                    if chunk:
                        result.append(chunk)
        return result
    
    def get_overlapping_chunks(self, position: int, window: int = 100) -> List[Chunk]:
        """获取位置附近的分块"""
        return self.get_chunks_by_range(position - window, position + window)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取分块统计信息"""
        if not self._chunks:
            return {"total_chunks": 0}
        
        chunks = list(self._chunks.values())
        total_tokens = sum(c.token_count for c in chunks)
        
        type_counts = {}
        for chunk in chunks:
            t = chunk.chunk_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_chunk_size": total_tokens / len(chunks),
            "min_chunk_size": min(c.token_count for c in chunks),
            "max_chunk_size": max(c.token_count for c in chunks),
            "type_distribution": type_counts,
            "avg_importance": sum(c.importance_score for c in chunks) / len(chunks)
        }
    
    def clear(self):
        """清空所有分块"""
        with self._lock:
            self._chunks.clear()
            self._chunk_index.clear()
            self._chunk_id_counter = 0
            self._position = 0
            self._pending_text = ""


class HierarchicalChunker(StreamingChunker):
    """分层分块器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy_levels = {
            "document": {"min_size": 2000, "max_size": 10000},
            "section": {"min_size": 500, "max_size": 2000},
            "paragraph": {"min_size": 100, "max_size": 500},
            "sentence": {"min_size": 20, "max_size": 100}
        }
    
    def chunk_hierarchical(self, text: str) -> Dict[str, List[Chunk]]:
        """分层分块"""
        result = {}
        
        # 文档级
        result["document"] = [Chunk(
            id=self._generate_chunk_id(),
            content=text,
            chunk_type=ChunkType.SEMANTIC,
            start_pos=0,
            end_pos=len(text),
            token_count=self.token_counter.count(text)
        )]
        
        # 章节级
        sections = self._split_sections(text)
        result["section"] = sections
        
        # 段落级
        paragraphs = []
        for section in sections:
            paragraphs.extend(self._split_paragraphs(section))
        result["paragraph"] = paragraphs
        
        # 句子级
        sentences = []
        for para in paragraphs:
            sentences.extend(self._split_sentences(para))
        result["sentence"] = sentences
        
        return result
    
    def _split_sections(self, text: str) -> List[Chunk]:
        """拆分章节"""
        # 按标题拆分
        header_pattern = re.compile(r'^#{1,3}\s+.+$', re.MULTILINE)
        matches = list(header_pattern.finditer(text))
        
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            
            if content:
                sections.append(Chunk(
                    id=self._generate_chunk_id(),
                    content=content,
                    chunk_type=ChunkType.HEADER,
                    start_pos=start,
                    end_pos=end,
                    token_count=self.token_counter.count(content)
                ))
        
        return sections
    
    def _split_paragraphs(self, parent_chunk: Chunk) -> List[Chunk]:
        """拆分段落"""
        paragraphs = re.split(r'\n\s*\n', parent_chunk.content)
        chunks = []
        pos = parent_chunk.start_pos
        
        for para in paragraphs:
            if para.strip():
                chunks.append(Chunk(
                    id=self._generate_chunk_id(),
                    content=para.strip(),
                    chunk_type=ChunkType.PARAGRAPH,
                    start_pos=pos,
                    end_pos=pos + len(para),
                    token_count=self.token_counter.count(para),
                    parent_id=parent_chunk.id
                ))
            pos += len(para) + 2
        
        return chunks
    
    def _split_sentences(self, parent_chunk: Chunk) -> List[Chunk]:
        """拆分句子"""
        sentences = re.split(r'([.!?。！？]\s+)', parent_chunk.content)
        chunks = []
        pos = parent_chunk.start_pos
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if sentence.strip():
                chunks.append(Chunk(
                    id=self._generate_chunk_id(),
                    content=sentence.strip(),
                    chunk_type=ChunkType.SENTENCE,
                    start_pos=pos,
                    end_pos=pos + len(sentence),
                    token_count=self.token_counter.count(sentence),
                    parent_id=parent_chunk.id
                ))
            pos += len(sentence)
        
        return chunks


# 工厂函数
def create_chunker(
    config: Optional[ChunkingConfig] = None,
    embedding_model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    hierarchical: bool = False
) -> StreamingChunker:
    """创建分块器"""
    if hierarchical:
        return HierarchicalChunker(config, embedding_model, tokenizer)
    return StreamingChunker(config, embedding_model, tokenizer)


# 便捷函数
def chunk_text(
    text: str,
    max_chunk_size: int = 512,
    min_chunk_size: int = 50,
    enable_semantic: bool = True
) -> List[Chunk]:
    """便捷分块函数"""
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        enable_semantic_boundary=enable_semantic
    )
    chunker = StreamingChunker(config)
    return chunker.chunk_text(text)
