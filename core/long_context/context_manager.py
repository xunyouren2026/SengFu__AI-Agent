"""
长上下文管理器 - 借鉴视频生成技术
====================================

本模块实现了统一的长上下文管理，整合多种技术。

重构说明：
- 内部使用core/unified_algorithms/统一核心
- 通过unified_adapter.py适配器保持API兼容
- 原有API完全保持不变

核心组件：
1. ContextChunker: 上下文分块器（基于UnifiedChunker）
2. ProgressiveLoader: 渐进式加载器（基于UnifiedProgressiveLoader）
3. BoundaryDetector: 边界检测器（基于UnifiedBoundaryDetector）
4. ContextFusion: 上下文融合

纯Python实现，仅使用标准库。
"""

from __future__ import annotations

import math
import random
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
from enum import Enum, auto

# 导入统一核心适配器
from .unified_adapter import (
    ContextChunkerAdapter,
    ProgressiveLoaderAdapter,
    BoundaryDetectorAdapter,
    ContextSegment,
    cosine_similarity,
    normalize_vector,
    compute_hash,
)

# 导入统一核心
from ..unified_algorithms.unified_chunking import (
    UnifiedChunker,
    UnifiedBoundaryDetector,
    UnifiedProgressiveLoader,
    UnifiedOverlapFusion,
    Chunk,
    ChunkingResult,
    Boundary,
)
from ..unified_algorithms.unified_config import (
    UnifiedAlgorithmConfig,
    ChunkingStrategy,
    BoundaryType,
)


# ============================================================================
# 工具函数（保持向后兼容）
# ============================================================================

# cosine_similarity, normalize_vector, compute_hash 现在从unified_adapter导入

def split_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    # 简化的句子分割
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# 数据类定义
# ============================================================================

class BoundaryType(Enum):
    """边界类型"""
    SENTENCE = auto()    # 句子边界
    PARAGRAPH = auto()   # 段落边界
    TOPIC = auto()       # 主题边界
    TEMPORAL = auto()    # 时间边界
    MANUAL = auto()      # 手动标记边界


# ContextSegment现在从unified_adapter导入


@dataclass
class ChunkConfig:
    """分块配置"""
    max_chunk_size: int = 512
    overlap_size: int = 64
    respect_boundaries: bool = True
    boundary_types: List[BoundaryType] = field(default_factory=lambda: [
        BoundaryType.PARAGRAPH, BoundaryType.SENTENCE
    ])


@dataclass
class ProgressiveLoadState:
    """渐进式加载状态"""
    stage: int = 0
    total_stages: int = 1
    loaded_segments: List[str] = field(default_factory=list)
    pending_segments: List[str] = field(default_factory=list)
    is_complete: bool = False


# ============================================================================
# 1. ContextChunker - 上下文分块器
# ============================================================================

class ContextChunker:
    """
    上下文分块器

    将长上下文智能分割成可管理的块：
    - 尊重语义边界（句子、段落）
    - 支持重叠保持连贯性
    - 自适应块大小

    借鉴视频分块思想：
    - 关键帧优先（重要段落）
    - 平滑过渡（重叠区域）

    重构说明：
    - 内部使用ContextChunkerAdapter包装UnifiedChunker
    - 保持原有API完全不变
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # 内部使用适配器
        self._adapter = ContextChunkerAdapter(
            max_chunk_size=self.config.max_chunk_size,
            overlap_size=self.config.overlap_size,
            respect_boundaries=self.config.respect_boundaries
        )

        # 历史记录（委托给适配器）
        self.chunk_history: List[Dict] = self._adapter.chunk_history
    
    def chunk(
        self,
        content: str,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ) -> List[ContextSegment]:
        """
        分块处理

        Args:
            content: 长文本内容
            embedding_fn: 可选的嵌入函数

        Returns:
            分块后的片段列表
        """
        # 委托给适配器
        return self._adapter.chunk(content, embedding_fn)
    
    def _detect_boundaries(self, content: str) -> List[Tuple[int, BoundaryType, float]]:
        """
        检测内容边界
        
        Returns:
            (位置, 边界类型, 边界分数) 列表
        """
        boundaries = []
        
        # 段落边界（空行）
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, content):
            pos = match.start()
            boundaries.append((pos, BoundaryType.PARAGRAPH, 0.9))
        
        # 句子边界（句号、问号、感叹号后跟空格或大写）
        sentence_pattern = r'[.!?。！？]\s+(?=[A-Z\u4e00-\u9fff])'
        for match in re.finditer(sentence_pattern, content):
            pos = match.end()
            boundaries.append((pos, BoundaryType.SENTENCE, 0.7))
        
        # 主题边界（关键词检测）
        topic_indicators = [
            '首先', '其次', '最后', '总之', '综上所述',
            'first', 'second', 'finally', 'in conclusion', 'summary'
        ]
        for indicator in topic_indicators:
            for match in re.finditer(r'\b' + indicator + r'\b', content, re.IGNORECASE):
                pos = match.start()
                boundaries.append((pos, BoundaryType.TOPIC, 0.8))
        
        # 按位置排序
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def _create_segments(
        self,
        content: str,
        boundaries: List[Tuple[int, BoundaryType, float]]
    ) -> List[ContextSegment]:
        """基于边界创建片段"""
        if not boundaries:
            # 没有边界，按固定大小分割
            return self._fixed_size_chunk(content)
        
        segments = []
        start_pos = 0
        current_chunk = ""
        
        for pos, boundary_type, score in boundaries:
            # 添加内容到当前块
            chunk_part = content[start_pos:pos]
            
            # 检查是否超出最大大小
            if len(current_chunk) + len(chunk_part) > self.config.max_chunk_size:
                # 保存当前块
                if current_chunk:
                    segment = ContextSegment(
                        segment_id=f"seg_{compute_hash(current_chunk)}_{len(segments)}",
                        content=current_chunk.strip(),
                        start_pos=start_pos - len(current_chunk),
                        end_pos=start_pos,
                        boundary_type=boundary_type,
                        boundary_score=score
                    )
                    segments.append(segment)
                current_chunk = chunk_part
            else:
                current_chunk += chunk_part
            
            start_pos = pos
        
        # 处理剩余内容
        remaining = content[start_pos:]
        if remaining:
            if len(current_chunk) + len(remaining) <= self.config.max_chunk_size:
                current_chunk += remaining
            else:
                if current_chunk:
                    segment = ContextSegment(
                        segment_id=f"seg_{compute_hash(current_chunk)}_{len(segments)}",
                        content=current_chunk.strip(),
                        start_pos=start_pos - len(current_chunk),
                        end_pos=start_pos,
                        boundary_type=BoundaryType.SENTENCE,
                        boundary_score=0.5
                    )
                    segments.append(segment)
                current_chunk = remaining
        
        # 添加最后一个块
        if current_chunk:
            segment = ContextSegment(
                segment_id=f"seg_{compute_hash(current_chunk)}_{len(segments)}",
                content=current_chunk.strip(),
                start_pos=start_pos,
                end_pos=len(content),
                boundary_type=BoundaryType.SENTENCE,
                boundary_score=0.5
            )
            segments.append(segment)
        
        return segments
    
    def _fixed_size_chunk(self, content: str) -> List[ContextSegment]:
        """固定大小分块（备用方案）"""
        segments = []
        step = self.config.max_chunk_size - self.config.overlap_size
        
        for i in range(0, len(content), step):
            chunk = content[i:i + self.config.max_chunk_size]
            segment = ContextSegment(
                segment_id=f"seg_{compute_hash(chunk)}_{i}",
                content=chunk,
                start_pos=i,
                end_pos=min(i + self.config.max_chunk_size, len(content)),
                boundary_type=BoundaryType.SENTENCE,
                boundary_score=0.5
            )
            segments.append(segment)
        
        return segments
    
    def _apply_overlap(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """应用重叠区域"""
        if len(segments) <= 1 or self.config.overlap_size <= 0:
            return segments
        
        for i in range(len(segments)):
            # 与前一片段重叠
            if i > 0:
                prev_segment = segments[i - 1]
                overlap_text = prev_segment.content[-self.config.overlap_size:]
                segments[i].overlap_prev = overlap_text
                segments[i].prev_segment_id = prev_segment.segment_id
            
            # 与后一片段重叠
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                overlap_text = next_segment.content[:self.config.overlap_size]
                segments[i].overlap_next = overlap_text
                segments[i].next_segment_id = next_segment.segment_id
        
        return segments
    
    def merge_chunks(
        self,
        segments: List[ContextSegment],
        merge_threshold: float = 0.8
    ) -> List[ContextSegment]:
        """
        合并相似块
        
        Args:
            segments: 片段列表
            merge_threshold: 合并相似度阈值
            
        Returns:
            合并后的片段列表
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_segment in segments[1:]:
            similarity = current.similarity_to(next_segment)
            
            if similarity > merge_threshold and \
               len(current.content) + len(next_segment.content) <= self.config.max_chunk_size:
                # 合并
                current.content += " " + next_segment.content
                current.end_pos = next_segment.end_pos
                current.embedding = normalize_vector([
                    (current.embedding[i] + next_segment.embedding[i]) / 2
                    for i in range(len(current.embedding))
                ]) if current.embedding and next_segment.embedding else []
                current.boundary_score = max(current.boundary_score, next_segment.boundary_score)
            else:
                merged.append(current)
                current = next_segment
        
        merged.append(current)
        return merged


# ============================================================================
# 2. OverlapFusion - 重叠融合器
# ============================================================================

class OverlapFusion:
    """
    重叠融合器
    
    处理相邻块之间的重叠区域：
    - 加权融合：根据位置加权
    - 平滑过渡：消除边界感
    - 去重：避免内容重复
    
    借鉴视频融合技术：
    - 交叉淡化（cross-fade）
    - 时间对齐
    """
    
    def __init__(self, fusion_method: str = 'weighted'):
        self.fusion_method = fusion_method
        self.fusion_history: List[Dict] = []
    
    def fuse(
        self,
        segments: List[ContextSegment],
        query_embedding: Optional[List[float]] = None
    ) -> str:
        """
        融合片段
        
        Args:
            segments: 片段列表
            query_embedding: 可选的查询嵌入（用于相关性加权）
            
        Returns:
            融合后的文本
        """
        if not segments:
            return ""
        
        if len(segments) == 1:
            return segments[0].get_full_content()
        
        if self.fusion_method == 'concat':
            return self._concat_fuse(segments)
        elif self.fusion_method == 'weighted':
            return self._weighted_fuse(segments, query_embedding)
        elif self.fusion_method == 'smart':
            return self._smart_fuse(segments, query_embedding)
        else:
            return self._concat_fuse(segments)
    
    def _concat_fuse(self, segments: List[ContextSegment]) -> str:
        """简单拼接（去重）"""
        result = segments[0].content
        
        for i in range(1, len(segments)):
            current = segments[i]
            
            # 检查与前一片段的重叠
            if current.overlap_prev and result.endswith(current.overlap_prev):
                # 去掉重复部分
                result = result[:-len(current.overlap_prev)]
            
            result += current.content
        
        return result
    
    def _weighted_fuse(
        self,
        segments: List[ContextSegment],
        query_embedding: Optional[List[float]]
    ) -> str:
        """加权融合"""
        # 计算每个片段的权重
        weights = []
        for segment in segments:
            if query_embedding and segment.embedding:
                relevance = cosine_similarity(query_embedding, segment.embedding)
                weight = 0.5 + 0.5 * relevance  # 基础权重0.5 + 相关性0.5
            else:
                weight = 1.0
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # 根据权重选择或加权内容
        # 简化处理：按权重排序后拼接
        weighted_segments = sorted(
            zip(segments, weights),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = ""
        seen_content = set()
        
        for segment, weight in weighted_segments:
            content = segment.content
            # 简单去重
            content_hash = compute_hash(content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                result += content + " "
        
        return result.strip()
    
    def _smart_fuse(
        self,
        segments: List[ContextSegment],
        query_embedding: Optional[List[float]]
    ) -> str:
        """智能融合（考虑语义连贯性）"""
        if not segments:
            return ""
        
        # 按原始顺序排列
        ordered_segments = sorted(segments, key=lambda s: s.start_pos)
        
        result_parts = []
        
        for i, segment in enumerate(ordered_segments):
            content = segment.content
            
            # 处理重叠区域
            if i > 0 and segment.overlap_prev:
                # 与前一片段重叠，使用更自然的过渡
                prev_content = ordered_segments[i - 1].content
                overlap = segment.overlap_prev
                
                # 找到重叠部分在前一片段中的位置
                if overlap in prev_content:
                    # 使用平滑过渡：取前一片段的后半部分 + 当前片段
                    transition_point = prev_content.rfind(overlap) + len(overlap) // 2
                    if transition_point > len(prev_content) * 0.7:
                        # 重叠较大，跳过当前片段的开头
                        content = content[len(overlap) // 2:]
            
            result_parts.append(content)
        
        return " ".join(result_parts)
    
    def fuse_embeddings(
        self,
        segments: List[ContextSegment],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        融合嵌入向量
        
        Args:
            segments: 片段列表
            weights: 可选的权重列表
            
        Returns:
            融合后的嵌入向量
        """
        if not segments:
            return []
        
        embeddings = [s.embedding for s in segments if s.embedding]
        if not embeddings:
            return []
        
        if weights is None:
            weights = [1.0] * len(embeddings)
        
        # 加权平均
        dim = len(embeddings[0])
        fused = [0.0] * dim
        total_weight = sum(weights)
        
        for emb, weight in zip(embeddings, weights):
            for i in range(dim):
                fused[i] += emb[i] * weight / total_weight
        
        return normalize_vector(fused)


# ============================================================================
# 3. BoundaryDetector - 边界检测器
# ============================================================================

class BoundaryDetector:
    """
    边界检测器

    检测上下文中的语义边界：
    - 主题切换
    - 时间跳跃
    - 情感变化

    借鉴视频镜头边界检测：
    - 帧间差异
    - 场景变化检测

    重构说明：
    - 内部使用BoundaryDetectorAdapter包装UnifiedBoundaryDetector
    - 保持原有API完全不变
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        window_size: int = 3
    ):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

        # 内部使用适配器
        self._adapter = BoundaryDetectorAdapter(
            similarity_threshold=similarity_threshold,
            window_size=window_size
        )

        # 历史记录（委托给适配器）
        self.detection_history: List[Dict] = self._adapter.detection_history
    
    def detect_boundaries(
        self,
        segments: List[ContextSegment]
    ) -> List[Tuple[int, BoundaryType, float]]:
        """
        检测片段间的边界

        Args:
            segments: 片段列表

        Returns:
            (片段索引, 边界类型, 边界强度) 列表
        """
        # 委托给适配器
        return self._adapter.detect_boundaries(segments)
    
    def _compute_semantic_difference(
        self,
        seg1: ContextSegment,
        seg2: ContextSegment
    ) -> float:
        """计算语义差异"""
        differences = []
        
        # 1. 嵌入相似度差异
        if seg1.embedding and seg2.embedding:
            similarity = cosine_similarity(seg1.embedding, seg2.embedding)
            differences.append(1 - similarity)
        
        # 2. 内容长度差异
        len_diff = abs(len(seg1.content) - len(seg2.content)) / max(
            len(seg1.content), len(seg2.content), 1
        )
        differences.append(len_diff * 0.3)
        
        # 3. 时间间隔（如果有时间戳）
        time_diff = abs(seg1.timestamp - seg2.timestamp)
        if time_diff > 60:  # 超过1分钟
            differences.append(min(time_diff / 3600, 1.0))  # 归一化到1小时
        
        return sum(differences) / max(len(differences), 1)
    
    def _classify_boundary(
        self,
        seg1: ContextSegment,
        seg2: ContextSegment,
        semantic_diff: float
    ) -> Tuple[BoundaryType, float]:
        """分类边界类型"""
        # 检查关键词指示主题变化
        topic_indicators = [
            '但是', '然而', '不过', '另一方面',
            'but', 'however', 'on the other hand', 'meanwhile'
        ]
        
        has_topic_indicator = any(
            indicator in seg2.content[:50].lower()
            for indicator in topic_indicators
        )
        
        if has_topic_indicator and semantic_diff > 0.6:
            return BoundaryType.TOPIC, semantic_diff
        
        # 检查时间指示
        time_indicators = [
            '后来', '之后', '然后', '接下来',
            'later', 'after', 'then', 'next'
        ]
        
        has_time_indicator = any(
            indicator in seg2.content[:50].lower()
            for indicator in time_indicators
        )
        
        if has_time_indicator:
            return BoundaryType.TEMPORAL, semantic_diff * 0.8
        
        # 默认句子边界
        return BoundaryType.SENTENCE, semantic_diff
    
    def split_at_boundaries(
        self,
        segments: List[ContextSegment]
    ) -> List[List[ContextSegment]]:
        """
        在边界处分割片段
        
        Returns:
            分割后的片段组
        """
        boundaries = self.detect_boundaries(segments)
        boundary_indices = set(b[0] for b in boundaries)
        
        groups = []
        current_group = []
        
        for i, segment in enumerate(segments):
            current_group.append(segment)
            
            if i in boundary_indices and current_group:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups


# ============================================================================
# 4. ProgressiveLoader - 渐进式加载器
# ============================================================================

class ProgressiveLoader:
    """
    渐进式加载器

    借鉴视频渐进训练思想：
    - 从粗粒度到细粒度逐步加载
    - 优先加载重要内容
    - 支持中断和恢复

    加载阶段：
    1. 摘要加载（全局概览）
    2. 关键帧加载（重要内容）
    3. 完整加载（所有内容）

    重构说明：
    - 内部使用ProgressiveLoaderAdapter包装UnifiedProgressiveLoader
    - 保持原有API完全不变
    """

    def __init__(
        self,
        num_stages: int = 3,
        keyframe_ratio: float = 0.2
    ):
        self.num_stages = num_stages
        self.keyframe_ratio = keyframe_ratio

        # 内部使用适配器
        self._adapter = ProgressiveLoaderAdapter(
            num_stages=num_stages,
            keyframe_ratio=keyframe_ratio
        )

        # 历史记录（委托给适配器）
        self.load_history: List[Dict] = self._adapter.load_history
    
    def create_loading_plan(
        self,
        segments: List[ContextSegment]
    ) -> List[List[str]]:
        """
        创建加载计划

        Args:
            segments: 所有片段

        Returns:
            每阶段加载的片段ID列表
        """
        return self._adapter.create_loading_plan(segments)

    def load_progressive(
        self,
        segments: List[ContextSegment],
        callback: Optional[Callable[[int, List[ContextSegment]], bool]] = None
    ) -> ProgressiveLoadState:
        """
        渐进式加载

        Args:
            segments: 所有片段
            callback: 每阶段回调函数(stage, loaded_segments) -> continue

        Returns:
            加载状态
        """
        return self._adapter.load_progressive(segments, callback)

    def get_iterator(
        self,
        segments: List[ContextSegment]
    ) -> Iterator[List[ContextSegment]]:
        """
        获取渐进式迭代器

        Yields:
            每阶段的片段列表
        """
        return self._adapter.get_iterator(segments)


# ============================================================================
# 5. LongContextManager - 长上下文管理器
# ============================================================================

class LongContextManager:
    """
    长上下文管理器
    
    统一管理长上下文的完整流程：
    1. 分块 -> 2. 检测边界 -> 3. 重叠融合 -> 4. 渐进加载
    
    借鉴视频处理流程：
    - 分镜（分块）
    - 场景检测（边界检测）
    - 转场融合（重叠融合）
    - 渐进播放（渐进加载）
    """
    
    def __init__(
        self,
        chunk_config: Optional[ChunkConfig] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None
    ):
        self.chunker = ContextChunker(chunk_config)
        self.fusion = OverlapFusion(fusion_method='smart')
        self.boundary_detector = BoundaryDetector()
        self.progressive_loader = ProgressiveLoader()
        self.embedding_fn = embedding_fn
        
        # 存储
        self.segments: Dict[str, ContextSegment] = {}
        self.segment_order: List[str] = []
        
        # 统计
        self.stats = {
            'total_processed': 0,
            'total_segments': 0,
            'total_boundaries': 0,
            'load_operations': 0
        }
    
    def process(self, content: str) -> List[ContextSegment]:
        """
        处理长上下文
        
        Args:
            content: 长文本内容
            
        Returns:
            处理后的片段列表
        """
        # 1. 分块
        segments = self.chunker.chunk(content, self.embedding_fn)
        
        # 2. 检测边界
        boundaries = self.boundary_detector.detect_boundaries(segments)
        
        # 3. 更新片段边界信息
        for idx, boundary_type, strength in boundaries:
            if idx < len(segments):
                segments[idx].boundary_type = boundary_type
                segments[idx].boundary_score = strength
        
        # 4. 存储
        for segment in segments:
            self.segments[segment.segment_id] = segment
            if segment.segment_id not in self.segment_order:
                self.segment_order.append(segment.segment_id)
        
        # 更新统计
        self.stats['total_processed'] += 1
        self.stats['total_segments'] += len(segments)
        self.stats['total_boundaries'] += len(boundaries)
        
        return segments
    
    def retrieve(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        fuse_results: bool = True
    ) -> str:
        """
        检索相关内容
        
        Args:
            query: 查询文本
            query_embedding: 查询嵌入
            top_k: 返回片段数
            fuse_results: 是否融合结果
            
        Returns:
            检索到的内容
        """
        if not self.segments:
            return ""
        
        # 生成查询嵌入
        if query_embedding is None and query and self.embedding_fn:
            query_embedding = self.embedding_fn(query)
        
        # 计算相似度
        segments = list(self.segments.values())
        
        if query_embedding:
            scored = [
                (seg, cosine_similarity(query_embedding, seg.embedding) if seg.embedding else 0)
                for seg in segments
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top_segments = [seg for seg, _ in scored[:top_k]]
        else:
            # 无查询，返回最近的
            top_segments = sorted(
                segments,
                key=lambda s: s.timestamp,
                reverse=True
            )[:top_k]
        
        # 融合或返回
        if fuse_results:
            return self.fusion.fuse(top_segments, query_embedding)
        else:
            return "\n\n".join(s.content for s in top_segments)
    
    def load_progressive(
        self,
        callback: Optional[Callable[[int, List[ContextSegment]], bool]] = None
    ) -> ProgressiveLoadState:
        """
        渐进式加载所有内容
        
        Args:
            callback: 阶段回调
            
        Returns:
            加载状态
        """
        segments = list(self.segments.values())
        state = self.progressive_loader.load_progressive(segments, callback)
        self.stats['load_operations'] += 1
        return state
    
    def get_context_window(
        self,
        segment_id: str,
        window_size: int = 2
    ) -> List[ContextSegment]:
        """
        获取上下文窗口
        
        Args:
            segment_id: 中心片段ID
            window_size: 每侧窗口大小
            
        Returns:
            上下文片段列表
        """
        if segment_id not in self.segments:
            return []
        
        # 找到位置
        try:
            center_idx = self.segment_order.index(segment_id)
        except ValueError:
            return []
        
        # 获取窗口
        start_idx = max(0, center_idx - window_size)
        end_idx = min(len(self.segment_order), center_idx + window_size + 1)
        
        window_ids = self.segment_order[start_idx:end_idx]
        return [self.segments[sid] for sid in window_ids if sid in self.segments]
    
    def split_by_boundaries(self) -> List[List[ContextSegment]]:
        """
        按边界分割
        
        Returns:
            分割后的片段组
        """
        segments = [self.segments[sid] for sid in self.segment_order if sid in self.segments]
        return self.boundary_detector.split_at_boundaries(segments)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'num_segments': len(self.segments),
            'chunker_history': len(self.chunker.chunk_history),
            'boundary_history': len(self.boundary_detector.detection_history),
            'load_history': len(self.progressive_loader.load_history)
        }
    
    def clear(self):
        """清空所有数据"""
        self.segments.clear()
        self.segment_order.clear()
        self.stats = {k: 0 for k in self.stats}
