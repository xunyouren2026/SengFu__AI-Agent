"""
统一分块/上下文管理模块

提供通用的分块和上下文管理功能，支持AGI和视频生成系统的需求。
支持视频（时间分块）和文本（语义分块）的统一处理。

核心组件：
- UnifiedChunker: 通用分块器
- UnifiedOverlapFusion: 重叠融合
- UnifiedBoundaryDetector: 边界检测
- UnifiedProgressiveLoader: 渐进式加载
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Iterator, Union
from enum import Enum, auto
import math

from .unified_config import (
    UnifiedAlgorithmConfig,
    ChunkingStrategy,
    BoundaryType,
    T, S
)


# ============================================================================
# 分块相关的数据结构
# ============================================================================

@dataclass
class Chunk(Generic[T]):
    """
    数据块
    
    存储分块后的数据单元。
    
    Attributes:
        data: 块数据
        index: 块索引
        start: 起始位置
        end: 结束位置
        metadata: 元数据
        boundary_type: 边界类型
    """
    data: List[T]
    index: int = 0
    start: int = 0
    end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    boundary_type: Optional[BoundaryType] = None
    
    def __len__(self) -> int:
        """获取块大小"""
        return len(self.data)
    
    def is_empty(self) -> bool:
        """检查块是否为空"""
        return len(self.data) == 0


@dataclass
class ChunkingResult(Generic[T]):
    """
    分块结果
    
    存储分块操作的完整结果。
    
    Attributes:
        chunks: 数据块列表
        total_items: 原始数据项总数
        strategy: 使用的分块策略
        overlap_size: 重叠大小
    """
    chunks: List[Chunk[T]]
    total_items: int = 0
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    overlap_size: int = 0
    
    def __len__(self) -> int:
        """获取块数量"""
        return len(self.chunks)
    
    def get_chunk(self, index: int) -> Optional[Chunk[T]]:
        """
        获取指定索引的块
        
        Args:
            index: 块索引
            
        Returns:
            数据块或None
        """
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None


@dataclass
class Boundary:
    """
    边界信息
    
    表示数据中的边界位置。
    
    Attributes:
        position: 边界位置
        type: 边界类型
        confidence: 置信度 (0.0-1.0)
        metadata: 额外信息
    """
    position: int
    type: BoundaryType
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 通用分块器
# ============================================================================

class UnifiedChunker(Generic[T]):
    """
    通用分块器
    
    支持多种分块策略：固定大小、语义分块、时间分块、自适应分块。
    适用于视频帧序列和文本token序列。
    
    Attributes:
        strategy: 分块策略
        chunk_size: 块大小
        overlap: 重叠大小
        boundary_detector: 边界检测器（可选）
    """
    
    def __init__(self,
                 strategy: Optional[ChunkingStrategy] = None,
                 chunk_size: Optional[int] = None,
                 overlap: Optional[int] = None,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化分块器
        
        Args:
            strategy: 分块策略
            chunk_size: 块大小
            overlap: 重叠大小
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.strategy = strategy or self.config.chunking_strategy
        self.chunk_size = chunk_size or self.config.chunk_size
        self.overlap = overlap or self.config.chunk_overlap
        
        self._boundary_detector: Optional[UnifiedBoundaryDetector] = None
    
    def set_boundary_detector(self, detector: UnifiedBoundaryDetector) -> None:
        """
        设置边界检测器
        
        Args:
            detector: 边界检测器
        """
        self._boundary_detector = detector
    
    def chunk(self, data: List[T]) -> ChunkingResult[T]:
        """
        对数据进行分块
        
        Args:
            data: 输入数据
            
        Returns:
            分块结果
        """
        if not data:
            return ChunkingResult(chunks=[], total_items=0, strategy=self.strategy)
        
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunk(data)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(data)
        elif self.strategy == ChunkingStrategy.TEMPORAL:
            return self._temporal_chunk(data)
        elif self.strategy == ChunkingStrategy.ADAPTIVE:
            return self._adaptive_chunk(data)
        else:
            return self._fixed_size_chunk(data)
    
    def _fixed_size_chunk(self, data: List[T]) -> ChunkingResult[T]:
        """固定大小分块"""
        chunks = []
        step = self.chunk_size - self.overlap
        
        index = 0
        for i in range(0, len(data), step):
            chunk_data = data[i:i + self.chunk_size]
            if chunk_data:  # 只添加非空块
                chunk = Chunk(
                    data=chunk_data,
                    index=index,
                    start=i,
                    end=min(i + self.chunk_size, len(data))
                )
                chunks.append(chunk)
                index += 1
        
        return ChunkingResult(
            chunks=chunks,
            total_items=len(data),
            strategy=ChunkingStrategy.FIXED_SIZE,
            overlap_size=self.overlap
        )
    
    def _semantic_chunk(self, data: List[T]) -> ChunkingResult[T]:
        """
        语义分块
        
        基于语义相似度进行分块。
        如果设置了边界检测器，会使用边界信息。
        """
        chunks = []
        
        if self._boundary_detector:
            # 使用边界检测器
            boundaries = self._boundary_detector.detect_boundaries(data)
            boundary_positions = [0] + [b.position for b in boundaries] + [len(data)]
            
            for i in range(len(boundary_positions) - 1):
                start = boundary_positions[i]
                end = boundary_positions[i + 1]
                chunk_data = data[start:end]
                
                if chunk_data:
                    chunk = Chunk(
                        data=chunk_data,
                        index=i,
                        start=start,
                        end=end,
                        boundary_type=boundaries[i].type if i < len(boundaries) else None
                    )
                    chunks.append(chunk)
        else:
            # 回退到基于相似度的简单语义分块
            chunks = self._similarity_based_chunk(data)
        
        return ChunkingResult(
            chunks=chunks,
            total_items=len(data),
            strategy=ChunkingStrategy.SEMANTIC,
            overlap_size=0
        )
    
    def _similarity_based_chunk(self, data: List[T]) -> List[Chunk[T]]:
        """基于相似度的语义分块"""
        chunks = []
        current_chunk = [data[0]] if data else []
        chunk_start = 0
        index = 0
        
        for i in range(1, len(data)):
            # 计算当前元素与前一个元素的相似度
            similarity = self._compute_similarity(data[i-1], data[i])
            
            # 如果相似度低或块已满，开始新块
            if similarity < 0.5 or len(current_chunk) >= self.chunk_size:
                if current_chunk:
                    chunk = Chunk(
                        data=current_chunk,
                        index=index,
                        start=chunk_start,
                        end=i
                    )
                    chunks.append(chunk)
                    index += 1
                
                current_chunk = [data[i]]
                chunk_start = i
            else:
                current_chunk.append(data[i])
        
        # 添加最后一个块
        if current_chunk:
            chunk = Chunk(
                data=current_chunk,
                index=index,
                start=chunk_start,
                end=len(data)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _temporal_chunk(self, data: List[T]) -> ChunkingResult[T]:
        """
        时间分块
        
        适用于视频帧序列，考虑时间连续性。
        """
        chunks = []
        step = self.chunk_size - self.overlap
        
        index = 0
        for i in range(0, len(data), step):
            chunk_data = data[i:i + self.chunk_size]
            if chunk_data:
                # 添加时间元数据
                metadata = {
                    'time_start': i,
                    'time_end': min(i + self.chunk_size, len(data)),
                    'duration': len(chunk_data)
                }
                
                chunk = Chunk(
                    data=chunk_data,
                    index=index,
                    start=i,
                    end=min(i + self.chunk_size, len(data)),
                    metadata=metadata,
                    boundary_type=BoundaryType.SHOT
                )
                chunks.append(chunk)
                index += 1
        
        return ChunkingResult(
            chunks=chunks,
            total_items=len(data),
            strategy=ChunkingStrategy.TEMPORAL,
            overlap_size=self.overlap
        )
    
    def _adaptive_chunk(self, data: List[T]) -> ChunkingResult[T]:
        """
        自适应分块
        
        根据数据特征动态调整块大小。
        """
        chunks = []
        current_chunk = []
        chunk_start = 0
        index = 0
        
        min_size = self.chunk_size // 2
        max_size = self.chunk_size * 2
        
        for i, item in enumerate(data):
            current_chunk.append(item)
            
            should_split = False
            
            # 检查是否达到最小大小
            if len(current_chunk) >= min_size:
                # 检查内容变化（简化实现）
                if len(current_chunk) >= 2:
                    similarity = self._compute_similarity(current_chunk[-2], current_chunk[-1])
                    if similarity < 0.3:  # 内容变化大
                        should_split = True
                
                # 检查是否达到最大大小
                if len(current_chunk) >= max_size:
                    should_split = True
            
            if should_split:
                chunk = Chunk(
                    data=current_chunk,
                    index=index,
                    start=chunk_start,
                    end=i + 1
                )
                chunks.append(chunk)
                index += 1
                
                # 保留重叠部分
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:]
                chunk_start = i + 1 - len(current_chunk)
        
        # 添加最后一个块
        if current_chunk:
            chunk = Chunk(
                data=current_chunk,
                index=index,
                start=chunk_start,
                end=len(data)
            )
            chunks.append(chunk)
        
        return ChunkingResult(
            chunks=chunks,
            total_items=len(data),
            strategy=ChunkingStrategy.ADAPTIVE,
            overlap_size=self.overlap
        )
    
    def _compute_similarity(self, a: T, b: T) -> float:
        """
        计算两个元素的相似度
        
        Args:
            a: 第一个元素
            b: 第二个元素
            
        Returns:
            相似度 (0.0-1.0)
        """
        try:
            if a == b:
                return 1.0
            
            # 尝试向量相似度
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                vec_a = [float(x) for x in a]
                vec_b = [float(x) for x in b]
                
                if len(vec_a) == len(vec_b) and len(vec_a) > 0:
                    dot = sum(x * y for x, y in zip(vec_a, vec_b))
                    norm_a = math.sqrt(sum(x * x for x in vec_a))
                    norm_b = math.sqrt(sum(x * x for x in vec_b))
                    
                    if norm_a > 0 and norm_b > 0:
                        return dot / (norm_a * norm_b)
            
            return 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def merge_chunks(self, chunks: List[Chunk[T]], 
                    max_gap: int = 1) -> ChunkingResult[T]:
        """
        合并相邻的块
        
        Args:
            chunks: 块列表
            max_gap: 最大允许间隙
            
        Returns:
            合并后的分块结果
        """
        if not chunks:
            return ChunkingResult(chunks=[], total_items=0, strategy=self.strategy)
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if chunk.start - current.end <= max_gap:
                # 合并
                current = Chunk(
                    data=current.data + chunk.data,
                    index=current.index,
                    start=current.start,
                    end=chunk.end,
                    metadata={**current.metadata, **chunk.metadata}
                )
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        
        # 重新索引
        for i, chunk in enumerate(merged):
            chunk.index = i
        
        total_items = sum(len(c.data) for c in merged)
        
        return ChunkingResult(
            chunks=merged,
            total_items=total_items,
            strategy=self.strategy,
            overlap_size=0
        )


# ============================================================================
# 重叠融合
# ============================================================================

class UnifiedOverlapFusion(Generic[T]):
    """
    重叠融合
    
    处理分块之间的重叠区域，平滑边界过渡。
    支持多种融合策略。
    
    Attributes:
        fusion_strategy: 融合策略
        blend_window: 混合窗口大小
    """
    
    def __init__(self, blend_window: int = 16):
        """
        初始化重叠融合器
        
        Args:
            blend_window: 混合窗口大小
        """
        self.blend_window = blend_window
    
    def fuse(self, chunks: List[Chunk[T]], 
             overlap_size: int) -> List[Chunk[T]]:
        """
        融合重叠的块
        
        Args:
            chunks: 块列表
            overlap_size: 重叠大小
            
        Returns:
            融合后的块列表
        """
        if not chunks or overlap_size <= 0:
            return chunks
        
        fused = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # 第一个块：保留全部
                fused.append(chunk)
            else:
                # 后续块：与前一块的尾部融合
                prev_chunk = chunks[i - 1]
                
                # 提取重叠区域
                overlap_start = max(0, len(prev_chunk.data) - overlap_size)
                prev_overlap = prev_chunk.data[overlap_start:]
                curr_overlap = chunk.data[:min(overlap_size, len(chunk.data))]
                
                # 融合重叠区域
                if len(prev_overlap) == len(curr_overlap) and len(prev_overlap) > 0:
                    fused_overlap = self._blend(prev_overlap, curr_overlap)
                    
                    # 更新前一个块的尾部
                    new_prev_data = prev_chunk.data[:overlap_start] + fused_overlap
                    fused[-1] = Chunk(
                        data=new_prev_data,
                        index=prev_chunk.index,
                        start=prev_chunk.start,
                        end=prev_chunk.end,
                        metadata=prev_chunk.metadata,
                        boundary_type=prev_chunk.boundary_type
                    )
                    
                    # 当前块去掉重叠部分
                    new_curr_data = chunk.data[len(curr_overlap):]
                    if new_curr_data:  # 只添加非空块
                        fused.append(Chunk(
                            data=new_curr_data,
                            index=chunk.index,
                            start=chunk.start + len(curr_overlap),
                            end=chunk.end,
                            metadata=chunk.metadata,
                            boundary_type=chunk.boundary_type
                        ))
                else:
                    fused.append(chunk)
        
        return fused
    
    def _blend(self, a: List[T], b: List[T]) -> List[T]:
        """
        混合两个序列
        
        使用线性插值进行混合。
        
        Args:
            a: 第一个序列
            b: 第二个序列
            
        Returns:
            混合后的序列
        """
        if len(a) != len(b):
            # 长度不匹配，返回较长的
            return a if len(a) > len(b) else b
        
        result = []
        n = len(a)
        
        for i in range(n):
            weight_a = 1.0 - (i / n)  # 从1递减到0
            weight_b = i / n          # 从0递增到1
            
            try:
                # 尝试数值混合
                if isinstance(a[i], (int, float)) and isinstance(b[i], (int, float)):
                    blended = weight_a * float(a[i]) + weight_b * float(b[i])
                    result.append(type(a[i])(blended))
                elif isinstance(a[i], (list, tuple)) and isinstance(b[i], (list, tuple)):
                    # 向量混合
                    blended_vec = []
                    for j in range(min(len(a[i]), len(b[i]))):
                        val = weight_a * float(a[i][j]) + weight_b * float(b[i][j])
                        blended_vec.append(val)
                    result.append(type(a[i])(blended_vec))
                else:
                    # 无法混合，根据权重选择
                    result.append(a[i] if weight_a > weight_b else b[i])
            except (ValueError, TypeError):
                result.append(a[i] if weight_a > weight_b else b[i])
        
        return result
    
    def smooth_boundaries(self, data: List[T], 
                         boundaries: List[int]) -> List[T]:
        """
        平滑边界处的过渡
        
        Args:
            data: 完整数据
            boundaries: 边界位置列表
            
        Returns:
            平滑后的数据
        """
        result = list(data)
        
        for boundary in boundaries:
            # 在边界周围应用平滑
            start = max(0, boundary - self.blend_window // 2)
            end = min(len(data), boundary + self.blend_window // 2)
            
            if end - start < 2:
                continue
            
            # 简单的移动平均平滑
            for i in range(start + 1, end - 1):
                try:
                    if isinstance(result[i], (int, float)):
                        smoothed = (float(result[i-1]) + float(result[i]) + float(result[i+1])) / 3
                        result[i] = type(result[i])(smoothed)
                except (ValueError, TypeError):
                    pass
        
        return result


# ============================================================================
# 边界检测
# ============================================================================

class UnifiedBoundaryDetector(Generic[T]):
    """
    边界检测器
    
    检测数据中的边界位置，支持：
    - 镜头边界（视频）
    - 场景边界（视频）
    - 话题边界（文本）
    - 段落边界（文本）
    
    Attributes:
        boundary_type: 边界类型
        threshold: 检测阈值
        min_segment_size: 最小段大小
    """
    
    def __init__(self,
                 boundary_type: BoundaryType = BoundaryType.TOPIC,
                 threshold: float = 0.5,
                 min_segment_size: int = 10):
        """
        初始化边界检测器
        
        Args:
            boundary_type: 边界类型
            threshold: 检测阈值
            min_segment_size: 最小段大小
        """
        self.boundary_type = boundary_type
        self.threshold = threshold
        self.min_segment_size = min_segment_size
    
    def detect_boundaries(self, data: List[T]) -> List[Boundary]:
        """
        检测数据中的边界
        
        Args:
            data: 输入数据
            
        Returns:
            边界列表
        """
        if not data or len(data) < self.min_segment_size * 2:
            return []
        
        if self.boundary_type in (BoundaryType.SHOT, BoundaryType.SCENE):
            return self._detect_video_boundaries(data)
        else:
            return self._detect_text_boundaries(data)
    
    def _detect_video_boundaries(self, data: List[T]) -> List[Boundary]:
        """
        检测视频边界
        
        基于帧间差异检测镜头/场景边界。
        """
        boundaries = []
        
        for i in range(1, len(data)):
            # 计算帧间差异
            diff = self._compute_difference(data[i-1], data[i])
            
            # 如果差异超过阈值，标记为边界
            if diff > self.threshold:
                # 检查最小段大小
                if not boundaries or i - boundaries[-1].position >= self.min_segment_size:
                    boundary = Boundary(
                        position=i,
                        type=self.boundary_type,
                        confidence=min(1.0, diff),
                        metadata={'difference': diff}
                    )
                    boundaries.append(boundary)
        
        return boundaries
    
    def _detect_text_boundaries(self, data: List[T]) -> List[Boundary]:
        """
        检测文本边界
        
        基于语义变化检测话题/段落边界。
        """
        boundaries = []
        
        # 滑动窗口计算语义一致性
        window_size = min(self.min_segment_size, len(data) // 4)
        if window_size < 2:
            window_size = 2
        
        for i in range(window_size, len(data) - window_size):
            # 计算前后窗口的相似度
            prev_window = data[i-window_size:i]
            next_window = data[i:i+window_size]
            
            similarity = self._compute_window_similarity(prev_window, next_window)
            
            # 如果相似度低，标记为边界
            if similarity < (1.0 - self.threshold):
                if not boundaries or i - boundaries[-1].position >= self.min_segment_size:
                    boundary = Boundary(
                        position=i,
                        type=self.boundary_type,
                        confidence=1.0 - similarity,
                        metadata={'similarity': similarity}
                    )
                    boundaries.append(boundary)
        
        return boundaries
    
    def _compute_difference(self, a: T, b: T) -> float:
        """
        计算两个元素的差异
        
        Args:
            a: 第一个元素
            b: 第二个元素
            
        Returns:
            差异值 (0.0-1.0)
        """
        try:
            # 尝试向量差异
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                vec_a = [float(x) for x in a]
                vec_b = [float(x) for x in b]
                
                if len(vec_a) == len(vec_b) and len(vec_a) > 0:
                    # 欧氏距离归一化
                    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(vec_a, vec_b)))
                    max_dist = math.sqrt(len(vec_a))  # 最大可能距离
                    return min(1.0, dist / max_dist) if max_dist > 0 else 0.0
            
            # 简单比较
            return 0.0 if a == b else 1.0
        except (ValueError, TypeError):
            return 0.0 if a == b else 1.0
    
    def _compute_window_similarity(self, window_a: List[T], window_b: List[T]) -> float:
        """
        计算两个窗口的相似度
        
        Args:
            window_a: 第一个窗口
            window_b: 第二个窗口
            
        Returns:
            相似度 (0.0-1.0)
        """
        if not window_a or not window_b:
            return 0.0
        
        # 计算所有元素对的平均相似度
        similarities = []
        for a in window_a:
            for b in window_b:
                diff = self._compute_difference(a, b)
                similarities.append(1.0 - diff)
        
        return sum(similarities) / len(similarities) if similarities else 0.0


# ============================================================================
# 渐进式加载
# ============================================================================

class UnifiedProgressiveLoader(Generic[T]):
    """
    渐进式加载器
    
    支持大数据集的渐进式加载，按需加载数据块。
    适用于视频流和长文本处理。
    
    Attributes:
        chunk_size: 块大小
        prefetch_size: 预取块数
        loaded_chunks: 已加载的块
    """
    
    def __init__(self,
                 chunk_size: Optional[int] = None,
                 prefetch_size: int = 2,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化渐进式加载器
        
        Args:
            chunk_size: 块大小
            prefetch_size: 预取块数
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.default_config()
        self.chunk_size = chunk_size or self.config.chunk_size
        self.prefetch_size = prefetch_size
        
        self._data_source: Optional[List[T]] = None
        self._chunker = UnifiedChunker(chunk_size=self.chunk_size)
        self._chunks: List[Chunk[T]] = []
        self._loaded_indices: set = set()
        self._current_index: int = 0
    
    def load_source(self, data: List[T]) -> None:
        """
        加载数据源
        
        Args:
            data: 数据源
        """
        self._data_source = data
        result = self._chunker.chunk(data)
        self._chunks = result.chunks
        self._loaded_indices.clear()
        self._current_index = 0
    
    def set_source(self, chunks: List[Chunk[T]]) -> None:
        """
        直接设置块源
        
        Args:
            chunks: 块列表
        """
        self._chunks = chunks
        self._loaded_indices.clear()
        self._current_index = 0
    
    def get_chunk(self, index: int) -> Optional[Chunk[T]]:
        """
        获取指定块（带预取）
        
        Args:
            index: 块索引
            
        Returns:
            数据块或None
        """
        if index < 0 or index >= len(self._chunks):
            return None
        
        # 标记为已加载
        self._loaded_indices.add(index)
        self._current_index = index
        
        # 预取后续块
        self._prefetch(index)
        
        return self._chunks[index]
    
    def _prefetch(self, current_index: int) -> None:
        """
        预取后续块
        
        Args:
            current_index: 当前索引
        """
        for i in range(1, self.prefetch_size + 1):
            prefetch_index = current_index + i
            if prefetch_index < len(self._chunks):
                self._loaded_indices.add(prefetch_index)
    
    def next_chunk(self) -> Optional[Chunk[T]]:
        """
        获取下一个块
        
        Returns:
            下一个数据块或None
        """
        return self.get_chunk(self._current_index + 1)
    
    def previous_chunk(self) -> Optional[Chunk[T]]:
        """
        获取上一个块
        
        Returns:
            上一个数据块或None
        """
        return self.get_chunk(self._current_index - 1)
    
    def iter_chunks(self) -> Iterator[Chunk[T]]:
        """
        迭代所有块
        
        Yields:
            数据块
        """
        for i in range(len(self._chunks)):
            chunk = self.get_chunk(i)
            if chunk:
                yield chunk
    
    def get_loaded_count(self) -> int:
        """
        获取已加载的块数
        
        Returns:
            已加载块数
        """
        return len(self._loaded_indices)
    
    def get_progress(self) -> float:
        """
        获取加载进度
        
        Returns:
            进度 (0.0-1.0)
        """
        if not self._chunks:
            return 0.0
        return self._current_index / len(self._chunks)
    
    def seek(self, position: float) -> Optional[Chunk[T]]:
        """
        跳转到指定位置
        
        Args:
            position: 位置 (0.0-1.0)
            
        Returns:
            目标块或None
        """
        if not self._chunks:
            return None
        
        index = int(position * len(self._chunks))
        index = max(0, min(index, len(self._chunks) - 1))
        return self.get_chunk(index)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取加载器统计信息
        
        Returns:
            统计信息
        """
        return {
            'total_chunks': len(self._chunks),
            'loaded_chunks': len(self._loaded_indices),
            'current_index': self._current_index,
            'progress': self.get_progress(),
            'prefetch_size': self.prefetch_size,
        }
