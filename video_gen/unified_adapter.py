"""
Video Generation Unified Adapter Module

适配器模块，将统一核心算法适配为视频生成专用组件。
保持原有API不变，内部使用统一核心实现。

适配器组件：
- VideoMemoryAdapter: 将UnifiedMemoryBank适配为视频帧记忆
- VideoAttentionAdapter: 将统一注意力适配为视频时空注意力
- VideoChunkerAdapter: 将统一分块器适配为视频时间分块
- VideoMoEAdapter: 将统一MoE适配为视频专家系统

使用示例：
    >>> from video_gen.unified_adapter import VideoMemoryAdapter
    >>> memory = VideoMemoryAdapter(capacity=1000)
    >>> memory.store_frame(frame_data, frame_idx=0, importance=0.8)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Tuple, Union
from enum import Enum, auto
import math

# 导入统一核心算法
from agi_unified_framework.core.unified_algorithms import (
    # 配置
    UnifiedAlgorithmConfig,
    ChunkingStrategy,
    BoundaryType,
    ExpertType,
    ConstraintPriority,
    # 记忆系统
    MemoryEntry,
    MemoryQuery,
    UnifiedMemoryBank,
    # 注意力机制
    UnifiedSlidingWindowAttention,
    UnifiedDynamicRouting,
    AttentionContext,
    # 分块系统
    UnifiedChunker,
    UnifiedOverlapFusion,
    UnifiedBoundaryDetector,
    Chunk,
    ChunkingResult,
    Boundary,
    # MoE系统
    MixtureOfExperts,
    Expert,
    ExpertOutput,
    RoutingInfo,
    PhysicalExpert,
    GenerationExpert,
    # 约束系统
    ConstraintManager,
    PhysicsConstraint,
    ConstraintCheckResult,
)


# ============================================================================
# 视频帧数据结构
# ============================================================================

@dataclass
class VideoFrame:
    """
    视频帧数据结构
    
    Attributes:
        data: 帧数据（可以是特征向量、像素值等）
        frame_idx: 帧索引
        timestamp: 时间戳
        metadata: 元数据
    """
    data: Any
    frame_idx: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatioTemporalQuery:
    """
    时空查询
    
    Attributes:
        spatial_query: 空间查询
        temporal_query: 时间查询
        top_k: 返回结果数量
        temporal_window: 时间窗口大小
    """
    spatial_query: Optional[Any] = None
    temporal_query: Optional[Any] = None
    top_k: int = 5
    temporal_window: int = 16


# ============================================================================
# Video Memory Adapter
# ============================================================================

class VideoMemoryAdapter:
    """
    视频记忆适配器
    
    将UnifiedMemoryBank适配为视频帧记忆系统。
    支持帧级别的存储、检索和时间关联。
    
    Attributes:
        capacity: 最大容量
        unified_memory: 底层统一记忆库
        frame_indices: 帧索引映射
    """
    
    def __init__(self, capacity: int = 10000, 
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化视频记忆适配器
        
        Args:
            capacity: 最大容量
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.video_optimized_config()
        self.capacity = capacity
        
        # 创建统一记忆库
        self.unified_memory = UnifiedMemoryBank(
            config=self.config.update(memory_capacity=capacity)
        )
        
        # 帧索引映射
        self._frame_indices: Dict[int, int] = {}  # frame_idx -> entry_id
        self._entry_id_counter = 0
    
    def store_frame(self, frame_data: Any, frame_idx: int = 0,
                    importance: float = 0.5, metadata: Optional[Dict] = None) -> bool:
        """
        存储视频帧
        
        Args:
            frame_data: 帧数据
            frame_idx: 帧索引
            importance: 重要性分数
            metadata: 元数据
            
        Returns:
            存储是否成功
        """
        # 创建视频帧对象
        frame = VideoFrame(
            data=frame_data,
            frame_idx=frame_idx,
            timestamp=float(frame_idx),
            metadata=metadata or {}
        )
        
        # 创建记忆条目
        entry = MemoryEntry(
            data=frame,
            timestamp=float(frame_idx),
            importance=importance,
            metadata={'frame_idx': frame_idx, 'entry_id': self._entry_id_counter}
        )
        
        # 存储到统一记忆库
        success = self.unified_memory.store(entry)
        
        if success:
            self._frame_indices[frame_idx] = self._entry_id_counter
            self._entry_id_counter += 1
        
        return success
    
    def store_frame_batch(self, frames: List[Tuple[Any, int]], 
                          importance: float = 0.5) -> int:
        """
        批量存储视频帧
        
        Args:
            frames: (帧数据, 帧索引) 列表
            importance: 重要性分数
            
        Returns:
            成功存储的数量
        """
        count = 0
        for frame_data, frame_idx in frames:
            if self.store_frame(frame_data, frame_idx, importance):
                count += 1
        return count
    
    def retrieve_by_frame_idx(self, frame_idx: int) -> Optional[VideoFrame]:
        """
        通过帧索引检索帧
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            视频帧或None
        """
        # 创建查询
        query = MemoryQuery(
            query_data=VideoFrame(data=None, frame_idx=frame_idx),
            top_k=1,
            threshold=0.0
        )
        
        results = self.unified_memory.retrieve(query)
        
        for entry in results:
            if isinstance(entry.data, VideoFrame) and entry.data.frame_idx == frame_idx:
                return entry.data
        
        return None
    
    def retrieve_similar_frames(self, query_frame: Any, top_k: int = 5,
                                temporal_window: Optional[Tuple[int, int]] = None) -> List[VideoFrame]:
        """
        检索相似帧
        
        Args:
            query_frame: 查询帧数据
            top_k: 返回结果数量
            temporal_window: 时间窗口 (start, end)
            
        Returns:
            相似帧列表
        """
        query = MemoryQuery(
            query_data=VideoFrame(data=query_frame),
            top_k=top_k * 2,  # 获取更多结果用于过滤
            threshold=0.1
        )
        
        results = self.unified_memory.retrieve(query)
        
        frames = []
        for entry in results:
            if isinstance(entry.data, VideoFrame):
                frame = entry.data
                
                # 应用时间窗口过滤
                if temporal_window:
                    start, end = temporal_window
                    if not (start <= frame.frame_idx <= end):
                        continue
                
                frames.append(frame)
                
                if len(frames) >= top_k:
                    break
        
        return frames
    
    def retrieve_temporal_sequence(self, start_idx: int, 
                                   end_idx: int) -> List[VideoFrame]:
        """
        检索时间序列
        
        Args:
            start_idx: 起始帧索引
            end_idx: 结束帧索引
            
        Returns:
            按时间排序的帧列表
        """
        frames = []
        
        for frame_idx in range(start_idx, end_idx + 1):
            frame = self.retrieve_by_frame_idx(frame_idx)
            if frame:
                frames.append(frame)
        
        # 按时间排序
        frames.sort(key=lambda f: f.frame_idx)
        return frames
    
    def update_frame_importance(self, frame_idx: int, importance: float) -> bool:
        """
        更新帧重要性
        
        Args:
            frame_idx: 帧索引
            importance: 新的重要性分数
            
        Returns:
            是否成功更新
        """
        # 在统一记忆库中查找并更新
        for entry in self.unified_memory.entries:
            if (isinstance(entry.data, VideoFrame) and 
                entry.data.frame_idx == frame_idx):
                entry.importance = max(0.0, min(1.0, importance))
                return True
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = self.unified_memory.get_stats()
        
        return {
            **base_stats,
            'frame_count': len(self._frame_indices),
            'frame_idx_range': (min(self._frame_indices.keys()), 
                               max(self._frame_indices.keys())) if self._frame_indices else (0, 0)
        }
    
    def clear(self) -> None:
        """清空所有记忆"""
        self.unified_memory.clear()
        self._frame_indices.clear()
        self._entry_id_counter = 0
    
    def size(self) -> int:
        """获取存储的帧数"""
        return self.unified_memory.size()


# ============================================================================
# Video Attention Adapter
# ============================================================================

class VideoAttentionAdapter:
    """
    视频注意力适配器
    
    将统一注意力机制适配为视频时空注意力。
    支持空间注意力和时间注意力的组合。
    
    Attributes:
        spatial_attention: 空间注意力
        temporal_attention: 时间注意力
        use_unified_core: 是否使用统一核心
    """
    
    def __init__(self, 
                 spatial_window_size: int = 16,
                 temporal_window_size: int = 16,
                 use_unified_core: bool = True,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化视频注意力适配器
        
        Args:
            spatial_window_size: 空间窗口大小
            temporal_window_size: 时间窗口大小
            use_unified_core: 是否使用统一核心
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.video_optimized_config()
        self.spatial_window_size = spatial_window_size
        self.temporal_window_size = temporal_window_size
        self.use_unified_core = use_unified_core
        
        if use_unified_core:
            # 使用统一核心的滑动窗口注意力
            self.spatial_attention = UnifiedSlidingWindowAttention(
                window_size=spatial_window_size,
                config=self.config
            )
            self.temporal_attention = UnifiedSlidingWindowAttention(
                window_size=temporal_window_size,
                config=self.config
            )
        else:
            self.spatial_attention = None
            self.temporal_attention = None
        
        # 动态路由
        self.dynamic_routing = UnifiedDynamicRouting(
            config=self.config
        )
    
    def compute_spatial_attention(self, query: List[Any], 
                                   key: List[Any], 
                                   value: List[Any]) -> List[Any]:
        """
        计算空间注意力
        
        Args:
            query: 查询
            key: 键
            value: 值
            
        Returns:
            注意力输出
        """
        if self.use_unified_core and self.spatial_attention:
            return self.spatial_attention.compute(query, key, value)
        
        # 回退到简单实现
        return self._simple_attention(query, key, value)
    
    def compute_temporal_attention(self, query: List[Any], 
                                    key: List[Any], 
                                    value: List[Any]) -> List[Any]:
        """
        计算时间注意力
        
        Args:
            query: 查询
            key: 键
            value: 值
            
        Returns:
            注意力输出
        """
        if self.use_unified_core and self.temporal_attention:
            return self.temporal_attention.compute(query, key, value)
        
        # 回退到简单实现
        return self._simple_attention(query, key, value)
    
    def compute_spatiotemporal_attention(self, 
                                          spatial_query: List[Any],
                                          spatial_key: List[Any],
                                          spatial_value: List[Any],
                                          temporal_query: List[Any],
                                          temporal_key: List[Any],
                                          temporal_value: List[Any],
                                          spatial_weight: float = 0.5) -> Tuple[List[Any], List[Any]]:
        """
        计算时空联合注意力
        
        Args:
            spatial_query: 空间查询
            spatial_key: 空间键
            spatial_value: 空间值
            temporal_query: 时间查询
            temporal_key: 时间键
            temporal_value: 时间值
            spatial_weight: 空间注意力权重
            
        Returns:
            (空间注意力输出, 时间注意力输出)
        """
        # 计算空间注意力
        spatial_out = self.compute_spatial_attention(
            spatial_query, spatial_key, spatial_value
        )
        
        # 计算时间注意力
        temporal_out = self.compute_temporal_attention(
            temporal_query, temporal_key, temporal_value
        )
        
        return spatial_out, temporal_out
    
    def route_tokens(self, tokens: List[Any], 
                     importance_scores: Optional[List[float]] = None) -> List[int]:
        """
        路由tokens到注意力计算
        
        Args:
            tokens: 输入tokens
            importance_scores: 重要性分数
            
        Returns:
            选中的token索引
        """
        decision = self.dynamic_routing.route(tokens, importance_scores)
        return decision.token_indices
    
    def _simple_attention(self, query: List[Any], 
                          key: List[Any], 
                          value: List[Any]) -> List[Any]:
        """简单注意力实现（回退）"""
        if not query or not key or not value:
            return []
        
        output = []
        for q in query:
            # 计算与所有key的相似度
            scores = []
            for k in key:
                score = self._compute_similarity(q, k)
                scores.append(score)
            
            # softmax
            weights = self._softmax(scores)
            
            # 加权求和
            result = self._weighted_sum(weights, value)
            output.append(result)
        
        return output
    
    def _compute_similarity(self, a: Any, b: Any) -> float:
        """计算相似度"""
        try:
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                vec_a = [float(x) for x in a]
                vec_b = [float(x) for x in b]
                
                if len(vec_a) == len(vec_b) and len(vec_a) > 0:
                    dot = sum(x * y for x, y in zip(vec_a, vec_b))
                    norm_a = math.sqrt(sum(x * x for x in vec_a))
                    norm_b = math.sqrt(sum(x * x for x in vec_b))
                    
                    if norm_a > 0 and norm_b > 0:
                        return dot / (norm_a * norm_b)
            
            return 1.0 if a == b else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _softmax(self, values: List[float]) -> List[float]:
        """计算softmax"""
        if not values:
            return []
        
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) for v in values]
        sum_exp = sum(exp_vals)
        
        if sum_exp == 0:
            return [1.0 / len(values)] * len(values)
        
        return [v / sum_exp for v in exp_vals]
    
    def _weighted_sum(self, weights: List[float], values: List[Any]) -> Any:
        """加权求和"""
        if not weights or not values:
            return values[0] if values else None
        
        try:
            if isinstance(values[0], (list, tuple)):
                result = []
                for i in range(len(values[0])):
                    val = sum(w * float(v[i]) for w, v in zip(weights, values))
                    result.append(val)
                return type(values[0])(result)
            elif isinstance(values[0], (int, float)):
                return sum(w * float(v) for w, v in zip(weights, values))
        except (ValueError, TypeError, IndexError):
            pass
        
        # 返回权重最高的值
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        return values[max_idx]
    
    def get_attention_weights(self) -> Dict[str, List[List[float]]]:
        """
        获取注意力权重
        
        Returns:
            注意力权重字典
        """
        weights = {}
        
        if self.use_unified_core:
            if self.spatial_attention:
                weights['spatial'] = self.spatial_attention.get_attention_weights()
            if self.temporal_attention:
                weights['temporal'] = self.temporal_attention.get_attention_weights()
        
        return weights
    
    def reset_cache(self) -> None:
        """重置缓存"""
        if self.use_unified_core:
            if self.spatial_attention:
                self.spatial_attention.reset_cache()
            if self.temporal_attention:
                self.temporal_attention.reset_cache()


# ============================================================================
# Video Chunker Adapter
# ============================================================================

class VideoChunkerAdapter:
    """
    视频分块适配器
    
    将统一分块器适配为视频时间分块。
    支持基于镜头边界的智能分块。
    
    Attributes:
        unified_chunker: 统一分块器
        boundary_detector: 边界检测器
        overlap_fusion: 重叠融合器
    """
    
    def __init__(self,
                 chunk_size: int = 16,
                 overlap: int = 4,
                 use_boundary_detection: bool = True,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化视频分块适配器
        
        Args:
            chunk_size: 块大小（帧数）
            overlap: 重叠大小（帧数）
            use_boundary_detection: 是否使用边界检测
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.video_optimized_config()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_boundary_detection = use_boundary_detection
        
        # 创建统一分块器（使用时间分块策略）
        self.unified_chunker = UnifiedChunker(
            strategy=ChunkingStrategy.TEMPORAL,
            chunk_size=chunk_size,
            overlap=overlap,
            config=self.config
        )
        
        # 创建边界检测器
        if use_boundary_detection:
            self.boundary_detector = UnifiedBoundaryDetector(
                boundary_type=BoundaryType.SHOT,
                threshold=0.5,
                min_segment_size=chunk_size // 2
            )
            self.unified_chunker.set_boundary_detector(self.boundary_detector)
        else:
            self.boundary_detector = None
        
        # 创建重叠融合器
        self.overlap_fusion = UnifiedOverlapFusion(blend_window=overlap)
    
    def chunk_video(self, frames: List[Any]) -> List[Dict[str, Any]]:
        """
        对视频进行分块
        
        Args:
            frames: 视频帧列表
            
        Returns:
            视频块列表，每个块包含：
            - frames: 帧数据列表
            - start_idx: 起始帧索引
            - end_idx: 结束帧索引
            - is_keyframe_chunk: 是否包含关键帧
        """
        # 使用统一分块器
        result = self.unified_chunker.chunk(frames)
        
        # 转换为视频块格式
        video_chunks = []
        for chunk in result.chunks:
            video_chunk = {
                'frames': chunk.data,
                'start_idx': chunk.start,
                'end_idx': chunk.end,
                'is_keyframe_chunk': chunk.boundary_type == BoundaryType.SHOT,
                'metadata': chunk.metadata
            }
            video_chunks.append(video_chunk)
        
        return video_chunks
    
    def detect_shot_boundaries(self, frames: List[Any]) -> List[int]:
        """
        检测镜头边界
        
        Args:
            frames: 视频帧列表
            
        Returns:
            边界帧索引列表
        """
        if not self.boundary_detector:
            return []
        
        boundaries = self.boundary_detector.detect_boundaries(frames)
        return [b.position for b in boundaries]
    
    def fuse_chunk_boundaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        融合块边界
        
        Args:
            chunks: 视频块列表
            
        Returns:
            融合后的块列表
        """
        # 转换为统一分块格式
        unified_chunks = []
        for i, chunk in enumerate(chunks):
            unified_chunk = Chunk(
                data=chunk['frames'],
                index=i,
                start=chunk['start_idx'],
                end=chunk['end_idx'],
                metadata=chunk.get('metadata', {})
            )
            unified_chunks.append(unified_chunk)
        
        # 使用重叠融合
        fused = self.overlap_fusion.fuse(unified_chunks, self.overlap)
        
        # 转换回视频块格式
        result = []
        for chunk in fused:
            video_chunk = {
                'frames': chunk.data,
                'start_idx': chunk.start,
                'end_idx': chunk.end,
                'metadata': chunk.metadata
            }
            result.append(video_chunk)
        
        return result
    
    def get_chunk_at_timestamp(self, chunks: List[Dict[str, Any]], 
                               frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        获取指定帧索引所在的块
        
        Args:
            chunks: 视频块列表
            frame_idx: 帧索引
            
        Returns:
            视频块或None
        """
        for chunk in chunks:
            if chunk['start_idx'] <= frame_idx < chunk['end_idx']:
                return chunk
        return None
    
    def merge_adjacent_chunks(self, chunks: List[Dict[str, Any]], 
                              max_gap: int = 1) -> List[Dict[str, Any]]:
        """
        合并相邻的块
        
        Args:
            chunks: 视频块列表
            max_gap: 最大允许间隙
            
        Returns:
            合并后的块列表
        """
        if not chunks:
            return []
        
        # 转换为统一分块格式
        unified_chunks = []
        for i, chunk in enumerate(chunks):
            unified_chunk = Chunk(
                data=chunk['frames'],
                index=i,
                start=chunk['start_idx'],
                end=chunk['end_idx'],
                metadata=chunk.get('metadata', {})
            )
            unified_chunks.append(unified_chunk)
        
        # 使用统一分块器的合并功能
        result = self.unified_chunker.merge_chunks(unified_chunks, max_gap)
        
        # 转换回视频块格式
        merged_chunks = []
        for chunk in result.chunks:
            video_chunk = {
                'frames': chunk.data,
                'start_idx': chunk.start,
                'end_idx': chunk.end,
                'metadata': chunk.metadata
            }
            merged_chunks.append(video_chunk)
        
        return merged_chunks


# ============================================================================
# Video MoE Adapter
# ============================================================================

class VideoMoEAdapter:
    """
    视频MoE适配器
    
    将统一MoE适配为视频专家系统。
    支持视频生成专用的专家类型。
    
    Attributes:
        unified_moe: 统一MoE系统
        video_experts: 视频专用专家
    """
    
    def __init__(self,
                 num_experts: int = 8,
                 top_k: int = 2,
                 config: Optional[UnifiedAlgorithmConfig] = None):
        """
        初始化视频MoE适配器
        
        Args:
            num_experts: 专家数量
            top_k: 激活的专家数
            config: 算法配置
        """
        self.config = config or UnifiedAlgorithmConfig.video_optimized_config()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 创建统一MoE系统
        self.unified_moe = MixtureOfExperts(
            num_experts=num_experts,
            top_k=top_k,
            config=self.config
        )
        
        # 初始化视频专用专家
        self._init_video_experts()
    
    def _init_video_experts(self) -> None:
        """初始化视频专用专家"""
        # 清除默认专家，添加视频专用专家
        self.unified_moe.experts.clear()
        
        expert_configs = [
            (ExpertType.PHYSICAL, 'physics'),
            (ExpertType.GENERATION, 'motion'),
            (ExpertType.GENERATION, 'appearance'),
            (ExpertType.PERCEPTION, 'temporal'),
            (ExpertType.REASONING, 'semantic'),
            (ExpertType.PHYSICAL, 'lighting'),
            (ExpertType.GENERATION, 'texture'),
            (ExpertType.PERCEPTION, 'spatial'),
        ]
        
        for i, (exp_type, exp_subtype) in enumerate(expert_configs[:self.num_experts]):
            if exp_type == ExpertType.PHYSICAL:
                expert = PhysicalExpert(i)
                expert.physics_params['subtype'] = exp_subtype
            elif exp_type == ExpertType.GENERATION:
                expert = GenerationExpert(i)
            elif exp_type == ExpertType.PERCEPTION:
                expert = PerceptionExpert(i)
            else:
                expert = PhysicalExpert(i)
            
            expert.expert_type = exp_type
            self.unified_moe.add_expert(expert)
    
    def process_frame(self, frame_data: Any, frame_idx: int = 0) -> Dict[str, Any]:
        """
        处理单个帧
        
        Args:
            frame_data: 帧数据
            frame_idx: 帧索引
            
        Returns:
            处理结果，包含：
            - output: 输出数据
            - expert_ids: 使用的专家ID
            - weights: 专家权重
            - confidence: 置信度
        """
        # 使用统一MoE处理
        expert_output = self.unified_moe.process(frame_data, frame_idx)
        
        # 获取路由信息
        routing_info = self.unified_moe.route(frame_data, frame_idx)
        
        return {
            'output': expert_output.data,
            'expert_ids': routing_info.expert_indices,
            'weights': routing_info.gate_weights,
            'confidence': expert_output.confidence,
            'load_balance_loss': routing_info.load_balance_loss
        }
    
    def process_frame_batch(self, frames: List[Any]) -> List[Dict[str, Any]]:
        """
        批量处理帧
        
        Args:
            frames: 帧数据列表
            
        Returns:
            处理结果列表
        """
        results = []
        for i, frame in enumerate(frames):
            result = self.process_frame(frame, i)
            results.append(result)
        return results
    
    def get_expert_by_type(self, expert_type: ExpertType) -> List[Expert]:
        """
        获取指定类型的专家
        
        Args:
            expert_type: 专家类型
            
        Returns:
            专家列表
        """
        return self.unified_moe.get_expert_by_type(expert_type)
    
    def add_custom_expert(self, expert: Expert) -> None:
        """
        添加自定义专家
        
        Args:
            expert: 专家实例
        """
        self.unified_moe.add_expert(expert)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息
        
        Returns:
            统计信息字典
        """
        moe_stats = self.unified_moe.get_stats()
        
        return {
            'total_tokens': moe_stats.total_tokens,
            'total_experts': moe_stats.total_experts,
            'active_experts': moe_stats.active_experts,
            'avg_experts_per_token': moe_stats.avg_experts_per_token,
            'expert_loads': moe_stats.expert_loads
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.unified_moe.reset_stats()


# ============================================================================
# 便捷函数
# ============================================================================

def create_video_unified_system(config: Optional[UnifiedAlgorithmConfig] = None) -> Dict[str, Any]:
    """
    创建视频生成的统一系统
    
    Args:
        config: 算法配置
        
    Returns:
        包含所有适配器的字典
    """
    cfg = config or UnifiedAlgorithmConfig.video_optimized_config()
    
    return {
        'config': cfg,
        'memory': VideoMemoryAdapter(config=cfg),
        'attention': VideoAttentionAdapter(config=cfg),
        'chunker': VideoChunkerAdapter(config=cfg),
        'moe': VideoMoEAdapter(config=cfg),
    }


# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 数据结构
    'VideoFrame',
    'SpatioTemporalQuery',
    
    # 适配器
    'VideoMemoryAdapter',
    'VideoAttentionAdapter',
    'VideoChunkerAdapter',
    'VideoMoEAdapter',
    
    # 便捷函数
    'create_video_unified_system',
]
