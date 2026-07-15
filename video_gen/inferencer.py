"""
Video Generation Inferencer Module

使用统一核心进行视频生成推理。
generate_long使用UnifiedChunker进行时间分块，
使用UnifiedBoundaryDetector检测镜头边界，
使用UnifiedOverlapFusion进行块间融合。

主要组件：
- VideoInferencer: 视频推理器
- LongVideoGenerator: 长视频生成器
- InferenceConfig: 推理配置
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Iterator
import math

# 导入统一核心算法
from agi_unified_framework.core.unified_algorithms import (
    UnifiedAlgorithmConfig,
    UnifiedChunker,
    UnifiedOverlapFusion,
    UnifiedBoundaryDetector,
    ChunkingStrategy,
    BoundaryType,
    Chunk,
)

from agi_unified_framework.video_gen.unified_adapter import (
    VideoChunkerAdapter,
    VideoMemoryAdapter,
    VideoAttentionAdapter,
    VideoMoEAdapter,
    create_video_unified_system,
)

from agi_unified_framework.video_gen.models.dit import (
    DiTModel,
    DiTConfig,
    create_dit_model,
)

from agi_unified_framework.video_gen.models.memory_bank import (
    MemoryBank,
    FrameBuffer,
)


# ============================================================================
# 推理配置
# ============================================================================

@dataclass
class InferenceConfig:
    """
    推理配置
    
    Attributes:
        # 生成参数
        num_frames: 生成帧数
        fps: 帧率
        resolution: 分辨率
        
        # 长视频生成参数
        chunk_size: 分块大小
        chunk_overlap: 块重叠大小
        use_boundary_detection: 是否使用边界检测
        use_overlap_fusion: 是否使用重叠融合
        
        # 统一核心配置
        use_unified_core: 是否使用统一核心
        unified_config: 统一算法配置
        
        # 推理参数
        num_inference_steps: 推理步数
        guidance_scale: 引导缩放
    """
    
    # 生成参数
    num_frames: int = 16
    fps: int = 8
    resolution: Tuple[int, int] = (512, 512)
    
    # 长视频生成参数
    chunk_size: int = 16
    chunk_overlap: int = 4
    use_boundary_detection: bool = True
    use_overlap_fusion: bool = True
    
    # 统一核心配置
    use_unified_core: bool = True
    unified_config: Optional[UnifiedAlgorithmConfig] = None
    
    # 推理参数
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    def __post_init__(self):
        """初始化后处理"""
        if self.unified_config is None:
            self.unified_config = UnifiedAlgorithmConfig.video_optimized_config()


# ============================================================================
# 视频推理器
# ============================================================================

class VideoInferencer:
    """
    视频推理器
    
    使用统一核心进行视频生成推理。
    
    Attributes:
        model: DiT模型
        config: 推理配置
        memory_bank: 记忆库
        frame_buffer: 帧缓冲区
        chunker_adapter: 视频分块适配器
    """
    
    def __init__(self,
                 model: Optional[DiTModel] = None,
                 config: Optional[InferenceConfig] = None):
        """
        初始化视频推理器
        
        Args:
            model: DiT模型
            config: 推理配置
        """
        self.config = config or InferenceConfig()
        
        # 初始化模型
        if model is None:
            self.model = create_dit_model(
                use_unified_core=self.config.use_unified_core
            )
        else:
            self.model = model
        
        # 初始化记忆库
        self.memory_bank = MemoryBank(
            capacity=10000,
            use_unified_core=self.config.use_unified_core
        )
        
        # 初始化帧缓冲区
        self.frame_buffer = FrameBuffer(max_size=self.config.chunk_size * 2)
        
        # 初始化分块适配器
        if self.config.use_unified_core:
            self.chunker_adapter = VideoChunkerAdapter(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                use_boundary_detection=self.config.use_boundary_detection,
                config=self.config.unified_config
            )
        else:
            self.chunker_adapter = None
    
    def generate(self,
                 prompt: str,
                 num_frames: Optional[int] = None,
                 **kwargs) -> List[Any]:
        """
        生成短视频
        
        Args:
            prompt: 文本提示
            num_frames: 帧数
            **kwargs: 其他参数
            
        Returns:
            生成的帧列表
        """
        n_frames = num_frames or self.config.num_frames
        
        # 使用模型生成
        frames = self.model.generate(
            shape=(n_frames,),
            steps=self.config.num_inference_steps,
            cfg_scale=self.config.guidance_scale
        )
        
        # 转换为列表格式
        if not isinstance(frames, list):
            frames = [frames]
        
        # 存储到记忆库
        for i, frame in enumerate(frames):
            self.memory_bank.store(
                frame_data=frame,
                frame_idx=i,
                importance=0.5,
                metadata={'prompt': prompt, 'generated': True}
            )
            self.frame_buffer.push(frame, i)
        
        return frames
    
    def generate_long(self,
                      prompt: str,
                      total_frames: int,
                      **kwargs) -> List[Any]:
        """
        生成长视频
        
        使用UnifiedChunker进行时间分块，
        使用UnifiedBoundaryDetector检测镜头边界，
        使用UnifiedOverlapFusion进行块间融合。
        
        Args:
            prompt: 文本提示
            total_frames: 总帧数
            **kwargs: 其他参数
            
        Returns:
            生成的帧列表
        """
        if not self.config.use_unified_core or self.chunker_adapter is None:
            # 回退到简单生成
            return self.generate(prompt, total_frames, **kwargs)
        
        # 步骤1: 将总帧数分块
        dummy_frames = list(range(total_frames))  # 用于分块的虚拟帧
        chunks = self.chunker_adapter.chunk_video(dummy_frames)
        
        print(f"生成长视频: {total_frames}帧, 分为{len(chunks)}个块")
        
        # 步骤2: 检测镜头边界（可选）
        if self.config.use_boundary_detection:
            boundaries = self._detect_boundaries_for_generation(chunks)
            print(f"检测到{len(boundaries)}个镜头边界")
        
        # 步骤3: 逐块生成
        generated_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_frames = self._generate_chunk(
                prompt=prompt,
                chunk_idx=i,
                start_idx=chunk['start_idx'],
                end_idx=chunk['end_idx'],
                is_keyframe_chunk=chunk.get('is_keyframe_chunk', False)
            )
            
            generated_chunks.append({
                'frames': chunk_frames,
                'start_idx': chunk['start_idx'],
                'end_idx': chunk['end_idx'],
                'metadata': chunk.get('metadata', {})
            })
            
            print(f"  块 {i+1}/{len(chunks)} 生成完成: {len(chunk_frames)}帧")
        
        # 步骤4: 使用UnifiedOverlapFusion进行块间融合
        if self.config.use_overlap_fusion and len(generated_chunks) > 1:
            final_frames = self._fuse_chunks(generated_chunks)
            print(f"融合完成: {len(final_frames)}帧")
        else:
            # 简单拼接
            final_frames = []
            for chunk in generated_chunks:
                final_frames.extend(chunk['frames'])
        
        # 步骤5: 存储到记忆库
        for i, frame in enumerate(final_frames):
            self.memory_bank.store(
                frame_data=frame,
                frame_idx=i,
                importance=0.5,
                metadata={
                    'prompt': prompt,
                    'generated': True,
                    'long_video': True,
                    'chunk_idx': i // self.config.chunk_size
                }
            )
        
        return final_frames
    
    def _generate_chunk(self,
                        prompt: str,
                        chunk_idx: int,
                        start_idx: int,
                        end_idx: int,
                        is_keyframe_chunk: bool = False) -> List[Any]:
        """
        生成单个块
        
        Args:
            prompt: 文本提示
            chunk_idx: 块索引
            start_idx: 起始帧索引
            end_idx: 结束帧索引
            is_keyframe_chunk: 是否是关键帧块
            
        Returns:
            生成的帧列表
        """
        chunk_size = end_idx - start_idx
        
        # 获取上下文（从前一块的末尾）
        context_frames = []
        if chunk_idx > 0:
            context_frames = self.frame_buffer.get_recent(
                n=min(self.config.chunk_overlap, self.frame_buffer.size())
            )
        
        # 生成块
        # 注意：实际实现中应该使用上下文进行条件生成
        frames = self.model.generate(
            shape=(chunk_size,),
            steps=self.config.num_inference_steps,
            cfg_scale=self.config.guidance_scale
        )
        
        if not isinstance(frames, list):
            frames = [frames] * chunk_size
        
        # 确保帧数正确
        while len(frames) < chunk_size:
            frames.append(frames[-1] if frames else None)
        frames = frames[:chunk_size]
        
        # 更新帧缓冲区
        for i, frame in enumerate(frames):
            global_idx = start_idx + i
            self.frame_buffer.push(frame, global_idx)
        
        return frames
    
    def _detect_boundaries_for_generation(self, 
                                          chunks: List[Dict[str, Any]]) -> List[int]:
        """
        为生成检测边界
        
        Args:
            chunks: 块列表
            
        Returns:
            边界位置列表
        """
        if self.chunker_adapter is None:
            return []
        
        # 使用分块适配器的边界检测
        # 注意：这里使用虚拟数据进行边界检测
        dummy_frames = list(range(sum(len(c.get('frames', [])) or 
                                      (c['end_idx'] - c['start_idx']) 
                                      for c in chunks)))
        
        return self.chunker_adapter.detect_shot_boundaries(dummy_frames)
    
    def _fuse_chunks(self, chunks: List[Dict[str, Any]]) -> List[Any]:
        """
        融合块
        
        使用UnifiedOverlapFusion进行块间融合。
        
        Args:
            chunks: 块列表
            
        Returns:
            融合后的帧列表
        """
        if self.chunker_adapter is None or len(chunks) <= 1:
            # 简单拼接
            result = []
            for chunk in chunks:
                result.extend(chunk['frames'])
            return result
        
        # 使用分块适配器的融合功能
        fused_chunks = self.chunker_adapter.fuse_chunk_boundaries(chunks)
        
        # 提取帧
        final_frames = []
        for chunk in fused_chunks:
            final_frames.extend(chunk['frames'])
        
        return final_frames
    
    def continue_generation(self,
                           prompt: str,
                           additional_frames: int,
                           **kwargs) -> List[Any]:
        """
        继续生成（在已有视频基础上）
        
        Args:
            prompt: 文本提示
            additional_frames: 额外帧数
            **kwargs: 其他参数
            
        Returns:
            新生成的帧列表
        """
        # 获取最近的帧作为上下文
        recent_frames = self.frame_buffer.get_recent(n=self.config.chunk_overlap)
        
        # 计算新的起始索引
        current_size = self.memory_bank.size()
        
        # 生成新帧
        new_frames = self.model.generate(
            shape=(additional_frames,),
            steps=self.config.num_inference_steps,
            cfg_scale=self.config.guidance_scale
        )
        
        if not isinstance(new_frames, list):
            new_frames = [new_frames] * additional_frames
        
        # 存储新帧
        for i, frame in enumerate(new_frames):
            global_idx = current_size + i
            self.memory_bank.store(
                frame_data=frame,
                frame_idx=global_idx,
                importance=0.5,
                metadata={'prompt': prompt, 'continued': True}
            )
            self.frame_buffer.push(frame, global_idx)
        
        return new_frames
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        获取生成统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_frames_generated': self.memory_bank.size(),
            'frame_buffer_size': self.frame_buffer.size(),
            'use_unified_core': self.config.use_unified_core,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'memory_stats': self.memory_bank.get_stats()
        }


# ============================================================================
# 长视频生成器
# ============================================================================

class LongVideoGenerator:
    """
    长视频生成器
    
    专门用于生成超长视频，支持流式生成。
    
    Attributes:
        inferencer: 视频推理器
        config: 推理配置
    """
    
    def __init__(self,
                 inferencer: Optional[VideoInferencer] = None,
                 config: Optional[InferenceConfig] = None):
        """
        初始化长视频生成器
        
        Args:
            inferencer: 视频推理器
            config: 推理配置
        """
        self.config = config or InferenceConfig()
        
        if inferencer is None:
            self.inferencer = VideoInferencer(config=self.config)
        else:
            self.inferencer = inferencer
    
    def generate_streaming(self,
                          prompt: str,
                          total_frames: int,
                          **kwargs) -> Iterator[List[Any]]:
        """
        流式生成长视频
        
        逐块生成并返回，适用于超长视频。
        
        Args:
            prompt: 文本提示
            total_frames: 总帧数
            **kwargs: 其他参数
            
        Yields:
            每块生成的帧列表
        """
        if not self.config.use_unified_core:
            # 回退到一次性生成
            frames = self.inferencer.generate_long(prompt, total_frames, **kwargs)
            yield frames
            return
        
        # 分块
        dummy_frames = list(range(total_frames))
        chunks = self.inferencer.chunker_adapter.chunk_video(dummy_frames)
        
        print(f"流式生成: {total_frames}帧, {len(chunks)}个块")
        
        for i, chunk in enumerate(chunks):
            chunk_frames = self.inferencer._generate_chunk(
                prompt=prompt,
                chunk_idx=i,
                start_idx=chunk['start_idx'],
                end_idx=chunk['end_idx'],
                is_keyframe_chunk=chunk.get('is_keyframe_chunk', False)
            )
            
            yield chunk_frames
    
    def generate_with_keyframes(self,
                                prompt: str,
                                keyframe_indices: List[int],
                                **kwargs) -> List[Any]:
        """
        基于关键帧生成
        
        在指定位置生成关键帧，然后插值填充。
        
        Args:
            prompt: 文本提示
            keyframe_indices: 关键帧索引列表
            **kwargs: 其他参数
            
        Returns:
            生成的帧列表
        """
        if not keyframe_indices:
            return []
        
        total_frames = keyframe_indices[-1] + 1
        frames = [None] * total_frames
        
        # 生成关键帧
        for idx in keyframe_indices:
            # 生成单个关键帧
            frame = self.inferencer.model.generate(
                shape=(1,),
                steps=self.config.num_inference_steps,
                cfg_scale=self.config.guidance_scale
            )
            frames[idx] = frame[0] if isinstance(frame, list) else frame
        
        # 插值填充（简化实现）
        for i in range(total_frames):
            if frames[i] is None:
                # 找到最近的关键帧进行插值
                frames[i] = self._interpolate_frame(frames, i, keyframe_indices)
        
        # 存储到记忆库
        for i, frame in enumerate(frames):
            self.inferencer.memory_bank.store(
                frame_data=frame,
                frame_idx=i,
                importance=1.0 if i in keyframe_indices else 0.3,
                metadata={
                    'prompt': prompt,
                    'is_keyframe': i in keyframe_indices
                }
            )
        
        return frames
    
    def _interpolate_frame(self,
                          frames: List[Any],
                          target_idx: int,
                          keyframe_indices: List[int]) -> Any:
        """
        插值帧
        
        Args:
            frames: 帧列表
            target_idx: 目标索引
            keyframe_indices: 关键帧索引
            
        Returns:
            插值后的帧
        """
        # 找到最近的前后关键帧
        prev_idx = None
        next_idx = None
        
        for idx in keyframe_indices:
            if idx < target_idx:
                prev_idx = idx
            elif idx > target_idx and next_idx is None:
                next_idx = idx
                break
        
        # 简单的插值
        if prev_idx is not None and frames[prev_idx] is not None:
            return frames[prev_idx]
        elif next_idx is not None and frames[next_idx] is not None:
            return frames[next_idx]
        else:
            # 返回默认值
            return 0.0


# ============================================================================
# 便捷函数
# ============================================================================

def create_video_inferencer(use_unified_core: bool = True,
                            **kwargs) -> VideoInferencer:
    """
    创建视频推理器
    
    Args:
        use_unified_core: 是否使用统一核心
        **kwargs: 其他配置参数
        
    Returns:
        视频推理器
    """
    config = InferenceConfig(use_unified_core=use_unified_core, **kwargs)
    return VideoInferencer(config=config)


def create_long_video_generator(use_unified_core: bool = True,
                                **kwargs) -> LongVideoGenerator:
    """
    创建长视频生成器
    
    Args:
        use_unified_core: 是否使用统一核心
        **kwargs: 其他配置参数
        
    Returns:
        长视频生成器
    """
    config = InferenceConfig(use_unified_core=use_unified_core, **kwargs)
    return LongVideoGenerator(config=config)


def generate_video(prompt: str,
                   num_frames: int = 16,
                   use_unified_core: bool = True,
                   **kwargs) -> List[Any]:
    """
    便捷函数：生成视频
    
    Args:
        prompt: 文本提示
        num_frames: 帧数
        use_unified_core: 是否使用统一核心
        **kwargs: 其他参数
        
    Returns:
        生成的帧列表
    """
    inferencer = create_video_inferencer(use_unified_core=use_unified_core)
    return inferencer.generate(prompt, num_frames, **kwargs)


def generate_long_video(prompt: str,
                        total_frames: int,
                        use_unified_core: bool = True,
                        **kwargs) -> List[Any]:
    """
    便捷函数：生成长视频
    
    Args:
        prompt: 文本提示
        total_frames: 总帧数
        use_unified_core: 是否使用统一核心
        **kwargs: 其他参数
        
    Returns:
        生成的帧列表
    """
    inferencer = create_video_inferencer(use_unified_core=use_unified_core)
    return inferencer.generate_long(prompt, total_frames, **kwargs)


# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 配置
    'InferenceConfig',
    
    # 推理器
    'VideoInferencer',
    'LongVideoGenerator',
    
    # 便捷函数
    'create_video_inferencer',
    'create_long_video_generator',
    'generate_video',
    'generate_long_video',
]
