"""
视频处理模块
提供视频帧采样和时序编码功能
"""
from .frame_sampler import (
    FrameSampler,
    UniformSampler,
    KeyFrameSampler,
    SceneDetectionSampler,
    AdaptiveSampler,
    MultiScaleSampler,
    create_frame_sampler
)

from .temporal_encoder import (
    TemporalEncoder,
    VideoEncoder,
    TemporalAttention,
    TemporalAttentionBlock,
    TemporalConvBlock,
    Conv3D,
    LayerNorm,
    create_temporal_encoder,
    create_video_encoder
)

__all__ = [
    # Frame Sampler
    'FrameSampler',
    'UniformSampler',
    'KeyFrameSampler',
    'SceneDetectionSampler',
    'AdaptiveSampler',
    'MultiScaleSampler',
    'create_frame_sampler',
    
    # Temporal Encoder
    'TemporalEncoder',
    'VideoEncoder',
    'TemporalAttention',
    'TemporalAttentionBlock',
    'TemporalConvBlock',
    'Conv3D',
    'LayerNorm',
    'create_temporal_encoder',
    'create_video_encoder'
]
