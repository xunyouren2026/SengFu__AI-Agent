"""
Video Generation - 视频生成模块
包含AnimateDiff和SVD视频生成管线。
"""

from .animate_diff import (
    AnimateDiffPipeline,
    MotionModule,
    SVDPipeline as AnimateDiffSVD,
    GenerationResult as AnimateDiffResult,
)
from .svd_pipe import (
    SVDPipeline,
    FrameInterpolator,
    TemporalConsistency,
    MotionController,
    ConditioningModule,
    VideoDecoder,
    SVDConfig,
    NoiseScheduler,
    VideoOutput,
    VideoFrame,
    ImageInput,
    LatentFrame,
    NoiseScheduleType,
)

__all__ = [
    "AnimateDiffPipeline",
    "MotionModule",
    "AnimateDiffSVD",
    "AnimateDiffResult",
    "SVDPipeline",
    "FrameInterpolator",
    "TemporalConsistency",
    "MotionController",
    "ConditioningModule",
    "VideoDecoder",
    "SVDConfig",
    "NoiseScheduler",
    "VideoOutput",
    "VideoFrame",
    "ImageInput",
    "LatentFrame",
    "NoiseScheduleType",
]
