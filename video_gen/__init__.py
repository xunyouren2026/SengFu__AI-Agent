"""
Video Generation Module

使用统一核心算法的视频生成模块。

主要组件：
- unified_adapter: 统一核心适配器
- models: 模型组件（DiT、MemoryBank等）
- inferencer: 推理器
- training: 训练器
- postprocess: 后处理

使用示例：
    >>> from agi_unified_framework.video_gen import VideoInferencer, DiTModel
    >>> model = DiTModel(use_unified_core=True)
    >>> inferencer = VideoInferencer(model)
    >>> frames = inferencer.generate("a cat playing", num_frames=16)
"""

from __future__ import annotations

# 导入适配器
from agi_unified_framework.video_gen.unified_adapter import (
    VideoMemoryAdapter,
    VideoAttentionAdapter,
    VideoChunkerAdapter,
    VideoMoEAdapter,
    VideoFrame,
    SpatioTemporalQuery,
    create_video_unified_system,
)

# 导入模型
from agi_unified_framework.video_gen.models.dit import (
    DiTConfig,
    DiTBlock,
    SpatialTemporalUNet,
    DiTModel,
    create_dit_model,
    create_spatial_temporal_unet,
)

from agi_unified_framework.video_gen.models.memory_bank import (
    FrameData,
    MemoryQueryResult,
    MemoryBank,
    FrameBuffer,
    TemporalMemory,
    create_memory_bank,
    create_frame_buffer,
    create_temporal_memory,
)

# 导入推理器
from agi_unified_framework.video_gen.inferencer import (
    InferenceConfig,
    VideoInferencer,
    LongVideoGenerator,
    create_video_inferencer,
    create_long_video_generator,
    generate_video,
    generate_long_video,
)

# 导入训练器
from agi_unified_framework.video_gen.training.trainer import (
    TrainingConfig,
    VideoTrainer,
    ConstraintAwareLoss,
    MoETrainer,
    create_video_trainer,
    train_model,
)

# 导入后处理
from agi_unified_framework.video_gen.postprocess.physics_corrector import (
    MotionVector,
    ObjectState,
    CorrectionResult,
    PhysicsCorrector,
    MotionSmoother,
    ConstraintValidator,
    create_physics_corrector,
    correct_video_physics,
    validate_video_physics,
)


# ============================================================================
# 版本信息
# ============================================================================

__version__ = "1.0.0"
__author__ = "AGI Unified Framework"


# ============================================================================
# 便捷入口函数
# ============================================================================

def create_video_generation_pipeline(
    use_unified_core: bool = True,
    use_constraints: bool = True,
    use_moe: bool = True,
    **kwargs
) -> dict:
    """
    创建完整的视频生成流水线
    
    Args:
        use_unified_core: 是否使用统一核心
        use_constraints: 是否使用约束系统
        use_moe: 是否使用MoE
        **kwargs: 其他配置参数
        
    Returns:
        包含所有组件的字典
    """
    # 创建模型
    model = create_dit_model(
        use_unified_core=use_unified_core,
        use_moe=use_moe,
        **kwargs
    )
    
    # 创建推理器
    inferencer = create_video_inferencer(
        use_unified_core=use_unified_core,
        **kwargs
    )
    inferencer.model = model
    
    # 创建训练器
    trainer = create_video_trainer(
        use_unified_core=use_unified_core,
        use_constraints=use_constraints,
        use_moe=use_moe,
        **kwargs
    )
    trainer.model = model
    
    # 创建物理校正器
    corrector = create_physics_corrector(
        use_unified_core=use_unified_core,
        **kwargs
    )
    
    return {
        'model': model,
        'inferencer': inferencer,
        'trainer': trainer,
        'corrector': corrector,
        'config': {
            'use_unified_core': use_unified_core,
            'use_constraints': use_constraints,
            'use_moe': use_moe,
        }
    }


# ============================================================================
# 导出列表
# ============================================================================

__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    
    # 适配器
    'VideoMemoryAdapter',
    'VideoAttentionAdapter',
    'VideoChunkerAdapter',
    'VideoMoEAdapter',
    'VideoFrame',
    'SpatioTemporalQuery',
    'create_video_unified_system',
    
    # 模型
    'DiTConfig',
    'DiTBlock',
    'SpatialTemporalUNet',
    'DiTModel',
    'create_dit_model',
    'create_spatial_temporal_unet',
    
    # 记忆库
    'FrameData',
    'MemoryQueryResult',
    'MemoryBank',
    'FrameBuffer',
    'TemporalMemory',
    'create_memory_bank',
    'create_frame_buffer',
    'create_temporal_memory',
    
    # 推理器
    'InferenceConfig',
    'VideoInferencer',
    'LongVideoGenerator',
    'create_video_inferencer',
    'create_long_video_generator',
    'generate_video',
    'generate_long_video',
    
    # 训练器
    'TrainingConfig',
    'VideoTrainer',
    'ConstraintAwareLoss',
    'MoETrainer',
    'create_video_trainer',
    'train_model',
    
    # 后处理
    'MotionVector',
    'ObjectState',
    'CorrectionResult',
    'PhysicsCorrector',
    'MotionSmoother',
    'ConstraintValidator',
    'create_physics_corrector',
    'correct_video_physics',
    'validate_video_physics',
    
    # 便捷函数
    'create_video_generation_pipeline',
]
