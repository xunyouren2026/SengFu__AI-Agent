"""
Video Generation Post-processing Module

视频生成后处理组件。

主要组件：
- PhysicsCorrector: 物理校正器
- MotionSmoother: 运动平滑器
- ConstraintValidator: 约束验证器
"""

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

__all__ = [
    'MotionVector',
    'ObjectState',
    'CorrectionResult',
    'PhysicsCorrector',
    'MotionSmoother',
    'ConstraintValidator',
    'create_physics_corrector',
    'correct_video_physics',
    'validate_video_physics',
]
