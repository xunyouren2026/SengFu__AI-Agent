"""
Video Generation Training Module

视频生成训练组件。

主要组件：
- VideoTrainer: 视频训练器
- ConstraintAwareLoss: 约束感知损失
- MoETrainer: MoE训练辅助
"""

from agi_unified_framework.video_gen.training.trainer import (
    TrainingConfig,
    VideoTrainer,
    ConstraintAwareLoss,
    MoETrainer,
    create_video_trainer,
    train_model,
)

__all__ = [
    'TrainingConfig',
    'VideoTrainer',
    'ConstraintAwareLoss',
    'MoETrainer',
    'create_video_trainer',
    'train_model',
]
