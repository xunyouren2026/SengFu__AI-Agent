"""
投影模块
提供特征投影和变换功能
"""
from .q_former import (
    QFormer,
    QFormerLayer,
    QFormerWithText,
    CrossAttention,
    SelfAttention,
    MLP,
    LayerNorm,
    create_qformer
)

from .visual_projector import (
    VisualProjector,
    MultiLayerVisualProjector,
    AdaptiveVisualProjector,
    LinearProjection,
    MLPProjection,
    ResidualProjection,
    create_visual_projector
)

__all__ = [
    # Q-Former
    'QFormer',
    'QFormerLayer',
    'QFormerWithText',
    'CrossAttention',
    'SelfAttention',
    'MLP',
    'LayerNorm',
    'create_qformer',
    
    # Visual Projector
    'VisualProjector',
    'MultiLayerVisualProjector',
    'AdaptiveVisualProjector',
    'LinearProjection',
    'MLPProjection',
    'ResidualProjection',
    'create_visual_projector'
]
