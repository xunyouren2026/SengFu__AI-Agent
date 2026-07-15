"""
多模态融合模块
提供跨模态交互和融合功能
"""
from .cross_attention import (
    CrossAttention,
    CrossAttentionBlock,
    BidirectionalCrossAttention,
    MultiModalCrossAttention,
    GatedCrossAttention,
    LayerNorm,
    create_cross_attention,
    create_bidirectional_cross_attention
)

from .adapters import (
    MLPAdapter,
    ResidualAdapter,
    QFormerAdapter,
    ModalityAdapter,
    SequentialAdapter,
    ParallelAdapter,
    create_mlp_adapter,
    create_residual_adapter,
    create_qformer_adapter
)

from .perceiver import (
    PerceiverResampler,
    PerceiverEncoder,
    PerceiverDecoder,
    MultiModalPerceiver,
    CrossAttention,
    SelfAttention,
    MLP,
    create_perceiver_resampler
)

from .late_fusion import (
    LateFusion,
    ConcatenationFusion,
    WeightedFusion,
    AttentionFusion,
    GatedFusion,
    TensorFusion,
    HierarchicalFusion,
    EnsembleFusion,
    create_attention_fusion,
    create_gated_fusion
)

__all__ = [
    # Cross Attention
    'CrossAttention',
    'CrossAttentionBlock',
    'BidirectionalCrossAttention',
    'MultiModalCrossAttention',
    'GatedCrossAttention',
    'LayerNorm',
    'create_cross_attention',
    'create_bidirectional_cross_attention',
    
    # Adapters
    'MLPAdapter',
    'ResidualAdapter',
    'QFormerAdapter',
    'ModalityAdapter',
    'SequentialAdapter',
    'ParallelAdapter',
    'create_mlp_adapter',
    'create_residual_adapter',
    'create_qformer_adapter',
    
    # Perceiver
    'PerceiverResampler',
    'PerceiverEncoder',
    'PerceiverDecoder',
    'MultiModalPerceiver',
    'SelfAttention',
    'MLP',
    'create_perceiver_resampler',
    
    # Late Fusion
    'LateFusion',
    'ConcatenationFusion',
    'WeightedFusion',
    'AttentionFusion',
    'GatedFusion',
    'TensorFusion',
    'HierarchicalFusion',
    'EnsembleFusion',
    'create_attention_fusion',
    'create_gated_fusion'
]
