"""
多模态模块
提供完整的多模态处理能力
"""

# 导入各子模块
from .vision import (
    CLIPVisionEncoder,
    CLIPImagePreprocessor,
    DINOv2Encoder,
    DINOv2Preprocessor,
    SigLIPVisionEncoder,
    SigLIPPreprocessor,
    create_clip_vision_encoder,
    create_dino_v2_encoder,
    create_siglip_encoder
)

from .audio import (
    WhisperEncoder,
    MelSpectrogram,
    CLAPEncoder,
    CLAPAudioEncoder,
    CLAPTextEncoder,
    create_whisper_encoder,
    create_clap_encoder
)

from .fusion import (
    CrossAttention,
    CrossAttentionBlock,
    BidirectionalCrossAttention,
    MultiModalCrossAttention,
    MLPAdapter,
    ResidualAdapter,
    QFormerAdapter,
    PerceiverResampler,
    MultiModalPerceiver,
    AttentionFusion,
    GatedFusion,
    ConcatenationFusion,
    WeightedFusion,
    create_cross_attention,
    create_mlp_adapter,
    create_perceiver_resampler,
    create_attention_fusion
)

from .video import (
    FrameSampler,
    UniformSampler,
    KeyFrameSampler,
    SceneDetectionSampler,
    TemporalEncoder,
    VideoEncoder,
    create_frame_sampler,
    create_video_encoder
)

from .alignment import (
    ContrastiveTrainer,
    SigLIPTrainer,
    ContrastiveLoss,
    SigmoidLoss,
    TripletLoss,
    MultiModalAlignmentLoss,
    create_contrastive_trainer,
    create_siglip_trainer
)

from .projection import (
    QFormer,
    QFormerWithText,
    VisualProjector,
    MultiLayerVisualProjector,
    AdaptiveVisualProjector,
    create_qformer,
    create_visual_projector
)

from .retrieval import (
    MultiModalIndex,
    FlatIndex,
    IVFIndex,
    SimpleCrossModalRetriever,
    WeightedCrossModalRetriever,
    HybridCrossModalRetriever,
    create_vector_index,
    create_cross_modal_retriever
)

from .multimodal_processor import (
    MultimodalProcessor,
    MultimodalPipeline,
    ImageProcessor,
    TextProcessor,
    AudioProcessor,
    VideoProcessor,
    create_multimodal_processor
)

from .embedding_cache import (
    EmbeddingCache,
    MultiLevelCache,
    PersistentCache,
    BatchCache,
    create_embedding_cache
)

__all__ = [
    # Vision
    'CLIPVisionEncoder',
    'CLIPImagePreprocessor',
    'DINOv2Encoder',
    'DINOv2Preprocessor',
    'SigLIPVisionEncoder',
    'SigLIPPreprocessor',
    'create_clip_vision_encoder',
    'create_dino_v2_encoder',
    'create_siglip_encoder',
    
    # Audio
    'WhisperEncoder',
    'MelSpectrogram',
    'CLAPEncoder',
    'CLAPAudioEncoder',
    'CLAPTextEncoder',
    'create_whisper_encoder',
    'create_clap_encoder',
    
    # Fusion
    'CrossAttention',
    'CrossAttentionBlock',
    'BidirectionalCrossAttention',
    'MultiModalCrossAttention',
    'MLPAdapter',
    'ResidualAdapter',
    'QFormerAdapter',
    'PerceiverResampler',
    'MultiModalPerceiver',
    'AttentionFusion',
    'GatedFusion',
    'ConcatenationFusion',
    'WeightedFusion',
    'create_cross_attention',
    'create_mlp_adapter',
    'create_perceiver_resampler',
    'create_attention_fusion',
    
    # Video
    'FrameSampler',
    'UniformSampler',
    'KeyFrameSampler',
    'SceneDetectionSampler',
    'TemporalEncoder',
    'VideoEncoder',
    'create_frame_sampler',
    'create_video_encoder',
    
    # Alignment
    'ContrastiveTrainer',
    'SigLIPTrainer',
    'ContrastiveLoss',
    'SigmoidLoss',
    'TripletLoss',
    'MultiModalAlignmentLoss',
    'create_contrastive_trainer',
    'create_siglip_trainer',
    
    # Projection
    'QFormer',
    'QFormerWithText',
    'VisualProjector',
    'MultiLayerVisualProjector',
    'AdaptiveVisualProjector',
    'create_qformer',
    'create_visual_projector',
    
    # Retrieval
    'MultiModalIndex',
    'FlatIndex',
    'IVFIndex',
    'SimpleCrossModalRetriever',
    'WeightedCrossModalRetriever',
    'HybridCrossModalRetriever',
    'create_vector_index',
    'create_cross_modal_retriever',
    
    # Processor
    'MultimodalProcessor',
    'MultimodalPipeline',
    'ImageProcessor',
    'TextProcessor',
    'AudioProcessor',
    'VideoProcessor',
    'create_multimodal_processor',
    
    # Cache
    'EmbeddingCache',
    'MultiLevelCache',
    'PersistentCache',
    'BatchCache',
    'create_embedding_cache'
]
