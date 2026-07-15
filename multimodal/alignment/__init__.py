"""
模态对齐模块
提供对比学习和模态对齐功能
"""
from .contrastive_trainer import (
    ContrastiveTrainer,
    MultiModalContrastiveTrainer,
    create_contrastive_trainer
)

from .siglip_trainer import (
    SigLIPTrainer,
    SigLIPWithHardNegatives,
    create_siglip_trainer
)

from .alignment_loss import (
    AlignmentLoss,
    ContrastiveLoss,
    SigmoidLoss,
    TripletLoss,
    CosineEmbeddingLoss,
    MSELoss,
    CombinedAlignmentLoss,
    MultiModalAlignmentLoss,
    create_contrastive_loss,
    create_sigmoid_loss,
    create_triplet_loss
)

__all__ = [
    # Contrastive Trainer
    'ContrastiveTrainer',
    'MultiModalContrastiveTrainer',
    'create_contrastive_trainer',
    
    # SigLIP Trainer
    'SigLIPTrainer',
    'SigLIPWithHardNegatives',
    'create_siglip_trainer',
    
    # Alignment Loss
    'AlignmentLoss',
    'ContrastiveLoss',
    'SigmoidLoss',
    'TripletLoss',
    'CosineEmbeddingLoss',
    'MSELoss',
    'CombinedAlignmentLoss',
    'MultiModalAlignmentLoss',
    'create_contrastive_loss',
    'create_sigmoid_loss',
    'create_triplet_loss'
]
