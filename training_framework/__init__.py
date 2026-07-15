"""
AGI Unified Framework - Training Framework Module

Provides training infrastructure including trainers, optimizers,
loss functions, and data loading utilities.
"""

from .trainer import (
    TrainerConfig,
    TrainingState,
    TrainingCallback,
    BaseTrainer,
    EarlyStopping,
    GradientAccumulator,
    CheckpointManager,
)
from .optimizers import (
    Optimizer,
    SGDOptimizer,
    AdamOptimizer,
    AdamWOptimizer,
    LRScheduler,
    CosineAnnealingLR,
    LinearWarmupLR,
    StepLR,
    ExponentialLR,
    OneCycleLR,
)
from .losses import (
    LossFunction,
    CrossEntropyLoss,
    MSELoss,
    L1Loss,
    HuberLoss,
    FocalLoss,
    ContrastiveLoss,
    KLDivergenceLoss,
    TripletLoss,
    CombinedLoss,
)
from .data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    BatchSampler,
    DistributedSampler,
    Collator,
)

__all__ = [
    # Trainer
    "TrainerConfig",
    "TrainingState",
    "TrainingCallback",
    "BaseTrainer",
    "EarlyStopping",
    "GradientAccumulator",
    "CheckpointManager",
    # Optimizers
    "Optimizer",
    "SGDOptimizer",
    "AdamOptimizer",
    "AdamWOptimizer",
    "LRScheduler",
    "CosineAnnealingLR",
    "LinearWarmupLR",
    "StepLR",
    "ExponentialLR",
    "OneCycleLR",
    # Losses
    "LossFunction",
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "HuberLoss",
    "FocalLoss",
    "ContrastiveLoss",
    "KLDivergenceLoss",
    "TripletLoss",
    "CombinedLoss",
    # Data
    "Dataset",
    "DataLoader",
    "RandomSampler",
    "SequentialSampler",
    "WeightedRandomSampler",
    "BatchSampler",
    "DistributedSampler",
    "Collator",
]
