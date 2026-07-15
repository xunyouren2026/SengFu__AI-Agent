"""
联邦学习客户端模块
"""
from .trainer import (
    TrainingConfig,
    LocalDataset,
    ModelWrapper,
    TrainingMetrics,
    ClientTrainer
)
from .selector import (
    SamplingStrategy,
    ClientStatistics,
    ImportanceWeighter,
    ClientSelector
)
from .personalization import (
    PersonalizationStrategy,
    LayerConfig,
    PersonalizedModel,
    AdapterLayer,
    LocalFineTuner,
    PersonalizationManager
)

__all__ = [
    # trainer
    'TrainingConfig',
    'LocalDataset',
    'ModelWrapper',
    'TrainingMetrics',
    'ClientTrainer',
    # selector
    'SamplingStrategy',
    'ClientStatistics',
    'ImportanceWeighter',
    'ClientSelector',
    # personalization
    'PersonalizationStrategy',
    'LayerConfig',
    'PersonalizedModel',
    'AdapterLayer',
    'LocalFineTuner',
    'PersonalizationManager'
]
