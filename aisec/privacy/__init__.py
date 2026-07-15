"""
AISEC Privacy Module
=====================
K-anonymity, l-diversity, t-closeness, synthetic data generation,
and differential privacy implementations.
"""

from .k_anonymity import (
    KAnonymizer,
    QuasiIdentifierDetector,
    GeneralizationHierarchy,
    KAnonymityChecker,
    LDiversityChecker,
    TCloseChecker,
    MondrianAlgorithm,
    AnonymizedDataset,
)
from .synthetic_data import (
    SyntheticDataGenerator,
    StatisticalProfiler,
    DistributionFitter,
    CorrelationPreserver,
    PrivacyNoise,
    CTGANSimulator,
    QualityMetrics,
    SyntheticValidator,
)

__all__ = [
    "KAnonymizer",
    "QuasiIdentifierDetector",
    "GeneralizationHierarchy",
    "KAnonymityChecker",
    "LDiversityChecker",
    "TCloseChecker",
    "MondrianAlgorithm",
    "AnonymizedDataset",
    "SyntheticDataGenerator",
    "StatisticalProfiler",
    "DistributionFitter",
    "CorrelationPreserver",
    "PrivacyNoise",
    "CTGANSimulator",
    "QualityMetrics",
    "SyntheticValidator",
]
