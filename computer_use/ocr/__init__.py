"""
OCR (Optical Character Recognition) Module

Provides image preprocessing and language detection for OCR pipelines.
"""

from .preprocess import (
    BinarizationMethod,
    DenoisingMethod,
    PreprocessingConfig,
    Binarizer,
    Denoiser,
    Deskewer,
    ContrastEnhancer,
    BorderRemover,
    LineSegmenter,
    PreprocessingPipeline,
)

from .language_detect import (
    Script,
    DetectionResult,
    UnicodeRangeAnalyzer,
    NGramProfile,
    ScriptClassifier,
    LanguageDetector,
)

__all__ = [
    "BinarizationMethod",
    "DenoisingMethod",
    "PreprocessingConfig",
    "Binarizer",
    "Denoiser",
    "Deskewer",
    "ContrastEnhancer",
    "BorderRemover",
    "LineSegmenter",
    "PreprocessingPipeline",
    "Script",
    "DetectionResult",
    "UnicodeRangeAnalyzer",
    "NGramProfile",
    "ScriptClassifier",
    "LanguageDetector",
]
