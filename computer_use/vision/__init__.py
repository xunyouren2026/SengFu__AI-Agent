"""
Computer Vision Module

Provides object detection (YOLO-style) and multi-scale template matching.
"""

from .yolo_detector import (
    BBox,
    Detection,
    GridCell,
    FeatureMapLevel,
    AnchorGenerator,
    NonMaxSuppression,
    BBoxEncoder,
    FeaturePyramid,
    DetectionPostProcessor,
    YOLODetector,
)

from .multi_scale import (
    MatchResult,
    PyramidType,
    PyramidLevel,
    ImagePyramid,
    ScaleEstimator,
    RotationSearch,
    AdaptiveSearchStrategy,
    MultiScaleMatcher,
)

__all__ = [
    "BBox",
    "Detection",
    "GridCell",
    "FeatureMapLevel",
    "AnchorGenerator",
    "NonMaxSuppression",
    "BBoxEncoder",
    "FeaturePyramid",
    "DetectionPostProcessor",
    "YOLODetector",
    "MatchResult",
    "PyramidType",
    "PyramidLevel",
    "ImagePyramid",
    "ScaleEstimator",
    "RotationSearch",
    "AdaptiveSearchStrategy",
    "MultiScaleMatcher",
]
