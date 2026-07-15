"""
AGI Unified Framework - 3D Generation Module
=============================================

3D生成模块，包含网格生成、Zero123多视角生成和TripoSR单图到3D网格生成。
仅使用 Python 标准库，无外部依赖。
"""

from .mesh_generator import Mesh3D, GenerationResult
from .zero123 import (
    Zero123Pipeline,
    CameraPoseGenerator,
    ViewConsistency,
    NovelViewSynthesizer,
    DepthEstimator,
)
from .triposr import (
    TripoSRPipeline,
    ImageEncoder,
    FeaturePyramid,
    VolumeReconstructor,
    ImplicitSurface,
    MarchingCubes,
    TextureGenerator,
    MeshPostProcessor,
    NormalEstimator,
    TripoSRConfig,
)

__all__ = [
    # mesh_generator
    "Mesh3D",
    "GenerationResult",
    # zero123
    "Zero123Pipeline",
    "CameraPoseGenerator",
    "ViewConsistency",
    "NovelViewSynthesizer",
    "DepthEstimator",
    # triposr
    "TripoSRPipeline",
    "ImageEncoder",
    "FeaturePyramid",
    "VolumeReconstructor",
    "ImplicitSurface",
    "MarchingCubes",
    "TextureGenerator",
    "MeshPostProcessor",
    "NormalEstimator",
    "TripoSRConfig",
]
