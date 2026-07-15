"""
3D Media - 3D媒体处理模块
包含3D网格简化功能。
"""

from .mesh_simplifier import (
    MeshSimplifier,
    EdgeCollapse,
    QuadricErrorMetric,
    VertexMerger,
    TopologyChecker,
    LODGenerator,
    MeshStats as MeshStatsComputer,
    TriangleMesh,
    SimplificationResult,
    LODLevel,
)

__all__ = [
    "MeshSimplifier",
    "EdgeCollapse",
    "QuadricErrorMetric",
    "VertexMerger",
    "TopologyChecker",
    "LODGenerator",
    "MeshStatsComputer",
    "TriangleMesh",
    "SimplificationResult",
    "LODLevel",
]
