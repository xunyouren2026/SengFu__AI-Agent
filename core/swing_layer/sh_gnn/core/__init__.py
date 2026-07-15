"""
SH-GNN Core - 球谐图神经网络核心

严格SO(3)等变的几何深度学习框架。
核心仅420行，将物理定律直接编译进网络架构。
"""

from .wigner_d import WignerDMatrix
from .spherical_harmonics import SphericalHarmonics
from .equivariant_conv import EquivariantConvLayer
from .parseval_scheduler import ParsevalScheduler
from .physics_loss import PhysicsConstraintLoss

__all__ = [
    'WignerDMatrix',
    'SphericalHarmonics', 
    'EquivariantConvLayer',
    'ParsevalScheduler',
    'PhysicsConstraintLoss',
]
