"""
AI科学中心 - AI for Science Hub

整合SH-GNN物理引擎的跨领域科学应用：
- 气候科学 (climate)
- 粒子物理 (particle_physics)
- 地震学 (seismology)
- 材料科学 (materials)
- 分子动力学 (molecular_dynamics)
- 流体动力学 (fluid_dynamics)
- 天文学 (astronomy)
"""

from .climate.climate_model import ClimateSHGNN, ExtremeWeatherDetector
from .particle_physics.detector import ParticleDetectorSHGNN
from .seismology.earthquake_model import SeismicSHGNN
from .materials.property_predictor import MaterialsPropertyPredictor
from .molecular_dynamics.md_simulator import MolecularDynamicsSHGNN
from .fluid_dynamics.cfd_solver import CFDSolverSHGNN
from .astronomy.cosmology_model import CosmologySHGNN

__all__ = [
    'ClimateSHGNN',
    'ExtremeWeatherDetector',
    'ParticleDetectorSHGNN',
    'SeismicSHGNN',
    'MaterialsPropertyPredictor',
    'MolecularDynamicsSHGNN',
    'CFDSolverSHGNN',
    'CosmologySHGNN',
]
