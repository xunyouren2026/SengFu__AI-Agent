"""
CAD Tools Module - CAD工具模块
提供OpenSCAD封装和STEP文件解析
"""

from .openscad_wrapper import (
    OpenSCADWrapper,
    OpenSCADBuilder,
    OpenSCADError,
    ExportFormat,
    Point3D as OpenSCADPoint3D,
    RenderResult,
    GeometryInfo,
    PredefinedModules
)

from .step_parser import (
    STEPParser,
    STEPAnalyzer,
    STEPEntity,
    STEPEntityType,
    STEPModel,
    Point3D as STEPPoint3D,
    Vector3D,
    GeometricCurve,
    GeometricSurface,
    Face,
    Edge,
    Vertex
)

__all__ = [
    # OpenSCAD
    "OpenSCADWrapper",
    "OpenSCADBuilder",
    "OpenSCADError",
    "ExportFormat",
    "RenderResult",
    "GeometryInfo",
    "PredefinedModules",
    # STEP
    "STEPParser",
    "STEPAnalyzer",
    "STEPEntity",
    "STEPEntityType",
    "STEPModel",
    "Vector3D",
    "GeometricCurve",
    "GeometricSurface",
    "Face",
    "Edge",
    "Vertex"
]
