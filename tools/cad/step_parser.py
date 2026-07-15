"""
STEP Parser - STEP文件解析器
提取几何信息
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class STEPEntityType(Enum):
    """STEP实体类型枚举"""
    CARTESIAN_POINT = "CARTESIAN_POINT"
    DIRECTION = "DIRECTION"
    VECTOR = "VECTOR"
    LINE = "LINE"
    CIRCLE = "CIRCLE"
    ELLIPSE = "ELLIPSE"
    B_SPLINE_CURVE = "B_SPLINE_CURVE"
    B_SPLINE_SURFACE = "B_SPLINE_SURFACE"
    PLANE = "PLANE"
    CYLINDRICAL_SURFACE = "CYLINDRICAL_SURFACE"
    CONICAL_SURFACE = "CONICAL_SURFACE"
    SPHERICAL_SURFACE = "SPHERICAL_SURFACE"
    TOROIDAL_SURFACE = "TOROIDAL_SURFACE"
    SURFACE_OF_REVOLUTION = "SURFACE_OF_REVOLUTION"
    ADVANCED_FACE = "ADVANCED_FACE"
    CLOSED_SHELL = "CLOSED_SHELL"
    MANIFOLD_SOLID_BREP = "MANIFOLD_SOLID_BREP"
    VERTEX_POINT = "VERTEX_POINT"
    EDGE_CURVE = "EDGE_CURVE"
    ORIENTED_EDGE = "ORIENTED_EDGE"
    FACE_BOUND = "FACE_BOUND"
    FACE_OUTER_BOUND = "FACE_OUTER_BOUND"


@dataclass
class Point3D:
    """三维点"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: "Point3D") -> float:
        """计算到另一点的距离"""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组"""
        return (self.x, self.y, self.z)
    
    def to_list(self) -> List[float]:
        """转换为列表"""
        return [self.x, self.y, self.z]


@dataclass
class Vector3D:
    """三维向量"""
    x: float
    y: float
    z: float
    
    def magnitude(self) -> float:
        """计算模"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> "Vector3D":
        """归一化"""
        mag = self.magnitude()
        if mag > 0:
            return Vector3D(self.x / mag, self.y / mag, self.z / mag)
        return Vector3D(0, 0, 0)
    
    def dot(self, other: "Vector3D") -> float:
        """点积"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: "Vector3D") -> "Vector3D":
        """叉积"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )


@dataclass
class STEPEntity:
    """STEP实体"""
    id: int
    entity_type: str
    name: str = ""
    attributes: List[Any] = field(default_factory=list)
    
    def get_attribute(self, index: int, default: Any = None) -> Any:
        """获取属性"""
        if 0 <= index < len(self.attributes):
            return self.attributes[index]
        return default


@dataclass
class GeometricCurve:
    """几何曲线"""
    curve_type: str
    start_point: Optional[Point3D] = None
    end_point: Optional[Point3D] = None
    center: Optional[Point3D] = None  # 圆弧中心
    radius: float = 0.0  # 圆弧半径
    normal: Optional[Vector3D] = None  # 圆弧法向量
    control_points: List[Point3D] = field(default_factory=list)  # B样条控制点
    knots: List[float] = field(default_factory=list)  # B样条节点
    weights: List[float] = field(default_factory=list)  # B样条权重


@dataclass
class GeometricSurface:
    """几何曲面"""
    surface_type: str
    origin: Optional[Point3D] = None
    normal: Optional[Vector3D] = None
    radius: float = 0.0
    semi_axis1: float = 0.0
    semi_axis2: float = 0.0
    major_radius: float = 0.0  # 圆环主半径
    minor_radius: float = 0.0  # 圆环次半径
    control_points: List[List[Point3D]] = field(default_factory=list)  # B样条控制点网格
    u_knots: List[float] = field(default_factory=list)
    v_knots: List[float] = field(default_factory=list)


@dataclass
class Face:
    """面"""
    id: int
    surface: Optional[GeometricSurface] = None
    bounds: List[int] = field(default_factory=list)  # 边界环ID
    orientation: bool = True
    area: float = 0.0


@dataclass
class Edge:
    """边"""
    id: int
    curve: Optional[GeometricCurve] = None
    start_vertex: int = 0
    end_vertex: int = 0
    orientation: bool = True
    length: float = 0.0


@dataclass
class Vertex:
    """顶点"""
    id: int
    point: Optional[Point3D] = None


@dataclass
class STEPModel:
    """STEP模型"""
    header: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[int, STEPEntity] = field(default_factory=dict)
    vertices: Dict[int, Vertex] = field(default_factory=dict)
    edges: Dict[int, Edge] = field(default_factory=dict)
    faces: Dict[int, Face] = field(default_factory=dict)
    shells: List[int] = field(default_factory=list)
    solids: List[int] = field(default_factory=list)
    
    # 统计信息
    total_vertices: int = 0
    total_edges: int = 0
    total_faces: int = 0
    total_shells: int = 0
    total_solids: int = 0
    
    # 边界框
    min_point: Optional[Point3D] = None
    max_point: Optional[Point3D] = None


class STEPParser:
    """STEP文件解析器"""
    
    def __init__(self):
        self._current_model: Optional[STEPModel] = None
        self._entity_map: Dict[int, STEPEntity] = {}
    
    def parse_file(self, file_path: str) -> STEPModel:
        """解析STEP文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            logger.error(f"Failed to parse STEP file: {e}")
            return STEPModel()
    
    def parse_string(self, content: str) -> STEPModel:
        """解析STEP字符串"""
        self._current_model = STEPModel()
        self._entity_map = {}
        
        # 分割HEADER和DATA部分
        header_match = re.search(r'HEADER;(.+?)ENDSEC;', content, re.DOTALL)
        data_match = re.search(r'DATA;(.+?)ENDSEC;', content, re.DOTALL)
        
        if header_match:
            self._parse_header(header_match.group(1))
        
        if data_match:
            self._parse_data(data_match.group(1))
        
        # 构建几何模型
        self._build_geometry()
        
        # 计算统计信息
        self._calculate_statistics()
        
        return self._current_model
    
    def _parse_header(self, header_content: str) -> None:
        """解析HEADER部分"""
        header = self._current_model.header
        
        # 提取文件描述
        file_desc_match = re.search(
            r'FILE_DESCRIPTION\s*\(\s*\(([^)]*)\)\s*,\s*\'([^\']*)\'\s*\)\s*;',
            header_content
        )
        if file_desc_match:
            header['description'] = file_desc_match.group(1).strip('"\'')
            header['implementation_level'] = file_desc_match.group(2)
        
        # 提取文件名
        file_name_match = re.search(
            r'FILE_NAME\s*\(\s*\'([^\']*)\'\s*,\s*\'([^\']*)\'\s*,',
            header_content
        )
        if file_name_match:
            header['name'] = file_name_match.group(1)
            header['time_stamp'] = file_name_match.group(2)
        
        # 提取作者和组织信息
        author_match = re.search(
            r'FILE_NAME\s*\([^)]*\)\s*;\s*',
            header_content
        )
    
    def _parse_data(self, data_content: str) -> None:
        """解析DATA部分"""
        # 匹配实体定义: #123 = ENTITY_TYPE(...) ;
        entity_pattern = re.compile(
            r'#(\d+)\s*=\s*([A-Z_]+)\s*\((.*?)\)\s*;',
            re.DOTALL
        )
        
        for match in entity_pattern.finditer(data_content):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            attributes_str = match.group(3)
            
            attributes = self._parse_attributes(attributes_str)
            
            entity = STEPEntity(
                id=entity_id,
                entity_type=entity_type,
                attributes=attributes
            )
            
            self._entity_map[entity_id] = entity
            self._current_model.entities[entity_id] = entity
    
    def _parse_attributes(self, attr_str: str) -> List[Any]:
        """解析属性列表"""
        attributes = []
        current = ""
        depth = 0
        in_string = False
        string_char = None
        
        i = 0
        while i < len(attr_str):
            char = attr_str[i]
            
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                    current += char
                elif char == '(':
                    depth += 1
                    current += char
                elif char == ')':
                    depth -= 1
                    current += char
                elif char == ',' and depth == 0:
                    attributes.append(self._parse_value(current.strip()))
                    current = ""
                else:
                    current += char
            else:
                current += char
                if char == string_char and (i == 0 or attr_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        if current.strip():
            attributes.append(self._parse_value(current.strip()))
        
        return attributes
    
    def _parse_value(self, value: str) -> Any:
        """解析单个值"""
        value = value.strip()
        
        if not value or value == '$':
            return None
        
        # 引用其他实体
        if value.startswith('#'):
            return int(value[1:])
        
        # 字符串
        if value.startswith('"') or value.startswith("'"):
            return value[1:-1]
        
        # 枚举值
        if value.startswith('.'):
            return value[1:-1]
        
        # 列表
        if value.startswith('('):
            return self._parse_attributes(value[1:-1])
        
        # 函数调用
        if '(' in value:
            func_match = re.match(r'([A-Z_]+)\s*\((.*)\)', value)
            if func_match:
                return {
                    'function': func_match.group(1),
                    'arguments': self._parse_attributes(func_match.group(2))
                }
        
        # 数字
        try:
            if '.' in value or 'E' in value.upper():
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _build_geometry(self) -> None:
        """构建几何模型"""
        model = self._current_model
        
        # 解析顶点
        for entity_id, entity in self._entity_map.items():
            if entity.entity_type == 'CARTESIAN_POINT':
                coords = entity.attributes
                if len(coords) >= 3:
                    point = Point3D(
                        float(coords[0]) if coords[0] is not None else 0,
                        float(coords[1]) if coords[1] is not None else 0,
                        float(coords[2]) if coords[2] is not None else 0
                    )
                    vertex = Vertex(id=entity_id, point=point)
                    model.vertices[entity_id] = vertex
        
        # 解析边
        for entity_id, entity in self._entity_map.items():
            if entity.entity_type == 'EDGE_CURVE':
                edge = Edge(id=entity_id)
                
                # 获取起点和终点
                if len(entity.attributes) >= 2:
                    edge.start_vertex = entity.attributes[0] if isinstance(entity.attributes[0], int) else 0
                    edge.end_vertex = entity.attributes[1] if isinstance(entity.attributes[1], int) else 0
                
                # 解析曲线几何
                if len(entity.attributes) >= 3:
                    curve_ref = entity.attributes[2]
                    if isinstance(curve_ref, int):
                        edge.curve = self._parse_curve(curve_ref)
                
                model.edges[entity_id] = edge
        
        # 解析面
        for entity_id, entity in self._entity_map.items():
            if entity.entity_type in ('ADVANCED_FACE', 'FACE_SURFACE'):
                face = Face(id=entity_id)
                
                # 获取边界
                if len(entity.attributes) >= 1:
                    bounds = entity.attributes[0]
                    if isinstance(bounds, list):
                        face.bounds = [b for b in bounds if isinstance(b, int)]
                
                # 获取曲面
                if len(entity.attributes) >= 2:
                    surface_ref = entity.attributes[1]
                    if isinstance(surface_ref, int):
                        face.surface = self._parse_surface(surface_ref)
                
                # 获取方向
                if len(entity.attributes) >= 3:
                    face.orientation = entity.attributes[2] == 'T'
                
                model.faces[entity_id] = face
        
        # 解析壳
        for entity_id, entity in self._entity_map.items():
            if entity.entity_type in ('CLOSED_SHELL', 'OPEN_SHELL'):
                model.shells.append(entity_id)
                if len(entity.attributes) >= 1:
                    faces = entity.attributes[0]
                    if isinstance(faces, list):
                        for face_ref in faces:
                            if isinstance(face_ref, int) and face_ref not in model.faces:
                                pass  # 面已在上面处理
        
        # 解析实体
        for entity_id, entity in self._entity_map.items():
            if entity.entity_type == 'MANIFOLD_SOLID_BREP':
                model.solids.append(entity_id)
    
    def _parse_curve(self, curve_id: int) -> Optional[GeometricCurve]:
        """解析曲线"""
        if curve_id not in self._entity_map:
            return None
        
        entity = self._entity_map[curve_id]
        curve = GeometricCurve(curve_type=entity.entity_type)
        
        if entity.entity_type == 'LINE':
            # LINE: 点, 方向
            if len(entity.attributes) >= 2:
                point_ref = entity.attributes[0]
                if isinstance(point_ref, int) and point_ref in self._entity_map:
                    point_entity = self._entity_map[point_ref]
                    if len(point_entity.attributes) >= 3:
                        curve.start_point = Point3D(
                            float(point_entity.attributes[0] or 0),
                            float(point_entity.attributes[1] or 0),
                            float(point_entity.attributes[2] or 0)
                        )
        
        elif entity.entity_type == 'CIRCLE':
            # CIRCLE: 轴位置, 半径
            if len(entity.attributes) >= 2:
                axis_ref = entity.attributes[0]
                radius = entity.attributes[1]
                
                curve.radius = float(radius) if radius is not None else 0
                
                if isinstance(axis_ref, int) and axis_ref in self._entity_map:
                    axis_entity = self._entity_map[axis_ref]
                    if axis_entity.entity_type == 'AXIS2_PLACEMENT_3D':
                        if len(axis_entity.attributes) >= 1:
                            center_ref = axis_entity.attributes[0]
                            if isinstance(center_ref, int) and center_ref in self._entity_map:
                                center_entity = self._entity_map[center_ref]
                                if len(center_entity.attributes) >= 3:
                                    curve.center = Point3D(
                                        float(center_entity.attributes[0] or 0),
                                        float(center_entity.attributes[1] or 0),
                                        float(center_entity.attributes[2] or 0)
                                    )
        
        elif entity.entity_type == 'B_SPLINE_CURVE_WITH_KNOTS':
            curve.curve_type = 'B_SPLINE'
            
            # 解析控制点
            if len(entity.attributes) >= 1:
                control_points_ref = entity.attributes[0]
                if isinstance(control_points_ref, int) and control_points_ref in self._entity_map:
                    cp_entity = self._entity_map[control_points_ref]
                    if len(cp_entity.attributes) >= 1:
                        points_list = cp_entity.attributes[0]
                        if isinstance(points_list, list):
                            for pt_ref in points_list:
                                if isinstance(pt_ref, int) and pt_ref in self._entity_map:
                                    pt_entity = self._entity_map[pt_ref]
                                    if len(pt_entity.attributes) >= 3:
                                        curve.control_points.append(Point3D(
                                            float(pt_entity.attributes[0] or 0),
                                            float(pt_entity.attributes[1] or 0),
                                            float(pt_entity.attributes[2] or 0)
                                        ))
            
            # 解析节点
            if len(entity.attributes) >= 4:
                knots_list = entity.attributes[3]
                if isinstance(knots_list, list):
                    curve.knots = [float(k) for k in knots_list if k is not None]
        
        return curve
    
    def _parse_surface(self, surface_id: int) -> Optional[GeometricSurface]:
        """解析曲面"""
        if surface_id not in self._entity_map:
            return None
        
        entity = self._entity_map[surface_id]
        surface = GeometricSurface(surface_type=entity.entity_type)
        
        if entity.entity_type == 'PLANE':
            # PLANE: 轴位置
            if len(entity.attributes) >= 1:
                axis_ref = entity.attributes[0]
                if isinstance(axis_ref, int) and axis_ref in self._entity_map:
                    axis_entity = self._entity_map[axis_ref]
                    if len(axis_entity.attributes) >= 1:
                        origin_ref = axis_entity.attributes[0]
                        if isinstance(origin_ref, int) and origin_ref in self._entity_map:
                            origin_entity = self._entity_map[origin_ref]
                            if len(origin_entity.attributes) >= 3:
                                surface.origin = Point3D(
                                    float(origin_entity.attributes[0] or 0),
                                    float(origin_entity.attributes[1] or 0),
                                    float(origin_entity.attributes[2] or 0)
                                )
        
        elif entity.entity_type == 'CYLINDRICAL_SURFACE':
            # CYLINDRICAL_SURFACE: 轴位置, 半径
            if len(entity.attributes) >= 2:
                surface.radius = float(entity.attributes[1] or 0)
        
        elif entity.entity_type == 'SPHERICAL_SURFACE':
            # SPHERICAL_SURFACE: 中心, 半径
            if len(entity.attributes) >= 2:
                surface.radius = float(entity.attributes[1] or 0)
        
        elif entity.entity_type == 'TOROIDAL_SURFACE':
            # TOROIDAL_SURFACE: 轴位置, 主半径, 次半径
            if len(entity.attributes) >= 3:
                surface.major_radius = float(entity.attributes[1] or 0)
                surface.minor_radius = float(entity.attributes[2] or 0)
        
        return surface
    
    def _calculate_statistics(self) -> None:
        """计算统计信息"""
        model = self._current_model
        
        model.total_vertices = len(model.vertices)
        model.total_edges = len(model.edges)
        model.total_faces = len(model.faces)
        model.total_shells = len(model.shells)
        model.total_solids = len(model.solids)
        
        # 计算边界框
        if model.vertices:
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for vertex in model.vertices.values():
                if vertex.point:
                    min_x = min(min_x, vertex.point.x)
                    min_y = min(min_y, vertex.point.y)
                    min_z = min(min_z, vertex.point.z)
                    max_x = max(max_x, vertex.point.x)
                    max_y = max(max_y, vertex.point.y)
                    max_z = max(max_z, vertex.point.z)
            
            model.min_point = Point3D(min_x, min_y, min_z)
            model.max_point = Point3D(max_x, max_y, max_z)


class STEPAnalyzer:
    """STEP文件分析器"""
    
    def __init__(self):
        self.parser = STEPParser()
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """分析STEP文件"""
        model = self.parser.parse_file(file_path)
        return self._generate_report(model)
    
    def analyze_string(self, content: str) -> Dict[str, Any]:
        """分析STEP字符串"""
        model = self.parser.parse_string(content)
        return self._generate_report(model)
    
    def _generate_report(self, model: STEPModel) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            'header': model.header,
            'statistics': {
                'vertices': model.total_vertices,
                'edges': model.total_edges,
                'faces': model.total_faces,
                'shells': model.total_shells,
                'solids': model.total_solids
            },
            'geometry': {
                'surface_types': self._count_surface_types(model),
                'curve_types': self._count_curve_types(model)
            }
        }
        
        if model.min_point and model.max_point:
            report['bounding_box'] = {
                'min': model.min_point.to_dict() if hasattr(model.min_point, 'to_dict') else {
                    'x': model.min_point.x, 'y': model.min_point.y, 'z': model.min_point.z
                },
                'max': model.max_point.to_dict() if hasattr(model.max_point, 'to_dict') else {
                    'x': model.max_point.x, 'y': model.max_point.y, 'z': model.max_point.z
                },
                'dimensions': {
                    'x': model.max_point.x - model.min_point.x,
                    'y': model.max_point.y - model.min_point.y,
                    'z': model.max_point.z - model.min_point.z
                }
            }
        
        return report
    
    def _count_surface_types(self, model: STEPModel) -> Dict[str, int]:
        """统计曲面类型"""
        types: Dict[str, int] = {}
        
        for face in model.faces.values():
            if face.surface:
                surf_type = face.surface.surface_type
                types[surf_type] = types.get(surf_type, 0) + 1
        
        return types
    
    def _count_curve_types(self, model: STEPModel) -> Dict[str, int]:
        """统计曲线类型"""
        types: Dict[str, int] = {}
        
        for edge in model.edges.values():
            if edge.curve:
                curve_type = edge.curve.curve_type
                types[curve_type] = types.get(curve_type, 0) + 1
        
        return types
    
    def get_model_info(self, file_path: str) -> STEPModel:
        """获取模型信息"""
        return self.parser.parse_file(file_path)
    
    def validate_step(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证STEP文件"""
        model = self.parser.parse_file(file_path)
        errors = []
        
        # 检查基本结构
        if model.total_solids == 0:
            errors.append("No solid bodies found")
        
        if model.total_faces == 0:
            errors.append("No faces found")
        
        # 检查几何完整性
        for edge_id, edge in model.edges.items():
            if edge.start_vertex == 0 or edge.end_vertex == 0:
                errors.append(f"Edge #{edge_id} has incomplete vertices")
        
        return len(errors) == 0, errors
    
    def extract_vertices(self, file_path: str) -> List[Point3D]:
        """提取所有顶点"""
        model = self.parser.parse_file(file_path)
        return [v.point for v in model.vertices.values() if v.point]
    
    def extract_edges(self, file_path: str) -> List[Tuple[Point3D, Point3D]]:
        """提取所有边（作为点对）"""
        model = self.parser.parse_file(file_path)
        edges = []
        
        for edge in model.edges.values():
            start = model.vertices.get(edge.start_vertex)
            end = model.vertices.get(edge.end_vertex)
            
            if start and end and start.point and end.point:
                edges.append((start.point, end.point))
        
        return edges
