"""
Slicer Engine - 3D模型切片引擎
将3D模型切片为G-code指令
"""

import math
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SlicerType(Enum):
    """切片器类型枚举"""
    CURA = "cura"
    PRUSASLICER = "prusaslicer"
    SLIC3R = "slic3r"
    SIMPLIFY3D = "simplify3d"


class InfillType(Enum):
    """填充类型枚举"""
    GRID = "grid"
    LINES = "lines"
    TRIANGULAR = "triangular"
    HONEYCOMB = "honeycomb"
    GYROID = "gyroid"
    CONCENTRIC = "concentric"
    RECTILINEAR = "rectilinear"


class SupportType(Enum):
    """支撑类型枚举"""
    NONE = "none"
    NORMAL = "normal"
    TREE = "tree"
    EVERYWHERE = "everywhere"


@dataclass
class Point3D:
    """三维坐标点"""
    x: float
    y: float
    z: float
    
    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "Point3D":
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def distance_to(self, other: "Point3D") -> float:
        """计算两点距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass
class Triangle:
    """三角形面片"""
    v1: Point3D
    v2: Point3D
    v3: Point3D
    normal: Optional[Point3D] = None
    
    def calculate_normal(self) -> Point3D:
        """计算法向量"""
        edge1 = self.v2 - self.v1
        edge2 = self.v3 - self.v1
        
        nx = edge1.y * edge2.z - edge1.z * edge2.y
        ny = edge1.z * edge2.x - edge1.x * edge2.z
        nz = edge1.x * edge2.y - edge1.y * edge2.x
        
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            return Point3D(nx/length, ny/length, nz/length)
        return Point3D(0, 0, 1)
    
    def intersect_plane(self, z: float) -> Optional[List[Tuple[Point3D, Point3D]]]:
        """计算与Z平面的交线"""
        edges = [(self.v1, self.v2), (self.v2, self.v3), (self.v3, self.v1)]
        intersections = []
        
        for p1, p2 in edges:
            if (p1.z - z) * (p2.z - z) < 0:
                t = (z - p1.z) / (p2.z - p1.z)
                x = p1.x + t * (p2.x - p1.x)
                y = p1.y + t * (p2.y - p1.y)
                intersections.append(Point3D(x, y, z))
        
        if len(intersections) == 2:
            return [(intersections[0], intersections[1])]
        return None


@dataclass
class SliceLayer:
    """切片层"""
    z_height: float
    contours: List[List[Point3D]] = field(default_factory=list)
    infill_paths: List[List[Point3D]] = field(default_factory=list)
    support_paths: List[List[Point3D]] = field(default_factory=list)
    is_first_layer: bool = False
    is_top_layer: bool = False


@dataclass
class SlicerConfig:
    """切片配置"""
    layer_height: float = 0.2
    first_layer_height: float = 0.3
    line_width: float = 0.4
    infill_density: float = 20.0
    infill_type: InfillType = InfillType.GRID
    top_layers: int = 3
    bottom_layers: int = 3
    wall_line_count: int = 2
    support_type: SupportType = SupportType.NORMAL
    support_angle: float = 45.0
    support_density: float = 15.0
    print_speed: int = 50
    first_layer_speed: int = 20
    travel_speed: int = 150
    retraction_distance: float = 6.0
    retraction_speed: int = 25
    z_offset: float = 0.0
    xy_offset: float = 0.0
    build_volume_x: float = 220.0
    build_volume_y: float = 220.0
    build_volume_z: float = 250.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "layer_height": self.layer_height,
            "first_layer_height": self.first_layer_height,
            "line_width": self.line_width,
            "infill_density": self.infill_density,
            "infill_type": self.infill_type.value,
            "top_layers": self.top_layers,
            "bottom_layers": self.bottom_layers,
            "wall_line_count": self.wall_line_count,
            "support_type": self.support_type.value,
            "support_angle": self.support_angle,
            "support_density": self.support_density,
            "print_speed": self.print_speed,
            "first_layer_speed": self.first_layer_speed,
            "travel_speed": self.travel_speed,
            "retraction_distance": self.retraction_distance,
            "retraction_speed": self.retraction_speed,
            "z_offset": self.z_offset,
            "xy_offset": self.xy_offset,
            "build_volume_x": self.build_volume_x,
            "build_volume_y": self.build_volume_y,
            "build_volume_z": self.build_volume_z
        }


@dataclass
class SlicerResult:
    """切片结果"""
    gcode: str
    total_layers: int
    estimated_time: float  # 分钟
    filament_used: float  # 毫米
    bounding_box: Tuple[Point3D, Point3D]
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class STLParser:
    """STL文件解析器"""
    
    @staticmethod
    def parse_binary(data: bytes) -> List[Triangle]:
        """解析二进制STL文件"""
        triangles = []
        
        # 跳过80字节头
        offset = 80
        num_triangles = int.from_bytes(data[offset:offset+4], 'little')
        offset += 4
        
        for _ in range(num_triangles):
            if offset + 50 > len(data):
                break
            
            # 法向量
            nx = struct_unpack_float(data[offset:offset+4])
            ny = struct_unpack_float(data[offset+4:offset+8])
            nz = struct_unpack_float(data[offset+8:offset+12])
            offset += 12
            
            # 顶点
            v1x = struct_unpack_float(data[offset:offset+4])
            v1y = struct_unpack_float(data[offset+4:offset+8])
            v1z = struct_unpack_float(data[offset+8:offset+12])
            offset += 12
            
            v2x = struct_unpack_float(data[offset:offset+4])
            v2y = struct_unpack_float(data[offset+4:offset+8])
            v2z = struct_unpack_float(data[offset+8:offset+12])
            offset += 12
            
            v3x = struct_unpack_float(data[offset:offset+4])
            v3y = struct_unpack_float(data[offset+4:offset+8])
            v3z = struct_unpack_float(data[offset+8:offset+12])
            offset += 12
            
            # 跳过属性字节计数
            offset += 2
            
            triangles.append(Triangle(
                v1=Point3D(v1x, v1y, v1z),
                v2=Point3D(v2x, v2y, v2z),
                v3=Point3D(v3x, v3y, v3z),
                normal=Point3D(nx, ny, nz)
            ))
        
        return triangles
    
    @staticmethod
    def parse_ascii(content: str) -> List[Triangle]:
        """解析ASCII STL文件"""
        triangles = []
        
        # 使用正则表达式解析
        facet_pattern = re.compile(
            r'facet\s+normal\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+'
            r'outer\s+loop\s+'
            r'vertex\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+'
            r'vertex\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+'
            r'vertex\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+'
            r'endloop\s+endfacet',
            re.IGNORECASE
        )
        
        for match in facet_pattern.finditer(content):
            groups = match.groups()
            triangles.append(Triangle(
                v1=Point3D(float(groups[3]), float(groups[4]), float(groups[5])),
                v2=Point3D(float(groups[6]), float(groups[7]), float(groups[8])),
                v3=Point3D(float(groups[9]), float(groups[10]), float(groups[11])),
                normal=Point3D(float(groups[0]), float(groups[1]), float(groups[2]))
            ))
        
        return triangles


def struct_unpack_float(data: bytes) -> float:
    """解包浮点数"""
    import struct
    return struct.unpack('<f', data)[0]


class GCodeGenerator:
    """G-code生成器"""
    
    def __init__(self, config: SlicerConfig):
        self.config = config
        self._gcode_lines: List[str] = []
        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._current_z: float = 0.0
        self._current_e: float = 0.0
        self._current_f: float = 0.0
        self._is_retracted: bool = False
    
    def add_comment(self, comment: str) -> None:
        """添加注释"""
        self._gcode_lines.append(f"; {comment}")
    
    def add_command(self, command: str) -> None:
        """添加命令"""
        self._gcode_lines.append(command)
    
    def set_position(self, x: Optional[float] = None, y: Optional[float] = None,
                     z: Optional[float] = None, e: Optional[float] = None,
                     f: Optional[float] = None) -> None:
        """设置位置"""
        parts = ["G1"]
        
        if x is not None:
            parts.append(f"X{x:.3f}")
            self._current_x = x
        if y is not None:
            parts.append(f"Y{y:.3f}")
            self._current_y = y
        if z is not None:
            parts.append(f"Z{z:.3f}")
            self._current_z = z
        if e is not None:
            parts.append(f"E{e:.5f}")
            self._current_e = e
        if f is not None:
            parts.append(f"F{int(f)}")
            self._current_f = f
        
        self._gcode_lines.append(" ".join(parts))
    
    def travel_to(self, x: float, y: float, z: Optional[float] = None) -> None:
        """移动到指定位置（不挤出）"""
        if self._is_retracted:
            self.unretract()
        
        if z is not None and z != self._current_z:
            self.set_position(z=z, f=self.config.travel_speed * 60)
        
        self.set_position(x=x, y=y, f=self.config.travel_speed * 60)
    
    def extrude_to(self, x: float, y: float, extrusion: float,
                   speed: Optional[int] = None) -> None:
        """挤出移动"""
        if self._is_retracted:
            self.unretract()
        
        new_e = self._current_e + extrusion
        feedrate = (speed or self.config.print_speed) * 60
        
        self.set_position(x=x, y=y, e=new_e, f=feedrate)
    
    def retract(self) -> None:
        """回抽"""
        if not self._is_retracted:
            new_e = self._current_e - self.config.retraction_distance
            self.set_position(e=new_e, f=self.config.retraction_speed * 60)
            self._is_retracted = True
    
    def unretract(self) -> None:
        """取消回抽"""
        if self._is_retracted:
            new_e = self._current_e + self.config.retraction_distance
            self.set_position(e=new_e, f=self.config.retraction_speed * 60)
            self._is_retracted = False
    
    def set_extruder_temp(self, temp: int, wait: bool = False) -> None:
        """设置挤出头温度"""
        cmd = "M109" if wait else "M104"
        self._gcode_lines.append(f"{cmd} S{temp}")
    
    def set_bed_temp(self, temp: int, wait: bool = False) -> None:
        """设置热床温度"""
        cmd = "M190" if wait else "M140"
        self._gcode_lines.append(f"{cmd} S{temp}")
    
    def set_fan_speed(self, speed: int) -> None:
        """设置风扇速度 (0-255)"""
        self._gcode_lines.append(f"M106 S{speed}")
    
    def home(self, axes: str = "XYZ") -> None:
        """归零"""
        self._gcode_lines.append(f"G28 {axes}")
    
    def get_gcode(self) -> str:
        """获取生成的G-code"""
        return "\n".join(self._gcode_lines)


class InfillGenerator:
    """填充路径生成器"""
    
    def __init__(self, config: SlicerConfig):
        self.config = config
    
    def generate_infill(self, contour: List[Point3D], z: float,
                        density: float) -> List[List[Point3D]]:
        """生成填充路径"""
        if density <= 0 or len(contour) < 3:
            return []
        
        # 获取边界框
        min_x = min(p.x for p in contour)
        max_x = max(p.x for p in contour)
        min_y = min(p.y for p in contour)
        max_y = max(p.y for p in contour)
        
        # 计算填充间距
        spacing = self.config.line_width / (density / 100.0)
        
        paths = []
        
        if self.config.infill_type == InfillType.LINES:
            paths = self._generate_lines_infill(contour, z, spacing, min_x, max_x, min_y, max_y)
        elif self.config.infill_type == InfillType.GRID:
            paths = self._generate_grid_infill(contour, z, spacing, min_x, max_x, min_y, max_y)
        elif self.config.infill_type == InfillType.RECTILINEAR:
            paths = self._generate_rectilinear_infill(contour, z, spacing, min_x, max_x, min_y, max_y)
        else:
            paths = self._generate_lines_infill(contour, z, spacing, min_x, max_x, min_y, max_y)
        
        return paths
    
    def _generate_lines_infill(self, contour: List[Point3D], z: float,
                               spacing: float, min_x: float, max_x: float,
                               min_y: float, max_y: float) -> List[List[Point3D]]:
        """生成线条填充"""
        paths = []
        y = min_y + spacing / 2
        direction = 1
        
        while y <= max_y:
            intersections = self._get_line_contour_intersections(contour, y)
            
            if len(intersections) >= 2:
                intersections.sort(key=lambda p: p.x)
                
                for i in range(0, len(intersections) - 1, 2):
                    if i + 1 < len(intersections):
                        if direction == 1:
                            path = [intersections[i], intersections[i + 1]]
                        else:
                            path = [intersections[i + 1], intersections[i]]
                        paths.append(path)
            
            y += spacing
            direction *= -1
        
        return paths
    
    def _generate_grid_infill(self, contour: List[Point3D], z: float,
                              spacing: float, min_x: float, max_x: float,
                              min_y: float, max_y: float) -> List[List[Point3D]]:
        """生成网格填充"""
        paths = []
        
        # 横向填充
        paths.extend(self._generate_lines_infill(contour, z, spacing, min_x, max_x, min_y, max_y))
        
        # 纵向填充（旋转90度）
        rotated_contour = [Point3D(p.y, -p.x, p.z) for p in contour]
        rotated_paths = self._generate_lines_infill(
            rotated_contour, z, spacing, min_y, max_y, -max_x, -min_x
        )
        
        for path in rotated_paths:
            rotated_back = [Point3D(-p.y, p.x, z) for p in path]
            paths.append(rotated_back)
        
        return paths
    
    def _generate_rectilinear_infill(self, contour: List[Point3D], z: float,
                                     spacing: float, min_x: float, max_x: float,
                                     min_y: float, max_y: float) -> List[List[Point3D]]:
        """生成直线填充"""
        return self._generate_lines_infill(contour, z, spacing, min_x, max_x, min_y, max_y)
    
    def _get_line_contour_intersections(self, contour: List[Point3D], y: float) -> List[Point3D]:
        """获取水平线与轮廓的交点"""
        intersections = []
        n = len(contour)
        
        for i in range(n):
            p1 = contour[i]
            p2 = contour[(i + 1) % n]
            
            if (p1.y - y) * (p2.y - y) < 0:
                t = (y - p1.y) / (p2.y - p1.y)
                x = p1.x + t * (p2.x - p1.x)
                intersections.append(Point3D(x, y, p1.z))
        
        return intersections


class SlicerEngine:
    """切片引擎主类"""
    
    def __init__(self, config: Optional[SlicerConfig] = None):
        self.config = config or SlicerConfig()
        self.triangles: List[Triangle] = []
        self.layers: List[SliceLayer] = []
        self._gcode_generator: Optional[GCodeGenerator] = None
        self._infill_generator: Optional[InfillGenerator] = None
    
    def load_stl(self, file_path: str) -> bool:
        """加载STL文件"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 检测文件类型
            if data[:5] == b'solid':
                # ASCII STL
                content = data.decode('utf-8', errors='ignore')
                self.triangles = STLParser.parse_ascii(content)
            else:
                # 二进制STL
                self.triangles = STLParser.parse_binary(data)
            
            logger.info(f"Loaded {len(self.triangles)} triangles from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load STL file: {e}")
            return False
    
    def load_stl_from_bytes(self, data: bytes) -> bool:
        """从字节数据加载STL"""
        try:
            if data[:5] == b'solid':
                content = data.decode('utf-8', errors='ignore')
                self.triangles = STLParser.parse_ascii(content)
            else:
                self.triangles = STLParser.parse_binary(data)
            
            logger.info(f"Loaded {len(self.triangles)} triangles")
            return True
        except Exception as e:
            logger.error(f"Failed to parse STL data: {e}")
            return False
    
    def get_bounding_box(self) -> Tuple[Point3D, Point3D]:
        """获取模型边界框"""
        if not self.triangles:
            return Point3D(0, 0, 0), Point3D(0, 0, 0)
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for tri in self.triangles:
            for v in [tri.v1, tri.v2, tri.v3]:
                min_x = min(min_x, v.x)
                min_y = min(min_y, v.y)
                min_z = min(min_z, v.z)
                max_x = max(max_x, v.x)
                max_y = max(max_y, v.y)
                max_z = max(max_z, v.z)
        
        return Point3D(min_x, min_y, min_z), Point3D(max_x, max_y, max_z)
    
    def center_model(self) -> None:
        """将模型居中"""
        min_pt, max_pt = self.get_bounding_box()
        center_x = (min_pt.x + max_pt.x) / 2
        center_y = (min_pt.y + max_pt.y) / 2
        
        offset_x = self.config.build_volume_x / 2 - center_x
        offset_y = self.config.build_volume_y / 2 - center_y
        
        for tri in self.triangles:
            tri.v1.x += offset_x
            tri.v1.y += offset_y
            tri.v2.x += offset_x
            tri.v2.y += offset_y
            tri.v3.x += offset_x
            tri.v3.y += offset_y
    
    def slice(self) -> List[SliceLayer]:
        """执行切片"""
        if not self.triangles:
            logger.warning("No triangles to slice")
            return []
        
        min_pt, max_pt = self.get_bounding_box()
        model_height = max_pt.z - min_pt.z
        
        # 计算层数
        first_layer_z = min_pt.z + self.config.first_layer_height
        remaining_height = model_height - self.config.first_layer_height
        num_layers = int(remaining_height / self.config.layer_height) + 1
        
        self.layers = []
        
        # 第一层
        layer = SliceLayer(z_height=first_layer_z, is_first_layer=True)
        layer.contours = self._slice_at_z(first_layer_z)
        self.layers.append(layer)
        
        # 其余层
        for i in range(1, num_layers):
            z = first_layer_z + i * self.config.layer_height
            is_top = i >= num_layers - self.config.top_layers
            is_bottom = i < self.config.bottom_layers
            
            layer = SliceLayer(z_height=z, is_top_layer=is_top)
            layer.contours = self._slice_at_z(z)
            
            # 生成填充
            if not is_bottom and not is_top:
                self._infill_generator = InfillGenerator(self.config)
                for contour in layer.contours:
                    infill = self._infill_generator.generate_infill(
                        contour, z, self.config.infill_density
                    )
                    layer.infill_paths.extend(infill)
            
            self.layers.append(layer)
        
        logger.info(f"Generated {len(self.layers)} layers")
        return self.layers
    
    def _slice_at_z(self, z: float) -> List[List[Point3D]]:
        """在指定Z高度切片"""
        segments: List[Tuple[Point3D, Point3D]] = []
        
        for tri in self.triangles:
            result = tri.intersect_plane(z)
            if result:
                segments.extend(result)
        
        # 将线段连接成轮廓
        contours = self._connect_segments(segments)
        return contours
    
    def _connect_segments(self, segments: List[Tuple[Point3D, Point3D]]) -> List[List[Point3D]]:
        """将线段连接成闭合轮廓"""
        if not segments:
            return []
        
        contours = []
        remaining = list(segments)
        tolerance = 0.001
        
        while remaining:
            contour = [remaining[0][0], remaining[0][1]]
            remaining.pop(0)
            
            changed = True
            while changed:
                changed = False
                for i, (p1, p2) in enumerate(remaining):
                    if contour[-1].distance_to(p1) < tolerance:
                        contour.append(p2)
                        remaining.pop(i)
                        changed = True
                        break
                    elif contour[-1].distance_to(p2) < tolerance:
                        contour.append(p1)
                        remaining.pop(i)
                        changed = True
                        break
                    elif contour[0].distance_to(p1) < tolerance:
                        contour.insert(0, p2)
                        remaining.pop(i)
                        changed = True
                        break
                    elif contour[0].distance_to(p2) < tolerance:
                        contour.insert(0, p1)
                        remaining.pop(i)
                        changed = True
                        break
            
            if len(contour) >= 3:
                contours.append(contour)
        
        return contours
    
    def generate_gcode(self, extruder_temp: int = 200, bed_temp: int = 60,
                       start_gcode: Optional[str] = None,
                       end_gcode: Optional[str] = None) -> SlicerResult:
        """生成G-code"""
        if not self.layers:
            self.slice()
        
        if not self.layers:
            return SlicerResult(
                gcode="",
                total_layers=0,
                estimated_time=0,
                filament_used=0,
                bounding_box=(Point3D(0, 0, 0), Point3D(0, 0, 0))
            )
        
        self._gcode_generator = GCodeGenerator(self.config)
        gcode = self._gcode_generator
        
        # 生成头部
        gcode.add_comment("Generated by SlicerEngine")
        gcode.add_comment(f"Layer height: {self.config.layer_height}mm")
        gcode.add_comment(f"Infill: {self.config.infill_density}%")
        
        # 启动G-code
        if start_gcode:
            for line in start_gcode.split('\n'):
                gcode.add_command(line)
        else:
            gcode.home()
            gcode.set_bed_temp(bed_temp, wait=True)
            gcode.set_extruder_temp(extruder_temp, wait=True)
            gcode.set_position(z=5, f=self.config.travel_speed * 60)
        
        # 生成每层
        total_filament = 0.0
        total_time = 0.0
        
        for i, layer in enumerate(self.layers):
            gcode.add_comment(f"Layer {i + 1}/{len(self.layers)}, Z={layer.z_height:.3f}")
            
            # 移动到层高度
            gcode.set_position(z=layer.z_height + self.config.z_offset,
                              f=self.config.travel_speed * 60)
            
            speed = self.config.first_layer_speed if layer.is_first_layer else self.config.print_speed
            
            # 打印轮廓
            for contour in layer.contours:
                if len(contour) < 2:
                    continue
                
                gcode.travel_to(contour[0].x, contour[0].y)
                
                for j in range(1, len(contour)):
                    p1, p2 = contour[j-1], contour[j]
                    distance = p1.distance_to(p2)
                    extrusion = self._calculate_extrusion(distance)
                    total_filament += extrusion
                    total_time += distance / speed
                    
                    gcode.extrude_to(p2.x, p2.y, extrusion, speed)
                
                # 闭合轮廓
                if len(contour) > 2:
                    distance = contour[-1].distance_to(contour[0])
                    extrusion = self._calculate_extrusion(distance)
                    total_filament += extrusion
                    total_time += distance / speed
                    gcode.extrude_to(contour[0].x, contour[0].y, extrusion, speed)
            
            # 打印填充
            for path in layer.infill_paths:
                if len(path) < 2:
                    continue
                
                gcode.travel_to(path[0].x, path[0].y)
                
                for j in range(1, len(path)):
                    p1, p2 = path[j-1], path[j]
                    distance = p1.distance_to(p2)
                    extrusion = self._calculate_extrusion(distance)
                    total_filament += extrusion
                    total_time += distance / speed
                    
                    gcode.extrude_to(p2.x, p2.y, extrusion, speed)
        
        # 结束G-code
        if end_gcode:
            for line in end_gcode.split('\n'):
                gcode.add_command(line)
        else:
            gcode.set_position(z=self.layers[-1].z_height + 10, f=self.config.travel_speed * 60)
            gcode.set_extruder_temp(0)
            gcode.set_bed_temp(0)
            gcode.home()
        
        min_pt, max_pt = self.get_bounding_box()
        
        return SlicerResult(
            gcode=gcode.get_gcode(),
            total_layers=len(self.layers),
            estimated_time=total_time / 60,  # 转换为分钟
            filament_used=total_filament,
            bounding_box=(min_pt, max_pt),
            statistics={
                "model_height": max_pt.z - min_pt.z,
                "model_width": max_pt.x - min_pt.x,
                "model_depth": max_pt.y - min_pt.y,
                "triangles": len(self.triangles)
            }
        )
    
    def _calculate_extrusion(self, distance: float) -> float:
        """计算挤出量"""
        # 简化的挤出计算
        area = self.config.line_width * self.config.layer_height
        filament_diameter = 1.75  # 假设1.75mm线材
        filament_area = math.pi * (filament_diameter / 2) ** 2
        return (area * distance) / filament_area
    
    def slice_to_gcode(self, stl_path: str, output_path: str,
                       extruder_temp: int = 200, bed_temp: int = 60) -> bool:
        """切片并保存G-code文件"""
        if not self.load_stl(stl_path):
            return False
        
        self.center_model()
        result = self.generate_gcode(extruder_temp, bed_temp)
        
        try:
            with open(output_path, 'w') as f:
                f.write(result.gcode)
            logger.info(f"G-code saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save G-code: {e}")
            return False
