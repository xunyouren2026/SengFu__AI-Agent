"""
OpenSCAD Wrapper - OpenSCAD封装
用于程序化3D建模
"""

import subprocess
import tempfile
import os
import re
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OpenSCADError(Exception):
    """OpenSCAD错误"""
    pass


class ExportFormat(Enum):
    """导出格式枚举"""
    STL = "stl"
    OFF = "off"
    AMF = "amf"
    DXF = "dxf"
    SVG = "svg"
    PNG = "png"
    OBJ = "obj"
    _3MF = "3mf"


@dataclass
class Point3D:
    """三维坐标点"""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class RenderResult:
    """渲染结果"""
    success: bool
    output_path: Optional[str] = None
    output_data: Optional[bytes] = None
    error_message: Optional[str] = None
    geometry_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryInfo:
    """几何信息"""
    vertices: int = 0
    faces: int = 0
    edges: int = 0
    volume: float = 0.0
    bounding_box: Tuple[Point3D, Point3D] = (Point3D(0, 0, 0), Point3D(0, 0, 0))


class OpenSCADBuilder:
    """OpenSCAD代码构建器"""
    
    def __init__(self):
        self._code_lines: List[str] = []
        self._indent_level: int = 0
        self._variables: Dict[str, Any] = {}
    
    def _indent(self) -> str:
        """获取缩进"""
        return "    " * self._indent_level
    
    def add_comment(self, comment: str) -> "OpenSCADBuilder":
        """添加注释"""
        self._code_lines.append(f"{self._indent()}// {comment}")
        return self
    
    def add_raw(self, code: str) -> "OpenSCADBuilder":
        """添加原始代码"""
        self._code_lines.append(f"{self._indent()}{code}")
        return self
    
    def set_variable(self, name: str, value: Any) -> "OpenSCADBuilder":
        """设置变量"""
        self._variables[name] = value
        if isinstance(value, bool):
            value_str = "true" if value else "false"
        elif isinstance(value, str):
            value_str = f'"{value}"'
        elif isinstance(value, (list, tuple)):
            value_str = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            value_str = str(value)
        
        self._code_lines.append(f"{self._indent()}{name} = {value_str};")
        return self
    
    def _begin_module(self, name: str, params: Optional[Dict[str, Any]] = None) -> "OpenSCADBuilder":
        """开始模块定义"""
        param_str = ""
        if params:
            param_str = "(" + ", ".join(f"{k}={v}" if v is not None else k for k, v in params.items()) + ")"
        
        self._code_lines.append(f"{self._indent()}module {name}{param_str} {{")
        self._indent_level += 1
        return self
    
    def _end_module(self) -> "OpenSCADBuilder":
        """结束模块定义"""
        self._indent_level -= 1
        self._code_lines.append(f"{self._indent()}}}")
        return self
    
    def cube(self, size: Union[float, Tuple[float, float, float], List[float]] = 1,
             center: bool = False) -> "OpenSCADBuilder":
        """创建立方体"""
        if isinstance(size, (tuple, list)):
            size_str = f"[{size[0]}, {size[1]}, {size[2]}]"
        else:
            size_str = str(size)
        
        self._code_lines.append(f"{self._indent()}cube({size_str}, center={str(center).lower()});")
        return self
    
    def sphere(self, radius: float = 1, segments: int = 16) -> "OpenSCADBuilder":
        """创建球体"""
        self._code_lines.append(f"{self._indent()}sphere(r={radius}, $fn={segments});")
        return self
    
    def cylinder(self, height: float = 1, radius: Optional[float] = None,
                 radius1: Optional[float] = None, radius2: Optional[float] = None,
                 segments: int = 16, center: bool = False) -> "OpenSCADBuilder":
        """创建圆柱体"""
        params = [f"h={height}"]
        
        if radius is not None:
            params.append(f"r={radius}")
        elif radius1 is not None and radius2 is not None:
            params.append(f"r1={radius1}")
            params.append(f"r2={radius2}")
        
        params.append(f"$fn={segments}")
        params.append(f"center={str(center).lower()}")
        
        self._code_lines.append(f"{self._indent()}cylinder({', '.join(params)});")
        return self
    
    def cone(self, height: float = 1, radius_bottom: float = 1,
             radius_top: float = 0, segments: int = 16, center: bool = False) -> "OpenSCADBuilder":
        """创建圆锥体"""
        return self.cylinder(height=height, radius1=radius_bottom, radius2=radius_top,
                            segments=segments, center=center)
    
    def polyhedron(self, points: List[Tuple[float, float, float]],
                   faces: List[List[int]], convexity: int = 1) -> "OpenSCADBuilder":
        """创建多面体"""
        points_str = "[" + ", ".join(f"[{p[0]}, {p[1]}, {p[2]}]" for p in points) + "]"
        faces_str = "[" + ", ".join("[" + ", ".join(str(f) for f in face) + "]" for face in faces) + "]"
        
        self._code_lines.append(f"{self._indent()}polyhedron(points={points_str}, faces={faces_str}, convexity={convexity});")
        return self
    
    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> "OpenSCADBuilder":
        """开始平移变换"""
        self._code_lines.append(f"{self._indent()}translate([{x}, {y}, {z}]) {{")
        self._indent_level += 1
        return self
    
    def rotate(self, x: float = 0, y: float = 0, z: float = 0) -> "OpenSCADBuilder":
        """开始旋转变换"""
        self._code_lines.append(f"{self._indent()}rotate([{x}, {y}, {z}]) {{")
        self._indent_level += 1
        return self
    
    def scale(self, x: float = 1, y: float = 1, z: float = 1) -> "OpenSCADBuilder":
        """开始缩放变换"""
        self._code_lines.append(f"{self._indent()}scale([{x}, {y}, {z}]) {{")
        self._indent_level += 1
        return self
    
    def mirror(self, x: float = 0, y: float = 0, z: float = 0) -> "OpenSCADBuilder":
        """开始镜像变换"""
        self._code_lines.append(f"{self._indent()}mirror([{x}, {y}, {z}]) {{")
        self._indent_level += 1
        return self
    
    def resize(self, x: float, y: float, z: float, auto: bool = False) -> "OpenSCADBuilder":
        """开始调整大小"""
        auto_str = str(auto).lower()
        self._code_lines.append(f"{self._indent()}resize([{x}, {y}, {z}], auto={auto_str}) {{")
        self._indent_level += 1
        return self
    
    def multmatrix(self, matrix: List[List[float]]) -> "OpenSCADBuilder":
        """开始矩阵变换"""
        matrix_str = "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in matrix) + "]"
        self._code_lines.append(f"{self._indent()}multmatrix({matrix_str}) {{")
        self._indent_level += 1
        return self
    
    def end(self) -> "OpenSCADBuilder":
        """结束变换"""
        self._indent_level -= 1
        self._code_lines.append(f"{self._indent()}}}")
        return self
    
    def union(self) -> "OpenSCADBuilder":
        """开始并集"""
        self._code_lines.append(f"{self._indent()}union() {{")
        self._indent_level += 1
        return self
    
    def difference(self) -> "OpenSCADBuilder":
        """开始差集"""
        self._code_lines.append(f"{self._indent()}difference() {{")
        self._indent_level += 1
        return self
    
    def intersection(self) -> "OpenSCADBuilder":
        """开始交集"""
        self._code_lines.append(f"{self._indent()}intersection() {{")
        self._indent_level += 1
        return self
    
    def hull(self) -> "OpenSCADBuilder":
        """开始凸包"""
        self._code_lines.append(f"{self._indent()}hull() {{")
        self._indent_level += 1
        return self
    
    def minkowski(self) -> "OpenSCADBuilder":
        """开始闵可夫斯基和"""
        self._code_lines.append(f"{self._indent()}minkowski() {{")
        self._indent_level += 1
        return self
    
    def linear_extrude(self, height: float, twist: float = 0,
                       scale: float = 1, segments: int = 10,
                       center: bool = False) -> "OpenSCADBuilder":
        """开始线性拉伸"""
        self._code_lines.append(
            f"{self._indent()}linear_extrude(height={height}, twist={twist}, "
            f"scale={scale}, segments={segments}, center={str(center).lower()}) {{"
        )
        self._indent_level += 1
        return self
    
    def rotate_extrude(self, angle: float = 360, segments: int = 10) -> "OpenSCADBuilder":
        """开始旋转拉伸"""
        self._code_lines.append(f"{self._indent()}rotate_extrude(angle={angle}, $fn={segments}) {{")
        self._indent_level += 1
        return self
    
    def import_file(self, file_path: str, convexity: int = 1) -> "OpenSCADBuilder":
        """导入文件"""
        self._code_lines.append(f'{self._indent()}import("{file_path}", convexity={convexity});')
        return self
    
    def surface(self, file_path: str, center: bool = False,
                invert: bool = False, convexity: int = 1) -> "OpenSCADBuilder":
        """从图像创建表面"""
        self._code_lines.append(
            f'{self._indent()}surface("{file_path}", center={str(center).lower()}, '
            f'invert={str(invert).lower()}, convexity={convexity});'
        )
        return self
    
    def text(self, text_str: str, size: float = 10, font: str = "",
             halign: str = "left", valign: str = "baseline",
             spacing: float = 1, direction: str = "ltr",
             language: str = "en", script: str = "",
             segments: int = 16) -> "OpenSCADBuilder":
        """创建文本"""
        params = [f'text="{text_str}"', f"size={size}"]
        
        if font:
            params.append(f'font="{font}"')
        
        params.extend([
            f'halign="{halign}"',
            f'valign="{valign}"',
            f"spacing={spacing}",
            f'direction="{direction}"',
            f'language="{language}"'
        ])
        
        if script:
            params.append(f'script="{script}"')
        
        params.append(f"$fn={segments}")
        
        self._code_lines.append(f"{self._indent()}text({', '.join(params)});")
        return self
    
    def projection(self, cut: bool = False) -> "OpenSCADBuilder":
        """开始投影"""
        self._code_lines.append(f"{self._indent()}projection(cut={str(cut).lower()}) {{")
        self._indent_level += 1
        return self
    
    def render(self, convexity: int = 1) -> "OpenSCADBuilder":
        """开始强制渲染"""
        self._code_lines.append(f"{self._indent()}render(convexity={convexity}) {{")
        self._indent_level += 1
        return self
    
    def color(self, color_name: str) -> "OpenSCADBuilder":
        """设置颜色"""
        self._code_lines.append(f'{self._indent()}color("{color_name}") {{')
        self._indent_level += 1
        return self
    
    def for_loop(self, var: str, start: int, end: int, step: int = 1) -> "OpenSCADBuilder":
        """开始for循环"""
        if step == 1:
            self._code_lines.append(f"{self._indent()}for({var} = [{start}:{end}]) {{")
        else:
            self._code_lines.append(f"{self._indent()}for({var} = [{start}:{step}:{end}]) {{")
        self._indent_level += 1
        return self
    
    def if_statement(self, condition: str) -> "OpenSCADBuilder":
        """开始if语句"""
        self._code_lines.append(f"{self._indent()}if({condition}) {{")
        self._indent_level += 1
        return self
    
    def else_statement(self) -> "OpenSCADBuilder":
        """else语句"""
        self._indent_level -= 1
        self._code_lines.append(f"{self._indent()}}} else {{")
        self._indent_level += 1
        return self
    
    def echo(self, *args: Any) -> "OpenSCADBuilder":
        """添加echo语句"""
        args_str = ", ".join(str(arg) for arg in args)
        self._code_lines.append(f"{self._indent()}echo({args_str});")
        return self
    
    def build(self) -> str:
        """构建代码"""
        return "\n".join(self._code_lines)
    
    def __str__(self) -> str:
        return self.build()


class OpenSCADWrapper:
    """OpenSCAD封装主类"""
    
    def __init__(self, openscad_path: str = "openscad"):
        self.openscad_path = openscad_path
        self._builder = OpenSCADBuilder()
        self._check_openscad()
    
    def _check_openscad(self) -> bool:
        """检查OpenSCAD是否可用"""
        try:
            result = subprocess.run(
                [self.openscad_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            logger.warning("OpenSCAD not found or not executable")
            return False
    
    @property
    def builder(self) -> OpenSCADBuilder:
        """获取构建器"""
        return self._builder
    
    def new_builder(self) -> OpenSCADBuilder:
        """创建新构建器"""
        return OpenSCADBuilder()
    
    def render(self, code: str, output_path: str,
               format: ExportFormat = ExportFormat.STL,
               camera: Optional[Tuple[float, float, float, float, float, float, float]] = None,
               img_width: int = 512, img_height: int = 512,
               preview: bool = False,
               render_all: bool = False,
               quiet: bool = True,
               timeout: int = 300) -> RenderResult:
        """渲染模型"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False) as f:
            f.write(code)
            scad_path = f.name
        
        try:
            cmd = [self.openscad_path]
            
            if quiet:
                cmd.append("-q")
            
            if preview:
                cmd.extend(["--preview", ""])
            elif render_all:
                cmd.append("--render")
            
            cmd.extend(["-o", output_path])
            
            if format == ExportFormat.PNG and camera:
                tx, ty, tz, rx, ry, rz, dist = camera
                cmd.extend([
                    "--camera",
                    f"{tx},{ty},{tz},{rx},{ry},{rz},{dist}",
                    "--imgsize", f"{img_width},{img_height}"
                ])
            
            cmd.append(scad_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # 读取输出文件
                output_data = None
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        output_data = f.read()
                
                return RenderResult(
                    success=True,
                    output_path=output_path,
                    output_data=output_data
                )
            else:
                return RenderResult(
                    success=False,
                    error_message=result.stderr or result.stdout
                )
        
        except subprocess.TimeoutExpired:
            return RenderResult(
                success=False,
                error_message=f"Render timeout after {timeout} seconds"
            )
        except Exception as e:
            return RenderResult(
                success=False,
                error_message=str(e)
            )
        finally:
            if os.path.exists(scad_path):
                os.unlink(scad_path)
    
    def render_to_stl(self, code: str, output_path: str,
                      timeout: int = 300) -> RenderResult:
        """渲染为STL"""
        return self.render(code, output_path, ExportFormat.STL, timeout=timeout)
    
    def render_to_obj(self, code: str, output_path: str,
                      timeout: int = 300) -> RenderResult:
        """渲染为OBJ"""
        return self.render(code, output_path, ExportFormat.OBJ, timeout=timeout)
    
    def render_to_png(self, code: str, output_path: str,
                      camera: Optional[Tuple[float, float, float, float, float, float, float]] = None,
                      width: int = 512, height: int = 512,
                      timeout: int = 300) -> RenderResult:
        """渲染为PNG图片"""
        return self.render(
            code, output_path, ExportFormat.PNG,
            camera=camera, img_width=width, img_height=height,
            timeout=timeout
        )
    
    def render_to_svg(self, code: str, output_path: str,
                      timeout: int = 300) -> RenderResult:
        """渲染为SVG"""
        return self.render(code, output_path, ExportFormat.SVG, timeout=timeout)
    
    def get_geometry_info(self, code: str) -> GeometryInfo:
        """获取几何信息"""
        # 使用csginfo获取几何信息
        with tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False) as f:
            f.write(code)
            scad_path = f.name
        
        try:
            # 尝试导出为OFF格式获取面和顶点数
            with tempfile.NamedTemporaryFile(suffix='.off', delete=False) as f:
                off_path = f.name
            
            result = subprocess.run(
                [self.openscad_path, "-o", off_path, scad_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            info = GeometryInfo()
            
            if result.returncode == 0 and os.path.exists(off_path):
                with open(off_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:
                    # OFF格式: 第一行是"OFF"，第二行是"vertices faces edges"
                    parts = lines[1].strip().split()
                    if len(parts) >= 3:
                        info.vertices = int(parts[0])
                        info.faces = int(parts[1])
                        info.edges = int(parts[2])
                
                os.unlink(off_path)
            
            return info
        
        except Exception as e:
            logger.error(f"Failed to get geometry info: {e}")
            return GeometryInfo()
        finally:
            if os.path.exists(scad_path):
                os.unlink(scad_path)
    
    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """验证代码"""
        errors = []
        
        # 基本语法检查
        brace_count = 0
        bracket_count = 0
        paren_count = 0
        
        for i, char in enumerate(code):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
            elif char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            
            if brace_count < 0:
                errors.append(f"Unmatched closing brace at position {i}")
            if bracket_count < 0:
                errors.append(f"Unmatched closing bracket at position {i}")
            if paren_count < 0:
                errors.append(f"Unmatched closing parenthesis at position {i}")
        
        if brace_count != 0:
            errors.append(f"Unmatched braces: {brace_count} remaining")
        if bracket_count != 0:
            errors.append(f"Unmatched brackets: {bracket_count} remaining")
        if paren_count != 0:
            errors.append(f"Unmatched parentheses: {paren_count} remaining")
        
        return len(errors) == 0, errors
    
    def execute_code(self, code: str, output_path: str,
                     format: ExportFormat = ExportFormat.STL) -> RenderResult:
        """执行代码并导出"""
        is_valid, errors = self.validate_code(code)
        
        if not is_valid:
            return RenderResult(
                success=False,
                error_message="Code validation failed: " + "; ".join(errors)
            )
        
        return self.render(code, output_path, format)


# 预定义模块
class PredefinedModules:
    """预定义的常用模块"""
    
    @staticmethod
    def rounded_cube(width: float, depth: float, height: float,
                     radius: float, segments: int = 16) -> str:
        """圆角立方体"""
        builder = OpenSCADBuilder()
        builder.add_comment("Rounded cube")
        builder.difference()
        builder.cube([width, depth, height])
        # 四个角的圆角
        for x in [radius, width - radius]:
            for y in [radius, depth - radius]:
                builder.translate(x, y, -0.01)
                builder.cylinder(height=height + 0.02, radius=radius, segments=segments)
                builder.end()
        builder.end()
        return builder.build()
    
    @staticmethod
    def hollow_cylinder(outer_radius: float, inner_radius: float,
                        height: float, segments: int = 32) -> str:
        """空心圆柱"""
        builder = OpenSCADBuilder()
        builder.add_comment("Hollow cylinder")
        builder.difference()
        builder.cylinder(height=height, radius=outer_radius, segments=segments)
        builder.translate(0, 0, -0.01)
        builder.cylinder(height=height + 0.02, radius=inner_radius, segments=segments)
        builder.end()
        builder.end()
        return builder.build()
    
    @staticmethod
    def torus(major_radius: float, minor_radius: float,
              major_segments: int = 32, minor_segments: int = 16) -> str:
        """圆环"""
        builder = OpenSCADBuilder()
        builder.add_comment("Torus")
        builder.rotate_extrude(segments=major_segments)
        builder.translate(major_radius, 0, 0)
        builder.circle(minor_radius, minor_segments)
        builder.end()
        builder.end()
        return builder.build()
    
    @staticmethod
    def gear(module: float, teeth: int, thickness: float,
             bore_diameter: float = 0) -> str:
        """简化齿轮"""
        pitch_diameter = module * teeth
        outer_diameter = pitch_diameter + 2 * module
        root_diameter = pitch_diameter - 2.5 * module
        
        builder = OpenSCADBuilder()
        builder.add_comment(f"Gear: {teeth} teeth")
        builder.difference()
        
        # 齿轮主体
        builder.cylinder(height=thickness, radius=outer_diameter / 2, segments=teeth * 2)
        
        # 齿槽
        tooth_angle = 360 / teeth
        for i in range(teeth):
            builder.rotate(0, 0, i * tooth_angle)
            builder.translate(pitch_diameter / 2, 0, -0.01)
            builder.cube([module * 2, module, thickness + 0.02])
            builder.end()
            builder.end()
        
        # 轴孔
        if bore_diameter > 0:
            builder.translate(0, 0, -0.01)
            builder.cylinder(height=thickness + 0.02, radius=bore_diameter / 2)
            builder.end()
        
        builder.end()
        return builder.build()
    
    @staticmethod
    def spring(radius: float, wire_radius: float, pitch: float,
               turns: int, segments: int = 32) -> str:
        """弹簧"""
        builder = OpenSCADBuilder()
        builder.add_comment("Spring")
        
        builder.for_loop("i", 0, turns * segments - 1)
        angle = "i * 360 / " + str(segments)
        builder.add_raw(f"    let(angle = {angle},")
        builder.add_raw(f"        z = i * {pitch} / {segments})")
        builder.translate(f"radius * cos(angle)", f"radius * sin(angle)", "z")
        builder.sphere(wire_radius, segments // 2)
        builder.end()
        builder.end()
        
        return builder.build()
