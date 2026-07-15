"""
G-code Analyzer - G-code分析器
分析G-code文件，估算打印时间、材料用量等
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GCodeCommandType(Enum):
    """G-code命令类型枚举"""
    MOVE = "G0"
    LINEAR_MOVE = "G1"
    DWELL = "G4"
    HOME = "G28"
    SET_POSITION = "G92"
    ARC_MOVE_CW = "G2"
    ARC_MOVE_CCW = "G3"
    SET_TEMP = "M104"
    WAIT_TEMP = "M109"
    SET_BED_TEMP = "M140"
    WAIT_BED_TEMP = "M190"
    FAN_ON = "M106"
    FAN_OFF = "M107"
    RETRACT = "retract"
    UNRETRACT = "unretract"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class GCodeLine:
    """G-code行"""
    line_number: int
    raw_line: str
    command: str
    params: Dict[str, float] = field(default_factory=dict)
    comment: Optional[str] = None
    command_type: GCodeCommandType = GCodeCommandType.UNKNOWN
    
    def has_param(self, param: str) -> bool:
        """检查是否有参数"""
        return param in self.params
    
    def get_param(self, param: str, default: float = 0.0) -> float:
        """获取参数值"""
        return self.params.get(param, default)


@dataclass
class Position:
    """打印头位置"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    e: float = 0.0
    f: float = 0.0  # 进给率
    
    def copy(self) -> "Position":
        """复制位置"""
        return Position(self.x, self.y, self.z, self.e, self.f)


@dataclass
class LayerInfo:
    """层信息"""
    layer_number: int
    z_height: float
    line_start: int
    line_end: int
    commands_count: int = 0
    extrusion_distance: float = 0.0
    travel_distance: float = 0.0
    time_seconds: float = 0.0


@dataclass
class FilamentUsage:
    """材料用量"""
    tool_id: int
    length_mm: float = 0.0  # 长度(毫米)
    volume_mm3: float = 0.0  # 体积(立方毫米)
    weight_g: float = 0.0  # 重量(克)
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "tool_id": self.tool_id,
            "length_mm": self.length_mm,
            "volume_mm3": self.volume_mm3,
            "weight_g": self.weight_g
        }


@dataclass
class PrintStatistics:
    """打印统计"""
    total_lines: int = 0
    total_layers: int = 0
    total_time_seconds: float = 0.0
    total_time_minutes: float = 0.0
    total_time_hours: float = 0.0
    total_filament_length: float = 0.0
    total_filament_volume: float = 0.0
    total_filament_weight: float = 0.0
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0, 0, 0), (0, 0, 0))
    min_x: float = 0.0
    max_x: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0
    min_z: float = 0.0
    max_z: float = 0.0
    travel_distance: float = 0.0
    extrusion_distance: float = 0.0
    retract_count: int = 0
    unretract_count: int = 0
    layer_height: float = 0.0
    avg_layer_time: float = 0.0
    max_layer_time: float = 0.0
    min_layer_time: float = float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_lines": self.total_lines,
            "total_layers": self.total_layers,
            "total_time_seconds": self.total_time_seconds,
            "total_time_minutes": self.total_time_minutes,
            "total_time_hours": self.total_time_hours,
            "total_filament_length": self.total_filament_length,
            "total_filament_volume": self.total_filament_volume,
            "total_filament_weight": self.total_filament_weight,
            "bounding_box": {
                "min": {"x": self.min_x, "y": self.min_y, "z": self.min_z},
                "max": {"x": self.max_x, "y": self.max_y, "z": self.max_z}
            },
            "dimensions": {
                "x": self.max_x - self.min_x,
                "y": self.max_y - self.min_y,
                "z": self.max_z - self.min_z
            },
            "travel_distance": self.travel_distance,
            "extrusion_distance": self.extrusion_distance,
            "retract_count": self.retract_count,
            "unretract_count": self.unretract_count,
            "layer_height": self.layer_height,
            "avg_layer_time": self.avg_layer_time,
            "max_layer_time": self.max_layer_time,
            "min_layer_time": self.min_layer_time if self.min_layer_time != float('inf') else 0
        }


class GCodeParser:
    """G-code解析器"""
    
    PARAM_PATTERN = re.compile(r'([A-Z])(-?\d*\.?\d+)')
    COMMENT_PATTERN = re.compile(r';(.*)$')
    
    def parse_line(self, line: str, line_number: int) -> GCodeLine:
        """解析单行G-code"""
        line = line.strip()
        
        if not line or line.startswith(';'):
            return GCodeLine(
                line_number=line_number,
                raw_line=line,
                command="",
                comment=line[1:] if line.startswith(';') else None,
                command_type=GCodeCommandType.COMMENT
            )
        
        # 提取注释
        comment = None
        comment_match = self.COMMENT_PATTERN.search(line)
        if comment_match:
            comment = comment_match.group(1).strip()
            line = line[:comment_match.start()].strip()
        
        # 提取命令和参数
        parts = line.split()
        if not parts:
            return GCodeLine(
                line_number=line_number,
                raw_line=line,
                command="",
                comment=comment,
                command_type=GCodeCommandType.COMMENT
            )
        
        command = parts[0].upper()
        params = {}
        
        for part in parts[1:]:
            match = self.PARAM_PATTERN.match(part.upper())
            if match:
                params[match.group(1)] = float(match.group(2))
        
        # 对于G0/G1，参数可能在同一字符串中
        for match in self.PARAM_PATTERN.finditer(line[len(command):]):
            params[match.group(1)] = float(match.group(2))
        
        command_type = self._get_command_type(command, params)
        
        return GCodeLine(
            line_number=line_number,
            raw_line=line,
            command=command,
            params=params,
            comment=comment,
            command_type=command_type
        )
    
    def _get_command_type(self, command: str, params: Dict[str, float]) -> GCodeCommandType:
        """获取命令类型"""
        if command in ("G0", "G00"):
            return GCodeCommandType.MOVE
        elif command in ("G1", "G01"):
            return GCodeCommandType.LINEAR_MOVE
        elif command in ("G2", "G02"):
            return GCodeCommandType.ARC_MOVE_CW
        elif command in ("G3", "G03"):
            return GCodeCommandType.ARC_MOVE_CCW
        elif command == "G4":
            return GCodeCommandType.DWELL
        elif command == "G28":
            return GCodeCommandType.HOME
        elif command == "G92":
            return GCodeCommandType.SET_POSITION
        elif command == "M104":
            return GCodeCommandType.SET_TEMP
        elif command == "M109":
            return GCodeCommandType.WAIT_TEMP
        elif command == "M140":
            return GCodeCommandType.SET_BED_TEMP
        elif command == "M190":
            return GCodeCommandType.WAIT_BED_TEMP
        elif command == "M106":
            return GCodeCommandType.FAN_ON
        elif command == "M107":
            return GCodeCommandType.FAN_OFF
        else:
            return GCodeCommandType.UNKNOWN
    
    def parse_file(self, file_path: str) -> List[GCodeLine]:
        """解析G-code文件"""
        lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f, 1):
                    lines.append(self.parse_line(line, i))
        except Exception as e:
            logger.error(f"Failed to parse G-code file: {e}")
        
        return lines
    
    def parse_string(self, content: str) -> List[GCodeLine]:
        """解析G-code字符串"""
        lines = []
        
        for i, line in enumerate(content.split('\n'), 1):
            lines.append(self.parse_line(line, i))
        
        return lines


class GCodeAnalyzer:
    """G-code分析器主类"""
    
    def __init__(self, filament_diameter: float = 1.75,
                 filament_density: float = 1.24,  # PLA密度 g/cm³
                 default_feedrate: float = 1500):  # 默认进给率 mm/min
        self.filament_diameter = filament_diameter
        self.filament_density = filament_density
        self.default_feedrate = default_feedrate
        self.parser = GCodeParser()
        
        # 计算横截面积
        self.filament_area = math.pi * (filament_diameter / 2) ** 2
    
    def analyze_file(self, file_path: str) -> PrintStatistics:
        """分析G-code文件"""
        lines = self.parser.parse_file(file_path)
        return self._analyze(lines)
    
    def analyze_string(self, content: str) -> PrintStatistics:
        """分析G-code字符串"""
        lines = self.parser.parse_string(content)
        return self._analyze(lines)
    
    def _analyze(self, lines: List[GCodeLine]) -> PrintStatistics:
        """执行分析"""
        stats = PrintStatistics(total_lines=len(lines))
        
        current_pos = Position()
        layers: List[LayerInfo] = []
        current_layer: Optional[LayerInfo] = None
        last_z = 0.0
        layer_z_values: List[float] = []
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        total_time = 0.0
        total_extrusion = 0.0
        total_travel = 0.0
        retract_count = 0
        unretract_count = 0
        
        for gcode_line in lines:
            if gcode_line.command_type == GCodeCommandType.COMMENT:
                continue
            
            # 处理移动命令
            if gcode_line.command_type in (GCodeCommandType.MOVE, GCodeCommandType.LINEAR_MOVE):
                new_pos = self._calculate_new_position(current_pos, gcode_line)
                
                # 计算距离
                dx = new_pos.x - current_pos.x
                dy = new_pos.y - current_pos.y
                dz = new_pos.z - current_pos.z
                de = new_pos.e - current_pos.e
                
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # 判断是挤出还是移动
                if de > 0:
                    total_extrusion += distance
                    
                    # 更新边界框
                    min_x = min(min_x, new_pos.x)
                    max_x = max(max_x, new_pos.x)
                    min_y = min(min_y, new_pos.y)
                    max_y = max(max_y, new_pos.y)
                else:
                    total_travel += distance
                
                # 计算时间
                feedrate = new_pos.f if new_pos.f > 0 else current_pos.f
                if feedrate > 0 and distance > 0:
                    time_seconds = (distance / feedrate) * 60
                    total_time += time_seconds
                    
                    if current_layer:
                        current_layer.time_seconds += time_seconds
                
                # 检测层变化
                if new_pos.z != last_z and new_pos.z > last_z:
                    if current_layer:
                        current_layer.line_end = gcode_line.line_number - 1
                        layers.append(current_layer)
                    
                    current_layer = LayerInfo(
                        layer_number=len(layers) + 1,
                        z_height=new_pos.z,
                        line_start=gcode_line.line_number
                    )
                    layer_z_values.append(new_pos.z)
                    last_z = new_pos.z
                
                # 更新Z边界
                if new_pos.z > 0:
                    min_z = min(min_z, new_pos.z)
                    max_z = max(max_z, new_pos.z)
                
                current_pos = new_pos
            
            # 处理停留命令
            elif gcode_line.command_type == GCodeCommandType.DWELL:
                if gcode_line.has_param('P'):
                    total_time += gcode_line.get_param('P') / 1000  # P是毫秒
                if gcode_line.has_param('S'):
                    total_time += gcode_line.get_param('S')  # S是秒
            
            # 处理归零
            elif gcode_line.command_type == GCodeCommandType.HOME:
                current_pos = Position(f=current_pos.f)
            
            # 处理位置设置
            elif gcode_line.command_type == GCodeCommandType.SET_POSITION:
                if gcode_line.has_param('X'):
                    current_pos.x = gcode_line.get_param('X')
                if gcode_line.has_param('Y'):
                    current_pos.y = gcode_line.get_param('Y')
                if gcode_line.has_param('Z'):
                    current_pos.z = gcode_line.get_param('Z')
                if gcode_line.has_param('E'):
                    current_pos.e = gcode_line.get_param('E')
            
            # 检测回抽/取消回抽
            if gcode_line.command_type == GCodeCommandType.LINEAR_MOVE:
                de = current_pos.e - (gcode_line.get_param('E', current_pos.e))
                if de < -0.5:  # 回抽
                    retract_count += 1
                elif de > 0.5 and retract_count > 0:  # 取消回抽
                    unretract_count += 1
        
        # 完成最后一层
        if current_layer:
            current_layer.line_end = len(lines)
            layers.append(current_layer)
        
        # 计算材料用量
        total_filament_length = abs(current_pos.e)
        total_filament_volume = total_filament_length * self.filament_area
        total_filament_weight = total_filament_volume * self.filament_density / 1000  # 转换为克
        
        # 计算层高
        layer_height = 0.0
        if len(layer_z_values) >= 2:
            layer_heights = [layer_z_values[i] - layer_z_values[i-1] 
                           for i in range(1, len(layer_z_values))]
            if layer_heights:
                layer_height = sum(layer_heights) / len(layer_heights)
        
        # 计算层时间统计
        layer_times = [layer.time_seconds for layer in layers if layer.time_seconds > 0]
        avg_layer_time = sum(layer_times) / len(layer_times) if layer_times else 0
        max_layer_time = max(layer_times) if layer_times else 0
        min_layer_time = min(layer_times) if layer_times else 0
        
        # 更新统计
        stats.total_layers = len(layers)
        stats.total_time_seconds = total_time
        stats.total_time_minutes = total_time / 60
        stats.total_time_hours = total_time / 3600
        stats.total_filament_length = total_filament_length
        stats.total_filament_volume = total_filament_volume
        stats.total_filament_weight = total_filament_weight
        stats.min_x = min_x if min_x != float('inf') else 0
        stats.max_x = max_x if max_x != float('-inf') else 0
        stats.min_y = min_y if min_y != float('inf') else 0
        stats.max_y = max_y if max_y != float('-inf') else 0
        stats.min_z = min_z if min_z != float('inf') else 0
        stats.max_z = max_z if max_z != float('-inf') else 0
        stats.travel_distance = total_travel
        stats.extrusion_distance = total_extrusion
        stats.retract_count = retract_count
        stats.unretract_count = unretract_count
        stats.layer_height = layer_height
        stats.avg_layer_time = avg_layer_time
        stats.max_layer_time = max_layer_time
        stats.min_layer_time = min_layer_time
        
        return stats
    
    def _calculate_new_position(self, current: Position, 
                                gcode: GCodeLine) -> Position:
        """计算新位置"""
        new_pos = current.copy()
        
        if gcode.has_param('X'):
            new_pos.x = gcode.get_param('X')
        if gcode.has_param('Y'):
            new_pos.y = gcode.get_param('Y')
        if gcode.has_param('Z'):
            new_pos.z = gcode.get_param('Z')
        if gcode.has_param('E'):
            new_pos.e = gcode.get_param('E')
        if gcode.has_param('F'):
            new_pos.f = gcode.get_param('F')
        
        return new_pos
    
    def get_layers(self, file_path: str) -> List[LayerInfo]:
        """获取层信息"""
        lines = self.parser.parse_file(file_path)
        return self._extract_layers(lines)
    
    def get_layers_from_string(self, content: str) -> List[LayerInfo]:
        """从字符串获取层信息"""
        lines = self.parser.parse_string(content)
        return self._extract_layers(lines)
    
    def _extract_layers(self, lines: List[GCodeLine]) -> List[LayerInfo]:
        """提取层信息"""
        layers: List[LayerInfo] = []
        current_layer: Optional[LayerInfo] = None
        last_z = 0.0
        current_pos = Position()
        
        for gcode_line in lines:
            if gcode_line.command_type in (GCodeCommandType.MOVE, GCodeCommandType.LINEAR_MOVE):
                new_z = gcode_line.get_param('Z', current_pos.z)
                
                if new_z != last_z and new_z > last_z:
                    if current_layer:
                        current_layer.line_end = gcode_line.line_number - 1
                        layers.append(current_layer)
                    
                    current_layer = LayerInfo(
                        layer_number=len(layers) + 1,
                        z_height=new_z,
                        line_start=gcode_line.line_number
                    )
                    last_z = new_z
                
                if gcode_line.has_param('X'):
                    current_pos.x = gcode_line.get_param('X')
                if gcode_line.has_param('Y'):
                    current_pos.y = gcode_line.get_param('Y')
                if gcode_line.has_param('Z'):
                    current_pos.z = gcode_line.get_param('Z')
                if gcode_line.has_param('E'):
                    current_pos.e = gcode_line.get_param('E')
                if gcode_line.has_param('F'):
                    current_pos.f = gcode_line.get_param('F')
        
        if current_layer:
            current_layer.line_end = len(lines)
            layers.append(current_layer)
        
        return layers
    
    def estimate_print_time(self, file_path: str,
                            acceleration: float = 500,  # mm/s²
                            jerk: float = 8.0) -> float:  # mm/s
        """估算打印时间（考虑加速度）"""
        stats = self.analyze_file(file_path)
        
        # 简化的时间估算，考虑加速/减速
        # 实际时间会比简单距离/速度估算更长
        acceleration_factor = 1.2  # 加速度补偿因子
        
        return stats.total_time_seconds * acceleration_factor
    
    def get_filament_usage_by_tool(self, file_path: str) -> Dict[int, FilamentUsage]:
        """获取各挤出头的材料用量"""
        lines = self.parser.parse_file(file_path)
        
        usages: Dict[int, FilamentUsage] = defaultdict(lambda: FilamentUsage(tool_id=0))
        current_tool = 0
        current_e = 0.0
        
        for gcode_line in lines:
            # 检测工具切换
            if gcode_line.command == "T":
                if gcode_line.params:
                    current_tool = int(gcode_line.get_param('T', 0))
            
            # 累积挤出量
            if gcode_line.command_type == GCodeCommandType.LINEAR_MOVE:
                if gcode_line.has_param('E'):
                    new_e = gcode_line.get_param('E')
                    delta_e = new_e - current_e
                    
                    if delta_e > 0:
                        usages[current_tool].length_mm += delta_e
                    
                    current_e = new_e
            
            # 处理位置重置
            elif gcode_line.command_type == GCodeCommandType.SET_POSITION:
                if gcode_line.has_param('E'):
                    current_e = gcode_line.get_param('E')
        
        # 计算体积和重量
        for tool_id, usage in usages.items():
            usage.tool_id = tool_id
            usage.volume_mm3 = usage.length_mm * self.filament_area
            usage.weight_g = usage.volume_mm3 * self.filament_density / 1000
        
        return dict(usages)
    
    def validate_gcode(self, file_path: str) -> Tuple[bool, List[str]]:
        """验证G-code文件"""
        lines = self.parser.parse_file(file_path)
        errors = []
        
        # 检查基本结构
        has_movement = False
        has_extrusion = False
        has_temp_set = False
        
        for gcode_line in lines:
            if gcode_line.command_type in (GCodeCommandType.MOVE, GCodeCommandType.LINEAR_MOVE):
                has_movement = True
                if gcode_line.has_param('E'):
                    has_extrusion = True
            
            if gcode_line.command_type in (GCodeCommandType.SET_TEMP, GCodeCommandType.WAIT_TEMP):
                has_temp_set = True
        
        if not has_movement:
            errors.append("No movement commands found")
        if not has_extrusion:
            errors.append("No extrusion commands found")
        
        # 检查温度设置
        if not has_temp_set:
            errors.append("Warning: No temperature settings found")
        
        return len(errors) == 0, errors
    
    def get_print_summary(self, file_path: str) -> Dict[str, Any]:
        """获取打印摘要"""
        stats = self.analyze_file(file_path)
        layers = self.get_layers(file_path)
        filament_usage = self.get_filament_usage_by_tool(file_path)
        is_valid, errors = self.validate_gcode(file_path)
        
        return {
            "valid": is_valid,
            "errors": errors,
            "statistics": stats.to_dict(),
            "layers": {
                "total": len(layers),
                "first_z": layers[0].z_height if layers else 0,
                "last_z": layers[-1].z_height if layers else 0
            },
            "filament": {
                f"tool_{tool}": usage.to_dict() 
                for tool, usage in filament_usage.items()
            },
            "estimated_time": {
                "seconds": stats.total_time_seconds,
                "minutes": stats.total_time_minutes,
                "hours": stats.total_time_hours,
                "formatted": self._format_time(stats.total_time_seconds)
            }
        }
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
