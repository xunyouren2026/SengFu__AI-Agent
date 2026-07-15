"""
CNC G-Code Generation Module

Provides toolpath generation from vector paths, feed rate calculation,
spindle speed control, roughing/finishing passes, tool change sequences,
and safety checks for CNC machining.
"""

from __future__ import annotations

import math
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MM_PER_INCH = 25.4
DEFAULT_SAFE_Z = 50.0  # mm
DEFAULT_RAPID_SPEED = 5000.0  # mm/min
DEFAULT_UNITS = "G21"  # Metric (mm)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MotionMode(Enum):
    RAPID = "G00"
    LINEAR = "G01"
    ARC_CW = "G02"
    ARC_CCW = "G03"


class CoordinateMode(Enum):
    ABSOLUTE = "G90"
    INCREMENTAL = "G91"


class Plane(Enum):
    XY = "G17"
    XZ = "G18"
    YZ = "G19"


class SpindleState(Enum):
    OFF = "M05"
    CW = "M03"
    CCW = "M04"


class CoolantState(Enum):
    OFF = "M09"
    FLOOD = "M08"
    MIST = "M07"


class ToolPathType(Enum):
    CONTOUR = "contour"
    POCKET = "pocket"
    DRILL = "drill"
    FACE = "face"
    PROFILE = "profile"
    SLOT = "slot"


class PassType(Enum):
    ROUGHING = "roughing"
    FINISHING = "finishing"
    SEMI_FINISHING = "semi_finishing"


class GCodeValidatorError:
    def __init__(self, line_num: int, message: str, severity: str = "error") -> None:
        self.line_num = line_num
        self.message = message
        self.severity = severity

    def __repr__(self) -> str:
        return f"[{self.severity.upper()}] Line {self.line_num}: {self.message}"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """2D point in mm."""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: Point2D) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Point2D:
        return Point2D(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f})"


@dataclass
class Point3D:
    """3D point in mm."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: Point3D) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def __repr__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class VectorPath:
    """A vector path consisting of points."""
    points: List[Point2D] = field(default_factory=list)
    closed: bool = False
    name: str = ""

    @property
    def length(self) -> float:
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i - 1].distance_to(self.points[i])
        if self.closed and len(self.points) > 1:
            total += self.points[-1].distance_to(self.points[0])
        return total

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Point2D:
        min_x, min_y, max_x, max_y = self.bounds
        return Point2D((min_x + max_x) / 2, (min_y + max_y) / 2)

    def reverse(self) -> VectorPath:
        return VectorPath(
            points=list(reversed(self.points)),
            closed=self.closed,
            name=self.name,
        )

    def offset(self, distance: float) -> VectorPath:
        """Create an offset path (simplified)."""
        if len(self.points) < 2:
            return VectorPath(points=list(self.points), closed=self.closed)
        offset_points: List[Point2D] = []
        for i in range(len(self.points)):
            prev_idx = (i - 1) % len(self.points) if self.closed else max(0, i - 1)
            next_idx = (i + 1) % len(self.points) if self.closed else min(len(self.points) - 1, i + 1)
            prev_pt = self.points[prev_idx]
            curr_pt = self.points[i]
            next_pt = self.points[next_idx]

            dx1 = curr_pt.x - prev_pt.x
            dy1 = curr_pt.y - prev_pt.y
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1) or 1e-10
            nx1 = -dy1 / len1
            ny1 = dx1 / len1

            dx2 = next_pt.x - curr_pt.x
            dy2 = next_pt.y - curr_pt.y
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2) or 1e-10
            nx2 = -dy2 / len2
            ny2 = dx2 / len2

            avg_nx = (nx1 + nx2) / 2
            avg_ny = (ny1 + ny2) / 2
            avg_len = math.sqrt(avg_nx * avg_nx + avg_ny * avg_ny) or 1e-10
            offset_points.append(Point2D(
                curr_pt.x + distance * avg_nx / avg_len,
                curr_pt.y + distance * avg_ny / avg_len,
            ))
        return VectorPath(points=offset_points, closed=self.closed)


@dataclass
class ToolInfo:
    """Information about a cutting tool."""
    tool_number: int = 1
    diameter: float = 6.0  # mm
    length_offset: float = 0.0
    radius_offset: float = 0.0
    max_rpm: int = 24000
    max_feed: float = 3000.0  # mm/min
    max_depth_of_cut: float = 2.0  # mm
    flutes: int = 2
    material: str = "carbide"
    description: str = ""


@dataclass
class GCodeCommand:
    """A single G-code command."""
    code: str = ""
    parameters: Dict[str, float] = field(default_factory=dict)
    comment: str = ""
    line_number: int = 0

    def __str__(self) -> str:
        parts = [self.code]
        for key in sorted(self.parameters.keys()):
            val = self.parameters[key]
            if val == int(val):
                parts.append(f"{key}{int(val)}")
            else:
                parts.append(f"{key}{val:.4f}")
        if self.comment:
            parts.append(f"; {self.comment}")
        return " ".join(parts)


@dataclass
class CutParameters:
    """Parameters for a cutting operation."""
    feed_rate: float = 500.0  # mm/min
    spindle_speed: int = 10000  # RPM
    depth_of_cut: float = 1.0  # mm
    step_over: float = 0.5  # fraction of tool diameter
    plunge_rate: float = 200.0  # mm/min
    retract_height: float = 5.0  # mm
    clearance_height: float = 10.0  # mm
    safe_z: float = DEFAULT_SAFE_Z


@dataclass
class ArcDefinition:
    """Defines an arc for G02/G03 commands."""
    start: Point2D
    end: Point2D
    center: Point2D
    clockwise: bool = True

    @property
    def radius(self) -> float:
        return self.start.distance_to(self.center)

    @property
    def i(self) -> float:
        return self.center.x - self.start.x

    @property
    def j(self) -> float:
        return self.center.y - self.start.y


# ---------------------------------------------------------------------------
# G-Code Validator
# ---------------------------------------------------------------------------

class GCodeValidator:
    """Validates G-code programs for common errors and safety issues."""

    VALID_G_CODES = {
        "G00", "G01", "G02", "G03", "G04", "G17", "G18", "G19",
        "G20", "G21", "G28", "G30", "G40", "G41", "G42", "G43",
        "G49", "G54", "G55", "G56", "G57", "G58", "G59",
        "G80", "G81", "G82", "G83", "G84", "G85", "G90", "G91",
        "G92", "G94", "G95",
    }
    VALID_M_CODES = {
        "M00", "M01", "M02", "M03", "M04", "M05", "M06", "M07",
        "M08", "M09", "M10", "M11", "M30", "M48", "M49",
    }
    REQUIRED_WORDS = {"X", "Y", "Z", "I", "J", "K", "F", "R", "S", "T", "H", "P", "Q"}

    def __init__(
        self,
        max_x: float = 500.0,
        max_y: float = 500.0,
        max_z: float = 200.0,
        max_feed: float = 10000.0,
        max_spindle_rpm: int = 30000,
        min_z: float = -10.0,
    ) -> None:
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.max_feed = max_feed
        self.max_spindle_rpm = max_spindle_rpm
        self.min_z = min_z

    def validate(self, gcode: str) -> List[GCodeValidatorError]:
        errors: List[GCodeValidatorError] = []
        lines = gcode.strip().split("\n")
        spindle_on = False
        current_feed: Optional[float] = None
        current_tool: Optional[int] = None

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("(") or line.startswith("%"):
                continue

            comment_idx = line.find(";")
            if comment_idx >= 0:
                line = line[:comment_idx].strip()
            if not line:
                continue

            words = self._parse_words(line)

            for word, value in words:
                if word == "G":
                    code = f"G{int(value):02d}"
                    if code not in self.VALID_G_CODES:
                        errors.append(GCodeValidatorError(
                            line_num, f"Unknown G-code: {code}", "warning"
                        ))
                elif word == "M":
                    code = f"M{int(value):02d}"
                    if code not in self.VALID_M_CODES:
                        errors.append(GCodeValidatorError(
                            line_num, f"Unknown M-code: {code}", "warning"
                        ))
                    if code == "M03" or code == "M04":
                        spindle_on = True
                    elif code == "M05":
                        spindle_on = False
                elif word == "F":
                    current_feed = value
                    if value > self.max_feed:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"Feed rate {value} exceeds maximum {self.max_feed}",
                            "error",
                        ))
                elif word == "S":
                    if value > self.max_spindle_rpm:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"Spindle speed {value} exceeds maximum {self.max_spindle_rpm}",
                            "error",
                        ))
                elif word == "T":
                    current_tool = int(value)
                elif word == "X":
                    if abs(value) > self.max_x:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"X position {value} exceeds limit {self.max_x}",
                            "error",
                        ))
                elif word == "Y":
                    if abs(value) > self.max_y:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"Y position {value} exceeds limit {self.max_y}",
                            "error",
                        ))
                elif word == "Z":
                    if value > self.max_z:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"Z position {value} exceeds maximum {self.max_z}",
                            "error",
                        ))
                    if value < self.min_z:
                        errors.append(GCodeValidatorError(
                            line_num,
                            f"Z position {value} below minimum {self.min_z}",
                            "error",
                        ))

        return errors

    def _parse_words(self, line: str) -> List[Tuple[str, float]]:
        words: List[Tuple[str, float]] = []
        pattern = re.compile(r'([A-Za-z])([+-]?\d*\.?\d+)')
        for match in pattern.finditer(line):
            letter = match.group(1).upper()
            try:
                value = float(match.group(2))
                words.append((letter, value))
            except ValueError:
                continue
        return words


# ---------------------------------------------------------------------------
# Feed Rate Calculator
# ---------------------------------------------------------------------------

class FeedRateCalculator:
    """Calculates optimal feed rates based on material, tool, and operation."""

    MATERIAL_DATA: Dict[str, Dict[str, float]] = {
        "aluminum": {
            "cutting_speed": 300.0,    # m/min
            "feed_per_tooth": 0.1,     # mm/tooth
            "depth_factor": 1.0,
        },
        "steel_mild": {
            "cutting_speed": 80.0,
            "feed_per_tooth": 0.08,
            "depth_factor": 0.5,
        },
        "steel_hard": {
            "cutting_speed": 40.0,
            "feed_per_tooth": 0.05,
            "depth_factor": 0.3,
        },
        "stainless": {
            "cutting_speed": 50.0,
            "feed_per_tooth": 0.06,
            "depth_factor": 0.4,
        },
        "brass": {
            "cutting_speed": 250.0,
            "feed_per_tooth": 0.12,
            "depth_factor": 0.8,
        },
        "wood": {
            "cutting_speed": 500.0,
            "feed_per_tooth": 0.2,
            "depth_factor": 1.5,
        },
        "plastic": {
            "cutting_speed": 400.0,
            "feed_per_tooth": 0.15,
            "depth_factor": 1.2,
        },
    }

    def __init__(self) -> None:
        self._overrides: Dict[str, float] = {}

    def calculate_feed_rate(
        self,
        material: str,
        tool: ToolInfo,
        pass_type: PassType = PassType.ROUGHING,
    ) -> float:
        mat_data = self.MATERIAL_DATA.get(material.lower())
        if mat_data is None:
            mat_data = self.MATERIAL_DATA["aluminum"]

        cutting_speed = mat_data["cutting_speed"]
        feed_per_tooth = mat_data["feed_per_tooth"]

        if pass_type == PassType.FINISHING:
            feed_per_tooth *= 0.5
        elif pass_type == PassType.SEMI_FINISHING:
            feed_per_tooth *= 0.75

        if "feed_override" in self._overrides:
            feed_per_tooth *= self._overrides["feed_override"]

        # Calculate spindle speed: N = (Vc * 1000) / (pi * D)
        spindle_speed = (cutting_speed * 1000.0) / (math.pi * tool.diameter)
        spindle_speed = min(spindle_speed, tool.max_rpm)

        # Calculate feed rate: F = N * f_z * z
        feed_rate = spindle_speed * feed_per_tooth * tool.flutes

        return min(feed_rate, tool.max_feed)

    def calculate_spindle_speed(
        self,
        material: str,
        tool: ToolInfo,
    ) -> int:
        mat_data = self.MATERIAL_DATA.get(material.lower())
        if mat_data is None:
            mat_data = self.MATERIAL_DATA["aluminum"]

        cutting_speed = mat_data["cutting_speed"]
        spindle_speed = (cutting_speed * 1000.0) / (math.pi * tool.diameter)
        return int(min(spindle_speed, tool.max_rpm))

    def calculate_depth_of_cut(
        self,
        material: str,
        tool: ToolInfo,
        pass_type: PassType = PassType.ROUGHING,
    ) -> float:
        mat_data = self.MATERIAL_DATA.get(material.lower())
        if mat_data is None:
            mat_data = self.MATERIAL_DATA["aluminum"]

        base_depth = tool.max_depth_of_cut * mat_data["depth_factor"]
        if pass_type == PassType.FINISHING:
            base_depth *= 0.3
        elif pass_type == PassType.SEMI_FINISHING:
            base_depth *= 0.6
        return base_depth

    def calculate_plunge_rate(self, feed_rate: float) -> float:
        return feed_rate * 0.3

    def set_override(self, name: str, factor: float) -> None:
        self._overrides[name] = factor


# ---------------------------------------------------------------------------
# Spindle Controller
# ---------------------------------------------------------------------------

class SpindleController:
    """Controls spindle operations in G-code generation."""

    def __init__(self) -> None:
        self._current_speed: int = 0
        self._current_state: SpindleState = SpindleState.OFF
        self._max_speed: int = 24000
        self._commands: List[GCodeCommand] = []

    def start_cw(self, rpm: int = 10000) -> List[GCodeCommand]:
        rpm = min(rpm, self._max_speed)
        self._current_speed = rpm
        self._current_state = SpindleState.CW
        self._commands = [
            GCodeCommand(code="M03", parameters={"S": rpm}, comment="Spindle ON CW"),
        ]
        return list(self._commands)

    def start_ccw(self, rpm: int = 10000) -> List[GCodeCommand]:
        rpm = min(rpm, self._max_speed)
        self._current_speed = rpm
        self._current_state = SpindleState.CCW
        self._commands = [
            GCodeCommand(code="M04", parameters={"S": rpm}, comment="Spindle ON CCW"),
        ]
        return list(self._commands)

    def stop(self) -> List[GCodeCommand]:
        self._current_speed = 0
        self._current_state = SpindleState.OFF
        self._commands = [
            GCodeCommand(code="M05", comment="Spindle OFF"),
        ]
        return list(self._commands)

    def set_speed(self, rpm: int) -> List[GCodeCommand]:
        rpm = min(rpm, self._max_speed)
        self._current_speed = rpm
        self._commands = [
            GCodeCommand(code="S", parameters={"S": rpm}, comment=f"Spindle speed {rpm} RPM"),
        ]
        return list(self._commands)

    def dwell(self, seconds: float) -> List[GCodeCommand]:
        self._commands.append(
            GCodeCommand(code="G04", parameters={"P": seconds * 1000}, comment=f"Dwell {seconds}s")
        )
        return list(self._commands)

    @property
    def current_speed(self) -> int:
        return self._current_speed

    @property
    def current_state(self) -> SpindleState:
        return self._current_state


# ---------------------------------------------------------------------------
# Tool Changer
# ---------------------------------------------------------------------------

class ToolChanger:
    """Manages tool change sequences."""

    def __init__(self) -> None:
        self._tools: Dict[int, ToolInfo] = {}
        self._current_tool: Optional[int] = None
        self._tool_table: List[GCodeCommand] = []

    def register_tool(self, tool: ToolInfo) -> None:
        self._tools[tool.tool_number] = tool

    def change_tool(self, tool_number: int) -> List[GCodeCommand]:
        if tool_number not in self._tools:
            raise ValueError(f"Tool {tool_number} not registered")

        commands: List[GCodeCommand] = []
        commands.append(GCodeCommand(code="M05", comment="Stop spindle before tool change"))
        commands.append(GCodeCommand(
            code="G00", parameters={"Z": DEFAULT_SAFE_Z},
            comment="Retract to safe Z",
        ))
        commands.append(GCodeCommand(
            code="M06", parameters={"T": tool_number},
            comment=f"Change to tool T{tool_number}",
        ))
        commands.append(GCodeCommand(
            code="G43", parameters={"H": tool_number},
            comment=f"Apply tool length offset H{tool_number}",
        ))
        self._current_tool = tool_number
        return commands

    def get_tool(self, tool_number: int) -> Optional[ToolInfo]:
        return self._tools.get(tool_number)

    def get_current_tool(self) -> Optional[ToolInfo]:
        if self._current_tool is not None:
            return self._tools.get(self._current_tool)
        return None

    def list_tools(self) -> List[ToolInfo]:
        return sorted(self._tools.values(), key=lambda t: t.tool_number)


# ---------------------------------------------------------------------------
# Pass Strategy
# ---------------------------------------------------------------------------

class PassStrategy:
    """Generates multiple roughing and finishing passes."""

    def __init__(
        self,
        tool: ToolInfo,
        feed_calculator: FeedRateCalculator,
        material: str = "aluminum",
    ) -> None:
        self.tool = tool
        self.feed_calculator = feed_calculator
        self.material = material

    def generate_passes(
        self,
        total_depth: float,
        finish_allowance: float = 0.2,
    ) -> List[Tuple[float, PassType]]:
        """Generate a list of (depth, pass_type) tuples."""
        roughing_depth = self.feed_calculator.calculate_depth_of_cut(
            self.material, self.tool, PassType.ROUGHING
        )
        finishing_depth = self.feed_calculator.calculate_depth_of_cut(
            self.material, self.tool, PassType.FINISHING
        )

        passes: List[Tuple[float, PassType]] = []
        remaining = total_depth - finish_allowance

        if remaining > 0:
            num_rough = max(1, int(math.ceil(remaining / roughing_depth)))
            actual_depth = remaining / num_rough
            for _ in range(num_rough):
                passes.append((actual_depth, PassType.ROUGHING))

        if finish_allowance > 0:
            passes.append((finish_allowance, PassType.FINISHING))

        return passes

    def calculate_step_over(self, pass_type: PassType) -> float:
        base_step = self.tool.diameter * 0.4
        if pass_type == PassType.ROUGHING:
            return base_step
        elif pass_type == PassType.SEMI_FINISHING:
            return base_step * 0.6
        else:
            return base_step * 0.3


# ---------------------------------------------------------------------------
# Toolpath Generator
# ---------------------------------------------------------------------------

class ToolpathGenerator:
    """Generates toolpaths from vector paths."""

    def __init__(
        self,
        tool: ToolInfo,
        cut_params: Optional[CutParameters] = None,
    ) -> None:
        self.tool = tool
        self.cut_params = cut_params or CutParameters()
        self._commands: List[GCodeCommand] = []

    def generate_contour_path(
        self,
        path: VectorPath,
        z_start: float = 0.0,
        z_end: float = -1.0,
        lead_in: float = 2.0,
        lead_out: float = 2.0,
    ) -> List[GCodeCommand]:
        """Generate a contour toolpath from a vector path."""
        commands: List[GCodeCommand] = []
        if len(path.points) < 2:
            return commands

        radius = self.tool.diameter / 2
        tool_path = path.offset(-radius)

        if not tool_path.points:
            return commands

        start = tool_path.points[0]
        direction = Point2D(
            tool_path.points[1].x - start.x,
            tool_path.points[1].y - start.y,
        )
        dir_len = math.sqrt(direction.x ** 2 + direction.y ** 2) or 1e-10
        lead_start = Point2D(
            start.x - lead_in * direction.x / dir_len,
            start.y - lead_in * direction.y / dir_len,
        )

        # Lead in
        commands.append(GCodeCommand(
            code="G00",
            parameters={"X": lead_start.x, "Y": lead_start.y, "Z": self.cut_params.safe_z},
            comment="Rapid to lead-in start",
        ))
        commands.append(GCodeCommand(
            code="G01",
            parameters={"Z": z_start, "F": self.cut_params.plunge_rate},
            comment="Plunge to start depth",
        ))

        # Cut path
        for i in range(len(tool_path.points)):
            pt = tool_path.points[i]
            if i == 0:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": pt.x, "Y": pt.y, "F": self.cut_params.feed_rate},
                    comment="Lead-in to contour",
                ))
            else:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": pt.x, "Y": pt.y, "F": self.cut_params.feed_rate},
                ))

        if path.closed and len(tool_path.points) > 1:
            commands.append(GCodeCommand(
                code="G01",
                parameters={
                    "X": tool_path.points[0].x,
                    "Y": tool_path.points[0].y,
                    "F": self.cut_params.feed_rate,
                },
                comment="Close contour",
            ))

        # Lead out
        end = tool_path.points[-1]
        commands.append(GCodeCommand(
            code="G01",
            parameters={"Z": self.cut_params.retract_height, "F": self.cut_params.plunge_rate},
            comment="Retract",
        ))

        return commands

    def generate_pocket_path(
        self,
        path: VectorPath,
        z_start: float = 0.0,
        z_end: float = -1.0,
        step_over: float = 0.5,
    ) -> List[GCodeCommand]:
        """Generate a pocket toolpath with zigzag pattern."""
        commands: List[GCodeCommand] = []
        if len(path.points) < 3:
            return commands

        min_x, min_y, max_x, max_y = path.bounds
        step = self.tool.diameter * step_over
        radius = self.tool.diameter / 2

        y = min_y + radius
        direction = 1  # 1 = left-to-right, -1 = right-to-left

        commands.append(GCodeCommand(
            code="G00",
            parameters={"X": min_x + radius, "Y": y, "Z": self.cut_params.safe_z},
            comment="Rapid to pocket start",
        ))
        commands.append(GCodeCommand(
            code="G01",
            parameters={"Z": z_start, "F": self.cut_params.plunge_rate},
            comment="Plunge to pocket depth",
        ))

        while y <= max_y - radius:
            if direction == 1:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": max_x - radius, "F": self.cut_params.feed_rate},
                ))
            else:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": min_x + radius, "F": self.cut_params.feed_rate},
                ))

            y += step
            if y <= max_y - radius:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"Y": y, "F": self.cut_params.feed_rate},
                ))
            direction *= -1

        commands.append(GCodeCommand(
            code="G00",
            parameters={"Z": self.cut_params.retract_height},
            comment="Retract from pocket",
        ))

        return commands

    def generate_drill_path(
        self,
        points: List[Point2D],
        z_start: float = 0.0,
        z_end: float = -5.0,
        peck_depth: float = 2.0,
        dwell: float = 0.5,
    ) -> List[GCodeCommand]:
        """Generate a drilling path."""
        commands: List[GCodeCommand] = []
        for i, pt in enumerate(points):
            commands.append(GCodeCommand(
                code="G00",
                parameters={"X": pt.x, "Y": pt.y, "Z": self.cut_params.safe_z},
                comment=f"Rapid to hole {i + 1}",
            ))
            commands.append(GCodeCommand(
                code="G83",
                parameters={
                    "Z": z_end,
                    "R": self.cut_params.retract_height,
                    "Q": peck_depth,
                    "F": self.cut_params.plunge_rate,
                    "P": dwell * 1000,
                },
                comment=f"Peck drill hole {i + 1}",
            ))
        commands.append(GCodeCommand(
            code="G00", parameters={"Z": self.cut_params.safe_z},
            comment="Retract after drilling",
        ))
        return commands

    def generate_face_path(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        z_start: float = 0.0,
        z_end: float = -0.5,
        step_over: float = 0.5,
    ) -> List[GCodeCommand]:
        """Generate a facing path."""
        commands: List[GCodeCommand] = []
        step = self.tool.diameter * step_over
        y = y_min
        direction = 1

        commands.append(GCodeCommand(
            code="G00",
            parameters={"X": x_min, "Y": y, "Z": self.cut_params.safe_z},
            comment="Rapid to face start",
        ))
        commands.append(GCodeCommand(
            code="G01",
            parameters={"Z": z_start, "F": self.cut_params.plunge_rate},
            comment="Plunge to face depth",
        ))

        while y <= y_max:
            if direction == 1:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": x_max, "F": self.cut_params.feed_rate},
                ))
            else:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"X": x_min, "F": self.cut_params.feed_rate},
                ))
            y += step
            if y <= y_max:
                commands.append(GCodeCommand(
                    code="G01",
                    parameters={"Y": y, "F": self.cut_params.feed_rate},
                ))
            direction *= -1

        commands.append(GCodeCommand(
            code="G00", parameters={"Z": self.cut_params.safe_z},
            comment="Retract after facing",
        ))
        return commands


# ---------------------------------------------------------------------------
# CNC Code Generator (Main Facade)
# ---------------------------------------------------------------------------

class CNCCodeGenerator:
    """Main facade for CNC G-code generation."""

    def __init__(
        self,
        tool: Optional[ToolInfo] = None,
        material: str = "aluminum",
        safe_z: float = DEFAULT_SAFE_Z,
    ) -> None:
        self.tool = tool or ToolInfo()
        self.material = material
        self.safe_z = safe_z
        self.feed_calculator = FeedRateCalculator()
        self.spindle = SpindleController()
        self.tool_changer = ToolChanger()
        self.validator = GCodeValidator()
        self.pass_strategy = PassStrategy(self.tool, self.feed_calculator, material)
        self.toolpath_gen = ToolpathGenerator(self.tool)
        self._program: List[GCodeCommand] = []
        self._line_counter = 0

    def start_program(self, program_name: str = "PROGRAM") -> None:
        self._program = []
        self._line_counter = 0
        self._add_command("%", comment=f"Program: {program_name}")
        self._add_command(DEFAULT_UNITS, comment="Metric units (mm)")
        self._add_command(CoordinateMode.ABSOLUTE.value, comment="Absolute positioning")
        self._add_command(Plane.XY.value, comment="XY plane selection")
        self._add_command("G40", comment="Cancel cutter compensation")
        self._add_command("G49", comment="Cancel tool length compensation")
        self._add_command("G80", comment="Cancel canned cycles")

    def end_program(self) -> None:
        self._program.extend(self.spindle.stop())
        self._add_command(
            "G00", parameters={"Z": self.safe_z}, comment="Retract to safe Z"
        )
        self._add_command(
            "G00", parameters={"X": 0, "Y": 0}, comment="Return to home"
        )
        self._add_command("M30", comment="End of program")
        self._add_command("%")

    def add_comment(self, text: str) -> None:
        self._add_command("", comment=text)

    def add_raw(self, gcode: str) -> None:
        cmd = GCodeCommand(code=gcode)
        cmd.line_number = self._line_counter
        self._line_counter += 1
        self._program.append(cmd)

    def tool_change(self, tool_number: int) -> None:
        self.tool_changer.register_tool(self.tool)
        commands = self.tool_changer.change_tool(tool_number)
        self._program.extend(commands)

    def rapid_move(self, x: float = 0.0, y: float = 0.0, z: Optional[float] = None) -> None:
        params: Dict[str, float] = {"X": x, "Y": y}
        if z is not None:
            params["Z"] = z
        self._add_command("G00", parameters=params, comment="Rapid move")

    def linear_move(self, x: float, y: float, z: float, feed: Optional[float] = None) -> None:
        params: Dict[str, float] = {"X": x, "Y": y, "Z": z}
        if feed:
            params["F"] = feed
        self._add_command("G01", parameters=params, comment="Linear move")

    def arc_cw(
        self, x: float, y: float, i: float, j: float,
        z: Optional[float] = None, feed: Optional[float] = None,
    ) -> None:
        params: Dict[str, float] = {"X": x, "Y": y, "I": i, "J": j}
        if z is not None:
            params["Z"] = z
        if feed:
            params["F"] = feed
        self._add_command("G02", parameters=params, comment="Arc CW")

    def arc_ccw(
        self, x: float, y: float, i: float, j: float,
        z: Optional[float] = None, feed: Optional[float] = None,
    ) -> None:
        params: Dict[str, float] = {"X": x, "Y": y, "I": i, "J": j}
        if z is not None:
            params["Z"] = z
        if feed:
            params["F"] = feed
        self._add_command("G03", parameters=params, comment="Arc CCW")

    def spindle_on(self, rpm: Optional[int] = None) -> None:
        speed = rpm or self.feed_calculator.calculate_spindle_speed(self.material, self.tool)
        self._program.extend(self.spindle.start_cw(speed))

    def spindle_off(self) -> None:
        self._program.extend(self.spindle.stop())

    def coolant_on(self, mode: CoolantState = CoolantState.FLOOD) -> None:
        self._add_command(mode.value, comment=f"Coolant {mode.name}")

    def coolant_off(self) -> None:
        self._add_command(CoolantState.OFF.value, comment="Coolant OFF")

    def dwell(self, seconds: float) -> None:
        self._program.extend(self.spindle.dwell(seconds))

    def generate_contour(
        self,
        path: VectorPath,
        total_depth: float = -2.0,
        finish_allowance: float = 0.2,
    ) -> None:
        passes = self.pass_strategy.generate_passes(
            abs(total_depth), finish_allowance
        )
        current_z = 0.0

        for depth, pass_type in passes:
            feed = self.feed_calculator.calculate_feed_rate(self.material, self.tool, pass_type)
            cut_params = CutParameters(
                feed_rate=feed,
                spindle_speed=self.feed_calculator.calculate_spindle_speed(self.material, self.tool),
                depth_of_cut=depth,
                plunge_rate=self.feed_calculator.calculate_plunge_rate(feed),
                retract_height=abs(total_depth) + 2.0,
                clearance_height=abs(total_depth) + 5.0,
                safe_z=self.safe_z,
            )
            gen = ToolpathGenerator(self.tool, cut_params)
            z_end = current_z - depth
            commands = gen.generate_contour_path(path, current_z, z_end)
            self._program.extend(commands)
            current_z = z_end

    def generate_pocket(
        self,
        path: VectorPath,
        total_depth: float = -2.0,
        step_over: float = 0.5,
    ) -> None:
        passes = self.pass_strategy.generate_passes(abs(total_depth), 0.0)
        current_z = 0.0
        for depth, pass_type in passes:
            feed = self.feed_calculator.calculate_feed_rate(self.material, self.tool, pass_type)
            cut_params = CutParameters(
                feed_rate=feed,
                spindle_speed=self.feed_calculator.calculate_spindle_speed(self.material, self.tool),
                plunge_rate=self.feed_calculator.calculate_plunge_rate(feed),
                safe_z=self.safe_z,
            )
            gen = ToolpathGenerator(self.tool, cut_params)
            z_end = current_z - depth
            commands = gen.generate_pocket_path(path, current_z, z_end, step_over)
            self._program.extend(commands)
            current_z = z_end

    def generate_drilling(
        self,
        points: List[Point2D],
        depth: float = -5.0,
        peck_depth: float = 2.0,
    ) -> None:
        feed = self.feed_calculator.calculate_feed_rate(self.material, self.tool, PassType.ROUGHING)
        cut_params = CutParameters(
            feed_rate=feed,
            plunge_rate=self.feed_calculator.calculate_plunge_rate(feed),
            safe_z=self.safe_z,
        )
        gen = ToolpathGenerator(self.tool, cut_params)
        commands = gen.generate_drill_path(points, 0.0, depth, peck_depth)
        self._program.extend(commands)

    def generate_facing(
        self,
        x_min: float, y_min: float, x_max: float, y_max: float,
        depth: float = -0.5,
    ) -> None:
        feed = self.feed_calculator.calculate_feed_rate(self.material, self.tool, PassType.ROUGHING)
        cut_params = CutParameters(
            feed_rate=feed,
            plunge_rate=self.feed_calculator.calculate_plunge_rate(feed),
            safe_z=self.safe_z,
        )
        gen = ToolpathGenerator(self.tool, cut_params)
        commands = gen.generate_face_path(x_min, y_min, x_max, y_max, 0.0, depth)
        self._program.extend(commands)

    def get_gcode(self) -> str:
        lines: List[str] = []
        for cmd in self._program:
            lines.append(str(cmd))
        return "\n".join(lines)

    def validate_program(self) -> List[GCodeValidatorError]:
        return self.validator.validate(self.get_gcode())

    def _add_command(
        self,
        code: str,
        parameters: Optional[Dict[str, float]] = None,
        comment: str = "",
    ) -> None:
        cmd = GCodeCommand(
            code=code,
            parameters=parameters or {},
            comment=comment,
            line_number=self._line_counter,
        )
        self._line_counter += 1
        self._program.append(cmd)

    @property
    def command_count(self) -> int:
        return len(self._program)

    def get_statistics(self) -> Dict[str, Any]:
        total_moves = 0
        rapid_moves = 0
        feed_moves = 0
        arc_moves = 0
        for cmd in self._program:
            if cmd.code == "G00":
                rapid_moves += 1
                total_moves += 1
            elif cmd.code == "G01":
                feed_moves += 1
                total_moves += 1
            elif cmd.code in ("G02", "G03"):
                arc_moves += 1
                total_moves += 1
        return {
            "total_commands": len(self._program),
            "total_moves": total_moves,
            "rapid_moves": rapid_moves,
            "feed_moves": feed_moves,
            "arc_moves": arc_moves,
        }
