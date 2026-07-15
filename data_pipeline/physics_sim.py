"""
AGI Unified Framework - 物理仿真数据生成器 (Physics Simulation Data Generator)

本模块提供纯Python实现的物理仿真数据生成功能，支持刚体动力学模拟、
碰撞检测、点云数据生成等核心能力。基于标准库实现，无需第三方依赖。

核心组件:
    - RigidBody: 刚体对象，封装位置、速度、质量、形状等物理属性
    - PhysicsScene: 物理场景，管理多个刚体、重力场和时间步进
    - PointCloudData: 点云数据结构，支持坐标、颜色、法线等信息
    - PhysicsDataGenerator: 物理仿真数据生成器，提供高层API

使用示例:
    >>> from agi_unified_framework.data_pipeline.physics_sim import PhysicsDataGenerator
    >>> generator = PhysicsDataGenerator(gravity=-9.81)
    >>> scene = generator.create_scene(num_bodies=10)
    >>> frames = generator.simulate(scene, num_steps=500)
    >>> point_cloud = generator.generate_point_cloud(scene)
"""

from __future__ import annotations

import math
import random
import json
import logging
import copy
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

__all__ = ["PhysicsDataGenerator", "RigidBody", "PhysicsScene", "PointCloudData"]
__version__ = "1.0.0"

logger = logging.getLogger(__name__)


# ==================== 形状类型枚举 ====================

class ShapeType(Enum):
    """刚体形状类型"""
    SPHERE = "sphere"       # 球体
    BOX = "box"             # 立方体
    CYLINDER = "cylinder"   # 圆柱体
    PLANE = "plane"         # 平面（无限大，用于地面）


# ==================== 刚体 ====================

@dataclass
class RigidBody:
    """
    刚体对象

    封装三维空间中的刚体物理属性，包括运动学状态（位置、速度、加速度）
    和物理属性（质量、形状、摩擦系数等）。支持欧拉积分法更新运动状态。

    Attributes:
        id: 刚体唯一标识符
        name: 刚体名称（用于调试和可视化）
        position: 三维位置坐标 [x, y, z]
        velocity: 三维速度向量 [vx, vy, vz]
        acceleration: 三维加速度向量 [ax, ay, az]
        mass: 质量（千克），0表示静态物体
        shape_type: 形状类型
        shape_params: 形状参数字典（半径、半尺寸等）
        color: RGBA颜色值 [r, g, b, a]，范围0~1
        friction: 摩擦系数
        restitution: 弹性恢复系数（0=完全非弹性，1=完全弹性）
        is_static: 是否为静态物体（不受力影响）
        angular_velocity: 角速度向量 [wx, wy, wz]
        orientation: 四元数表示的姿态 [w, x, y, z]
        force_accumulator: 累积力向量
    """
    id: int
    name: str = ""
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    mass: float = 1.0
    shape_type: ShapeType = ShapeType.SPHERE
    shape_params: Dict[str, float] = field(default_factory=lambda: {"radius": 0.5})
    color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0])
    friction: float = 0.5
    restitution: float = 0.3
    is_static: bool = False
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    force_accumulator: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def kinetic_energy(self) -> float:
        """计算刚体的平动动能: E_k = 0.5 * m * |v|^2"""
        if self.is_static or self.mass <= 0:
            return 0.0
        v_sq = sum(v * v for v in self.velocity)
        return 0.5 * self.mass * v_sq

    @property
    def potential_energy(self, gravity: float = -9.81) -> float:
        """计算刚体的重力势能: E_p = m * |g| * h"""
        if self.is_static or self.mass <= 0:
            return 0.0
        return self.mass * abs(gravity) * self.position[2]

    @property
    def momentum(self) -> List[float]:
        """计算动量向量: p = m * v"""
        if self.is_static or self.mass <= 0:
            return [0.0, 0.0, 0.0]
        return [self.mass * v for v in self.velocity]

    @property
    def speed(self) -> float:
        """计算速率: |v|"""
        return math.sqrt(sum(v * v for v in self.velocity))

    def apply_force(self, force: List[float]) -> None:
        """
        施加外力到刚体

        Args:
            force: 力向量 [fx, fy, fz]
        """
        if self.is_static:
            return
        for i in range(3):
            self.force_accumulator[i] += force[i]

    def apply_impulse(self, impulse: List[float]) -> None:
        """
        施加冲量到刚体（直接改变速度）

        Args:
            impulse: 冲量向量 [jx, jy, jz]
        """
        if self.is_static or self.mass <= 0:
            return
        for i in range(3):
            self.velocity[i] += impulse[i] / self.mass

    def clear_forces(self) -> None:
        """清除累积力"""
        self.force_accumulator = [0.0, 0.0, 0.0]

    def integrate(self, dt: float) -> None:
        """
        使用半隐式欧拉法积分更新运动状态

        更新步骤:
        1. 根据累积力计算加速度: a = F / m
        2. 更新速度: v = v + a * dt
        3. 更新位置: x = x + v * dt
        4. 清除累积力

        Args:
            dt: 时间步长（秒）
        """
        if self.is_static or self.mass <= 0:
            return

        # 计算加速度 a = F / m
        for i in range(3):
            self.acceleration[i] = self.force_accumulator[i] / self.mass

        # 更新速度（半隐式欧拉：先更新速度，再更新位置）
        for i in range(3):
            self.velocity[i] += self.acceleration[i] * dt

        # 更新位置
        for i in range(3):
            self.position[i] += self.velocity[i] * dt

        # 清除累积力
        self.clear_forces()

    def get_bounding_radius(self) -> float:
        """
        获取包围球半径

        Returns:
            包围球半径
        """
        if self.shape_type == ShapeType.SPHERE:
            return self.shape_params.get("radius", 0.5)
        elif self.shape_type == ShapeType.BOX:
            hx = self.shape_params.get("half_x", 0.5)
            hy = self.shape_params.get("half_y", 0.5)
            hz = self.shape_params.get("half_z", 0.5)
            return math.sqrt(hx * hx + hy * hy + hz * hz)
        elif self.shape_type == ShapeType.CYLINDER:
            r = self.shape_params.get("radius", 0.5)
            h = self.shape_params.get("half_height", 0.5)
            return math.sqrt(r * r + h * h)
        return 0.5

    def distance_to(self, other: RigidBody) -> float:
        """
        计算到另一个刚体的距离

        Args:
            other: 另一个刚体

        Returns:
            两刚体中心之间的欧氏距离
        """
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        dz = self.position[2] - other.position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "mass": self.mass,
            "shape_type": self.shape_type.value,
            "shape_params": self.shape_params,
            "color": self.color,
            "friction": self.friction,
            "restitution": self.restitution,
            "is_static": self.is_static,
        }

    def clone(self) -> RigidBody:
        """创建刚体的深拷贝"""
        return copy.deepcopy(self)


# ==================== 物理场景 ====================

class PhysicsScene:
    """
    物理场景

    管理物理仿真中的所有刚体对象、重力场和时间步进。
    提供碰撞检测、场景状态快照、能量计算等功能。

    Attributes:
        gravity: 重力加速度向量 [gx, gy, gz]
        time_step: 仿真时间步长（秒）
        time: 当前仿真时间
        step_count: 已执行的仿真步数
        bodies: 场景中的刚体列表
        ground_y: 地面高度（y轴），低于此高度的物体会被弹回

    使用示例:
        >>> scene = PhysicsScene(gravity=[0, -9.81, 0])
        >>> ball = RigidBody(id=0, position=[0, 10, 0], mass=1.0)
        >>> scene.add_body(ball)
        >>> scene.step()  # 执行一步仿真
    """

    def __init__(
        self,
        gravity: List[float] = None,
        time_step: float = 1.0 / 60.0,
        ground_y: float = 0.0,
        damping: float = 0.999,
    ):
        """
        初始化物理场景

        Args:
            gravity: 重力加速度向量，默认 [0, -9.81, 0]
            time_step: 仿真时间步长
            ground_y: 地面高度
            damping: 速度阻尼系数（模拟空气阻力），1.0表示无阻尼
        """
        self.gravity = gravity if gravity is not None else [0.0, -9.81, 0.0]
        self.time_step = time_step
        self.time = 0.0
        self.step_count = 0
        self.ground_y = ground_y
        self.damping = damping

        # 刚体管理
        self.bodies: List[RigidBody] = []
        self._next_id = 0

        # 碰撞记录
        self.collision_records: List[Dict[str, Any]] = []

        # 能量历史
        self.energy_history: List[Dict[str, float]] = []

        logger.debug(
            f"PhysicsScene 初始化: gravity={self.gravity}, "
            f"time_step={self.time_step}"
        )

    def add_body(self, body: RigidBody) -> None:
        """
        添加刚体到场景

        Args:
            body: 要添加的刚体对象
        """
        body.id = self._next_id
        self._next_id += 1
        self.bodies.append(body)
        logger.debug(f"添加刚体: id={body.id}, name={body.name}")

    def remove_body(self, body_id: int) -> bool:
        """
        从场景中移除刚体

        Args:
            body_id: 要移除的刚体ID

        Returns:
            是否成功移除
        """
        for i, body in enumerate(self.bodies):
            if body.id == body_id:
                self.bodies.pop(i)
                logger.debug(f"移除刚体: id={body_id}")
                return True
        return False

    def get_body(self, body_id: int) -> Optional[RigidBody]:
        """
        根据ID获取刚体

        Args:
            body_id: 刚体ID

        Returns:
            刚体对象，不存在则返回None
        """
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None

    def step(self, num_steps: int = 1) -> None:
        """
        执行仿真步进

        每步执行:
        1. 对所有非静态刚体施加重力
        2. 检测并处理碰撞
        3. 积分更新运动状态
        4. 应用速度阻尼
        5. 记录能量

        Args:
            num_steps: 要执行的步数
        """
        for _ in range(num_steps):
            # 施加重力
            self._apply_gravity()

            # 碰撞检测与响应
            self._detect_and_resolve_collisions()

            # 积分更新
            for body in self.bodies:
                body.integrate(self.time_step)

            # 应用阻尼
            for body in self.bodies:
                if not body.is_static:
                    for i in range(3):
                        body.velocity[i] *= self.damping

            # 地面碰撞
            self._enforce_ground()

            # 更新时间和步数
            self.time += self.time_step
            self.step_count += 1

            # 记录能量
            if self.step_count % 10 == 0:
                self.energy_history.append(self.compute_energy())

    def _apply_gravity(self) -> None:
        """对所有非静态刚体施加重力"""
        for body in self.bodies:
            if not body.is_static and body.mass > 0:
                gravity_force = [
                    body.mass * g for g in self.gravity
                ]
                body.apply_force(gravity_force)

    def _detect_and_resolve_collisions(self) -> None:
        """
        碰撞检测与响应

        使用球体包围盒进行快速碰撞检测，对碰撞的刚体对
        计算碰撞法线和冲量，应用弹性碰撞公式。
        """
        self.collision_records.clear()

        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                body_a = self.bodies[i]
                body_b = self.bodies[j]

                # 跳过两个静态物体
                if body_a.is_static and body_b.is_static:
                    continue

                # 计算距离
                dist = body_a.distance_to(body_b)
                radius_a = body_a.get_bounding_radius()
                radius_b = body_b.get_bounding_radius()
                min_dist = radius_a + radius_b

                if dist < min_dist and dist > 1e-8:
                    # 碰撞发生，计算碰撞法线
                    normal = [
                        (body_b.position[k] - body_a.position[k]) / dist
                        for k in range(3)
                    ]

                    # 计算相对速度在法线方向的分量
                    rel_vel = [
                        body_a.velocity[k] - body_b.velocity[k]
                        for k in range(3)
                    ]
                    vel_along_normal = sum(
                        rel_vel[k] * normal[k] for k in range(3)
                    )

                    # 如果物体正在分离，不处理
                    if vel_along_normal > 0:
                        continue

                    # 计算弹性系数（取两者中的较小值）
                    restitution = min(
                        body_a.restitution, body_b.restitution
                    )

                    # 计算冲量标量
                    if body_a.is_static:
                        inv_mass_a = 0.0
                    else:
                        inv_mass_a = 1.0 / body_a.mass

                    if body_b.is_static:
                        inv_mass_b = 0.0
                    else:
                        inv_mass_b = 1.0 / body_b.mass

                    impulse_scalar = (
                        -(1.0 + restitution) * vel_along_normal
                        / (inv_mass_a + inv_mass_b)
                    )

                    # 应用冲量
                    impulse = [impulse_scalar * n for n in normal]

                    if not body_a.is_static:
                        body_a.apply_impulse(impulse)
                    if not body_b.is_static:
                        body_b.apply_impulse([-imp for imp in impulse])

                    # 位置修正（防止穿透）
                    overlap = min_dist - dist
                    correction = [
                        overlap * n * 0.5 for n in normal
                    ]
                    if not body_a.is_static:
                        for k in range(3):
                            body_a.position[k] -= correction[k]
                    if not body_b.is_static:
                        for k in range(3):
                            body_b.position[k] += correction[k]

                    # 记录碰撞
                    self.collision_records.append({
                        "body_a": body_a.id,
                        "body_b": body_b.id,
                        "normal": normal,
                        "impulse": impulse_scalar,
                        "overlap": overlap,
                    })

    def _enforce_ground(self) -> None:
        """
        强制执行地面约束

        检测所有刚体是否低于地面，如果是则将其弹回地面
        并反转垂直方向速度（乘以弹性系数）。
        """
        for body in self.bodies:
            if body.is_static:
                continue

            radius = body.get_bounding_radius()
            ground_limit = self.ground_y + radius

            if body.position[1] < ground_limit:
                # 位置修正
                body.position[1] = ground_limit

                # 速度反弹
                if body.velocity[1] < 0:
                    body.velocity[1] = -body.velocity[1] * body.restitution

                    # 摩擦力（减少水平速度）
                    friction_factor = 1.0 - body.friction * self.time_step
                    friction_factor = max(0.0, friction_factor)
                    body.velocity[0] *= friction_factor
                    body.velocity[2] *= friction_factor

    def compute_energy(self) -> Dict[str, float]:
        """
        计算系统总能量

        Returns:
            包含动能、势能和总能量的字典
        """
        kinetic = 0.0
        potential = 0.0

        for body in self.bodies:
            kinetic += body.kinetic_energy
            potential += body.potential_energy(self.gravity[1])

        return {
            "kinetic": kinetic,
            "potential": potential,
            "total": kinetic + potential,
            "time": self.time,
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        获取场景状态快照

        Returns:
            包含所有刚体状态和场景信息的字典
        """
        return {
            "time": self.time,
            "step_count": self.step_count,
            "num_bodies": len(self.bodies),
            "bodies": [body.to_dict() for body in self.bodies],
            "energy": self.compute_energy(),
            "num_collisions": len(self.collision_records),
        }

    def reset(self) -> None:
        """重置场景（清除所有刚体和状态）"""
        self.bodies.clear()
        self.collision_records.clear()
        self.energy_history.clear()
        self.time = 0.0
        self.step_count = 0
        self._next_id = 0
        logger.debug("PhysicsScene 已重置")

    def __len__(self) -> int:
        """返回场景中的刚体数量"""
        return len(self.bodies)

    def __repr__(self) -> str:
        return (
            f"PhysicsScene(bodies={len(self.bodies)}, "
            f"time={self.time:.3f}s, "
            f"steps={self.step_count})"
        )


# ==================== 点云数据 ====================

@dataclass
class PointCloudData:
    """
    点云数据结构

    封装三维点云数据，包含坐标、颜色、法线等信息。
    提供点云变换、过滤、统计等实用方法。

    Attributes:
        points: 点坐标列表，每个点为 [x, y, z]
        colors: 点颜色列表，每个颜色为 [r, g, b, a]，范围0~1
        normals: 点法线列表，每个法线为 [nx, ny, nz]
        labels: 点标签列表（用于语义分割）
        metadata: 附加元数据

    使用示例:
        >>> pc = PointCloudData(
        ...     points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ...     colors=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]],
        ... )
        >>> print(f"点云包含 {len(pc)} 个点")
    """
    points: List[List[float]] = field(default_factory=list)
    colors: List[List[float]] = field(default_factory=list)
    normals: List[List[float]] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        """点云中的点数"""
        return len(self.points)

    @property
    def bounding_box(self) -> Tuple[List[float], List[float]]:
        """
        计算点云的轴对齐包围盒 (AABB)

        Returns:
            (min_coords, max_coords) 最小和最大坐标
        """
        if not self.points:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

        min_coords = [
            min(p[i] for p in self.points) for i in range(3)
        ]
        max_coords = [
            max(p[i] for p in self.points) for i in range(3)
        ]
        return min_coords, max_coords

    @property
    def centroid(self) -> List[float]:
        """
        计算点云质心

        Returns:
            质心坐标 [cx, cy, cz]
        """
        if not self.points:
            return [0.0, 0.0, 0.0]

        n = len(self.points)
        return [
            sum(p[i] for p in self.points) / n for i in range(3)
        ]

    def compute_normals(self) -> None:
        """
        估算点云法线（使用最近邻平面拟合的简化版本）

        对每个点，找到最近的若干邻居，拟合平面并取法线方向。
        简化实现使用相邻点的叉积近似。
        """
        if len(self.points) < 3:
            return

        self.normals = []

        for i in range(len(self.points)):
            # 找到最近的两个邻居
            distances = []
            for j in range(len(self.points)):
                if i == j:
                    continue
                dx = self.points[j][0] - self.points[i][0]
                dy = self.points[j][1] - self.points[i][1]
                dz = self.points[j][2] - self.points[i][2]
                distances.append((math.sqrt(dx*dx + dy*dy + dz*dz), j))

            distances.sort(key=lambda x: x[0])

            if len(distances) >= 2:
                # 取最近两个邻居构建两个向量
                j1 = distances[0][1]
                j2 = distances[1][1]

                v1 = [
                    self.points[j1][k] - self.points[i][k]
                    for k in range(3)
                ]
                v2 = [
                    self.points[j2][k] - self.points[i][k]
                    for k in range(3)
                ]

                # 叉积得到法线
                nx = v1[1] * v2[2] - v1[2] * v2[1]
                ny = v1[2] * v2[0] - v1[0] * v2[2]
                nz = v1[0] * v2[1] - v1[1] * v2[0]

                # 归一化
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-8:
                    nx /= length
                    ny /= length
                    nz /= length
                else:
                    nx, ny, nz = 0.0, 1.0, 0.0

                self.normals.append([nx, ny, nz])
            else:
                self.normals.append([0.0, 1.0, 0.0])

    def translate(self, offset: List[float]) -> PointCloudData:
        """
        平移点云

        Args:
            offset: 平移向量 [dx, dy, dz]

        Returns:
            平移后的新点云对象
        """
        new_points = [
            [p[i] + offset[i] for i in range(3)]
            for p in self.points
        ]
        return PointCloudData(
            points=new_points,
            colors=list(self.colors),
            normals=list(self.normals),
            labels=list(self.labels),
            metadata=dict(self.metadata),
        )

    def scale(self, factor: float) -> PointCloudData:
        """
        缩放点云

        Args:
            factor: 缩放因子

        Returns:
            缩放后的新点云对象
        """
        centroid = self.centroid
        new_points = [
            [
                centroid[i] + (p[i] - centroid[i]) * factor
                for i in range(3)
            ]
            for p in self.points
        ]
        return PointCloudData(
            points=new_points,
            colors=list(self.colors),
            normals=list(self.normals),
            labels=list(self.labels),
            metadata=dict(self.metadata),
        )

    def filter_by_distance(self, max_distance: float) -> PointCloudData:
        """
        过滤距离质心过远的点

        Args:
            max_distance: 最大允许距离

        Returns:
            过滤后的新点云对象
        """
        centroid = self.centroid
        filtered_points = []
        filtered_colors = []
        filtered_normals = []
        filtered_labels = []

        for i, p in enumerate(self.points):
            dx = p[0] - centroid[0]
            dy = p[1] - centroid[1]
            dz = p[2] - centroid[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist <= max_distance:
                filtered_points.append(p)
                if i < len(self.colors):
                    filtered_colors.append(self.colors[i])
                if i < len(self.normals):
                    filtered_normals.append(self.normals[i])
                if i < len(self.labels):
                    filtered_labels.append(self.labels[i])

        return PointCloudData(
            points=filtered_points,
            colors=filtered_colors,
            normals=filtered_normals,
            labels=filtered_labels,
            metadata=dict(self.metadata),
        )

    def random_subsample(self, num_points: int) -> PointCloudData:
        """
        随机下采样

        Args:
            num_points: 目标点数

        Returns:
            下采样后的新点云对象
        """
        if num_points >= len(self.points):
            return self.clone()

        indices = random.sample(range(len(self.points)), num_points)
        indices.sort()

        return PointCloudData(
            points=[self.points[i] for i in indices],
            colors=[self.colors[i] for i in indices if i < len(self.colors)],
            normals=[self.normals[i] for i in indices if i < len(self.normals)],
            labels=[self.labels[i] for i in indices if i < len(self.labels)],
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "num_points": self.num_points,
            "points": self.points,
            "colors": self.colors,
            "normals": self.normals,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    def clone(self) -> PointCloudData:
        """创建点云的深拷贝"""
        return PointCloudData(
            points=[list(p) for p in self.points],
            colors=[list(c) for c in self.colors],
            normals=[list(n) for n in self.normals],
            labels=list(self.labels),
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return self.num_points

    def __repr__(self) -> str:
        return (
            f"PointCloudData(num_points={self.num_points}, "
            f"has_colors={len(self.colors) > 0}, "
            f"has_normals={len(self.normals) > 0})"
        )


# ==================== 物理仿真数据生成器 ====================

class PhysicsDataGenerator:
    """
    物理仿真数据生成器

    提供高层次的物理仿真数据生成API，支持以下场景:
    - 自由落体模拟
    - 多体碰撞模拟
    - 堆叠稳定性模拟
    - 点云数据生成
    - 场景序列数据生成

    所有仿真基于纯Python实现，使用欧拉积分法和球体碰撞检测。

    使用示例:
        >>> generator = PhysicsDataGenerator(gravity=-9.81)
        >>> scene = generator.create_free_fall_scene()
        >>> frames = generator.simulate(scene, num_steps=300)
        >>> point_cloud = generator.generate_point_cloud(scene)
    """

    def __init__(
        self,
        gravity: float = -9.81,
        time_step: float = 1.0 / 60.0,
        default_restitution: float = 0.6,
        default_friction: float = 0.4,
        seed: Optional[int] = None,
    ):
        """
        初始化物理仿真数据生成器

        Args:
            gravity: 重力加速度（m/s^2），负值表示向下
            time_step: 默认仿真时间步长
            default_restitution: 默认弹性恢复系数
            default_friction: 默认摩擦系数
            seed: 随机种子（为None时不设置）
        """
        self.gravity = gravity
        self.time_step = time_step
        self.default_restitution = default_restitution
        self.default_friction = default_friction

        if seed is not None:
            random.seed(seed)

        logger.info(
            f"PhysicsDataGenerator 初始化: gravity={self.gravity}, "
            f"time_step={self.time_step}"
        )

    def create_scene(
        self,
        num_bodies: int = 5,
        spawn_range: float = 5.0,
        height_range: Tuple[float, float] = (2.0, 10.0),
        mass_range: Tuple[float, float] = (0.5, 5.0),
        radius_range: Tuple[float, float] = (0.2, 0.8),
    ) -> PhysicsScene:
        """
        创建随机物理场景

        在指定范围内随机生成多个球体刚体，用于碰撞和自由落体仿真。

        Args:
            num_bodies: 刚体数量
            spawn_range: 水平生成范围（正负对称）
            height_range: 高度生成范围 [min, max]
            mass_range: 质量生成范围 [min, max]
            radius_range: 半径生成范围 [min, max]

        Returns:
            配置好的PhysicsScene对象
        """
        scene = PhysicsScene(
            gravity=[0.0, self.gravity, 0.0],
            time_step=self.time_step,
        )

        for i in range(num_bodies):
            radius = random.uniform(*radius_range)
            mass = random.uniform(*mass_range)
            height = random.uniform(*height_range)

            body = RigidBody(
                id=i,
                name=f"body_{i}",
                position=[
                    random.uniform(-spawn_range, spawn_range),
                    height,
                    random.uniform(-spawn_range, spawn_range),
                ],
                velocity=[
                    random.uniform(-2, 2),
                    random.uniform(-1, 1),
                    random.uniform(-2, 2),
                ],
                mass=mass,
                shape_type=ShapeType.SPHERE,
                shape_params={"radius": radius},
                color=[
                    random.random(),
                    random.random(),
                    random.random(),
                    1.0,
                ],
                restitution=self.default_restitution,
                friction=self.default_friction,
            )

            scene.add_body(body)

        logger.info(f"创建物理场景: {num_bodies} 个刚体")
        return scene

    def create_free_fall_scene(
        self,
        num_bodies: int = 3,
        height: float = 10.0,
    ) -> PhysicsScene:
        """
        创建自由落体场景

        多个球体从同一高度自由下落，观察重力加速度效果。

        Args:
            num_bodies: 球体数量
            height: 初始高度

        Returns:
            配置好的PhysicsScene对象
        """
        scene = PhysicsScene(
            gravity=[0.0, self.gravity, 0.0],
            time_step=self.time_step,
        )

        for i in range(num_bodies):
            radius = random.uniform(0.3, 0.7)
            body = RigidBody(
                id=i,
                name=f"freefall_{i}",
                position=[
                    (i - num_bodies / 2) * 2.0,
                    height + i * 1.0,
                    0.0,
                ],
                velocity=[0.0, 0.0, 0.0],
                mass=random.uniform(0.5, 3.0),
                shape_type=ShapeType.SPHERE,
                shape_params={"radius": radius},
                color=[
                    random.random(),
                    random.random(),
                    random.random(),
                    1.0,
                ],
                restitution=self.default_restitution,
                friction=self.default_friction,
            )
            scene.add_body(body)

        logger.info(f"创建自由落体场景: {num_bodies} 个球体, 高度={height}m")
        return scene

    def create_stacking_scene(
        self,
        num_blocks: int = 5,
        block_size: float = 1.0,
    ) -> PhysicsScene:
        """
        创建堆叠场景

        多个立方体从上方落下堆叠，测试稳定性。

        Args:
            num_blocks: 立方体数量
            block_size: 立方体尺寸

        Returns:
            配置好的PhysicsScene对象
        """
        scene = PhysicsScene(
            gravity=[0.0, self.gravity, 0.0],
            time_step=self.time_step,
        )

        half = block_size / 2.0
        for i in range(num_blocks):
            body = RigidBody(
                id=i,
                name=f"block_{i}",
                position=[
                    random.uniform(-0.1, 0.1),
                    8.0 + i * 1.5,
                    random.uniform(-0.1, 0.1),
                ],
                velocity=[0.0, 0.0, 0.0],
                mass=random.uniform(1.0, 3.0),
                shape_type=ShapeType.BOX,
                shape_params={
                    "half_x": half,
                    "half_y": half,
                    "half_z": half,
                },
                color=[
                    random.random(),
                    random.random(),
                    random.random(),
                    1.0,
                ],
                restitution=0.2,
                friction=0.8,
            )
            scene.add_body(body)

        logger.info(f"创建堆叠场景: {num_blocks} 个立方体")
        return scene

    def simulate(
        self,
        scene: PhysicsScene,
        num_steps: int = 300,
        record_interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        运行物理仿真并记录状态序列

        Args:
            scene: 物理场景
            num_steps: 仿真步数
            record_interval: 记录间隔（每N步记录一次）

        Returns:
            状态快照列表，每个快照包含所有刚体的位置、速度等信息
        """
        frames = []

        for step in range(num_steps):
            scene.step()

            if step % record_interval == 0:
                snapshot = scene.get_state_snapshot()
                snapshot["frame_index"] = len(frames)
                frames.append(snapshot)

        logger.info(
            f"仿真完成: {num_steps} 步, 记录 {len(frames)} 帧"
        )
        return frames

    def generate_point_cloud(
        self,
        scene: PhysicsScene,
        points_per_body: int = 100,
        add_noise: float = 0.01,
    ) -> PointCloudData:
        """
        从物理场景生成点云数据

        对场景中的每个刚体，在其表面均匀采样点，生成模拟的
        三维扫描点云数据。

        Args:
            scene: 物理场景
            points_per_body: 每个刚体表面采样点数
            add_noise: 添加的高斯噪声标准差

        Returns:
            生成的PointCloudData对象
        """
        all_points = []
        all_colors = []
        all_labels = []

        for body_idx, body in enumerate(scene.bodies):
            if body.shape_type == ShapeType.SPHERE:
                points = self._sample_sphere_surface(
                    body.position,
                    body.shape_params.get("radius", 0.5),
                    points_per_body,
                )
            elif body.shape_type == ShapeType.BOX:
                points = self._sample_box_surface(
                    body.position,
                    body.shape_params.get("half_x", 0.5),
                    body.shape_params.get("half_y", 0.5),
                    body.shape_params.get("half_z", 0.5),
                    points_per_body,
                )
            else:
                # 默认使用球体采样
                points = self._sample_sphere_surface(
                    body.position,
                    body.get_bounding_radius(),
                    points_per_body,
                )

            # 添加高斯噪声
            if add_noise > 0:
                for p in points:
                    for k in range(3):
                        p[k] += random.gauss(0, add_noise)

            all_points.extend(points)
            all_colors.extend([body.color] * len(points))
            all_labels.extend([body_idx] * len(points))

        point_cloud = PointCloudData(
            points=all_points,
            colors=all_colors,
            labels=all_labels,
            metadata={
                "num_bodies": len(scene.bodies),
                "points_per_body": points_per_body,
                "scene_time": scene.time,
            },
        )

        logger.info(
            f"生成点云: {point_cloud.num_points} 个点, "
            f"{len(scene.bodies)} 个刚体"
        )
        return point_cloud

    def generate_dataset(
        self,
        num_scenes: int = 100,
        num_bodies_per_scene: int = 5,
        num_steps: int = 300,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成物理仿真数据集

        Args:
            num_scenes: 场景数量
            num_bodies_per_scene: 每个场景的刚体数量
            num_steps: 每个场景的仿真步数
            output_dir: 输出目录（为None时不保存到磁盘）

        Returns:
            数据集列表，每个元素包含场景配置和仿真帧
        """
        dataset = []

        for scene_idx in range(num_scenes):
            # 创建随机场景
            scene = self.create_scene(
                num_bodies=num_bodies_per_scene,
            )

            # 运行仿真
            frames = self.simulate(scene, num_steps=num_steps)

            # 生成点云
            point_cloud = self.generate_point_cloud(scene)

            scene_data = {
                "scene_id": scene_idx,
                "config": {
                    "num_bodies": num_bodies_per_scene,
                    "num_steps": num_steps,
                    "gravity": self.gravity,
                },
                "frames": frames,
                "point_cloud": point_cloud.to_dict(),
                "energy_history": scene.energy_history,
            }

            dataset.append(scene_data)

            if (scene_idx + 1) % 10 == 0:
                logger.info(
                    f"数据集生成进度: {scene_idx + 1}/{num_scenes}"
                )

        # 保存到磁盘
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 保存为JSON
            output_file = output_path / "physics_dataset.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            logger.info(f"数据集已保存到: {output_file}")

        logger.info(
            f"数据集生成完成: {num_scenes} 个场景, "
            f"每个 {num_steps} 步仿真"
        )
        return dataset

    @staticmethod
    def _sample_sphere_surface(
        center: List[float],
        radius: float,
        num_points: int,
    ) -> List[List[float]]:
        """
        在球体表面均匀采样点（使用斐波那契球面采样）

        Args:
            center: 球心坐标
            radius: 球体半径
            num_points: 采样点数

        Returns:
            采样点坐标列表
        """
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2

        for i in range(num_points):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / num_points)

            x = center[0] + radius * math.sin(phi) * math.cos(theta)
            y = center[1] + radius * math.cos(phi)
            z = center[2] + radius * math.sin(phi) * math.sin(theta)

            points.append([x, y, z])

        return points

    @staticmethod
    def _sample_box_surface(
        center: List[float],
        half_x: float,
        half_y: float,
        half_z: float,
        num_points: int,
    ) -> List[List[float]]:
        """
        在立方体表面均匀采样点

        Args:
            center: 立方体中心坐标
            half_x, half_y, half_z: 三个轴的半尺寸
            num_points: 采样点数

        Returns:
            采样点坐标列表
        """
        points = []
        points_per_face = max(1, num_points // 6)

        # 六个面的法线和参数
        faces = [
            # (轴, 方向, 固定值, 变化轴1, 变化轴2)
            (1, 1, half_y, 0, 2),   # +Y面（顶面）
            (1, -1, -half_y, 0, 2),  # -Y面（底面）
            (0, 1, half_x, 1, 2),   # +X面
            (0, -1, -half_x, 1, 2),  # -X面
            (2, 1, half_z, 0, 1),   # +Z面
            (2, -1, -half_z, 0, 1),  # -Z面
        ]

        for axis, direction, fixed_val, axis1, axis2 in faces:
            half_vals = [half_x, half_y, half_z]
            for _ in range(points_per_face):
                point = list(center)
                point[axis] = center[axis] + direction * fixed_val
                point[axis1] = center[axis1] + random.uniform(
                    -half_vals[axis1], half_vals[axis1]
                )
                point[axis2] = center[axis2] + random.uniform(
                    -half_vals[axis2], half_vals[axis2]
                )
                points.append(point)

        return points

    def __repr__(self) -> str:
        return (
            f"PhysicsDataGenerator(gravity={self.gravity}, "
            f"time_step={self.time_step})"
        )
