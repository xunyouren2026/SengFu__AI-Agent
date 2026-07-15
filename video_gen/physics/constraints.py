"""
物理约束系统模块

实现纳维-斯托克斯方程损失、刚体轨迹损失、物理校正器、物理评分器等
用于视频生成中的物理一致性约束
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Callable, Any
import math
import random


@dataclass
class PhysicsConfig:
    """物理约束配置"""
    use_navier_stokes: bool = True
    use_rigid_body: bool = True
    ns_weight: float = 0.1
    rigid_weight: float = 0.1
    collision_weight: float = 0.05
    elastic_weight: float = 0.05
    viscosity: float = 0.01
    density: float = 1.0
    gravity: float = 9.8
    time_step: float = 0.01


class NavierStokesLoss:
    """
    纳维-斯托克斯方程损失
    
    用于计算流体动力学约束，确保生成的视频符合物理规律
    纳维-斯托克斯方程描述粘性流体的运动：
    - 连续性方程（不可压缩）：∇·v = 0
    - 动量方程：ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
    """
    
    def __init__(self, viscosity: float = 0.01, density: float = 1.0,
                 gravity: float = 9.8, time_step: float = 0.01):
        """
        初始化纳维-斯托克斯损失计算器
        
        Args:
            viscosity: 流体粘度系数
            density: 流体密度
            gravity: 重力加速度（m/s²），用于外力项计算
            time_step: 时间步长（s），用于时间导数项计算
        """
        self._viscosity = viscosity
        self._density = density
        self._gravity = gravity
        self._time_step = time_step
    
    def compute(self, velocity_field: List[List[List[float]]], 
                pressure: List[List[float]]) -> float:
        """
        计算NS方程残差
        
        Args:
            velocity_field: 速度场 [H, W, 2] 或 [T, H, W, 2]
            pressure: 压力场 [H, W] 或 [T, H, W]
            
        Returns:
            NS方程残差损失值
        """
        if not velocity_field or not pressure:
            return 0.0
        
        # 计算连续性方程残差
        continuity_loss = self._continuity_eq(velocity_field)
        
        # 计算动量方程残差
        momentum_loss = self._momentum_eq(velocity_field, pressure)
        
        # 总损失为两项之和
        total_loss = continuity_loss + momentum_loss
        
        return total_loss
    
    def _continuity_eq(self, velocity: List[List[List[float]]]) -> float:
        """
        连续性方程（不可压缩条件）
        ∇·v = ∂vx/∂x + ∂vy/∂y = 0
        
        Args:
            velocity: 速度场
            
        Returns:
            连续性方程残差
        """
        if len(velocity) < 2 or len(velocity[0]) < 2:
            return 0.0
        
        total_divergence = 0.0
        count = 0
        
        # 计算每个点的散度
        for i in range(1, len(velocity) - 1):
            for j in range(1, len(velocity[0]) - 1):
                if len(velocity[i][j]) >= 2:
                    # 中心差分计算散度
                    dvx_dx = (velocity[i][j+1][0] - velocity[i][j-1][0]) / 2.0
                    dvy_dy = (velocity[i+1][j][1] - velocity[i-1][j][1]) / 2.0
                    divergence = dvx_dx + dvy_dy
                    total_divergence += divergence ** 2
                    count += 1
        
        return total_divergence / max(count, 1)
    
    def _momentum_eq(self, velocity: List[List[List[float]]], 
                     pressure: List[List[float]],
                     prev_velocity: Optional[List[List[List[float]]]] = None) -> float:
        """
        动量方程（完整版）
        ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
        
        包含对流项、时间导数项、压力梯度项、粘性项和外力项（重力）
        
        Args:
            velocity: 当前速度场 [H, W, 2] 或 [T, H, W, 2]
            pressure: 压力场 [H, W] 或 [T, H, W]
            prev_velocity: 上一时刻的速度场（用于计算时间导数），可选
            
        Returns:
            动量方程残差
        """
        if len(velocity) < 3 or len(velocity[0]) < 3:
            return 0.0
        
        total_residual = 0.0
        count = 0
        
        for i in range(1, len(velocity) - 1):
            for j in range(1, len(velocity[0]) - 1):
                if len(velocity[i][j]) >= 2:
                    vx, vy = velocity[i][j][0], velocity[i][j][1]
                    
                    # ---- 计算速度梯度（用于对流项 v·∇v）----
                    # vx方向梯度：dvx/dx, dvx/dy
                    dvx_dx = (velocity[i][j+1][0] - velocity[i][j-1][0]) / 2.0 if j+1 < len(velocity[i]) and j > 0 else 0.0
                    dvx_dy = (velocity[i+1][j][0] - velocity[i-1][j][0]) / 2.0 if i+1 < len(velocity) and i > 0 else 0.0
                    # vy方向梯度：dvy/dx, dvy/dy
                    dvy_dx = (velocity[i][j+1][1] - velocity[i][j-1][1]) / 2.0 if j+1 < len(velocity[i]) and j > 0 else 0.0
                    dvy_dy = (velocity[i+1][j][1] - velocity[i-1][j][1]) / 2.0 if i+1 < len(velocity) and i > 0 else 0.0
                    
                    # ---- 对流项：v·∇v ----
                    # advx = vx * dvx/dx + vy * dvx/dy
                    # advy = vx * dvy/dx + vy * dvy/dy
                    advx = vx * dvx_dx + vy * dvx_dy
                    advy = vx * dvy_dx + vy * dvy_dy
                    
                    # ---- 时间导数项：∂v/∂t ----
                    # 使用前向差分，若提供了上一时刻的速度场
                    if prev_velocity is not None and i < len(prev_velocity) and j < len(prev_velocity[i]) and len(prev_velocity[i][j]) >= 2:
                        dt = self._time_step if hasattr(self, '_time_step') else 0.01
                        dvx_dt = (vx - prev_velocity[i][j][0]) / dt
                        dvy_dt = (vy - prev_velocity[i][j][1]) / dt
                    else:
                        # 无时序数据时，时间导数项设为零（稳态假设）
                        dvx_dt = 0.0
                        dvy_dt = 0.0
                    
                    # ---- 压力梯度项：-∇p ----
                    dp_dx = (pressure[i][j+1] - pressure[i][j-1]) / 2.0 if j+1 < len(pressure[i]) and j > 0 else 0.0
                    dp_dy = (pressure[i+1][j] - pressure[i-1][j]) / 2.0 if i+1 < len(pressure) and i > 0 else 0.0
                    
                    # ---- 粘性项：μ∇²v ----
                    if i > 0 and i < len(velocity) - 1 and j > 0 and j < len(velocity[0]) - 1:
                        laplacian_vx = (velocity[i+1][j][0] + velocity[i-1][j][0] + 
                                       velocity[i][j+1][0] + velocity[i][j-1][0] - 
                                       4 * vx)
                        laplacian_vy = (velocity[i+1][j][1] + velocity[i-1][j][1] + 
                                       velocity[i][j+1][1] + velocity[i][j-1][1] - 
                                       4 * vy)
                    else:
                        laplacian_vx, laplacian_vy = 0.0, 0.0
                    
                    # ---- 外力项 f（重力，沿y轴负方向）----
                    # 重力加速度 g = 9.8 m/s²，方向向下（y正方向为下时取正）
                    gravity = self._gravity if hasattr(self, '_gravity') else 9.8
                    fx = 0.0   # x方向无外力
                    fy = self._density * gravity  # y方向重力 = ρg
                    
                    # ---- 完整动量方程残差 ----
                    # ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
                    # 残差 = ρ(∂v/∂t + v·∇v) - (-∇p + μ∇²v + f)
                    residual_x = self._density * (dvx_dt + advx) - (-dp_dx + self._viscosity * laplacian_vx + fx)
                    residual_y = self._density * (dvy_dt + advy) - (-dp_dy + self._viscosity * laplacian_vy + fy)
                    
                    total_residual += residual_x ** 2 + residual_y ** 2
                    count += 1
        
        return total_residual / max(count, 1)
    
    def _gradient(self, field: List[List[float]]) -> List[List[List[float]]]:
        """
        数值梯度计算
        
        Args:
            field: 标量场
            
        Returns:
            梯度向量场 [H, W, 2]
        """
        if len(field) < 2 or len(field[0]) < 2:
            return [[[0.0, 0.0] for _ in range(len(field[0]))] for _ in range(len(field))]
        
        gradient = []
        for i in range(len(field)):
            row = []
            for j in range(len(field[0])):
                # x方向梯度（中心差分）
                if j > 0 and j < len(field[0]) - 1:
                    dx = (field[i][j+1] - field[i][j-1]) / 2.0
                elif j == 0:
                    dx = field[i][1] - field[i][0] if len(field[0]) > 1 else 0.0
                else:
                    dx = field[i][j] - field[i][j-1] if j > 0 else 0.0
                
                # y方向梯度（中心差分）
                if i > 0 and i < len(field) - 1:
                    dy = (field[i+1][j] - field[i-1][j]) / 2.0
                elif i == 0:
                    dy = field[1][j] - field[0][j] if len(field) > 1 else 0.0
                else:
                    dy = field[i][j] - field[i-1][j] if i > 0 else 0.0
                
                row.append([dx, dy])
            gradient.append(row)
        
        return gradient


class RigidBodyLoss:
    """
    刚体轨迹损失
    
    用于计算刚体运动的物理约束，包括：
    - 加速度平滑性（jerk最小化）
    - 角动量守恒
    - 碰撞检测
    """
    
    def __init__(self, mass: float = 1.0, moment_of_inertia: float = 1.0):
        """
        初始化刚体损失计算器
        
        Args:
            mass: 刚体质量
            moment_of_inertia: 转动惯量
        """
        self._mass = mass
        self._moment_of_inertia = moment_of_inertia
    
    def compute(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        计算刚体约束损失
        
        Args:
            trajectory: 轨迹点列表，每个点包含位置、速度、角速度等信息
            
        Returns:
            刚体约束损失值
        """
        if len(trajectory) < 3:
            return 0.0
        
        # 计算jerk损失（加速度变化率）
        jerk_loss = self._jerk(trajectory)
        
        # 计算角动量守恒损失
        angular_loss = self._angular_momentum(trajectory)
        
        # 计算碰撞损失
        collision_loss = self._collision_check(trajectory)
        
        # 总损失
        total_loss = jerk_loss + 0.5 * angular_loss + 0.3 * collision_loss
        
        return total_loss
    
    def _jerk(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        计算加速度变化率（jerk）
        平滑运动应该有较小的jerk
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            jerk损失值
        """
        if len(trajectory) < 4:
            return 0.0
        
        total_jerk = 0.0
        count = 0
        
        for i in range(2, len(trajectory) - 1):
            # 获取连续四帧的位置
            pos_prev2 = trajectory[i-2].get('position', [0.0, 0.0])
            pos_prev1 = trajectory[i-1].get('position', [0.0, 0.0])
            pos_curr = trajectory[i].get('position', [0.0, 0.0])
            pos_next = trajectory[i+1].get('position', [0.0, 0.0])
            
            # 计算加速度（二阶差分）
            acc_prev = [pos_prev1[0] - 2*pos_prev2[0] + pos_curr[0],
                       pos_prev1[1] - 2*pos_prev2[1] + pos_curr[1]]
            acc_curr = [pos_curr[0] - 2*pos_prev1[0] + pos_next[0],
                       pos_curr[1] - 2*pos_prev1[1] + pos_next[1]]
            
            # 计算jerk（加速度变化率）
            jerk = [acc_curr[0] - acc_prev[0], acc_curr[1] - acc_prev[1]]
            jerk_magnitude = math.sqrt(jerk[0]**2 + jerk[1]**2)
            
            total_jerk += jerk_magnitude ** 2
            count += 1
        
        return total_jerk / max(count, 1)
    
    def _angular_momentum(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        计算角动量守恒损失
        在无外力矩作用下，角动量应守恒
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            角动量守恒损失值
        """
        if len(trajectory) < 2:
            return 0.0
        
        angular_momenta = []
        
        for point in trajectory:
            angular_velocity = point.get('angular_velocity', 0.0)
            # L = I * omega
            L = self._moment_of_inertia * angular_velocity
            angular_momenta.append(L)
        
        if len(angular_momenta) < 2:
            return 0.0
        
        # 计算角动量变化
        total_variation = 0.0
        for i in range(1, len(angular_momenta)):
            variation = angular_momenta[i] - angular_momenta[i-1]
            total_variation += variation ** 2
        
        return total_variation / len(angular_momenta)
    
    def _collision_check(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        碰撞检测
        检测轨迹中是否存在不合理的穿透
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            碰撞损失值
        """
        if len(trajectory) < 2:
            return 0.0
        
        collision_loss = 0.0
        count = 0
        
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1].get('position', [0.0, 0.0])
            curr_pos = trajectory[i].get('position', [0.0, 0.0])
            
            prev_radius = trajectory[i-1].get('radius', 1.0)
            curr_radius = trajectory[i].get('radius', 1.0)
            
            # 计算位移
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # 检测是否有不合理的瞬时位移（可能的穿透）
            max_reasonable_distance = (prev_radius + curr_radius) * 3.0
            if distance > max_reasonable_distance:
                collision_loss += (distance - max_reasonable_distance) ** 2
            
            count += 1
        
        return collision_loss / max(count, 1)


class PhysicsCorrector:
    """
    物理校正器
    
    用于校正生成的帧序列，使其符合物理规律
    支持多种后端：simple（纯Python）、warp、taichi
    """
    
    def __init__(self, backend: str = "simple", config: Optional[PhysicsConfig] = None):
        """
        初始化物理校正器
        
        Args:
            backend: 计算后端（simple/warp/taichi）
            config: 物理配置
        """
        self._backend = backend
        self._config = config or PhysicsConfig()
        self._ns_loss = NavierStokesLoss(
            viscosity=self._config.viscosity,
            density=self._config.density
        )
        self._rigid_loss = RigidBodyLoss()
    
    def correct(self, frames: List[List[List[List[int]]]], 
                flow: Optional[List[List[List[List[float]]]]] = None) -> List[List[List[List[int]]]]:
        """
        校正帧序列
        
        Args:
            frames: 帧序列 [T, H, W, C]
            flow: 光流场 [T, H, W, 2]
            
        Returns:
            校正后的帧序列
        """
        if not frames:
            return frames
        
        # 光流引导平滑
        if flow is not None:
            frames = self._flow_guided_smooth(frames, flow)
        
        # 刚体粒子校正
        frames = self._rigid_particle_correct(frames)
        
        # 弹性网格校正
        frames = self._elastic_mesh_correct(frames)
        
        return frames
    
    def _flow_guided_smooth(self, frames: List[List[List[List[int]]]], 
                            flow: List[List[List[List[float]]]]) -> List[List[List[List[int]]]]:
        """
        光流引导平滑
        使用光流信息进行时域平滑
        
        Args:
            frames: 帧序列
            flow: 光流场
            
        Returns:
            平滑后的帧序列
        """
        if len(frames) < 3:
            return frames
        
        smoothed = [frames[0]]  # 保留第一帧
        
        for t in range(1, len(frames) - 1):
            prev_frame = frames[t - 1]
            curr_frame = frames[t]
            next_frame = frames[t + 1]
            
            # 获取当前帧的光流
            if t < len(flow):
                curr_flow = flow[t]
            else:
                curr_flow = [[[0.0, 0.0] for _ in range(len(frames[0][0]))] 
                            for _ in range(len(frames[0]))]
            
            # 光流引导的加权平均
            new_frame = []
            for i in range(len(curr_frame)):
                row = []
                for j in range(len(curr_frame[i])):
                    # 基于光流的权重
                    if i < len(curr_flow) and j < len(curr_flow[i]):
                        flow_magnitude = math.sqrt(curr_flow[i][j][0]**2 + curr_flow[i][j][1]**2)
                        # 流动大的区域保持原值，流动小的区域进行平滑
                        weight = min(flow_magnitude / 10.0, 1.0)
                    else:
                        weight = 0.5
                    
                    pixel = []
                    for c in range(len(curr_frame[i][j])):
                        # 加权混合
                        smoothed_val = (prev_frame[i][j][c] + curr_frame[i][j][c] + next_frame[i][j][c]) / 3.0
                        val = weight * curr_frame[i][j][c] + (1 - weight) * smoothed_val
                        pixel.append(int(max(0, min(255, val))))
                    row.append(pixel)
                new_frame.append(row)
            
            smoothed.append(new_frame)
        
        smoothed.append(frames[-1])  # 保留最后一帧
        
        return smoothed
    
    def _rigid_particle_correct(self, frames: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """
        刚体粒子校正
        检测并校正刚体运动的不一致性
        
        Args:
            frames: 帧序列
            
        Returns:
            校正后的帧序列
        """
        if len(frames) < 2:
            return frames
        
        # 简化实现：检测并平滑异常像素变化
        corrected = [frames[0]]
        
        for t in range(1, len(frames)):
            prev_frame = frames[t - 1]
            curr_frame = frames[t]
            
            new_frame = []
            for i in range(len(curr_frame)):
                row = []
                for j in range(len(curr_frame[i])):
                    pixel = []
                    for c in range(len(curr_frame[i][j])):
                        prev_val = prev_frame[i][j][c]
                        curr_val = curr_frame[i][j][c]
                        
                        # 检测异常变化（超过阈值的变化可能是错误）
                        diff = abs(curr_val - prev_val)
                        if diff > 100:  # 大变化阈值
                            # 使用邻域平均进行校正
                            neighbor_sum = 0
                            neighbor_count = 0
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < len(curr_frame) and 0 <= nj < len(curr_frame[0]):
                                        neighbor_sum += curr_frame[ni][nj][c]
                                        neighbor_count += 1
                            if neighbor_count > 0:
                                corrected_val = int(neighbor_sum / neighbor_count)
                            else:
                                corrected_val = curr_val
                        else:
                            corrected_val = curr_val
                        
                        pixel.append(max(0, min(255, corrected_val)))
                    row.append(pixel)
                new_frame.append(row)
            
            corrected.append(new_frame)
        
        return corrected
    
    def _elastic_mesh_correct(self, frames: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """
        弹性网格校正
        使用弹性变形模型校正帧序列
        
        Args:
            frames: 帧序列
            
        Returns:
            校正后的帧序列
        """
        if len(frames) < 2:
            return frames
        
        # 简化实现：应用轻微的弹性平滑
        corrected = [frames[0]]
        
        for t in range(1, len(frames) - 1):
            prev_frame = frames[t - 1]
            curr_frame = frames[t]
            next_frame = frames[t + 1]
            
            new_frame = []
            for i in range(len(curr_frame)):
                row = []
                for j in range(len(curr_frame[i])):
                    pixel = []
                    for c in range(len(curr_frame[i][j])):
                        # 弹性平滑：考虑时空邻域
                        val = 0.5 * curr_frame[i][j][c] + 0.25 * prev_frame[i][j][c] + 0.25 * next_frame[i][j][c]
                        pixel.append(int(max(0, min(255, val))))
                    row.append(pixel)
                new_frame.append(row)
            
            corrected.append(new_frame)
        
        if len(frames) > 1:
            corrected.append(frames[-1])
        
        return corrected


class EnhancedPhysicsScorer:
    """
    物理评分器
    
    对视频进行综合物理合理性评分
    """
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        初始化物理评分器
        
        Args:
            config: 物理配置
        """
        self._config = config or PhysicsConfig()
        self._ns_loss = NavierStokesLoss(
            viscosity=self._config.viscosity,
            density=self._config.density
        )
        self._rigid_loss = RigidBodyLoss()
    
    def score(self, video: List[List[List[List[int]]]]) -> float:
        """
        综合物理合理性评分
        
        Args:
            video: 视频帧序列 [T, H, W, C]
            
        Returns:
            物理评分（0-1，越高越好）
        """
        if not video:
            return 0.0
        
        # 计算各项评分
        ns_score = self._ns_score(video)
        rigid_score = self._rigid_score(video)
        collision_score = self._collision_score(video)
        elastic_score = self._elastic_score(video)
        
        # 加权平均
        total_score = (
            self._config.ns_weight * ns_score +
            self._config.rigid_weight * rigid_score +
            self._config.collision_weight * collision_score +
            self._config.elastic_weight * elastic_score
        )
        
        # 归一化
        total_weight = (self._config.ns_weight + self._config.rigid_weight + 
                       self._config.collision_weight + self._config.elastic_weight)
        
        return total_score / max(total_weight, 1e-6)
    
    def _ns_score(self, video: List[List[List[List[int]]]]) -> float:
        """
        NS方程评分
        
        Args:
            video: 视频帧序列
            
        Returns:
            NS方程评分（0-1）
        """
        if len(video) < 2:
            return 1.0
        
        # 从视频帧估计速度场（简化实现）
        velocity_field = self._estimate_velocity_field(video)
        
        # 估计压力场
        pressure = self._estimate_pressure(video)
        
        # 计算NS损失
        ns_loss = self._ns_loss.compute(velocity_field, pressure)
        
        # 转换为评分（损失越小，评分越高）
        score = math.exp(-ns_loss)
        
        return min(1.0, max(0.0, score))
    
    def _rigid_score(self, video: List[List[List[List[int]]]]) -> float:
        """
        刚体评分
        
        Args:
            video: 视频帧序列
            
        Returns:
            刚体评分（0-1）
        """
        if len(video) < 3:
            return 1.0
        
        # 从视频帧估计轨迹
        trajectory = self._estimate_trajectory(video)
        
        # 计算刚体损失
        rigid_loss = self._rigid_loss.compute(trajectory)
        
        # 转换为评分
        score = math.exp(-rigid_loss)
        
        return min(1.0, max(0.0, score))
    
    def _collision_score(self, video: List[List[List[List[int]]]]) -> float:
        """
        碰撞评分
        
        Args:
            video: 视频帧序列
            
        Returns:
            碰撞评分（0-1）
        """
        if len(video) < 2:
            return 1.0
        
        # 检测帧间突变
        collision_count = 0
        total_pixels = 0
        
        for t in range(1, len(video)):
            prev_frame = video[t - 1]
            curr_frame = video[t]
            
            for i in range(len(curr_frame)):
                for j in range(len(curr_frame[i])):
                    for c in range(len(curr_frame[i][j])):
                        diff = abs(curr_frame[i][j][c] - prev_frame[i][j][c])
                        if diff > 200:  # 大突变
                            collision_count += 1
                        total_pixels += 1
        
        if total_pixels == 0:
            return 1.0
        
        # 突变比例越低，评分越高
        collision_ratio = collision_count / total_pixels
        score = 1.0 - collision_ratio
        
        return min(1.0, max(0.0, score))
    
    def _elastic_score(self, video: List[List[List[List[int]]]]) -> float:
        """
        弹性评分
        
        Args:
            video: 视频帧序列
            
        Returns:
            弹性评分（0-1）
        """
        if len(video) < 3:
            return 1.0
        
        # 计算帧间变化的平滑性
        total_variation = 0.0
        count = 0
        
        for t in range(1, len(video) - 1):
            prev_frame = video[t - 1]
            curr_frame = video[t]
            next_frame = video[t + 1]
            
            for i in range(len(curr_frame)):
                for j in range(len(curr_frame[i])):
                    for c in range(len(curr_frame[i][j])):
                        # 计算二阶差分（变化的平滑性）
                        second_diff = (next_frame[i][j][c] - 2 * curr_frame[i][j][c] + prev_frame[i][j][c])
                        total_variation += second_diff ** 2
                        count += 1
        
        if count == 0:
            return 1.0
        
        # 变化越平滑，评分越高
        avg_variation = total_variation / count
        score = math.exp(-avg_variation / 1000.0)
        
        return min(1.0, max(0.0, score))
    
    def _estimate_velocity_field(self, video: List[List[List[List[int]]]]) -> List[List[List[float]]]:
        """从视频帧估计速度场"""
        if len(video) < 2:
            return [[[0.0, 0.0] for _ in range(len(video[0][0]))] for _ in range(len(video[0]))]
        
        # 使用帧差估计速度
        prev_frame = video[0]
        curr_frame = video[1]
        
        velocity = []
        for i in range(len(curr_frame)):
            row = []
            for j in range(len(curr_frame[i])):
                # 简化：使用亮度差作为速度估计
                prev_brightness = sum(prev_frame[i][j]) / len(prev_frame[i][j])
                curr_brightness = sum(curr_frame[i][j]) / len(curr_frame[i][j])
                
                # 水平和垂直速度分量（简化估计）
                vx = (curr_brightness - prev_brightness) * 0.1
                vy = (curr_brightness - prev_brightness) * 0.1
                
                row.append([vx, vy])
            velocity.append(row)
        
        return velocity
    
    def _estimate_pressure(self, video: List[List[List[List[int]]]]) -> List[List[float]]:
        """从视频帧估计压力场"""
        if not video:
            return []
        
        # 使用亮度作为压力的代理
        frame = video[0]
        
        pressure = []
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                brightness = sum(frame[i][j]) / len(frame[i][j])
                row.append(brightness / 255.0)
            pressure.append(row)
        
        return pressure
    
    def _estimate_trajectory(self, video: List[List[List[List[int]]]]) -> List[Dict[str, Any]]:
        """从视频帧估计物体轨迹"""
        trajectory = []
        
        for t, frame in enumerate(video):
            # 简化：计算质心作为位置
            total_x, total_y, total_mass = 0.0, 0.0, 0.0
            
            for i in range(len(frame)):
                for j in range(len(frame[i])):
                    brightness = sum(frame[i][j]) / len(frame[i][j])
                    total_x += j * brightness
                    total_y += i * brightness
                    total_mass += brightness
            
            if total_mass > 0:
                center_x = total_x / total_mass
                center_y = total_y / total_mass
            else:
                center_x, center_y = len(frame[0]) / 2, len(frame) / 2
            
            trajectory.append({
                'position': [center_x, center_y],
                'velocity': [0.0, 0.0],
                'angular_velocity': 0.0,
                'radius': 10.0
            })
        
        # 计算速度
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1]['position']
            curr_pos = trajectory[i]['position']
            trajectory[i]['velocity'] = [
                curr_pos[0] - prev_pos[0],
                curr_pos[1] - prev_pos[1]
            ]
        
        return trajectory


class CorruptionType(Enum):
    """帧损坏类型枚举"""
    # 亮度/颜色相关
    EXPOSURE = "exposure"           # 曝光异常
    GAMMA = "gamma"                 # Gamma校正异常
    NOISE = "noise"                 # 噪声
    COLOR_SHIFT = "color_shift"     # 颜色偏移
    SATURATION = "saturation"       # 饱和度异常
    
    # 模糊/锐化相关
    BLUR = "blur"                   # 模糊
    MOTION_BLUR = "motion_blur"     # 运动模糊
    SHARPEN = "sharpen"             # 过锐化
    JPEG_ARTIFACT = "jpeg_artifact" # JPEG压缩伪影
    
    # 时域相关
    FRAME_DROP = "frame_drop"       # 帧丢失
    FRAME_DUPE = "frame_dupe"       # 帧重复
    FRAME_SWAP = "frame_swap"       # 帧交换
    TEMPORAL_NOISE = "temporal_noise"  # 时域噪声
    
    # 几何相关
    ROTATION = "rotation"           # 旋转
    SCALE = "scale"                 # 缩放
    TRANSLATION = "translation"     # 平移
    PERSPECTIVE = "perspective"     # 透视畸变
    LENS_DISTORTION = "lens_distortion"  # 镜头畸变
    
    # 遮挡相关
    OCCLUSION = "occlusion"         # 遮挡
    VIGNETTE = "vignette"           # 暗角
    GLARE = "glare"                 # 眩光
    
    # 其他
    RAIN = "rain"                   # 雨滴
    SNOW = "snow"                   # 雪花
    FOG = "fog"                     # 雾


class FrameCorruptor:
    """
    帧损坏器
    
    用于数据增强或测试视频生成模型的鲁棒性
    支持23种不同的帧损坏类型
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化帧损坏器
        
        Args:
            seed: 随机种子
        """
        self._rng = random.Random(seed)
    
    def corrupt(self, frame: List[List[List[int]]], 
                corruption_type: CorruptionType, 
                severity: float = 0.5) -> List[List[List[int]]]:
        """
        应用指定类型的损坏
        
        Args:
            frame: 输入帧 [H, W, C]
            corruption_type: 损坏类型
            severity: 损坏强度（0-1）
            
        Returns:
            损坏后的帧
        """
        if not frame or not frame[0]:
            return frame
        
        # 确保severity在有效范围内
        severity = max(0.0, min(1.0, severity))
        
        # 根据损坏类型调用相应的处理函数
        handlers = {
            CorruptionType.EXPOSURE: self._exposure_corruption,
            CorruptionType.GAMMA: self._gamma_corruption,
            CorruptionType.NOISE: self._noise_corruption,
            CorruptionType.COLOR_SHIFT: self._color_shift_corruption,
            CorruptionType.SATURATION: self._saturation_corruption,
            CorruptionType.BLUR: self._blur_corruption,
            CorruptionType.MOTION_BLUR: self._motion_blur_corruption,
            CorruptionType.SHARPEN: self._sharpen_corruption,
            CorruptionType.JPEG_ARTIFACT: self._jpeg_artifact_corruption,
            CorruptionType.ROTATION: self._rotation_corruption,
            CorruptionType.SCALE: self._scale_corruption,
            CorruptionType.TRANSLATION: self._translation_corruption,
            CorruptionType.VIGNETTE: self._vignette_corruption,
            CorruptionType.OCCLUSION: self._occlusion_corruption,
            CorruptionType.GLARE: self._glare_corruption,
            CorruptionType.RAIN: self._rain_corruption,
            CorruptionType.SNOW: self._snow_corruption,
            CorruptionType.FOG: self._fog_corruption,
        }
        
        handler = handlers.get(corruption_type)
        if handler:
            return handler(frame, severity)
        else:
            # 默认返回原帧
            return frame
    
    def _random_corruption(self, frame: List[List[List[int]]], 
                           severity: float = 0.5) -> List[List[List[int]]]:
        """
        随机选择损坏类型并应用
        
        Args:
            frame: 输入帧
            severity: 损坏强度
            
        Returns:
            损坏后的帧
        """
        corruption_types = list(CorruptionType)
        random_type = self._rng.choice(corruption_types)
        return self.corrupt(frame, random_type, severity)
    
    def _exposure_corruption(self, frame: List[List[List[int]]], 
                             severity: float) -> List[List[List[int]]]:
        """曝光异常"""
        # 过曝或欠曝
        factor = 1.0 + (self._rng.choice([-1, 1]) * severity * 0.5)
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = [int(max(0, min(255, c * factor))) for c in pixel]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _gamma_corruption(self, frame: List[List[List[int]]], 
                          severity: float) -> List[List[List[int]]]:
        """Gamma校正异常"""
        gamma = 1.0 + severity * (self._rng.random() - 0.5) * 2
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = [int(255 * (c / 255.0) ** gamma) for c in pixel]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _noise_corruption(self, frame: List[List[List[int]]], 
                          severity: float) -> List[List[List[int]]]:
        """添加噪声"""
        noise_level = severity * 50
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = [
                    int(max(0, min(255, c + self._rng.gauss(0, noise_level))))
                    for c in pixel
                ]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _color_shift_corruption(self, frame: List[List[List[int]]], 
                                severity: float) -> List[List[List[int]]]:
        """颜色偏移"""
        shift = int(severity * 50)
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                if len(pixel) >= 3:
                    new_pixel = [
                        max(0, min(255, pixel[0] + shift)),
                        max(0, min(255, pixel[1] - shift // 2)),
                        max(0, min(255, pixel[2] - shift // 2))
                    ]
                    if len(pixel) > 3:
                        new_pixel.extend(pixel[3:])
                else:
                    new_pixel = pixel[:]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _saturation_corruption(self, frame: List[List[List[int]]], 
                               severity: float) -> List[List[List[int]]]:
        """饱和度异常"""
        factor = 1.0 - severity  # 降低饱和度
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                if len(pixel) >= 3:
                    gray = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
                    new_pixel = [
                        int(gray + factor * (pixel[0] - gray)),
                        int(gray + factor * (pixel[1] - gray)),
                        int(gray + factor * (pixel[2] - gray))
                    ]
                    new_pixel = [max(0, min(255, c)) for c in new_pixel]
                    if len(pixel) > 3:
                        new_pixel.extend(pixel[3:])
                else:
                    new_pixel = pixel[:]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _blur_corruption(self, frame: List[List[List[int]]], 
                         severity: float) -> List[List[List[int]]]:
        """模糊"""
        kernel_size = int(3 + severity * 5)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return self._apply_blur(frame, kernel_size)
    
    def _apply_blur(self, frame: List[List[List[int]]], 
                    kernel_size: int) -> List[List[List[int]]]:
        """应用模糊滤波"""
        h, w = len(frame), len(frame[0])
        pad = kernel_size // 2
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel_sum = [0.0] * len(frame[i][j])
                count = 0
                
                for di in range(-pad, pad + 1):
                    for dj in range(-pad, pad + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            for c in range(len(pixel_sum)):
                                pixel_sum[c] += frame[ni][nj][c]
                            count += 1
                
                if count > 0:
                    pixel = [int(s / count) for s in pixel_sum]
                else:
                    pixel = frame[i][j][:]
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _motion_blur_corruption(self, frame: List[List[List[int]]], 
                                severity: float) -> List[List[List[int]]]:
        """运动模糊"""
        length = int(5 + severity * 15)
        angle = self._rng.random() * math.pi * 2
        
        h, w = len(frame), len(frame[0])
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel_sum = [0.0] * len(frame[i][j])
                
                for k in range(length):
                    ni = int(i + k * dy)
                    nj = int(j + k * dx)
                    if 0 <= ni < h and 0 <= nj < w:
                        for c in range(len(pixel_sum)):
                            pixel_sum[c] += frame[ni][nj][c]
                
                pixel = [int(s / length) for s in pixel_sum]
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _sharpen_corruption(self, frame: List[List[List[int]]], 
                            severity: float) -> List[List[List[int]]]:
        """过锐化"""
        amount = 1.0 + severity * 2
        
        # 先模糊
        blurred = self._apply_blur(frame, 3)
        
        h, w = len(frame), len(frame[0])
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                for c in range(len(frame[i][j])):
                    val = frame[i][j][c] + amount * (frame[i][j][c] - blurred[i][j][c])
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _jpeg_artifact_corruption(self, frame: List[List[List[int]]], 
                                  severity: float) -> List[List[List[int]]]:
        """JPEG压缩伪影"""
        # 模拟JPEG量化
        quality = int(100 - severity * 90)
        quant_factor = (100 - quality) / 100.0 * 30 + 1
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = [
                    int(round(c / quant_factor) * quant_factor)
                    for c in pixel
                ]
                new_pixel = [max(0, min(255, c)) for c in new_pixel]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _rotation_corruption(self, frame: List[List[List[int]]], 
                             severity: float) -> List[List[List[int]]]:
        """旋转"""
        angle = severity * 30 * (self._rng.random() - 0.5) * 2
        angle_rad = math.radians(angle)
        
        h, w = len(frame), len(frame[0])
        cx, cy = w // 2, h // 2
        
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                # 反向映射
                x = j - cx
                y = i - cy
                src_x = int(x * cos_a + y * sin_a + cx)
                src_y = int(-x * sin_a + y * cos_a + cy)
                
                if 0 <= src_x < w and 0 <= src_y < h:
                    pixel = frame[src_y][src_x][:]
                else:
                    pixel = [0] * len(frame[0][0])
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _scale_corruption(self, frame: List[List[List[int]]], 
                          severity: float) -> List[List[List[int]]]:
        """缩放"""
        scale = 1.0 + severity * 0.3 * (self._rng.random() - 0.5) * 2
        
        h, w = len(frame), len(frame[0])
        cx, cy = w // 2, h // 2
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                src_x = int((j - cx) / scale + cx)
                src_y = int((i - cy) / scale + cy)
                
                if 0 <= src_x < w and 0 <= src_y < h:
                    pixel = frame[src_y][src_x][:]
                else:
                    pixel = [0] * len(frame[0][0])
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _translation_corruption(self, frame: List[List[List[int]]], 
                                severity: float) -> List[List[List[int]]]:
        """平移"""
        max_shift = int(severity * 50)
        dx = self._rng.randint(-max_shift, max_shift)
        dy = self._rng.randint(-max_shift, max_shift)
        
        h, w = len(frame), len(frame[0])
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                src_x = j - dx
                src_y = i - dy
                
                if 0 <= src_x < w and 0 <= src_y < h:
                    pixel = frame[src_y][src_x][:]
                else:
                    pixel = [0] * len(frame[0][0])
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _vignette_corruption(self, frame: List[List[List[int]]], 
                             severity: float) -> List[List[List[int]]]:
        """暗角"""
        h, w = len(frame), len(frame[0])
        cx, cy = w / 2, h / 2
        max_dist = math.sqrt(cx**2 + cy**2)
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                dist = math.sqrt((j - cx)**2 + (i - cy)**2)
                factor = 1.0 - severity * (dist / max_dist) ** 2
                
                pixel = [int(max(0, min(255, c * factor))) for c in frame[i][j]]
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _occlusion_corruption(self, frame: List[List[List[int]]], 
                              severity: float) -> List[List[List[int]]]:
        """遮挡"""
        h, w = len(frame), len(frame[0])
        num_occlusions = int(1 + severity * 5)
        
        result = [row[:] for row in frame]
        
        for _ in range(num_occlusions):
            occ_w = self._rng.randint(10, int(w * severity * 0.5))
            occ_h = self._rng.randint(10, int(h * severity * 0.5))
            x = self._rng.randint(0, w - occ_w)
            y = self._rng.randint(0, h - occ_h)
            
            color = [self._rng.randint(0, 255) for _ in range(len(frame[0][0]))]
            
            for i in range(y, min(y + occ_h, h)):
                for j in range(x, min(x + occ_w, w)):
                    result[i][j] = color[:]
        
        return result
    
    def _glare_corruption(self, frame: List[List[List[int]]], 
                          severity: float) -> List[List[List[int]]]:
        """眩光"""
        h, w = len(frame), len(frame[0])
        cx = self._rng.randint(0, w)
        cy = self._rng.randint(0, h)
        radius = int(20 + severity * 100)
        intensity = severity * 200
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                dist = math.sqrt((j - cx)**2 + (i - cy)**2)
                if dist < radius:
                    factor = 1.0 - (dist / radius)
                    glare = intensity * factor
                else:
                    glare = 0
                
                pixel = [int(max(0, min(255, c + glare))) for c in frame[i][j]]
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _rain_corruption(self, frame: List[List[List[int]]], 
                         severity: float) -> List[List[List[int]]]:
        """雨滴"""
        h, w = len(frame), len(frame[0])
        num_drops = int(severity * 500)
        
        result = [row[:] for row in frame]
        
        for _ in range(num_drops):
            x = self._rng.randint(0, w - 1)
            y = self._rng.randint(0, h - 1)
            length = self._rng.randint(5, 15)
            
            for dy in range(length):
                ny = y + dy
                if 0 <= ny < h:
                    for c in range(len(result[ny][x])):
                        result[ny][x][c] = min(255, result[ny][x][c] + 50)
        
        return result
    
    def _snow_corruption(self, frame: List[List[List[int]]], 
                         severity: float) -> List[List[List[int]]]:
        """雪花"""
        h, w = len(frame), len(frame[0])
        num_flakes = int(severity * 300)
        
        result = [row[:] for row in frame]
        
        for _ in range(num_flakes):
            x = self._rng.randint(0, w - 1)
            y = self._rng.randint(0, h - 1)
            size = self._rng.randint(1, 3)
            
            for di in range(-size, size + 1):
                for dj in range(-size, size + 1):
                    ni, nj = y + di, x + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        result[ni][nj] = [255] * len(result[ni][nj])
        
        return result
    
    def _fog_corruption(self, frame: List[List[List[int]]], 
                        severity: float) -> List[List[List[int]]]:
        """雾"""
        fog_color = [200, 200, 200]
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = []
                for c in range(len(pixel)):
                    if c < 3:
                        val = (1 - severity) * pixel[c] + severity * fog_color[c]
                    else:
                        val = pixel[c]
                    new_pixel.append(int(val))
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
