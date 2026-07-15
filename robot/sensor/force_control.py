#!/usr/bin/env python3
"""
力控模块 - 完整实现
支持：
- 力/位混合控制（Hybrid Force/Position Control）
- 阻抗控制（Impedance Control）
- 导纳控制（Admittance Control）
- 实时力控循环（独立线程）
- 安全力限制
"""

import threading
import time
from typing import Tuple, List, Optional, Callable
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller_base import RobotControllerBase


def create_diag_matrix(diagonal: List[float]) -> List[List[float]]:
    """创建对角矩阵（标准库实现，替代numpy.diag）"""
    n = len(diagonal)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = diagonal[i]
    return matrix


def mat_vec_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """矩阵-向量乘法（标准库实现）"""
    n = len(vector)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]
    return result


def mat_add(mat1: List[List[float]], mat2: List[List[float]]) -> List[List[float]]:
    """矩阵加法"""
    n = len(mat1)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = mat1[i][j] + mat2[i][j]
    return result


def invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """矩阵求逆（使用高斯-约旦消元法）"""
    n = len(matrix)
    # 创建增广矩阵 [A | I]
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    
    # 前向消元
    for col in range(n):
        # 找到主元
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        if abs(aug[col][col]) < 1e-10:
            raise ValueError("Matrix is singular")
        
        # 归一化主元行
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        
        # 消元
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    
    # 提取逆矩阵
    inverse = [row[n:] for row in aug]
    return inverse


class SimpleArray:
    """简单数组类，替代numpy.ndarray的部分功能"""
    def __init__(self, data: List[float]):
        self._data = list(data)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __add__(self, other):
        if isinstance(other, SimpleArray):
            return SimpleArray([a + b for a, b in zip(self._data, other._data)])
        return SimpleArray([a + other for a in self._data])
    
    def __sub__(self, other):
        if isinstance(other, SimpleArray):
            return SimpleArray([a - b for a, b in zip(self._data, other._data)])
        return SimpleArray([a - other for a in self._data])
    
    def __mul__(self, scalar: float):
        return SimpleArray([a * scalar for a in self._data])
    
    def copy(self):
        return SimpleArray(list(self._data))
    
    @property
    def shape(self):
        return (len(self._data),)
    
    def tolist(self):
        return list(self._data)


class ForceController:
    """力控制器（完整实现）"""

    def __init__(self, robot: RobotControllerBase, control_cycle: float = 0.01):
        """
        初始化力控制器
        robot: 机器人控制器实例（需支持 set_force_torque, get_force_torque, get_tcp_pose, move_cartesian）
        control_cycle: 控制周期（秒），默认100Hz
        """
        self.robot = robot
        self.dt = control_cycle
        self._enabled = False
        self._running = False
        self._control_thread = None
        self._lock = threading.Lock()

        # 阻抗控制参数（6维）
        # M: 惯性矩阵 (6x6)
        # D: 阻尼矩阵 (6x6)
        # K: 刚度矩阵 (6x6)
        self.M = create_diag_matrix([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        self.D = create_diag_matrix([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        self.K = create_diag_matrix([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])

        # 期望值
        self._desired_pose = None           # 期望位姿 (6,)
        self._desired_force = [0.0] * 6      # 期望力/力矩 (6,)
        self._actual_pose = None            # 实际位姿 (6,)
        self._actual_force = [0.0] * 6      # 实际力/力矩 (6,)

        # 力/位混合控制掩码：True表示力控，False表示位置控
        self._force_mask = [False] * 6

        # 安全限制
        self.max_force = 100.0      # 最大力 (N)
        self.max_torque = 50.0      # 最大力矩 (Nm)
        self.max_velocity = 0.1     # 最大线速度 (m/s)

        # 积分项（用于PI控制）
        self._force_error_integral = [0.0] * 6

    # ==================== 参数设置 ====================
    def set_impedance_params(self, M: List[List[float]], D: List[List[float]], K: List[List[float]]):
        """设置阻抗参数（6x6矩阵）"""
        with self._lock:
            self.M = M
            self.D = D
            self.K = K

    def set_desired_pose(self, pose: Tuple[float, float, float, float, float, float]):
        """设置期望位姿 (x, y, z, rx, ry, rz)"""
        self._desired_pose = list(pose)

    def set_desired_force(self, force: Tuple[float, float, float], torque: Tuple[float, float, float]):
        """设置期望力/力矩"""
        self._desired_force = list(force) + list(torque)

    def set_force_mask(self, mask: List[bool]):
        """
        设置力/位混合控制掩码
        mask[i] = True 表示第i维进行力控，False表示位置控
        顺序: [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        assert len(mask) == 6, "Mask must have 6 elements"
        self._force_mask = list(mask)

    def set_safety_limits(self, max_force: float = 100.0, max_torque: float = 50.0, max_velocity: float = 0.1):
        """设置安全限制"""
        self.max_force = max_force
        self.max_torque = max_torque
        self.max_velocity = max_velocity

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """限制值在范围内"""
        return max(min_val, min(max_val, value))

    # ==================== 力控模式开关 ====================
    def enable(self):
        """启用力控模式（启动独立控制线程）"""
        if self._enabled:
            return
        self._enabled = True
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("[ForceControl] Enabled")

    def disable(self):
        """禁用力控模式"""
        self._enabled = False
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        print("[ForceControl] Disabled")

    def is_enabled(self) -> bool:
        return self._enabled

    # ==================== 力控核心算法 ====================
    def _control_loop(self):
        """实时力控循环（在独立线程中运行）"""
        last_time = time.time()
        # 状态变量（用于阻抗控制中的导数计算）
        x_prev = None
        dx_prev = None

        while self._running and self.robot.is_connected:
            now = time.time()
            dt = now - last_time
            if dt < self.dt:
                time.sleep(self.dt - dt)
                continue
            last_time = now

            try:
                # 1. 获取当前状态
                pose = self.robot.get_tcp_pose()
                self._actual_pose = list(pose)
                force, torque = self.robot.get_force_torque()
                self._actual_force = list(force) + list(torque)

                # 2. 计算控制输出（根据当前模式）
                if self._desired_pose is not None:
                    # 有期望位姿，使用阻抗控制或力/位混合控制
                    if any(self._force_mask):
                        cmd = self._hybrid_control(dt, x_prev, dx_prev)
                    else:
                        cmd = self._impedance_control(dt, x_prev, dx_prev)
                else:
                    # 无期望位姿，纯力控制（直接力跟踪）
                    cmd = self._pure_force_control()

                # 3. 安全限制
                cmd = self._apply_safety_limits(cmd)

                # 4. 发送力/力矩指令
                self.robot.set_force_torque(tuple(cmd[:3]), tuple(cmd[3:6]))

                # 5. 更新状态变量
                if self._desired_pose is not None:
                    x_prev = list(self._actual_pose)
                    if dx_prev is not None and dt > 0:
                        dx_prev = [(self._actual_pose[i] - x_prev[i]) / dt for i in range(6)]
                    else:
                        dx_prev = [0.0] * 6

            except Exception as e:
                print(f"[ForceControl] Loop error: {e}")

    def _impedance_control(self, dt: float, x_prev: List[float], dx_prev: List[float]) -> List[float]:
        """
        阻抗控制：M * dd_e + D * de + K * e = F_ext
        输出：所需的末端力/力矩
        """
        # 位置误差
        e = [self._desired_pose[i] - self._actual_pose[i] for i in range(6)]

        # 速度误差（使用数值微分）
        if x_prev is not None and dt > 0:
            de = [(self._actual_pose[i] - x_prev[i]) / dt for i in range(6)]
            if dx_prev is not None:
                dde = [(de[i] - dx_prev[i]) / dt for i in range(6)]
            else:
                dde = [0.0] * 6
        else:
            de = [0.0] * 6
            dde = [0.0] * 6

        # 阻抗控制律: F = K @ e + D @ de + M @ dde
        Ke = mat_vec_mul(self.K, e)
        Dde = mat_vec_mul(self.D, de)
        Mdde = mat_vec_mul(self.M, dde)
        F_desired = [Ke[i] + Dde[i] + Mdde[i] for i in range(6)]
        return F_desired

    def _hybrid_control(self, dt: float, x_prev: List[float], dx_prev: List[float]) -> List[float]:
        """
        力/位混合控制
        根据 _force_mask 选择每个维度是力控还是位置控
        """
        # 位置误差
        e = [self._desired_pose[i] - self._actual_pose[i] for i in range(6)]

        # 速度误差
        if x_prev is not None and dt > 0:
            de = [(self._actual_pose[i] - x_prev[i]) / dt for i in range(6)]
        else:
            de = [0.0] * 6

        # 力误差
        force_error = [self._desired_force[i] - self._actual_force[i] for i in range(6)]
        # 积分项（抗饱和）
        self._force_error_integral = [self._force_error_integral[i] + force_error[i] * dt for i in range(6)]
        self._force_error_integral = [self._clip(val, -10, 10) for val in self._force_error_integral]

        # 控制输出
        cmd = [0.0] * 6
        for i in range(6):
            if self._force_mask[i]:
                # 力控通道：PI控制
                cmd[i] = 0.5 * force_error[i] + 0.1 * self._force_error_integral[i]
            else:
                # 位置控通道：PD控制
                cmd[i] = 2.0 * e[i] + 0.5 * de[i]
        return cmd

    def _pure_force_control(self) -> List[float]:
        """纯力控制（无位置期望）"""
        force_error = [self._desired_force[i] - self._actual_force[i] for i in range(6)]
        # PI控制
        self._force_error_integral = [self._force_error_integral[i] + force_error[i] * self.dt for i in range(6)]
        self._force_error_integral = [self._clip(val, -10, 10) for val in self._force_error_integral]
        cmd = [0.5 * force_error[i] + 0.1 * self._force_error_integral[i] for i in range(6)]
        return cmd

    def _apply_safety_limits(self, cmd: List[float]) -> List[float]:
        """应用安全限制"""
        # 力限制
        cmd[0] = self._clip(cmd[0], -self.max_force, self.max_force)
        cmd[1] = self._clip(cmd[1], -self.max_force, self.max_force)
        cmd[2] = self._clip(cmd[2], -self.max_force, self.max_force)
        # 力矩限制
        cmd[3] = self._clip(cmd[3], -self.max_torque, self.max_torque)
        cmd[4] = self._clip(cmd[4], -self.max_torque, self.max_torque)
        cmd[5] = self._clip(cmd[5], -self.max_torque, self.max_torque)
        return cmd

    # ==================== 导纳控制（Admittance Control） ====================
    def admittance_control_step(self, measured_force: List[float], desired_pose: List[float],
                                dt: float, x_prev: List[float], v_prev: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        导纳控制：测量力 -> 修正期望位置
        返回: (new_desired_pose, new_velocity, new_acceleration)
        """
        # 力误差
        force_error = [self._desired_force[i] - measured_force[i] for i in range(6)]
        # 导纳模型：M * a + D * v + K * x = F
        # 求解加速度
        # D @ v_prev
        Dv_prev = mat_vec_mul(self.D, v_prev)
        # K @ (desired_pose - self._desired_pose)
        pose_diff = [desired_pose[i] - self._desired_pose[i] for i in range(6)]
        K_pose_diff = mat_vec_mul(self.K, pose_diff)
        # force_error - D @ v_prev - K @ (desired_pose - self._desired_pose)
        numerator = [force_error[i] - Dv_prev[i] - K_pose_diff[i] for i in range(6)]
        # M_inv @ numerator
        M_inv = invert_matrix(self.M)
        acc = mat_vec_mul(M_inv, numerator)
        # 积分得到速度和位置
        new_velocity = [v_prev[i] + acc[i] * dt for i in range(6)]
        new_pose = [desired_pose[i] + new_velocity[i] * dt for i in range(6)]
        return new_pose, new_velocity, acc

    # ==================== 辅助方法 ====================
    def reset_integral(self):
        """重置积分项"""
        self._force_error_integral = [0.0] * 6

    def get_actual_force(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """获取当前实际力/力矩"""
        f = (self._actual_force[0], self._actual_force[1], self._actual_force[2])
        t = (self._actual_force[3], self._actual_force[4], self._actual_force[5])
        return (f, t)

    def get_actual_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取当前实际位姿"""
        if self._actual_pose is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return tuple(self._actual_pose)


# ==================== 简单力控示例（无独立线程，用于快速测试） ====================
class SimpleForceController:
    """简化版力控制器（非实时，单步执行）"""

    def __init__(self, robot: RobotControllerBase):
        self.robot = robot

    def force_move(self, target_force: Tuple[float, float, float],
                   target_torque: Tuple[float, float, float],
                   max_duration: float = 5.0, gain: float = 0.01) -> bool:
        """
        简单力控移动：根据力误差调整末端位置
        适用于不需要高实时性的场景
        """
        start_time = time.time()
        while time.time() - start_time < max_duration:
            # 获取当前力
            f, t = self.robot.get_force_torque()
            error_f = (target_force[0] - f[0], target_force[1] - f[1], target_force[2] - f[2])
            error_t = (target_torque[0] - t[0], target_torque[1] - t[1], target_torque[2] - t[2])
            # 计算位移增量
            delta = (error_f[0] * gain, error_f[1] * gain, error_f[2] * gain,
                     error_t[0] * gain, error_t[1] * gain, error_t[2] * gain)
            # 获取当前位姿并移动
            pose = self.robot.get_tcp_pose()
            new_pose = tuple(pose[i] + delta[i] for i in range(6))
            self.robot.move_cartesian(new_pose, velocity=0.02)
            # 检查是否达到目标力
            if abs(error_f[0]) < 5.0 and abs(error_f[1]) < 5.0 and abs(error_f[2]) < 5.0:
                return True
            time.sleep(0.05)
        return False
