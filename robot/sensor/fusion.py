#!/usr/bin/env python3
"""
传感器融合模块 - 生产级实现
基于扩展卡尔曼滤波（EKF）融合 IMU、视觉、力觉数据
输出：6自由度位姿（x, y, z, roll, pitch, yaw）

注意：使用纯Python标准库实现，不依赖numpy/scipy
"""

from typing import Tuple, Optional, Dict, Any, List
import logging
import math

logger = logging.getLogger(__name__)


class SimpleMatrix:
    """简单矩阵类 - 纯Python实现，支持基本运算"""
    
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    @staticmethod
    def eye(n: int) -> 'SimpleMatrix':
        """单位矩阵"""
        return SimpleMatrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'SimpleMatrix':
        """零矩阵"""
        return SimpleMatrix([[0.0] * cols for _ in range(rows)])
    
    def __add__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        """矩阵加法"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")
        return SimpleMatrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def __sub__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        """矩阵减法"""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction")
        return SimpleMatrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def __matmul__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        """矩阵乘法"""
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply: {self.cols} != {other.rows}")
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return SimpleMatrix(result)
    
    def __mul__(self, scalar: float) -> 'SimpleMatrix':
        """标量乘法"""
        return SimpleMatrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def __rmul__(self, scalar: float) -> 'SimpleMatrix':
        return self.__mul__(scalar)
    
    def transpose(self) -> 'SimpleMatrix':
        """转置"""
        return SimpleMatrix([
            [self.data[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ])
    
    def inverse(self) -> 'SimpleMatrix':
        """矩阵求逆（高斯-约旦消元）"""
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")
        n = self.rows
        # 增广矩阵 [A|I]
        aug = [self.data[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        for i in range(n):
            # 找到主元
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            if abs(aug[i][i]) < 1e-10:
                raise ValueError("Matrix is singular")
            
            # 归一化
            pivot = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= pivot
            
            # 消元
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        
        # 提取逆矩阵部分
        inv_data = [row[n:] for row in aug]
        return SimpleMatrix(inv_data)
    
    def to_list(self) -> List[List[float]]:
        return self.data
    
    def get_row(self, i: int) -> List[float]:
        return self.data[i][:]
    
    def get_col(self, j: int) -> List[float]:
        return [self.data[i][j] for i in range(self.rows)]


class EKF:
    """扩展卡尔曼滤波器（6自由度位姿）- 纯Python实现"""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        # 状态: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.dim_x = 12
        self.dim_z = 6  # 观测维度（位置+姿态）
        
        self.x = [0.0] * self.dim_x  # 状态向量
        # 协方差矩阵 P (12x12)
        self.P = [[0.1 if i == j else 0.0 for j in range(self.dim_x)] for i in range(self.dim_x)]
        
        # 状态转移矩阵 F（恒定速度模型）
        self.F = [[1.0 if i == j else 0.0 for j in range(self.dim_x)] for i in range(self.dim_x)]
        for i in range(3):
            self.F[i][i + 6] = dt
        for i in range(3):
            self.F[3 + i][9 + i] = dt  # 角度速度
        
        # 观测矩阵 H（观测位置和姿态）(6x12)
        self.H = [[0.0] * self.dim_x for _ in range(self.dim_z)]
        for i in range(6):
            self.H[i][i] = 1.0
        
        # 过程噪声协方差 Q
        self.Q = [[0.01 if i == j else 0.0 for j in range(self.dim_x)] for i in range(self.dim_x)]
        # 观测噪声协方差 R（可动态调整）
        self.R = [[0.1 if i == j else 0.0 for j in range(self.dim_z)] for i in range(self.dim_z)]
    
    def _mat_vec_mul(self, mat: List[List[float]], vec: List[float]) -> List[float]:
        """矩阵向量乘法"""
        return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]
    
    def _mat_mul(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """矩阵乘法"""
        rows, cols, inner = len(a), len(b[0]), len(a[0])
        return [
            [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(rows)
        ]
    
    def _mat_transpose(self, mat: List[List[float]]) -> List[List[float]]:
        """矩阵转置"""
        return [list(row) for row in zip(*mat)]
    
    def _mat_add(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """矩阵加法"""
        return [
            [a[i][j] + b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))
        ]
    
    def _mat_sub(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """矩阵减法"""
        return [
            [a[i][j] - b[i][j] for j in range(len(a[0]))]
            for i in range(len(a))
        ]
    
    def _mat_scale(self, mat: List[List[float]], scalar: float) -> List[List[float]]:
        """矩阵标量乘法"""
        return [[mat[i][j] * scalar for j in range(len(mat[0]))] for i in range(len(mat))]
    
    def _mat_inverse(self, mat: List[List[float]]) -> List[List[float]]:
        """矩阵求逆"""
        n = len(mat)
        # 增广矩阵
        aug = [mat[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        for i in range(n):
            # 找主元
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            if abs(aug[i][i]) < 1e-10:
                raise ValueError("Matrix is singular")
            
            # 归一化
            pivot = aug[i][i]
            for j in range(2 * n):
                aug[i][j] /= pivot
            
            # 消元
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        
        return [row[n:] for row in aug]
    
    def predict(self):
        """预测步骤"""
        # x = F @ x
        self.x = self._mat_vec_mul(self.F, self.x)
        # P = F @ P @ F.T + Q
        FP = self._mat_mul(self.F, self.P)
        self.P = self._mat_add(self._mat_mul(FP, self._mat_transpose(self.F)), self.Q)
    
    def update(self, z: List[float], R_scale: float = 1.0):
        """更新步骤
        :param z: 观测向量 (x,y,z,roll,pitch,yaw)
        :param R_scale: 观测噪声缩放因子（可根据传感器置信度调整）
        """
        R = [[self.R[i][j] * R_scale for j in range(self.dim_z)] for i in range(self.dim_z)]
        # y = z - H @ x (残差)
        Hx = self._mat_vec_mul(self.H, self.x)
        y = [z[i] - Hx[i] for i in range(self.dim_z)]
        # S = H @ P @ H.T + R
        PHt = self._mat_mul(self.P, self._mat_transpose(self.H))
        S = self._mat_add(self._mat_mul(self.H, PHt), R)
        # K = P @ H.T @ inv(S)
        K = self._mat_mul(PHt, self._mat_inverse(S))
        # x = x + K @ y
        Ky = self._mat_vec_mul(K, y)
        self.x = [self.x[i] + Ky[i] for i in range(self.dim_x)]
        # P = (I - K @ H) @ P
        KH = self._mat_mul(K, self.H)
        I = [[1.0 if i == j else 0.0 for j in range(self.dim_x)] for i in range(self.dim_x)]
        IKH = [[I[i][j] - KH[i][j] for j in range(self.dim_x)] for i in range(self.dim_x)]
        self.P = self._mat_mul(IKH, self.P)
    
    def get_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取当前估计的位姿"""
        return (self.x[0], self.x[1], self.x[2],
                self.x[3], self.x[4], self.x[5])
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """获取当前线速度"""
        return (self.x[6], self.x[7], self.x[8])
    
    def set_state(self, pose: Tuple, velocity: Tuple = None):
        """手动设置初始状态"""
        self.x[:6] = list(pose)
        if velocity:
            self.x[6:9] = list(velocity)


class SensorFusion:
    """
    传感器融合器
    整合 IMU（加速度计/陀螺仪）、视觉定位、力觉数据
    """
    
    def __init__(self, dt: float = 0.01):
        self.ekf = EKF(dt)
        self.dt = dt
        self._last_imu_time = 0.0
        self._gyro_bias = [0.0, 0.0, 0.0]  # 陀螺仪零偏估计
    
    def _vec_add(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> List[float]:
        return [a[i] + b[i] for i in range(3)]
    
    def _vec_sub(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> List[float]:
        return [a[i] - b[i] for i in range(3)]
    
    def _vec_scale(self, v: Tuple[float, float, float], s: float) -> List[float]:
        return [v[i] * s for i in range(3)]
    
    def _vec_norm(self, v: Tuple[float, float, float]) -> float:
        return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    def predict_imu(self, accel: Tuple[float, float, float],
                    gyro: Tuple[float, float, float],
                    dt: float) -> List[float]:
        """
        使用 IMU 数据进行预测（运动学积分）
        返回预测后的状态（可直接用于 EKF 预测）
        """
        # 去除陀螺仪零偏
        gyro_corrected = self._vec_sub(gyro, tuple(self._gyro_bias))
        # 简单的运动学积分（实际可用四元数）
        # 此处仅更新角速度状态
        self.ekf.x[9:12] = list(gyro_corrected)
        # 加速度积分得到速度增量（简化）
        accel_list = self._vec_scale(accel, dt)
        self.ekf.x[6:9] = self._vec_add(tuple(self.ekf.x[6:9]), accel_list)
        # 位置增量
        vel_list = self._vec_scale(self.ekf.x[6:9], dt)
        self.ekf.x[0:3] = self._vec_add(tuple(self.ekf.x[0:3]), vel_list)
        # 角度增量
        gyro_dt = self._vec_scale(gyro_corrected, dt)
        self.ekf.x[3:6] = self._vec_add(tuple(self.ekf.x[3:6]), gyro_dt)
        return self.ekf.x.copy()
    
    def fuse_imu(self, accel: Tuple[float, float, float],
                 gyro: Tuple[float, float, float],
                 dt: float):
        """融合 IMU 数据（预测步骤）"""
        self.predict_imu(accel, gyro, dt)
        self.ekf.predict()  # EKF 预测（使用运动模型）
    
    def fuse_vision(self, pose: Tuple[float, float, float, float, float, float],
                    confidence: float = 0.9):
        """融合视觉定位数据（更新步骤）"""
        # 根据置信度调整观测噪声
        R_scale = 1.0 / max(0.1, confidence)
        self.ekf.update(list(pose), R_scale)
    
    def fuse_force(self, force: Tuple[float, float, float],
                   torque: Tuple[float, float, float]):
        """
        融合力/力矩数据（用于接触约束修正）
        简单实现：根据力的大小修正位置（假设力大说明有接触，停止运动）
        """
        force_magnitude = self._vec_norm(force)
        if force_magnitude > 50.0:  # 阈值
            # 减小速度，防止穿透
            self.ekf.x[6:9] = self._vec_scale(self.ekf.x[6:9], 0.5)
            logger.debug("Force feedback: reduced velocity")
    
    def get_pose(self) -> Tuple[float, float, float, float, float, float]:
        """获取融合后的位姿"""
        return self.ekf.get_pose()
    
    def get_velocity(self) -> Tuple[float, float, float]:
        return self.ekf.get_velocity()
    
    def reset(self, initial_pose: Tuple = (0, 0, 0, 0, 0, 0)):
        self.ekf.set_state(initial_pose)
        self._gyro_bias = [0.0, 0.0, 0.0]
    
    def calibrate_gyro_bias(self, gyro_samples: List[Tuple[float, float, float]]):
        """校准陀螺仪零偏（静止时采样）"""
        if len(gyro_samples) < 100:
            return
        # 计算平均值
        avg = [0.0, 0.0, 0.0]
        for sample in gyro_samples:
            for i in range(3):
                avg[i] += sample[i]
        avg = [avg[i] / len(gyro_samples) for i in range(3)]
        self._gyro_bias = avg
        logger.info(f"Gyro bias calibrated: {self._gyro_bias}")


# ==================== 使用示例 ====================
async def example_sensor_fusion():
    sf = SensorFusion(dt=0.01)
    # 模拟 IMU 数据
    for i in range(100):
        accel = (0, 0, 9.8)  # 静止时重力
        gyro = (0.01, 0.02, 0.01)
        sf.fuse_imu(accel, gyro, 0.01)
        if i % 50 == 0:
            # 模拟视觉观测
            vision_pose = (0.1, 0.05, 0.02, 0.0, 0.0, 0.0)
            sf.fuse_vision(vision_pose, confidence=0.8)
        await asyncio.sleep(0.01)
    print(f"Final pose: {sf.get_pose()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_sensor_fusion())
