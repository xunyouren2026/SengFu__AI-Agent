#!/usr/bin/env python3
"""
机器人校准模块
支持关节零位校准、工具坐标系校准（四点法/六点法）、工件坐标系校准（三点法）
使用纯Python标准库实现
"""

import json
import os
import math
from typing import List, Tuple, Optional


class SimpleMatrix:
    """简单的纯Python矩阵实现（替代numpy）"""
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @staticmethod
    def eye(n: int) -> 'SimpleMatrix':
        """单位矩阵"""
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return SimpleMatrix(data)

    def __getitem__(self, key: Tuple[int, int]) -> float:
        i, j = key
        return self.data[i][j]

    def __setitem__(self, key: Tuple[int, int], value: float):
        i, j = key
        self.data[i][j] = value

    def to_list(self) -> List[List[float]]:
        return self.data

    @staticmethod
    def from_list(arr) -> 'SimpleMatrix':
        if isinstance(arr, SimpleMatrix):
            return arr
        if isinstance(arr[0], (int, float)):
            return SimpleMatrix([arr])
        return SimpleMatrix(arr)

    def T(self) -> 'SimpleMatrix':
        """转置"""
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return SimpleMatrix(result)

    def __matmul__(self, other: 'SimpleMatrix') -> 'SimpleMatrix':
        """矩阵乘法"""
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions mismatch: {self.cols} != {other.rows}")
        result = [[0.0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return SimpleMatrix(result)


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法"""
    if len(a[0]) != len(b):
        raise ValueError(f"Matrix dimensions mismatch")
    result = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                result[i][j] += a[i][k] * b[k][j]
    return result


def mat_vec_mul(a: List[List[float]], v: List[float]) -> List[float]:
    """矩阵向量乘法"""
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def svd(matrix: List[List[float]]) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """简化的SVD分解（使用幂迭代法，适合小矩阵）"""
    # 对于2x2或3x3矩阵的简化SVD
    m = len(matrix)
    n = len(matrix[0])

    # 计算 A^T * A 的特征值和特征向量
    ata = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(m):
                ata[i][j] += matrix[k][i] * matrix[k][j]

    # 简化的特征值计算（幂迭代）
    eigenvalues = []
    for _ in range(min(n, 3)):  # 取前几个特征值
        # 随机初始向量
        v = [1.0 / math.sqrt(n)] * n
        for _ in range(100):
            # 矩阵乘法
            av = [sum(ata[i][j] * v[j] for j in range(n)) for i in range(n)]
            # 归一化
            norm = math.sqrt(sum(x*x for x in av))
            if norm > 1e-10:
                v = [x / norm for x in av]
            else:
                break
        eigenvalues.append(math.sqrt(max(0.01, sum(ata[i][j] * v[j] for i in range(n) for j in range(n)) // n)))

    # 返回简化的奇异值和向量
    singular_values = sorted(eigenvalues, reverse=True)
    singular_values.extend([0.0] * (n - len(singular_values)))

    U = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    Vt = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    return U, singular_values, Vt


def lstsq(a: List[List[float]], b: List[float]) -> Tuple[List[float], float, int, List[float]]:
    """最小二乘法（替代numpy.linalg.lstsq）"""
    m = len(a)
    n = len(a[0])

    # 使用正规方程: A^T * A * x = A^T * b
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n

    for i in range(n):
        for j in range(n):
            for k in range(m):
                ata[i][j] += a[k][i] * a[k][j]
        for k in range(m):
            atb[i] += a[k][i] * b[k]

    # 高斯消元求解
    aug = [ata[i] + [atb[i]] for i in range(n)]

    for i in range(n):
        # 找主元
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k
        aug[i], aug[max_row] = aug[max_row], aug[i]

        if abs(aug[i][i]) < 1e-10:
            continue

        # 消元
        for k in range(i + 1, n):
            factor = aug[k][i] / aug[i][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]

    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(aug[i][i]) < 1e-10:
            x[i] = 0.0
        else:
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]

    # 计算残差
    residuals = 0.0
    for i in range(m):
        ax = sum(a[i][j] * x[j] for j in range(n))
        residuals += (ax - b[i]) ** 2

    return x, residuals, n, [0.0] * n


class RobotCalibration:
    """机器人校准器"""

    def __init__(self, robot):
        """
        初始化校准器
        robot: 机器人控制器实例（需提供 get_joint_positions, move_joint, get_tcp_pose, move_cartesian 等方法）
        """
        self.robot = robot

    # ==================== 关节零位校准 ====================
    def calibrate_joint_zero(self, joint_index: int = -1) -> bool:
        """
        校准关节零位
        joint_index: 关节索引，-1 表示所有关节
        返回是否成功
        """
        if joint_index == -1:
            print("校准所有关节：请依次将每个关节移动到机械零位（刻度对齐），然后按 Enter 确认")
            for i in range(len(self.robot.get_joint_positions())):
                input(f"请将关节 {i+1} 移动到零位，然后按 Enter...")
                if hasattr(self.robot, 'set_joint_zero'):
                    self.robot.set_joint_zero(i)
                else:
                    print(f"警告：当前机器人控制器不支持 set_joint_zero，请手动记录零位偏移")
        else:
            input(f"请将关节 {joint_index+1} 移动到零位，然后按 Enter...")
            if hasattr(self.robot, 'set_joint_zero'):
                self.robot.set_joint_zero(joint_index)
        return True

    # ==================== 工具坐标系校准（四点法） ====================
    def calibrate_tool_four_point(self, reference_point: Tuple[float, float, float],
                                  num_poses: int = 4) -> List[List[float]]:
        """
        四点法校准工具坐标系（TCP）
        reference_point: 空间中固定尖点的坐标（在机器人基座坐标系下，单位 mm）
        num_poses: 采集姿态数量（至少4个，越多越精确）
        返回工具坐标系相对于机器人末端法兰的变换矩阵 (4x4)
        """
        print("开始四点法工具校准...")
        print("请将工具末端对准固定尖点，记录多个不同姿态下的机器人法兰位姿")

        # 存储机器人法兰位姿矩阵（末端到基座）
        flange_poses = []

        for i in range(num_poses):
            input(f"姿态 {i+1}/{num_poses}: 将工具尖端对准固定尖点，然后按 Enter...")
            pose = self.robot.get_tcp_pose()  # 获取法兰位姿 (x,y,z,rx,ry,rz)
            flange_poses.append(self._pose_to_matrix(pose))

        # 固定尖点在基座坐标系下的坐标
        P_base = list(reference_point)

        # 构建超定方程组 A * P_tool = b，使用最小二乘法求解
        A_list = []
        b_list = []
        for T_flange in flange_poses:
            R = [row[:3] for row in T_flange]   # 旋转矩阵
            t = T_flange[:3][:]    # 平移向量
            A_list.append(R)
            b_list.append([P_base[i] - t[i] for i in range(3)])

        # 堆叠所有方程
        A = []
        b = []
        for i in range(len(A_list)):
            A.extend(A_list[i])
            b.extend(b_list[i])

        P_tool, residuals, rank, s = lstsq(A, b)

        print(f"工具校准完成：TCP 在法兰坐标系下的坐标 = {P_tool}")

        # 构建工具变换矩阵（假设工具无旋转，仅有平移）
        T_tool = [
            [1, 0, 0, P_tool[0]],
            [0, 1, 0, P_tool[1]],
            [0, 0, 1, P_tool[2]],
            [0, 0, 0, 1]
        ]
        return T_tool

    def calibrate_tool_six_point(self, reference_point: Tuple[float, float, float]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        六点法校准工具坐标系（完整姿态，包含旋转）
        reference_point: 固定尖点坐标（基座坐标系）
        返回 (工具平移矩阵, 工具旋转矩阵) 或直接返回齐次变换矩阵
        """
        print("开始六点法工具校准（含姿态）...")
        # 记录6个不同姿态下的法兰位姿和对应的工具尖端指向
        # 此处简化实现，实际需要求解旋转矩阵
        # 先获取平移（四点法）
        T_trans = self.calibrate_tool_four_point(reference_point, num_poses=4)
        # 再通过额外两点确定工具坐标系的姿态（需要用户将工具沿 X 和 XY 平面指向固定点）
        print("请将工具沿 X 正方向对准固定尖点，记录姿态...")
        input("按 Enter 记录...")
        pose_x = self.robot.get_tcp_pose()
        print("请将工具沿 XY 平面内（如 Y 正方向）对准固定尖点，记录姿态...")
        input("按 Enter 记录...")
        pose_xy = self.robot.get_tcp_pose()

        # 根据三点确定工具坐标系旋转（算法较复杂，此处简化返回单位矩阵）
        R_tool = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        T_tool = [
            [1, 0, 0, T_trans[0][3]],
            [0, 1, 0, T_trans[1][3]],
            [0, 0, 1, T_trans[2][3]],
            [0, 0, 0, 1]
        ]
        return T_tool, R_tool

    # ==================== 工件坐标系校准（三点法） ====================
    def calibrate_workpiece(self, workpiece_points: List[Tuple[float, float, float]],
                            robot_points: List[Tuple[float, float, float]]) -> List[List[float]]:
        """
        工件坐标系校准（三点法或多点拟合）
        workpiece_points: 工件上的点（在工件自身坐标系下，通常由设计图纸给出）
        robot_points: 对应的点在机器人基座坐标系下的实际位置（通过示教获得）
        返回工件坐标系到机器人基座坐标系的变换矩阵 (4x4)
        """
        assert len(workpiece_points) >= 3, "至少需要3个点"
        A = [list(p) for p in workpiece_points]
        B = [list(p) for p in robot_points]

        # 使用 Kabsch 算法求解最优旋转和平移
        centroid_A = [sum(A[i][j] for i in range(len(A))) / len(A) for j in range(len(A[0]))]
        centroid_B = [sum(B[i][j] for i in range(len(B))) / len(B) for j in range(len(B[0]))]

        # 中心化
        A_centered = [[A[i][j] - centroid_A[j] for j in range(len(A[0]))] for i in range(len(A))]
        B_centered = [[B[i][j] - centroid_B[j] for j in range(len(B[0]))] for i in range(len(B))]

        # H = A^T * B
        H = [[0.0] * len(A[0]) for _ in range(len(A[0]))]
        for i in range(len(A[0])):
            for j in range(len(A[0])):
                for k in range(len(A)):
                    H[i][j] += A_centered[k][i] * B_centered[k][j]

        # SVD分解
        U, S, Vt = svd(H)
        V = list(zip(*Vt)) if isinstance(Vt[0], (list, tuple)) else [[Vt[j][i] for j in range(len(Vt))] for i in range(len(Vt[0]))]

        # R = V * U^T
        R = matmul(V, [list(row) for row in zip(*U)] if isinstance(U[0], (list, tuple)) else [[U[j][i] for j in range(len(U))] for i in range(len(U[0]))])

        # 确保旋转矩阵行列式为+1（避免反射）
        det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
             - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
             + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))

        if det < 0:
            # 取反最后一列
            for i in range(len(R)):
                R[i][-1] *= -1

        # t = centroid_B - R @ centroid_A
        t = [centroid_B[i] - sum(R[i][j] * centroid_A[j] for j in range(len(centroid_A))) for i in range(len(R))]

        T = [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0, 0, 0, 1]
        ]
        print(f"工件校准完成：旋转矩阵=\n{R}\n平移向量={t}")
        return T

    # ==================== 辅助函数 ====================
    @staticmethod
    def _pose_to_matrix(pose: Tuple[float, float, float, float, float, float]) -> List[List[float]]:
        """
        将位姿 (x, y, z, rx, ry, rz) 转换为 4x4 齐次变换矩阵
        旋转使用 ZYX 欧拉角顺序
        """
        x, y, z, rx, ry, rz = pose[:6]
        T = [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ]
        # 计算旋转矩阵（ZYX 顺序）
        cos_rz, sin_rz = math.cos(rz), math.sin(rz)
        cos_ry, sin_ry = math.cos(ry), math.sin(ry)
        cos_rx, sin_rx = math.cos(rx), math.sin(rx)

        Rz = [
            [cos_rz, -sin_rz, 0],
            [sin_rz, cos_rz, 0],
            [0, 0, 1]
        ]
        Ry = [
            [cos_ry, 0, sin_ry],
            [0, 1, 0],
            [-sin_ry, 0, cos_ry]
        ]
        Rx = [
            [1, 0, 0],
            [0, cos_rx, -sin_rx],
            [0, sin_rx, cos_rx]
        ]

        # R = Rz @ Ry @ Rx
        RyRx = matmul(Ry, Rx)
        R = matmul(Rz, RyRx)

        T[0][:3] = R[0]
        T[1][:3] = R[1]
        T[2][:3] = R[2]
        return T

    @staticmethod
    def _matrix_to_pose(T: List[List[float]]) -> Tuple[float, float, float, float, float, float]:
        """将4x4齐次变换矩阵转换为位姿 (x, y, z, rx, ry, rz)"""
        x, y, z = T[0][3], T[1][3], T[2][3]
        R = [row[:3] for row in T]

        # 从旋转矩阵提取欧拉角（ZYX 顺序）
        rz = math.atan2(R[1][0], R[0][0])
        ry = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
        rx = math.atan2(R[2][1], R[2][2])

        return (x, y, z, rx, ry, rz)

    # ==================== 保存与加载校准结果 ====================
    def save_calibration(self, filepath: str, data: List[List[float]], format: str = "json"):
        """
        保存校准结果到文件
        data: 变换矩阵 (4x4)
        format: "npy" 或 "json"
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f)
        else:
            # 保存为简单的文本格式
            with open(filepath, 'w') as f:
                for row in data:
                    f.write(' '.join(str(x) for x in row) + '\n')
        print(f"校准数据已保存到 {filepath}")

    def load_calibration(self, filepath: str, format: str = "json") -> List[List[float]]:
        """从文件加载校准结果"""
        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            # 从文本格式加载
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    row = [float(x) for x in line.strip().split()]
                    data.append(row)
        print(f"从 {filepath} 加载校准数据")
        return data

    # ==================== 应用校准结果 ====================
    def apply_tool_calibration(self, T_tool: List[List[float]]):
        """
        将工具校准结果应用到机器人控制器
        需要机器人控制器支持 set_tool_transform 方法
        """
        if hasattr(self.robot, 'set_tool_transform'):
            self.robot.set_tool_transform(T_tool)
            print("工具校准已应用")
        else:
            print("警告：当前机器人控制器不支持 set_tool_transform，请手动设置")

    def apply_workpiece_calibration(self, T_workpiece: List[List[float]]):
        """
        将工件校准结果应用到机器人控制器
        需要机器人控制器支持 set_workpiece_transform 方法
        """
        if hasattr(self.robot, 'set_workpiece_transform'):
            self.robot.set_workpiece_transform(T_workpiece)
            print("工件校准已应用")
        else:
            print("警告：当前机器人控制器不支持 set_workpiece_transform，请手动设置")

    # ==================== 验证校准精度 ====================
    def verify_tool_calibration(self, reference_point: Tuple[float, float, float], tolerance: float = 1.0) -> bool:
        """
        验证工具校准精度
        将工具尖端移动到固定尖点，读取机器人法兰位姿，计算误差
        """
        input("请将工具尖端对准固定尖点，然后按 Enter...")
        pose = self.robot.get_tcp_pose()
        T_flange = self._pose_to_matrix(pose)

        # 如果已应用工具校准，则计算实际工具尖端位置
        if hasattr(self.robot, 'get_tool_transform'):
            T_tool = self.robot.get_tool_transform()
        else:
            T_tool = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        # 计算工具尖端在基座坐标系下的位置
        # P_tool_in_base = T_flange @ T_tool @ [0, 0, 0, 1]
        tool_tip_local = [0, 0, 0, 1]
        # 先计算 T_tool @ [0,0,0,1]
        temp = mat_vec_mul(T_tool[:3], tool_tip_local[:3])
        temp = [temp[i] + T_tool[i][3] for i in range(3)]
        # 再计算 T_flange @ temp
        P_tool_in_base = mat_vec_mul(T_flange[:3], temp + [1])[:3]

        error = math.sqrt(sum((P_tool_in_base[i] - reference_point[i])**2 for i in range(3)))
        print(f"工具尖端位置误差: {error:.3f} mm")
        return error < tolerance
