#!/usr/bin/env python3
"""
手眼标定模块
支持眼在手外（Eye-to-Hand）和眼在手上（Eye-in-Hand）两种模式
使用纯Python实现的标定算法
"""

import math
from typing import List, Tuple, Optional

from ..controller_base import RobotControllerBase


def mat_identity() -> List[List[float]]:
    """创建4x4单位矩阵"""
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def mat_mult(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """矩阵乘法 (4x4)"""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += A[i][k] * B[k][j]
    return result


def mat_vec_mult(M: List[List[float]], v: List[float]) -> List[float]:
    """矩阵与向量乘法 (4x4) @ (4,)"""
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3],
        1.0
    ]


def mat_inv(M: List[List[float]]) -> List[List[float]]:
    """求4x4矩阵的逆矩阵（高斯-约旦消元法）"""
    n = 4
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    
    for i in range(n):
        # 寻找主元
        pivot = i
        for j in range(i, n):
            if abs(augmented[j][i]) > abs(augmented[pivot][i]):
                pivot = j
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        
        if abs(augmented[i][i]) < 1e-10:
            continue
        
        # 归一化行
        div = augmented[i][i]
        for j in range(2*n):
            augmented[i][j] /= div
        
        # 消元
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                for k in range(2*n):
                    augmented[j][k] -= factor * augmented[i][k]
    
    return [row[n:] for row in augmented]


def rotvec_to_matrix(rx: float, ry: float, rz: float) -> List[List[float]]:
    """旋转向量转旋转矩阵（Rodrigues公式）"""
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta < 1e-10:
        return [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]
    
    kx, ky, kz = rx/theta, ry/theta, rz/theta
    c = math.cos(theta)
    s = math.sin(theta)
    t = 1.0 - c
    
    return [
        [t*kx*kx + c,     t*kx*ky - s*kz, t*kx*kz + s*ky, 0.0],
        [t*kx*ky + s*kz,  t*ky*ky + c,     t*ky*kz - s*kx, 0.0],
        [t*kx*kz - s*ky,  t*ky*kz + s*kx, t*kz*kz + c,     0.0],
        [0.0,             0.0,             0.0,             1.0]
    ]


def matrix_to_rotvec(R: List[List[float]]) -> Tuple[float, float, float]:
    """旋转矩阵转旋转向量"""
    cos_angle = (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.acos(cos_angle)
    
    if abs(angle) < 1e-6:
        return (0.0, 0.0, 0.0)
    
    # 从旋转矩阵提取轴角
    denom = 2.0 * math.sin(angle)
    rx = (R[2][1] - R[1][2]) / denom
    ry = (R[0][2] - R[2][0]) / denom
    rz = (R[1][0] - R[0][1]) / denom
    
    return (rx * angle, ry * angle, rz * angle)


class SimpleMatrix:
    """简单矩阵类，用于手眼标定"""
    
    @staticmethod
    def eye(size: int) -> List[List[float]]:
        """创建单位矩阵"""
        return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    
    @staticmethod
    def zeros(rows: int, cols: int) -> List[List[float]]:
        """创建零矩阵"""
        return [[0.0] * cols for _ in range(rows)]
    
    @staticmethod
    def from_list(data: List[List[float]]) -> List[List[float]]:
        """复制矩阵"""
        return [row[:] for row in data]


class HandEyeCalibration:
    """手眼标定器 - 纯Python实现"""

    def __init__(self, robot: RobotControllerBase, camera_matrix: List[List[float]], dist_coeffs: List[float],
                 mode: str = "eye_in_hand"):
        """
        robot: 机器人控制器
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs: 相机畸变系数 (1x5)
        mode: "eye_in_hand" (眼在手上) 或 "eye_to_hand" (眼在手外)
        """
        self.robot = robot
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.mode = mode
        self._calibrated = False
        self._robot_to_cam = mat_identity()   # 机器人基座到相机坐标系的变换
        self._cam_to_robot = mat_identity()    # 相机到机器人基座的变换

    def collect_data(self, num_poses: int = 15) -> Tuple[List[List[List[float]]], List[List[List[float]]]]:
        """
        采集标定数据（简化版）
        num_poses: 需要采集的位姿数量
        返回: (robot_poses_list, camera_poses_list)
        
        注意：实际使用时需要集成相机采集和标定板检测
        """
        robot_poses = []
        camera_poses = []

        for i in range(num_poses):
            print(f"Move robot to pose {i+1}/{num_poses}, then press Enter...")
            input()
            # 获取机器人当前位姿
            pose = self.robot.get_tcp_pose()
            robot_poses.append(self._pose_to_matrix(pose))

            # 获取相机图像并检测标定板（需要实际相机驱动）
            # 这里使用模拟数据
            cam_pose = self._capture_camera_pose()
            if cam_pose is not None:
                camera_poses.append(cam_pose)
            else:
                print("Camera pose not detected, retry")
                i -= 1

        return robot_poses, camera_poses

    def calibrate(self, robot_poses: List[List[List[float]]], camera_poses: List[List[List[float]]]) -> bool:
        """
        执行手眼标定 - 使用TSAI方法简化实现
        """
        if len(robot_poses) < 3 or len(camera_poses) < 3:
            print("Need at least 3 pose pairs for calibration")
            return False
        
        if self.mode == "eye_in_hand":
            # 眼在手上：求解机器人末端到相机的变换
            # 简化实现：使用平均变换
            R_cam2gripper_list = []
            t_cam2gripper_list = []
            
            for i in range(len(robot_poses)):
                gripper_to_base = robot_poses[i]
                target_to_cam = camera_poses[i]
                
                # 计算变换
                base_to_gripper = mat_inv(gripper_to_base)
                cam_to_target = mat_inv(target_to_cam)
                
                # 简化：取平移向量的平均值作为标定结果
                R_cam2gripper_list.append(target_to_cam[:3][:3])
                t_cam2gripper_list.append([
                    (gripper_to_base[0][3] + target_to_cam[0][3]) / 2,
                    (gripper_to_base[1][3] + target_to_cam[1][3]) / 2,
                    (gripper_to_base[2][3] + target_to_cam[2][3]) / 2
                ])
            
            # 平均化结果
            avg_R = SimpleMatrix.eye(3)
            avg_t = [0.0, 0.0, 0.0]
            for i in range(3):
                for j in range(3):
                    sum_val = sum(R[j][i] for R in R_cam2gripper_list)
                    avg_R[j][i] = sum_val / len(R_cam2gripper_list)
                avg_t[i] = sum(t[i] for t in t_cam2gripper_list) / len(t_cam2gripper_list)
            
            for i in range(3):
                for j in range(3):
                    self._robot_to_cam[i][j] = avg_R[i][j]
                self._robot_to_cam[i][3] = avg_t[i]
        else:
            # 眼在手外：求解机器人基座到相机的变换
            avg_R = SimpleMatrix.eye(3)
            avg_t = [0.0, 0.0, 0.0]
            
            for cam_pose in camera_poses:
                for i in range(3):
                    for j in range(3):
                        avg_R[i][j] += cam_pose[i][j] / len(camera_poses)
                    avg_t[i] += cam_pose[i][3] / len(camera_poses)
            
            for i in range(3):
                for j in range(3):
                    self._cam_to_robot[i][j] = avg_R[i][j]
                self._cam_to_robot[i][3] = avg_t[i]
            
            self._robot_to_cam = mat_inv(self._cam_to_robot)

        self._calibrated = True
        print("Calibration completed successfully")
        return True

    def transform_point(self, point_in_camera: List[float]) -> List[float]:
        """将相机坐标系下的点转换到机器人基座坐标系"""
        if not self._calibrated:
            raise RuntimeError("Calibration not performed")
        p_hom = point_in_camera + [1.0]
        if self.mode == "eye_in_hand":
            # 点从相机坐标系 -> 机器人末端 -> 机器人基座
            gripper_to_base = self._pose_to_matrix(self.robot.get_tcp_pose())
            p_in_base = mat_vec_mult(mat_mult(gripper_to_base, self._robot_to_cam), p_hom)
        else:
            p_in_base = mat_vec_mult(self._cam_to_robot, p_hom)
        return p_in_base[:3]

    def get_transform(self) -> List[List[float]]:
        """获取手眼变换矩阵 (4x4)"""
        return [row[:] for row in self._robot_to_cam]

    @staticmethod
    def _pose_to_matrix(pose: Tuple[float, ...]) -> List[List[float]]:
        x, y, z, rx, ry, rz = pose[:6]
        T = mat_identity()
        T[0][3] = x
        T[1][3] = y
        T[2][3] = z
        # 欧拉角转旋转矩阵
        R = rotvec_to_matrix(rx, ry, rz)
        for i in range(3):
            for j in range(3):
                T[i][j] = R[i][j]
        return T

    @staticmethod
    def _rvec_tvec_to_matrix(rvec: Tuple[float, float, float], tvec: Tuple[float, float, float]) -> List[List[float]]:
        R = rotvec_to_matrix(*rvec)
        T = mat_identity()
        for i in range(3):
            for j in range(3):
                T[i][j] = R[i][j]
            T[i][3] = tvec[i]
        return T

    def _capture_camera_pose(self) -> Optional[List[List[float]]]:
        """
        模拟图像采集（实际应使用真实相机）
        返回相机检测到的标定板位姿
        """
        # 这里返回模拟数据，实际需要集成相机驱动和标定板检测
        # 返回一个随机位姿作为模拟
        import random
        # 模拟检测成功
        if random.random() > 0.1:
            rx = random.uniform(-0.1, 0.1)
            ry = random.uniform(-0.1, 0.1)
            rz = random.uniform(-0.1, 0.1)
            tx = random.uniform(-0.5, 0.5)
            ty = random.uniform(-0.5, 0.5)
            tz = random.uniform(0.3, 1.0)
            return HandEyeCalibration._rvec_tvec_to_matrix(
                (rx, ry, rz),
                (tx, ty, tz)
            )
        return None
