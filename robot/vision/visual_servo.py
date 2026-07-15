#!/usr/bin/env python3
"""
视觉伺服模块
基于相机反馈的闭环控制（眼在手外/眼在手上）
使用纯Python标准库实现（无OpenCV依赖）
"""

import math
import struct
import hashlib
from typing import Tuple, Optional, Callable, List


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


class SimpleImage:
    """简化的图像表示（替代cv2/numpy数组）"""
    def __init__(self, width: int, height: int, channels: int = 3):
        self.width = width
        self.height = height
        self.channels = channels
        # 用一维列表存储像素数据
        self.pixels = [[0, 0, 0] for _ in range(width * height)]

    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        """获取像素值 (B, G, R)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            return tuple(self.pixels[idx])
        return (0, 0, 0)

    def set_pixel(self, x: int, y: int, b: int, g: int, r: int):
        """设置像素值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            self.pixels[idx] = [b, g, r]

    @staticmethod
    def from_array(data: List[List[List[int]]]) -> 'SimpleImage':
        """从3D列表创建图像"""
        height = len(data)
        width = len(data[0]) if height > 0 else 0
        channels = len(data[0][0]) if width > 0 else 3
        img = SimpleImage(width, height, channels)
        img.pixels = [list(pixel) for row in data for pixel in row]
        return img


class SimpleFeatureDetector:
    """简化的特征检测器（替代ORB/SIFT）"""
    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self.features = []

    def detect_and_compute(self, image: SimpleImage) -> Tuple[List, List]:
        """检测并计算特征"""
        # 简化的角点检测（使用简单的像素差异）
        features = []
        descriptors = []

        step = max(1, (image.width * image.height) // (self.max_features * 10))
        pixels = image.pixels

        for i in range(0, image.height - 2, step):
            for j in range(0, image.width - 2, step):
                if len(features) >= self.max_features:
                    break

                # 简单的Harris-like角点响应
                idx = i * image.width + j
                if idx + image.width * 2 + 2 >= len(pixels):
                    continue

                c = pixels[idx]
                r = pixels[idx + 2]
                b = pixels[idx + image.width * 2]

                # 计算梯度
                dx = (r[0] + r[1] + r[2]) - (c[0] + c[1] + c[2])
                dy = (b[0] + b[1] + b[2]) - (c[0] + c[1] + c[2])

                corner_score = dx * dx + dy * dy

                if corner_score > 100:  # 阈值
                    features.append({'x': j, 'y': i})
                    # 简化的描述符：周围9个像素的梯度信息
                    desc = [
                        dx % 256, dy % 256,
                        sum(c) % 256,  # 中心亮度
                    ]
                    # 添加周围像素信息
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < image.height and 0 <= nj < image.width:
                                nidx = ni * image.width + nj
                                desc.extend(pixels[nidx])
                            else:
                                desc.extend([0, 0, 0])
                    descriptors.append(desc)

        return features, descriptors


class SimpleBFMatcher:
    """简化的暴力匹配器（替代cv2.BFMatcher）"""
    def __init__(self, norm_type: str = "hamming"):
        self.norm_type = norm_type

    def knn_match(self, desc1: List, desc2: List, k: int = 2) -> List[List]:
        """KNN匹配"""
        if not desc1 or not desc2:
            return []

        matches = []
        for i, d1 in enumerate(desc1):
            distances = []
            for j, d2 in enumerate(desc2):
                dist = self._compute_distance(d1, d2)
                distances.append((dist, j))

            distances.sort(key=lambda x: x[0])
            # 返回k个最近邻
            k_neighbors = [(d, idx) for d, idx in distances[:k]]
            matches.append(k_neighbors)

        return matches

    def _compute_distance(self, d1: List, d2: List) -> float:
        """计算描述符距离"""
        if self.norm_type == "hamming":
            # 汉明距离
            return sum(b1 != b2 for b1, b2 in zip(d1, d2))
        else:
            # 欧氏距离
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(d1, d2)))


def find_homography(src_pts: List, dst_pts: List) -> Tuple:
    """简化的单应性矩阵计算（替代cv2.findHomography）"""
    # 简化的实现：返回仿射变换矩阵
    n = len(src_pts)
    if n < 4:
        return None, None

    # 计算简单的平移+缩放变换
    src_center = [sum(p[0] for p in src_pts) / n, sum(p[1] for p in src_pts) / n]
    dst_center = [sum(p[0] for p in dst_pts) / n, sum(p[1] for p in dst_pts) / n]

    scale = 1.0
    if n > 0:
        src_dist = math.sqrt(sum((p[0] - src_center[0])**2 + (p[1] - src_center[1])**2 for p in src_pts) / n)
        dst_dist = math.sqrt(sum((p[0] - dst_center[0])**2 + (p[1] - dst_center[1])**2 for p in dst_pts) / n)
        if src_dist > 0:
            scale = dst_dist / src_dist

    # 3x3单应性矩阵
    H = [
        [scale, 0, dst_center[0] - scale * src_center[0]],
        [0, scale, dst_center[1] - scale * src_center[1]],
        [0, 0, 1]
    ]
    mask = [[1] for _ in range(n)]

    return H, mask


class VisualServo:
    """视觉伺服控制器"""

    def __init__(self, robot, camera_matrix: List[List[float]],
                 camera_to_robot_transform: List[List[float]] = None):
        """
        robot: 机器人控制器
        camera_matrix: 相机内参矩阵 (3x3)
        camera_to_robot_transform: 相机坐标系到机器人基座的变换矩阵 (4x4)
        """
        self.robot = robot
        self.K = SimpleMatrix(camera_matrix) if isinstance(camera_matrix, list) else camera_matrix
        self.T_cam_to_robot = SimpleMatrix(camera_to_robot_transform) if camera_to_robot_transform else SimpleMatrix.eye(4)
        self._feature_detector = None
        self._target_features = None
        self._target_image = None

    def set_target_image(self, target_image: SimpleImage, feature_type: str = "orb"):
        """设置目标图像，提取特征"""
        self._target_image = target_image
        if feature_type == "orb":
            self._feature_detector = SimpleFeatureDetector(nfeatures=500)
        else:
            self._feature_detector = SimpleFeatureDetector(nfeatures=500)
        kp, des = self._feature_detector.detect_and_compute(target_image)
        self._target_features = (kp, des)
        return len(kp)

    def compute_pose_error(self, current_image: SimpleImage) -> Tuple[List[float], float]:
        """计算当前图像与目标图像的位姿误差（返回平移和旋转误差）"""
        if self._target_features is None:
            return [0.0, 0.0, 0.0], 0.0

        kp_cur, des_cur = self._feature_detector.detect_and_compute(current_image)

        if not des_cur or len(kp_cur) < 4:
            return [0.0, 0.0, 0.0], 1e6

        # 特征匹配
        bf = SimpleBFMatcher(norm_type="hamming")
        matches = bf.knn_match(self._target_features[1], des_cur, k=2)

        # 应用比率测试
        good = []
        for match_list in matches:
            if len(match_list) >= 2:
                m, n = match_list[0], match_list[1]
                if m[0] < 0.75 * n[0]:  # 距离比率测试
                    good.append({'query': m[1], 'train': m[1]})  # 简化表示

        if len(good) < 4:
            return [0.0, 0.0, 0.0], 1e6

        # 获取匹配点坐标
        src_pts = [[self._target_features[0][i]['x'], self._target_features[0][i]['y']] for i in range(len(good)) if i < len(self._target_features[0])]
        dst_pts = [[kp_cur[i]['x'], kp_cur[i]['y']] for i in range(len(good)) if i < len(kp_cur)]

        # 计算单应性矩阵
        H, mask = find_homography(src_pts, dst_pts)
        if H is None:
            return [0.0, 0.0, 0.0], 1e6

        # 提取平移误差
        t_error = [H[0][2], H[1][2], 0.0]

        # 简化的重投影误差
        reproj_error = 10.0  # 默认误差值

        return t_error, reproj_error

    def _clip_velocity(self, velocity: List[float], max_velocity: float) -> List[float]:
        """限制速度大小"""
        magnitude = math.sqrt(sum(v*v for v in velocity))
        if magnitude > max_velocity:
            scale = max_velocity / magnitude
            return [v * scale for v in velocity]
        return velocity

    def servo_step(self, current_image: SimpleImage, gain: float = 0.05, max_velocity: float = 0.1) -> bool:
        """单步视觉伺服控制"""
        t_error, err = self.compute_pose_error(current_image)

        if err < 5.0:  # 误差足够小
            return True

        # 计算速度指令（眼在手外情况）
        # 假设相机固定，机器人移动末端
        velocity = [-gain * t_error[0], -gain * t_error[1], -gain * t_error[2]]
        velocity = self._clip_velocity(velocity, max_velocity)

        # 获取当前位姿
        pose = self.robot.get_tcp_pose()
        new_pose = (
            pose[0] + velocity[0],
            pose[1] + velocity[1],
            pose[2] + velocity[2],
            pose[3], pose[4], pose[5]
        )
        self.robot.move_cartesian(new_pose, velocity=0.05)
        return False

    def run_servo_loop(self, camera_func: Callable, max_steps: int = 200):
        """运行完整视觉伺服循环"""
        for step in range(max_steps):
            img = camera_func()
            if img is None:
                break
            done = self.servo_step(img)
            if done:
                print(f"Visual servo converged in {step+1} steps")
                break
        return step + 1
