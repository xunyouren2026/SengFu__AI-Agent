"""
后处理模块

实现帧插值、超分辨率、风格化、相机运动、时域平滑、水印、嘴唇同步等功能
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
import math
import random


@dataclass
class InterpolationConfig:
    """帧插值配置"""
    method: str = "rife"  # rife/linear/optical_flow
    num_frames: int = 1
    blend_mode: str = "linear"


@dataclass
class SuperResolutionConfig:
    """超分辨率配置"""
    model_type: str = "esrgan"  # esrgan/anime/realcugan
    scale: int = 4
    tile_size: int = 512


class FrameInterpolator:
    """
    帧插值器
    
    支持多种插值方法：
    - rife: 基于光流的插值（模拟）
    - linear: 线性混合
    - optical_flow: 光流扭曲
    """
    
    def __init__(self, method: str = "rife"):
        """
        初始化帧插值器
        
        Args:
            method: 插值方法（rife/linear/optical_flow）
        """
        self._method = method
    
    def interpolate(self, frame_a: List[List[List[int]]], 
                    frame_b: List[List[List[int]]], 
                    num_frames: int = 1) -> List[List[List[List[int]]]]:
        """
        在两帧之间插值生成中间帧
        
        Args:
            frame_a: 起始帧 [H, W, C]
            frame_b: 结束帧 [H, W, C]
            num_frames: 要生成的中间帧数量
            
        Returns:
            插值帧列表（包含起始帧、中间帧和结束帧）
        """
        if not frame_a or not frame_b:
            return [frame_a, frame_b] if frame_a or frame_b else []
        
        if num_frames <= 0:
            return [frame_a, frame_b]
        
        # 根据方法选择插值策略
        if self._method == "linear":
            return self._linear_interpolate(frame_a, frame_b, num_frames)
        elif self._method == "optical_flow":
            return self._optical_flow_interpolate(frame_a, frame_b, num_frames)
        else:  # rife
            return self._rife_interpolate(frame_a, frame_b, num_frames)
    
    def _linear_interpolate(self, frame_a: List[List[List[int]]], 
                            frame_b: List[List[List[int]]], 
                            num_frames: int) -> List[List[List[List[int]]]]:
        """
        线性插值
        
        Args:
            frame_a: 起始帧
            frame_b: 结束帧
            num_frames: 中间帧数量
            
        Returns:
            插值帧列表
        """
        result = [frame_a]
        
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            interpolated = self._linear_blend(frame_a, frame_b, alpha)
            result.append(interpolated)
        
        result.append(frame_b)
        
        return result
    
    def _linear_blend(self, a: List[List[List[int]]], 
                      b: List[List[List[int]]], 
                      alpha: float) -> List[List[List[int]]]:
        """
        线性混合两帧
        
        Args:
            a: 帧A
            b: 帧B
            alpha: 混合系数（0-1）
            
        Returns:
            混合后的帧
        """
        result = []
        
        for i in range(len(a)):
            row = []
            for j in range(len(a[i])):
                pixel = []
                for c in range(len(a[i][j])):
                    val = (1 - alpha) * a[i][j][c] + alpha * b[i][j][c]
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _optical_flow_interpolate(self, frame_a: List[List[List[int]]], 
                                  frame_b: List[List[List[int]]], 
                                  num_frames: int) -> List[List[List[List[int]]]]:
        """
        光流插值
        
        Args:
            frame_a: 起始帧
            frame_b: 结束帧
            num_frames: 中间帧数量
            
        Returns:
            插值帧列表
        """
        # 估计光流
        flow = self._estimate_flow(frame_a, frame_b)
        
        result = [frame_a]
        
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            
            # 正向扭曲
            warped_a = self._optical_flow_warp(frame_a, flow, alpha)
            # 反向扭曲
            warped_b = self._optical_flow_warp(frame_b, flow, alpha - 1)
            
            # 混合
            interpolated = self._linear_blend(warped_a, warped_b, 0.5)
            result.append(interpolated)
        
        result.append(frame_b)
        
        return result
    
    def _rife_interpolate(self, frame_a: List[List[List[int]]], 
                          frame_b: List[List[List[int]]], 
                          num_frames: int) -> List[List[List[List[int]]]]:
        """
        RIFE风格插值（简化模拟）
        RIFE (Real-Time Intermediate Flow Estimation) 是一种基于光流的实时插值方法
        
        Args:
            frame_a: 起始帧
            frame_b: 结束帧
            num_frames: 中间帧数量
            
        Returns:
            插值帧列表
        """
        # 估计双向光流
        flow_forward = self._estimate_flow(frame_a, frame_b)
        flow_backward = self._estimate_flow(frame_b, frame_a)
        
        result = [frame_a]
        
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            
            # 双向扭曲并混合
            warped_a = self._optical_flow_warp(frame_a, flow_forward, t)
            warped_b = self._optical_flow_warp(frame_b, flow_backward, 1 - t)
            
            # 使用软最小化混合权重
            blend_weight = self._soft_blend_weight(warped_a, warped_b, t)
            interpolated = self._weighted_blend(warped_a, warped_b, blend_weight)
            
            result.append(interpolated)
        
        result.append(frame_b)
        
        return result
    
    def _estimate_flow(self, frame_a: List[List[List[int]]], 
                       frame_b: List[List[List[int]]]) -> List[List[List[float]]]:
        """
        Lucas-Kanade光流算法
        基于三个假设：亮度恒定、时间连续（小运动）、空间一致性（邻域相同运动）
        
        算法步骤：
        1. 计算空间梯度 Ix, Iy 和时间梯度 It
        2. 在每个像素的局部窗口内建立超定方程组
        3. 通过最小二乘法求解光流 (u, v)
        4. 迭代优化以提高精度
        
        Args:
            frame_a: 起始帧 [H, W, C]
            frame_b: 结束帧 [H, W, C]
            
        Returns:
            光流场 [H, W, 2]，每个像素包含(u, v)位移
        """
        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        
        if h < 3 or w < 3:
            # 图像太小，返回零光流
            return [[[0.0, 0.0] for _ in range(w)] for _ in range(h)]
        
        # Lucas-Kanade参数
        window_size = 5  # 局部窗口大小（奇数）
        half_w = window_size // 2
        num_iterations = 3  # 迭代次数
        epsilon = 1e-4  # 收敛阈值
        
        # ---- 第一步：计算图像梯度 ----
        # 将帧转为灰度（简化：取RGB均值）
        gray_a = [[0.0] * w for _ in range(h)]
        gray_b = [[0.0] * w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if len(frame_a[i][j]) >= 3:
                    gray_a[i][j] = (frame_a[i][j][0] + frame_a[i][j][1] + frame_a[i][j][2]) / 3.0
                if len(frame_b[i][j]) >= 3:
                    gray_b[i][j] = (frame_b[i][j][0] + frame_b[i][j][1] + frame_b[i][j][2]) / 3.0
        
        # 空间梯度 Ix = dI/dx, Iy = dI/dy（使用中心差分）
        Ix = [[0.0] * w for _ in range(h)]
        Iy = [[0.0] * w for _ in range(h)]
        # 时间梯度 It = dI/dt（前向差分）
        It = [[0.0] * w for _ in range(h)]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # 空间梯度（在两帧的平均灰度上计算）
                avg_gray = (gray_a[i][j] + gray_b[i][j]) / 2.0
                avg_left = (gray_a[i][j-1] + gray_b[i][j-1]) / 2.0
                avg_right = (gray_a[i][j+1] + gray_b[i][j+1]) / 2.0
                avg_up = (gray_a[i-1][j] + gray_b[i-1][j]) / 2.0
                avg_down = (gray_a[i+1][j] + gray_b[i+1][j]) / 2.0
                
                Ix[i][j] = (avg_right - avg_left) / 2.0
                Iy[i][j] = (avg_down - avg_up) / 2.0
                It[i][j] = gray_b[i][j] - gray_a[i][j]
        
        # ---- 第二步：在每个像素的窗口内求解光流 ----
        # 初始化光流为零
        flow_u = [[0.0] * w for _ in range(h)]
        flow_v = [[0.0] * w for _ in range(h)]
        
        for iteration in range(num_iterations):
            new_u = [[0.0] * w for _ in range(h)]
            new_v = [[0.0] * w for _ in range(h)]
            
            for i in range(half_w, h - half_w):
                for j in range(half_w, w - half_w):
                    # 在局部窗口内累加梯度乘积
                    sum_Ix2 = 0.0  # ΣIx²
                    sum_Iy2 = 0.0  # ΣIy²
                    sum_IxIy = 0.0  # ΣIx*Iy
                    sum_IxIt = 0.0  # ΣIx*It
                    sum_IyIt = 0.0  # ΣIy*It
                    
                    for di in range(-half_w, half_w + 1):
                        for dj in range(-half_w, half_w + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                ix = Ix[ni][nj]
                                iy = Iy[ni][nj]
                                it = It[ni][nj]
                                
                                sum_Ix2 += ix * ix
                                sum_Iy2 += iy * iy
                                sum_IxIy += ix * iy
                                sum_IxIt += ix * it
                                sum_IyIt += iy * it
                    
                    # ---- 第三步：求解2x2线性方程组 ----
                    # [ΣIx²  ΣIxIy] [u]   [-ΣIxIt]
                    # [ΣIxIy ΣIy² ] [v] = [-ΣIyIt]
                    det = sum_Ix2 * sum_Iy2 - sum_IxIy * sum_IxIy
                    
                    if abs(det) > epsilon:
                        # 矩阵可逆，直接求解
                        u = (sum_Iy2 * (-sum_IxIt) - sum_IxIy * (-sum_IyIt)) / det
                        v = (sum_Ix2 * (-sum_IyIt) - sum_IxIy * (-sum_IxIt)) / det
                    else:
                        # 矩阵奇异，使用伪逆近似
                        # 沿梯度最大的方向求解
                        if sum_Ix2 > sum_Iy2:
                            u = -sum_IxIt / (sum_Ix2 + epsilon)
                            v = 0.0
                        else:
                            u = 0.0
                            v = -sum_IyIt / (sum_Iy2 + epsilon)
                    
                    # ---- 第四步：迭代更新 ----
                    # 将当前迭代的增量累加到已有光流上
                    new_u[i][j] = flow_u[i][j] + u
                    new_v[i][j] = flow_v[i][j] + v
            
            # 更新光流
            flow_u = new_u
            flow_v = new_v
        
        # 组装光流场
        flow = []
        for i in range(h):
            row = []
            for j in range(w):
                row.append([flow_u[i][j], flow_v[i][j]])
            flow.append(row)
        
        return flow
    
    def _optical_flow_warp(self, frame: List[List[List[int]]], 
                           flow: List[List[List[float]]], 
                           scale: float) -> List[List[List[int]]]:
        """
        光流扭曲
        
        Args:
            frame: 输入帧
            flow: 光流场
            scale: 流缩放因子
            
        Returns:
            扭曲后的帧
        """
        h = len(frame)
        w = len(frame[0]) if h > 0 else 0
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                # 获取流向量
                if i < len(flow) and j < len(flow[i]):
                    fx, fy = flow[i][j]
                else:
                    fx, fy = 0.0, 0.0
                
                # 计算源位置
                src_x = int(j + fx * scale)
                src_y = int(i + fy * scale)
                
                # 边界检查
                if 0 <= src_x < w and 0 <= src_y < h:
                    pixel = frame[src_y][src_x][:]
                else:
                    pixel = [0] * len(frame[0][0])
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _soft_blend_weight(self, frame_a: List[List[List[int]]], 
                           frame_b: List[List[List[int]]], 
                           t: float) -> List[List[float]]:
        """
        计算软混合权重
        
        Args:
            frame_a: 帧A
            frame_b: 帧B
            t: 时间参数
            
        Returns:
            权重矩阵
        """
        h = len(frame_a)
        w = len(frame_a[0]) if h > 0 else 0
        
        weights = []
        for i in range(h):
            row = []
            for j in range(w):
                # 基于像素差异调整权重
                diff = sum(abs(frame_a[i][j][c] - frame_b[i][j][c]) 
                          for c in range(min(len(frame_a[i][j]), len(frame_b[i][j]))))
                
                # 差异大的区域使用更平滑的过渡
                weight = t + 0.1 * math.tanh(diff / 100.0 - 0.5)
                weight = max(0.0, min(1.0, weight))
                
                row.append(weight)
            weights.append(row)
        
        return weights
    
    def _weighted_blend(self, frame_a: List[List[List[int]]], 
                        frame_b: List[List[List[int]]], 
                        weights: List[List[float]]) -> List[List[List[int]]]:
        """
        加权混合
        
        Args:
            frame_a: 帧A
            frame_b: 帧B
            weights: 权重矩阵
            
        Returns:
            混合后的帧
        """
        result = []
        
        for i in range(len(frame_a)):
            row = []
            for j in range(len(frame_a[i])):
                weight = weights[i][j] if i < len(weights) and j < len(weights[i]) else 0.5
                pixel = []
                for c in range(len(frame_a[i][j])):
                    val = (1 - weight) * frame_a[i][j][c] + weight * frame_b[i][j][c]
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result


class SuperResolver:
    """
    超分辨率器
    
    支持多种超分模型：
    - esrgan: ESRGAN通用模型
    - anime: 动漫专用模型
    - realcugan: Real-CUGAN模型
    """
    
    def __init__(self, model_type: str = "esrgan", scale: int = 4):
        """
        初始化超分辨率器
        
        Args:
            model_type: 模型类型
            scale: 放大倍数
        """
        self._model_type = model_type
        self._scale = scale
    
    def upscale(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        超分单帧
        
        Args:
            frame: 输入帧 [H, W, C]
            
        Returns:
            超分后的帧 [H*scale, W*scale, C]
        """
        if not frame or not frame[0]:
            return frame
        
        h, w = len(frame), len(frame[0])
        new_h, new_w = h * self._scale, w * self._scale
        
        # 简化实现：使用双三次插值
        result = []
        
        for i in range(new_h):
            row = []
            for j in range(new_w):
                # 映射回原图坐标
                src_i = i / self._scale
                src_j = j / self._scale
                
                # 双三次插值
                pixel = self._bicubic_interpolate(frame, src_i, src_j)
                row.append(pixel)
            result.append(row)
        
        # 应用残差增强（模拟ESRGAN的效果）
        if self._model_type == "esrgan":
            result = self._enhance_details(result)
        elif self._model_type == "anime":
            result = self._enhance_anime(result)
        
        return result
    
    def upscale_video(self, frames: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """
        超分视频
        
        Args:
            frames: 视频帧序列
            
        Returns:
            超分后的视频帧序列
        """
        return [self.upscale(frame) for frame in frames]
    
    def _bicubic_interpolate(self, frame: List[List[List[int]]], 
                             y: float, x: float) -> List[int]:
        """
        双三次插值
        
        Args:
            frame: 输入帧
            y: y坐标（浮点）
            x: x坐标（浮点）
            
        Returns:
            插值像素
        """
        h, w = len(frame), len(frame[0])
        channels = len(frame[0][0])
        
        # 获取整数部分和小数部分
        x0, y0 = int(x), int(y)
        dx, dy = x - x0, y - y0
        
        pixel = []
        
        for c in range(channels):
            # 4x4邻域加权
            val = 0.0
            weight_sum = 0.0
            
            for di in range(-1, 3):
                for dj in range(-1, 3):
                    ni, nj = y0 + di, x0 + dj
                    
                    if 0 <= ni < h and 0 <= nj < w:
                        # 三次核函数
                        wx = self._cubic_kernel(dj - dx)
                        wy = self._cubic_kernel(di - dy)
                        weight = wx * wy
                        
                        val += weight * frame[ni][nj][c]
                        weight_sum += weight
            
            if weight_sum > 0:
                val /= weight_sum
            
            pixel.append(int(max(0, min(255, val))))
        
        return pixel
    
    def _cubic_kernel(self, x: float) -> float:
        """
        三次核函数
        
        Args:
            x: 输入值
            
        Returns:
            核函数值
        """
        abs_x = abs(x)
        
        if abs_x <= 1:
            return 1 - 2 * abs_x ** 2 + abs_x ** 3
        elif abs_x < 2:
            return 4 - 8 * abs_x + 5 * abs_x ** 2 - abs_x ** 3
        else:
            return 0.0
    
    def _enhance_details(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        细节增强（模拟ESRGAN）
        
        Args:
            frame: 输入帧
            
        Returns:
            增强后的帧
        """
        # 应用残差密集块的效果（简化）
        result = []
        
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                pixel = []
                for c in range(len(frame[i][j])):
                    # 边缘增强
                    if i > 0 and i < len(frame) - 1 and j > 0 and j < len(frame[0]) - 1:
                        # Laplacian锐化
                        laplacian = (
                            frame[i-1][j][c] + frame[i+1][j][c] +
                            frame[i][j-1][c] + frame[i][j+1][c] -
                            4 * frame[i][j][c]
                        )
                        val = frame[i][j][c] - 0.1 * laplacian
                    else:
                        val = frame[i][j][c]
                    
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _enhance_anime(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        动漫风格增强
        
        Args:
            frame: 输入帧
            
        Returns:
            增强后的帧
        """
        # 动漫风格：边缘增强 + 颜色量化
        result = []
        
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                pixel = []
                for c in range(len(frame[i][j])):
                    # 边缘检测
                    if i > 0 and i < len(frame) - 1 and j > 0 and j < len(frame[0]) - 1:
                        gx = (frame[i][j+1][c] - frame[i][j-1][c]) / 2
                        gy = (frame[i+1][j][c] - frame[i-1][j][c]) / 2
                        edge = math.sqrt(gx**2 + gy**2)
                        
                        # 边缘增强
                        val = frame[i][j][c] + edge * 0.3
                    else:
                        val = frame[i][j][c]
                    
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _residual_dense_block(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        残差密集块（RRDB的核心组件）
        包含3层卷积+ReLU激活，每层输出与输入拼接后通过1x1卷积压缩，
        最终通过残差连接与原始输入相加
        
        Args:
            x: 输入特征图 [H, W, C]
            
        Returns:
            输出特征图 [H, W, C]
        """
        if not x or not x[0]:
            return x
        
        h = len(x)
        w = len(x[0])
        c = len(x[0][0]) if x[0] else 0
        
        if c == 0:
            return x
        
        # 保存原始输入用于残差连接
        residual = x
        
        # ---- 密集块：3层3x3卷积 + ReLU + 拼接 ----
        # 每层卷积的输出通道数（简化为与输入相同）
        conv_out_channels = c
        # 拼接后的总通道数 = 原始通道 + 3层卷积输出
        total_channels = c + conv_out_channels * 3
        
        # 逐层处理
        features = [x]  # 初始特征列表
        
        for layer_idx in range(3):
            # 获取上一层的输出
            prev = features[-1]
            prev_c = len(prev[0][0]) if prev and prev[0] else 0
            
            # 3x3卷积 + ReLU
            conv_out = self._conv3x3_relu(prev, prev_c, conv_out_channels)
            features.append(conv_out)
        
        # ---- 1x1卷积压缩：将拼接后的特征压缩回原始通道数 ----
        # 拼接所有层输出
        concatenated = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                for feat in features:
                    if i < len(feat) and j < len(feat[i]):
                        pixel.extend(feat[i][j])
                row.append(pixel)
            concatenated.append(row)
        
        # 1x1卷积压缩（线性投影）
        concat_c = len(concatenated[0][0]) if concatenated and concatenated[0] else 0
        compressed = self._conv1x1(concatenated, concat_c, c)
        
        # ---- 残差连接：output = x + 0.2 * compressed ----
        # 0.2为残差缩放因子，防止残差过大导致训练不稳定
        scale = 0.2
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                for ch in range(c):
                    orig_val = residual[i][j][ch] if ch < len(residual[i][j]) else 0.0
                    comp_val = compressed[i][j][ch] if i < len(compressed) and j < len(compressed[i]) and ch < len(compressed[i][j]) else 0.0
                    pixel.append(orig_val + scale * comp_val)
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _conv3x3_relu(self, x: List[List[List[float]]], in_c: int, out_c: int) -> List[List[List[float]]]:
        """
        3x3卷积 + ReLU激活
        使用简化的均匀随机权重进行卷积运算
        
        Args:
            x: 输入特征图 [H, W, in_c]
            in_c: 输入通道数
            out_c: 输出通道数
            
        Returns:
            卷积后的特征图 [H, W, out_c]
        """
        import random
        rng = random.Random(789)
        h = len(x)
        w = len(x[0]) if h > 0 else 0
        
        # Xavier初始化3x3卷积核权重 [out_c, in_c, 3, 3]
        limit = math.sqrt(6.0 / (in_c * 9 + out_c * 9))
        kernel = [[[[rng.uniform(-limit, limit) for _ in range(3)] for _ in range(3)] for _ in range(in_c)] for _ in range(out_c)]
        bias = [0.0] * out_c
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                for oc in range(out_c):
                    val = bias[oc]
                    # 3x3卷积窗口
                    for ic in range(min(in_c, len(x[i][j]))):
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w and ic < len(x[ni][nj]):
                                    val += x[ni][nj][ic] * kernel[oc][ic][di + 1][dj + 1]
                    # ReLU激活
                    pixel.append(max(0.0, val))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _conv1x1(self, x: List[List[List[float]]], in_c: int, out_c: int) -> List[List[List[float]]]:
        """
        1x1卷积（逐点卷积）
        等价于对每个像素位置进行线性投影
        
        Args:
            x: 输入特征图 [H, W, in_c]
            in_c: 输入通道数
            out_c: 输出通道数
            
        Returns:
            卷积后的特征图 [H, W, out_c]
        """
        import random
        rng = random.Random(1011)
        h = len(x)
        w = len(x[0]) if h > 0 else 0
        
        # Xavier初始化1x1卷积核权重 [out_c, in_c]
        limit = math.sqrt(6.0 / (in_c + out_c))
        weights = [[rng.uniform(-limit, limit) for _ in range(in_c)] for _ in range(out_c)]
        bias = [0.0] * out_c
        
        result = []
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                for oc in range(out_c):
                    val = bias[oc]
                    for ic in range(min(in_c, len(x[i][j]))):
                        val += x[i][j][ic] * weights[oc][ic]
                    pixel.append(val)
                row.append(pixel)
            result.append(row)
        
        return result


class Stylizer:
    """
    风格化器
    
    支持多种艺术风格：
    - cartoon: 卡通风格
    - oil_painting: 油画风格
    - sketch: 素描风格
    - vintage: 怀旧风格
    - pixel: 像素化风格
    - watercolor: 水彩风格
    """
    
    def __init__(self):
        """初始化风格化器"""
        self._styles: Dict[str, Callable] = {
            "cartoon": self._cartoon,
            "oil_painting": self._oil_painting,
            "sketch": self._sketch,
            "vintage": self._vintage,
            "pixel": self._pixel,
            "watercolor": self._watercolor,
        }
    
    def apply(self, frame: List[List[List[int]]], 
              style: str) -> List[List[List[int]]]:
        """
        应用风格
        
        Args:
            frame: 输入帧
            style: 风格名称
            
        Returns:
            风格化后的帧
        """
        if style not in self._styles:
            return frame
        
        return self._styles[style](frame)
    
    def _cartoon(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        卡通风格
        特点：边缘增强 + 颜色量化
        
        Args:
            frame: 输入帧
            
        Returns:
            卡通风格帧
        """
        # 边缘检测
        edges = self._detect_edges(frame)
        
        # 颜色量化
        quantized = self._quantize_colors(frame, levels=8)
        
        # 合并
        result = []
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                pixel = []
                edge_strength = edges[i][j] / 255.0
                
                for c in range(len(frame[i][j])):
                    # 边缘处变暗
                    val = quantized[i][j][c] * (1 - edge_strength * 0.5)
                    pixel.append(int(max(0, min(255, val))))
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _detect_edges(self, frame: List[List[List[int]]]) -> List[List[int]]:
        """
        边缘检测（Sobel算子）
        
        Args:
            frame: 输入帧
            
        Returns:
            边缘图
        """
        h, w = len(frame), len(frame[0])
        edges = []
        
        for i in range(h):
            row = []
            for j in range(w):
                if i > 0 and i < h - 1 and j > 0 and j < w - 1:
                    # Sobel算子
                    gx = 0
                    gy = 0
                    
                    for c in range(min(3, len(frame[i][j]))):
                        gx += (
                            -frame[i-1][j-1][c] + frame[i-1][j+1][c] +
                            -2 * frame[i][j-1][c] + 2 * frame[i][j+1][c] +
                            -frame[i+1][j-1][c] + frame[i+1][j+1][c]
                        )
                        gy += (
                            -frame[i-1][j-1][c] - 2 * frame[i-1][j][c] - frame[i-1][j+1][c] +
                            frame[i+1][j-1][c] + 2 * frame[i+1][j][c] + frame[i+1][j+1][c]
                        )
                    
                    edge = math.sqrt(gx**2 + gy**2) / (3 * 4 * 255) * 255
                else:
                    edge = 0
                
                row.append(int(min(255, edge)))
            edges.append(row)
        
        return edges
    
    def _quantize_colors(self, frame: List[List[List[int]]], 
                         levels: int = 8) -> List[List[List[int]]]:
        """
        颜色量化
        
        Args:
            frame: 输入帧
            levels: 量化级别
            
        Returns:
            量化后的帧
        """
        step = 256 // levels
        
        result = []
        for row in frame:
            new_row = []
            for pixel in row:
                new_pixel = [
                    int((c // step) * step + step // 2)
                    for c in pixel
                ]
                new_pixel = [max(0, min(255, c)) for c in new_pixel]
                new_row.append(new_pixel)
            result.append(new_row)
        
        return result
    
    def _oil_painting(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        油画风格
        特点：局部颜色统计 + 笔触效果
        
        Args:
            frame: 输入帧
            
        Returns:
            油画风格帧
        """
        h, w = len(frame), len(frame[0])
        radius = 3
        intensity_levels = 20
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                # 统计邻域内的颜色强度分布
                intensity_count = [0] * intensity_levels
                avg_color = [[0, 0, 0] for _ in range(intensity_levels)]
                
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            # 计算强度
                            intensity = sum(frame[ni][nj][:3]) / 3
                            bin_idx = int(intensity / 256 * intensity_levels)
                            bin_idx = min(bin_idx, intensity_levels - 1)
                            
                            intensity_count[bin_idx] += 1
                            for c in range(min(3, len(frame[ni][nj]))):
                                avg_color[bin_idx][c] += frame[ni][nj][c]
                
                # 选择出现最多的强度级别
                max_idx = intensity_count.index(max(intensity_count))
                
                if intensity_count[max_idx] > 0:
                    pixel = [
                        avg_color[max_idx][c] // intensity_count[max_idx]
                        for c in range(3)
                    ]
                    if len(frame[i][j]) > 3:
                        pixel.extend(frame[i][j][3:])
                else:
                    pixel = frame[i][j][:]
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _sketch(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        素描风格
        特点：边缘提取 + 反色
        
        Args:
            frame: 输入帧
            
        Returns:
            素描风格帧
        """
        # 边缘检测
        edges = self._detect_edges(frame)
        
        # 反色并增强
        result = []
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                edge_val = edges[i][j]
                # 素描效果：白底黑线
                val = 255 - edge_val
                pixel = [val] * min(3, len(frame[i][j]))
                if len(frame[i][j]) > 3:
                    pixel.extend(frame[i][j][3:])
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _vintage(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        怀旧风格
        特点：暖色调 + 降低饱和度 + 暗角
        
        Args:
            frame: 输入帧
            
        Returns:
            怀旧风格帧
        """
        h, w = len(frame), len(frame[0])
        cx, cy = w / 2, h / 2
        max_dist = math.sqrt(cx**2 + cy**2)
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                if len(frame[i][j]) >= 3:
                    r, g, b = frame[i][j][:3]
                    
                    # 暖色调
                    new_r = int(r * 1.1 + 20)
                    new_g = int(g * 0.9)
                    new_b = int(b * 0.8)
                    
                    # 降低饱和度
                    gray = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b
                    sat_factor = 0.7
                    new_r = int(gray + sat_factor * (new_r - gray))
                    new_g = int(gray + sat_factor * (new_g - gray))
                    new_b = int(gray + sat_factor * (new_b - gray))
                    
                    # 暗角
                    dist = math.sqrt((j - cx)**2 + (i - cy)**2)
                    vignette = 1 - 0.3 * (dist / max_dist) ** 2
                    
                    new_r = int(new_r * vignette)
                    new_g = int(new_g * vignette)
                    new_b = int(new_b * vignette)
                    
                    pixel = [
                        max(0, min(255, new_r)),
                        max(0, min(255, new_g)),
                        max(0, min(255, new_b))
                    ]
                    
                    if len(frame[i][j]) > 3:
                        pixel.extend(frame[i][j][3:])
                else:
                    pixel = frame[i][j][:]
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _pixel(self, frame: List[List[List[int]]], 
               pixel_size: int = 8) -> List[List[List[int]]]:
        """
        像素化风格
        
        Args:
            frame: 输入帧
            pixel_size: 像素块大小
            
        Returns:
            像素化风格帧
        """
        h, w = len(frame), len(frame[0])
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                # 计算像素块的平均颜色
                block_i = (i // pixel_size) * pixel_size
                block_j = (j // pixel_size) * pixel_size
                
                total = [0] * len(frame[i][j])
                count = 0
                
                for di in range(pixel_size):
                    for dj in range(pixel_size):
                        ni, nj = block_i + di, block_j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            for c in range(len(total)):
                                total[c] += frame[ni][nj][c]
                            count += 1
                
                if count > 0:
                    pixel = [t // count for t in total]
                else:
                    pixel = frame[i][j][:]
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _watercolor(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        水彩风格
        特点：颜色扩散 + 边缘模糊
        
        Args:
            frame: 输入帧
            
        Returns:
            水彩风格帧
        """
        # 中值滤波（模拟水彩的扩散效果）
        median = self._median_filter(frame, radius=3)
        
        # 边缘软化
        result = self._soft_edges(median)
        
        return result
    
    def _median_filter(self, frame: List[List[List[int]]], 
                       radius: int = 2) -> List[List[List[int]]]:
        """
        中值滤波
        
        Args:
            frame: 输入帧
            radius: 滤波半径
            
        Returns:
            滤波后的帧
        """
        h, w = len(frame), len(frame[0])
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                pixel = []
                
                for c in range(len(frame[i][j])):
                    values = []
                    
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                values.append(frame[ni][nj][c])
                    
                    values.sort()
                    median_val = values[len(values) // 2] if values else 0
                    pixel.append(median_val)
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _soft_edges(self, frame: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        边缘软化
        
        Args:
            frame: 输入帧
            
        Returns:
            边缘软化后的帧
        """
        edges = self._detect_edges(frame)
        
        result = []
        for i in range(len(frame)):
            row = []
            for j in range(len(frame[i])):
                edge_strength = edges[i][j] / 255.0
                pixel = []
                
                for c in range(len(frame[i][j])):
                    # 边缘处轻微模糊
                    if edge_strength > 0.3 and i > 0 and i < len(frame) - 1 and j > 0 and j < len(frame[0]) - 1:
                        val = (
                            frame[i][j][c] * 0.5 +
                            frame[i-1][j][c] * 0.125 +
                            frame[i+1][j][c] * 0.125 +
                            frame[i][j-1][c] * 0.125 +
                            frame[i][j+1][c] * 0.125
                        )
                    else:
                        val = frame[i][j][c]
                    
                    pixel.append(int(val))
                row.append(pixel)
            result.append(row)
        
        return result


class CameraMotion:
    """
    相机运动模拟器
    
    支持多种相机运动：
    - pan: 平移
    - tilt: 俯仰
    - zoom: 推拉
    - rotate: 旋转
    - dolly: 轨道路径
    """
    
    def apply(self, frames: List[List[List[List[int]]]], 
              motion_type: str, 
              params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        应用相机运动
        
        Args:
            frames: 视频帧序列
            motion_type: 运动类型
            params: 运动参数
            
        Returns:
            运动后的视频帧序列
        """
        if not frames:
            return frames
        
        handlers = {
            "pan": self._pan,
            "tilt": self._tilt,
            "zoom": self._zoom,
            "rotate": self._rotate,
            "dolly": self._dolly,
        }
        
        handler = handlers.get(motion_type)
        if handler:
            return handler(frames, params)
        else:
            return frames
    
    def _pan(self, frames: List[List[List[List[int]]]], 
             params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        平移
        
        Args:
            frames: 视频帧序列
            params: 参数（direction: 方向, speed: 速度）
            
        Returns:
            平移后的视频帧序列
        """
        direction = params.get("direction", "right")  # left/right
        speed = params.get("speed", 2)
        
        result = []
        
        for t, frame in enumerate(frames):
            offset = (t + 1) * speed
            if direction == "left":
                offset = -offset
            
            shifted = self._shift_frame(frame, offset, 0)
            result.append(shifted)
        
        return result
    
    def _tilt(self, frames: List[List[List[List[int]]]], 
              params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        俯仰
        
        Args:
            frames: 视频帧序列
            params: 参数（direction: 方向, speed: 速度）
            
        Returns:
            俯仰后的视频帧序列
        """
        direction = params.get("direction", "down")  # up/down
        speed = params.get("speed", 2)
        
        result = []
        
        for t, frame in enumerate(frames):
            offset = (t + 1) * speed
            if direction == "up":
                offset = -offset
            
            shifted = self._shift_frame(frame, 0, offset)
            result.append(shifted)
        
        return result
    
    def _shift_frame(self, frame: List[List[List[int]]], 
                     dx: int, dy: int) -> List[List[List[int]]]:
        """
        平移帧
        
        Args:
            frame: 输入帧
            dx: x方向偏移
            dy: y方向偏移
            
        Returns:
            平移后的帧
        """
        h, w = len(frame), len(frame[0])
        channels = len(frame[0][0])
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                src_i = i - dy
                src_j = j - dx
                
                if 0 <= src_i < h and 0 <= src_j < w:
                    pixel = frame[src_i][src_j][:]
                else:
                    pixel = [0] * channels
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _zoom(self, frames: List[List[List[List[int]]]], 
              params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        推拉
        
        Args:
            frames: 视频帧序列
            params: 参数（direction: 方向, speed: 速度）
            
        Returns:
            推拉后的视频帧序列
        """
        direction = params.get("direction", "in")  # in/out
        speed = params.get("speed", 0.02)
        
        h = len(frames[0]) if frames else 0
        w = len(frames[0][0]) if h > 0 else 0
        cx, cy = w / 2, h / 2
        
        result = []
        
        for t, frame in enumerate(frames):
            if direction == "in":
                scale = 1.0 + (t + 1) * speed
            else:
                scale = 1.0 - (t + 1) * speed
                scale = max(0.5, scale)  # 防止缩放太小
            
            zoomed = self._scale_frame(frame, scale, cx, cy)
            result.append(zoomed)
        
        return result
    
    def _scale_frame(self, frame: List[List[List[int]]], 
                     scale: float, cx: float, cy: float) -> List[List[List[int]]]:
        """
        缩放帧
        
        Args:
            frame: 输入帧
            scale: 缩放因子
            cx: 中心x
            cy: 中心y
            
        Returns:
            缩放后的帧
        """
        h, w = len(frame), len(frame[0])
        channels = len(frame[0][0])
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                src_i = int((i - cy) / scale + cy)
                src_j = int((j - cx) / scale + cx)
                
                if 0 <= src_i < h and 0 <= src_j < w:
                    pixel = frame[src_i][src_j][:]
                else:
                    pixel = [0] * channels
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _rotate(self, frames: List[List[List[List[int]]]], 
                params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        旋转
        
        Args:
            frames: 视频帧序列
            params: 参数（center: 中心, angle: 角度增量）
            
        Returns:
            旋转后的视频帧序列
        """
        center = params.get("center", None)
        angle_increment = params.get("angle", 2)  # 每帧旋转角度
        
        h = len(frames[0]) if frames else 0
        w = len(frames[0][0]) if h > 0 else 0
        
        if center is None:
            center = (w / 2, h / 2)
        
        cx, cy = center
        
        result = []
        
        for t, frame in enumerate(frames):
            angle = math.radians((t + 1) * angle_increment)
            rotated = self._rotate_frame(frame, angle, cx, cy)
            result.append(rotated)
        
        return result
    
    def _rotate_frame(self, frame: List[List[List[int]]], 
                      angle: float, cx: float, cy: float) -> List[List[List[int]]]:
        """
        旋转帧
        
        Args:
            frame: 输入帧
            angle: 旋转角度（弧度）
            cx: 中心x
            cy: 中心y
            
        Returns:
            旋转后的帧
        """
        h, w = len(frame), len(frame[0])
        channels = len(frame[0][0])
        
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        result = []
        
        for i in range(h):
            row = []
            for j in range(w):
                # 反向映射
                x = j - cx
                y = i - cy
                src_j = int(x * cos_a + y * sin_a + cx)
                src_i = int(-x * sin_a + y * cos_a + cy)
                
                if 0 <= src_i < h and 0 <= src_j < w:
                    pixel = frame[src_i][src_j][:]
                else:
                    pixel = [0] * channels
                
                row.append(pixel)
            result.append(row)
        
        return result
    
    def _dolly(self, frames: List[List[List[List[int]]]], 
               params: Dict[str, Any]) -> List[List[List[List[int]]]]:
        """
        轨道路径
        
        Args:
            frames: 视频帧序列
            params: 参数（path: 路径点列表）
            
        Returns:
            轨道运动后的视频帧序列
        """
        path = params.get("path", [])
        
        if not path:
            return frames
        
        h = len(frames[0]) if frames else 0
        w = len(frames[0][0]) if h > 0 else 0
        
        result = []
        
        for t, frame in enumerate(frames):
            # 插值获取当前位置
            path_idx = min(t, len(path) - 1)
            
            if path_idx < len(path) - 1:
                alpha = (t - path_idx) / 1.0
                x1, y1 = path[path_idx]
                x2, y2 = path[path_idx + 1]
                dx = int((1 - alpha) * x1 + alpha * x2)
                dy = int((1 - alpha) * y1 + alpha * y2)
            else:
                dx, dy = path[-1]
            
            shifted = self._shift_frame(frame, dx, dy)
            result.append(shifted)
        
        return result


class TemporalSmoother:
    """
    时域平滑器
    
    使用高斯滤波对视频进行时域平滑
    """
    
    def __init__(self, window_size: int = 5, sigma: float = 1.0):
        """
        初始化时域平滑器
        
        Args:
            window_size: 窗口大小
            sigma: 高斯核标准差
        """
        self._window_size = window_size
        self._sigma = sigma
    
    def smooth(self, frames: List[List[List[List[int]]]]) -> List[List[List[List[int]]]]:
        """
        高斯滤波平滑
        
        Args:
            frames: 视频帧序列
            
        Returns:
            平滑后的视频帧序列
        """
        if len(frames) < self._window_size:
            return frames
        
        # 生成高斯核
        kernel = self._gaussian_kernel(self._window_size, self._sigma)
        half_window = self._window_size // 2
        
        result = []
        
        for t in range(len(frames)):
            if t < half_window or t >= len(frames) - half_window:
                # 边界帧保持不变
                result.append(frames[t])
                continue
            
            # 加权平均
            h = len(frames[t])
            w = len(frames[t][0])
            channels = len(frames[t][0][0])
            
            new_frame = []
            
            for i in range(h):
                row = []
                for j in range(w):
                    pixel = [0.0] * channels
                    
                    for k in range(self._window_size):
                        frame_idx = t - half_window + k
                        weight = kernel[k]
                        
                        for c in range(channels):
                            pixel[c] += weight * frames[frame_idx][i][j][c]
                    
                    pixel = [int(max(0, min(255, c))) for c in pixel]
                    row.append(pixel)
                
                new_frame.append(row)
            
            result.append(new_frame)
        
        return result
    
    def _gaussian_kernel(self, size: int, sigma: float) -> List[float]:
        """
        高斯核
        
        Args:
            size: 核大小
            sigma: 标准差
            
        Returns:
            归一化的高斯核
        """
        half = size // 2
        kernel = []
        
        for i in range(size):
            x = i - half
            val = math.exp(-x**2 / (2 * sigma**2))
            kernel.append(val)
        
        # 归一化
        total = sum(kernel)
        kernel = [k / total for k in kernel]
        
        return kernel


class Watermarker:
    """
    水印器
    
    支持添加文字水印
    """
    
    def add(self, frame: List[List[List[int]]], 
            text: str, 
            position: str = "bottom_right", 
            opacity: float = 0.5) -> List[List[List[int]]]:
        """
        添加水印
        
        Args:
            frame: 输入帧
            text: 水印文字
            position: 位置（top_left/top_right/bottom_left/bottom_right/center）
            opacity: 不透明度（0-1）
            
        Returns:
            添加水印后的帧
        """
        if not frame or not text:
            return frame
        
        h, w = len(frame), len(frame[0])
        
        # 生成文字图像（简化：使用ASCII字符渲染）
        text_img = self._render_text(text)
        text_h, text_w = len(text_img), len(text_img[0]) if text_img else 0
        
        if text_h == 0 or text_w == 0:
            return frame
        
        # 计算位置
        positions = {
            "top_left": (10, 10),
            "top_right": (w - text_w - 10, 10),
            "bottom_left": (10, h - text_h - 10),
            "bottom_right": (w - text_w - 10, h - text_h - 10),
            "center": ((w - text_w) // 2, (h - text_h) // 2),
        }
        
        x, y = positions.get(position, (w - text_w - 10, h - text_h - 10))
        
        # 混合文字
        result = [row[:] for row in frame]
        result = self._blend_text(result, text_img, x, y, opacity)
        
        return result
    
    def _render_text(self, text: str) -> List[List[int]]:
        """
        渲染文字（简化实现）
        
        Args:
            text: 文字内容
            
        Returns:
            文字图像（灰度）
        """
        # 简化：使用固定大小的文字块
        char_width = 8
        char_height = 12
        
        width = len(text) * char_width
        height = char_height
        
        # 创建文字图像
        text_img = [[0 for _ in range(width)] for _ in range(height)]
        
        # 简化：用矩形表示每个字符
        for i, char in enumerate(text):
            if char != ' ':
                x_start = i * char_width + 1
                x_end = x_start + char_width - 2
                y_start = 2
                y_end = height - 2
                
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        if 0 <= y < height and 0 <= x < width:
                            text_img[y][x] = 255
        
        return text_img
    
    def _blend_text(self, frame: List[List[List[int]]], 
                    text_img: List[List[int]], 
                    x: int, y: int, 
                    opacity: float) -> List[List[List[int]]]:
        """
        混合文字
        
        Args:
            frame: 输入帧
            text_img: 文字图像
            x: x位置
            y: y位置
            opacity: 不透明度
            
        Returns:
            混合后的帧
        """
        h, w = len(frame), len(frame[0])
        text_h, text_w = len(text_img), len(text_img[0])
        
        for i in range(text_h):
            for j in range(text_w):
                frame_y = y + i
                frame_x = x + j
                
                if 0 <= frame_y < h and 0 <= frame_x < w:
                    text_val = text_img[i][j] / 255.0
                    
                    for c in range(len(frame[frame_y][frame_x])):
                        original = frame[frame_y][frame_x][c]
                        # 白色文字
                        blended = original * (1 - text_val * opacity) + 255 * text_val * opacity
                        frame[frame_y][frame_x][c] = int(max(0, min(255, blended)))
        
        return frame


class LipSync:
    """
    嘴唇同步器
    
    实现音频驱动的口型同步
    """
    
    def sync(self, video_frames: List[List[List[List[int]]]], 
             audio: List[float]) -> List[List[List[List[int]]]]:
        """
        音频驱动口型同步
        
        Args:
            video_frames: 视频帧序列
            audio: 音频特征（简化为能量序列）
            
        Returns:
            同步后的视频帧序列
        """
        if not video_frames:
            return video_frames
        
        # 检测人脸
        faces = [self._detect_face(frame) for frame in video_frames]
        
        # 生成唇部运动
        lip_motions = self._generate_lip_motion(audio, len(video_frames))
        
        # 应用唇部运动
        result = []
        
        for t, frame in enumerate(video_frames):
            if t < len(faces) and faces[t] and t < len(lip_motions):
                modified_frame = self._apply_lip_motion(frame, faces[t], lip_motions[t])
                result.append(modified_frame)
            else:
                result.append(frame)
        
        return result
    
    def _detect_face(self, frame: List[List[List[int]]]) -> Optional[Dict[str, Any]]:
        """
        基于Haar级联的简化人脸检测
        使用肤色检测 + 边缘检测 + 比例验证的三阶段方法：
        1. 肤色检测：在YCbCr色彩空间中检测肤色区域
        2. 边缘检测：使用Sobel算子检测肤色区域的边缘
        3. 比例验证：根据人脸宽高比和面积过滤候选区域
        
        Args:
            frame: 输入帧 [H, W, C]
            
        Returns:
            人脸信息字典，包含位置和大小
        """
        if not frame:
            return None
        
        h, w = len(frame), len(frame[0])
        
        # ---- 第一阶段：肤色检测（YCbCr色彩空间）----
        # 在YCbCr空间中，肤色满足：Cb in [77, 127], Cr in [133, 173]
        skin_mask = [[False] * w for _ in range(h)]
        skin_pixels = []
        
        for i in range(h):
            for j in range(w):
                if len(frame[i][j]) >= 3:
                    r, g, b = frame[i][j][:3]
                    # RGB转YCbCr
                    y_val = 0.299 * r + 0.587 * g + 0.114 * b
                    cb = 128 - 0.169 * r - 0.331 * g + 0.500 * b
                    cr = 128 + 0.500 * r - 0.419 * g - 0.081 * b
                    
                    # 肤色范围判断
                    if 77 <= cb <= 127 and 133 <= cr <= 173:
                        skin_mask[i][j] = True
                        skin_pixels.append((i, j))
        
        if len(skin_pixels) < 50:
            # 未检测到足够的肤色像素，返回默认位置
            return {
                'bbox': (w // 2 - 50, h // 2 - 50, 100, 100),
                'mouth_region': (w // 2 - 20, h // 2 + 20, 40, 20),
                'confidence': 0.3
            }
        
        # ---- 第二阶段：边缘检测 ----
        # 对肤色掩码应用Sobel算子，检测肤色区域的边界
        edge_strength = [[0] * w for _ in range(h)]
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if not skin_mask[i][j]:
                    continue
                # 检查邻域是否有非肤色像素（肤色区域边界即为边缘）
                gx = 0
                gy = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            val = 1 if skin_mask[ni][nj] else 0
                            # Sobel核
                            if (di, dj) in [(-1, -1), (1, 1)]:
                                gx -= val
                                gy -= val
                            elif (di, dj) in [(-1, 1), (1, -1)]:
                                gx += val
                                gy += val
                            elif dj == 1:
                                gx += 2 * val
                            elif dj == -1:
                                gx -= 2 * val
                            elif di == 1:
                                gy += 2 * val
                            elif di == -1:
                                gy -= 2 * val
                
                edge_strength[i][j] = int(math.sqrt(gx ** 2 + gy ** 2))
        
        # ---- 第三阶段：连通区域分析与比例验证 ----
        # 简化实现：使用滑动窗口在肤色密集区域寻找最大的人脸候选
        # 将图像划分为网格，统计每个网格的肤色像素密度
        grid_size = 16
        grid_h = (h + grid_size - 1) // grid_size
        grid_w = (w + grid_size - 1) // grid_size
        density = [[0] * grid_w for _ in range(grid_h)]
        
        for si, sj in skin_pixels:
            gi, gj = si // grid_size, sj // grid_size
            if gi < grid_h and gj < grid_w:
                density[gi][gj] += 1
        
        # 找到肤色密度最高的区域作为人脸候选
        max_density = 0
        best_gi, best_gj = 0, 0
        for gi in range(grid_h):
            for gj in range(grid_w):
                if density[gi][gj] > max_density:
                    max_density = density[gi][gj]
                    best_gi, best_gj = gi, gj
        
        # 从最高密度网格扩展，找到连通的肤色区域边界框
        # 使用简单的区域生长方法
        cx = (best_gj + 0.5) * grid_size
        cy = (best_gi + 0.5) * grid_size
        search_radius = min(w, h) // 3
        
        # 在搜索范围内统计肤色像素
        region_pixels = []
        for si, sj in skin_pixels:
            if abs(si - cy) < search_radius and abs(sj - cx) < search_radius:
                region_pixels.append((si, sj))
        
        if not region_pixels:
            return {
                'bbox': (w // 2 - 50, h // 2 - 50, 100, 100),
                'mouth_region': (w // 2 - 20, h // 2 + 20, 40, 20),
                'confidence': 0.3
            }
        
        # 计算边界框
        ys = [p[0] for p in region_pixels]
        xs = [p[1] for p in region_pixels]
        
        face_top = min(ys)
        face_bottom = max(ys)
        face_left = min(xs)
        face_right = max(xs)
        
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        
        # ---- 比例验证 ----
        # 人脸宽高比通常在 0.6 ~ 1.4 之间
        if face_width > 0:
            aspect_ratio = face_height / face_width
        else:
            aspect_ratio = 1.0
        
        # 面积占图像比例应在合理范围内（0.5% ~ 60%）
        face_area = face_width * face_height
        image_area = h * w
        area_ratio = face_area / image_area if image_area > 0 else 0
        
        # 计算置信度
        confidence = 0.0
        # 宽高比得分（越接近1.0越好）
        if 0.6 <= aspect_ratio <= 1.4:
            confidence += 0.3
        else:
            confidence += 0.1 * max(0, 1.0 - abs(aspect_ratio - 1.0))
        
        # 面积比例得分
        if 0.005 <= area_ratio <= 0.6:
            confidence += 0.3
        else:
            confidence += 0.1
        
        # 肤色密度得分
        density_score = min(1.0, len(region_pixels) / (face_area + 1) * 10)
        confidence += 0.4 * density_score
        
        confidence = min(1.0, confidence)
        
        # 估计嘴部位置（人脸下三分之一区域）
        mouth_y = face_top + face_height * 2 // 3
        mouth_x = face_left + face_width // 2
        mouth_width = max(10, face_width // 2)
        mouth_height = max(5, face_height // 6)
        
        return {
            'bbox': (face_left, face_top, face_width, face_height),
            'mouth_region': (mouth_x - mouth_width // 2, mouth_y, mouth_width, mouth_height),
            'confidence': confidence
        }
    
    def _extract_mouth_region(self, face: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        """
        提取嘴部区域
        
        Args:
            face: 人脸信息
            
        Returns:
            嘴部区域 (x, y, width, height)
        """
        if not face or 'mouth_region' not in face:
            return None
        
        return face['mouth_region']
    
    def _generate_lip_motion(self, audio: List[float], 
                             num_frames: int) -> List[Dict[str, Any]]:
        """
        生成唇部运动
        
        Args:
            audio: 音频特征
            num_frames: 帧数
            
        Returns:
            唇部运动参数列表
        """
        motions = []
        
        for t in range(num_frames):
            # 获取当前帧对应的音频能量
            audio_idx = int(t * len(audio) / num_frames) if audio else 0
            energy = audio[audio_idx] if audio_idx < len(audio) else 0.0
            
            # 根据音频能量生成唇部开合程度
            # 能量越大，嘴巴张得越大
            openness = min(1.0, energy / 100.0) if audio else 0.3
            
            # 添加一些随机变化使运动更自然
            variation = 0.1 * math.sin(t * 0.5)
            openness = max(0.0, min(1.0, openness + variation))
            
            motions.append({
                'openness': openness,
                'width': 1.0 - 0.2 * openness,  # 嘴巴张开时变窄
                'expression': 'neutral'
            })
        
        return motions
    
    def _apply_lip_motion(self, frame: List[List[List[int]]], 
                          face: Dict[str, Any], 
                          lip_motion: Dict[str, Any]) -> List[List[List[int]]]:
        """
        应用唇部运动
        
        Args:
            frame: 输入帧
            face: 人脸信息
            lip_motion: 唇部运动参数
            
        Returns:
            修改后的帧
        """
        if not face or 'mouth_region' not in face:
            return frame
        
        result = [row[:] for row in frame]
        
        mouth_x, mouth_y, mouth_w, mouth_h = face['mouth_region']
        openness = lip_motion.get('openness', 0.3)
        width_factor = lip_motion.get('width', 1.0)
        
        # 简化：通过修改嘴部区域的颜色来模拟嘴唇运动
        # 实际应用中应使用更复杂的形变或生成模型
        
        # 嘴部张开时，内部变暗（模拟口腔）
        inner_h = int(mouth_h * openness)
        
        for i in range(mouth_y, min(mouth_y + mouth_h, len(frame))):
            for j in range(mouth_x, min(mouth_x + mouth_w, len(frame[0]))):
                # 计算在嘴部区域内的相对位置
                rel_y = (i - mouth_y) / mouth_h
                rel_x = (j - mouth_x) / mouth_w - 0.5
                
                # 根据开合程度和位置决定是否为嘴部内部
                if rel_y > (1 - openness) and abs(rel_x) < width_factor / 2:
                    # 嘴部内部，变暗
                    for c in range(len(result[i][j])):
                        result[i][j][c] = int(result[i][j][c] * 0.3)
        
        return result
