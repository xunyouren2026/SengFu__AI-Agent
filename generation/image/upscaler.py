"""
超分辨率模块

提供图像超分辨率功能，包括Real-ESRGAN风格上采样和分块处理
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict


@dataclass
class UpscalerConfig:
    """超分辨率配置"""
    scale_factor: int = 4  # 放大倍数 (2/4/8)
    denoise_strength: float = 0.5
    tile_size: int = 512
    overlap: int = 64


class RealESRGANStyle:
    """
    Real-ESRGAN风格上采样
    
    实现类似Real-ESRGAN的超分辨率算法
    """
    
    def __init__(self, scale: int = 4, num_features: int = 64):
        self._scale = scale
        self._num_features = num_features
    
    def upscale(self, image: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        上采样图像
        
        Args:
            image: 输入图像 [H][W][3]
            
        Returns:
            上采样后的图像 [H*scale][W*scale][3]
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 转换为浮点数
        float_image = [[[float(c) / 255.0 for c in pixel] for pixel in row] for row in image]
        
        # 特征提取 (简化版)
        features = self._extract_features(float_image)
        
        # 残差密集块处理
        for _ in range(3):  # 3个残差密集块
            features = self._residual_dense_block(features)
        
        # 上采样
        upsampled = self._upsample_block(features, self._scale)
        
        # 重构图像
        result = self._reconstruct(upsampled, float_image)
        
        # 转换回整数
        return [[[int(max(0, min(255, c * 255))) for c in pixel] for pixel in row] for row in result]
    
    def _extract_features(self, image: List[List[List[float]]]) -> List[List[List[float]]]:
        """提取特征"""
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 简化的特征提取 (3x3卷积模拟)
        features = [[[0.0 for _ in range(self._num_features)] 
                     for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                for f in range(self._num_features):
                    # 简化的卷积操作
                    value = 0.0
                    for c in range(3):
                        value += image[y][x][c] * (random.random() * 0.1 - 0.05)
                    features[y][x][f] = max(0, value)  # ReLU激活
        
        return features
    
    def _residual_dense_block(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        残差密集块
        
        Args:
            x: 输入特征
            
        Returns:
            处理后的特征
        """
        height = len(x)
        width = len(x[0]) if height > 0 else 0
        channels = len(x[0][0]) if width > 0 else 0
        
        # 密集连接 (简化版)
        out = [[[v for v in pixel] for pixel in row] for row in x]
        
        # 模拟卷积层
        for y in range(1, height - 1):
            for x_pos in range(1, width - 1):
                for c in range(channels):
                    # 3x3卷积
                    value = 0.0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            for oc in range(channels):
                                value += out[y + dy][x_pos + dx][oc] * 0.01
                    out[y][x_pos][c] += max(0, value)  # 残差连接
        
        return out
    
    def _pixel_shuffle(self, x: List[List[List[float]]], scale: int) -> List[List[List[float]]]:
        """
        像素重组 (Pixel Shuffle)
        
        Args:
            x: 输入特征
            scale: 上采样倍数
            
        Returns:
            重组后的特征
        """
        height = len(x)
        width = len(x[0]) if height > 0 else 0
        channels = len(x[0][0]) if width > 0 else 0
        
        new_height = height * scale
        new_width = width * scale
        new_channels = channels // (scale * scale)
        
        if new_channels <= 0:
            new_channels = 1
        
        result = [[[0.0 for _ in range(new_channels)] 
                   for _ in range(new_width)] for _ in range(new_height)]
        
        for y in range(height):
            for x_pos in range(width):
                for c in range(min(channels, new_channels * scale * scale)):
                    # 计算目标位置
                    oc = c % new_channels
                    sy = (c // new_channels) // scale
                    sx = (c // new_channels) % scale
                    
                    ty = y * scale + sy
                    tx = x_pos * scale + sx
                    
                    if ty < new_height and tx < new_width:
                        result[ty][tx][oc] = x[y][x_pos][c % channels]
        
        return result
    
    def _upsample_block(self, x: List[List[List[float]]], scale: int) -> List[List[List[float]]]:
        """
        上采样块
        
        Args:
            x: 输入特征
            scale: 上采样倍数
            
        Returns:
            上采样后的特征
        """
        height = len(x)
        width = len(x[0]) if height > 0 else 0
        channels = len(x[0][0]) if width > 0 else 0
        
        # 增加通道数以支持像素重组
        expanded_channels = channels * scale * scale
        
        # 扩展通道 (简化的卷积)
        expanded = [[[0.0 for _ in range(expanded_channels)] 
                     for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x_pos in range(width):
                for ec in range(expanded_channels):
                    value = 0.0
                    for c in range(channels):
                        value += x[y][x_pos][c] * (0.9 + random.random() * 0.2)
                    expanded[y][x_pos][ec] = value / channels
        
        # 像素重组
        return self._pixel_shuffle(expanded, scale)
    
    def _conv_block(self, x: List[List[List[float]]], 
                    out_channels: int, 
                    kernel: int = 3) -> List[List[List[float]]]:
        """
        卷积块
        
        Args:
            x: 输入特征
            out_channels: 输出通道数
            kernel: 卷积核大小
            
        Returns:
            卷积后的特征
        """
        height = len(x)
        width = len(x[0]) if height > 0 else 0
        in_channels = len(x[0][0]) if width > 0 else 0
        
        result = [[[0.0 for _ in range(out_channels)] 
                   for _ in range(width)] for _ in range(height)]
        
        half_k = kernel // 2
        
        for y in range(height):
            for x_pos in range(width):
                for oc in range(out_channels):
                    value = 0.0
                    for ky in range(-half_k, half_k + 1):
                        for kx in range(-half_k, half_k + 1):
                            py = max(0, min(height - 1, y + ky))
                            px = max(0, min(width - 1, x_pos + kx))
                            for ic in range(in_channels):
                                value += x[py][px][ic] * 0.01
                    result[y][x_pos][oc] = max(0, value)
        
        return result
    
    def _reconstruct(self, features: List[List[List[float]]], 
                     original: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        重构图像
        
        Args:
            features: 上采样后的特征
            original: 原始图像
            
        Returns:
            重构的图像
        """
        height = len(features)
        width = len(features[0]) if height > 0 else 0
        
        # 特征到RGB的映射
        result = [[[0.0 for _ in range(3)] for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    # 简化的特征组合
                    value = 0.0
                    for f in features[y][x]:
                        value += f * 0.1
                    result[y][x][c] = max(0, min(1.0, value + 0.5))
        
        return result


class TileUpscaler:
    """
    分块上采样器
    
    用于处理大图像的分块上采样
    """
    
    def __init__(self, upscaler: RealESRGANStyle, tile_size: int = 512, overlap: int = 64):
        self._upscaler = upscaler
        self._tile_size = tile_size
        self._overlap = overlap
    
    def upscale_tiled(self, image: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        分块上采样
        
        Args:
            image: 输入图像
            
        Returns:
            上采样后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 分割为块
        tiles = self._split_to_tiles(image)
        
        # 上采样每个块
        upscaled_tiles = []
        for tile in tiles:
            upscaled = self._upscaler.upscale(tile)
            upscaled_tiles.append(upscaled)
        
        # 合并块
        original_size = (height, width)
        result = self._merge_tiles(upscaled_tiles, original_size, self._overlap)
        
        return result
    
    def _split_to_tiles(self, image: List[List[List[int]]]) -> List[List[List[List[int]]]]:
        """
        分割为块
        
        Args:
            image: 输入图像
            
        Returns:
            图像块列表
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        tiles = []
        stride = self._tile_size - self._overlap
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # 计算块边界
                y_end = min(y + self._tile_size, height)
                x_end = min(x + self._tile_size, width)
                y_start = max(0, y_end - self._tile_size)
                x_start = max(0, x_end - self._tile_size)
                
                # 提取块
                tile = []
                for ty in range(y_start, y_end):
                    row = []
                    for tx in range(x_start, x_end):
                        row.append(image[ty][tx][:])
                    tile.append(row)
                
                tiles.append(tile)
        
        return tiles
    
    def _merge_tiles(self, tiles: List[List[List[List[int]]]], 
                     original_size: Tuple[int, int],
                     overlap: int) -> List[List[List[int]]]:
        """
        合并块
        
        Args:
            tiles: 上采样后的块列表
            original_size: 原始图像尺寸
            overlap: 重叠区域大小
            
        Returns:
            合并后的图像
        """
        orig_height, orig_width = original_size
        scale = self._upscaler._scale
        
        new_height = orig_height * scale
        new_width = orig_width * scale
        
        # 初始化结果图像
        result = [[[0, 0, 0] for _ in range(new_width)] for _ in range(new_height)]
        weights = [[0.0 for _ in range(new_width)] for _ in range(new_height)]
        
        stride = (self._tile_size - self._overlap) * scale
        tile_idx = 0
        
        for y in range(0, new_height, stride):
            for x in range(0, new_width, stride):
                if tile_idx >= len(tiles):
                    break
                
                tile = tiles[tile_idx]
                tile_height = len(tile)
                tile_width = len(tile[0]) if tile_height > 0 else 0
                
                for ty in range(tile_height):
                    for tx in range(tile_width):
                        ry = y + ty
                        rx = x + tx
                        
                        if ry < new_height and rx < new_width:
                            # 计算权重 (边缘区域权重较低)
                            weight = 1.0
                            if ty < overlap or ty >= tile_height - overlap:
                                weight *= 0.5
                            if tx < overlap or tx >= tile_width - overlap:
                                weight *= 0.5
                            
                            for c in range(3):
                                result[ry][rx][c] += int(tile[ty][tx][c] * weight)
                            weights[ry][rx] += weight
                
                tile_idx += 1
        
        # 归一化
        for y in range(new_height):
            for x in range(new_width):
                if weights[y][x] > 0:
                    for c in range(3):
                        result[y][x][c] = int(result[y][x][c] / weights[y][x])
        
        return result
    
    def _blend_overlap(self, region_a: List[List[List[int]]], 
                       region_b: List[List[List[int]]], 
                       alpha: float) -> List[List[List[int]]]:
        """
        混合重叠区域
        
        Args:
            region_a: 区域A
            region_b: 区域B
            alpha: 混合系数
            
        Returns:
            混合后的区域
        """
        height = min(len(region_a), len(region_b))
        width = min(len(region_a[0]) if region_a else 0, len(region_b[0]) if region_b else 0)
        
        result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    val_a = region_a[y][x][c]
                    val_b = region_b[y][x][c]
                    result[y][x][c] = int(val_a * (1 - alpha) + val_b * alpha)
        
        return result


class UpscalePipeline:
    """
    超分辨率管线
    
    整合上采样、锐化和去噪的完整处理管线
    """
    
    def __init__(self, config: Optional[UpscalerConfig] = None):
        self._config = config or UpscalerConfig()
        self._upscaler = RealESRGANStyle(scale=self._config.scale_factor)
        self._tile_upscaler = TileUpscaler(
            self._upscaler,
            tile_size=self._config.tile_size,
            overlap=self._config.overlap
        )
    
    def upscale(self, image: List[List[List[int]]], 
                scale: int = 4, 
                use_tiling: bool = False) -> Dict[str, Any]:
        """
        超分辨率处理
        
        Args:
            image: 输入图像
            scale: 放大倍数
            use_tiling: 是否使用分块处理
            
        Returns:
            处理结果
        """
        # 预处理
        preprocessed = self._preprocess_image(image)
        
        # 更新缩放倍数
        self._upscaler._scale = scale
        
        # 上采样
        if use_tiling:
            upscaled = self._tile_upscaler.upscale_tiled(preprocessed)
        else:
            upscaled = self._upscaler.upscale(preprocessed)
        
        # 后处理
        result = self._postprocess_image(upscaled)
        
        # 锐化
        result = self._sharpen(result, amount=1.0)
        
        # 去噪
        if self._config.denoise_strength > 0:
            result = self._denoise(result, strength=self._config.denoise_strength)
        
        return {
            'image': result,
            'scale': scale,
            'original_size': (len(image), len(image[0]) if image else 0),
            'new_size': (len(result), len(result[0]) if result else 0)
        }
    
    def _preprocess_image(self, image: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 复制图像
        result = [[[c for c in pixel] for pixel in row] for row in image]
        
        # 归一化亮度
        total_brightness = 0
        count = 0
        for y in range(height):
            for x in range(width):
                r, g, b = result[y][x]
                total_brightness += (r + g + b) / 3.0
                count += 1
        
        avg_brightness = total_brightness / count if count > 0 else 128
        
        # 调整亮度到标准范围
        target_brightness = 128
        adjustment = target_brightness / (avg_brightness + 1e-6)
        
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    result[y][x][c] = int(max(0, min(255, result[y][x][c] * adjustment)))
        
        return result
    
    def _postprocess_image(self, image: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        后处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            后处理后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        result = [[[c for c in pixel] for pixel in row] for row in image]
        
        # 对比度增强
        contrast_factor = 1.1
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    val = result[y][x][c]
                    # 以128为中心进行对比度调整
                    new_val = 128 + (val - 128) * contrast_factor
                    result[y][x][c] = int(max(0, min(255, new_val)))
        
        return result
    
    def _sharpen(self, image: List[List[List[int]]], amount: float = 1.0) -> List[List[List[int]]]:
        """
        锐化图像
        
        Args:
            image: 输入图像
            amount: 锐化强度
            
        Returns:
            锐化后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 拉普拉斯锐化核
        kernel = [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]
        
        result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                for c in range(3):
                    value = 0
                    for ky in range(3):
                        for kx in range(3):
                            value += image[y + ky - 1][x + kx - 1][c] * kernel[ky][kx]
                    
                    # 混合原始图像和锐化结果
                    orig_val = image[y][x][c]
                    sharp_val = int(value)
                    final_val = int(orig_val * (1 - amount * 0.3) + sharp_val * amount * 0.3)
                    result[y][x][c] = max(0, min(255, final_val))
        
        # 复制边界
        for y in range(height):
            result[y][0] = image[y][0][:]
            result[y][width - 1] = image[y][width - 1][:]
        for x in range(width):
            result[0][x] = image[0][x][:]
            result[height - 1][x] = image[height - 1][x][:]
        
        return result
    
    def _denoise(self, image: List[List[List[int]]], strength: float = 0.5) -> List[List[List[int]]]:
        """
        去噪
        
        Args:
            image: 输入图像
            strength: 去噪强度
            
        Returns:
            去噪后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        # 高斯滤波去噪
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                for c in range(3):
                    # 3x3高斯滤波
                    value = 0
                    weights = [
                        [1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]
                    ]
                    weight_sum = 16
                    
                    for ky in range(3):
                        for kx in range(3):
                            value += image[y + ky - 1][x + kx - 1][c] * weights[ky][kx]
                    
                    filtered_val = value / weight_sum
                    orig_val = image[y][x][c]
                    
                    # 混合原始图像和滤波结果
                    final_val = int(orig_val * (1 - strength) + filtered_val * strength)
                    result[y][x][c] = max(0, min(255, final_val))
        
        # 复制边界
        for y in range(height):
            result[y][0] = image[y][0][:]
            result[y][width - 1] = image[y][width - 1][:]
        for x in range(width):
            result[0][x] = image[0][x][:]
            result[height - 1][x] = image[height - 1][x][:]
        
        return result


# 导出
__all__ = [
    'UpscalerConfig',
    'RealESRGANStyle',
    'TileUpscaler',
    'UpscalePipeline',
]
