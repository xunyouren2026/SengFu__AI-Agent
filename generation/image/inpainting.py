"""
图像局部重绘模块

提供图像修复(inpainting)和外扩(outpainting)功能
"""

import math
import random
from typing import List, Tuple, Any, Optional, Dict


class MaskProcessor:
    """
    掩码处理器
    
    处理修复掩码，包括膨胀、羽化边缘和高斯金字塔
    """
    
    def __init__(self, blur_radius: int = 4):
        self._blur_radius = blur_radius
    
    def process_mask(self, mask_image: List[List[int]]) -> List[List[float]]:
        """
        处理掩码
        
        Args:
            mask_image: 二值掩码图像 [H][W]
            
        Returns:
            处理后的浮点掩码 [H][W]
        """
        # 膨胀掩码
        dilated = self._dilate_mask(mask_image, iterations=3)
        
        # 羽化边缘
        feathered = self._feather_edges(dilated, self._blur_radius)
        
        return feathered
    
    def _dilate_mask(self, mask: List[List[int]], iterations: int = 3) -> List[List[int]]:
        """
        膨胀掩码
        
        Args:
            mask: 二值掩码
            iterations: 膨胀次数
            
        Returns:
            膨胀后的掩码
        """
        height = len(mask)
        width = len(mask[0]) if height > 0 else 0
        
        result = [row[:] for row in mask]
        
        for _ in range(iterations):
            new_result = [row[:] for row in result]
            
            for y in range(height):
                for x in range(width):
                    if result[y][x] > 0:
                        # 膨胀到8邻域
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    new_result[ny][nx] = 255
            
            result = new_result
        
        return result
    
    def _feather_edges(self, mask: List[List[int]], radius: int) -> List[List[float]]:
        """
        羽化边缘
        
        对掩码边缘进行高斯模糊处理
        
        Args:
            mask: 二值掩码
            radius: 羽化半径
            
        Returns:
            羽化后的浮点掩码
        """
        height = len(mask)
        width = len(mask[0]) if height > 0 else 0
        
        # 转换为浮点数
        float_mask = [[float(v) / 255.0 for v in row] for row in mask]
        
        if radius <= 0:
            return float_mask
        
        # 高斯模糊
        result = [[0.0 for _ in range(width)] for _ in range(height)]
        
        # 创建高斯核
        kernel_size = 2 * radius + 1
        sigma = radius / 2.0
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
        half_k = kernel_size // 2
        
        for y in range(height):
            for x in range(width):
                value = 0.0
                weight = 0.0
                
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        py = y + ky - half_k
                        px = x + kx - half_k
                        
                        if 0 <= py < height and 0 <= px < width:
                            w = kernel[ky][kx]
                            value += float_mask[py][px] * w
                            weight += w
                
                result[y][x] = value / weight if weight > 0 else 0.0
        
        return result
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> List[List[float]]:
        """创建高斯核"""
        kernel = [[0.0 for _ in range(size)] for _ in range(size)]
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                dy = y - center
                dx = x - center
                kernel[y][x] = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
        
        return kernel
    
    def _create_gaussian_pyramid(self, mask: List[List[float]], levels: int = 3) -> List[List[List[float]]]:
        """
        创建高斯金字塔
        
        Args:
            mask: 浮点掩码
            levels: 金字塔层数
            
        Returns:
            高斯金字塔列表
        """
        pyramid = [mask]
        
        current = mask
        for _ in range(levels - 1):
            # 下采样
            height = len(current)
            width = len(current[0]) if height > 0 else 0
            
            new_height = height // 2
            new_width = width // 2
            
            downsampled = [[0.0 for _ in range(new_width)] for _ in range(new_height)]
            
            for y in range(new_height):
                for x in range(new_width):
                    # 2x2平均池化
                    values = []
                    for dy in range(2):
                        for dx in range(2):
                            py = min(2 * y + dy, height - 1)
                            px = min(2 * x + dx, width - 1)
                            values.append(current[py][px])
                    downsampled[y][x] = sum(values) / len(values)
            
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid


class Inpainter:
    """
    局部重绘器
    
    实现图像修复功能，支持多种融合算法
    """
    
    def __init__(self, pipeline: Any = None):
        self._pipeline = pipeline
        self._mask_processor = MaskProcessor(blur_radius=4)
    
    def inpaint(self, image: List[List[List[int]]], 
                mask: List[List[int]], 
                prompt: str,
                **kwargs) -> Dict[str, Any]:
        """
        局部重绘
        
        Args:
            image: 原始图像 [H][W][3]
            mask: 修复掩码 [H][W]
            prompt: 文本提示
            **kwargs: 其他参数
            
        Returns:
            修复结果
        """
        # 处理掩码
        processed_mask = self._mask_processor.process_mask(mask)
        
        # 准备掩码图像
        masked_image, mask_latent = self._prepare_masked_image(image, processed_mask)
        
        # 编码掩码区域
        encoded_region = self._encode_masked_region(image, processed_mask)
        
        # 去噪重绘
        steps = kwargs.get('steps', 20)
        text_embeds = self._encode_text(prompt)
        
        generated_latent = self._denoise_inpaint(
            masked_image, mask_latent, text_embeds, steps
        )
        
        # 解码生成结果
        generated_image = self._latent_to_image(generated_latent)
        
        # 混合结果
        result = self._blend_result(image, generated_image, processed_mask)
        
        return {
            'image': result,
            'mask': processed_mask,
            'prompt': prompt
        }
    
    def _prepare_masked_image(self, image: List[List[List[int]]], 
                              mask: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        准备掩码图像
        
        Args:
            image: 原始图像
            mask: 处理后的掩码
            
        Returns:
            (掩码后的图像, 掩码潜在变量)
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 创建掩码后的图像
        masked = [[0.0 for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                # 转换为灰度并应用掩码
                r, g, b = image[y][x]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                masked[y][x] = gray * (1.0 - mask[y][x])
        
        # 创建掩码潜在变量 (简化版)
        mask_latent = [[mask[y][x] for x in range(width)] for y in range(height)]
        
        return masked, mask_latent
    
    def _encode_masked_region(self, image: List[List[List[int]]], 
                              mask: List[List[float]]) -> List[List[float]]:
        """
        编码掩码区域
        
        Args:
            image: 原始图像
            mask: 掩码
            
        Returns:
            编码后的区域特征
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 提取掩码区域的特征
        features = [[0.0 for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                if mask[y][x] > 0.5:
                    r, g, b = image[y][x]
                    # 简单的特征编码
                    features[y][x] = (r + g + b) / (3.0 * 255.0)
        
        return features
    
    def _denoise_inpaint(self, latent: List[List[float]], 
                         mask_latent: List[List[float]],
                         text_embeds: List[float],
                         steps: int) -> List[List[float]]:
        """
        去噪重绘
        
        Args:
            latent: 初始潜在变量
            mask_latent: 掩码潜在变量
            text_embeds: 文本嵌入
            steps: 去噪步数
            
        Returns:
            去噪后的潜在变量
        """
        height = len(latent)
        width = len(latent[0]) if height > 0 else 0
        
        result = [row[:] for row in latent]
        
        for step in range(steps):
            t = 1.0 - step / steps
            
            for y in range(height):
                for x in range(width):
                    # 仅在掩码区域进行去噪
                    if mask_latent[y][x] > 0.1:
                        noise = random.gauss(0, t * 0.1)
                        result[y][x] = result[y][x] * (1 - 0.1 * t) + noise
        
        return result
    
    def _encode_text(self, prompt: str) -> List[float]:
        """编码文本 (简化)"""
        embed_dim = 768
        hash_val = sum(ord(c) for c in prompt)
        random.seed(hash_val)
        return [random.gauss(0, 0.1) for _ in range(embed_dim)]
    
    def _latent_to_image(self, latent: List[List[float]]) -> List[List[List[int]]]:
        """潜在变量转图像 (简化)"""
        height = len(latent)
        width = len(latent[0]) if height > 0 else 0
        
        image = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                val = int((latent[y][x] + 1.0) * 127.5)
                val = max(0, min(255, val))
                image[y][x] = [val, val, val]
        
        return image
    
    def _blend_result(self, original: List[List[List[int]]], 
                      generated: List[List[List[int]]], 
                      mask: List[List[float]]) -> List[List[List[int]]]:
        """
        混合结果
        
        Args:
            original: 原始图像
            generated: 生成图像
            mask: 混合掩码
            
        Returns:
            混合后的图像
        """
        height = len(original)
        width = len(original[0]) if height > 0 else 0
        
        result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        for y in range(height):
            for x in range(width):
                m = mask[y][x]
                for c in range(3):
                    orig_val = original[y][x][c]
                    gen_val = generated[y][x][c]
                    # 线性混合
                    result[y][x][c] = int(orig_val * (1.0 - m) + gen_val * m)
        
        return result
    
    def _poisson_blend(self, source: List[List[List[int]]], 
                       target: List[List[List[int]]], 
                       mask: List[List[float]]) -> List[List[List[int]]]:
        """
        泊松融合 (简化版)
        
        Args:
            source: 源图像
            target: 目标图像
            mask: 融合掩码
            
        Returns:
            融合后的图像
        """
        height = len(source)
        width = len(source[0]) if height > 0 else 0
        
        result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        
        # 迭代求解泊松方程
        iterations = 50
        
        # 初始化
        for y in range(height):
            for x in range(width):
                for c in range(3):
                    result[y][x][c] = target[y][x][c]
        
        # 迭代优化
        for _ in range(iterations):
            new_result = [[[v for v in pixel] for pixel in row] for row in result]
            
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if mask[y][x] > 0.1:
                        for c in range(3):
                            # 拉普拉斯平滑
                            laplacian = (
                                result[y-1][x][c] + result[y+1][x][c] +
                                result[y][x-1][c] + result[y][x+1][c]
                            ) / 4.0
                            
                            # 添加源图像的梯度信息
                            source_grad = source[y][x][c] - (
                                source[y-1][x][c] + source[y+1][x][c] +
                                source[y][x-1][c] + source[y][x+1][c]
                            ) / 4.0
                            
                            new_val = laplacian + source_grad * 0.5
                            new_result[y][x][c] = int(max(0, min(255, new_val)))
            
            result = new_result
        
        return result
    
    def _seamless_clone(self, source: List[List[List[int]]], 
                        target: List[List[List[int]]], 
                        mask: List[List[float]],
                        center: Tuple[int, int]) -> List[List[List[int]]]:
        """
        无缝克隆
        
        Args:
            source: 源图像
            target: 目标图像
            mask: 克隆掩码
            center: 克隆中心位置
            
        Returns:
            克隆后的图像
        """
        height = len(target)
        width = len(target[0]) if height > 0 else 0
        
        result = [[[v for v in pixel] for pixel in row] for row in target]
        
        cy, cx = center
        src_height = len(source)
        src_width = len(source[0]) if src_height > 0 else 0
        
        # 计算偏移
        offset_y = cy - src_height // 2
        offset_x = cx - src_width // 2
        
        # 应用克隆
        for y in range(src_height):
            for x in range(src_width):
                ty = y + offset_y
                tx = x + offset_x
                
                if 0 <= ty < height and 0 <= tx < width:
                    m = mask[y][x] if y < len(mask) and x < len(mask[0]) else 0.0
                    if m > 0.1:
                        for c in range(3):
                            # 混合源和目标
                            blend = int(target[ty][tx][c] * (1.0 - m) + source[y][x][c] * m)
                            result[ty][tx][c] = max(0, min(255, blend))
        
        return result


class Outpainter:
    """
    图像外扩器
    
    实现图像外扩功能，扩展图像边界
    """
    
    def __init__(self, inpainter: Optional[Inpainter] = None):
        self._inpainter = inpainter or Inpainter()
    
    def outpaint(self, image: List[List[List[int]]], 
                 direction: str,
                 pixels: int,
                 prompt: str,
                 **kwargs) -> Dict[str, Any]:
        """
        外扩图像
        
        Args:
            image: 原始图像
            direction: 扩展方向 ('left', 'right', 'top', 'bottom', 'all')
            pixels: 扩展像素数
            prompt: 文本提示
            **kwargs: 其他参数
            
        Returns:
            外扩结果
        """
        # 扩展画布
        extended = self._extend_canvas(image, direction, pixels)
        
        # 创建外扩掩码
        mask = self._create_outpaint_mask(
            (len(image), len(image[0]) if image else 0),
            direction, pixels
        )
        
        # 填充扩展区域
        filled = self._fill_extended_region(extended, (len(image), len(image[0]) if image else 0), direction)
        
        # 使用修复器处理
        result = self._inpainter.inpaint(filled, mask, prompt, **kwargs)
        
        return result
    
    def _create_outpaint_mask(self, image_size: Tuple[int, int], 
                              direction: str, 
                              pixels: int) -> List[List[int]]:
        """
        创建外扩掩码
        
        Args:
            image_size: (height, width)
            direction: 扩展方向
            pixels: 扩展像素数
            
        Returns:
            外扩掩码
        """
        height, width = image_size
        
        # 计算新尺寸
        new_height = height
        new_width = width
        
        if direction in ['top', 'all']:
            new_height += pixels
        if direction in ['bottom', 'all']:
            new_height += pixels
        if direction in ['left', 'all']:
            new_width += pixels
        if direction in ['right', 'all']:
            new_width += pixels
        
        mask = [[0 for _ in range(new_width)] for _ in range(new_height)]
        
        # 标记扩展区域
        offset_y = pixels if direction in ['top', 'all'] else 0
        offset_x = pixels if direction in ['left', 'all'] else 0
        
        for y in range(new_height):
            for x in range(new_width):
                in_original = (
                    offset_y <= y < offset_y + height and
                    offset_x <= x < offset_x + width
                )
                if not in_original:
                    mask[y][x] = 255
        
        return mask
    
    def _extend_canvas(self, image: List[List[List[int]]], 
                       direction: str, 
                       pixels: int) -> List[List[List[int]]]:
        """
        扩展画布
        
        Args:
            image: 原始图像
            direction: 扩展方向
            pixels: 扩展像素数
            
        Returns:
            扩展后的图像
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 计算新尺寸
        new_height = height
        new_width = width
        top_pad = 0
        left_pad = 0
        
        if direction in ['top', 'all']:
            new_height += pixels
            top_pad = pixels
        if direction in ['bottom', 'all']:
            new_height += pixels
        if direction in ['left', 'all']:
            new_width += pixels
            left_pad = pixels
        if direction in ['right', 'all']:
            new_width += pixels
        
        # 创建扩展后的图像
        extended = [[[0, 0, 0] for _ in range(new_width)] for _ in range(new_height)]
        
        # 复制原始图像
        for y in range(height):
            for x in range(width):
                extended[y + top_pad][x + left_pad] = image[y][x][:]
        
        return extended
    
    def _fill_extended_region(self, extended: List[List[List[int]]], 
                              original_size: Tuple[int, int], 
                              direction: str) -> List[List[List[int]]]:
        """
        填充扩展区域
        
        使用边缘像素填充扩展区域作为初始值
        
        Args:
            extended: 扩展后的图像
            original_size: 原始图像尺寸
            direction: 扩展方向
            
        Returns:
            填充后的图像
        """
        height = len(extended)
        width = len(extended[0]) if height > 0 else 0
        orig_height, orig_width = original_size
        
        result = [row[:] for row in extended]
        
        pixels = (height - orig_height) // 2 if direction == 'all' else height - orig_height
        
        # 填充顶部
        if direction in ['top', 'all']:
            for y in range(pixels):
                for x in range(width):
                    src_y = pixels
                    if src_y < height:
                        result[y][x] = result[src_y][x][:]
        
        # 填充底部
        if direction in ['bottom', 'all']:
            for y in range(orig_height + pixels, height):
                for x in range(width):
                    src_y = orig_height + pixels - 1
                    if src_y >= 0:
                        result[y][x] = result[src_y][x][:]
        
        # 填充左侧
        if direction in ['left', 'all']:
            for y in range(height):
                for x in range(pixels):
                    src_x = pixels
                    if src_x < width:
                        result[y][x] = result[y][src_x][:]
        
        # 填充右侧
        if direction in ['right', 'all']:
            for y in range(height):
                for x in range(orig_width + pixels, width):
                    src_x = orig_width + pixels - 1
                    if src_x >= 0:
                        result[y][x] = result[y][src_x][:]
        
        return result


# 导出
__all__ = [
    'MaskProcessor',
    'Inpainter',
    'Outpainter',
]
