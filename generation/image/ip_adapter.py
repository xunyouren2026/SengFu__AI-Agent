"""
图像提示适配器模块

提供图像提示适配(IP-Adapter)和风格迁移功能
"""

import math
import random
from typing import List, Tuple, Any, Optional, Dict


class ImageEncoder:
    """
    图像编码器
    
    将图像编码为嵌入向量
    """
    
    def __init__(self, embed_dim: int = 768):
        self._embed_dim = embed_dim
    
    def encode(self, image: List[List[List[int]]]) -> List[float]:
        """
        编码图像为嵌入向量
        
        Args:
            image: RGB图像 [H][W][3]
            
        Returns:
            嵌入向量 [embed_dim]
        """
        # 分块
        patches = self._patchify(image, patch_size=16)
        
        # 提取特征
        features = self._extract_features(patches)
        
        # 全局平均池化
        embedding = self._global_average_pooling(features)
        
        return embedding
    
    def _patchify(self, image: List[List[List[int]]], 
                  patch_size: int = 16) -> List[List[List[List[int]]]]:
        """
        将图像分割为块
        
        Args:
            image: RGB图像
            patch_size: 块大小
            
        Returns:
            图像块列表
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        patches = []
        
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                patch = []
                for py in range(y, min(y + patch_size, height)):
                    row = []
                    for px in range(x, min(x + patch_size, width)):
                        row.append(image[py][px][:])
                    patch.append(row)
                patches.append(patch)
        
        return patches
    
    def _extract_features(self, patches: List[List[List[List[int]]]]) -> List[List[float]]:
        """
        提取特征
        
        Args:
            patches: 图像块列表
            
        Returns:
            特征列表
        """
        features = []
        
        for patch in patches:
            # 计算每个块的统计特征
            patch_features = self._compute_patch_features(patch)
            features.append(patch_features)
        
        return features
    
    def _compute_patch_features(self, patch: List[List[List[int]]]) -> List[float]:
        """
        计算块的特征
        
        Args:
            patch: 图像块
            
        Returns:
            特征向量
        """
        height = len(patch)
        width = len(patch[0]) if height > 0 else 0
        
        if height == 0 or width == 0:
            return [0.0] * 16
        
        # 计算颜色统计
        r_sum = g_sum = b_sum = 0
        r_sq = g_sq = b_sq = 0
        
        for y in range(height):
            for x in range(width):
                r, g, b = patch[y][x]
                r_sum += r
                g_sum += g
                b_sum += b
                r_sq += r * r
                g_sq += g * g
                b_sq += b * b
        
        count = height * width
        
        # 均值
        r_mean = r_sum / count
        g_mean = g_sum / count
        b_mean = b_sum / count
        
        # 标准差
        r_std = math.sqrt(max(0, r_sq / count - r_mean * r_mean))
        g_std = math.sqrt(max(0, g_sq / count - g_mean * g_mean))
        b_std = math.sqrt(max(0, b_sq / count - b_mean * b_mean))
        
        # 边缘特征 (简化版梯度)
        edge_h = 0
        edge_v = 0
        for y in range(height - 1):
            for x in range(width - 1):
                gray_y = sum(patch[y][x]) / 3
                gray_y1 = sum(patch[y + 1][x]) / 3
                gray_x1 = sum(patch[y][x + 1]) / 3
                edge_v += abs(gray_y - gray_y1)
                edge_h += abs(gray_y - gray_x1)
        
        edge_score = (edge_h + edge_v) / (count * 255.0 + 1e-6)
        
        # 组合特征
        features = [
            r_mean / 255.0, g_mean / 255.0, b_mean / 255.0,
            r_std / 255.0, g_std / 255.0, b_std / 255.0,
            edge_score,
            (r_mean + g_mean + b_mean) / (3 * 255.0),  # 亮度
            (r_mean - g_mean) / 255.0,  # RG差
            (g_mean - b_mean) / 255.0,  # GB差
            r_std / (r_mean + 1),  # 变异系数
            g_std / (g_mean + 1),
            b_std / (b_mean + 1),
            math.log(r_mean + 1) / math.log(256),  # 对数亮度
            math.log(g_mean + 1) / math.log(256),
            math.log(b_mean + 1) / math.log(256),
        ]
        
        return features
    
    def _global_average_pooling(self, features: List[List[float]]) -> List[float]:
        """
        全局平均池化
        
        Args:
            features: 特征列表
            
        Returns:
            池化后的嵌入向量
        """
        if not features:
            return [0.0] * self._embed_dim
        
        num_features = len(features[0])
        
        # 计算每个特征维度的平均值
        pooled = [0.0] * num_features
        for feat in features:
            for i, v in enumerate(feat):
                if i < num_features:
                    pooled[i] += v
        
        pooled = [v / len(features) for v in pooled]
        
        # 扩展到embed_dim维度
        if len(pooled) < self._embed_dim:
            # 重复和插值
            result = []
            repeat = self._embed_dim // len(pooled)
            remainder = self._embed_dim % len(pooled)
            
            for v in pooled:
                result.extend([v] * repeat)
            
            # 添加剩余维度
            for i in range(remainder):
                result.append(pooled[i % len(pooled)])
            
            return result[:self._embed_dim]
        else:
            return pooled[:self._embed_dim]


class IPAdapter:
    """
    图像提示适配器
    
    实现IP-Adapter功能，将参考图像作为生成条件
    """
    
    def __init__(self, image_encoder: Optional[ImageEncoder] = None):
        self._image_encoder = image_encoder or ImageEncoder(embed_dim=768)
        self._ip_layers: List[Dict[str, Any]] = []
        self._reference_images: List[Tuple[List[float], float]] = []  # (embedding, weight)
        self._init_ip_layers()
    
    def _init_ip_layers(self):
        """初始化IP-Adapter层"""
        # 模拟IP-Adapter层参数
        for i in range(12):  # 12层
            layer = {
                'layer_idx': i,
                'ip_proj_weight': [random.gauss(0, 0.02) for _ in range(768)],
                'ip_proj_bias': [random.gauss(0, 0.01) for _ in range(768)],
            }
            self._ip_layers.append(layer)
    
    def set_reference_image(self, image: List[List[List[int]]], weight: float = 1.0):
        """
        设置参考图像
        
        Args:
            image: 参考图像
            weight: 权重
        """
        embedding = self._encode_reference(image)
        self._reference_images = [(embedding, weight)]
    
    def add_reference_image(self, image: List[List[List[int]]], weight: float = 1.0):
        """
        添加参考图像
        
        Args:
            image: 参考图像
            weight: 权重
        """
        embedding = self._encode_reference(image)
        self._reference_images.append((embedding, weight))
    
    def clear_reference_images(self):
        """清除参考图像"""
        self._reference_images = []
    
    def _encode_reference(self, image: List[List[List[int]]]) -> List[float]:
        """
        编码参考图像
        
        Args:
            image: 参考图像
            
        Returns:
            编码后的嵌入向量
        """
        return self._image_encoder.encode(image)
    
    def _compute_ip_features(self, reference_embeds: List[Tuple[List[float], float]]) -> List[float]:
        """
        计算IP特征
        
        Args:
            reference_embeds: 参考图像嵌入列表
            
        Returns:
            IP特征
        """
        if not reference_embeds:
            return [0.0] * 768
        
        # 加权平均
        combined = [0.0] * 768
        total_weight = 0.0
        
        for embed, weight in reference_embeds:
            total_weight += weight
            for i, v in enumerate(embed):
                if i < 768:
                    combined[i] += v * weight
        
        if total_weight > 0:
            combined = [v / total_weight for v in combined]
        
        # 应用IP投影
        for layer in self._ip_layers:
            projected = []
            for i in range(768):
                val = combined[i] * layer['ip_proj_weight'][i] + layer['ip_proj_bias'][i]
                projected.append(max(0, val))  # ReLU
            combined = projected
        
        return combined
    
    def _inject_to_unet(self, unet_features: List, 
                        ip_features: List[float], 
                        scale: float) -> List:
        """
        注入IP特征到UNet
        
        Args:
            unet_features: UNet特征
            ip_features: IP特征
            scale: 注入强度
            
        Returns:
            注入后的特征
        """
        result = []
        
        for feat in unet_features:
            if isinstance(feat, (int, float)):
                # 简单的特征融合
                ip_signal = sum(ip_features) / len(ip_features) if ip_features else 0
                adjusted = feat + ip_signal * scale * 0.1
                result.append(adjusted)
            elif isinstance(feat, list):
                # 递归处理列表
                adjusted = self._inject_to_unet(feat, ip_features, scale)
                result.append(adjusted)
            else:
                result.append(feat)
        
        return result
    
    def _attention_with_ip(self, query: List[float], 
                           key: List[float], 
                           value: List[float],
                           ip_key: List[float], 
                           ip_value: List[float], 
                           scale: float) -> List[float]:
        """
        IP注意力机制
        
        Args:
            query: 查询向量
            key: 键向量
            value: 值向量
            ip_key: IP键向量
            ip_value: IP值向量
            scale: 注意力强度
            
        Returns:
            注意力输出
        """
        # 简化的注意力计算
        # 标准注意力
        standard_attn = []
        for i in range(min(len(query), len(key), len(value))):
            score = query[i] * key[i] / math.sqrt(len(query))
            standard_attn.append(score * value[i])
        
        # IP注意力
        ip_attn = []
        for i in range(min(len(query), len(ip_key), len(ip_value))):
            score = query[i] * ip_key[i] / math.sqrt(len(query))
            ip_attn.append(score * ip_value[i])
        
        # 合并
        result = []
        for i in range(len(standard_attn)):
            if i < len(ip_attn):
                val = standard_attn[i] * (1 - scale) + ip_attn[i] * scale
            else:
                val = standard_attn[i]
            result.append(val)
        
        return result
    
    def generate(self, prompt: str, 
                 reference_images: Optional[List[List[List[List[int]]]]] = None,
                 ip_scale: float = 1.0, 
                 **kwargs) -> Dict[str, Any]:
        """
        生成图像
        
        Args:
            prompt: 文本提示
            reference_images: 参考图像列表
            ip_scale: IP适配强度
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        # 设置参考图像
        if reference_images:
            self.clear_reference_images()
            for img in reference_images:
                self.add_reference_image(img)
        
        # 计算IP特征
        ip_features = self._compute_ip_features(self._reference_images)
        
        # 模拟生成过程
        steps = kwargs.get('steps', 20)
        height = kwargs.get('height', 512)
        width = kwargs.get('width', 512)
        
        # 初始化潜在变量
        latent = self._init_latent(height, width)
        
        # 文本编码
        text_embeds = self._encode_text(prompt)
        
        # 去噪过程
        for step in range(steps):
            t = 1.0 - step / steps
            
            # 标准去噪
            for y in range(len(latent)):
                for x in range(len(latent[y])):
                    noise = random.gauss(0, t * 0.1)
                    latent[y][x] = latent[y][x] * (1 - 0.1 * t) + noise
            
            # 注入IP特征
            if ip_scale > 0:
                flat_latent = [v for row in latent for v in row]
                adjusted = self._inject_to_unet(flat_latent, ip_features, ip_scale)
                
                # 恢复形状
                idx = 0
                for y in range(len(latent)):
                    for x in range(len(latent[y])):
                        if idx < len(adjusted):
                            latent[y][x] = adjusted[idx]
                            idx += 1
        
        # 解码为图像
        generated_image = self._latent_to_image(latent)
        
        return {
            'image': generated_image,
            'prompt': prompt,
            'ip_scale': ip_scale,
            'num_references': len(self._reference_images)
        }
    
    def _init_latent(self, height: int, width: int) -> List[List[float]]:
        """初始化潜在变量"""
        return [[random.gauss(0, 1) for _ in range(width // 8)] 
                for _ in range(height // 8)]
    
    def _encode_text(self, prompt: str) -> List[float]:
        """编码文本"""
        embed_dim = 768
        hash_val = sum(ord(c) for c in prompt)
        random.seed(hash_val)
        return [random.gauss(0, 0.1) for _ in range(embed_dim)]
    
    def _latent_to_image(self, latent: List[List[float]]) -> List[List[List[int]]]:
        """潜在变量转图像"""
        height = len(latent)
        width = len(latent[0]) if height > 0 else 0
        
        image = [[[0, 0, 0] for _ in range(width * 8)] for _ in range(height * 8)]
        
        for y in range(height * 8):
            for x in range(width * 8):
                ly = y // 8
                lx = x // 8
                if ly < height and lx < width:
                    val = int((latent[ly][lx] + 1.0) * 127.5)
                    val = max(0, min(255, val))
                    image[y][x] = [val, val, val]
        
        return image


class StyleTransfer:
    """
    风格迁移
    
    实现神经风格迁移算法
    """
    
    def __init__(self, content_weight: float = 1.0, style_weight: float = 10.0):
        self._content_weight = content_weight
        self._style_weight = style_weight
    
    def transfer(self, content_image: List[List[List[int]]], 
                 style_image: List[List[List[int]]], 
                 iterations: int = 100) -> Dict[str, Any]:
        """
        风格迁移
        
        Args:
            content_image: 内容图像
            style_image: 风格图像
            iterations: 优化迭代次数
            
        Returns:
            迁移结果
        """
        # 提取内容特征
        content_features = self._extract_features(content_image)
        
        # 提取风格特征
        style_features = self._extract_features(style_image)
        
        # 计算风格Gram矩阵
        style_gram = self._compute_gram_matrix(style_features)
        
        # 初始化生成图像 (从内容图像开始)
        generated = [[[c for c in pixel] for pixel in row] for row in content_image]
        
        # 优化
        generated = self._optimize(generated, content_features, style_gram, iterations)
        
        return {
            'image': generated,
            'content_weight': self._content_weight,
            'style_weight': self._style_weight,
            'iterations': iterations
        }
    
    def _extract_features(self, image: List[List[List[int]]]) -> List[List[List[float]]]:
        """
        提取图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            特征图
        """
        height = len(image)
        width = len(image[0]) if height > 0 else 0
        
        # 简化的特征提取 (多尺度)
        features = []
        
        # 原始尺度
        for y in range(height):
            row = []
            for x in range(width):
                r, g, b = image[y][x]
                # RGB特征
                row.append([r / 255.0, g / 255.0, b / 255.0])
            features.append(row)
        
        return features
    
    def _compute_gram_matrix(self, features: List[List[List[float]]]) -> List[List[float]]:
        """
        计算Gram矩阵
        
        Args:
            features: 特征图
            
        Returns:
            Gram矩阵
        """
        height = len(features)
        width = len(features[0]) if height > 0 else 0
        channels = len(features[0][0]) if width > 0 else 0
        
        # 将特征展平
        flat_features = []
        for y in range(height):
            for x in range(width):
                flat_features.append(features[y][x])
        
        # 计算Gram矩阵
        gram = [[0.0 for _ in range(channels)] for _ in range(channels)]
        
        for i in range(channels):
            for j in range(channels):
                value = 0.0
                for feat in flat_features:
                    if i < len(feat) and j < len(feat):
                        value += feat[i] * feat[j]
                gram[i][j] = value / (height * width)
        
        return gram
    
    def _compute_content_loss(self, generated: List[List[List[float]]], 
                              content: List[List[List[float]]]) -> float:
        """
        计算内容损失
        
        Args:
            generated: 生成图像特征
            content: 内容图像特征
            
        Returns:
            内容损失
        """
        loss = 0.0
        count = 0
        
        for y in range(min(len(generated), len(content))):
            for x in range(min(len(generated[y]), len(content[y]))):
                for c in range(min(len(generated[y][x]), len(content[y][x]))):
                    diff = generated[y][x][c] - content[y][x][c]
                    loss += diff * diff
                    count += 1
        
        return loss / (count + 1e-6)
    
    def _compute_style_loss(self, generated: List[List[List[float]]], 
                            style_gram: List[List[float]]) -> float:
        """
        计算风格损失
        
        Args:
            generated: 生成图像特征
            style_gram: 风格Gram矩阵
            
        Returns:
            风格损失
        """
        # 计算生成图像的Gram矩阵
        gen_gram = self._compute_gram_matrix(generated)
        
        # 计算Frobenius范数差异
        loss = 0.0
        count = 0
        
        for i in range(min(len(gen_gram), len(style_gram))):
            for j in range(min(len(gen_gram[i]), len(style_gram[i]))):
                diff = gen_gram[i][j] - style_gram[i][j]
                loss += diff * diff
                count += 1
        
        return loss / (count + 1e-6)
    
    def _optimize(self, generated: List[List[List[int]]], 
                  content: List[List[List[float]]], 
                  style_gram: List[List[float]], 
                  iterations: int) -> List[List[List[int]]]:
        """
        优化生成图像
        
        Args:
            generated: 初始生成图像
            content: 内容特征
            style_gram: 风格Gram矩阵
            iterations: 迭代次数
            
        Returns:
            优化后的图像
        """
        # 转换为浮点数
        result = [[[float(c) / 255.0 for c in pixel] for pixel in row] for row in generated]
        
        learning_rate = 0.1
        
        for iter_idx in range(iterations):
            # 提取当前特征
            current_features = result
            
            # 计算损失
            content_loss = self._compute_content_loss(current_features, content)
            style_loss = self._compute_style_loss(current_features, style_gram)
            
            total_loss = (self._content_weight * content_loss + 
                         self._style_weight * style_loss)
            
            # 简化的梯度下降 (使用有限差分近似)
            if iter_idx < iterations - 1:
                for y in range(len(result)):
                    for x in range(len(result[y])):
                        for c in range(3):
                            # 简化的梯度计算
                            gradient = 0.0
                            
                            # 内容梯度 (向内容图像靠近)
                            if y < len(content) and x < len(content[y]) and c < len(content[y][x]):
                                content_grad = result[y][x][c] - content[y][x][c]
                                gradient += self._content_weight * content_grad
                            
                            # 风格梯度 (简化的风格匹配)
                            style_grad = random.gauss(0, 0.01) * self._style_weight * 0.1
                            gradient += style_grad
                            
                            # 更新
                            result[y][x][c] -= learning_rate * gradient
                            result[y][x][c] = max(0.0, min(1.0, result[y][x][c]))
            
            # 衰减学习率
            if iter_idx % 20 == 0:
                learning_rate *= 0.9
        
        # 转换回整数
        return [[[int(max(0, min(255, c * 255))) for c in pixel] for pixel in row] for row in result]


# 导出
__all__ = [
    'ImageEncoder',
    'IPAdapter',
    'StyleTransfer',
]
