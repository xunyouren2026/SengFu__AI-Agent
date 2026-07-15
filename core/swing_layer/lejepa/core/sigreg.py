"""
SIGReg - 草图化各向同性高斯正则化

LeJEPA的核心创新，强制特征服从各向同性高斯分布，
从根本上防止表征坍缩。

核心约200行。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SIGReg(nn.Module):
    """
    草图化各向同性高斯正则化 (Sketched Isotropic Gaussian Regularization)
    
    通过随机投影和正态性检验，强制特征分布为各向同性高斯分布，
    从而防止JEPA架构中的表征坍缩问题。
    
    核心思想：
    1. 将高维特征随机投影到多个一维子空间
    2. 对每个投影进行正态性检验（偏度、峰度）
    3. 计算与标准正态分布的统计距离作为正则化损失
    """
    
    def __init__(
        self,
        num_projections: int = 64,
        normalize: bool = True,
        eps: float = 1e-6
    ):
        """
        初始化SIGReg
        
        Args:
            num_projections: 随机投影的数量
            normalize: 是否对特征进行归一化
            eps: 数值稳定性常数
        """
        super().__init__()
        self.num_projections = num_projections
        self.normalize = normalize
        self.eps = eps
        
        # 投影矩阵将在第一次前向传播时初始化
        self.register_buffer('projection_matrix', None)
        self._feature_dim = None
    
    def _init_projection_matrix(self, feature_dim: int, device: torch.device):
        """初始化随机投影矩阵"""
        self._feature_dim = feature_dim
        # 从标准正态分布采样投影矩阵
        projection = torch.randn(
            feature_dim, self.num_projections,
            device=device, dtype=torch.float32
        )
        # 归一化投影向量
        projection = projection / (torch.norm(projection, dim=0, keepdim=True) + self.eps)
        self.projection_matrix = projection
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算SIGReg损失
        
        Args:
            z: (batch_size, feature_dim)的特征张量
        
        Returns:
            loss: SIGReg正则化损失
            stats: 包含偏度、峰度等统计信息的字典
        """
        batch_size, feature_dim = z.shape
        
        # 初始化投影矩阵（如果尚未初始化）
        if self.projection_matrix is None or self._feature_dim != feature_dim:
            self._init_projection_matrix(feature_dim, z.device)
        
        # 可选：归一化特征
        if self.normalize:
            z = F.layer_norm(z, z.shape[-1:])
        
        # 1. 随机投影：将高维特征投影到多个一维子空间
        # projected: (batch_size, num_projections)
        projected = z @ self.projection_matrix
        
        # 2. 计算统计量
        stats = self._compute_statistics(projected)
        
        # 3. 计算SIGReg损失
        # 惩罚偏离标准正态分布的程度
        skewness_loss = torch.mean(stats['skewness'] ** 2)
        kurtosis_loss = torch.mean(stats['kurtosis'] ** 2)
        
        # 总损失
        loss = skewness_loss + kurtosis_loss
        
        # 附加统计信息
        stats.update({
            'sigreg_loss': loss.item(),
            'skewness_loss': skewness_loss.item(),
            'kurtosis_loss': kurtosis_loss.item(),
        })
        
        return loss, stats
    
    def _compute_statistics(self, x: torch.Tensor) -> dict:
        """
        计算投影后特征的统计量
        
        Args:
            x: (batch_size, num_projections)的投影特征
        
        Returns:
            包含均值、方差、偏度、峰度的字典
        """
        # 沿batch维度计算统计量
        # mean: (num_projections,)
        mean = torch.mean(x, dim=0)
        
        # 中心化
        x_centered = x - mean
        
        # 方差
        variance = torch.mean(x_centered ** 2, dim=0)
        std = torch.sqrt(variance + self.eps)
        
        # 标准化
        x_normalized = x_centered / (std + self.eps)
        
        # 偏度 (skewness)：衡量分布的不对称性
        # 标准正态分布的偏度为0
        skewness = torch.mean(x_normalized ** 3, dim=0)
        
        # 峰度 (kurtosis)：衡量分布的尾部厚度
        # 标准正态分布的峰度为3，这里计算超额峰度（减去3）
        kurtosis = torch.mean(x_normalized ** 4, dim=0) - 3.0
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }
    
    def check_collapse(self, z: torch.Tensor, threshold: float = 0.1) -> bool:
        """
        检查是否发生表征坍缩
        
        Args:
            z: 特征张量
            threshold: 坍缩检测阈值
        
        Returns:
            是否发生坍缩
        """
        with torch.no_grad():
            loss, stats = self.forward(z)
            # 如果损失低于阈值，认为发生坍缩
            return loss.item() < threshold
    
    def get_feature_distribution(self, z: torch.Tensor) -> dict:
        """
        获取特征分布的详细信息
        
        Args:
            z: 特征张量
        
        Returns:
            分布统计信息
        """
        with torch.no_grad():
            _, stats = self.forward(z)
        
        # 计算额外的分布指标
        feature_mean = torch.mean(z, dim=0)
        feature_std = torch.std(z, dim=0)
        
        return {
            'feature_mean': feature_mean.cpu().numpy(),
            'feature_std': feature_std.cpu().numpy(),
            'mean_feature_norm': torch.mean(torch.norm(z, dim=1)).item(),
            'max_feature_norm': torch.max(torch.norm(z, dim=1)).item(),
            'min_feature_norm': torch.min(torch.norm(z, dim=1)).item(),
            **{k: v.cpu().numpy() if torch.is_tensor(v) else v 
               for k, v in stats.items()},
        }


class SIGRegScheduler:
    """
    SIGReg正则化强度调度器
    
    动态调整SIGReg的权重，在训练初期较弱，后期增强。
    """
    
    def __init__(
        self,
        initial_weight: float = 0.01,
        final_weight: float = 1.0,
        warmup_steps: int = 1000,
        schedule_type: str = 'linear'
    ):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.current_step = 0
    
    def step(self) -> float:
        """获取当前权重并更新步数"""
        if self.current_step >= self.warmup_steps:
            weight = self.final_weight
        else:
            progress = self.current_step / self.warmup_steps
            if self.schedule_type == 'linear':
                weight = self.initial_weight + (self.final_weight - self.initial_weight) * progress
            elif self.schedule_type == 'cosine':
                weight = self.initial_weight + (self.final_weight - self.initial_weight) * \
                         (1 - math.cos(progress * math.pi)) / 2
            else:
                weight = self.initial_weight
        
        self.current_step += 1
        return weight
    
    def get_weight(self) -> float:
        """获取当前权重（不更新步数）"""
        if self.current_step >= self.warmup_steps:
            return self.final_weight
        
        progress = self.current_step / self.warmup_steps
        if self.schedule_type == 'linear':
            return self.initial_weight + (self.final_weight - self.initial_weight) * progress
        elif self.schedule_type == 'cosine':
            return self.initial_weight + (self.final_weight - self.initial_weight) * \
                   (1 - math.cos(progress * math.pi)) / 2
        return self.initial_weight


# 便捷函数
def compute_sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 64,
    return_stats: bool = False
) -> torch.Tensor:
    """
    便捷函数：计算SIGReg损失
    
    Args:
        z: 特征张量
        num_projections: 投影数量
        return_stats: 是否返回统计信息
    
    Returns:
        SIGReg损失（或(loss, stats)元组）
    """
    sigreg = SIGReg(num_projections=num_projections)
    loss, stats = sigreg(z)
    
    if return_stats:
        return loss, stats
    return loss
