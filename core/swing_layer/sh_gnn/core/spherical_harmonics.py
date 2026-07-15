"""
球谐函数计算

基于scipy的特殊函数实现，约100行。
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import sph_harm
from typing import List, Tuple
import math


class SphericalHarmonics:
    """
    球谐函数计算类
    
    球谐函数是拉普拉斯方程在球坐标系下的解，
    构成了球面上函数的完备正交基。
    """
    
    def __init__(self, l_max: int = 4):
        """
        初始化
        
        Args:
            l_max: 最大阶数
        """
        self.l_max = l_max
        self._precomputed = {}
    
    def compute(self, l: int, m: int, theta: float, phi: float) -> complex:
        """
        计算球谐函数 Y_l^m(theta, phi)
        
        Args:
            l: 阶数（0 <= l <= l_max）
            m: 次数（-l <= m <= l）
            theta: 极角 [0, pi]
            phi: 方位角 [0, 2*pi]
        
        Returns:
            复数值
        """
        return sph_harm(m, l, phi, theta)
    
    def compute_batch(
        self,
        l: int,
        m: int,
        theta: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """
        批量计算球谐函数
        
        Args:
            theta: (N,)极角张量
            phi: (N,)方位角张量
        
        Returns:
            (N,)复数值张量
        """
        result = []
        for t, p in zip(theta.cpu().numpy(), phi.cpu().numpy()):
            y = self.compute(l, m, float(t), float(p))
            result.append(complex(y))
        
        return torch.tensor(result, dtype=torch.complex64, device=theta.device)
    
    def compute_all_orders(
        self,
        theta: float,
        phi: float
    ) -> List[complex]:
        """
        计算所有阶数的球谐函数
        
        Args:
            theta: 极角
            phi: 方位角
        
        Returns:
            扁平化列表，按(l, m)字典序排列
        """
        result = []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                y = self.compute(l, m, theta, phi)
                result.append(y)
        return result
    
    def get_num_coefficients(self) -> int:
        """获取系数总数"""
        return (self.l_max + 1) ** 2
    
    def cartesian_to_spherical(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        笛卡尔坐标转球坐标
        
        Args:
            x, y, z: 坐标张量
        
        Returns:
            (r, theta, phi)
        """
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        return r, theta, phi


class SphericalHarmonicsEmbedding(nn.Module):
    """
    球谐函数嵌入层
    
    将3D坐标编码为球谐系数。
    """
    
    def __init__(self, l_max: int = 4):
        super().__init__()
        self.l_max = l_max
        self.sh = SphericalHarmonics(l_max)
        self.num_coeffs = self.sh.get_num_coefficients()
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            coords: (N, 3)的3D坐标
        
        Returns:
            (N, num_coeffs)的球谐系数
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r, theta, phi = self.sh.cartesian_to_spherical(x, y, z)
        
        # 计算所有球谐系数
        coeffs = []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                y_lm = self.sh.compute_batch(l, m, theta, phi)
                coeffs.append(y_lm.real)
        
        return torch.stack(coeffs, dim=-1)
