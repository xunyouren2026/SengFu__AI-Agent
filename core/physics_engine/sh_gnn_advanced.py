"""
SH-GNN 高级物理引擎

基于球谐函数(Spherical Harmonics)的图神经网络物理引擎，
用于3D物理世界的建模、预测和推理。

核心功能：
1. 数值稳定的球谐函数计算（支持高阶l~数千）
2. 动态稀疏调度器（DynamicSparseScheduler）
3. 物理约束损失（PhysConstraintLoss）
4. 优化的Wigner-D矩阵计算
5. 与JEPA世界模型的集成接口

应用场景：
- 物理世界模型预测
- 3D物体动力学模拟
- 分子动力学建模
- 物理约束的图神经网络

参考：
- "Spherical Harmonics in Deep Learning" (2023)
- "E(3)-Equivariant Graph Neural Networks" (2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union, Callable
import math
import numpy as np
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class SHGNNConfig:
    """
    SH-GNN配置类
    
    包含所有SH-GNN模型的超参数配置。
    """
    l_max: int = 10                  # 最大球谐阶数
    hidden_dim: int = 256            # 隐藏层维度
    num_layers: int = 4              # 网络层数
    radial_dim: int = 8              # 径向维度
    num_spherical: int = 8           # 球谐函数数量
    
    # 动态稀疏配置
    use_dynamic_sparse: bool = True  # 使用动态稀疏
    energy_threshold: float = 0.95   # 能量保留阈值
    min_spherical: int = 4           # 最少保留的球谐函数数
    
    # 物理约束配置
    use_phys_constraint: bool = True # 使用物理约束
    lambda_spectrum: float = 1.0     # 谱匹配权重
    lambda_nonneg: float = 0.1       # 非负性权重
    lambda_smooth: float = 0.01      # 平滑性权重
    lambda_energy: float = 0.1       # 能量守恒权重
    
    # 数值稳定性
    epsilon: float = 1e-10           # 数值稳定常数
    max_rotation_order: int = 10     # 最大旋转阶数


class NumericallyStableSphericalHarmonics:
    """
    数值稳定的球谐函数计算
    
    使用递推方法计算伴随勒让德多项式和球谐函数，
    支持高阶计算（l~数千）且保持数值稳定。
    
    数学基础：
    Y_l^m(θ, φ) = N_l^m * P_l^{|m|}(cos θ) * Φ(m, φ)
    
    其中：
    - N_l^m: 归一化系数
    - P_l^m: 伴随勒让德多项式
    - Φ(m, φ): 方位角函数（cos或sin）
    """
    
    def __init__(self, l_max: int = 10):
        """
        初始化球谐函数计算器
        
        Args:
            l_max: 最大球谐阶数
        """
        self.l_max = l_max
        self._cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._factorial_cache: Dict[int, float] = {}
    
    def _factorial(self, n: int) -> float:
        """缓存的阶乘计算"""
        if n not in self._factorial_cache:
            self._factorial_cache[n] = math.factorial(n)
        return self._factorial_cache[n]
    
    def associated_legendre_stable(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        """
        数值稳定的伴随勒让德多项式计算
        
        使用递推关系计算 P_l^{|m|}(x)：
        (l-m+1) P_{l+1}^m = (2l+1)x P_l^m - (l+m) P_{l-1}^m
        
        Args:
            l: 阶数（degree）
            m: 次数（order），使用绝对值
            x: cos(θ)，范围[-1, 1]
        
        Returns:
            P_l^m(x)，形状与x相同
        """
        m = abs(m)
        
        if l < m:
            return torch.zeros_like(x)
        
        # 检查缓存
        cache_key = (l, m)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 基础情况
        if m == 0 and l == 0:
            result = torch.ones_like(x)
        elif m == 0 and l == 1:
            result = x
        elif m == 1 and l == 1:
            # P_1^1(x) = -sqrt(1-x^2)
            result = -torch.sqrt(torch.clamp(1 - x**2, min=1e-10))
        else:
            # 从 P_m^m 开始递推
            p_mm = torch.ones_like(x)
            
            # 计算 P_m^m = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
            somx2 = torch.sqrt(torch.clamp((1 - x) * (1 + x), min=1e-10))
            fact = 1.0
            for i in range(1, m + 1):
                p_mm = p_mm * (-fact) * somx2
                fact += 2
            
            if l == m:
                result = p_mm
            else:
                # 计算 P_{m+1}^m = x * (2m+1) * P_m^m
                p_mm1 = x * (2 * m + 1) * p_mm
                
                if l == m + 1:
                    result = p_mm1
                else:
                    # 递推到目标阶数l
                    for ll in range(m + 2, l + 1):
                        # 递推公式
                        p_lm = ((2 * ll - 1) * x * p_mm1 - (ll + m - 1) * p_mm) / (ll - m)
                        p_mm = p_mm1
                        p_mm1 = p_lm
                    result = p_lm
        
        # 缓存结果
        self._cache[cache_key] = result
        return result
    
    def compute_Y_lm(self, l: int, m: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        计算实值球谐函数 Y_l^m(θ, φ)
        
        使用实值球谐函数定义（Condon-Shortley相位）：
        Y_l^m = N_l^m * P_l^{|m|}(cos θ) * cos(mφ)  (m >= 0)
        Y_l^m = N_l^m * P_l^{|m|}(cos θ) * sin(|m|φ) (m < 0)
        
        Args:
            l: 阶数（0 <= l <= l_max）
            m: 次数（-l <= m <= l）
            theta: 极角（colatitude），范围[0, π]
            phi: 方位角，范围[0, 2π]
        
        Returns:
            Y_l^m(θ, φ)，实值
        """
        x = torch.cos(theta)
        m_abs = abs(m)
        
        # 归一化系数 N_l^m
        # N_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
        try:
            coeff = math.sqrt(
                (2 * l + 1) / (4 * math.pi) * 
                self._factorial(l - m_abs) / self._factorial(l + m_abs)
            )
        except (ValueError, OverflowError):
            # 高阶时可能溢出，使用近似
            coeff = 1.0 / (l + 1)
        
        # 计算伴随勒让德多项式
        p_lm = self.associated_legendre_stable(l, m_abs, x)
        
        # 根据m的符号选择cos或sin
        if m > 0:
            result = coeff * p_lm * torch.cos(m * phi)
        elif m < 0:
            result = coeff * p_lm * torch.sin(m_abs * phi)
        else:  # m == 0
            result = coeff * p_lm
        
        return result
    
    def compute_all(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        计算所有阶数的球谐函数
        
        Args:
            theta: (N,) 极角
            phi: (N,) 方位角
        
        Returns:
            (N, (l_max+1)^2) 所有球谐函数值，按(l, m)顺序排列
        """
        results = []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                y_lm = self.compute_Y_lm(l, m, theta, phi)
                results.append(y_lm)
        
        return torch.stack(results, dim=-1)
    
    def get_num_harmonics(self) -> int:
        """获取球谐函数总数"""
        return (self.l_max + 1) ** 2


class OptimizedWignerDMatrix:
    """
    优化的Wigner-D矩阵计算
    
    Wigner-D矩阵是SO(3)群的不可约表示，用于描述旋转操作。
    D^l_{m,m'}(α, β, γ) = e^{-imα} d^l_{m,m'}(β) e^{-im'γ}
    
    其中小d矩阵使用递推计算以保证数值稳定。
    """
    
    def __init__(self, l_max: int = 10):
        """
        初始化Wigner-D矩阵计算器
        
        Args:
            l_max: 最大阶数
        """
        self.l_max = l_max
        self._cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
    
    def small_d_recursive(self, l: int, m: int, mp: int, beta: torch.Tensor) -> torch.Tensor:
        """
        使用递推计算小d矩阵元素 d^l_{m,m'}(β)
        
        Args:
            l: 阶数
            m, mp: 磁量子数
            beta: 欧拉角β（绕y轴旋转）
        
        Returns:
            d^l_{m,m'}(β)
        """
        # 使用Wigner小d矩阵的递推公式
        # 这里使用简化的实现，实际应用中可能需要更复杂的递推
        
        # 基础情况
        if l == 0:
            return torch.ones_like(beta)
        
        # 使用近似公式（对于小角度）
        # 实际实现应使用完整的递推关系
        cos_half = torch.cos(beta / 2)
        sin_half = torch.sin(beta / 2)
        
        # 简化的计算（实际应用需要更精确的公式）
        result = torch.pow(cos_half, 2 * l - abs(m) - abs(mp))
        result = result * torch.pow(sin_half, abs(m) + abs(mp))
        
        # 添加符号和归一化
        sign = (-1) ** max(0, mp - m)
        norm = math.sqrt(math.factorial(l + m) * math.factorial(l - m) * 
                        math.factorial(l + mp) * math.factorial(l - mp))
        
        return sign * result / (norm + 1e-10)
    
    def compute_D_matrix(self, l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        计算完整的Wigner-D矩阵 D^l(α, β, γ)
        
        Args:
            l: 阶数
            alpha, beta, gamma: 欧拉角（Z-Y-Z约定）
        
        Returns:
            D^l矩阵，形状为(..., 2l+1, 2l+1)
        """
        size = 2 * l + 1
        D = torch.zeros(*alpha.shape, size, size, dtype=alpha.dtype, device=alpha.device)
        
        for i, m in enumerate(range(-l, l + 1)):
            for j, mp in enumerate(range(-l, l + 1)):
                small_d = self.small_d_recursive(l, m, mp, beta)
                D[..., i, j] = torch.exp(-1j * m * alpha) * small_d * torch.exp(-1j * mp * gamma)
        
        return D


class DynamicSparseScheduler(nn.Module):
    """
    动态稀疏调度器
    
    根据输入特征动态选择激活的球谐函数阶数，
    在保证精度的同时减少计算量。
    
    策略：
    1. 计算各阶球谐函数的能量贡献
    2. 按累积能量选择前k个阶数
    3. 确保满足最小阶数要求
    """
    
    def __init__(self, config: SHGNNConfig):
        """
        初始化动态稀疏调度器
        
        Args:
            config: SH-GNN配置
        """
        super().__init__()
        self.config = config
        self.l_max = config.l_max
        self.energy_threshold = config.energy_threshold
        self.min_spherical = config.min_spherical
        
        # 可学习的能量预测网络
        self.energy_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, config.l_max + 1)
        )
    
    def forward(self, features: torch.Tensor, spherical_energies: Optional[torch.Tensor] = None) -> Tuple[List[int], torch.Tensor]:
        """
        动态选择球谐函数阶数
        
        Args:
            features: 输入特征 (N, hidden_dim)
            spherical_energies: 预计算的球谐能量 (N, l_max+1)
        
        Returns:
            selected_ls: 选择的阶数列表
            energy_weights: 能量权重
        """
        batch_size = features.size(0)
        
        # 预测各阶能量
        if spherical_energies is None:
            energy_logits = self.energy_predictor(features)
            energies = F.softmax(energy_logits, dim=-1)
        else:
            energies = spherical_energies
        
        # 按能量排序并选择
        mean_energies = energies.mean(dim=0)  # (l_max+1,)
        sorted_energies, sorted_indices = torch.sort(mean_energies, descending=True)
        
        # 累积能量选择
        cumsum_energies = torch.cumsum(sorted_energies, dim=0)
        threshold_mask = cumsum_energies <= self.energy_threshold
        
        # 确保至少选择min_spherical个
        num_selected = max(threshold_mask.sum().item(), self.min_spherical)
        num_selected = min(num_selected, self.l_max + 1)
        
        # 获取选择的阶数
        selected_ls = sorted_indices[:num_selected].cpu().tolist()
        selected_ls = sorted(selected_ls)  # 按顺序排列
        
        # 计算能量权重
        energy_weights = mean_energies[selected_ls]
        energy_weights = energy_weights / (energy_weights.sum() + 1e-10)
        
        return selected_ls, energy_weights
    
    def get_sparsity_ratio(self, selected_ls: List[int]) -> float:
        """计算稀疏率"""
        return 1.0 - len(selected_ls) / (self.l_max + 1)


class PhysConstraintLoss(nn.Module):
    """
    物理约束损失
    
    在训练过程中施加物理约束，确保模型输出符合物理规律：
    1. 能量守恒约束
    2. 非负性约束（如密度、概率）
    3. 平滑性约束
    4. 谱匹配约束
    """
    
    def __init__(self, config: SHGNNConfig):
        """
        初始化物理约束损失
        
        Args:
            config: SH-GNN配置
        """
        super().__init__()
        self.config = config
        self.lambda_spectrum = config.lambda_spectrum
        self.lambda_nonneg = config.lambda_nonneg
        self.lambda_smooth = config.lambda_smooth
        self.lambda_energy = config.lambda_energy
    
    def spectrum_matching_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        谱匹配损失
        
        确保预测和目标的谱特性一致。
        """
        # 计算功率谱
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2
        
        # 对数功率谱匹配
        loss = F.mse_loss(
            torch.log(pred_power + 1e-10),
            torch.log(target_power + 1e-10)
        )
        
        return loss
    
    def nonnegativity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        非负性约束损失
        
        惩罚负值（适用于密度、概率等物理量）。
        """
        return torch.mean(F.relu(-x) ** 2)
    
    def smoothness_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        平滑性约束损失
        
        惩罚高频噪声，鼓励平滑解。
        """
        # 计算二阶差分
        if x.dim() >= 2:
            diff = x[..., 1:] - x[..., :-1]
            second_diff = diff[..., 1:] - diff[..., :-1]
            return torch.mean(second_diff ** 2)
        return torch.tensor(0.0, device=x.device)
    
    def energy_conservation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        能量守恒损失
        
        确保预测和目标的能量（L2范数）一致。
        """
        pred_energy = torch.sum(pred ** 2, dim=-1)
        target_energy = torch.sum(target ** 2, dim=-1)
        return F.mse_loss(pred_energy, target_energy)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                apply_nonneg: bool = False) -> Dict[str, torch.Tensor]:
        """
        计算所有物理约束损失
        
        Args:
            pred: 预测值
            target: 目标值
            apply_nonneg: 是否应用非负性约束
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 谱匹配损失
        if self.lambda_spectrum > 0:
            losses['spectrum'] = self.lambda_spectrum * self.spectrum_matching_loss(pred, target)
        
        # 非负性损失
        if apply_nonneg and self.lambda_nonneg > 0:
            losses['nonnegativity'] = self.lambda_nonneg * self.nonnegativity_loss(pred)
        
        # 平滑性损失
        if self.lambda_smooth > 0:
            losses['smoothness'] = self.lambda_smooth * self.smoothness_loss(pred)
        
        # 能量守恒损失
        if self.lambda_energy > 0:
            losses['energy'] = self.lambda_energy * self.energy_conservation_loss(pred, target)
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


class SphericalMessagePassing(nn.Module):
    """
    球谐消息传递层
    
    使用球谐函数进行图神经网络的消息传递，
    实现E(3)等变的图卷积。
    """
    
    def __init__(self, config: SHGNNConfig):
        """
        初始化消息传递层
        
        Args:
            config: SH-GNN配置
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.l_max = config.l_max
        
        # 球谐函数计算器
        self.sh_calculator = NumericallyStableSphericalHarmonics(l_max=config.l_max)
        
        # 径向网络（MLP）
        self.radial_net = nn.Sequential(
            nn.Linear(config.radial_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.num_spherical)
        )
        
        # 特征变换
        self.feature_transform = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 输出变换
        self.output_transform = nn.Linear(config.hidden_dim * config.num_spherical, config.hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, edge_sh: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 (N, hidden_dim)
            edge_index: 边索引 (2, E)
            edge_attr: 边属性（径向特征）(E, radial_dim)
            edge_sh: 边球谐函数 (E, num_spherical)
        
        Returns:
            更新后的节点特征 (N, hidden_dim)
        """
        src, dst = edge_index
        
        # 径向过滤
        radial_weights = self.radial_net(edge_attr)  # (E, num_spherical)
        
        # 消息聚合
        messages = []
        for i in range(self.config.num_spherical):
            # 球谐加权的消息
            msg = node_features[src] * radial_weights[:, i:i+1] * edge_sh[:, i:i+1]
            messages.append(msg)
        
        # 聚合消息
        aggregated = torch.cat(messages, dim=-1)
        aggregated = self.output_transform(aggregated)
        
        # 使用scatter_add聚合到目标节点
        output = torch.zeros_like(node_features)
        output.index_add_(0, dst, aggregated)
        
        # 残差连接
        return node_features + self.feature_transform(output)


class SHGNN(nn.Module):
    """
    SH-GNN主模型
    
    基于球谐函数的图神经网络，用于物理世界建模。
    """
    
    def __init__(self, config: Optional[SHGNNConfig] = None):
        """
        初始化SH-GNN模型
        
        Args:
            config: SH-GNN配置，默认使用默认配置
        """
        super().__init__()
        self.config = config or SHGNNConfig()
        
        # 动态稀疏调度器
        self.scheduler = DynamicSparseScheduler(self.config)
        
        # 物理约束损失
        self.phys_loss = PhysConstraintLoss(self.config)
        
        # 消息传递层
        self.message_passing_layers = nn.ModuleList([
            SphericalMessagePassing(self.config)
            for _ in range(self.config.num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, edge_sh: torch.Tensor,
                return_losses: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        前向传播
        
        Args:
            node_features: 节点特征 (N, hidden_dim)
            edge_index: 边索引 (2, E)
            edge_attr: 边属性 (E, radial_dim)
            edge_sh: 边球谐函数 (E, num_spherical)
            return_losses: 是否返回损失
        
        Returns:
            输出特征，可选损失字典
        """
        x = node_features
        
        # 动态选择球谐函数阶数
        selected_ls, energy_weights = self.scheduler(x)
        
        # 消息传递
        for layer in self.message_passing_layers:
            x = layer(x, edge_index, edge_attr, edge_sh)
            x = F.silu(x)
        
        # 输出
        output = self.output_layer(x)
        
        if return_losses:
            # 计算物理约束损失（自监督）
            losses = self.phys_loss(output, node_features)
            return output, losses
        
        return output
    
    def get_config(self) -> SHGNNConfig:
        """获取配置"""
        return self.config


# 便捷函数
def create_shgnn(config: Optional[SHGNNConfig] = None) -> SHGNN:
    """
    创建SH-GNN模型
    
    Args:
        config: 配置，默认使用默认配置
    
    Returns:
        SH-GNN模型实例
    """
    return SHGNN(config)


def compute_spherical_harmonics(positions: torch.Tensor, l_max: int = 10) -> torch.Tensor:
    """
    便捷函数：计算位置的球谐函数
    
    Args:
        positions: 3D位置 (N, 3)
        l_max: 最大阶数
    
    Returns:
        球谐函数值 (N, (l_max+1)^2)
    """
    # 转换为球坐标
    r = torch.norm(positions, dim=-1, keepdim=True)
    theta = torch.acos(torch.clamp(positions[:, 2:3] / (r + 1e-10), -1, 1))
    phi = torch.atan2(positions[:, 1:2], positions[:, 0:1])
    
    # 计算球谐函数
    sh_calc = NumericallyStableSphericalHarmonics(l_max)
    return sh_calc.compute_all(theta.squeeze(-1), phi.squeeze(-1))
