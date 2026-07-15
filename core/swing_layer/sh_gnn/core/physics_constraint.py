"""
物理约束损失 - Physics Constraint Loss

基于物理定律的正则化，保证模型预测符合物理规律。

L_phys = λ1 * Σ w_l (Ĉ_l - C_l^theory)²      ← 谱匹配（Fisher加权）
       + λ2 * Σ ReLU(-Ĉ_l)                    ← 非负性强制
       + λ3 * Σ (Ĉ_{l-1} - 2Ĉ_l + Ĉ_{l+1})²   ← 平滑正则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失模块
    
    将物理定律编译到损失函数中，确保模型学习物理一致的表示。
    """
    
    def __init__(
        self,
        lambda_spectral: float = 1.0,
        lambda_nonnegativity: float = 0.5,
        lambda_smoothness: float = 0.3,
        lambda_energy_conservation: float = 0.2,
        use_fisher_weighting: bool = True,
        theoretical_model: Optional[str] = None
    ):
        """
        初始化物理约束损失
        
        Args:
            lambda_spectral: 谱匹配损失权重
            lambda_nonnegativity: 非负性约束权重
            lambda_smoothness: 平滑性约束权重
            lambda_energy_conservation: 能量守恒约束权重
            use_fisher_weighting: 是否使用Fisher信息加权
            theoretical_model: 理论模型类型（'cmb', 'turbulence', 'quantum'等）
        """
        super().__init__()
        self.lambda_spectral = lambda_spectral
        self.lambda_nonnegativity = lambda_nonnegativity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_energy_conservation = lambda_energy_conservation
        self.use_fisher_weighting = use_fisher_weighting
        self.theoretical_model = theoretical_model
        
        # Fisher信息矩阵（用于最优加权）
        self.register_buffer('fisher_weights', torch.ones(100))  # 最大支持100阶
        
    def forward(
        self,
        predicted_power: torch.Tensor,
        target_power: Optional[torch.Tensor] = None,
        sh_coeffs: Optional[torch.Tensor] = None,
        l_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算物理约束损失
        
        Args:
            predicted_power: (batch, l_max) 预测的功率谱
            target_power: (batch, l_max) 目标功率谱（可选）
            sh_coeffs: (batch, (l_max+1)^2) 球谐系数（用于能量计算）
            l_values: (l_max,) 阶数值
            
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=predicted_power.device)
        
        # 1. 谱匹配损失（与理论模型对比）
        if target_power is not None and self.lambda_spectral > 0:
            spectral_loss = self.spectral_matching_loss(
                predicted_power, target_power, l_values
            )
            total_loss += self.lambda_spectral * spectral_loss
            loss_dict['spectral'] = spectral_loss.item()
        
        # 2. 非负性约束
        if self.lambda_nonnegativity > 0:
            nonneg_loss = self.nonnegativity_loss(predicted_power)
            total_loss += self.lambda_nonnegativity * nonneg_loss
            loss_dict['nonnegativity'] = nonneg_loss.item()
        
        # 3. 平滑性约束
        if self.lambda_smoothness > 0:
            smooth_loss = self.smoothness_loss(predicted_power)
            total_loss += self.lambda_smoothness * smooth_loss
            loss_dict['smoothness'] = smooth_loss.item()
        
        # 4. 能量守恒约束
        if sh_coeffs is not None and self.lambda_energy_conservation > 0:
            energy_loss = self.energy_conservation_loss(sh_coeffs, predicted_power)
            total_loss += self.lambda_energy_conservation * energy_loss
            loss_dict['energy_conservation'] = energy_loss.item()
        
        loss_dict['total_physics'] = total_loss.item()
        return total_loss, loss_dict
    
    def spectral_matching_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        l_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        谱匹配损失（Fisher加权）
        
        使用Fisher信息矩阵进行最优加权，对高信噪比区域赋予更高权重。
        
        Args:
            predicted: (batch, l_max) 预测功率谱
            target: (batch, l_max) 目标功率谱
            l_values: (l_max,) 阶数值
            
        Returns:
            加权MSE损失
        """
        l_max = predicted.shape[1]
        
        # 计算Fisher权重
        if self.use_fisher_weighting:
            # Fisher信息 ~ 1/方差，对高信噪比区域更敏感
            with torch.no_grad():
                variance = torch.var(target, dim=0) + 1e-8
                fisher_w = 1.0 / variance
                fisher_w = fisher_w / torch.sum(fisher_w) * l_max  # 归一化
        else:
            fisher_w = torch.ones(l_max, device=predicted.device)
        
        # 加权MSE
        squared_error = (predicted - target) ** 2
        weighted_error = squared_error * fisher_w.unsqueeze(0)
        
        return torch.mean(weighted_error)
    
    def nonnegativity_loss(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        非负性约束
        
        功率谱必须非负（物理要求）。
        
        Args:
            power_spectrum: (batch, l_max) 功率谱
            
            Returns:
            ReLU(-C_l)的均值
        """
        return torch.mean(F.relu(-power_spectrum))
    
    def smoothness_loss(self, power_spectrum: torch.Tensor) -> torch.Tensor:
        """
        平滑性约束（二阶差分）
        
        物理功率谱通常是平滑的，惩罚剧烈变化。
        
        Args:
            power_spectrum: (batch, l_max) 功率谱
            
        Returns:
            二阶差分的L2范数
        """
        if power_spectrum.shape[1] < 3:
            return torch.tensor(0.0, device=power_spectrum.device)
        
        # 计算二阶差分: C_{l-1} - 2C_l + C_{l+1}
        second_diff = (
            power_spectrum[:, :-2] - 
            2 * power_spectrum[:, 1:-1] + 
            power_spectrum[:, 2:]
        )
        
        return torch.mean(second_diff ** 2)
    
    def energy_conservation_loss(
        self,
        sh_coeffs: torch.Tensor,
        power_spectrum: torch.Tensor
    ) -> torch.Tensor:
        """
        能量守恒约束（Parseval定理）
        
        球谐系数能量应等于功率谱积分。
        
        Args:
            sh_coeffs: (batch, (l_max+1)^2) 球谐系数
            power_spectrum: (batch, l_max) 功率谱
            
        Returns:
            能量守恒误差
        """
        batch_size = sh_coeffs.shape[0]
        l_max = power_spectrum.shape[1]
        
        # 从球谐系数计算能量
        idx = 0
        coeff_energy = torch.zeros(batch_size, l_max, device=sh_coeffs.device)
        for l in range(l_max):
            num_m = 2 * l + 1
            coeffs_l = sh_coeffs[:, idx:idx + num_m]
            coeff_energy[:, l] = torch.sum(coeffs_l ** 2, dim=1)
            idx += num_m
        
        # 功率谱能量（积分近似）
        l_values = torch.arange(l_max, device=power_spectrum.device, dtype=torch.float32)
        # C_l * (2l+1) / (4π) 的积分
        spectrum_energy = power_spectrum * (2 * l_values.unsqueeze(0) + 1) / (4 * np.pi)
        
        # 能量守恒误差
        energy_diff = torch.abs(coeff_energy - spectrum_energy)
        return torch.mean(energy_diff)
    
    def compute_theoretical_power_spectrum(
        self,
        l_values: torch.Tensor,
        model_type: str = 'cmb',
        params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        计算理论功率谱
        
        Args:
            l_values: (l_max,) 阶数值
            model_type: 模型类型
            params: 模型参数
            
        Returns:
            (l_max,) 理论功率谱
        """
        if model_type == 'cmb':
            return self._cmb_power_spectrum(l_values, params)
        elif model_type == 'turbulence':
            return self._turbulence_power_spectrum(l_values, params)
        elif model_type == 'quantum':
            return self._quantum_power_spectrum(l_values, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _cmb_power_spectrum(
        self,
        l_values: torch.Tensor,
        params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        CMB功率谱模型（简化版）
        
        基于标准ΛCDM模型的近似。
        """
        if params is None:
            params = {
                'A_s': 2.1e-9,  # 振幅
                'n_s': 0.965,   # 谱指数
                'l_pivot': 0.05 # 基准尺度
            }
        
        A_s = params['A_s']
        n_s = params['n_s']
        
        # 简化模型: C_l ∝ A_s * (l/2)^(n_s-1) * 振荡项
        l_norm = l_values / 2.0
        power = A_s * torch.pow(l_norm, n_s - 1)
        
        # 添加声学峰（简化）
        acoustic_peaks = 1 + 0.5 * torch.sin(l_values / 100 * np.pi)
        power = power * acoustic_peaks
        
        return power
    
    def _turbulence_power_spectrum(
        self,
        l_values: torch.Tensor,
        params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        湍流功率谱（Kolmogorov谱）
        
        E(k) ~ k^(-5/3)
        """
        if params is None:
            params = {'epsilon': 1.0, 'eta': 0.001}
        
        epsilon = params['epsilon']  # 能量耗散率
        eta = params['eta']          # Kolmogorov尺度
        
        # Kolmogorov谱
        k = l_values.float()
        power = epsilon ** (2/3) * torch.pow(k, -5/3)
        
        # 粘性截断
        power = power * torch.exp(-(k * eta) ** 2)
        
        return power
    
    def _quantum_power_spectrum(
        self,
        l_values: torch.Tensor,
        params: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        量子场论功率谱（简化）
        """
        if params is None:
            params = {'m': 1.0, 'H': 1.0}
        
        m = params['m']  # 质量
        H = params['H']  # Hubble参数
        
        # 简化模型
        k = l_values.float()
        power = torch.pow(k, 3) / (k ** 2 + (m/H) ** 2) ** 2
        
        return power


class PhysicalConsistencyChecker(nn.Module):
    """
    物理一致性检查器
    
    检查模型输出是否满足基本物理定律。
    """
    
    def __init__(self, tolerance: float = 1e-6):
        super().__init__()
        self.tolerance = tolerance
        
    def check_all(
        self,
        power_spectrum: torch.Tensor,
        sh_coeffs: Optional[torch.Tensor] = None
    ) -> Dict[str, bool]:
        """
        执行所有物理一致性检查
        
        Args:
            power_spectrum: 功率谱
            sh_coeffs: 球谐系数
            
        Returns:
            各检查项的通过状态
        """
        results = {}
        
        # 1. 非负性检查
        results['nonnegativity'] = self.check_nonnegativity(power_spectrum)
        
        # 2. 有限性检查
        results['finiteness'] = self.check_finiteness(power_spectrum)
        
        # 3. 能量守恒检查
        if sh_coeffs is not None:
            results['energy_conservation'] = self.check_energy_conservation(
                sh_coeffs, power_spectrum
            )
        
        # 4. 渐近行为检查
        results['asymptotic'] = self.check_asymptotic_behavior(power_spectrum)
        
        return results
    
    def check_nonnegativity(self, power_spectrum: torch.Tensor) -> bool:
        """检查功率谱非负"""
        return torch.all(power_spectrum >= -self.tolerance).item()
    
    def check_finiteness(self, power_spectrum: torch.Tensor) -> bool:
        """检查功率谱有限"""
        return torch.all(torch.isfinite(power_spectrum)).item()
    
    def check_energy_conservation(
        self,
        sh_coeffs: torch.Tensor,
        power_spectrum: torch.Tensor
    ) -> bool:
        """检查能量守恒"""
        # 简化的能量守恒检查
        l_max = min(int(np.sqrt(sh_coeffs.shape[1])) - 1, power_spectrum.shape[1])
        
        idx = 0
        coeff_energy = 0.0
        for l in range(l_max):
            num_m = 2 * l + 1
            coeffs_l = sh_coeffs[:, idx:idx + num_m]
            coeff_energy += torch.sum(coeffs_l ** 2).item()
            idx += num_m
        
        spectrum_energy = torch.sum(power_spectrum[:, :l_max]).item()
        
        relative_error = abs(coeff_energy - spectrum_energy) / (spectrum_energy + 1e-10)
        return relative_error < 0.1  # 允许10%误差
    
    def check_asymptotic_behavior(self, power_spectrum: torch.Tensor) -> bool:
        """检查渐近行为（高阶衰减）"""
        # 检查高阶是否衰减
        high_l_values = power_spectrum[:, -10:]  # 最后10个值
        low_l_values = power_spectrum[:, :10]    # 前10个值
        
        high_mean = torch.mean(high_l_values)
        low_mean = torch.mean(low_l_values)
        
        # 高阶应该比低阶小（或至少不发散）
        return (high_mean <= low_mean * 10).item()
