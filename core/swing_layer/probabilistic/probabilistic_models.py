"""
AGI统一框架 - 概率编程与不确定性估计
实现变分推断、贝叶斯神经网络、蒙特卡洛Dropout等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


# ==================== 变分推断 ====================

class ReparameterizedGaussian(nn.Module):
    """重参数化的高斯分布"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(log_sigma)
        
        # 重参数化采样
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        
        # KL散度
        kl = 0.5 * (sigma.pow(2) + mu.pow(2) - 1 - 2 * log_sigma).sum(dim=-1)
        
        return z, kl


class VariationalEncoder(nn.Module):
    """变分自编码器编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        
        # 重参数化
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        
        return z, mu, log_var


class VariationalDecoder(nn.Module):
    """变分自编码器解码器"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    """变分自编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoder(latent_dim, hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z, mu, log_var = self.encoder(x)
        x_recon = self.decoder(z)
        
        # 重建损失
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return {
            'reconstruction': x_recon,
            'latent': z,
            'mu': mu,
            'log_var': log_var,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'loss': recon_loss + kl_loss
        }
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """从先验采样"""
        z = torch.randn(num_samples, self.encoder.mu_layer.out_features, device=device)
        return self.decoder(z)


# ==================== 贝叶斯神经网络 ====================

class BayesianLinear(nn.Module):
    """贝叶斯线性层"""
    
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 采样权重
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        # 前向传播
        output = F.linear(x, weight, bias)
        
        # KL散度
        kl_weight = self._kl_divergence(self.weight_mu, weight_sigma, self.prior_sigma)
        kl_bias = self._kl_divergence(self.bias_mu, bias_sigma, self.prior_sigma)
        kl = kl_weight + kl_bias
        
        return output, kl
    
    def _kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor,
                       prior_sigma: float) -> torch.Tensor:
        """计算KL散度"""
        prior_sigma_sq = prior_sigma ** 2
        sigma_sq = sigma ** 2
        
        kl = 0.5 * (
            (sigma_sq / prior_sigma_sq) +
            (mu ** 2 / prior_sigma_sq) - 
            1 +
            torch.log(prior_sigma_sq / sigma_sq)
        )
        
        return kl.sum()


class BayesianNeuralNetwork(nn.Module):
    """贝叶斯神经网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 prior_sigma: float = 1.0):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(BayesianLinear(dims[i], dims[i+1], prior_sigma))
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kl_total = 0.0
        
        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            kl_total = kl_total + kl
            
            if i < len(self.layers) - 1:
                x = F.relu(x)
                
        return x, kl_total
    
    def predict(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """预测（多次采样）"""
        predictions = []
        
        for _ in range(num_samples):
            pred, _ = self.forward(x)
            predictions.append(pred)
            
        predictions = torch.stack(predictions)
        
        return predictions.mean(dim=0), predictions.std(dim=0)


# ==================== 蒙特卡洛Dropout ====================

class MCDropout(nn.Module):
    """蒙特卡洛Dropout"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True)


class MCDropoutNetwork(nn.Module):
    """带MC Dropout的网络"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.dropout_layers.append(MCDropout(dropout_rate))
            
        self.output_layer = nn.Linear(dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = layer(x)
            x = F.relu(x)
            x = dropout(x)
            
        return self.output_layer(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                  num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """带不确定性估计的预测"""
        predictions = []
        
        for _ in range(num_samples):
            pred = self.forward(x)
            predictions.append(pred)
            
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


# ==================== 深度集成 ====================

class DeepEnsemble(nn.Module):
    """深度集成"""
    
    def __init__(self, model_class: type, model_args: Dict[str, Any],
                 num_models: int = 5):
        super().__init__()
        
        self.models = nn.ModuleList([
            model_class(**model_args) for _ in range(num_models)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """带不确定性估计的预测"""
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)
        
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std


# ==================== 正态化流 ====================

class AffineCouplingLayer(nn.Module):
    """仿射耦合层"""
    
    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        
        self.register_buffer('mask', mask)
        
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_x = x * self.mask
        
        scale = self.scale_net(masked_x) * (1 - self.mask)
        translate = self.translate_net(masked_x) * (1 - self.mask)
        
        y = masked_x + (1 - self.mask) * (x * torch.exp(scale) + translate)
        
        log_det = scale.sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_y = y * self.mask
        
        scale = self.scale_net(masked_y) * (1 - self.mask)
        translate = self.translate_net(masked_y) * (1 - self.mask)
        
        x = masked_y + (1 - self.mask) * ((y - translate) * torch.exp(-scale))
        
        log_det = -scale.sum(dim=-1)
        
        return x, log_det


class NormalizingFlow(nn.Module):
    """正态化流"""
    
    def __init__(self, dim: int, num_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        
        masks = []
        for i in range(num_layers):
            if i % 2 == 0:
                mask = torch.zeros(dim)
                mask[:dim//2] = 1
            else:
                mask = torch.zeros(dim)
                mask[dim//2:] = 1
            masks.append(mask)
        
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim, mask)
            for mask in masks
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = 0.0
        
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det
            
        return x, log_det_total
    
    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = 0.0
        
        for layer in reversed(self.layers):
            y, log_det = layer.inverse(y)
            log_det_total = log_det_total + log_det
            
        return y, log_det_total
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """计算对数概率"""
        z, log_det = self.forward(x)
        
        # 基础分布为标准正态
        log_prob_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        
        return log_prob_base + log_det
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """采样"""
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x


# ==================== 能量模型 ====================

class EnergyBasedModel(nn.Module):
    """能量模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_net(x)
    
    def negative_log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """负对数似然"""
        energy = self.forward(x)
        
        # 使用Langevin动力学估计配分函数
        # 这里简化处理
        return energy.mean()
    
    def langevin_dynamics(self, x: torch.Tensor, num_steps: int = 100,
                          step_size: float = 0.1, noise_scale: float = 0.01) -> torch.Tensor:
        """Langevin动力学采样"""
        x = x.clone().detach().requires_grad_(True)
        
        for _ in range(num_steps):
            energy = self.forward(x).sum()
            energy.backward()
            
            with torch.no_grad():
                grad = x.grad
                x = x - step_size * grad + noise_scale * torch.randn_like(x)
                x = x.detach().requires_grad_(True)
                
        return x.detach()


# ==================== 高斯过程 ====================

class RBFKernel(nn.Module):
    """RBF核"""
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        length_scale = torch.exp(self.log_length_scale)
        variance = torch.exp(self.log_variance)
        
        dist = torch.cdist(x1 / length_scale, x2 / length_scale)
        
        return variance * torch.exp(-0.5 * dist ** 2)


class SparseGaussianProcess(nn.Module):
    """稀疏高斯过程"""
    
    def __init__(self, input_dim: int, num_inducing: int = 100,
                 output_dim: int = 1):
        super().__init__()
        
        self.num_inducing = num_inducing
        self.output_dim = output_dim
        
        # 诱导点
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, input_dim))
        
        # 核函数
        self.kernel = RBFKernel()
        
        # 变分参数
        self.mu = nn.Parameter(torch.zeros(num_inducing, output_dim))
        self.L = nn.Parameter(torch.eye(num_inducing))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算核矩阵
        K_zz = self.kernel(self.inducing_points, self.inducing_points)
        K_xz = self.kernel(x, self.inducing_points)
        
        # Cholesky分解
        L_zz = torch.linalg.cholesky(K_zz + 1e-6 * torch.eye(self.num_inducing, device=x.device))
        
        # 预测
        alpha = torch.linalg.solve(L_zz, self.mu)
        alpha = torch.linalg.solve(L_zz.T, alpha)
        
        mean = K_xz @ alpha
        
        # 方差
        K_xx = self.kernel(x, x)
        v = torch.linalg.solve(L_zz, K_xz.T)
        var = K_xx.diag() - (v ** 2).sum(dim=0)
        var = var.unsqueeze(-1) + 1e-6
        
        return mean, var.sqrt()


# ==================== 工具函数 ====================

def elbo_loss(recon_x: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """ELBO损失"""
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
    """高斯负对数似然"""
    var = std ** 2
    nll = 0.5 * torch.log(var) + 0.5 * (target - mean) ** 2 / var
    return nll.sum(dim=-1)


def kl_divergence_normal(mu1: torch.Tensor, sigma1: torch.Tensor,
                         mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """两个高斯分布之间的KL散度"""
    kl = (
        torch.log(sigma2 / sigma1) +
        (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) -
        0.5
    )
    return kl.sum(dim=-1)
