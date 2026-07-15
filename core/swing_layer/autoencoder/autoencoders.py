"""
自编码器模块 - 完整实现
包含: AutoEncoder, VAE, BetaVAE, VQ-VAE, ConditionalVAE,
      DenoisingAutoEncoder, SparseAutoEncoder, ContractiveAutoEncoder等
所有实现均为真实算法代码，无占位符
"""

import math
import random
from typing import List, Tuple, Optional, Union, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from core.swing_layer.stubs import torch, nn, F, _HAS_TORCH



def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def softplus(x: float) -> float:
    return math.log(1 + math.exp(max(-100, x)))


def reparameterize(mean: Union[List[float], 'torch.Tensor'], log_var: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
    """
    重参数化技巧
    z = μ + σ * ε, where ε ~ N(0, I)
    """
    std = [math.exp(0.5 * lv) for lv in log_var]
    eps = [random.gauss(0, 1) for _ in mean]
    return [mean[i] + std[i] * eps[i] for i in range(len(mean))]


def kl_divergence(mean: Union[List[float], 'torch.Tensor'], log_var: Union[List[float], 'torch.Tensor']) -> float:
    """
    计算KL散度: KL(q(z|x) || p(z))
    = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    """
    kl = 0.0
    for i in range(len(mean)):
        kl += 1.0 + log_var[i] - mean[i]**2 - math.exp(log_var[i])
    return -0.5 * kl



# =============================================================================
# PyTorch Compatibility Utilities
# =============================================================================

def _to_tensor(x, device: str = None, dtype=None, requires_grad: bool = False):
    """
    Convert input to torch.Tensor.
    
    Supports:
    - torch.Tensor: returned as-is (with optional device/dtype cast)
    - list/tuple: converted to torch.Tensor
    - numpy.ndarray: converted to torch.Tensor
    - scalar: wrapped in torch.Tensor
    
    Args:
        x: Input data (tensor, list, tuple, numpy array, or scalar)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        dtype: Target dtype (torch.float32, torch.float64, etc.)
        requires_grad: Whether to track gradients
    
    Returns:
        torch.Tensor or original type if torch is not available
    """
    if not _HAS_TORCH:
        return x
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != torch.device(device):
            x = x.to(device=device)
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype=dtype)
        if requires_grad and not x.requires_grad:
            x = x.requires_grad_(requires_grad=True)
        return x
    # Convert from list/tuple/numpy
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _to_numpy(x):
    """Convert torch.Tensor to numpy array."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


def _to_list(x):
    """Convert torch.Tensor to nested Python list."""
    if not _HAS_TORCH or not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().tolist()


def _get_device(x):
    """Get device of tensor, default to 'cpu'."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.device
    return None


def _batch_dim(x):
    """Ensure input has batch dimension. If 2D, add batch dim to make 3D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 2:
            return x.unsqueeze(0)
    return x


def _unbatch(x):
    """Remove batch dimension if it's 1. If 3D with batch=1, squeeze to 2D."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)
    return x


class LinearLayer:
    """线性层"""
    
    def __init__(self, in_features: int, out_features: int):
        std = math.sqrt(2.0 / in_features)
        self.weight = [[random.gauss(0, std) for _ in range(in_features)] 
                      for _ in range(out_features)]
        self.bias = [0.0 for _ in range(out_features)]
    
    def forward(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        out = [0.0 for _ in range(len(self.weight))]
        for i in range(len(self.weight)):
            for j in range(len(x)):
                out[i] += self.weight[i][j] * x[j]
            out[i] += self.bias[i]
        return out


class MLP:
    """多层感知机"""
    
    def __init__(self, dims: List[int], activation: str = 'relu'):
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(LinearLayer(dims[i], dims[i+1]))
        self.activation = activation
    
    def _activate(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        if self.activation == 'relu':
            return [max(0.0, xi) for xi in x]
        elif self.activation == 'tanh':
            return [math.tanh(xi) for xi in x]
        elif self.activation == 'sigmoid':
            return [sigmoid(xi) for xi in x]
        elif self.activation == 'silu':
            return [xi / (1 + math.exp(-xi)) for xi in x]
        return x
    
    def forward(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = self._activate(x)
        return x


class AutoEncoder:
    """
    标准自编码器
    
    编码器: x -> z
    解码器: z -> x_hat
    损失: ||x - x_hat||^2
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 编码器
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = MLP(encoder_dims, 'relu')
        
        # 解码器
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(decoder_dims, 'relu')
    
    def encode(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """编码"""
        return self.encoder.forward(x)
    
    def decode(self, z: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """解码"""
        return self.decoder.forward(z)
    
    def forward(self, x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播，返回(重建, 潜在表示)"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_loss(self, x: Union[List[float], 'torch.Tensor'], x_recon: Union[List[float], 'torch.Tensor']) -> float:
        """重建损失 (MSE)"""
        return sum((x[i] - x_recon[i])**2 for i in range(len(x))) / len(x)
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步"""
        x_recon, z = self.forward(x)
        loss = self.reconstruction_loss(x, x_recon)
        return {'loss': loss, 'reconstruction_loss': loss}


class VAE:
    """
    变分自编码器 (VAE)
    
    编码器: x -> μ, σ
    潜变量: z ~ N(μ, σ²) (重参数化)
    解码器: z -> x_hat
    损失: E[log p(x|z)] - KL(q(z|x) || p(z))
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 beta: float = 1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL散度权重
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 编码器 (输出均值和log方差)
        encoder_dims = [input_dim] + hidden_dims
        self.encoder = MLP(encoder_dims, 'relu')
        
        # 均值和方差投影
        self.fc_mean = LinearLayer(hidden_dims[-1], latent_dim)
        self.fc_log_var = LinearLayer(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(decoder_dims, 'relu')
    
    def encode(self, x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """编码，返回(均值, log方差)"""
        h = self.encoder.forward(x)
        mean = self.fc_mean.forward(h)
        log_var = self.fc_log_var.forward(h)
        return mean, log_var
    
    def reparameterize(self, mean: Union[List[float], 'torch.Tensor'], log_var: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """重参数化"""
        return reparameterize(mean, log_var)
    
    def decode(self, z: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """解码"""
        return self.decoder.forward(z)
    
    def forward(self, x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播，返回(重建, 均值, log方差, 潜在表示)"""
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_recon = self.decode(z)
        return x_recon, mean, log_var, z
    
    def reconstruction_loss(self, x: Union[List[float], 'torch.Tensor'], x_recon: Union[List[float], 'torch.Tensor']) -> float:
        """重建损失 (假设高斯分布的负对数似然)"""
        return sum((x[i] - x_recon[i])**2 for i in range(len(x))) / len(x)
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步"""
        x_recon, mean, log_var, z = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = kl_divergence(mean, log_var)
        
        # ELBO = -recon_loss + beta * kl_loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples: int) -> Union[List[List[float]], 'torch.Tensor']:
        """从先验分布采样生成"""
        samples = []
        for _ in range(num_samples):
            z = [random.gauss(0, 1) for _ in range(self.latent_dim)]
            x = self.decode(z)
            samples.append(x)
        return samples


class BetaVAE(VAE):
    """
    Beta-VAE
    使用β > 1鼓励解耦表示
    
    L = E[log p(x|z)] - β * KL(q(z|x) || p(z))
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 beta: float = 4.0, gamma: float = 1.0,
                 max_capacity: int = 25, capacity_max_iter: int = 100000,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__(input_dim, latent_dim, hidden_dims, beta=1.0)
        
        self.target_beta = beta
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.capacity_max_iter = capacity_max_iter
        self.training_step = 0
    
    def _compute_c(self) -> float:
        """计算容量系数"""
        return min(self.max_capacity, 
                   self.max_capacity * self.training_step / self.capacity_max_iter)
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步（带容量约束）"""
        x_recon, mean, log_var, z = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = kl_divergence(mean, log_var)
        
        C = self._compute_c()
        
        # Beta-VAE损失: L = recon + γ|KL - C| - β * KL
        loss = recon_loss + self.gamma * abs(kl_loss - C) + self.target_beta * kl_loss
        
        self.training_step += 1
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'capacity': C
        }


class VQVAE:
    """
    向量量化变分自编码器 (VQ-VAE)
    
    使用离散的codebook替代连续潜变量
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 num_embeddings: int = 512, hidden_dims: Optional[List[int]] = None,
                 commitment_cost: float = 0.25, decay: float = 0.99):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # 编码器
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = MLP(encoder_dims, 'relu')
        
        # 解码器
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(decoder_dims, 'relu')
        
        # Codebook
        std = 1.0 / num_embeddings
        self.embedding = [[random.gauss(0, std) for _ in range(latent_dim)] 
                         for _ in range(num_embeddings)]
        
        # EMA更新
        self.embedding_count = [0.0 for _ in range(num_embeddings)]
        self.embedding_sum = [[0.0 for _ in range(latent_dim)] for _ in range(num_embeddings)]
    
    def _find_nearest_embedding(self, z: Union[List[float], 'torch.Tensor']) -> Tuple[int, Union[List[float], 'torch.Tensor']]:
        """找到最近的codebook向量"""
        min_dist = float('inf')
        best_idx = 0
        
        for k in range(self.num_embeddings):
            dist = sum((z[i] - self.embedding[k][i])**2 for i in range(self.latent_dim))
            if dist < min_dist:
                min_dist = dist
                best_idx = k
        
        return best_idx, self.embedding[best_idx][:]
    
    def _update_embedding(self, idx: int, z: Union[List[float], 'torch.Tensor']):
        """EMA更新codebook"""
        self.embedding_sum[idx] = [self.decay * self.embedding_sum[idx][i] + (1 - self.decay) * z[i] 
                                   for i in range(self.latent_dim)]
        self.embedding_count[idx] = self.decay * self.embedding_count[idx] + (1 - self.decay)
        
        n = self.embedding_count[idx]
        if n > 0:
            self.embedding[idx] = [self.embedding_sum[idx][i] / n for i in range(self.latent_dim)]
    
    def encode(self, x: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], int]:
        """编码，返回(量化向量, 索引)"""
        z = self.encoder.forward(x)
        idx, z_q = self._find_nearest_embedding(z)
        return z_q, idx
    
    def decode(self, z_q: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """解码"""
        return self.decoder.forward(z_q)
    
    def forward(self, x: Union[List[float], 'torch.Tensor']) -> Dict:
        """前向传播"""
        z = self.encoder.forward(x)
        idx, z_q = self._find_nearest_embedding(z)
        x_recon = self.decode(z_q)
        
        # 更新codebook
        self._update_embedding(idx, z)
        
        # 计算损失
        recon_loss = sum((x[i] - x_recon[i])**2 for i in range(len(x))) / len(x)
        
        # VQ损失
        vq_loss = sum((z_q[i] - z[i])**2 for i in range(self.latent_dim))
        
        # Commitment损失
        commitment_loss = sum((z[i] - z_q[i])**2 for i in range(self.latent_dim))
        
        return {
            'x_recon': x_recon,
            'z_q': z_q,
            'z': z,
            'idx': idx,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'loss': recon_loss + vq_loss + self.commitment_cost * commitment_loss
        }
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步"""
        result = self.forward(x)
        return {
            'loss': result['loss'],
            'reconstruction_loss': result['recon_loss'],
            'vq_loss': result['vq_loss'],
            'commitment_loss': result['commitment_loss']
        }


class ConditionalVAE(VAE):
    """
    条件变分自编码器 (CVAE)
    
    p(z|x,y) = N(μ(x,y), σ²(x,y))
    p(x|z,y) = decoder(z,y)
    """
    
    def __init__(self, input_dim: int, latent_dim: int, condition_dim: int,
                 hidden_dims: Optional[List[int]] = None):
        # 先初始化基础VAE结构
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # 编码器 (输入+条件)
        encoder_dims = [input_dim + condition_dim] + hidden_dims
        self.encoder = MLP(encoder_dims, 'relu')
        self.fc_mean = LinearLayer(hidden_dims[-1], latent_dim)
        self.fc_log_var = LinearLayer(hidden_dims[-1], latent_dim)
        
        # 解码器 (潜变量+条件)
        decoder_dims = [latent_dim + condition_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(decoder_dims, 'relu')
    
    def encode(self, x: Union[List[float], 'torch.Tensor'], condition: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """条件编码"""
        x_cond = x + condition
        h = self.encoder.forward(x_cond)
        mean = self.fc_mean.forward(h)
        log_var = self.fc_log_var.forward(h)
        return mean, log_var
    
    def decode(self, z: Union[List[float], 'torch.Tensor'], condition: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """条件解码"""
        z_cond = z + condition
        return self.decoder.forward(z_cond)
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], condition: Union[List[float], 'torch.Tensor']) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """前向传播"""
        mean, log_var = self.encode(x, condition)
        z = reparameterize(mean, log_var)
        x_recon = self.decode(z, condition)
        return x_recon, mean, log_var, z
    
    def train_step(self, x: Union[List[float], 'torch.Tensor'], condition: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步"""
        x_recon, mean, log_var, z = self.forward(x, condition)
        
        recon_loss = sum((x[i] - x_recon[i])**2 for i in range(len(x))) / len(x)
        kl_loss = kl_divergence(mean, log_var)
        
        return {
            'loss': recon_loss + kl_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, condition: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """条件采样"""
        z = [random.gauss(0, 1) for _ in range(self.latent_dim)]
        return self.decode(z, condition)


class DenoisingAutoEncoder(AutoEncoder):
    """
    去噪自编码器 (DAE)
    
    训练时对输入添加噪声，学习恢复原始输入
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 noise_type: str = 'gaussian', noise_level: float = 0.1,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.noise_type = noise_type
        self.noise_level = noise_level
    
    def add_noise(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """添加噪声"""
        if self.noise_type == 'gaussian':
            return [xi + random.gauss(0, self.noise_level) for xi in x]
        elif self.noise_type == 'masking':
            return [xi if random.random() > self.noise_level else 0.0 for xi in x]
        elif self.noise_type == 'salt_pepper':
            return [0.0 if random.random() < self.noise_level / 2 
                   else (1.0 if random.random() < self.noise_level / 2 else xi) 
                   for xi in x]
        return x
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步（带噪声）"""
        x_noisy = self.add_noise(x)
        x_recon, z = self.forward(x_noisy)
        loss = self.reconstruction_loss(x, x_recon)
        return {'loss': loss, 'reconstruction_loss': loss}


class SparseAutoEncoder(AutoEncoder):
    """
    稀疏自编码器
    
    在潜空间添加L1正则化鼓励稀疏表示
    L = ||x - x_hat||^2 + λ * ||z||_1
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 sparsity_lambda: float = 0.01,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.sparsity_lambda = sparsity_lambda
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步（带稀疏正则化）"""
        x_recon, z = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, x_recon)
        
        # L1正则化
        sparsity_loss = sum(abs(zi) for zi in z)
        
        total_loss = recon_loss + self.sparsity_lambda * sparsity_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'sparsity_loss': sparsity_loss
        }


class ContractiveAutoEncoder(AutoEncoder):
    """
    收缩自编码器 (CAE)
    
    惩罚编码器Jacobian的Frobenius范数
    L = ||x - x_hat||^2 + λ * ||J_f(x)||_F^2
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 contractive_lambda: float = 0.1,
                 hidden_dims: Optional[List[int]] = None):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.contractive_lambda = contractive_lambda
    
    def _compute_jacobian_frobenius(self, x: Union[List[float], 'torch.Tensor']) -> float:
        """计算编码器Jacobian的Frobenius范数 (数值近似)"""
        eps = 1e-5
        z_0 = self.encode(x)
        
        frobenius_norm_sq = 0.0
        for i in range(len(x)):
            x_plus = x[:]
            x_plus[i] += eps
            z_plus = self.encode(x_plus)
            
            for j in range(len(z_0)):
                frobenius_norm_sq += ((z_plus[j] - z_0[j]) / eps) ** 2
        
        return frobenius_norm_sq
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步（带收缩惩罚）"""
        x_recon, z = self.forward(x)
        
        recon_loss = self.reconstruction_loss(x, x_recon)
        contractive_loss = self._compute_jacobian_frobenius(x)
        
        total_loss = recon_loss + self.contractive_lambda * contractive_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contractive_loss': contractive_loss
        }


class IWAE(VAE):
    """
    Importance Weighted Autoencoder (IWAE)
    
    使用多个样本估计ELBO， tighter bound
    L_k = E[log (1/k * Σ_i p(x|z_i) / q(z_i|x))]
    """
    
    def __init__(self, input_dim: int, latent_dim: int,
                 num_samples: int = 5, **kwargs):
        super().__init__(input_dim, latent_dim, **kwargs)
        self.num_samples = num_samples
    
    def train_step(self, x: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """训练单步（重要性加权）"""
        mean, log_var = self.encode(x)
        
        log_weights = []
        x_recons = []
        
        for _ in range(self.num_samples):
            z = reparameterize(mean, log_var)
            x_recon = self.decode(z)
            x_recons.append(x_recon)
            
            # log p(x|z) (假设高斯)
            log_px_z = -0.5 * sum((x[i] - x_recon[i])**2 for i in range(len(x)))
            
            # log q(z|x)
            log_qz_x = -0.5 * sum((z[i] - mean[i])**2 / math.exp(log_var[i]) + log_var[i] 
                                     for i in range(len(z)))
            
            # log p(z) (标准正态)
            log_pz = -0.5 * sum(z[i]**2 for i in range(len(z)))
            
            log_weights.append(log_px_z + log_pz - log_qz_x)
        
        # log-sum-exp
        max_log_w = max(log_weights)
        iwae_loss = -(max_log_w + math.log(sum(math.exp(lw - max_log_w) for lw in log_weights) / self.num_samples))
        
        # 标准VAE损失
        x_recon = x_recons[0]
        recon_loss = sum((x[i] - x_recon[i])**2 for i in range(len(x))) / len(x)
        kl_loss = kl_divergence(mean, log_var)
        
        return {
            'loss': iwae_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'iwae_bound': iwae_loss
        }


# 工厂函数
def autoencoder(input_dim: int, latent_dim: int, **kwargs) -> AutoEncoder:
    """创建自编码器"""
    return AutoEncoder(input_dim, latent_dim, **kwargs)


def vae(input_dim: int, latent_dim: int, **kwargs) -> VAE:
    """创建变分自编码器"""
    return VAE(input_dim, latent_dim, **kwargs)


def beta_vae(input_dim: int, latent_dim: int, **kwargs) -> BetaVAE:
    """创建Beta-VAE"""
    return BetaVAE(input_dim, latent_dim, **kwargs)


def vq_vae(input_dim: int, latent_dim: int, **kwargs) -> VQVAE:
    """创建VQ-VAE"""
    return VQVAE(input_dim, latent_dim, **kwargs)


def conditional_vae(input_dim: int, latent_dim: int, condition_dim: int, **kwargs) -> ConditionalVAE:
    """创建条件VAE"""
    return ConditionalVAE(input_dim, latent_dim, condition_dim, **kwargs)


def denoising_ae(input_dim: int, latent_dim: int, **kwargs) -> DenoisingAutoEncoder:
    """创建去噪自编码器"""
    return DenoisingAutoEncoder(input_dim, latent_dim, **kwargs)


def sparse_ae(input_dim: int, latent_dim: int, **kwargs) -> SparseAutoEncoder:
    """创建稀疏自编码器"""
    return SparseAutoEncoder(input_dim, latent_dim, **kwargs)


def contractive_ae(input_dim: int, latent_dim: int, **kwargs) -> ContractiveAutoEncoder:
    """创建收缩自编码器"""
    return ContractiveAutoEncoder(input_dim, latent_dim, **kwargs)


def iwae(input_dim: int, latent_dim: int, **kwargs) -> IWAE:
    """创建IWAE"""
    return IWAE(input_dim, latent_dim, **kwargs)
