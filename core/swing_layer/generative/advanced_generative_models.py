"""
高级生成模型 - Advanced Generative Models
实现Flow Matching、VQ-VAE、Consistency Models等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque

# ==================== VQ-VAE ====================

class VectorQuantizer(nn.Module):
    """向量量化层"""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 嵌入表
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA统计
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 展平
        z_flat = z.view(-1, self.embedding_dim)
        
        # 计算距离
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        # 最近邻编码
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view_as(z)
        
        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = z + (quantized - z).detach()
        
        # 困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # EMA更新
        if self.training:
            self._ema_update(z_flat, encodings)
        
        encoding_indices = encoding_indices.view(z.size(0), -1)
        
        return quantized, loss, perplexity, encoding_indices
    
    def _ema_update(self, z_flat: torch.Tensor, encodings: torch.Tensor):
        """EMA更新"""
        # 簇大小
        n = torch.sum(encodings, dim=0)
        self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * n
        
        # 嵌入求和
        dw = torch.matmul(encodings.t(), z_flat)
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        
        # 归一化
        n = self.ema_cluster_size
        self.embedding.weight.data = self.ema_w / (n.unsqueeze(1) + self.epsilon)


class VectorQuantizerEMA(nn.Module):
    """EMA向量量化层"""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # 嵌入表
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_count', torch.ones(num_embeddings))
        self.register_buffer('ema_weight', embed.clone())
    
    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        z_flat = z.view(-1, self.embedding_dim)
        
        # L2距离
        distances = torch.cdist(z_flat.unsqueeze(0), self.embedding.unsqueeze(0)).squeeze(0) ** 2
        
        # 编码
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding[encoding_indices].view_as(z)
        
        # 损失
        commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())
        
        # 直通估计
        quantized = z + (quantized - z).detach()
        
        # EMA更新
        if self.training:
            with torch.no_grad():
                self._update_ema(z_flat, encoding_indices)
        
        return quantized, commitment_loss, encoding_indices.view(z.size(0), -1)
    
    def _update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """更新EMA"""
        # 统计每个编码的使用次数
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        cluster_size = one_hot.sum(dim=0)
        
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * cluster_size
        
        # 嵌入求和
        embed_sum = one_hot.t() @ z_flat
        self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * embed_sum
        
        # 更新嵌入
        n = self.ema_count.unsqueeze(1)
        self.embedding = self.ema_weight / (n + 1e-5)


class VQVAEEncoder(nn.Module):
    """VQ-VAE编码器"""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_resolutions: int = 3,
    ):
        super().__init__()
        
        layers = []
        channels = in_channels
        
        # 初始卷积
        layers.append(nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1))
        layers.append(nn.ReLU())
        channels = hidden_channels
        
        # 下采样
        for _ in range(num_resolutions - 1):
            layers.append(nn.Conv2d(channels, channels, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
        
        # 残差块
        for _ in range(2):
            layers.append(ResidualBlock(channels, channels))
        
        # 输出
        layers.append(nn.Conv2d(channels, embedding_dim, 3, padding=1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQVAEDecoder(nn.Module):
    """VQ-VAE解码器"""
    
    def __init__(
        self,
        out_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_resolutions: int = 3,
    ):
        super().__init__()
        
        layers = []
        
        # 输入
        layers.append(nn.Conv2d(embedding_dim, hidden_channels, 3, padding=1))
        
        # 残差块
        for _ in range(2):
            layers.append(ResidualBlock(hidden_channels, hidden_channels))
        
        # 上采样
        for _ in range(num_resolutions - 1):
            layers.append(nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
        
        # 最后上采样
        layers.append(nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1))
        layers.append(nn.ReLU())
        
        # 输出
        layers.append(nn.Conv2d(hidden_channels, out_channels, 3, padding=1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VQVAE(nn.Module):
    """VQ-VAE"""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        num_resolutions: int = 3,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        
        self.encoder = VQVAEEncoder(in_channels, hidden_channels, embedding_dim, num_resolutions)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQVAEDecoder(in_channels, hidden_channels, embedding_dim, num_resolutions)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 编码
        z = self.encoder(x)
        
        # 量化
        z_q, vq_loss, perplexity, indices = self.quantizer(z)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss, perplexity, indices
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        z = self.encoder(x)
        _, _, _, indices = self.quantizer(z)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """解码"""
        z_q = self.quantizer.embedding(indices)
        return self.decoder(z_q)


# ==================== Flow Matching ====================

class FlowMatching(nn.Module):
    """Flow Matching模型"""
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.data_dim = data_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 速度场网络
        layers = []
        layers.append(nn.Linear(data_dim + time_embed_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, data_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """计算速度场"""
        # 时间嵌入
        t_embed = self.time_embed(t.unsqueeze(-1))
        
        # 拼接
        inp = torch.cat([x, t_embed], dim=-1)
        
        # 速度
        v = self.net(inp)
        
        return v
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """计算Flow Matching损失"""
        batch_size = x_0.size(0)
        
        # 采样时间
        t = torch.rand(batch_size, device=x_0.device)
        
        # 插值
        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        
        # 目标速度
        v_target = x_1 - x_0
        
        # 预测速度
        v_pred = self.forward(x_t, t)
        
        # 损失
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def sample(
        self,
        x_0: torch.Tensor,
        num_steps: int = 100,
        method: str = 'euler',
    ) -> torch.Tensor:
        """生成样本"""
        dt = 1.0 / num_steps
        x = x_0.clone()
        
        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            
            if method == 'euler':
                v = self.forward(x, t_tensor)
                x = x + v * dt
            
            elif method == 'rk4':
                # Runge-Kutta 4阶
                k1 = self.forward(x, t_tensor)
                k2 = self.forward(x + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                k3 = self.forward(x + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                k4 = self.forward(x + dt * k3, t_tensor + dt)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return x


class ConditionalFlowMatching(FlowMatching):
    """条件Flow Matching"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算条件Flow Matching损失"""
        batch_size = x_0.size(0)
        
        # 采样时间
        t = torch.rand(batch_size, device=x_0.device)
        
        # 条件路径
        if condition is not None:
            # 使用条件信息
            x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
            x_t = torch.cat([x_t, condition], dim=-1)
        else:
            x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        
        # 目标速度
        v_target = x_1 - x_0
        
        # 预测速度
        v_pred = self.forward(x_t, t)
        
        # 损失
        loss = F.mse_loss(v_pred, v_target)
        
        return loss


class OptimalTransportFlowMatching(FlowMatching):
    """最优传输Flow Matching"""
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """计算最优传输Flow Matching损失"""
        batch_size = x_0.size(0)
        
        # 计算最优传输计划（简化：使用身份映射）
        # 实际应用中可以使用Sinkhorn算法
        
        # 采样时间
        t = torch.rand(batch_size, device=x_0.device)
        
        # 最优传输路径（直线）
        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        
        # 目标速度（常数速度场）
        v_target = x_1 - x_0
        
        # 预测速度
        v_pred = self.forward(x_t, t)
        
        # 损失
        loss = F.mse_loss(v_pred, v_target)
        
        return loss


# ==================== Consistency Models ====================

class ConsistencyModel(nn.Module):
    """Consistency Model"""
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.sigma_data = sigma_data
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 网络
        layers = []
        layers.append(nn.Linear(data_dim + hidden_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, data_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        # 跳跃连接
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        
        # 时间嵌入
        sigma_embed = self.time_embed(sigma.unsqueeze(-1))
        
        # 网络输入
        inp = torch.cat([x * c_in.unsqueeze(-1), sigma_embed], dim=-1)
        
        # 网络输出
        F_out = self.net(inp)
        
        # 组合
        out = c_skip.unsqueeze(-1) * x + c_out.unsqueeze(-1) * F_out
        
        return out
    
    def consistency_loss(
        self,
        x: torch.Tensor,
        sigma_1: torch.Tensor,
        sigma_2: torch.Tensor,
    ) -> torch.Tensor:
        """一致性损失"""
        # 噪声
        noise = torch.randn_like(x)
        
        # 加噪
        x_1 = x + sigma_1.unsqueeze(-1) * noise
        x_2 = x + sigma_2.unsqueeze(-1) * noise
        
        # 预测
        out_1 = self.forward(x_1, sigma_1)
        out_2 = self.forward(x_2, sigma_2)
        
        # 损失
        loss = F.mse_loss(out_1, out_2)
        
        return loss
    
    def sample(
        self,
        noise: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """生成样本"""
        # 初始噪声
        sigma_max = torch.tensor(80.0, device=noise.device)
        x = noise * sigma_max
        
        # 多步采样
        for _ in range(num_steps):
            sigma = torch.full((x.size(0),), sigma_max, device=x.device)
            x = self.forward(x, sigma)
        
        return x


# ==================== Latent Diffusion ====================

class LatentEncoder(nn.Module):
    """潜在编码器"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        hidden_channels: int = 64,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim * 2, 3, padding=1),
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码"""
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class LatentDecoder(nn.Module):
    """潜在解码器"""
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 4,
        hidden_channels: int = 64,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """解码"""
        return self.net(z)


class LatentDiffusion(nn.Module):
    """潜在扩散模型"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        hidden_channels: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        super().__init__()
        
        self.encoder = LatentEncoder(in_channels, latent_dim, hidden_channels)
        self.decoder = LatentDecoder(in_channels, latent_dim, hidden_channels)
        
        self.num_timesteps = num_timesteps
        self.latent_dim = latent_dim
        
        # 噪声调度
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())
        
        # UNet去噪网络
        self.unet = SimpleUNet(latent_dim, hidden_channels)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """前向扩散"""
        z = self.encode(x)
        
        # 加噪
        noise = torch.randn_like(z)
        z_t = (
            self.sqrt_alphas_cumprod[t][:, None, None, None] * z +
            self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * noise
        )
        
        # 预测噪声
        noise_pred = self.unet(z_t, t)
        
        return noise_pred, noise
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        batch_size = x.size(0)
        
        # 随机时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
        # 前向
        noise_pred, noise = self.forward(x, t)
        
        # 损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def sample(
        self,
        batch_size: int,
        img_size: int = 64,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """生成样本"""
        # 从纯噪声开始
        z = torch.randn(batch_size, self.latent_dim, img_size // 4, img_size // 4, device=device)
        
        # 逐步去噪
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.unet(z, t_tensor)
            
            # 去噪步骤
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = 0
            
            z = (1 / alpha.sqrt()) * (z - beta / (1 - alpha_cumprod).sqrt() * noise_pred) + beta.sqrt() * noise
        
        # 解码
        x = self.decode(z)
        
        return x


class SimpleUNet(nn.Module):
    """简单UNet"""
    
    def __init__(self, channels: int, hidden_channels: int = 64):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        
        # 编码器
        self.enc1 = nn.Conv2d(channels, hidden_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        
        # 中间
        self.mid = nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1)
        
        # 解码器
        self.dec3 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Conv2d(hidden_channels * 2, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 时间嵌入
        t_embed = self.time_embed(t.unsqueeze(-1).float() / 1000)
        
        # 编码
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        h3 = F.silu(self.enc3(h2))
        
        # 中间
        h = F.silu(self.mid(h3))
        
        # 解码
        h = F.silu(self.dec3(h))
        h = torch.cat([h, h2], dim=1)
        h = F.silu(self.dec2(h))
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h)
        
        return h


# ==================== 主函数 ====================

def main():
    """测试高级生成模型"""
    print("高级生成模型测试")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试VQ-VAE
    print("\n测试VQ-VAE...")
    vqvae = VQVAE(
        in_channels=3,
        hidden_channels=64,
        embedding_dim=32,
        num_embeddings=256,
    ).to(device)
    
    x = torch.randn(2, 3, 32, 32).to(device)
    x_recon, vq_loss, perplexity, indices = vqvae(x)
    print(f"VQ-VAE output shape: {x_recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}, Perplexity: {perplexity.item():.2f}")
    
    # 测试Flow Matching
    print("\n测试Flow Matching...")
    fm = FlowMatching(data_dim=32, hidden_dim=64).to(device)
    
    x_0 = torch.randn(16, 32).to(device)
    x_1 = torch.randn(16, 32).to(device)
    
    loss = fm.compute_loss(x_0, x_1)
    print(f"Flow Matching loss: {loss.item():.4f}")
    
    samples = fm.sample(x_0, num_steps=10)
    print(f"Generated samples shape: {samples.shape}")
    
    # 测试Consistency Model
    print("\n测试Consistency Model...")
    cm = ConsistencyModel(data_dim=32, hidden_dim=64).to(device)
    
    x = torch.randn(16, 32).to(device)
    sigma_1 = torch.ones(16).to(device) * 1.0
    sigma_2 = torch.ones(16).to(device) * 0.5
    
    loss = cm.consistency_loss(x, sigma_1, sigma_2)
    print(f"Consistency loss: {loss.item():.4f}")
    
    # 测试Latent Diffusion
    print("\n测试Latent Diffusion...")
    ld = LatentDiffusion(
        in_channels=3,
        latent_dim=4,
        hidden_channels=32,
        num_timesteps=100,
    ).to(device)
    
    x = torch.randn(2, 3, 32, 32).to(device)
    loss = ld.compute_loss(x)
    print(f"Latent Diffusion loss: {loss.item():.4f}")
    
    print("\n高级生成模型测试完成")


if __name__ == "__main__":
    main()
