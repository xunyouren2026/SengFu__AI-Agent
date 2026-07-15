"""
AGI统一框架 - 自动编码器与生成模型
实现VAE、GAN、Diffusion等生成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


# ==================== VAE变体 ====================

class BetaVAE(nn.Module):
    """β-VAE: 可调节KL权重的变分自编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 latent_dim: int = 64, beta: float = 4.0):
        super().__init__()
        self.beta = beta
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), torch.exp(self.fc_var(h))
    
    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        x_recon = self.decode(z)
        
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = 0.5 * torch.sum(var + mu.pow(2) - 1 - var.log())
        
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'reconstruction': x_recon,
            'latent': z,
            'mu': mu,
            'var': var,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'loss': loss
        }


class ConditionalVAE(nn.Module):
    """条件变分自编码器"""
    
    def __init__(self, input_dim: int, condition_dim: int,
                 hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        
        total_input = input_dim + condition_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        h = self.encoder(torch.cat([x, c], dim=-1))
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        
        # 重参数化
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        
        # 解码
        x_recon = self.decoder(torch.cat([z, c], dim=-1))
        
        # 损失
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return {
            'reconstruction': x_recon,
            'latent': z,
            'loss': recon_loss + kl_loss
        }


# ==================== GAN ====================

class VanillaGAN(nn.Module):
    """标准GAN"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # 判别器
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
    
    def discriminator_loss(self, real_output: torch.Tensor, 
                          fake_output: torch.Tensor) -> torch.Tensor:
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss


class WGAN(nn.Module):
    """Wasserstein GAN"""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def criticize(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)
    
    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                        lambda_gp: float = 10.0) -> torch.Tensor:
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, device=real.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        
        critic_output = self.criticize(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_output),
            create_graph=True
        )[0]
        
        gradient_norm = gradients.view(batch_size, -1).norm(dim=1)
        return lambda_gp * ((gradient_norm - 1) ** 2).mean()


class StyleGAN(nn.Module):
    """StyleGAN生成器 (简化版)"""
    
    def __init__(self, latent_dim: int = 512, hidden_dim: int = 512,
                 num_layers: int = 8):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # 映射网络
        self.mapping = nn.Sequential(
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2))
              for _ in range(8)]
        )
        
        # 合成网络
        self.synthesis = nn.ModuleList()
        for i in range(num_layers):
            self.synthesis.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            
        # 样式调制层
        self.style_layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim * 2) for _ in range(num_layers)
        ])
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 映射到样式空间
        w = self.mapping(z)
        
        # 合成
        x = torch.zeros(z.size(0), 512, device=z.device)
        
        for i, (synth_layer, style_layer) in enumerate(zip(self.synthesis, self.style_layers)):
            # 样式调制
            style = style_layer(w)
            scale, bias = style.chunk(2, dim=-1)
            
            x = synth_layer(x)
            x = x * scale + bias
            
        return x


# ==================== Diffusion Model ====================

class GaussianDiffusion(nn.Module):
    """高斯扩散模型"""
    
    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # β调度
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
        
        # 后验分布参数
        self.register_buffer('posterior_variance',
                           self.betas * (1.0 - self.alphas_cumprod) / (1.0 - self.alphas_cumprod))
        
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_losses(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        return F.mse_loss(noise, predicted_noise)
    
    def p_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """反向采样一步"""
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        
        predicted_noise = self.model(x, t_tensor)
        
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        mean = (1.0 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1.0 - alpha_cumprod) * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        else:
            return mean
    
    def sample(self, batch_size: int, device: torch.device,
               shape: Tuple[int, ...]) -> torch.Tensor:
        """生成样本"""
        x = torch.randn(batch_size, *shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
            
        return x


class UNetBlock(nn.Module):
    """UNet块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_dim: int = 256, up: bool = False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm = nn.GroupNorm(8, out_channels)
        
        if up:
            self.updown = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.updown = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        t = self.time_mlp(t)[:, :, None, None]
        x = x + t
        x = self.norm(x)
        x = F.silu(x)
        return self.updown(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==================== Flow-based Models ====================

class RealNVP(nn.Module):
    """Real NVP流模型"""
    
    def __init__(self, dim: int, num_flows: int = 6, hidden_dim: int = 64):
        super().__init__()
        
        self.dim = dim
        self.num_flows = num_flows
        
        # 耦合层
        self.s_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2)
            )
            for _ in range(num_flows)
        ])
        
        self.t_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2)
            )
            for _ in range(num_flows)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.num_flows):
            # 掩码
            if i % 2 == 0:
                mask = torch.zeros(self.dim, device=x.device)
                mask[:self.dim // 2] = 1
            else:
                mask = torch.zeros(self.dim, device=x.device)
                mask[self.dim // 2:] = 1
                
            mask = mask.bool()
            
            # 耦合变换
            x_masked = x[:, mask]
            s = self.s_nets[i](x_masked)
            t = self.t_nets[i](x_masked)
            
            x[:, ~mask] = x[:, ~mask] * torch.exp(s) + t
            log_det = log_det + s.sum(dim=-1)
            
        return x, log_det
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = y.clone()
        
        for i in reversed(range(self.num_flows)):
            if i % 2 == 0:
                mask = torch.zeros(self.dim, device=y.device)
                mask[:self.dim // 2] = 1
            else:
                mask = torch.zeros(self.dim, device=y.device)
                mask[self.dim // 2:] = 1
                
            mask = mask.bool()
            
            x_masked = x[:, mask]
            s = self.s_nets[i](x_masked)
            t = self.t_nets[i](x_masked)
            
            x[:, ~mask] = (x[:, ~mask] - t) * torch.exp(-s)
            
        return x
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_prob_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_prob_base + log_det


# ==================== Autoregressive Models ====================

class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_hidden: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 创建掩码
        masks = self._create_masks(input_dim, hidden_dim, num_hidden)
        
        # 构建网络
        layers = []
        dims = [input_dim] + [hidden_dim] * num_hidden + [input_dim * 2]
        
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            layer.weight.data = layer.weight.data * masks[i]
            layers.append(layer)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                
        self.net = nn.Sequential(*layers)
        
    def _create_masks(self, input_dim: int, hidden_dim: int, 
                     num_hidden: int) -> List[torch.Tensor]:
        """创建自回归掩码"""
        m0 = torch.arange(1, input_dim + 1)
        
        masks = []
        m_prev = m0
        
        for i in range(num_hidden):
            m = torch.randint(1, input_dim, (hidden_dim,))
            mask = (m_prev[:, None] <= m[None, :]).float()
            masks.append(mask)
            m_prev = m
            
        mask = (m_prev[:, None] <= m0[None, :]).float()
        masks.append(mask)
        
        # 输出掩码 (mu和log_var)
        mask = torch.cat([mask, mask], dim=1)
        masks[-1] = mask
        
        return masks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        mu, log_var = output.chunk(2, dim=-1)
        return mu, log_var


class PixelCNN(nn.Module):
    """PixelCNN自回归模型"""
    
    def __init__(self, num_channels: int = 3, hidden_dim: int = 64,
                 num_layers: int = 5):
        super().__init__()
        
        self.num_channels = num_channels
        
        # 掩码卷积层
        self.layers = nn.ModuleList()
        
        # 第一层 (中心像素被掩码)
        self.layers.append(MaskedConv2d(num_channels, hidden_dim, 7, mask_type='A'))
        
        # 中间层
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                MaskedConv2d(hidden_dim, hidden_dim, 7, mask_type='B'),
                nn.ReLU()
            ))
            
        # 输出层
        self.output = nn.Conv2d(hidden_dim, num_channels * 256, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers[0](x)
        h = F.relu(h)
        
        for layer in self.layers[1:]:
            h = layer(h)
            
        return self.output(h)


class MaskedConv2d(nn.Conv2d):
    """掩码卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, mask_type: str = 'A'):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.mask_type = mask_type
        self.register_buffer('mask', self._create_mask(kernel_size))
        
    def _create_mask(self, kernel_size: int) -> torch.Tensor:
        mask = torch.ones(kernel_size, kernel_size)
        center = kernel_size // 2
        
        # 掩码中心及之后
        mask[center + 1:, :] = 0
        mask[center, center + (1 if self.mask_type == 'A' else 0):] = 0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)
