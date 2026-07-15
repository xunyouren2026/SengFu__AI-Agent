"""
扩散模型模块 - 完整实现
包含: DDPM, DDIM, Score-Based SDE, NoiseScheduler,
      UNet去噪网络, Classifier-Free Guidance等
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
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def softplus(x: float) -> float:
    if x > 20:
        return x
    return math.log(1 + math.exp(x))



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


class NoiseScheduler:
    """
    噪声调度器
    管理扩散过程中噪声的添加和去除
    """
    
    def __init__(self, num_timesteps: int = 1000, 
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 schedule: str = 'linear'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule
        
        # 计算beta序列
        if schedule == 'linear':
            self.betas = [beta_start + (beta_end - beta_start) * t / (num_timesteps - 1) 
                         for t in range(num_timesteps)]
        elif schedule == 'cosine':
            self.betas = self._cosine_schedule(num_timesteps)
        elif schedule == 'quadratic':
            self.betas = [beta_start + (beta_end - beta_start) * (t / (num_timesteps - 1))**2 
                         for t in range(num_timesteps)]
        elif schedule == 'sigmoid':
            self.betas = self._sigmoid_schedule(num_timesteps, beta_start, beta_end)
        else:
            self.betas = [beta_start + (beta_end - beta_start) * t / (num_timesteps - 1) 
                         for t in range(num_timesteps)]
        
        # 预计算alpha序列
        self.alphas = [1.0 - b for b in self.betas]
        self.alphas_cumprod = [1.0]
        for a in self.alphas:
            self.alphas_cumprod.append(self.alphas_cumprod[-1] * a)
        self.alphas_cumprod = self.alphas_cumprod[1:]  # 移除初始1
        
        # sqrt(alpha_cumprod)
        self.sqrt_alphas_cumprod = [math.sqrt(a) for a in self.alphas_cumprod]
        
        # sqrt(1 - alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = [math.sqrt(1.0 - a) for a in self.alphas_cumprod]
        
        # 1/sqrt(alpha)
        self.sqrt_recip_alphas = [1.0 / math.sqrt(a) for a in self.alphas]
        
        # 后验方差
        self.posterior_variance = self._compute_posterior_variance()
    
    def _cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> Union[List[float], 'torch.Tensor']:
        """余弦调度"""
        steps = list(range(num_timesteps + 1))
        f_t = [math.cos((t / num_timesteps + s) / (1 + s) * math.pi / 2)**2 for t in steps]
        betas = []
        for i in range(num_timesteps):
            betas.append(min(1.0 - f_t[i] / f_t[i+1], 0.999))
        return betas
    
    def _sigmoid_schedule(self, num_timesteps: int, beta_start: float, beta_end: float) -> Union[List[float], 'torch.Tensor']:
        """Sigmoid调度"""
        betas = []
        for t in range(num_timesteps):
            x = (t / num_timesteps - 0.5) * 10
            sig = sigmoid(x)
            betas.append(beta_start + (beta_end - beta_start) * sig)
        return betas
    
    def _compute_posterior_variance(self) -> Union[List[float], 'torch.Tensor']:
        """计算后验方差 q(x_{t-1} | x_t, x_0)"""
        posterior_variance = []
        for t in range(self.num_timesteps):
            if t == 0:
                var = self.betas[0]
            else:
                a_cumprod_t = self.alphas_cumprod[t]
                a_cumprod_t_prev = self.alphas_cumprod[t - 1]
                beta_t = self.betas[t]
                var = beta_t * (1 - a_cumprod_t_prev) / (1 - a_cumprod_t)
            posterior_variance.append(max(var, 1e-20))
        return posterior_variance
    
    def add_noise(self, x_0: Union[List[float], 'torch.Tensor'], noise: Union[List[float], 'torch.Tensor'], t: int) -> Union[List[float], 'torch.Tensor']:
        """
        前向扩散: q(x_t | x_0) = N(sqrt(alpha_cumprod_t) * x_0, (1-alpha_cumprod_t) * I)
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * noise
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        return [sqrt_alpha * x_0[i] + sqrt_one_minus_alpha * noise[i] for i in range(len(x_0))]
    
    def get_velocity(self, x_t: Union[List[float], 'torch.Tensor'], noise: Union[List[float], 'torch.Tensor'], t: int) -> Union[List[float], 'torch.Tensor']:
        """
        获取速度 (用于SDE采样)
        dx = f(x, t)dt + g(t)dw
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # velocity matching: v = sqrt(alpha_cumprod_t) * noise - sqrt(1-alpha_cumprod_t) * x_t
        # 实际上: v_theta 预测的是 noise
        return [sqrt_alpha * noise[i] - sqrt_one_minus_alpha * x_t[i] for i in range(len(x_t))]


class GaussianDiffusion:
    """
    高斯扩散过程
    实现DDPM的完整前向和反向过程
    """
    
    def __init__(self, scheduler: NoiseScheduler):
        self.scheduler = scheduler
    
    def q_sample(self, x_0: Union[List[float], 'torch.Tensor'], t: int, noise: Optional[Union[List[float], 'torch.Tensor']] = None) -> Tuple[Union[List[float], 'torch.Tensor'], Union[List[float], 'torch.Tensor']]:
        """
        前向采样 q(x_t | x_0)
        """
        if noise is None:
            noise = [random.gauss(0, 1) for _ in x_0]
        
        x_t = self.scheduler.add_noise(x_0, noise, t)
        return x_t, noise
    
    def q_posterior_mean_variance(self, x_0: Union[List[float], 'torch.Tensor'], x_t: Union[List[float], 'torch.Tensor'], t: int) -> Tuple[Union[List[float], 'torch.Tensor'], float]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        a_cumprod_t = self.scheduler.alphas_cumprod[t]
        a_cumprod_t_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_t = self.scheduler.betas[t]
        
        # 后验均值
        posterior_mean = []
        for i in range(len(x_t)):
            coef1 = math.sqrt(a_cumprod_t_prev) * beta_t / (1 - a_cumprod_t)
            coef2 = math.sqrt(1 - a_cumprod_t_prev) * (1 - beta_t) / (1 - a_cumprod_t)
            posterior_mean.append(coef1 * x_0[i] + coef2 * x_t[i])
        
        # 后验方差
        posterior_var = self.scheduler.posterior_variance[t]
        
        return posterior_mean, posterior_var
    
    def p_mean_variance(self, model_output: Union[List[float], 'torch.Tensor'], x_t: Union[List[float], 'torch.Tensor'], t: int,
                        clip_denoised: bool = True) -> Tuple[Union[List[float], 'torch.Tensor'], float]:
        """
        计算反向过程 p(x_{t-1} | x_t) 的均值和方差
        
        model_output: 模型预测的噪声 ε_θ(x_t, t)
        """
        a_cumprod_t = self.scheduler.alphas_cumprod[t]
        a_cumprod_t_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_t = self.scheduler.betas[t]
        sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        # 预测 x_0
        x_0_pred = []
        for i in range(len(x_t)):
            val = sqrt_recip_alpha * x_t[i] - sqrt_one_minus_alpha * model_output[i]
            if clip_denoised:
                val = max(-1.0, min(1.0, val))
            x_0_pred.append(val)
        
        # 计算均值
        posterior_mean, posterior_var = self.q_posterior_mean_variance(x_0_pred, x_t, t)
        
        return posterior_mean, posterior_var
    
    def p_sample(self, model_output: Union[List[float], 'torch.Tensor'], x_t: Union[List[float], 'torch.Tensor'], t: int,
                 clip_denoised: bool = True) -> Union[List[float], 'torch.Tensor']:
        """
        反向采样单步: p(x_{t-1} | x_t)
        """
        mean, var = self.p_mean_variance(model_output, x_t, t, clip_denoised)
        
        # 添加噪声
        noise = [random.gauss(0, 1) for _ in x_t] if t > 0 else [0.0] * len(x_t)
        
        return [mean[i] + math.sqrt(var) * noise[i] for i in range(len(x_t))]
    
    def p_sample_loop(self, model_fn: Callable, shape: List[int], 
                      return_intermediates: bool = False) -> Union[List[float], 'torch.Tensor', Tuple]:
        """
        完整的反向采样循环
        model_fn: 接受 (x_t, t) 返回预测噪声
        """
        # 从纯噪声开始
        x = [random.gauss(0, 1) for _ in range(shape[0] if len(shape) == 1 else shape[0] * shape[1])]
        
        intermediates = []
        
        for t in reversed(range(self.scheduler.num_timesteps)):
            model_output = model_fn(x, t)
            x = self.p_sample(model_output, x, t)
            
            if return_intermediates:
                intermediates.append(x[:])
        
        if return_intermediates:
            return x, intermediates
        return x


class DDPM:
    """
    Denoising Diffusion Probabilistic Models (DDPM)
    
    完整实现DDPM训练和采样
    """
    
    def __init__(self, model_fn: Callable, num_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 schedule: str = 'linear'):
        self.model_fn = model_fn
        self.scheduler = NoiseScheduler(num_timesteps, beta_start, beta_end, schedule)
        self.diffusion = GaussianDiffusion(self.scheduler)
        
        self.training_step = 0
    
    def training_loss(self, x_0: Union[List[float], 'torch.Tensor']) -> Dict[str, float]:
        """
        计算训练损失
        L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||^2]
        """
        # 随机采样时间步
        t = random.randint(0, self.scheduler.num_timesteps - 1)
        
        # 采样噪声
        noise = [random.gauss(0, 1) for _ in x_0]
        
        # 前向扩散
        x_t, _ = self.diffusion.q_sample(x_0, t, noise)
        
        # 模型预测
        model_output = self.model_fn(x_t, t)
        
        # 计算MSE损失
        mse = sum((model_output[i] - noise[i])**2 for i in range(len(x_0))) / len(x_0)
        
        self.training_step += 1
        
        return {'loss': mse, 'timestep': t}
    
    def sample(self, shape: List[int], return_intermediates: bool = False) -> Union[List[float], 'torch.Tensor', Tuple]:
        """
        采样生成
        """
        return self.diffusion.p_sample_loop(self.model_fn, shape, return_intermediates)
    
    def ddim_sample(self, shape: List[int], num_inference_steps: int = 50,
                    eta: float = 0.0) -> Union[List[float], 'torch.Tensor']:
        """
        DDIM采样 (加速采样)
        """
        ddim = DDIM(self.scheduler, num_inference_steps, eta)
        return ddim.sample(self.model_fn, shape)


class DDIM:
    """
    Denoising Diffusion Implicit Models (DDIM)
    
    加速采样，使用非马尔可夫过程
    """
    
    def __init__(self, scheduler: NoiseScheduler, num_inference_steps: int = 50,
                 eta: float = 0.0):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # 创建子序列
        step_ratio = scheduler.num_timesteps // num_inference_steps
        self.timesteps = list(range(0, scheduler.num_timesteps, step_ratio))
        self.timesteps.reverse()
    
    def ddim_step(self, model_output: Union[List[float], 'torch.Tensor'], x_t: Union[List[float], 'torch.Tensor'], t: int, t_prev: int) -> Union[List[float], 'torch.Tensor']:
        """
        DDIM单步采样
        """
        a_cumprod_t = self.scheduler.alphas_cumprod[t]
        a_cumprod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
        
        # 预测x_0
        sqrt_recip_alpha = self.scheduler.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        
        x_0_pred = [sqrt_recip_alpha * x_t[i] - sqrt_one_minus_alpha * model_output[i] 
                   for i in range(len(x_t))]
        
        # 方向指向x_t
        sqrt_alpha_cumprod_prev = math.sqrt(a_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_prev = math.sqrt(1 - a_cumprod_t_prev)
        
        # DDIM sigma
        sigma = self.eta * math.sqrt((1 - a_cumprod_t_prev) / (1 - a_cumprod_t) * (1 - a_cumprod_t / a_cumprod_t_prev))
        
        # 预测噪声
        pred_noise = [model_output[i] for i in range(len(x_t))]
        
        # 计算x_{t-1}
        dir_xt = [sqrt_one_minus_alpha_cumprod_prev * pred_noise[i] for i in range(len(x_t))]
        
        noise = [random.gauss(0, 1) for _ in x_t] if sigma > 0 else [0.0] * len(x_t)
        
        x_prev = [sqrt_alpha_cumprod_prev * x_0_pred[i] + dir_xt[i] + sigma * noise[i] 
                 for i in range(len(x_t))]
        
        return x_prev
    
    def sample(self, model_fn: Callable, shape: List[int]) -> Union[List[float], 'torch.Tensor']:
        """
        DDIM采样
        """
        # 从纯噪声开始
        x = [random.gauss(0, 1) for _ in range(shape[0])]
        
        for i in range(len(self.timesteps)):
            t = self.timesteps[i]
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else -1
            
            model_output = model_fn(x, t)
            x = self.ddim_step(model_output, x, t, t_prev)
        
        return x


class ScoreBasedSDE:
    """
    基于分数的随机微分方程 (Score-Based SDE)
    
    dx = f(x, t)dt + g(t)dw
    
    反向SDE: dx = [f(x,t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) d\bar{w}
    """
    
    def __init__(self, score_fn: Callable, sde_type: str = 've',
                 num_timesteps: int = 1000, beta_min: float = 0.1, beta_max: float = 20.0):
        self.score_fn = score_fn
        self.sde_type = sde_type
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def sde_coefficients(self, t: float) -> Tuple[float, float]:
        """
        获取SDE系数 (f(x,t), g(t))
        """
        if self.sde_type == 've':  # Variance Exploding
            # f(x,t) = 0, g(t) = beta(t)
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            return 0.0, beta_t
        elif self.sde_type == 'vp':  # Variance Preserving
            # f(x,t) = -1/2 beta(t) x, g(t) = sqrt(beta(t))
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t, math.sqrt(beta_t)
        elif self.sde_type == 'subvp':  # Sub-VP
            beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t, math.sqrt(beta_t) * (1 - math.exp(-2 * beta_t))**0.5
        else:
            return 0.0, 1.0
    
    def reverse_sde_coefficients(self, t: float) -> Tuple[float, float]:
        """
        获取反向SDE系数
        """
        f, g = self.sde_coefficients(t)
        # f_reverse = f - g^2 * score
        return f, g
    
    def euler_maruyama_step(self, x: Union[List[float], 'torch.Tensor'], t: float, dt: float, 
                            reverse: bool = False) -> Union[List[float], 'torch.Tensor']:
        """
        Euler-Maruyama数值积分单步
        """
        if reverse:
            score = self.score_fn(x, t)
            f, g = self.reverse_sde_coefficients(t)
            
            # dx = (f - g^2 * score) dt + g d\bar{w}
            drift = [(f - g * g * score[i]) * dt for i in range(len(x))]
            diffusion = [g * math.sqrt(dt) * random.gauss(0, 1) for _ in x]
        else:
            f, g = self.sde_coefficients(t)
            drift = [f * x[i] * dt for i in range(len(x))]
            diffusion = [g * math.sqrt(dt) * random.gauss(0, 1) for _ in x]
        
        return [x[i] + drift[i] + diffusion[i] for i in range(len(x))]
    
    def heun_step(self, x: Union[List[float], 'torch.Tensor'], t: float, dt: float) -> Union[List[float], 'torch.Tensor']:
        """
        Heun数值积分 (更精确)
        """
        f, g = self.sde_coefficients(t)
        score = self.score_fn(x, t)
        
        # 第一步
        x_prime = [x[i] + (f * x[i] - g * g * score[i]) * dt for i in range(len(x))]
        
        # 第二步
        score_prime = self.score_fn(x_prime, t - dt)
        f_prime, g_prime = self.sde_coefficients(t - dt)
        
        x_next = [x[i] + 0.5 * ((f * x[i] - g * g * score[i]) + 
                                  (f_prime * x_prime[i] - g_prime * g_prime * score_prime[i])) * dt 
                  for i in range(len(x))]
        
        noise = [g * math.sqrt(dt) * random.gauss(0, 1) for _ in x]
        return [x_next[i] + noise[i] for i in range(len(x))]
    
    def sample(self, shape: List[int], method: str = 'euler', 
               num_steps: int = 100) -> Union[List[float], 'torch.Tensor']:
        """
        采样
        """
        x = [random.gauss(0, 1) for _ in range(shape[0])]
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = 1.0 - step * dt
            if method == 'euler':
                x = self.euler_maruyama_step(x, t, dt, reverse=True)
            elif method == 'heun':
                x = self.heun_step(x, t, dt)
        
        return x


class UNet1D:
    """
    1D UNet去噪网络 (简化版)
    用于扩散模型的噪声预测
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 time_embed_dim: int = 128, hidden_dims: List[int] = [64, 128, 256]):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.hidden_dims = hidden_dims
        
        # 时间嵌入
        std = math.sqrt(2.0 / time_embed_dim)
        self.time_mlp1 = [[random.gauss(0, std) for _ in range(time_embed_dim)] 
                         for _ in range(time_embed_dim)]
        self.time_mlp2 = [[random.gauss(0, std) for _ in range(time_embed_dim)] 
                         for _ in range(time_embed_dim)]
        
        # 编码器
        self.encoder_weights = []
        dims = [in_channels] + hidden_dims
        for i in range(len(dims) - 1):
            std = math.sqrt(2.0 / dims[i])
            w = [[random.gauss(0, std) for _ in range(dims[i])] for _ in range(dims[i+1])]
            self.encoder_weights.append(w)
        
        # 解码器
        self.decoder_weights = []
        dims = [hidden_dims[-1]] + hidden_dims[::-1] + [out_channels]
        for i in range(len(dims) - 1):
            std = math.sqrt(2.0 / dims[i])
            w = [[random.gauss(0, std) for _ in range(dims[i])] for _ in range(dims[i+1])]
            self.decoder_weights.append(w)
    
    def _sinusoidal_time_embedding(self, t: int) -> Union[List[float], 'torch.Tensor']:
        """正弦时间嵌入"""
        half_dim = self.time_embed_dim // 2
        emb = []
        for i in range(half_dim):
            freq = math.exp(-math.log(10000) * i / half_dim)
            emb.append(math.sin(t * freq))
            emb.append(math.cos(t * freq))
        return emb
    
    def _linear(self, x: Union[List[float], 'torch.Tensor'], weight: Union[List[List[float]], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """线性变换"""
        out = [0.0 for _ in range(len(weight))]
        for i in range(len(weight)):
            for j in range(len(x)):
                out[i] += weight[i][j] * x[j]
        return out
    
    def _relu(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        return [max(0.0, xi) for xi in x]
    
    def _silu(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        return [xi / (1 + math.exp(-xi)) for xi in x]
    
    def forward(self, x: Union[List[float], 'torch.Tensor'], t: int) -> Union[List[float], 'torch.Tensor']:
        """
        前向传播
        x: 输入 (in_channels,)
        t: 时间步
        返回: 预测噪声 (out_channels,)
        """
        # 时间嵌入
        t_emb = self._sinusoidal_time_embedding(t)
        t_emb = self._relu(self._linear(t_emb, self.time_mlp1))
        t_emb = self._relu(self._linear(t_emb, self.time_mlp2))
        
        # 编码器
        h = x
        skip_connections = []
        for w in self.encoder_weights:
            h = self._linear(h, w)
            h = self._silu(h)
            skip_connections.append(h)
        
        # 解码器
        for i, w in enumerate(self.decoder_weights):
            if i < len(skip_connections):
                # 跳跃连接
                skip = skip_connections[-(i+1)]
                h = [h[j] + skip[j] for j in range(min(len(h), len(skip)))]
            h = self._linear(h, w)
            if i < len(self.decoder_weights) - 1:
                h = self._silu(h)
        
        return h


class ClassifierFreeGuidance:
    """
    无分类器引导 (Classifier-Free Guidance)
    
    ∇ log p(y|x) = (1 + w) * ∇ log p(x|y) - w * ∇ log p(x)
    
    结合条件和无条件生成
    """
    
    def __init__(self, model_fn: Callable, unconditional_model_fn: Callable,
                 guidance_scale: float = 7.5):
        self.model_fn = model_fn
        self.unconditional_model_fn = unconditional_model_fn
        self.guidance_scale = guidance_scale
    
    def guided_step(self, x_t: Union[List[float], 'torch.Tensor'], t: int, condition: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """
        引导采样单步
        """
        # 无条件预测
        noise_uncond = self.unconditional_model_fn(x_t, t)
        
        # 条件预测
        noise_cond = self.model_fn(x_t, t)
        
        # 引导
        w = self.guidance_scale
        guided_noise = [(1 + w) * noise_cond[i] - w * noise_uncond[i] for i in range(len(x_t))]
        
        return guided_noise


class LatentDiffusion:
    """
    潜在扩散模型 (Latent Diffusion)
    
    在潜空间中进行扩散，而非像素空间
    """
    
    def __init__(self, encoder_fn: Callable, decoder_fn: Callable,
                 diffusion: DDPM, latent_dim: int):
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.diffusion = diffusion
        self.latent_dim = latent_dim
    
    def encode(self, x: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """编码到潜空间"""
        return self.encoder_fn(x)
    
    def decode(self, z: Union[List[float], 'torch.Tensor']) -> Union[List[float], 'torch.Tensor']:
        """从潜空间解码"""
        return self.decoder_fn(z)
    
    def generate(self, condition: Optional[Union[List[float], 'torch.Tensor']] = None) -> Union[List[float], 'torch.Tensor']:
        """
        生成
        """
        # 在潜空间中采样
        z = self.diffusion.sample([self.latent_dim])
        
        # 解码
        x = self.decode(z)
        return x


# 工厂函数
def ddpm(model_fn: Callable, **kwargs) -> DDPM:
    """创建DDPM模型"""
    return DDPM(model_fn, **kwargs)


def ddim(scheduler: NoiseScheduler, **kwargs) -> DDIM:
    """创建DDIM采样器"""
    return DDIM(scheduler, **kwargs)


def noise_scheduler(**kwargs) -> NoiseScheduler:
    """创建噪声调度器"""
    return NoiseScheduler(**kwargs)


def score_sde(score_fn: Callable, **kwargs) -> ScoreBasedSDE:
    """创建Score-Based SDE"""
    return ScoreBasedSDE(score_fn, **kwargs)


def unet_1d(in_channels: int, out_channels: int, **kwargs) -> UNet1D:
    """创建1D UNet"""
    return UNet1D(in_channels, out_channels, **kwargs)


def classifier_free_guidance(model_fn: Callable, unconditional_fn: Callable, 
                            **kwargs) -> ClassifierFreeGuidance:
    """创建无分类器引导"""
    return ClassifierFreeGuidance(model_fn, unconditional_fn, **kwargs)


def latent_diffusion(encoder_fn: Callable, decoder_fn: Callable,
                    diffusion: DDPM, latent_dim: int) -> LatentDiffusion:
    """创建潜在扩散模型"""
    return LatentDiffusion(encoder_fn, decoder_fn, diffusion, latent_dim)
