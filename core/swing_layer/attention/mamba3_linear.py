"""
Mamba-3 Linear Attention - 线性复杂度注意力机制

基于ICLR 2026最新研究，实现二阶精度的状态空间模型。
核心特性：
- O(n)线性复杂度替代O(n²)二次复杂度
- 二阶梯形离散化提高精度
- 支持百万级token长上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class Mamba3Block(nn.Module):
    """
    Mamba-3核心块
    
    使用二阶梯形离散化的选择性状态空间模型。
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
    ):
        """
        初始化Mamba-3块
        
        Args:
            d_model: 模型维度
            d_state: 状态空间维度
            d_conv: 卷积核大小
            expand: 扩展因子
            dt_rank: 时间步长秩
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 输入投影（x到z, x, B, C, Δ）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + d_state * 2 + self.dt_rank, bias=bias)

        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # 激活函数
        self.act = nn.SiLU()

        # 时间步长投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 初始化dt_proj权重
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # 初始化dt_proj偏置
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_proj.bias.data = dt

        # SSM参数A和D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, dim = x.shape

        # 输入投影
        xzBCDelta = self.in_proj(x)  # (batch, seq_len, d_inner*2 + d_state*2 + dt_rank)
        
        # 分割
        xz_chunk = self.d_inner * 2
        x_chunk = xz_chunk + self.d_state
        b_chunk = x_chunk + self.d_state
        
        xz = xzBCDelta[:, :, :xz_chunk]
        B = xzBCDelta[:, :, xz_chunk:x_chunk]
        C = xzBCDelta[:, :, x_chunk:b_chunk]
        delta = xzBCDelta[:, :, b_chunk:]

        # 分割x和z
        x_inner, z = xz.chunk(2, dim=-1)

        # 应用卷积
        x_conv = self.act(self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2))

        # 计算时间步长
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)

        # 离散化SSM参数
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 二阶梯形离散化（Mamba-3核心创新）
        # 比Mamba-2的欧拉离散化更精确
        discrete_A, discrete_B = self._trapezoidal_discretization(A, B, delta)

        # 选择性扫描（使用PyTorch实现简化版）
        y = self._selective_scan(x_conv, discrete_A, discrete_B, C, self.D)

        # 门控机制
        y = y * self.act(z)

        # 输出投影
        output = self.out_proj(y)

        return output

    def _trapezoidal_discretization(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        二阶梯形离散化
        
        比欧拉方法更精确，误差为O(Δt²)而非O(Δt)。
        
        Args:
            A: (d_inner, d_state) 连续时间状态矩阵
            B: (batch, seq_len, d_state) 输入矩阵
            delta: (batch, seq_len, d_inner) 时间步长
        
        Returns:
            discrete_A: 离散化后的A
            discrete_B: 离散化后的B
        """
        # 扩展维度
        delta = delta.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        B = B.unsqueeze(2)  # (batch, seq_len, 1, d_state)

        # 梯形离散化公式
        # Ā = (I + Δt/2 * A)⁻¹ (I - Δt/2 * A)
        # B̄ = (I + Δt/2 * A)⁻¹ Δt B
        
        I = torch.eye(self.d_state, device=A.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 计算(I + Δt/2 * A)
        left = I + delta / 2 * A  # (batch, seq_len, d_inner, d_state, d_state)
        
        # 简化：假设A是对角矩阵
        left_diag = 1 + delta / 2 * A  # (batch, seq_len, d_inner, d_state)
        
        # 计算离散化A
        discrete_A = (1 - delta / 2 * A) / left_diag
        
        # 计算离散化B
        discrete_B = delta * B / left_diag

        return discrete_A.squeeze(2), discrete_B.squeeze(2)

    def _selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """
        选择性扫描
        
        Args:
            x: (batch, seq_len, d_inner)
            A: (batch, seq_len, d_inner, d_state)
            B: (batch, seq_len, d_inner, d_state)
            C: (batch, seq_len, d_state)
            D: (d_inner,)
        
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[-1]

        # 初始化状态
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        ys = []
        
        for t in range(seq_len):
            # 更新状态: h_t = A_t * h_{t-1} + B_t * x_t
            h = A[:, t] * h + B[:, t] * x[:, t:t+1, :].transpose(1, 2)
            
            # 输出: y_t = C_t @ h_t + D * x_t
            y = torch.sum(h * C[:, t:t+1, :].transpose(1, 2), dim=-1)
            y = y + D * x[:, t, :]
            
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        return y


class LinearAttention(nn.Module):
    """
    线性注意力
    
    使用核技巧将注意力复杂度从O(n²)降低到O(n)。
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        feature_dim: int = 64,
        eps: float = 1e-6
    ):
        """
        初始化线性注意力
        
        Args:
            dim: 输入维度
            num_heads: 注意力头数
            qk_dim: Q/K维度
            v_dim: V维度
            feature_dim: 特征映射维度
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim // num_heads
        self.v_dim = v_dim or dim // num_heads
        self.feature_dim = feature_dim
        self.eps = eps
        
        self.q_proj = nn.Linear(dim, num_heads * self.qk_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.qk_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.v_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.v_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.qk_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.qk_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.v_dim).transpose(1, 2)
        
        # 特征映射（使用ReLU核）
        q = F.relu(q) + self.eps
        k = F.relu(k) + self.eps
        
        # 线性注意力计算
        # O(n)复杂度：先计算KV聚合，再与Q相乘
        kv = torch.matmul(k.transpose(-2, -1), v)  # (batch, heads, qk_dim, v_dim)
        z = 1 / (torch.sum(q, dim=-1, keepdim=True) + self.eps)  # (batch, heads, seq_len, 1)
        
        output = torch.matmul(q, kv) * z  # (batch, heads, seq_len, v_dim)
        
        # 重塑并投影
        output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.out_proj(output)
        
        return output


class HybridMambaTransformer(nn.Module):
    """
    混合Mamba-Transformer架构
    
    结合Mamba的线性复杂度和Transformer的表达能力。
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_state: int = 64,
        mamba_ratio: float = 0.5,
    ):
        """
        初始化混合架构
        
        Args:
            d_model: 模型维度
            n_layers: 层数
            n_heads: 注意力头数
            d_state: Mamba状态维度
            mamba_ratio: Mamba层比例
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i < n_layers * mamba_ratio:
                self.layers.append(Mamba3Block(d_model=d_model, d_state=d_state))
            else:
                self.layers.append(LinearAttention(dim=d_model, num_heads=n_heads))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = x + layer(x)  # 残差连接
        return x


# 便捷函数
def create_mamba3_model(
    d_model: int = 512,
    n_layers: int = 12,
    **kwargs
) -> HybridMambaTransformer:
    """便捷函数：创建Mamba-3模型"""
    return HybridMambaTransformer(d_model=d_model, n_layers=n_layers, **kwargs)
