"""
序列模型模块 - Sequence Models
实现LSTM、GRU、Transformer、State Space Models等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass

# ==================== LSTM变体 ====================

class LSTM(nn.Module):
    """标准LSTM"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.layers.append(
                LSTMCell(layer_input_size, hidden_size, bias)
            )
            if bidirectional:
                self.layers.append(
                    LSTMCell(layer_input_size, hidden_size, bias)
                )
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        if self.batch_first:
            batch_size = x.size(0)
            seq_len = x.size(1)
        else:
            batch_size = x.size(1)
            seq_len = x.size(0)
            x = x.transpose(0, 1)
        
        # 初始化隐藏状态
        if hx is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
        else:
            h_0, c_0 = hx
        
        # 处理每一层
        h_n = []
        c_n = []
        
        for layer_idx in range(self.num_layers):
            layer_offset = layer_idx * self.num_directions
            
            if self.bidirectional:
                # 前向
                forward_layer = self.layers[layer_offset]
                forward_h = h_0[layer_offset]
                forward_c = c_0[layer_offset]
                
                # 反向
                backward_layer = self.layers[layer_offset + 1]
                backward_h = h_0[layer_offset + 1]
                backward_c = c_0[layer_offset + 1]
                
                # 前向传播
                forward_outputs = []
                h, c = forward_h, forward_c
                for t in range(seq_len):
                    h, c = forward_layer(x[:, t, :], (h, c))
                    forward_outputs.append(h)
                    if self.dropout > 0 and layer_idx < self.num_layers - 1:
                        h = F.dropout(h, p=self.dropout, training=self.training)
                
                # 反向传播
                backward_outputs = []
                h, c = backward_h, backward_c
                for t in range(seq_len - 1, -1, -1):
                    h, c = backward_layer(x[:, t, :], (h, c))
                    backward_outputs.insert(0, h)
                    if self.dropout > 0 and layer_idx < self.num_layers - 1:
                        h = F.dropout(h, p=self.dropout, training=self.training)
                
                # 拼接
                outputs = torch.stack([
                    torch.cat([f, b], dim=-1)
                    for f, b in zip(forward_outputs, backward_outputs)
                ], dim=1)
                
                h_n.extend([forward_outputs[-1], backward_outputs[0]])
                c_n.extend([forward_outputs[-1], backward_outputs[0]])
            else:
                layer = self.layers[layer_idx]
                h = h_0[layer_offset]
                c = c_0[layer_offset]
                
                outputs = []
                for t in range(seq_len):
                    h, c = layer(x[:, t, :], (h, c))
                    outputs.append(h)
                    if self.dropout > 0 and layer_idx < self.num_layers - 1:
                        h = F.dropout(h, p=self.dropout, training=self.training)
                
                outputs = torch.stack(outputs, dim=1)
                h_n.append(h)
                c_n.append(c)
            
            x = outputs
        
        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, (h_n, c_n)


class LSTMCell(nn.Module):
    """LSTM单元"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hx
        
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h, self.weight_hh, self.bias_hh)
        
        i, f, g, o = gates.chunk(4, dim=-1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class GRU(nn.Module):
    """GRU"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            self.layers.append(GRUCell(layer_input_size, hidden_size, bias))
            if bidirectional:
                self.layers.append(GRUCell(layer_input_size, hidden_size, bias))
    
    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        if self.batch_first:
            batch_size = x.size(0)
            seq_len = x.size(1)
        else:
            batch_size = x.size(1)
            seq_len = x.size(0)
            x = x.transpose(0, 1)
        
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device)
        
        h_n = []
        
        for layer_idx in range(self.num_layers):
            layer_offset = layer_idx * self.num_directions
            
            if self.bidirectional:
                forward_layer = self.layers[layer_offset]
                backward_layer = self.layers[layer_offset + 1]
                
                forward_h = h_0[layer_offset]
                backward_h = h_0[layer_offset + 1]
                
                forward_outputs = []
                h = forward_h
                for t in range(seq_len):
                    h = forward_layer(x[:, t, :], h)
                    forward_outputs.append(h)
                
                backward_outputs = []
                h = backward_h
                for t in range(seq_len - 1, -1, -1):
                    h = backward_layer(x[:, t, :], h)
                    backward_outputs.insert(0, h)
                
                outputs = torch.stack([
                    torch.cat([f, b], dim=-1)
                    for f, b in zip(forward_outputs, backward_outputs)
                ], dim=1)
                
                h_n.extend([forward_outputs[-1], backward_outputs[0]])
            else:
                layer = self.layers[layer_idx]
                h = h_0[layer_offset]
                
                outputs = []
                for t in range(seq_len):
                    h = layer(x[:, t, :], h)
                    outputs.append(h)
                    if self.dropout > 0 and layer_idx < self.num_layers - 1:
                        h = F.dropout(h, p=self.dropout, training=self.training)
                
                outputs = torch.stack(outputs, dim=1)
                h_n.append(h)
            
            x = outputs
        
        h_n = torch.stack(h_n, dim=0)
        
        if not self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, h_n


class GRUCell(nn.Module):
    """GRU单元"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h, self.weight_hh, self.bias_hh)
        
        r, z, n = gates.chunk(3, dim=-1)
        
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)
        
        h_new = (1 - z) * n + z * h
        
        return h_new


# ==================== Peephole LSTM ====================

class PeepholeLSTMCell(nn.Module):
    """Peephole LSTM单元"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        # Peephole权重
        self.w_peep_i = nn.Parameter(torch.randn(hidden_size))
        self.w_peep_f = nn.Parameter(torch.randn(hidden_size))
        self.w_peep_o = nn.Parameter(torch.randn(hidden_size))
        
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
    
    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = hx
        
        gates = F.linear(x, self.weight_ih) + F.linear(h, self.weight_hh) + self.bias
        
        i, f, g, o = gates.chunk(4, dim=-1)
        
        # Peephole连接
        i = torch.sigmoid(i + self.w_peep_i * c)
        f = torch.sigmoid(f + self.w_peep_f * c)
        g = torch.tanh(g)
        
        c_new = f * c + i * g
        
        o = torch.sigmoid(o + self.w_peep_o * c_new)
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


# ==================== State Space Models ====================

class S4Block(nn.Module):
    """S4 (Structured State Space) 块"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM参数
        self.log_dt = nn.Parameter(torch.randn(d_model))
        self.log_A_real = nn.Parameter(torch.randn(d_model, d_state))
        self.A_imag = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state, 2))
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2))
        self.D = nn.Parameter(torch.randn(d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, d_model = x.shape
        
        # 计算SSM
        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        B = self.B[..., 0] + 1j * self.B[..., 1]
        C = self.C[..., 0] + 1j * self.C[..., 1]
        
        # 离散化
        dA = torch.exp(A * dt.unsqueeze(-1))
        dB = B * dt.unsqueeze(-1)
        
        # 递归计算
        h = x.new_zeros(batch_size, d_model, self.d_state, dtype=torch.complex64)
        outputs = []
        
        for t in range(seq_len):
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y = 2 * (C * h).real.sum(dim=-1) + self.D * x[:, t, :]
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class MambaBlock(nn.Module):
    """Mamba块"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # 投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 卷积
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # A参数
        A = torch.arange(1, d_state + 1).float().unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, d_model = x.shape
        
        # 投影
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # 卷积
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM
        A = -torch.exp(self.A_log.float())
        
        # 计算delta和B, C
        x_dbl = self.x_proj(x_conv)
        delta, BC = x_dbl.split([self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # 选择性扫描（简化实现）
        B = BC
        C = BC
        
        # 递归
        h = x.new_zeros(batch_size, self.d_inner, self.d_state)
        y = []
        
        for t in range(seq_len):
            delta_t = delta[:, t, :].unsqueeze(-1)
            A_t = torch.exp(A.unsqueeze(0) * delta_t)
            B_t = B[:, t, :].unsqueeze(1)
            
            h = A_t * h + B_t * x_conv[:, t, :].unsqueeze(-1)
            
            C_t = C[:, t, :].unsqueeze(1)
            y_t = (C_t * h).sum(dim=-1) + self.D * x_conv[:, t, :]
            y.append(y_t)
        
        y = torch.stack(y, dim=1)
        
        # 门控
        output = y * F.silu(z)
        output = self.out_proj(output)
        
        return output


# ==================== 长序列Transformer ====================

class LongformerAttention(nn.Module):
    """Longformer注意力"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_window: int = 512,
        attention_dilation: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attention_window = attention_window
        self.attention_dilation = attention_dilation
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, hidden_size = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 局部注意力
        output = torch.zeros_like(Q)
        
        for i in range(seq_len):
            # 计算窗口
            window_start = max(0, i - self.attention_window // 2)
            window_end = min(seq_len, i + self.attention_window // 2 + 1)
            
            q_i = Q[:, :, i:i+1, :]
            k_window = K[:, :, window_start:window_end, :]
            v_window = V[:, :, window_start:window_end, :]
            
            attn = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            
            output[:, :, i:i+1, :] = torch.matmul(attn, v_window)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.output(output)
        
        return output


class PerformerAttention(nn.Module):
    """Performer注意力 (线性复杂度)"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_features: int = 256,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_features = num_features
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # 随机特征
        self.register_buffer('projection_matrix', torch.randn(num_features, self.head_dim))
    
    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        """核函数近似"""
        x_proj = torch.matmul(x, self.projection_matrix.t())
        return torch.exp(x_proj - x_proj.max(dim=-1, keepdim=True)[0]) / math.sqrt(self.num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, hidden_size = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 缩放
        Q = Q / math.sqrt(self.head_dim)
        
        # 核变换
        Q_prime = self._kernel(Q)
        K_prime = self._kernel(K)
        
        # 线性注意力
        KV = torch.einsum('bshd,bshm->bhmd', K_prime, V)
        output = torch.einsum('bshd,bhmd->bshm', Q_prime, KV)
        
        # 归一化
        K_sum = K_prime.sum(dim=1, keepdim=True)
        normalizer = torch.einsum('bshd,bhd->bsh', Q_prime, K_sum.squeeze(1))
        output = output / (normalizer.unsqueeze(-1) + 1e-6)
        
        output = output.view(batch_size, seq_len, hidden_size)
        output = self.output(output)
        
        return output


# ==================== 序列到序列模型 ====================

class Seq2Seq(nn.Module):
    """序列到序列模型"""
    
    def __init__(
        self,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 编码器
        self.encoder_embed = nn.Embedding(encoder_vocab_size, hidden_size)
        self.encoder_pos = nn.Parameter(torch.randn(1, 512, hidden_size) * 0.02)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads=8, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 解码器
        self.decoder_embed = nn.Embedding(decoder_vocab_size, hidden_size)
        self.decoder_pos = nn.Parameter(torch.randn(1, 512, hidden_size) * 0.02)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_heads=8, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_size, decoder_vocab_size)
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """编码"""
        x = self.encoder_embed(src)
        x = x + self.encoder_pos[:, :src.size(1), :]
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """解码"""
        x = self.decoder_embed(tgt)
        x = x + self.decoder_pos[:, :tgt.size(1), :]
        
        for layer in self.decoder_layers:
            x = layer(x, memory)
        
        return self.output_proj(x)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播"""
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), memory, memory)[0]
        x = x + self.mlp(self.norm3(x))
        return x


# ==================== 主函数 ====================

def main():
    """测试序列模型"""
    print("序列模型测试")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试LSTM
    print("\n测试LSTM...")
    lstm = LSTM(input_size=64, hidden_size=128, num_layers=2).to(device)
    x = torch.randn(2, 10, 64).to(device)
    output, (h_n, c_n) = lstm(x)
    print(f"LSTM output shape: {output.shape}, hidden shape: {h_n.shape}")
    
    # 测试GRU
    print("\n测试GRU...")
    gru = GRU(input_size=64, hidden_size=128, num_layers=2).to(device)
    output, h_n = gru(x)
    print(f"GRU output shape: {output.shape}, hidden shape: {h_n.shape}")
    
    # 测试Mamba
    print("\n测试Mamba...")
    mamba = MambaBlock(d_model=64, d_state=16).to(device)
    output = mamba(x)
    print(f"Mamba output shape: {output.shape}")
    
    # 测试Performer
    print("\n测试Performer...")
    performer = PerformerAttention(hidden_size=64, num_heads=4).to(device)
    output = performer(x)
    print(f"Performer output shape: {output.shape}")
    
    # 测试Seq2Seq
    print("\n测试Seq2Seq...")
    seq2seq = Seq2Seq(encoder_vocab_size=1000, decoder_vocab_size=1000, hidden_size=128).to(device)
    src = torch.randint(0, 1000, (2, 20)).to(device)
    tgt = torch.randint(0, 1000, (2, 15)).to(device)
    output = seq2seq(src, tgt)
    print(f"Seq2Seq output shape: {output.shape}")
    
    print("\n序列模型测试完成")


if __name__ == "__main__":
    main()
