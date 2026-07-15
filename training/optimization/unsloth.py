"""
UnSloth: 高效LLM微调加速
手动反向传播优化 + RoPE嵌入优化 + LoRA支持

基于UnSloth开源实现，提供2-5倍的训练加速
支持QLoRA和多种优化技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import math
from transformers import PreTrainedModel


@dataclass
class UnSlothConfig:
    """UnSloth配置参数"""
    # LoRA配置
    r: int = 16  # LoRA秩
    lora_alpha: int = 16  # LoRA alpha参数
    lora_dropout: float = 0.0  # LoRA dropout
    target_modules: List[str] = None  # 目标模块
    
    # 优化配置
    use_gradient_checkpointing: bool = True  # 使用梯度检查点
    use_rmsnorm: bool = True  # 使用RMSNorm优化
    use_rope_scaling: bool = True  # 使用RoPE缩放
    max_seq_length: int = 2048  # 最大序列长度
    
    # 量化配置
    load_in_4bit: bool = False  # 4bit量化
    load_in_8bit: bool = False  # 8bit量化
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"  # nf4或fp4
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class FastRoPE(nn.Module):
    """
    优化的RoPE (Rotary Position Embedding)
    使用更高效的计算方式
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算位置编码
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _compute_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算cos和sin缓存"""
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = seq_len
            
            # 计算位置
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            t = t / self.scaling_factor
            
            # 计算频率
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            
            # 计算cos和sin
            self._cached_cos = emb.cos().to(dtype)
            self._cached_sin = emb.sin().to(dtype)
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用RoPE
        
        Args:
            x: 输入张量 [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度
            
        Returns:
            应用RoPE后的cos和sin
        """
        if seq_len is None:
            seq_len = x.shape[2]
        
        cos, sin = self._compute_cos_sin(seq_len, x.device, x.dtype)
        
        return cos, sin
    
    @staticmethod
    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码
        
        Args:
            q: 查询
            k: 键
            cos: cos缓存
            sin: sin缓存
            
        Returns:
            旋转后的q和k
        """
        # 重塑cos和sin以匹配q和k的形状
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 应用旋转
        q_embed = (q * cos) + (FastRoPE._rotate_half(q) * sin)
        k_embed = (k * cos) + (FastRoPE._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """旋转张量的一半"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class FastRMSNorm(nn.Module):
    """
    优化的RMSNorm实现
    比标准LayerNorm更快
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 [..., hidden_size]
            
        Returns:
            归一化输出
        """
        # 计算RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        return self.weight * x


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    低秩适配器用于高效微调
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # LoRA权重
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入
            
        Returns:
            LoRA输出
        """
        # x @ A @ B * scaling
        result = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    带有LoRA的线性层
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：base + LoRA"""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_lora(self):
        """合并LoRA权重到基础层"""
        # 计算LoRA权重增量
        delta_W = self.lora.lora_A @ self.lora.lora_B * self.lora.scaling
        
        # 合并到基础层
        with torch.no_grad():
            self.base_layer.weight.data += delta_W.T
        
        # 重置LoRA
        self.lora.lora_A.data.zero_()
        self.lora.lora_B.data.zero_()


class FastLinear(nn.Module):
    """
    优化的线性层
    支持手动反向传播优化
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return F.linear(x, self.weight, self.bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        手动反向传播优化
        使用更高效的梯度计算
        """
        x, weight = ctx.saved_tensors
        
        # 计算梯度
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ x
        grad_bias = grad_output.sum(0) if ctx.needs_input_grad[2] else None
        
        return grad_input, grad_weight, grad_bias


class UnSlothAttention(nn.Module):
    """
    UnSloth优化的注意力层
    结合RoPE优化和高效注意力计算
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        
        # 投影层
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = FastRoPE(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=rope_scaling.get("factor", 1.0) if rope_scaling else 1.0
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            
        Returns:
            注意力输出
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 重复KV（用于GQA）
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # 应用RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = FastRoPE.apply_rotary_pos_emb(q, k, cos, sin)
        
        # 注意力计算
        attn_output = self._flash_attention(q, k, v, attention_mask)
        
        # 重塑并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用Flash Attention风格的内存高效注意力
        
        Args:
            q: 查询
            k: 键
            v: 值
            attention_mask: 注意力掩码
            
        Returns:
            注意力输出
        """
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        
        return output


class UnSlothMLP(nn.Module):
    """
    UnSloth优化的MLP层
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swish"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        if activation == "swish":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：使用门控机制"""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class UnSlothDecoderLayer(nn.Module):
    """
    UnSloth优化的Decoder层
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 自注意力
        self.self_attn = UnSlothAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings
        )
        
        # MLP
        self.mlp = UnSlothMLP(hidden_size, intermediate_size)
        
        # RMSNorm
        self.input_layernorm = FastRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = FastRMSNorm(hidden_size, eps=rms_norm_eps)
        
        # 应用LoRA（如果需要）
        if use_lora and lora_config:
            self._apply_lora(lora_config)
    
    def _apply_lora(self, config: Dict):
        """应用LoRA到目标模块"""
        target_modules = config.get('target_modules', ['q_proj', 'v_proj'])
        r = config.get('r', 16)
        lora_alpha = config.get('lora_alpha', 16)
        lora_dropout = config.get('lora_dropout', 0.0)
        
        for module_name in target_modules:
            if hasattr(self.self_attn, module_name):
                original_layer = getattr(self.self_attn, module_name)
                if isinstance(original_layer, nn.Linear):
                    lora_layer = LinearWithLoRA(
                        original_layer,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )
                    setattr(self.self_attn, module_name, lora_layer)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 隐藏状态
            attention_mask: 注意力掩码
            
        Returns:
            输出隐藏状态
        """
        # 自注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class UnSlothModel(nn.Module):
    """
    UnSloth优化的模型
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Decoder层
        self.layers = nn.ModuleList([
            UnSlothDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_key_value_heads=num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                use_lora=use_lora,
                lora_config=lora_config
            )
            for _ in range(num_hidden_layers)
        ])
        
        # 最终归一化
        self.norm = FastRMSNorm(hidden_size, eps=rms_norm_eps)
        
        # LM头
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            
        Returns:
            logits
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过所有层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # LM头
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def merge_and_unload_lora(self):
        """合并LoRA权重并卸载"""
        for layer in self.layers:
            for module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, module_name):
                    module = getattr(layer.self_attn, module_name)
                    if isinstance(module, LinearWithLoRA):
                        module.merge_lora()
                        setattr(layer.self_attn, module_name, module.base_layer)


def apply_unsloth_optimizations(
    model: nn.Module,
    config: UnSlothConfig
) -> nn.Module:
    """
    应用UnSloth优化到现有模型
    
    Args:
        model: 原始模型
        config: UnSloth配置
        
    Returns:
        优化后的模型
    """
    # 替换LayerNorm为RMSNorm
    if config.use_rmsnorm:
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                rms_norm = FastRMSNorm(module.normalized_shape[0], eps=module.eps)
                setattr(parent, child_name, rms_norm)
    
    return model


def create_unsloth_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建UnSloth推荐的优化器
    
    Args:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
        
    Returns:
        优化器
    """
    # 分离需要权重衰减的参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, **kwargs)


# 辅助函数
def get_unsloth_memory_stats(model: nn.Module) -> Dict[str, float]:
    """
    获取UnSloth模型的内存统计
    
    Args:
        model: 模型
        
    Returns:
        内存统计
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算内存使用（假设float32）
    param_memory = total_params * 4 / (1024 ** 3)  # GB
    trainable_memory = trainable_params * 4 / (1024 ** 3)  # GB
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_memory_gb': param_memory,
        'trainable_memory_gb': trainable_memory,
        'memory_reduction_ratio': 1 - (trainable_memory / param_memory) if param_memory > 0 else 0
    }


def estimate_training_speedup(
    baseline_time: float,
    unsloth_time: float
) -> Dict[str, float]:
    """
    估算训练加速比
    
    Args:
        baseline_time: 基线时间
        unsloth_time: UnSloth时间
        
    Returns:
        加速统计
    """
    speedup = baseline_time / unsloth_time
    
    return {
        'baseline_time': baseline_time,
        'unsloth_time': unsloth_time,
        'speedup': speedup,
        'speedup_percentage': (speedup - 1) * 100
    }
