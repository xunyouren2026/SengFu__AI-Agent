"""
Infini-Attention - 无限上下文注意力机制（PyTorch真实实现）

Google 2024论文实现，通过压缩记忆实现O(1)复杂度无限上下文
集成到 transformers 注意力机制

作者: UFO Framework Team
"""

import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InfiniAttentionConfig:
    """Infini-Attention配置"""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # GQA支持
    head_dim: int = 64

    # 记忆配置
    memory_size: int = 512  # 压缩记忆的大小
    compression_ratio: float = 0.1

    # 注意力配置
    attention_dropout: float = 0.0
    use_delta_rule: bool = True  # 使用增量更新规则

    # 门控配置
    use_gating: bool = True
    gate_init: float = 0.0

    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads


class CompressiveMemory(nn.Module):
    """
    压缩记忆矩阵 - PyTorch实现

    核心思想：将历史信息压缩为固定大小的矩阵
    使用增量更新规则保持记忆的时效性
    """

    def __init__(self, config: InfiniAttentionConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.memory_size
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.compression_ratio = config.compression_ratio

        # 压缩记忆矩阵 [memory_size, head_dim]
        self.register_buffer(
            "memory",
            torch.zeros(config.memory_size, config.head_dim, dtype=config.dtype)
        )

        # 访问计数（用于LRU替换）
        self.register_buffer(
            "access_count",
            torch.zeros(config.memory_size, dtype=torch.float32)
        )

        # 归一化项（用于稳定训练）
        self.register_buffer(
            "normalizer",
            torch.ones(config.memory_size, dtype=torch.float32)
        )

        # 统计信息
        self.stats = {
            'total_updates': 0,
            'total_retrievals': 0,
            'memory_utilization': 0.0,
        }

    def reset_memory(self):
        """重置记忆"""
        self.memory.zero_()
        self.access_count.zero_()
        self.normalizer.fill_(1.0)
        self.stats = {
            'total_updates': 0,
            'total_retrievals': 0,
            'memory_utilization': 0.0,
        }

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> None:
        """
        更新记忆（增量式）

        Args:
            key: 键 [batch_size, seq_len, head_dim]
            value: 值 [batch_size, seq_len, head_dim]
            importance: 重要性权重 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = key.shape

        # 展平batch和seq维度
        key_flat = key.reshape(-1, self.head_dim)  # [B*L, head_dim]
        value_flat = value.reshape(-1, self.head_dim)  # [B*L, head_dim]

        if importance is not None:
            importance_flat = importance.reshape(-1)  # [B*L]
        else:
            importance_flat = torch.ones(key_flat.size(0), device=key.device)

        # 增量更新
        for i in range(key_flat.size(0)):
            k = key_flat[i]  # [head_dim]
            v = value_flat[i]  # [head_dim]
            imp = importance_flat[i].item()

            # 找到最不重要的记忆槽位（LRU策略）
            min_idx = torch.argmin(self.access_count).item()

            # 增量更新记忆
            # M_new = M_old * (1 - beta) + v * beta * imp
            beta = self.compression_ratio
            self.memory[min_idx] = (
                self.memory[min_idx] * (1 - beta) +
                v * beta * imp
            )

            # 更新归一化项
            self.normalizer[min_idx] = (
                self.normalizer[min_idx] * (1 - beta) +
                beta * imp
            )

            # 更新访问计数
            self.access_count[min_idx] += imp

        self.stats['total_updates'] += 1
        self.stats['memory_utilization'] = (self.access_count > 0).float().mean().item()

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 32
    ) -> torch.Tensor:
        """
        检索记忆

        Args:
            query: 查询 [batch_size, seq_len, head_dim]
            top_k: 检索数量

        Returns:
            检索结果 [batch_size, seq_len, head_dim]
        """
        batch_size, seq_len, _ = query.shape

        # 计算查询与记忆的相似度
        # query: [B, L, head_dim], memory: [M, head_dim]
        similarities = torch.matmul(query, self.memory.T)  # [B, L, M]

        # 应用归一化
        normalized_memory = self.memory / (self.normalizer.unsqueeze(1) + 1e-8)

        # Top-k检索
        top_k = min(top_k, self.memory_size)
        topk_values, topk_indices = torch.topk(similarities, top_k, dim=-1)  # [B, L, K]

        # 计算注意力权重
        attn_weights = F.softmax(topk_values / math.sqrt(self.head_dim), dim=-1)  # [B, L, K]

        # 加权求和
        # gathered: [B, L, K, head_dim]
        gathered = F.embedding(topk_indices, normalized_memory)  # [B, L, K, head_dim]

        # 加权求和
        result = torch.sum(
            gathered * attn_weights.unsqueeze(-1),  # [B, L, K, head_dim]
            dim=-2  # sum over K
        )  # [B, L, head_dim]

        self.stats['total_retrievals'] += 1

        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return self.stats.copy()


class InfiniAttention(nn.Module):
    """
    Infini-Attention真实PyTorch实现

    公式: A = softmax(QK^T / sqrt(d)) V + σ(Q) M

    其中M是压缩记忆矩阵
    """

    def __init__(self, config: InfiniAttentionConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Q, K, V投影
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

        # 压缩记忆（每个头一个记忆）
        self.memories = nn.ModuleList([
            CompressiveMemory(config) for _ in range(self.num_key_value_heads)
        ])

        # 门控参数（用于平衡局部注意力和记忆注意力）
        if config.use_gating:
            self.gate = nn.Parameter(torch.full((self.num_key_value_heads,), config.gate_init))
        else:
            self.register_buffer('gate', torch.zeros(self.num_key_value_heads))

        # 记忆投影
        self.memory_proj = nn.Linear(config.head_dim, config.head_dim, bias=False)

        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # 统计
        self.stats = {
            'total_forward': 0,
            'avg_memory_contribution': 0.0,
        }

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)
        nn.init.normal_(self.memory_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_memory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len]
            position_ids: [batch_size, seq_len]
            past_key_value: 缓存的KV
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存
            use_memory: 是否使用压缩记忆

        Returns:
            (输出, 注意力权重, KV缓存)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V投影
        query_states = self.q_proj(hidden_states)  # [B, L, num_heads * head_dim]
        key_states = self.k_proj(hidden_states)    # [B, L, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [B, L, num_kv_heads * head_dim]

        # 重塑为多头
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, num_heads, L, head_dim]

        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [B, num_kv_heads, L, head_dim]

        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [B, num_kv_heads, L, head_dim]

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        kv_seq_len = key_states.shape[2]

        # 保存KV缓存
        present_key_value = (key_states, value_states) if use_cache else None

        # 重复K, V以匹配Q的头数（GQA）
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # 标准注意力计算
        attn_output, attn_weights = self._compute_attention(
            query_states, key_states, value_states, attention_mask
        )

        # 记忆注意力
        memory_contribution = 0.0
        if use_memory:
            memory_output = self._compute_memory_attention(query_states)

            # 门控融合
            if self.config.use_gating:
                gate = torch.sigmoid(self.gate).view(1, -1, 1, 1)  # [1, num_kv_heads, 1, 1]
                attn_output = (1 - gate) * attn_output + gate * memory_output
                memory_contribution = gate.mean().item()
            else:
                attn_output = attn_output + memory_output

            # 更新记忆
            self._update_memory(key_states, value_states)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.o_proj(attn_output)

        # 更新统计
        self.stats['total_forward'] += 1
        n = self.stats['total_forward']
        self.stats['avg_memory_contribution'] = (
            (n - 1) * self.stats['avg_memory_contribution'] + memory_contribution
        ) / n

        return output, attn_weights, present_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        重复KV以匹配Q的头数（GQA）

        Args:
            hidden_states: [B, num_kv_heads, L, head_dim]
            n_rep: 重复次数

        Returns:
            [B, num_heads, L, head_dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算标准注意力

        Args:
            query: [B, num_heads, L, head_dim]
            key: [B, num_heads, L, head_dim]
            value: [B, num_heads, L, head_dim]
            attention_mask: [B, 1, L, L]

        Returns:
            (输出, 注意力权重)
        """
        # 计算注意力分数
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, num_heads, L, L]

        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, value)  # [B, num_heads, L, head_dim]

        return attn_output, attn_weights

    def _compute_memory_attention(self, query: torch.Tensor) -> torch.Tensor:
        """
        计算记忆注意力

        Args:
            query: [B, num_heads, L, head_dim]

        Returns:
            [B, num_heads, L, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        # 按KV头分组
        num_groups = self.num_key_value_heads
        heads_per_group = num_heads // num_groups

        memory_outputs = []

        for g in range(num_groups):
            # 获取该组的查询
            group_query = query[:, g * heads_per_group:(g + 1) * heads_per_group, :, :]
            # [B, heads_per_group, L, head_dim]

            # 重塑以检索记忆
            group_query_flat = group_query.transpose(1, 2).reshape(-1, head_dim)
            # [B * heads_per_group * L, head_dim]

            # 从记忆检索
            memory_retrieved = self.memories[g].retrieve(
                group_query_flat.view(batch_size, heads_per_group * seq_len, head_dim),
                top_k=32
            )
            # [B, heads_per_group * L, head_dim]

            # 投影
            memory_retrieved = self.memory_proj(memory_retrieved)

            # 重塑回原始形状
            memory_retrieved = memory_retrieved.view(batch_size, heads_per_group, seq_len, head_dim)

            memory_outputs.append(memory_retrieved)

        # 合并所有组
        memory_output = torch.cat(memory_outputs, dim=1)  # [B, num_heads, L, head_dim]

        return memory_output

    def _update_memory(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        更新压缩记忆

        Args:
            key: [B, num_heads, L, head_dim]
            value: [B, num_heads, L, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = key.shape

        # 按KV头分组
        num_groups = self.num_key_value_heads
        heads_per_group = num_heads // num_groups

        for g in range(num_groups):
            # 获取该组的key和value
            group_key = key[:, g * heads_per_group:(g + 1) * heads_per_group, :, :]
            group_value = value[:, g * heads_per_group:(g + 1) * heads_per_group, :, :]

            # 重塑以更新记忆
            group_key_flat = group_key.transpose(1, 2).reshape(batch_size, seq_len, -1)
            group_value_flat = group_value.transpose(1, 2).reshape(batch_size, seq_len, -1)

            # 更新记忆
            self.memories[g].update(group_key_flat, group_value_flat)

    def reset_memory(self):
        """重置所有记忆"""
        for memory in self.memories:
            memory.reset_memory()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        memory_stats = [m.get_stats() for m in self.memories]
        return {
            **self.stats,
            'memory_stats': memory_stats,
        }


class InfiniTransformerBlock(nn.Module):
    """
    Infini-Attention Transformer块

    完整的Transformer块，包含Infini-Attention和前馈网络
    """

    def __init__(self, config: InfiniAttentionConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # 输入层归一化
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # Infini-Attention
        self.self_attn = InfiniAttention(config, layer_idx)

        # 后注意力层归一化
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 前馈网络
        self.mlp = self._build_mlp()

    def _build_mlp(self) -> nn.Module:
        """构建前馈网络"""
        intermediate_size = self.config.hidden_size * 4
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, self.config.hidden_size),
            nn.Dropout(self.config.attention_dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_memory: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播

        Args:
            hidden_states: [B, L, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: 缓存的KV
            output_attentions: 是否输出注意力
            use_cache: 是否使用缓存
            use_memory: 是否使用压缩记忆

        Returns:
            (输出, 注意力权重, KV缓存)
        """
        residual = hidden_states

        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # Infini-Attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            use_memory=use_memory,
        )

        # 残差连接
        hidden_states = residual + attn_output

        # 前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, present_key_value


class InfiniAttentionModel(nn.Module):
    """
    完整的Infini-Attention模型

    支持无限上下文的语言模型
    """

    def __init__(self, config: InfiniAttentionConfig, num_layers: int = 12):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        # 词嵌入
        self.embed_tokens = nn.Embedding(50000, config.hidden_size)

        # Transformer层
        self.layers = nn.ModuleList([
            InfiniTransformerBlock(config, i) for i in range(num_layers)
        ])

        # 最终层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, 50000, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        use_memory: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            position_ids: [B, L]
            past_key_values: 缓存的KV列表
            use_cache: 是否使用缓存
            use_memory: 是否使用压缩记忆
            output_attentions: 是否输出注意力
            output_hidden_states: 是否输出隐藏状态

        Returns:
            包含logits和其他输出的字典
        """
        batch_size, seq_len = input_ids.shape

        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)

        # 处理位置ID
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # 处理注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=input_ids.device
            )

        # 转换为因果掩码
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask, seq_len, inputs_embeds.dtype
        )

        # Transformer层
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                use_memory=use_memory,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

            if use_cache:
                next_cache += (layer_outputs[2],)

        # 最终层归一化
        hidden_states = self.norm(hidden_states)

        # 语言模型头
        logits = self.lm_head(hidden_states)

        return {
            'logits': logits,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def _prepare_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """准备因果注意力掩码"""
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=dtype),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展注意力掩码
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            causal_mask = causal_mask + (1.0 - mask.float()) * float('-inf')

        return causal_mask

    def reset_memory(self):
        """重置所有层的记忆"""
        for layer in self.layers:
            layer.self_attn.reset_memory()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        layer_stats = [layer.self_attn.get_stats() for layer in self.layers]
        return {
            'num_layers': self.num_layers,
            'layer_stats': layer_stats,
        }


def create_infini_attention(
    hidden_size: int = 768,
    num_heads: int = 12,
    memory_size: int = 512,
    **kwargs
) -> InfiniAttention:
    """
    便捷函数：创建Infini-Attention

    Args:
        hidden_size: 隐藏维度
        num_heads: 注意力头数
        memory_size: 记忆大小
        **kwargs: 其他配置参数

    Returns:
        InfiniAttention实例
    """
    config = InfiniAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        memory_size=memory_size,
        **kwargs
    )
    return InfiniAttention(config)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Infini-Attention PyTorch实现测试")
    print("=" * 60)

    # 测试配置
    config = InfiniAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        memory_size=256,
        device="cpu",
        dtype=torch.float32,
    )

    # 创建Infini-Attention
    print("\n[1] 创建Infini-Attention")
    attention = InfiniAttention(config)
    print(f"  参数数量: {sum(p.numel() for p in attention.parameters()):,}")

    # 测试前向传播
    print("\n[2] 测试前向传播")
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, attn_weights, _ = attention(hidden_states, use_memory=True)
    print(f"  输入形状: {hidden_states.shape}")
    print(f"  输出形状: {output.shape}")

    # 测试长序列
    print("\n[3] 测试长序列处理")
    for length in [128, 256, 512, 1024]:
        long_hidden = torch.randn(1, length, config.hidden_size)
        output, _, _ = attention(long_hidden, use_memory=True)
        print(f"  序列长度 {length}: 输出形状 {output.shape}")

    # 测试记忆统计
    print("\n[4] 测试记忆统计")
    stats = attention.get_stats()
    print(f"  前向传播次数: {stats['total_forward']}")
    print(f"  平均记忆贡献: {stats['avg_memory_contribution']:.4f}")

    # 测试完整模型
    print("\n[5] 测试完整模型")
    model = InfiniAttentionModel(config, num_layers=4)
    input_ids = torch.randint(0, 50000, (2, 32))
    outputs = model(input_ids, use_memory=True)
    print(f"  输入形状: {input_ids.shape}")
    print(f"  Logits形状: {outputs['logits'].shape}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
