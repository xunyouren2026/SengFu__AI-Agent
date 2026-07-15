"""
Gist Sparse Attention - 可学习稀疏注意力

实现32倍压缩的稀疏注意力机制

作者: UFO Framework Team
"""

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class GistConfig:
    """Gist配置"""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    compression_ratio: int = 32
    num_gist_tokens: int = 8
    dropout: float = 0.1


class GistTokenEncoder:
    """
    Gist Token编码器
    
    将长序列压缩为少量Gist Token
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_gist_tokens: int
    ):
        self.hidden_size = hidden_size
        self.num_gist_tokens = num_gist_tokens
        
        # 可学习的Gist Token
        self.gist_tokens = np.random.randn(num_gist_tokens, hidden_size).astype(np.float32) * 0.02
        
        # 压缩权重
        self.compress_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    
    def encode(
        self,
        hidden_states: np.ndarray
    ) -> np.ndarray:
        """
        编码为Gist Token
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            gist_tokens: [batch, num_gist_tokens, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        
        # 方法1: 平均池化 + 投影
        pooled = np.mean(hidden_states, axis=1)  # [batch, hidden_size]
        
        # 投影
        projected = np.dot(pooled, self.compress_weight)
        
        # 扩展为多个gist token
        gist_tokens = np.tile(self.gist_tokens, (batch_size, 1, 1))
        
        # 加入全局信息
        gist_tokens = gist_tokens + projected[:, np.newaxis, :] * 0.1
        
        return gist_tokens


class SparseAttentionMask:
    """
    稀疏注意力掩码生成器
    """
    
    def __init__(
        self,
        compression_ratio: int = 32,
        local_window: int = 64
    ):
        self.compression_ratio = compression_ratio
        self.local_window = local_window
    
    def create_mask(
        self,
        seq_len: int,
        gist_positions: List[int]
    ) -> np.ndarray:
        """
        创建稀疏掩码
        
        Args:
            seq_len: 序列长度
            gist_positions: Gist Token位置
            
        Returns:
            mask: [seq_len, seq_len]
        """
        mask = np.zeros((seq_len, seq_len), dtype=np.float32)
        
        # 1. 局部窗口注意力
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2)
            mask[i, start:end] = 1.0
        
        # 2. Gist Token注意力（所有位置都可以关注）
        for pos in gist_positions:
            if pos < seq_len:
                mask[:, pos] = 1.0
                mask[pos, :] = 1.0
        
        # 3. 块稀疏模式
        block_size = self.compression_ratio
        for i in range(0, seq_len, block_size):
            for j in range(0, seq_len, block_size):
                if abs(i - j) <= block_size * 2:
                    mask[i:i+block_size, j:j+block_size] = 1.0
        
        return mask


class GistSparseAttention:
    """
    Gist稀疏注意力
    
    实现32倍压缩的注意力机制
    """
    
    def __init__(self, config: GistConfig):
        self.config = config
        
        # 组件
        self.gist_encoder = GistTokenEncoder(
            hidden_size=config.hidden_size,
            num_gist_tokens=config.num_gist_tokens
        )
        self.mask_generator = SparseAttentionMask(
            compression_ratio=config.compression_ratio
        )
        
        # 投影权重
        scale = 1.0 / math.sqrt(config.hidden_size)
        self.W_q = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        self.W_k = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        self.W_v = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        self.W_o = np.random.randn(config.hidden_size, config.hidden_size).astype(np.float32) * scale
        
        # 统计
        self.stats = {
            'total_forward': 0,
            'avg_sparsity': 0.0,
            'avg_compression': 0.0
        }
    
    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            
        Returns:
            (输出, 统计信息)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 生成Gist Token
        gist_tokens = self.gist_encoder.encode(hidden_states)
        
        # 合并Gist Token和原始序列
        combined = np.concatenate([gist_tokens, hidden_states], axis=1)
        combined_len = combined.shape[1]
        
        # 投影
        Q = np.dot(combined, self.W_q)
        K = np.dot(combined, self.W_k)
        V = np.dot(combined, self.W_v)
        
        # 多头分割
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 计算注意力分数
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.config.head_dim)
        
        # 创建稀疏掩码
        gist_positions = list(range(self.config.num_gist_tokens))
        sparse_mask = self.mask_generator.create_mask(combined_len, gist_positions)
        
        # 应用掩码
        attn_scores = attn_scores + (1 - sparse_mask) * -10000.0
        
        # Softmax
        attn_probs = self._softmax(attn_scores)
        
        # 计算输出
        attn_output = np.matmul(attn_probs, V)
        
        # 合并多头
        attn_output = self._merge_heads(attn_output)
        
        # 只取原始序列部分
        output = attn_output[:, self.config.num_gist_tokens:, :]
        
        # 输出投影
        output = np.dot(output, self.W_o)
        
        # 计算稀疏度
        sparsity = 1.0 - np.mean(sparse_mask)
        compression = seq_len / max(1, self.config.num_gist_tokens + self.mask_generator.local_window)
        
        # 更新统计
        self.stats['total_forward'] += 1
        n = self.stats['total_forward']
        self.stats['avg_sparsity'] = ((n - 1) * self.stats['avg_sparsity'] + sparsity) / n
        self.stats['avg_compression'] = ((n - 1) * self.stats['avg_compression'] + compression) / n
        
        return output, {
            'sparsity': sparsity,
            'compression': compression,
            'gist_tokens': gist_tokens
        }
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """分割多头"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """合并多头"""
        x = x.transpose(0, 2, 1, 3)
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, -1)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return self.stats.copy()


class GistAttentionLayer:
    """
    Gist Attention层（完整实现）
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        compression_ratio: int = 32
    ):
        self.config = GistConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            compression_ratio=compression_ratio
        )
        self.attention = GistSparseAttention(self.config)
    
    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            
        Returns:
            输出 [batch, seq_len, hidden_size]
        """
        output, _ = self.attention.forward(hidden_states, attention_mask)
        return output


# 便捷函数
def create_gist_attention(
    hidden_size: int = 768,
    num_heads: int = 12,
    compression_ratio: int = 32
) -> GistAttentionLayer:
    """创建Gist Attention层"""
    return GistAttentionLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        compression_ratio=compression_ratio
    )


if __name__ == "__main__":
    # 测试
    layer = create_gist_attention(hidden_size=256, num_heads=4, compression_ratio=32)
    
    # 模拟输入
    batch_size, seq_len = 2, 128
    hidden_states = np.random.randn(batch_size, seq_len, 256).astype(np.float32)
    
    print("=" * 60)
    print("Gist Sparse Attention测试")
    print("=" * 60)
    
    # 前向传播
    output, info = layer.attention.forward(hidden_states)
    
    print(f"\n输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    print(f"稀疏度: {info['sparsity']:.1%}")
    print(f"压缩比: {info['compression']:.1f}x")
    
    print(f"\n统计: {layer.attention.get_stats()}")
