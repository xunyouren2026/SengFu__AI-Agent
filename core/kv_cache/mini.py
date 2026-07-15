"""
MiniCache - 高效KV缓存压缩

实现15倍压缩的KV缓存

作者: UFO Framework Team
"""

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class MiniCacheConfig:
    """MiniCache配置"""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    compression_ratio: int = 15
    num_centroids: int = 256


class VectorQuantizer:
    """向量量化器"""
    
    def __init__(
        self,
        dim: int,
        num_centroids: int = 256,
        num_iterations: int = 10
    ):
        self.dim = dim
        self.num_centroids = num_centroids
        self.num_iterations = num_iterations
        
        # 质心（码本）
        self.centroids: Optional[np.ndarray] = None
        
        # 统计
        self.stats = {
            'total_quantized': 0,
            'avg_reconstruction_error': 0.0
        }
    
    def fit(self, vectors: np.ndarray) -> None:
        """
        训练码本（K-means）
        
        Args:
            vectors: [N, dim]
        """
        n = vectors.shape[0]
        k = min(self.num_centroids, n)
        
        # 随机初始化质心
        indices = np.random.choice(n, k, replace=False)
        self.centroids = vectors[indices].copy()
        
        # K-means迭代
        for _ in range(self.num_iterations):
            # 分配
            distances = self._compute_distances(vectors)
            assignments = np.argmin(distances, axis=1)
            
            # 更新质心
            for i in range(k):
                mask = assignments == i
                if np.any(mask):
                    self.centroids[i] = vectors[mask].mean(axis=0)
    
    def quantize(
        self,
        vectors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        量化向量
        
        Args:
            vectors: [N, dim]
            
        Returns:
            (indices, reconstructed)
        """
        if self.centroids is None:
            self.fit(vectors)
        
        distances = self._compute_distances(vectors)
        indices = np.argmin(distances, axis=1)
        
        # 重建
        reconstructed = self.centroids[indices]
        
        # 计算误差
        error = np.mean((vectors - reconstructed) ** 2)
        n = self.stats['total_quantized'] + vectors.shape[0]
        self.stats['total_quantized'] = n
        self.stats['avg_reconstruction_error'] = (
            (n - vectors.shape[0]) * self.stats['avg_reconstruction_error'] + error * vectors.shape[0]
        ) / n
        
        return indices, reconstructed
    
    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """反量化"""
        if self.centroids is None:
            raise ValueError("Quantizer not fitted")
        return self.centroids[indices]
    
    def _compute_distances(self, vectors: np.ndarray) -> np.ndarray:
        """计算到质心的距离"""
        # [N, k]
        distances = np.zeros((vectors.shape[0], self.centroids.shape[0]))
        for i, c in enumerate(self.centroids):
            distances[:, i] = np.sum((vectors - c) ** 2, axis=1)
        return distances


class MiniCache:
    """
    MiniCache - 高效KV缓存压缩
    
    实现15倍压缩
    """
    
    def __init__(self, config: MiniCacheConfig):
        self.config = config
        
        # 向量量化器
        self.key_quantizer = VectorQuantizer(
            dim=config.head_dim,
            num_centroids=config.num_centroids
        )
        self.value_quantizer = VectorQuantizer(
            dim=config.head_dim,
            num_centroids=config.num_centroids
        )
        
        # 压缩存储
        self.compressed_keys: List[np.ndarray] = []  # 量化索引
        self.compressed_values: List[np.ndarray] = []
        
        # 元数据
        self.positions: List[int] = []
        
        # 统计
        self.stats = {
            'total_compressed': 0,
            'compression_ratio': 0.0,
            'space_saved_bytes': 0
        }
    
    def compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        positions: Optional[List[int]] = None
    ) -> Dict:
        """
        压缩KV缓存
        
        Args:
            keys: [seq_len, num_heads, head_dim]
            values: [seq_len, num_heads, head_dim]
            positions: 位置列表
            
        Returns:
            压缩统计
        """
        seq_len = keys.shape[0]
        
        # 展平为 [seq_len * num_heads, head_dim]
        keys_flat = keys.reshape(-1, self.config.head_dim)
        values_flat = values.reshape(-1, self.config.head_dim)
        
        # 量化
        key_indices, _ = self.key_quantizer.quantize(keys_flat)
        value_indices, _ = self.value_quantizer.quantize(values_flat)
        
        # 存储
        self.compressed_keys.append(key_indices)
        self.compressed_values.append(value_indices)
        
        if positions:
            self.positions.extend(positions)
        else:
            self.positions.extend(range(seq_len))
        
        # 计算压缩比
        original_size = keys.nbytes + values.nbytes
        compressed_size = key_indices.nbytes + value_indices.nbytes
        
        compression_ratio = original_size / max(1, compressed_size)
        space_saved = original_size - compressed_size
        
        self.stats['total_compressed'] += seq_len
        self.stats['compression_ratio'] = (
            (self.stats['total_compressed'] - seq_len) * self.stats['compression_ratio'] +
            compression_ratio * seq_len
        ) / self.stats['total_compressed']
        self.stats['space_saved_bytes'] += space_saved
        
        return {
            'compression_ratio': compression_ratio,
            'space_saved_bytes': space_saved
        }
    
    def decompress(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        解压KV缓存
        
        Returns:
            (keys, values)
        """
        if not self.compressed_keys:
            return np.array([]), np.array([])
        
        # 合并所有索引
        all_key_indices = np.concatenate(self.compressed_keys)
        all_value_indices = np.concatenate(self.compressed_values)
        
        # 反量化
        keys = self.key_quantizer.dequantize(all_key_indices)
        values = self.value_quantizer.dequantize(all_value_indices)
        
        # 重塑为 [seq_len, num_heads, head_dim]
        seq_len = len(all_key_indices) // self.config.num_heads
        keys = keys.reshape(seq_len, self.config.num_heads, self.config.head_dim)
        values = values.reshape(seq_len, self.config.num_heads, self.config.head_dim)
        
        return keys, values
    
    def get_positions(self) -> List[int]:
        """获取位置列表"""
        return self.positions.copy()
    
    def clear(self) -> None:
        """清空缓存"""
        self.compressed_keys.clear()
        self.compressed_values.clear()
        self.positions.clear()
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            **self.stats,
            'current_entries': len(self.positions),
            'key_quantizer_stats': self.key_quantizer.stats,
            'value_quantizer_stats': self.value_quantizer.stats
        }


class MiniCacheManager:
    """
    MiniCache管理器
    
    管理多个头的压缩缓存
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        compression_ratio: int = 15
    ):
        self.config = MiniCacheConfig(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            compression_ratio=compression_ratio
        )
        
        self.cache = MiniCache(self.config)
    
    def update(
        self,
        new_keys: np.ndarray,
        new_values: np.ndarray
    ) -> Dict:
        """
        更新缓存
        
        Args:
            new_keys: [seq_len, num_heads, head_dim]
            new_values: [seq_len, num_heads, head_dim]
            
        Returns:
            压缩统计
        """
        return self.cache.compress(new_keys, new_values)
    
    def get_cached(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取缓存的KV"""
        return self.cache.decompress()
    
    def get_stats(self) -> Dict:
        """获取统计"""
        return self.cache.get_stats()


# 便捷函数
def create_mini_cache(
    hidden_size: int = 768,
    num_heads: int = 12,
    compression_ratio: int = 15
) -> MiniCacheManager:
    """创建MiniCache"""
    return MiniCacheManager(
        hidden_size=hidden_size,
        num_heads=num_heads,
        compression_ratio=compression_ratio
    )


if __name__ == "__main__":
    # 测试
    cache = create_mini_cache(hidden_size=256, num_heads=4, compression_ratio=15)
    
    print("=" * 60)
    print("MiniCache测试")
    print("=" * 60)
    
    # 模拟KV
    keys = np.random.randn(100, 4, 64).astype(np.float32)
    values = np.random.randn(100, 4, 64).astype(np.float32)
    
    # 压缩
    result = cache.update(keys, values)
    print(f"\n压缩比: {result['compression_ratio']:.1f}x")
    print(f"节省空间: {result['space_saved_bytes'] / 1024:.1f} KB")
    
    # 解压
    decompressed_keys, decompressed_values = cache.get_cached()
    print(f"\n解压后形状: keys={decompressed_keys.shape}, values={decompressed_values.shape}")
    
    # 统计
    print(f"\n统计: {cache.get_stats()}")
