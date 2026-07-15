"""
AGI Unified Framework - VSA (Vector Symbolic Architecture) 超维计算模块
==========================================================================

本模块实现了超维计算（Hyperdimensional Computing）的核心算法。

VSA简介：
- 使用高维随机向量（通常1000-10000维）表示概念
- 支持绑定（binding）、捆绑（bundling）、置换（permutation）操作
- 具有噪声容忍和计算效率高的特点
- 广泛应用于记忆编码、类比推理、快速学习等场景

主要功能：
1. 超向量基本操作
2. 绑定操作（绑定、解绑）
3. 捆绑操作（捆绑、清理）
4. 置换操作（循环移位）
5. 项记忆系统
6. 序列编码

使用示例：
    from core.vsa import HyperVector, Bundle, bind, permute
    
    # 创建超向量
    cat = HyperVector.random('cat')
    dog = HyperVector.random('dog')
    
    # 绑定：创建复合概念
    animal_cat = bind(cat, animal_concept)
    
    # 捆绑：合并多个概念
    scene = bundle(cat, on_mat, near_bowl)
    
    # 清理：从捆绑中提取
    recovered_cat = cleanup(scene, [cat, dog, bird])
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


# =============================================================================
# 常量定义
# =============================================================================

class VDimensionality(Enum):
    """向量维度枚举"""
    D100 = 100
    D500 = 500
    D1000 = 1000
    D2000 = 2000
    D5000 = 5000
    D10000 = 10000
    
    @classmethod
    def default(cls):
        return cls.D1000


class VSBinding(Enum):
    """绑定类型枚举"""
    MULTIPLICATIVE = "multiplicative"    # 逐元素乘法
    CIRCULAR_CONVOLUTION = "circular_convolution"  # 循环卷积
    XNOR = "xnor"  # 二值XOR


@dataclass
class VSConfig:
    """VSA配置"""
    dimensionality: int = 1000
    binding_type: VSBinding = VSBinding.MULTIPLICATIVE
    bundling_type: str = "sum"  # sum, product, majority
    vector_type: str = "binary"  # binary, bipolar, real
    seed: Optional[int] = None
    
    # 清理参数
    cleanup_threshold: float = 0.3
    
    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)


# =============================================================================
# 基础超向量类
# =============================================================================

class HyperVector:
    """
    超向量类
    
    表示一个高维向量，用于表示概念或信息。
    支持多种类型：二进制、双极、实数。
    
    Attributes:
        data: 向量数据
        vtype: 向量类型
        name: 名称（可选）
    """
    
    def __init__(self, data: np.ndarray, vtype: Optional[str] = None, 
                 name: Optional[str] = None):
        """
        创建超向量
        
        Args:
            data: 向量数据
            vtype: 向量类型 ('binary', 'bipolar', 'real')
            name: 名称
        """
        self.data = data
        self.name = name
        
        # 自动推断类型
        if vtype is None:
            unique_vals = np.unique(data)
            if set(unique_vals).issubset({0, 1}):
                self.vtype = 'binary'
            elif set(unique_vals).issubset({-1, 1}):
                self.vtype = 'bipolar'
            else:
                self.vtype = 'real'
        else:
            self.vtype = vtype
    
    @property
    def dim(self) -> int:
        """返回向量维度"""
        return len(self.data)
    
    @property
    def shape(self) -> Tuple[int]:
        """返回向量形状"""
        return self.data.shape
    
    # =====================================================================
    # 工厂方法
    # =====================================================================
    
    @classmethod
    def random(cls, name: Optional[str] = None, dim: int = 1000,
              vtype: str = 'binary', seed: Optional[int] = None) -> 'HyperVector':
        """
        创建随机超向量
        
        Args:
            name: 名称
            dim: 维度
            vtype: 类型
            seed: 随机种子
            
        Returns:
            随机超向量
        """
        if seed is not None:
            np.random.seed(seed)
        
        if vtype == 'binary':
            data = np.random.randint(0, 2, dim)
        elif vtype == 'bipolar':
            data = np.random.choice([-1, 1], dim)
        else:  # real
            data = np.random.randn(dim)
        
        return cls(data, vtype, name)
    
    @classmethod
    def zero(cls, name: Optional[str] = None, dim: int = 1000,
            vtype: str = 'binary') -> 'HyperVector':
        """创建零向量"""
        return cls(np.zeros(dim), vtype, name)
    
    @classmethod
    def constant(cls, value: Union[int, float], name: Optional[str] = None,
                dim: int = 1000, vtype: str = 'binary') -> 'HyperVector':
        """创建常量向量"""
        if vtype == 'binary':
            value = 1 if value else 0
        return cls(np.full(dim, value), vtype, name)
    
    @classmethod
    def from_hash(cls, content: str, name: Optional[str] = None,
                  dim: int = 1000) -> 'HyperVector':
        """
        从字符串创建超向量
        
        使用哈希确保相同内容产生相同的向量。
        
        Args:
            content: 内容字符串
            name: 名称
            dim: 维度
            
        Returns:
            基于哈希的超向量
        """
        # 生成确定性的随机向量
        hash_bytes = hashlib.sha256(content.encode()).digest()
        
        # 使用多个哈希循环填充向量
        data = np.zeros(dim, dtype=np.float64)
        for i in range(dim):
            hash_input = content + str(i)
            h = hashlib.sha256(hash_input.encode()).digest()
            # 使用前8字节作为float64
            value = struct.unpack('d', h[:8])[0]
            data[i] = value
        
        # 归一化到[0, 1]或[-1, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-10)
        if random.random() > 0.5:
            data = 2 * data - 1  # 转双极
        
        # 二值化
        binary_data = (data > 0).astype(int)
        
        return cls(binary_data, 'binary', name or content[:20])
    
    # =====================================================================
    # 操作方法
    # =====================================================================
    
    def copy(self) -> 'HyperVector':
        """复制向量"""
        return HyperVector(self.data.copy(), self.vtype, self.name)
    
    def binarize(self, threshold: float = 0.5) -> 'HyperVector':
        """二值化"""
        if self.vtype == 'binary':
            return self.copy()
        
        binary_data = (self.data > threshold).astype(int)
        return HyperVector(binary_data, 'binary', self.name)
    
    def normalize(self) -> 'HyperVector':
        """归一化"""
        if self.vtype == 'real':
            norm = np.linalg.norm(self.data)
            if norm > 0:
                data = self.data / norm
            else:
                data = self.data
            return HyperVector(data, 'real', self.name)
        return self.copy()
    
    # =====================================================================
    # 距离和相似度
    # =====================================================================
    
    def similarity(self, other: 'HyperVector') -> float:
        """
        计算与另一个向量的相似度
        
        使用余弦相似度或汉明相似度。
        
        Args:
            other: 另一个超向量
            
        Returns:
            相似度值 [0, 1] 或 [-1, 1]
        """
        if self.dim != other.dim:
            raise ValueError(f"维度不匹配: {self.dim} vs {other.dim}")
        
        # 二值向量使用汉明相似度
        if self.vtype == 'binary' and other.vtype == 'binary':
            matches = np.sum(self.data == other.data)
            return matches / self.dim
        
        # 双极或实数向量使用余弦相似度
        dot = np.dot(self.data, other.data)
        norm1 = np.linalg.norm(self.data)
        norm2 = np.linalg.norm(other.data)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_sim = dot / (norm1 * norm2)
        # 归一化到[0, 1]
        return (cos_sim + 1) / 2
    
    def distance(self, other: 'HyperVector') -> float:
        """计算距离（1 - 相似度）"""
        return 1 - self.similarity(other)
    
    def hamming_similarity(self, other: 'HyperVector') -> float:
        """汉明相似度"""
        if self.vtype != 'binary' or other.vtype != 'binary':
            raise ValueError("汉明相似度仅适用于二进制向量")
        return np.sum(self.data == other.data) / self.dim
    
    def dot_product(self, other: 'HyperVector') -> float:
        """点积"""
        if self.dim != other.dim:
            raise ValueError(f"维度不匹配: {self.dim} vs {other.dim}")
        return float(np.dot(self.data, other.data))
    
    # =====================================================================
    # 数学运算
    # =====================================================================
    
    def __add__(self, other: Union['HyperVector', float]) -> 'HyperVector':
        """捆绑操作"""
        if isinstance(other, HyperVector):
            if self.dim != other.dim:
                raise ValueError("维度不匹配")
            if self.vtype == 'binary' and other.vtype == 'binary':
                # 二值向量使用多数投票
                data = ((self.data + other.data) >= 1).astype(int)
            else:
                data = self.data + other.data
            return HyperVector(data, 'real', f"bundled({self.name}, {other.name})")
        else:
            return HyperVector(self.data + other, self.vtype, self.name)
    
    def __mul__(self, other: Union['HyperVector', float]) -> 'HyperVector':
        """绑定操作"""
        if isinstance(other, HyperVector):
            if self.dim != other.dim:
                raise ValueError("维度不匹配")
            if self.vtype == 'binary' and other.vtype == 'binary':
                # 二值XOR绑定
                data = np.logical_xor(self.data, other.data).astype(int)
            else:
                data = self.data * other.data
            return HyperVector(data, self.vtype, f"bound({self.name}, {other.name})")
        else:
            return HyperVector(self.data * other, self.vtype, self.name)
    
    def __matmul__(self, other: 'HyperVector') -> float:
        """点积"""
        return self.dot_product(other)
    
    def permute(self, shifts: Optional[int] = None) -> 'HyperVector':
        """
        置换操作（循环移位）
        
        Args:
            shifts: 移位量，None表示随机移位
            
        Returns:
            置换后的向量
        """
        if shifts is None:
            shifts = np.random.randint(1, self.dim)
        
        permuted = np.roll(self.data, shifts)
        return HyperVector(permuted, self.vtype, f"perm({self.name})")
    
    def unpermute(self, shifts: int) -> 'HyperVector':
        """反置换"""
        permuted = np.roll(self.data, -shifts)
        return HyperVector(permuted, self.vtype, f"unperm({self.name})")
    
    def threshold(self) -> 'HyperVector':
        """阈值化（用于从实数向量生成二值向量）"""
        threshold = np.mean(self.data)
        binary_data = (self.data > threshold).astype(int)
        return HyperVector(binary_data, 'binary', self.name)
    
    # =====================================================================
    # 统计方法
    # =====================================================================
    
    def mean(self) -> float:
        """平均值"""
        return float(np.mean(self.data))
    
    def std(self) -> float:
        """标准差"""
        return float(np.std(self.data))
    
    def sparsity(self) -> float:
        """稀疏度（零元素比例）"""
        if self.vtype == 'binary':
            return 1 - np.mean(self.data)
        return 0.0
    
    # =====================================================================
    # 表示方法
    # =====================================================================
    
    def __repr__(self) -> str:
        vtype_short = {'binary': 'B', 'bipolar': 'P', 'real': 'R'}[self.vtype]
        name_str = f"'{self.name}'" if self.name else "unnamed"
        preview = str(self.data[:5])
        return f"HyperVector({self.dim}D, {vtype_short}, {name_str}, {preview}...)"
    
    def __str__(self) -> str:
        return f"{self.name or 'HV'}[{self.vtype}]({self.dim})"
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return self.data.copy()
    
    @classmethod
    def from_array(cls, arr: np.ndarray, name: Optional[str] = None) -> 'HyperVector':
        """从numpy数组创建"""
        return cls(arr.copy(), name=name)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data.tolist(),
            'type': self.vtype,
            'name': self.name,
            'dim': self.dim
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HyperVector':
        """从字典创建"""
        return cls(
            np.array(d['data']),
            vtype=d.get('type'),
            name=d.get('name')
        )


# =============================================================================
# VSA 操作函数
# =============================================================================

def bind(hv1: HyperVector, hv2: HyperVector, binding_type: VSBinding = VSBinding.MULTIPLICATIVE) -> HyperVector:
    """
    绑定操作
    
    绑定用于将两个概念组合成一个复合概念。
    绑定的结果可以解绑恢复原始概念。
    
    Args:
        hv1: 第一个超向量
        hv2: 第二个超向量
        binding_type: 绑定类型
        
    Returns:
        绑定后的超向量
    """
    if binding_type == VSBinding.MULTIPLICATIVE:
        return hv1 * hv2
    elif binding_type == VSBinding.CIRCULAR_CONVOLUTION:
        # 循环卷积
        conv = np.fft.ifft(np.fft.fft(hv1.data) * np.fft.fft(hv2.data)).real
        return HyperVector(conv, hv1.vtype, f"conv({hv1.name}, {hv2.name})")
    elif binding_type == VSBinding.XNOR:
        # XNOR (用于二值向量)
        if hv1.vtype != 'binary' or hv2.vtype != 'binary':
            raise ValueError("XNOR绑定仅适用于二值向量")
        xnor = np.logical_not(np.logical_xor(hv1.data, hv2.data)).astype(int)
        return HyperVector(xnor, 'binary', f"xnor({hv1.name}, {hv2.name})")
    else:
        raise ValueError(f"未知的绑定类型: {binding_type}")


def unbind(compound: HyperVector, hv: HyperVector, 
          binding_type: VSBinding = VSBinding.MULTIPLICATIVE) -> HyperVector:
    """
    解绑操作
    
    从复合向量中恢复其中一个原始向量。
    
    Args:
        compound: 复合超向量
        hv: 已知的原始向量之一
        binding_type: 绑定类型
        
    Returns:
        恢复的另一个原始向量
    """
    if binding_type == VSBinding.MULTIPLICATIVE:
        return compound * hv  # 对于乘法绑定，解绑就是再次乘法
    elif binding_type == VSBinding.CIRCULAR_CONVOLUTION:
        # 频域解卷积
        spec_compound = np.fft.fft(compound.data)
        spec_hv = np.fft.fft(hv.data)
        # 避免除以零
        spec_hv = np.where(np.abs(spec_hv) < 1e-10, 1e-10, spec_hv)
        spec_result = spec_compound / spec_hv
        result = np.fft.ifft(spec_result).real
        return HyperVector(result, compound.vtype)
    else:
        raise ValueError(f"未知的绑定类型: {binding_type}")


def bundle(hvs: List[HyperVector], bundling_type: str = 'sum') -> HyperVector:
    """
    捆绑操作
    
    将多个超向量合并成一个。
    捆绑后的向量可以通过清理操作恢复原始向量。
    
    Args:
        hvs: 超向量列表
        bundling_type: 捆绑类型 ('sum', 'product', 'majority')
        
    Returns:
        捆绑后的超向量
    """
    if not hvs:
        raise ValueError("捆绑列表不能为空")
    
    dim = hvs[0].dim
    
    # 检查维度一致性
    for hv in hvs[1:]:
        if hv.dim != dim:
            raise ValueError(f"维度不匹配: {dim} vs {hv.dim}")
    
    if bundling_type == 'sum':
        # 求和捆绑
        data = sum(hv.data for hv in hvs)
        return HyperVector(data, 'real', f"bundle({len(hvs)})")
    
    elif bundling_type == 'product':
        # 乘积捆绑
        data = hvs[0].data.copy()
        for hv in hvs[1:]:
            data = data * hv.data
        return HyperVector(data, hvs[0].vtype, f"product({len(hvs)})")
    
    elif bundling_type == 'majority':
        # 多数投票（仅适用于二值向量）
        if not all(hv.vtype == 'binary' for hv in hvs):
            raise ValueError("多数投票仅适用于二值向量")
        votes = sum(hv.data for hv in hvs)
        threshold = len(hvs) / 2
        data = (votes > threshold).astype(int)
        return HyperVector(data, 'binary', f"majority({len(hvs)})")
    
    else:
        raise ValueError(f"未知的捆绑类型: {bundling_type}")


def cleanup(bundled: HyperVector, candidates: List[HyperVector],
           threshold: float = 0.3) -> Tuple[Optional[HyperVector], float]:
    """
    清理操作
    
    从捆绑向量中恢复最匹配的原始向量。
    
    Args:
        bundled: 捆绑向量
        candidates: 候选向量列表
        threshold: 相似度阈值
        
    Returns:
        (最匹配的向量, 相似度) 或 (None, 0) 如果没有匹配
    """
    if not candidates:
        return None, 0.0
    
    best_match = None
    best_similarity = threshold
    
    for candidate in candidates:
        sim = bundled.similarity(candidate)
        if sim > best_similarity:
            best_similarity = sim
            best_match = candidate
    
    return best_match, best_similarity


def permute(hv: HyperVector, k: Optional[int] = None) -> HyperVector:
    """
    置换操作
    
    对向量进行循环移位。
    多次置换需要记住移位量以便恢复。
    
    Args:
        hv: 超向量
        k: 移位量，None表示随机
        
    Returns:
        置换后的向量
    """
    return hv.permute(k)


def sequence_encode(items: List[HyperVector]) -> HyperVector:
    """
    序列编码
    
    将有序的项目序列编码为单个超向量。
    使用位置置换来保留顺序信息。
    
    Args:
        items: 项目超向量列表
        
    Returns:
        编码后的序列向量
    """
    if not items:
        raise ValueError("序列不能为空")
    
    # 为每个位置创建位置向量
    position_vectors = []
    for i, item in enumerate(items):
        # 位置向量 = 随机置换基向量
        pos_vec = HyperVector.random(f"pos_{i}", dim=item.dim, vtype=item.vtype)
        position_vectors.append(pos_vec)
        
        # 绑定项目和位置
        items[i] = bind(item, pos_vec)
    
    # 捆绑所有绑定后的项目
    return bundle(items)


def ngram_encode(ngrams: List[Tuple[HyperVector, ...]]) -> HyperVector:
    """
    N-gram编码
    
    将n-gram序列编码为超向量。
    
    Args:
        ngrams: n-gram元组列表
        
    Returns:
        编码后的超向量
    """
    if not ngrams:
        raise ValueError("N-gram列表不能为空")
    
    encoded_ngrams = []
    for ngram in ngrams:
        # 捆绑n-gram中的所有元素
        encoded = bundle(list(ngram))
        encoded_ngrams.append(encoded)
    
    # 捆绑所有n-gram
    return bundle(encoded_ngrams)


# =============================================================================
# 项记忆系统
# =============================================================================

class ItemMemory:
    """
    项记忆
    
    存储符号到超向量的映射。
    支持添加、检索和批量操作。
    
    Attributes:
        items: 符号到超向量的映射
        config: 配置
    """
    
    def __init__(self, config: Optional[VSConfig] = None):
        self.config = config or VSConfig()
        self.items: Dict[str, HyperVector] = {}
        self.reverse_index: Dict[str, HyperVector] = {}  # 双向索引
    
    def add(self, symbol: str, hv: HyperVector) -> None:
        """
        添加项
        
        Args:
            symbol: 符号
            hv: 超向量
        """
        if hv.dim != self.config.dimensionality:
            hv = HyperVector(hv.data, hv.vtype, hv.name)
        
        self.items[symbol] = hv
        self.reverse_index[symbol] = hv
    
    def get(self, symbol: str) -> Optional[HyperVector]:
        """获取项"""
        return self.items.get(symbol)
    
    def recall(self, hv: HyperVector, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        检索最相似的项
        
        Args:
            hv: 查询向量
            top_k: 返回前k个结果
            
        Returns:
            [(符号, 相似度), ...]
        """
        results = []
        for symbol, item_hv in self.items.items():
            sim = hv.similarity(item_hv)
            results.append((symbol, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def contains(self, symbol: str) -> bool:
        """检查是否包含符号"""
        return symbol in self.items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self.items
    
    def __getitem__(self, symbol: str) -> HyperVector:
        return self.items[symbol]
    
    def keys(self):
        return self.items.keys()
    
    def values(self):
        return self.items.values()
    
    def items_iter(self):
        return self.items.items()


class ContinuousItemMemory(ItemMemory):
    """
    连续项记忆
    
    支持增量学习和在线更新的项记忆。
    使用指数衰减来平衡新旧知识。
    """
    
    def __init__(self, config: Optional[VSConfig] = None,
                decay_rate: float = 0.99):
        super().__init__(config)
        self.decay_rate = decay_rate
        self.access_counts: Dict[str, int] = {}
        self.last_access: Dict[str, float] = {}
    
    def add(self, symbol: str, hv: HyperVector) -> None:
        """添加或更新项（带访问计数）"""
        super().add(symbol, hv)
        self.access_counts[symbol] = 1
        self.last_access[symbol] = time.time()
    
    def get_importance(self, symbol: str) -> float:
        """
        计算符号的重要性分数
        
        基于访问频率和最近访问时间。
        """
        if symbol not in self.access_counts:
            return 0.0
        
        count = self.access_counts[symbol]
        last_time = self.last_access.get(symbol, 0)
        
        # 简单的重要性公式
        time_factor = math.exp(-0.01 * (time.time() - last_time))
        return count * time_factor
    
    def record_access(self, symbol: str) -> None:
        """记录访问"""
        if symbol in self.access_counts:
            self.access_counts[symbol] += 1
        self.last_access[symbol] = time.time()


# 导入时间函数
import time


# =============================================================================
# 图结构编码
# =============================================================================

class GraphEncoder:
    """
    图结构编码器
    
    将图结构（节点和边）编码为超向量表示。
    """
    
    def __init__(self, config: Optional[VSConfig] = None):
        self.config = config or VSConfig()
        self.node_memory = ItemMemory(config)
        self.edge_memory = ItemMemory(config)
        
        # 图的基础向量
        self.graph_baseline = HyperVector.random('graph', config.dimensionality, 'bipolar')
    
    def encode_node(self, node_id: str, features: Optional[HyperVector] = None) -> HyperVector:
        """
        编码节点
        
        Args:
            node_id: 节点ID
            features: 节点特征向量
            
        Returns:
            节点超向量
        """
        # 基础节点向量
        node_baseline = HyperVector.from_hash(node_id, f"node_{node_id}", 
                                              self.config.dimensionality)
        
        if features is not None:
            # 绑定特征
            node_hv = bind(node_baseline, features)
        else:
            node_hv = node_baseline
        
        self.node_memory.add(node_id, node_hv)
        return node_hv
    
    def encode_edge(self, edge_type: str, node1_id: str, node2_id: str) -> HyperVector:
        """
        编码边
        
        Args:
            edge_type: 边类型
            node1_id: 起点节点ID
            node2_id: 终点节点ID
            
        Returns:
            边超向量
        """
        # 获取节点向量
        node1_hv = self.node_memory.get(node1_id)
        node2_hv = self.node_memory.get(node2_id)
        
        if node1_hv is None or node2_hv is None:
            raise ValueError("节点未编码")
        
        # 边向量 = 节点绑定 + 边类型
        edge_baseline = HyperVector.from_hash(edge_type, f"edge_{edge_type}",
                                             self.config.dimensionality)
        node_bind = bind(node1_hv, node2_hv)
        edge_hv = bind(node_bind, edge_baseline)
        
        edge_key = f"{node1_id}_{edge_type}_{node2_id}"
        self.edge_memory.add(edge_key, edge_hv)
        return edge_hv
    
    def encode_graph(self, edges: List[Tuple[str, str, str]]) -> HyperVector:
        """
        编码整个图
        
        Args:
            edges: [(node1_id, edge_type, node2_id), ...]
            
        Returns:
            图超向量
        """
        edge_hvs = []
        for node1_id, edge_type, node2_id in edges:
            edge_hv = self.encode_edge(edge_type, node1_id, node2_id)
            edge_hvs.append(edge_hv)
        
        # 捆绑所有边
        graph_hv = bundle(edge_hvs)
        
        # 与图基础向量绑定
        return bind(graph_hv, self.graph_baseline)
    
    def query_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        查询邻居节点
        
        Args:
            node_id: 节点ID
            edge_type: 边类型过滤
            
        Returns:
            邻居节点ID列表
        """
        node_hv = self.node_memory.get(node_id)
        if node_hv is None:
            return []
        
        neighbors = []
        for edge_key, edge_hv in self.edge_memory.items_iter():
            # 解码边的节点
            parts = edge_key.split('_')
            if len(parts) >= 3:
                n1, et, n2 = parts[0], parts[1], '_'.join(parts[2:])
                
                if edge_type and et != edge_type:
                    continue
                
                if n1 == node_id:
                    neighbors.append(n2)
                elif n2 == node_id:
                    neighbors.append(n1)
        
        return neighbors


# =============================================================================
# 配置工厂
# =============================================================================

def create_vs_config(dim: int = 1000, vtype: str = 'binary',
                    seed: Optional[int] = None) -> VSConfig:
    """创建VSA配置"""
    return VSConfig(
        dimensionality=dim,
        vector_type=vtype,
        seed=seed
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 类
    'HyperVector',
    'ItemMemory',
    'ContinuousItemMemory',
    'GraphEncoder',
    
    # 配置
    'VSConfig',
    'VDimensionality',
    'VSBinding',
    'create_vs_config',
    
    # 操作函数
    'bind',
    'unbind',
    'bundle',
    'cleanup',
    'permute',
    'sequence_encode',
    'ngram_encode',
]
