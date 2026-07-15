"""
统一算法配置模块

提供所有统一算法的集中配置管理，支持AGI和视频生成系统的通用配置需求。
使用泛型设计，不依赖具体数据类型。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable, Union
from enum import Enum, auto


# ============================================================================
# 泛型类型定义
# ============================================================================

T = TypeVar('T')  # 通用类型参数
S = TypeVar('S')  # 序列元素类型（视频帧/文本token）
K = TypeVar('K')  # 键类型
V = TypeVar('V')  # 值类型


# ============================================================================
# 枚举类型定义
# ============================================================================

class MemoryRetrievalMode(Enum):
    """记忆检索模式"""
    SIMILARITY = auto()      # 基于相似度的检索
    LEARNABLE = auto()       # 可学习的检索
    HYBRID = auto()          # 混合检索


class CompressionStrategy(Enum):
    """压缩策略"""
    FIXED_RATIO = auto()     # 固定压缩率
    ADAPTIVE = auto()        # 自适应压缩
    LOSSLESS = auto()        # 无损压缩


class AttentionPattern(Enum):
    """注意力模式"""
    DENSE = auto()           # 密集注意力
    SPARSE_RANDOM = auto()   # 随机稀疏
    SPARSE_BLOCK = auto()    # 块稀疏
    SPARSE_STRIDED = auto()  # 步进稀疏


class ChunkingStrategy(Enum):
    """分块策略"""
    FIXED_SIZE = auto()      # 固定大小
    SEMANTIC = auto()        # 语义分块
    TEMPORAL = auto()        # 时间分块
    ADAPTIVE = auto()        # 自适应分块


class BoundaryType(Enum):
    """边界类型"""
    SHOT = auto()            # 镜头边界（视频）
    SCENE = auto()           # 场景边界（视频）
    TOPIC = auto()           # 话题边界（文本）
    PARAGRAPH = auto()       # 段落边界（文本）


class ExpertType(Enum):
    """专家类型"""
    PHYSICAL = auto()        # 物理专家
    REASONING = auto()       # 推理专家
    MEMORY = auto()          # 记忆专家
    PERCEPTION = auto()      # 感知专家
    GENERATION = auto()      # 生成专家


class ConstraintPriority(Enum):
    """约束优先级"""
    HARD = auto()            # 硬约束（必须满足）
    SOFT = auto()            # 软约束（尽量满足）
    GUIDANCE = auto()        # 指导性约束


# ============================================================================
# 统一算法配置类
# ============================================================================

@dataclass
class UnifiedAlgorithmConfig:
    """
    统一算法配置类
    
    集中管理所有统一算法的配置参数，支持AGI和视频生成系统的通用需求。
    所有参数都有合理的默认值，可根据具体任务进行调整。
    
    Attributes:
        # 记忆系统配置
        memory_capacity: 记忆库最大容量
        memory_compression_rate: 默认压缩率 (0.0-1.0)
        memory_retrieval_mode: 记忆检索模式
        
        # 注意力配置
        attention_window_size: 滑动窗口大小
        attention_cache_zones: 缓存区域数量
        attention_pattern: 注意力稀疏模式
        
        # 分块配置
        chunk_size: 默认分块大小
        chunk_overlap: 分块重叠大小
        chunking_strategy: 分块策略
        
        # MoE配置
        num_experts: 专家数量
        top_k_experts: 激活的专家数量
        expert_capacity: 每个专家的容量
        
        # 约束配置
        constraint_tolerance: 约束容差
        
        # 通用配置
        device: 运行设备标识
        dtype: 数据类型
        random_seed: 随机种子
    """
    
    # 记忆系统配置
    memory_capacity: int = 10000
    memory_compression_rate: float = 0.5
    memory_retrieval_mode: MemoryRetrievalMode = MemoryRetrievalMode.HYBRID
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    
    # 注意力配置
    attention_window_size: int = 2048
    attention_cache_zones: int = 3
    attention_pattern: AttentionPattern = AttentionPattern.SPARSE_BLOCK
    
    # 分块配置
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.ADAPTIVE
    
    # MoE配置
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity: float = 1.0
    
    # 约束配置
    constraint_tolerance: float = 0.01
    
    # 通用配置
    device: str = "cpu"
    dtype: str = "float32"
    random_seed: Optional[int] = None
    
    # 扩展配置（用于存储特定任务的额外参数）
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """配置验证和初始化后处理"""
        # 验证压缩率在有效范围
        if not 0.0 <= self.memory_compression_rate <= 1.0:
            raise ValueError(f"压缩率必须在 [0.0, 1.0] 范围内，当前值: {self.memory_compression_rate}")
        
        # 验证专家数量关系
        if self.top_k_experts > self.num_experts:
            raise ValueError(f"top_k_experts ({self.top_k_experts}) 不能大于 num_experts ({self.num_experts})")
        
        # 验证窗口大小
        if self.attention_window_size <= 0:
            raise ValueError(f"注意力窗口大小必须为正数，当前值: {self.attention_window_size}")
    
    def update(self, **kwargs) -> UnifiedAlgorithmConfig:
        """
        更新配置参数
        
        Args:
            **kwargs: 要更新的配置项
            
        Returns:
            更新后的新配置对象
        """
        current = {
            'memory_capacity': self.memory_capacity,
            'memory_compression_rate': self.memory_compression_rate,
            'memory_retrieval_mode': self.memory_retrieval_mode,
            'compression_strategy': self.compression_strategy,
            'attention_window_size': self.attention_window_size,
            'attention_cache_zones': self.attention_cache_zones,
            'attention_pattern': self.attention_pattern,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunking_strategy': self.chunking_strategy,
            'num_experts': self.num_experts,
            'top_k_experts': self.top_k_experts,
            'expert_capacity': self.expert_capacity,
            'constraint_tolerance': self.constraint_tolerance,
            'device': self.device,
            'dtype': self.dtype,
            'random_seed': self.random_seed,
            'extra_config': self.extra_config.copy(),
        }
        current.update(kwargs)
        return UnifiedAlgorithmConfig(**current)
    
    def get_extra(self, key: str, default: Any = None) -> Any:
        """
        获取扩展配置项
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        return self.extra_config.get(key, default)
    
    def set_extra(self, key: str, value: Any) -> None:
        """
        设置扩展配置项
        
        Args:
            key: 配置键
            value: 配置值
        """
        self.extra_config[key] = value
    
    @classmethod
    def default_config(cls) -> UnifiedAlgorithmConfig:
        """获取默认配置"""
        return cls()
    
    @classmethod
    def video_optimized_config(cls) -> UnifiedAlgorithmConfig:
        """获取视频优化的配置"""
        return cls(
            memory_capacity=50000,
            memory_compression_rate=0.3,
            attention_window_size=4096,
            chunk_size=1024,
            chunking_strategy=ChunkingStrategy.TEMPORAL,
            num_experts=16,
        )
    
    @classmethod
    def agi_optimized_config(cls) -> UnifiedAlgorithmConfig:
        """获取AGI优化的配置"""
        return cls(
            memory_capacity=100000,
            memory_compression_rate=0.7,
            attention_window_size=8192,
            chunk_size=256,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            num_experts=32,
            top_k_experts=4,
        )


# ============================================================================
# 配置工具函数
# ============================================================================

def merge_configs(base: UnifiedAlgorithmConfig, override: UnifiedAlgorithmConfig) -> UnifiedAlgorithmConfig:
    """
    合并两个配置，override 中的非默认值会覆盖 base 中的值
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = UnifiedAlgorithmConfig.default_config()
    
    # 复制 base 的配置
    for field_name in base.__dataclass_fields__:
        if field_name != 'extra_config':
            setattr(result, field_name, getattr(base, field_name))
    
    # 合并 extra_config
    result.extra_config = {**base.extra_config, **override.extra_config}
    
    # 应用 override 的非默认值
    default = UnifiedAlgorithmConfig.default_config()
    for field_name in override.__dataclass_fields__:
        if field_name != 'extra_config':
            override_val = getattr(override, field_name)
            default_val = getattr(default, field_name)
            if override_val != default_val:
                setattr(result, field_name, override_val)
    
    return result


def config_from_dict(config_dict: Dict[str, Any]) -> UnifiedAlgorithmConfig:
    """
    从字典创建配置对象
    
    Args:
        config_dict: 配置字典
        
    Returns:
        配置对象
    """
    # 处理枚举类型
    enum_fields = {
        'memory_retrieval_mode': MemoryRetrievalMode,
        'compression_strategy': CompressionStrategy,
        'attention_pattern': AttentionPattern,
        'chunking_strategy': ChunkingStrategy,
    }
    
    processed = {}
    for key, value in config_dict.items():
        if key in enum_fields and isinstance(value, str):
            processed[key] = enum_fields[key][value.upper()]
        else:
            processed[key] = value
    
    return UnifiedAlgorithmConfig(**processed)


def config_to_dict(config: UnifiedAlgorithmConfig) -> Dict[str, Any]:
    """
    将配置对象转换为字典
    
    Args:
        config: 配置对象
        
    Returns:
        配置字典
    """
    result = {}
    for field_name in config.__dataclass_fields__:
        value = getattr(config, field_name)
        if isinstance(value, Enum):
            result[field_name] = value.name
        else:
            result[field_name] = value
    return result
