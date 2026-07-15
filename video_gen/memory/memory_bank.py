"""
记忆模块实现
包含5种记忆模块：MemoryEntry, MemoryBank, AdaptiveMemoryCompressor, 
LightweightMemory, MemoryFusion, HierarchicalMemory
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math
import time


# ============================================================================
# 1. MemoryEntry - 记忆条目数据类
# ============================================================================

@dataclass
class MemoryEntry:
    """记忆条目，存储单个记忆单元"""
    key: List[float]
    value: List[float]
    importance: float = 1.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    
    def __post_init__(self):
        """初始化后验证"""
        if not isinstance(self.key, list):
            self.key = list(self.key) if hasattr(self.key, '__iter__') else [float(self.key)]
        if not isinstance(self.value, list):
            self.value = list(self.value) if hasattr(self.value, '__iter__') else [float(self.value)]
    
    def update_access(self) -> None:
        """更新访问计数和时间戳"""
        self.access_count += 1
        self.timestamp = time.time()
    
    def decay_importance(self, decay_rate: float = 0.99) -> None:
        """衰减重要性"""
        self.importance *= decay_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'value': self.value,
            'importance': self.importance,
            'timestamp': self.timestamp,
            'access_count': self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """从字典创建"""
        return cls(
            key=data['key'],
            value=data['value'],
            importance=data.get('importance', 1.0),
            timestamp=data.get('timestamp', time.time()),
            access_count=data.get('access_count', 0)
        )


# ============================================================================
# 2. MemoryBank - 外部记忆库
# ============================================================================

class MemoryBank:
    """外部记忆库，支持基于重要性的存储和检索"""
    
    def __init__(self, capacity: int = 10000):
        """
        初始化记忆库
        
        Args:
            capacity: 记忆容量上限
        """
        self._capacity = capacity
        self._memory: List[MemoryEntry] = []
        self._importance_weights: List[float] = []
        self._key_dim: Optional[int] = None
    
    def store(self, key: List[float], value: List[float], importance: float = 1.0) -> None:
        """
        存储记忆
        
        Args:
            key: 记忆键（用于检索）
            value: 记忆值
            importance: 重要性权重
        """
        # 检查容量
        if len(self._memory) >= self._capacity:
            self._evict_low_importance()
        
        # 创建记忆条目
        entry = MemoryEntry(key=key, value=value, importance=importance)
        
        # 设置键维度
        if self._key_dim is None:
            self._key_dim = len(key)
        
        # 存储记忆
        self._memory.append(entry)
        self._importance_weights.append(importance)
    
    def retrieve(self, query: List[float], top_k: int = 10) -> List[MemoryEntry]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            top_k: 返回前k个最相似的记忆
            
        Returns:
            最相似的记忆列表
        """
        if not self._memory:
            return []
        
        # 计算相似度
        similarities = []
        for i, entry in enumerate(self._memory):
            sim = self._compute_similarity(query, entry.key)
            # 结合重要性加权
            weighted_sim = sim * (1 + 0.1 * self._importance_weights[i])
            similarities.append((weighted_sim, i, entry))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, idx, entry in similarities[:top_k]:
            # 更新访问信息
            entry.update_access()
            self._update_importance(entry)
            results.append(entry)
        
        return results
    
    def _compute_similarity(self, query: List[float], key: List[float]) -> float:
        """
        计算余弦相似度
        
        Args:
            query: 查询向量
            key: 键向量
            
        Returns:
            余弦相似度
        """
        if len(query) != len(key):
            # 维度不匹配时返回0
            return 0.0
        
        # 计算点积
        dot_product = sum(q * k for q, k in zip(query, key))
        
        # 计算范数
        norm_query = math.sqrt(sum(q * q for q in query))
        norm_key = math.sqrt(sum(k * k for k in key))
        
        if norm_query == 0 or norm_key == 0:
            return 0.0
        
        return dot_product / (norm_query * norm_key)
    
    def _update_importance(self, entry: MemoryEntry) -> None:
        """
        更新重要性
        
        Args:
            entry: 记忆条目
        """
        # 基于访问次数增加重要性
        entry.importance = min(10.0, entry.importance + 0.1 * entry.access_count)
        
        # 更新权重列表
        try:
            idx = self._memory.index(entry)
            self._importance_weights[idx] = entry.importance
        except ValueError:
            pass
    
    def _evict_low_importance(self) -> None:
        """淘汰低重要性记忆"""
        if not self._memory:
            return
        
        # 找到最低重要性的记忆
        min_idx = 0
        min_importance = self._importance_weights[0]
        
        for i, imp in enumerate(self._importance_weights):
            if imp < min_importance:
                min_importance = imp
                min_idx = i
        
        # 淘汰该记忆
        self._memory.pop(min_idx)
        self._importance_weights.pop(min_idx)
    
    def clear(self) -> None:
        """清空记忆库"""
        self._memory.clear()
        self._importance_weights.clear()
    
    def __len__(self) -> int:
        return len(self._memory)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._memory:
            return {
                'size': 0,
                'capacity': self._capacity,
                'avg_importance': 0.0,
                'total_access': 0
            }
        
        return {
            'size': len(self._memory),
            'capacity': self._capacity,
            'avg_importance': sum(self._importance_weights) / len(self._importance_weights),
            'total_access': sum(e.access_count for e in self._memory)
        }


# ============================================================================
# 3. AdaptiveMemoryCompressor - 自适应记忆压缩器
# ============================================================================

class LearnableCompressor:
    """可学习的压缩器"""
    
    def __init__(self, input_dim: int, compression_ratio: int):
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.output_dim = input_dim // compression_ratio
        
        # 随机初始化投影矩阵
        self._projection = self._init_projection()
        self._bias = [0.0] * self.output_dim
    
    def _init_projection(self) -> List[List[float]]:
        """初始化投影矩阵"""
        import random
        scale = 1.0 / math.sqrt(self.input_dim)
        return [
            [random.gauss(0, scale) for _ in range(self.input_dim)]
            for _ in range(self.output_dim)
        ]
    
    def compress(self, x: List[float]) -> List[float]:
        """压缩向量"""
        result = []
        for i in range(self.output_dim):
            val = sum(self._projection[i][j] * x[j] for j in range(len(x)))
            result.append(val + self._bias[i])
        return result
    
    def decompress(self, compressed: List[float]) -> List[float]:
        """解压向量（近似重建）"""
        result = []
        for i in range(self.input_dim):
            val = 0.0
            for j in range(len(compressed)):
                if j < len(self._projection) and i < len(self._projection[j]):
                    val += self._projection[j][i] * compressed[j]
            result.append(val)
        return result


class AdaptiveMemoryCompressor:
    """自适应记忆压缩器，支持多尺度压缩"""
    
    def __init__(self, input_dim: int = 512, compression_ratios: List[int] = None):
        """
        初始化压缩器
        
        Args:
            input_dim: 输入维度
            compression_ratios: 压缩比例列表
        """
        self._input_dim = input_dim
        self._compression_ratios = compression_ratios or [2, 4, 8]
        self._compressors: List[LearnableCompressor] = []
        
        # 初始化多尺度压缩器
        for ratio in self._compression_ratios:
            if input_dim % ratio == 0:
                self._compressors.append(LearnableCompressor(input_dim, ratio))
        
        # 可学习查询参数
        self._query_weights = self._init_query_weights()
    
    def _init_query_weights(self) -> List[List[float]]:
        """初始化查询权重"""
        import random
        num_scales = len(self._compression_ratios)
        return [
            [random.gauss(0, 0.1) for _ in range(self._input_dim)]
            for _ in range(num_scales)
        ]
    
    def compress(self, memory: List[List[float]]) -> List[List[List[float]]]:
        """
        多尺度压缩
        
        Args:
            memory: 记忆矩阵
            
        Returns:
            多尺度压缩结果
        """
        compressed_results = []
        
        for compressor in self._compressors:
            scale_compressed = []
            for vec in memory:
                if len(vec) == self._input_dim:
                    compressed = compressor.compress(vec)
                    scale_compressed.append(compressed)
            compressed_results.append(scale_compressed)
        
        return compressed_results
    
    def _learnable_query(self, x: List[float]) -> List[float]:
        """
        可学习查询
        
        Args:
            x: 输入向量
            
        Returns:
            查询向量列表（每个尺度一个）
        """
        queries = []
        for i, weights in enumerate(self._query_weights):
            query = []
            for j in range(min(len(x), len(weights))):
                query.append(x[j] * weights[j])
            queries.append(query)
        return queries
    
    def reconstruct(self, compressed: List[List[List[float]]]) -> List[List[float]]:
        """
        重建记忆
        
        Args:
            compressed: 多尺度压缩结果
            
        Returns:
            重建的记忆矩阵
        """
        reconstructed = []
        
        if not compressed or not compressed[0]:
            return reconstructed
        
        # 使用最高分辨率（最小压缩比）的压缩器重建
        num_vectors = len(compressed[0])
        
        for i in range(num_vectors):
            # 加权融合多尺度重建
            fused = [0.0] * self._input_dim
            total_weight = 0.0
            
            for scale_idx, scale_compressed in enumerate(compressed):
                if i < len(scale_compressed):
                    compressor = self._compressors[scale_idx]
                    recon = compressor.decompress(scale_compressed[i])
                    
                    # 较低压缩比的权重更高
                    weight = 1.0 / self._compression_ratios[scale_idx]
                    
                    for j in range(min(len(fused), len(recon))):
                        fused[j] += weight * recon[j]
                    total_weight += weight
            
            if total_weight > 0:
                fused = [v / total_weight for v in fused]
            reconstructed.append(fused)
        
        return reconstructed
    
    def update_query_weights(self, gradients: List[List[float]], lr: float = 0.01) -> None:
        """
        更新查询权重
        
        Args:
            gradients: 梯度
            lr: 学习率
        """
        for i in range(len(self._query_weights)):
            for j in range(min(len(self._query_weights[i]), len(gradients[i]))):
                self._query_weights[i][j] -= lr * gradients[i][j]


# ============================================================================
# 4. LightweightMemory - LoRA风格轻量记忆
# ============================================================================

class LightweightMemory:
    """LoRA风格轻量记忆，使用低秩矩阵进行记忆注入"""
    
    def __init__(self, input_dim: int = 512, output_dim: int = 512, rank: int = 16):
        """
        初始化轻量记忆
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            rank: 低秩矩阵的秩
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._rank = rank
        
        # 初始化低秩矩阵 A 和 B
        # A: input_dim x rank
        # B: rank x output_dim
        self._A = self._init_matrix(input_dim, rank, scale=0.01)
        self._B = self._init_matrix(rank, output_dim, scale=1.0)
        
        # 缩放因子
        self._scaling = 1.0 / rank
    
    def _init_matrix(self, rows: int, cols: int, scale: float = 1.0) -> List[List[float]]:
        """初始化矩阵"""
        import random
        return [
            [random.gauss(0, scale) for _ in range(cols)]
            for _ in range(rows)
        ]
    
    def forward(self, x: List[float]) -> List[float]:
        """
        低秩记忆注入
        
        Args:
            x: 输入向量
            
        Returns:
            记忆注入后的输出
        """
        # x: [input_dim]
        # A: [input_dim, rank]
        # B: [rank, output_dim]
        
        # 计算 x @ A -> [rank]
        intermediate = []
        for j in range(self._rank):
            val = 0.0
            for i in range(min(len(x), self._input_dim)):
                val += x[i] * self._A[i][j]
            intermediate.append(val)
        
        # 计算 intermediate @ B -> [output_dim]
        output = []
        for j in range(self._output_dim):
            val = 0.0
            for i in range(self._rank):
                val += intermediate[i] * self._B[i][j]
            output.append(val * self._scaling)
        
        return output
    
    def update(self, delta_A: List[List[float]], delta_B: List[List[float]]) -> None:
        """
        更新记忆矩阵
        
        Args:
            delta_A: A矩阵的更新量
            delta_B: B矩阵的更新量
        """
        # 更新A
        for i in range(min(len(self._A), len(delta_A))):
            for j in range(min(len(self._A[i]), len(delta_A[i]))):
                self._A[i][j] += delta_A[i][j]
        
        # 更新B
        for i in range(min(len(self._B), len(delta_B))):
            for j in range(min(len(self._B[i]), len(delta_B[i]))):
                self._B[i][j] += delta_B[i][j]
    
    def get_memory_effect(self) -> List[List[float]]:
        """获取完整的记忆效果矩阵 (A @ B)"""
        result = []
        for i in range(self._input_dim):
            row = []
            for j in range(self._output_dim):
                val = 0.0
                for k in range(self._rank):
                    val += self._A[i][k] * self._B[k][j]
                row.append(val * self._scaling)
            result.append(row)
        return result
    
    def reset_memory(self) -> None:
        """重置记忆"""
        self._A = self._init_matrix(self._input_dim, self._rank, scale=0.01)
        self._B = self._init_matrix(self._rank, self._output_dim, scale=1.0)
    
    def get_parameters(self) -> Tuple[List[List[float]], List[List[float]]]:
        """获取参数"""
        return self._A, self._B
    
    def set_parameters(self, A: List[List[float]], B: List[List[float]]) -> None:
        """设置参数"""
        self._A = [row[:] for row in A]
        self._B = [row[:] for row in B]


# ============================================================================
# 5. MemoryFusion - 记忆融合
# ============================================================================

class MemoryFusion:
    """记忆融合，支持多种融合策略"""
    
    def __init__(self, dim: int = 512, gate_type: str = "linear"):
        """
        初始化记忆融合
        
        Args:
            dim: 向量维度
            gate_type: 门控类型 ("linear", "cross_attention", "gated")
        """
        self._dim = dim
        self._gate_type = gate_type
        
        # 门控参数
        self._gate_alpha = 0.5
        self._gate_weights = self._init_gate_weights()
    
    def _init_gate_weights(self) -> List[float]:
        """初始化门控权重"""
        import random
        return [random.gauss(0, 0.1) for _ in range(self._dim)]
    
    def forward(self, internal_mem: List[List[float]], 
                external_mem: List[List[float]]) -> List[List[float]]:
        """
        融合记忆
        
        Args:
            internal_mem: 内部记忆
            external_mem: 外部记忆
            
        Returns:
            融合后的记忆
        """
        if not internal_mem and not external_mem:
            return []
        
        if not internal_mem:
            return external_mem
        if not external_mem:
            return internal_mem
        
        if self._gate_type == "linear":
            return self._linear_gate(internal_mem, external_mem, self._gate_alpha)
        elif self._gate_type == "cross_attention":
            return self._cross_attention(internal_mem, external_mem, external_mem)
        elif self._gate_type == "gated":
            return self._gated_fusion(internal_mem, external_mem)
        else:
            return self._linear_gate(internal_mem, external_mem, 0.5)
    
    def _linear_gate(self, a: List[List[float]], b: List[List[float]], 
                     alpha: float) -> List[List[float]]:
        """
        线性门控融合
        
        Args:
            a: 记忆A
            b: 记忆B
            alpha: 融合权重
            
        Returns:
            融合结果
        """
        result = []
        max_len = max(len(a), len(b))
        
        for i in range(max_len):
            if i < len(a) and i < len(b):
                # 线性组合
                fused = [
                    alpha * a[i][j] + (1 - alpha) * b[i][j]
                    for j in range(min(len(a[i]), len(b[i])))
                ]
            elif i < len(a):
                fused = a[i][:]
            else:
                fused = b[i][:]
            result.append(fused)
        
        return result
    
    def _cross_attention(self, q: List[List[float]], 
                         internal: List[List[float]], 
                         external: List[List[float]]) -> List[List[float]]:
        """
        交叉注意力融合
        
        Args:
            q: 查询（内部记忆）
            internal: 内部记忆
            external: 外部记忆
            
        Returns:
            融合结果
        """
        result = []
        
        for query in q:
            # 计算与内部记忆的注意力
            internal_attn = self._compute_attention(query, internal)
            
            # 计算与外部记忆的注意力
            external_attn = self._compute_attention(query, external)
            
            # 加权融合
            fused = []
            max_len = max(len(internal_attn), len(external_attn))
            
            for j in range(max_len):
                val = 0.0
                if j < len(internal_attn):
                    val += 0.5 * internal_attn[j]
                if j < len(external_attn):
                    val += 0.5 * external_attn[j]
                fused.append(val)
            
            result.append(fused)
        
        return result
    
    def _compute_attention(self, query: List[float], 
                           memory: List[List[float]]) -> List[float]:
        """计算注意力加权输出"""
        if not memory:
            return [0.0] * len(query)
        
        # 计算注意力分数
        scores = []
        for mem in memory:
            score = sum(q * m for q, m in zip(query, mem))
            scores.append(score)
        
        # Softmax
        max_score = max(scores) if scores else 0
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        attn_weights = [e / sum_exp for e in exp_scores]
        
        # 加权求和
        result = [0.0] * len(query)
        for i, mem in enumerate(memory):
            for j in range(min(len(result), len(mem))):
                result[j] += attn_weights[i] * mem[j]
        
        return result
    
    def _gated_fusion(self, internal: List[List[float]], 
                      external: List[List[float]]) -> List[List[float]]:
        """门控融合"""
        result = []
        
        for i in range(max(len(internal), len(external))):
            if i < len(internal) and i < len(external):
                # 计算动态门控
                gate = self._compute_dynamic_gate(internal[i], external[i])
                
                fused = [
                    gate * internal[i][j] + (1 - gate) * external[i][j]
                    for j in range(min(len(internal[i]), len(external[i])))
                ]
            elif i < len(internal):
                fused = internal[i][:]
            else:
                fused = external[i][:]
            result.append(fused)
        
        return result
    
    def _compute_dynamic_gate(self, a: List[float], b: List[float]) -> float:
        """计算动态门控值"""
        # 基于向量范数的门控
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a + norm_b == 0:
            return 0.5
        
        return norm_a / (norm_a + norm_b)
    
    def set_gate_alpha(self, alpha: float) -> None:
        """设置门控alpha"""
        self._gate_alpha = max(0.0, min(1.0, alpha))


# ============================================================================
# 6. HierarchicalMemory - 分层记忆
# ============================================================================

class HierarchicalMemory:
    """分层记忆，支持短、中、长期记忆"""
    
    def __init__(self, 
                 short_term_capacity: int = 100,
                 mid_term_capacity: int = 1000,
                 long_term_capacity: int = 10000):
        """
        初始化分层记忆
        
        Args:
            short_term_capacity: 短期记忆容量
            mid_term_capacity: 中期记忆容量
            long_term_capacity: 长期记忆容量
        """
        # 短期记忆：容量小，更新快
        self._short_term = MemoryBank(capacity=short_term_capacity)
        
        # 中期记忆
        self._mid_term = MemoryBank(capacity=mid_term_capacity)
        
        # 长期记忆：容量大，更新慢
        self._long_term = MemoryBank(capacity=long_term_capacity)
        
        # 提升阈值
        self._promote_threshold = 3  # 访问次数阈值
        self._promote_importance = 2.0  # 重要性阈值
    
    def store(self, key: List[float], value: List[float], 
              timescale: str = "short", importance: float = 1.0) -> None:
        """
        按时间尺度存储记忆
        
        Args:
            key: 记忆键
            value: 记忆值
            timescale: 时间尺度 ("short", "mid", "long")
            importance: 重要性
        """
        if timescale == "short":
            self._short_term.store(key, value, importance)
        elif timescale == "mid":
            self._mid_term.store(key, value, importance)
        elif timescale == "long":
            self._long_term.store(key, value, importance)
        else:
            # 默认存储到短期记忆
            self._short_term.store(key, value, importance)
    
    def retrieve(self, query: List[float], 
                 timescales: List[str] = None,
                 top_k: int = 10) -> List[Tuple[str, MemoryEntry]]:
        """
        多尺度检索
        
        Args:
            query: 查询向量
            timescales: 要检索的时间尺度列表
            top_k: 每个尺度返回的数量
            
        Returns:
            (时间尺度, 记忆条目) 的列表
        """
        if timescales is None:
            timescales = ["short", "mid", "long"]
        
        results = []
        
        for scale in timescales:
            if scale == "short":
                memories = self._short_term.retrieve(query, top_k)
            elif scale == "mid":
                memories = self._mid_term.retrieve(query, top_k)
            elif scale == "long":
                memories = self._long_term.retrieve(query, top_k)
            else:
                continue
            
            for mem in memories:
                results.append((scale, mem))
        
        return results
    
    def _promote_to_longer_term(self, entry: MemoryEntry, 
                                from_scale: str) -> bool:
        """
        提升到更长期记忆
        
        Args:
            entry: 记忆条目
            from_scale: 当前时间尺度
            
        Returns:
            是否成功提升
        """
        # 检查是否满足提升条件
        if entry.access_count < self._promote_threshold:
            return False
        if entry.importance < self._promote_importance:
            return False
        
        # 提升记忆
        if from_scale == "short":
            self._mid_term.store(entry.key, entry.value, entry.importance)
            return True
        elif from_scale == "mid":
            self._long_term.store(entry.key, entry.value, entry.importance)
            return True
        
        return False
    
    def consolidate(self) -> Dict[str, int]:
        """
        记忆巩固：将满足条件的记忆提升到更长期
        
        Returns:
            各尺度提升的数量
        """
        promoted = {"short_to_mid": 0, "mid_to_long": 0}
        
        # 检查短期记忆是否需要提升
        for entry in self._short_term._memory[:]:
            if self._promote_to_longer_term(entry, "short"):
                promoted["short_to_mid"] += 1
        
        # 检查中期记忆是否需要提升
        for entry in self._mid_term._memory[:]:
            if self._promote_to_longer_term(entry, "mid"):
                promoted["mid_to_long"] += 1
        
        return promoted
    
    def decay_all(self, decay_rate: float = 0.99) -> None:
        """对所有记忆进行衰减"""
        for entry in self._short_term._memory:
            entry.decay_importance(decay_rate)
        for entry in self._mid_term._memory:
            entry.decay_importance(decay_rate)
        for entry in self._long_term._memory:
            entry.decay_importance(decay_rate)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "short_term": self._short_term.get_statistics(),
            "mid_term": self._mid_term.get_statistics(),
            "long_term": self._long_term.get_statistics(),
            "total_memories": (
                len(self._short_term) + 
                len(self._mid_term) + 
                len(self._long_term)
            )
        }
    
    def clear_all(self) -> None:
        """清空所有记忆"""
        self._short_term.clear()
        self._mid_term.clear()
        self._long_term.clear()
    
    def get_memory_bank(self, timescale: str) -> Optional[MemoryBank]:
        """获取指定时间尺度的记忆库"""
        if timescale == "short":
            return self._short_term
        elif timescale == "mid":
            return self._mid_term
        elif timescale == "long":
            return self._long_term
        return None


# ============================================================================
# 辅助函数
# ============================================================================

def create_memory_entry(key: List[float], value: List[float], 
                        importance: float = 1.0) -> MemoryEntry:
    """创建记忆条目的便捷函数"""
    return MemoryEntry(key=key, value=value, importance=importance)


def batch_store(memory_bank: MemoryBank, 
                keys: List[List[float]], 
                values: List[List[float]], 
                importances: List[float] = None) -> None:
    """批量存储记忆"""
    if importances is None:
        importances = [1.0] * len(keys)
    
    for key, value, imp in zip(keys, values, importances):
        memory_bank.store(key, value, imp)


def compute_memory_similarity(mem1: MemoryEntry, mem2: MemoryEntry) -> float:
    """计算两个记忆条目的相似度"""
    # 键相似度
    key_sim = _cosine_similarity(mem1.key, mem2.key)
    # 值相似度
    val_sim = _cosine_similarity(mem1.value, mem2.value)
    # 综合相似度
    return 0.5 * key_sim + 0.5 * val_sim


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    if len(a) != len(b):
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("记忆模块测试")
    print("=" * 60)
    
    # 测试 MemoryBank
    print("\n1. 测试 MemoryBank")
    bank = MemoryBank(capacity=100)
    
    # 存储一些记忆
    for i in range(10):
        key = [float(i), float(i+1), float(i+2)]
        value = [float(i*2), float(i*2+1)]
        bank.store(key, value, importance=float(i+1))
    
    print(f"存储了 {len(bank)} 条记忆")
    
    # 检索记忆
    query = [5.0, 6.0, 7.0]
    results = bank.retrieve(query, top_k=3)
    print(f"检索到 {len(results)} 条相似记忆")
    
    # 测试 AdaptiveMemoryCompressor
    print("\n2. 测试 AdaptiveMemoryCompressor")
    compressor = AdaptiveMemoryCompressor(input_dim=16, compression_ratios=[2, 4])
    
    memory = [[float(i+j) for j in range(16)] for i in range(5)]
    compressed = compressor.compress(memory)
    print(f"压缩后尺度数: {len(compressed)}")
    
    reconstructed = compressor.reconstruct(compressed)
    print(f"重建向量数: {len(reconstructed)}")
    
    # 测试 LightweightMemory
    print("\n3. 测试 LightweightMemory")
    lora_mem = LightweightMemory(input_dim=16, output_dim=16, rank=4)
    
    x = [1.0] * 16
    output = lora_mem.forward(x)
    print(f"LoRA输出维度: {len(output)}")
    
    # 测试 MemoryFusion
    print("\n4. 测试 MemoryFusion")
    fusion = MemoryFusion(dim=8, gate_type="linear")
    
    internal = [[1.0, 2.0, 3.0] for _ in range(3)]
    external = [[4.0, 5.0, 6.0] for _ in range(3)]
    fused = fusion.forward(internal, external)
    print(f"融合后向量数: {len(fused)}")
    
    # 测试 HierarchicalMemory
    print("\n5. 测试 HierarchicalMemory")
    hier_mem = HierarchicalMemory(
        short_term_capacity=10,
        mid_term_capacity=100,
        long_term_capacity=1000
    )
    
    # 存储到不同尺度
    hier_mem.store([1.0, 2.0], [3.0, 4.0], timescale="short")
    hier_mem.store([5.0, 6.0], [7.0, 8.0], timescale="mid")
    hier_mem.store([9.0, 10.0], [11.0, 12.0], timescale="long")
    
    stats = hier_mem.get_statistics()
    print(f"总记忆数: {stats['total_memories']}")
    
    print("\n" + "=" * 60)
    print("所有测试完成")
    print("=" * 60)
