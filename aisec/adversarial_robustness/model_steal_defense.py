"""
模型窃取防御 - 防止模型被窃取
"""
import hashlib
import time
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class DefenseStrategy(Enum):
    """防御策略"""
    OUTPUT_ROUNDING = "output_rounding"
    NOISE_INJECTION = "noise_injection"
    QUERY_LIMITING = "query_limiting"
    WATERMARKING = "watermarking"
    ADVERSARIAL_TRAP = "adversarial_trap"
    CONFIDENCE_MASKING = "confidence_masking"


@dataclass
class QueryRecord:
    """查询记录"""
    query_id: str
    input_hash: str
    timestamp: float
    output: Any
    client_id: str = ""


@dataclass
class DefenseConfig:
    """防御配置"""
    strategies: List[DefenseStrategy] = field(default_factory=lambda: [DefenseStrategy.OUTPUT_ROUNDING])
    query_limit: int = 1000
    time_window: int = 3600
    noise_scale: float = 0.01
    rounding_precision: int = 3
    trap_samples: int = 100


class ModelStealDefense:
    """模型窃取防御"""
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self._config = config or DefenseConfig()
        self._query_history: Dict[str, List[QueryRecord]] = defaultdict(list)
        self._client_queries: Dict[str, List[float]] = defaultdict(list)
        self._trap_inputs: Set[str] = set()
        self._detection_threshold = 0.8
    
    def defend_output(
        self,
        output: List[float],
        input_data: List[float],
        client_id: str = ""
    ) -> Tuple[List[float], Dict[str, Any]]:
        """防御处理输出"""
        defended_output = output.copy()
        applied_defenses = []
        
        for strategy in self._config.strategies:
            if strategy == DefenseStrategy.OUTPUT_ROUNDING:
                defended_output = self._apply_rounding(defended_output)
                applied_defenses.append("rounding")
            
            elif strategy == DefenseStrategy.NOISE_INJECTION:
                defended_output = self._apply_noise(defended_output)
                applied_defenses.append("noise")
            
            elif strategy == DefenseStrategy.CONFIDENCE_MASKING:
                defended_output = self._mask_confidence(defended_output)
                applied_defenses.append("confidence_mask")
        
        # 记录查询
        self._record_query(input_data, defended_output, client_id)
        
        return defended_output, {"defenses": applied_defenses}
    
    def _apply_rounding(self, output: List[float]) -> List[float]:
        """应用输出舍入"""
        precision = self._config.rounding_precision
        return [round(x, precision) for x in output]
    
    def _apply_noise(self, output: List[float]) -> List[float]:
        """注入噪声"""
        import random
        scale = self._config.noise_scale
        return [x + random.gauss(0, scale * abs(x)) for x in output]
    
    def _mask_confidence(self, output: List[float]) -> List[float]:
        """掩盖置信度"""
        # 降低最大值的显著性
        if not output:
            return output
        
        max_val = max(output)
        min_val = min(output)
        range_val = max_val - min_val
        
        if range_val > 0:
            # 压缩范围
            compressed = [
                min_val + (x - min_val) * 0.8
                for x in output
            ]
            return compressed
        
        return output
    
    def _record_query(
        self,
        input_data: List[float],
        output: List[float],
        client_id: str
    ) -> None:
        """记录查询"""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:16]
        query_id = hashlib.md5(f"{input_hash}{time.time()}".encode()).hexdigest()[:16]
        
        record = QueryRecord(
            query_id=query_id,
            input_hash=input_hash,
            timestamp=time.time(),
            output=output,
            client_id=client_id
        )
        
        self._query_history[client_id].append(record)
        self._client_queries[client_id].append(record.timestamp)
    
    def check_query_limit(self, client_id: str) -> Tuple[bool, int]:
        """检查查询限制"""
        current_time = time.time()
        cutoff = current_time - self._config.time_window
        
        # 清理旧记录
        self._client_queries[client_id] = [
            t for t in self._client_queries[client_id] if t > cutoff
        ]
        
        query_count = len(self._client_queries[client_id])
        is_allowed = query_count < self._config.query_limit
        
        return is_allowed, self._config.query_limit - query_count
    
    def detect_extraction_attack(self, client_id: str) -> Tuple[bool, float, str]:
        """检测模型窃取攻击"""
        queries = self._query_history.get(client_id, [])
        
        if len(queries) < 100:
            return False, 0.0, "insufficient_data"
        
        # 检测指标
        scores = []
        
        # 1. 查询频率
        if len(queries) > 500:
            scores.append(0.3)
        
        # 2. 输入多样性
        unique_inputs = len(set(q.input_hash for q in queries))
        diversity_ratio = unique_inputs / len(queries)
        if diversity_ratio > 0.9:
            scores.append(0.3)
        
        # 3. 输出分布
        outputs = [q.output for q in queries if q.output]
        if outputs:
            output_variance = self._calculate_output_variance(outputs)
            if output_variance > 0.5:
                scores.append(0.2)
        
        # 4. 输入空间覆盖
        if self._check_input_coverage(queries):
            scores.append(0.2)
        
        total_score = sum(scores)
        is_attack = total_score > self._detection_threshold
        
        reason = "extraction_detected" if is_attack else "normal"
        
        return is_attack, total_score, reason
    
    def _calculate_output_variance(self, outputs: List[List[float]]) -> float:
        """计算输出方差"""
        if not outputs or not outputs[0]:
            return 0.0
        
        # 取第一个维度
        first_dim = [o[0] if o else 0 for o in outputs]
        mean = sum(first_dim) / len(first_dim)
        variance = sum((x - mean) ** 2 for x in first_dim) / len(first_dim)
        
        return math.sqrt(variance)
    
    def _check_input_coverage(self, queries: List[QueryRecord]) -> bool:
        """检查输入空间覆盖"""
        # 简化：检查是否有大量不同的输入
        unique_inputs = set(q.input_hash for q in queries)
        return len(unique_inputs) > len(queries) * 0.8
    
    def generate_trap_samples(self, num_samples: int = 100) -> List[List[float]]:
        """生成陷阱样本"""
        import random
        trap_samples = []
        
        for _ in range(num_samples):
            # 生成随机陷阱输入
            sample = [random.gauss(0, 1) for _ in range(10)]
            trap_samples.append(sample)
            
            # 记录陷阱
            input_hash = hashlib.md5(str(sample).encode()).hexdigest()[:16]
            self._trap_inputs.add(input_hash)
        
        return trap_samples
    
    def is_trap_triggered(self, input_data: List[float]) -> bool:
        """检查是否触发陷阱"""
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:16]
        return input_hash in self._trap_inputs
    
    def get_statistics(self, client_id: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        if client_id:
            queries = self._query_history.get(client_id, [])
            return {
                "total_queries": len(queries),
                "unique_inputs": len(set(q.input_hash for q in queries)),
                "first_query": min((q.timestamp for q in queries), default=0),
                "last_query": max((q.timestamp for q in queries), default=0)
            }
        
        return {
            "total_clients": len(self._query_history),
            "total_queries": sum(len(q) for q in self._query_history.values()),
            "trap_samples": len(self._trap_inputs)
        }
    
    def reset_client(self, client_id: str) -> None:
        """重置客户端记录"""
        self._query_history.pop(client_id, None)
        self._client_queries.pop(client_id, None)
