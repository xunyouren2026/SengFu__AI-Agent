"""
联邦知识蒸馏
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import math


class DistillationType(Enum):
    """蒸馏类型"""
    LOGITS = "logits"  # Logits蒸馏
    FEATURES = "features"  # 特征蒸馏
    ATTENTION = "attention"  # 注意力蒸馏
    RELATION = "relation"  # 关系蒸馏


class KnowledgeBuffer:
    """知识缓冲区"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: List[Dict[str, Any]] = []
    
    def add(self, knowledge: Dict[str, Any]) -> None:
        """添加知识"""
        self._buffer.append(knowledge)
        
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有知识"""
        return self._buffer.copy()
    
    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


class FederatedDistillation:
    """
    联邦知识蒸馏
    
    在服务器端聚合客户端知识
    """
    
    def __init__(
        self,
        distillation_type: DistillationType = DistillationType.LOGITS,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        """
        Args:
            distillation_type: 蒸馏类型
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
        """
        self.distillation_type = distillation_type
        self.temperature = temperature
        self.alpha = alpha
        
        self._teacher_knowledge: Dict[str, KnowledgeBuffer] = {}
        self._student_knowledge: Dict[str, KnowledgeBuffer] = {}
        
        # 统计
        self._total_distillations = 0
    
    def collect_teacher_knowledge(
        self,
        client_id: str,
        logits: List[List[float]],
        labels: List[int]
    ) -> None:
        """
        收集教师知识
        
        Args:
            client_id: 客户端ID
            logits: 模型输出logits
            labels: 真实标签
        """
        if client_id not in self._teacher_knowledge:
            self._teacher_knowledge[client_id] = KnowledgeBuffer()
        
        knowledge = {
            'logits': logits,
            'labels': labels,
            'timestamp': datetime.now().timestamp()
        }
        
        self._teacher_knowledge[client_id].add(knowledge)
    
    def aggregate_teacher_knowledge(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        聚合教师知识
        
        Args:
            weights: 客户端权重
        """
        all_knowledge = []
        
        for client_id, buffer in self._teacher_knowledge.items():
            weight = weights.get(client_id, 1.0) if weights else 1.0
            
            for knowledge in buffer.get_all():
                knowledge['weight'] = weight
                all_knowledge.append(knowledge)
        
        if not all_knowledge:
            return {}
        
        # 聚合logits
        aggregated_logits: List[List[float]] = []
        
        # 简化实现：加权平均
        for knowledge in all_knowledge:
            weight = knowledge['weight']
            logits = knowledge['logits']
            
            if not aggregated_logits:
                aggregated_logits = [[w * v for v in row] for row in logits]
            else:
                for i, row in enumerate(logits):
                    if i < len(aggregated_logits):
                        for j, v in enumerate(row):
                            if j < len(aggregated_logits[i]):
                                aggregated_logits[i][j] += weight * v
        
        # 归一化
        total_weight = sum(k['weight'] for k in all_knowledge)
        if total_weight > 0:
            aggregated_logits = [
                [v / total_weight for v in row]
                for row in aggregated_logits
            ]
        
        return {
            'aggregated_logits': aggregated_logits,
            'num_samples': len(all_knowledge),
            'temperature': self.temperature
        }
    
    def compute_distillation_loss(
        self,
        student_logits: List[List[float]],
        teacher_logits: List[List[float]]
    ) -> float:
        """
        计算蒸馏损失
        
        使用KL散度
        """
        if not student_logits or not teacher_logits:
            return 0.0
        
        total_loss = 0.0
        count = 0
        
        for s_row, t_row in zip(student_logits, teacher_logits):
            # 软化
            s_soft = self._softmax(s_row, self.temperature)
            t_soft = self._softmax(t_row, self.temperature)
            
            # KL散度
            kl = self._kl_divergence(s_soft, t_soft)
            total_loss += kl
            count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def _softmax(self, logits: List[float], temperature: float) -> List[float]:
        """计算softmax"""
        max_val = max(logits)
        exp_vals = [math.exp((v - max_val) / temperature) for v in logits]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]
    
    def _kl_divergence(
        self,
        p: List[float],
        q: List[float]
    ) -> float:
        """计算KL散度"""
        kl = 0.0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)
        return kl
    
    def distill(
        self,
        student_id: str,
        student_logits: List[List[float]],
        teacher_knowledge: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        执行蒸馏
        
        Args:
            student_id: 学生客户端ID
            student_logits: 学生输出
            teacher_knowledge: 教师知识
        
        Returns:
            (蒸馏损失, 教师知识)
        """
        if teacher_knowledge is None:
            teacher_knowledge = self.aggregate_teacher_knowledge()
        
        teacher_logits = teacher_knowledge.get('aggregated_logits', [])
        
        loss = self.compute_distillation_loss(student_logits, teacher_logits)
        
        self._total_distillations += 1
        
        return loss, teacher_knowledge
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'distillation_type': self.distillation_type.value,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'total_distillations': self._total_distillations,
            'num_teachers': len(self._teacher_knowledge),
            'num_students': len(self._student_knowledge)
        }


class FeatureDistillation(FederatedDistillation):
    """
    特征蒸馏
    
    在特征层面进行知识蒸馏
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__(
            distillation_type=DistillationType.FEATURES,
            temperature=temperature
        )
    
    def compute_feature_loss(
        self,
        student_features: List[List[float]],
        teacher_features: List[List[float]]
    ) -> float:
        """计算特征损失"""
        if not student_features or not teacher_features:
            return 0.0
        
        total_loss = 0.0
        count = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # L2距离
            loss = sum((s - t) ** 2 for s, t in zip(s_feat, t_feat))
            total_loss += loss
            count += 1
        
        return total_loss / count if count > 0 else 0.0


class AttentionDistillation(FederatedDistillation):
    """
    注意力蒸馏
    
    在注意力图层面进行知识蒸馏
    """
    
    def __init__(self):
        super().__init__(distillation_type=DistillationType.ATTENTION)
    
    def compute_attention_loss(
        self,
        student_attention: List[List[List[float]]],
        teacher_attention: List[List[List[float]]]
    ) -> float:
        """计算注意力损失"""
        if not student_attention or not teacher_attention:
            return 0.0
        
        total_loss = 0.0
        count = 0
        
        for s_attn, t_attn in zip(student_attention, teacher_attention):
            # MSE损失
            for s_row, t_row in zip(s_attn, t_attn):
                for s_val, t_val in zip(s_row, t_row):
                    total_loss += (s_val - t_val) ** 2
                    count += 1
        
        return total_loss / count if count > 0 else 0.0
