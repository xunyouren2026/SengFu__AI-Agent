"""
全局对抗样本存储
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import copy


class AdversarialType(Enum):
    """对抗样本类型"""
    FGSM = "fgsm"
    PGD = "pgd"
    CW = "cw"
    DEEPFOOL = "deepfool"
    AUTO_ATTACK = "auto_attack"


class AdversarialSample:
    """对抗样本"""
    
    def __init__(
        self,
        sample_id: str,
        original_data: Any,
        adversarial_data: Any,
        label: int,
        attack_type: AdversarialType,
        epsilon: float = 0.1,
        source_client: Optional[str] = None
    ):
        self.sample_id = sample_id
        self.original_data = original_data
        self.adversarial_data = adversarial_data
        self.label = label
        self.attack_type = attack_type
        self.epsilon = epsilon
        self.source_client = source_client
        self.created_at = datetime.now().timestamp()
        self.is_verified = False
        self.success_rate: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    def compute_perturbation_norm(self) -> float:
        """计算扰动范数"""
        # 简化实现
        if isinstance(self.original_data, list) and isinstance(self.adversarial_data, list):
            diff = [o - a for o, a in zip(self.original_data, self.adversarial_data)]
            return sum(d ** 2 for d in diff) ** 0.5
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'sample_id': self.sample_id,
            'label': self.label,
            'attack_type': self.attack_type.value,
            'epsilon': self.epsilon,
            'source_client': self.source_client,
            'created_at': self.created_at,
            'is_verified': self.is_verified,
            'success_rate': self.success_rate,
            'perturbation_norm': self.compute_perturbation_norm()
        }


class GlobalAdversarialStore:
    """
    全局对抗样本存储
    
    在联邦学习节点间共享对抗样本
    """
    
    def __init__(
        self,
        max_samples: int = 10000,
        verify_threshold: float = 0.5
    ):
        self.max_samples = max_samples
        self.verify_threshold = verify_threshold
        
        self._samples: Dict[str, AdversarialSample] = {}
        self._by_type: Dict[AdversarialType, Set[str]] = {}
        self._by_client: Dict[str, Set[str]] = {}
        self._by_label: Dict[int, Set[str]] = {}
        
        # 统计
        self._total_added = 0
        self._total_verified = 0
    
    def add(
        self,
        sample: AdversarialSample
    ) -> bool:
        """
        添加对抗样本
        
        Args:
            sample: 对抗样本
        
        Returns:
            是否成功
        """
        if len(self._samples) >= self.max_samples:
            self._evict_old()
        
        sample_id = sample.sample_id
        self._samples[sample_id] = sample
        
        # 按类型索引
        if sample.attack_type not in self._by_type:
            self._by_type[sample.attack_type] = set()
        self._by_type[sample.attack_type].add(sample_id)
        
        # 按客户端索引
        if sample.source_client:
            if sample.source_client not in self._by_client:
                self._by_client[sample.source_client] = set()
            self._by_client[sample.source_client].add(sample_id)
        
        # 按标签索引
        if sample.label not in self._by_label:
            self._by_label[sample.label] = set()
        self._by_label[sample.label].add(sample_id)
        
        self._total_added += 1
        return True
    
    def get(self, sample_id: str) -> Optional[AdversarialSample]:
        """获取对抗样本"""
        return self._samples.get(sample_id)
    
    def get_by_type(
        self,
        attack_type: AdversarialType
    ) -> List[AdversarialSample]:
        """按类型获取"""
        ids = self._by_type.get(attack_type, set())
        return [self._samples[sid] for sid in ids if sid in self._samples]
    
    def get_by_client(
        self,
        client_id: str
    ) -> List[AdversarialSample]:
        """按客户端获取"""
        ids = self._by_client.get(client_id, set())
        return [self._samples[sid] for sid in ids if sid in self._samples]
    
    def get_by_label(
        self,
        label: int
    ) -> List[AdversarialSample]:
        """按标签获取"""
        ids = self._by_label.get(label, set())
        return [self._samples[sid] for sid in ids if sid in self._samples]
    
    def get_verified(self) -> List[AdversarialSample]:
        """获取已验证的样本"""
        return [s for s in self._samples.values() if s.is_verified]
    
    def verify(
        self,
        sample_id: str,
        success_rate: float
    ) -> bool:
        """
        验证对抗样本
        
        Args:
            sample_id: 样本ID
            success_rate: 攻击成功率
        """
        if sample_id not in self._samples:
            return False
        
        sample = self._samples[sample_id]
        sample.success_rate = success_rate
        sample.is_verified = success_rate >= self.verify_threshold
        
        if sample.is_verified:
            self._total_verified += 1
        
        return sample.is_verified
    
    def merge(
        self,
        other_samples: List[AdversarialSample],
        source_node: str
    ) -> int:
        """
        合并来自其他节点的样本
        
        Returns:
            合并的样本数量
        """
        merged = 0
        for sample in other_samples:
            if sample.sample_id not in self._samples:
                sample.metadata['merged_from'] = source_node
                self.add(sample)
                merged += 1
        
        return merged
    
    def export_for_client(
        self,
        client_id: str,
        max_count: int = 100
    ) -> List[AdversarialSample]:
        """
        为客户端导出样本
        
        排除该客户端自己生成的样本
        """
        own_ids = self._by_client.get(client_id, set())
        
        candidates = [
            s for s in self._samples.values()
            if s.sample_id not in own_ids and s.is_verified
        ]
        
        # 按成功率排序
        candidates.sort(key=lambda s: s.success_rate, reverse=True)
        
        return candidates[:max_count]
    
    def remove(self, sample_id: str) -> bool:
        """移除样本"""
        if sample_id not in self._samples:
            return False
        
        sample = self._samples[sample_id]
        
        # 从索引中移除
        if sample.attack_type in self._by_type:
            self._by_type[sample.attack_type].discard(sample_id)
        
        if sample.source_client and sample.source_client in self._by_client:
            self._by_client[sample.source_client].discard(sample_id)
        
        if sample.label in self._by_label:
            self._by_label[sample.label].discard(sample_id)
        
        del self._samples[sample_id]
        return True
    
    def _evict_old(self) -> None:
        """淘汰旧样本"""
        # 移除最旧的10%
        n_remove = len(self._samples) // 10
        sorted_samples = sorted(
            self._samples.values(),
            key=lambda s: s.created_at
        )
        
        for sample in sorted_samples[:n_remove]:
            self.remove(sample.sample_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        verified_count = sum(1 for s in self._samples.values() if s.is_verified)
        
        type_counts = {
            t.value: len(ids) for t, ids in self._by_type.items()
        }
        
        return {
            'total_samples': len(self._samples),
            'verified_samples': verified_count,
            'total_added': self._total_added,
            'total_verified': self._total_verified,
            'max_samples': self.max_samples,
            'by_type': type_counts,
            'unique_clients': len(self._by_client),
            'unique_labels': len(self._by_label)
        }


class AdversarialDefenseTracker:
    """
    对抗防御跟踪器
    
    跟踪模型对对抗样本的防御效果
    """
    
    def __init__(self):
        self._defense_history: List[Dict[str, Any]] = []
        self._model_versions: Dict[int, float] = {}  # version -> defense_rate
    
    def record_defense(
        self,
        model_version: int,
        sample_id: str,
        defended: bool,
        confidence: float
    ) -> None:
        """记录防御结果"""
        self._defense_history.append({
            'model_version': model_version,
            'sample_id': sample_id,
            'defended': defended,
            'confidence': confidence,
            'timestamp': datetime.now().timestamp()
        })
    
    def compute_defense_rate(
        self,
        model_version: int
    ) -> float:
        """计算模型的防御率"""
        records = [
            r for r in self._defense_history
            if r['model_version'] == model_version
        ]
        
        if not records:
            return 0.0
        
        defended = sum(1 for r in records if r['defended'])
        return defended / len(records)
    
    def get_defense_trend(self) -> List[Tuple[int, float]]:
        """获取防御趋势"""
        versions = sorted(set(r['model_version'] for r in self._defense_history))
        return [(v, self.compute_defense_rate(v)) for v in versions]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_records': len(self._defense_history),
            'model_versions': len(self._model_versions),
            'defense_trend': self.get_defense_trend()[-10:]  # 最近10个版本
        }
