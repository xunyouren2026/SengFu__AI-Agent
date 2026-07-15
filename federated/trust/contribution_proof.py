"""
贡献证明
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import random


class ProofType(Enum):
    """证明类型"""
    GRADIENT = "gradient"  # 梯度证明
    LOSS = "loss"  # 损失证明
    DATA = "data"  # 数据证明
    COMPUTATION = "computation"  # 计算证明


class ContributionProof:
    """贡献证明"""
    
    def __init__(
        self,
        proof_id: str,
        client_id: str,
        proof_type: ProofType,
        proof_data: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        self.proof_id = proof_id
        self.client_id = client_id
        self.proof_type = proof_type
        self.proof_data = proof_data
        self.timestamp = timestamp or datetime.now().timestamp()
        self.verified = False
        self.contribution_value: float = 0.0
    
    def compute_hash(self) -> str:
        """计算证明哈希"""
        content = f"{self.proof_id}{self.client_id}{self.proof_type.value}{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proof_id': self.proof_id,
            'client_id': self.client_id,
            'proof_type': self.proof_type.value,
            'proof_data': self.proof_data,
            'timestamp': self.timestamp,
            'verified': self.verified,
            'contribution_value': self.contribution_value
        }


class ContributionVerifier:
    """贡献验证器"""
    
    def __init__(
        self,
        tolerance: float = 0.01,
        min_samples: int = 10
    ):
        self.tolerance = tolerance
        self.min_samples = min_samples
    
    def verify_gradient_proof(
        self,
        proof: ContributionProof,
        expected_norm_range: Tuple[float, float] = (0.0, 100.0)
    ) -> bool:
        """验证梯度证明"""
        gradient = proof.proof_data.get('gradient', {})
        
        if not gradient:
            return False
        
        # 计算梯度范数
        norm = 0.0
        for key, value in gradient.items():
            if isinstance(value, (int, float)):
                norm += value ** 2
            elif isinstance(value, list):
                norm += sum(v ** 2 for v in value if isinstance(v, (int, float)))
        
        norm = norm ** 0.5
        
        # 检查是否在合理范围
        min_norm, max_norm = expected_norm_range
        return min_norm <= norm <= max_norm
    
    def verify_loss_proof(
        self,
        proof: ContributionProof,
        previous_loss: float
    ) -> bool:
        """验证损失证明"""
        reported_loss = proof.proof_data.get('loss')
        
        if reported_loss is None:
            return False
        
        # 损失应该下降或保持
        return reported_loss <= previous_loss + self.tolerance
    
    def verify_data_proof(
        self,
        proof: ContributionProof
    ) -> bool:
        """验证数据证明"""
        num_samples = proof.proof_data.get('num_samples', 0)
        data_hash = proof.proof_data.get('data_hash')
        
        # 检查样本数量
        if num_samples < self.min_samples:
            return False
        
        # 检查数据哈希
        if data_hash is None:
            return False
        
        return True


class ContributionProofSystem:
    """
    贡献证明系统
    
    生成和验证贡献证明
    """
    
    def __init__(self):
        self._proofs: Dict[str, ContributionProof] = {}
        self._client_proofs: Dict[str, List[str]] = {}  # client_id -> proof_ids
        self._verifier = ContributionVerifier()
        
        self._total_proofs = 0
        self._total_verified = 0
    
    def generate_gradient_proof(
        self,
        client_id: str,
        gradient: Dict[str, Any],
        loss: float,
        num_samples: int
    ) -> ContributionProof:
        """
        生成梯度证明
        
        Args:
            client_id: 客户端ID
            gradient: 梯度
            loss: 损失值
            num_samples: 样本数
        """
        proof_id = f"grad_{client_id}_{self._total_proofs}"
        
        proof_data = {
            'gradient': gradient,
            'loss': loss,
            'num_samples': num_samples,
            'gradient_norm': self._compute_norm(gradient)
        }
        
        proof = ContributionProof(
            proof_id=proof_id,
            client_id=client_id,
            proof_type=ProofType.GRADIENT,
            proof_data=proof_data
        )
        
        self._store_proof(proof)
        return proof
    
    def generate_loss_proof(
        self,
        client_id: str,
        initial_loss: float,
        final_loss: float,
        num_epochs: int
    ) -> ContributionProof:
        """生成损失证明"""
        proof_id = f"loss_{client_id}_{self._total_proofs}"
        
        proof_data = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': initial_loss - final_loss,
            'num_epochs': num_epochs
        }
        
        proof = ContributionProof(
            proof_id=proof_id,
            client_id=client_id,
            proof_type=ProofType.LOSS,
            proof_data=proof_data
        )
        
        self._store_proof(proof)
        return proof
    
    def generate_data_proof(
        self,
        client_id: str,
        num_samples: int,
        data_stats: Dict[str, Any]
    ) -> ContributionProof:
        """生成数据证明"""
        proof_id = f"data_{client_id}_{self._total_proofs}"
        
        # 计算数据哈希
        data_hash = self._compute_data_hash(data_stats)
        
        proof_data = {
            'num_samples': num_samples,
            'data_hash': data_hash,
            'stats': data_stats
        }
        
        proof = ContributionProof(
            proof_id=proof_id,
            client_id=client_id,
            proof_type=ProofType.DATA,
            proof_data=proof_data
        )
        
        self._store_proof(proof)
        return proof
    
    def verify(
        self,
        proof: ContributionProof,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float]:
        """
        验证证明
        
        Args:
            proof: 贡献证明
            context: 验证上下文
        
        Returns:
            (是否验证通过, 贡献值)
        """
        context = context or {}
        
        if proof.proof_type == ProofType.GRADIENT:
            valid = self._verifier.verify_gradient_proof(proof)
        elif proof.proof_type == ProofType.LOSS:
            previous_loss = context.get('previous_loss', float('inf'))
            valid = self._verifier.verify_loss_proof(proof, previous_loss)
        elif proof.proof_type == ProofType.DATA:
            valid = self._verifier.verify_data_proof(proof)
        else:
            valid = True
        
        if valid:
            proof.verified = True
            proof.contribution_value = self._compute_contribution_value(proof)
            self._total_verified += 1
        
        return valid, proof.contribution_value
    
    def _store_proof(self, proof: ContributionProof) -> None:
        """存储证明"""
        self._proofs[proof.proof_id] = proof
        self._total_proofs += 1
        
        if proof.client_id not in self._client_proofs:
            self._client_proofs[proof.client_id] = []
        self._client_proofs[proof.client_id].append(proof.proof_id)
    
    def _compute_norm(self, gradient: Dict[str, Any]) -> float:
        """计算梯度范数"""
        norm = 0.0
        for value in gradient.values():
            if isinstance(value, (int, float)):
                norm += value ** 2
            elif isinstance(value, list):
                norm += sum(v ** 2 for v in value if isinstance(v, (int, float)))
        return norm ** 0.5
    
    def _compute_data_hash(self, data_stats: Dict[str, Any]) -> str:
        """计算数据哈希"""
        content = str(sorted(data_stats.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _compute_contribution_value(self, proof: ContributionProof) -> float:
        """计算贡献值"""
        if proof.proof_type == ProofType.GRADIENT:
            norm = proof.proof_data.get('gradient_norm', 0)
            num_samples = proof.proof_data.get('num_samples', 0)
            return norm * num_samples / 1000  # 归一化
        
        elif proof.proof_type == ProofType.LOSS:
            reduction = proof.proof_data.get('loss_reduction', 0)
            return max(0, reduction)  # 损失减少为正贡献
        
        elif proof.proof_type == ProofType.DATA:
            num_samples = proof.proof_data.get('num_samples', 0)
            return num_samples / 1000  # 归一化
        
        return 0.0
    
    def get_proof(self, proof_id: str) -> Optional[ContributionProof]:
        """获取证明"""
        return self._proofs.get(proof_id)
    
    def get_client_proofs(
        self,
        client_id: str
    ) -> List[ContributionProof]:
        """获取客户端的所有证明"""
        proof_ids = self._client_proofs.get(client_id, [])
        return [self._proofs[pid] for pid in proof_ids if pid in self._proofs]
    
    def get_total_contribution(
        self,
        client_id: str
    ) -> float:
        """获取客户端总贡献"""
        proofs = self.get_client_proofs(client_id)
        return sum(p.contribution_value for p in proofs if p.verified)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_proofs': self._total_proofs,
            'total_verified': self._total_verified,
            'verification_rate': self._total_verified / self._total_proofs if self._total_proofs > 0 else 0,
            'num_clients': len(self._client_proofs)
        }
