"""
指纹验证 - 模型指纹验证
"""
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .model_hash import ModelFingerprint, ModelHasher


class VerificationStatus(Enum):
    """验证状态"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    """验证结果"""
    status: VerificationStatus
    fingerprint_id: str
    similarity: float
    matched_model: Optional[str]
    timestamp: float
    details: Dict[str, Any]


class FingerprintVerifier:
    """指纹验证器"""
    
    def __init__(self, hasher: Optional[ModelHasher] = None):
        self._hasher = hasher or ModelHasher()
        self._registered_models: Dict[str, ModelFingerprint] = {}
        self._verification_history: List[VerificationResult] = []
        self._similarity_threshold = 0.95
        self._suspicious_threshold = 0.7
    
    def register_model(
        self,
        model_name: str,
        fingerprint: ModelFingerprint
    ) -> None:
        """注册模型指纹"""
        self._registered_models[model_name] = fingerprint
    
    def verify_fingerprint(
        self,
        fingerprint: ModelFingerprint,
        expected_model: Optional[str] = None
    ) -> VerificationResult:
        """验证指纹"""
        import time
        
        # 如果指定了期望模型
        if expected_model:
            expected_fp = self._registered_models.get(expected_model)
            
            if expected_fp:
                similarities = self._hasher.compare_fingerprints(fingerprint, expected_fp)
                overall = similarities.get("overall", 0)
                
                if overall >= self._similarity_threshold:
                    status = VerificationStatus.VALID
                elif overall >= self._suspicious_threshold:
                    status = VerificationStatus.SUSPICIOUS
                else:
                    status = VerificationStatus.INVALID
                
                result = VerificationResult(
                    status=status,
                    fingerprint_id=fingerprint.fingerprint_id,
                    similarity=overall,
                    matched_model=expected_model if status == VerificationStatus.VALID else None,
                    timestamp=time.time(),
                    details={"similarities": similarities}
                )
                
                self._verification_history.append(result)
                return result
        
        # 在所有注册模型中查找匹配
        matches = self._hasher.identify_model(
            fingerprint,
            threshold=self._suspicious_threshold
        )
        
        if matches:
            best_match, best_similarity = matches[0]
            
            if best_similarity >= self._similarity_threshold:
                status = VerificationStatus.VALID
            else:
                status = VerificationStatus.SUSPICIOUS
            
            result = VerificationResult(
                status=status,
                fingerprint_id=fingerprint.fingerprint_id,
                similarity=best_similarity,
                matched_model=best_match.model_name,
                timestamp=time.time(),
                details={"all_matches": [(m.model_name, s) for m, s in matches[:5]]}
            )
        else:
            result = VerificationResult(
                status=VerificationStatus.UNKNOWN,
                fingerprint_id=fingerprint.fingerprint_id,
                similarity=0.0,
                matched_model=None,
                timestamp=time.time(),
                details={"message": "No matching model found"}
            )
        
        self._verification_history.append(result)
        return result
    
    def verify_model_weights(
        self,
        weights: Dict[str, Any],
        expected_model: str
    ) -> VerificationResult:
        """验证模型权重"""
        # 创建临时指纹
        fingerprint = self._hasher.create_fingerprint(
            model_name="verification_target",
            weights=weights
        )
        
        return self.verify_fingerprint(fingerprint, expected_model)
    
    def check_model_integrity(
        self,
        model_name: str,
        current_weights: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查模型完整性"""
        registered_fp = self._registered_models.get(model_name)
        
        if not registered_fp:
            return False, {"error": "Model not registered"}
        
        # 计算当前权重哈希
        current_hash = self._hasher.compute_weight_hash(current_weights)
        expected_hash = registered_fp.hash_values.get("weights", "")
        
        is_valid = current_hash == expected_hash
        
        return is_valid, {
            "current_hash": current_hash,
            "expected_hash": expected_hash,
            "match": is_valid
        }
    
    def detect_model_modification(
        self,
        original_fingerprint: ModelFingerprint,
        current_fingerprint: ModelFingerprint
    ) -> Dict[str, Any]:
        """检测模型修改"""
        similarities = self._hasher.compare_fingerprints(
            original_fingerprint,
            current_fingerprint
        )
        
        modifications = []
        
        for key, similarity in similarities.items():
            if key != "overall" and similarity < 0.99:
                modifications.append({
                    "component": key,
                    "similarity": similarity,
                    "change_detected": True
                })
        
        return {
            "has_modifications": len(modifications) > 0,
            "modifications": modifications,
            "overall_similarity": similarities.get("overall", 0)
        }
    
    def set_thresholds(
        self,
        similarity: float = 0.95,
        suspicious: float = 0.7
    ) -> None:
        """设置阈值"""
        self._similarity_threshold = similarity
        self._suspicious_threshold = suspicious
    
    def get_registered_models(self) -> List[str]:
        """获取已注册模型"""
        return list(self._registered_models.keys())
    
    def get_verification_history(
        self,
        limit: int = 100
    ) -> List[VerificationResult]:
        """获取验证历史"""
        return self._verification_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._verification_history:
            return {
                "total_verifications": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "suspicious_count": 0
            }
        
        status_counts = {}
        for result in self._verification_history:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_verifications": len(self._verification_history),
            "registered_models": len(self._registered_models),
            **status_counts
        }
    
    def clear_history(self) -> None:
        """清除历史"""
        self._verification_history.clear()
    
    def export_registry(self) -> str:
        """导出注册表"""
        return json.dumps({
            name: self._hasher.export_fingerprint(fp)
            for name, fp in self._registered_models.items()
        }, indent=2)
    
    def import_registry(self, data: str) -> int:
        """导入注册表"""
        registry = json.loads(data)
        count = 0
        
        for name, fp_data in registry.items():
            fingerprint = self._hasher.import_fingerprint(fp_data)
            self._registered_models[name] = fingerprint
            count += 1
        
        return count
