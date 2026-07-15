"""
模型指纹 - AI模型指纹识别
"""
import hashlib
import json
import struct
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FingerprintType(Enum):
    """指纹类型"""
    WEIGHT_HASH = "weight_hash"
    ARCHITECTURE = "architecture"
    OUTPUT_PATTERN = "output_pattern"
    GRADIENT_PATTERN = "gradient_pattern"
    COMBINED = "combined"


@dataclass
class ModelFingerprint:
    """模型指纹"""
    fingerprint_id: str
    fingerprint_type: FingerprintType
    model_name: str
    hash_values: Dict[str, str]
    features: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelHasher:
    """模型哈希器"""
    
    def __init__(self):
        self._fingerprints: Dict[str, ModelFingerprint] = {}
    
    def compute_weight_hash(
        self,
        weights: Dict[str, Any],
        sample_ratio: float = 0.1
    ) -> str:
        """计算权重哈希"""
        hash_components = []
        
        for name, weight in weights.items():
            # 采样计算
            if hasattr(weight, '__len__'):
                n = len(weight)
                sample_size = max(1, int(n * sample_ratio))
                indices = [i * n // sample_size for i in range(sample_size)]
                
                samples = []
                for idx in indices:
                    if idx < len(weight):
                        val = weight[idx]
                        if isinstance(val, (int, float)):
                            samples.append(struct.pack('d', float(val)))
                
                if samples:
                    component_hash = hashlib.sha256(b''.join(samples)).hexdigest()
                    hash_components.append(f"{name}:{component_hash}")
        
        # 组合所有组件
        combined = hashlib.sha256('|'.join(hash_components).encode()).hexdigest()
        return combined
    
    def compute_architecture_hash(
        self,
        architecture: Dict[str, Any]
    ) -> str:
        """计算架构哈希"""
        # 提取架构特征
        features = {
            "layers": architecture.get("layers", []),
            "layer_types": architecture.get("layer_types", []),
            "activation_functions": architecture.get("activations", []),
            "input_shape": str(architecture.get("input_shape", "")),
            "output_shape": str(architecture.get("output_shape", "")),
        }
        
        return hashlib.sha256(json.dumps(features, sort_keys=True).encode()).hexdigest()
    
    def compute_output_pattern_hash(
        self,
        outputs: List[List[float]],
        inputs: List[List[float]]
    ) -> str:
        """计算输出模式哈希"""
        patterns = []
        
        # 计算输出统计特征
        for output in outputs:
            if output:
                mean = sum(output) / len(output)
                variance = sum((x - mean) ** 2 for x in output) / len(output)
                max_val = max(output)
                min_val = min(output)
                
                pattern = f"{mean:.6f}:{variance:.6f}:{max_val:.6f}:{min_val:.6f}"
                patterns.append(pattern)
        
        return hashlib.sha256('|'.join(patterns).encode()).hexdigest()
    
    def create_fingerprint(
        self,
        model_name: str,
        weights: Optional[Dict[str, Any]] = None,
        architecture: Optional[Dict[str, Any]] = None,
        outputs: Optional[List[List[float]]] = None,
        inputs: Optional[List[List[float]]] = None
    ) -> ModelFingerprint:
        """创建模型指纹"""
        import time
        
        hash_values = {}
        features = {}
        
        if weights:
            hash_values["weights"] = self.compute_weight_hash(weights)
            features["weight_count"] = len(weights)
            features["total_params"] = sum(
                len(w) if hasattr(w, '__len__') else 1
                for w in weights.values()
            )
        
        if architecture:
            hash_values["architecture"] = self.compute_architecture_hash(architecture)
            features["layer_count"] = len(architecture.get("layers", []))
        
        if outputs and inputs:
            hash_values["output_pattern"] = self.compute_output_pattern_hash(outputs, inputs)
            features["output_samples"] = len(outputs)
        
        # 组合哈希
        combined = hashlib.sha256(
            json.dumps(hash_values, sort_keys=True).encode()
        ).hexdigest()
        hash_values["combined"] = combined
        
        fingerprint_id = combined[:16]
        
        fingerprint = ModelFingerprint(
            fingerprint_id=fingerprint_id,
            fingerprint_type=FingerprintType.COMBINED,
            model_name=model_name,
            hash_values=hash_values,
            features=features,
            created_at=time.time()
        )
        
        self._fingerprints[fingerprint_id] = fingerprint
        return fingerprint
    
    def get_fingerprint(self, fingerprint_id: str) -> Optional[ModelFingerprint]:
        """获取指纹"""
        return self._fingerprints.get(fingerprint_id)
    
    def get_all_fingerprints(self) -> List[ModelFingerprint]:
        """获取所有指纹"""
        return list(self._fingerprints.values())
    
    def compare_fingerprints(
        self,
        fp1: ModelFingerprint,
        fp2: ModelFingerprint
    ) -> Dict[str, float]:
        """比较两个指纹"""
        similarities = {}
        
        for key in fp1.hash_values:
            if key in fp2.hash_values:
                # 计算汉明距离相似度
                h1 = fp1.hash_values[key]
                h2 = fp2.hash_values[key]
                
                if len(h1) == len(h2):
                    diff = sum(c1 != c2 for c1, c2 in zip(h1, h2))
                    similarity = 1 - diff / len(h1)
                    similarities[key] = similarity
        
        # 计算综合相似度
        if similarities:
            similarities["overall"] = sum(similarities.values()) / len(similarities)
        
        return similarities
    
    def identify_model(
        self,
        fingerprint: ModelFingerprint,
        threshold: float = 0.9
    ) -> List[Tuple[ModelFingerprint, float]]:
        """识别模型"""
        matches = []
        
        for stored_fp in self._fingerprints.values():
            if stored_fp.fingerprint_id == fingerprint.fingerprint_id:
                continue
            
            similarities = self.compare_fingerprints(fingerprint, stored_fp)
            overall = similarities.get("overall", 0)
            
            if overall >= threshold:
                matches.append((stored_fp, overall))
        
        # 按相似度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def export_fingerprint(self, fingerprint: ModelFingerprint) -> str:
        """导出指纹"""
        return json.dumps({
            "fingerprint_id": fingerprint.fingerprint_id,
            "fingerprint_type": fingerprint.fingerprint_type.value,
            "model_name": fingerprint.model_name,
            "hash_values": fingerprint.hash_values,
            "features": fingerprint.features,
            "created_at": fingerprint.created_at,
            "metadata": fingerprint.metadata
        }, indent=2)
    
    def import_fingerprint(self, data: str) -> ModelFingerprint:
        """导入指纹"""
        obj = json.loads(data)
        
        return ModelFingerprint(
            fingerprint_id=obj["fingerprint_id"],
            fingerprint_type=FingerprintType(obj["fingerprint_type"]),
            model_name=obj["model_name"],
            hash_values=obj["hash_values"],
            features=obj.get("features", {}),
            created_at=obj.get("created_at", 0.0),
            metadata=obj.get("metadata", {})
        )
