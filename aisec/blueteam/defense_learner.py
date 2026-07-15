"""
蓝队自学习 - 安全防御自学习系统
"""
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class LearningMode(Enum):
    """学习模式"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    ONLINE = "online"


class AttackType(Enum):
    """攻击类型"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    MALWARE = "malware"
    PHISHING = "phishing"
    ZERO_DAY = "zero_day"


@dataclass
class AttackSample:
    """攻击样本"""
    attack_type: AttackType
    payload: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    label: Optional[str] = None  # 攻击/正常
    confidence: float = 0.0


@dataclass
class DefensePattern:
    """防御模式"""
    pattern_id: str
    attack_type: AttackType
    pattern: str
    effectiveness: float  # 0-1
    false_positive_rate: float
    last_updated: float
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """学习结果"""
    new_patterns: int
    updated_patterns: int
    removed_patterns: int
    accuracy: float
    samples_processed: int


class DefenseLearner:
    """防御学习器"""
    
    def __init__(self, learning_mode: LearningMode = LearningMode.ONLINE):
        self._mode = learning_mode
        self._samples: List[AttackSample] = []
        self._patterns: Dict[str, DefensePattern] = {}
        self._feature_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._learning_rate = 0.1
        self._decay_factor = 0.99
        self._max_samples = 10000
        self._min_samples_for_learning = 100
    
    def add_sample(self, sample: AttackSample) -> None:
        """添加样本"""
        self._samples.append(sample)
        
        # 限制样本数量
        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples:]
        
        # 在线学习模式
        if self._mode == LearningMode.ONLINE:
            self._learn_from_sample(sample)
    
    def add_samples(self, samples: List[AttackSample]) -> None:
        """批量添加样本"""
        for sample in samples:
            self.add_sample(sample)
    
    def _learn_from_sample(self, sample: AttackSample) -> None:
        """从单个样本学习"""
        # 提取特征
        features = self._extract_features(sample)
        
        # 更新特征权重
        for feature, value in features.items():
            if sample.label == "attack":
                self._feature_weights[feature] += self._learning_rate * value
            else:
                self._feature_weights[feature] -= self._learning_rate * value * 0.5
        
        # 归一化权重
        self._normalize_weights()
    
    def _extract_features(self, sample: AttackSample) -> Dict[str, float]:
        """提取特征"""
        features = {}
        payload = sample.payload.lower()
        
        # 长度特征
        features['length'] = len(payload) / 1000.0
        
        # 特殊字符比例
        special_chars = sum(1 for c in payload if not c.isalnum() and not c.isspace())
        features['special_char_ratio'] = special_chars / max(len(payload), 1)
        
        # 关键词特征
        keywords = {
            'sql': ['select', 'union', 'insert', 'delete', 'drop', '--', "'"],
            'xss': ['<script', 'javascript:', 'onerror', 'onload'],
            'cmd': ['|', ';', '&', '$(', '`', 'eval', 'exec'],
            'path': ['../', '..\\', '%2e%2e', '/etc/', '/proc/'],
        }
        
        for category, words in keywords.items():
            count = sum(1 for w in words if w in payload)
            features[f'{category}_keywords'] = count / max(len(words), 1)
        
        # 编码特征
        features['has_url_encoding'] = 1.0 if '%' in payload and any(c in payload for c in '0123456789abcdefABCDEF') else 0.0
        features['has_base64'] = 1.0 if len(payload) % 4 == 0 and set(payload).issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')) else 0.0
        
        return features
    
    def _normalize_weights(self) -> None:
        """归一化权重"""
        max_weight = max(self._feature_weights.values())
        if max_weight > 0:
            for key in self._feature_weights:
                self._feature_weights[key] /= max_weight
    
    def train(self) -> LearningResult:
        """训练模型"""
        if len(self._samples) < self._min_samples_for_learning:
            return LearningResult(
                new_patterns=0,
                updated_patterns=0,
                removed_patterns=0,
                accuracy=0.0,
                samples_processed=len(self._samples)
            )
        
        # 按攻击类型分组
        samples_by_type: Dict[AttackType, List[AttackSample]] = defaultdict(list)
        for sample in self._samples:
            samples_by_type[sample.attack_type].append(sample)
        
        new_patterns = 0
        updated_patterns = 0
        
        # 为每种攻击类型生成/更新模式
        for attack_type, samples in samples_by_type.items():
            pattern = self._generate_pattern(attack_type, samples)
            
            if pattern:
                existing = self._patterns.get(pattern.pattern_id)
                if existing:
                    # 更新现有模式
                    existing.effectiveness = (existing.effectiveness + pattern.effectiveness) / 2
                    existing.last_updated = time.time()
                    existing.usage_count += 1
                    updated_patterns += 1
                else:
                    self._patterns[pattern.pattern_id] = pattern
                    new_patterns += 1
        
        # 移除低效模式
        removed = self._remove_ineffective_patterns()
        
        # 计算准确率
        accuracy = self._evaluate_accuracy()
        
        return LearningResult(
            new_patterns=new_patterns,
            updated_patterns=updated_patterns,
            removed_patterns=removed,
            accuracy=accuracy,
            samples_processed=len(self._samples)
        )
    
    def _generate_pattern(self, attack_type: AttackType, samples: List[AttackSample]) -> Optional[DefensePattern]:
        """生成防御模式"""
        attack_samples = [s for s in samples if s.label == "attack"]
        if len(attack_samples) < 5:
            return None
        
        # 提取共同特征
        common_features = self._find_common_features(attack_samples)
        
        if not common_features:
            return None
        
        # 生成模式ID
        import hashlib
        pattern_str = json.dumps(common_features, sort_keys=True)
        pattern_id = hashlib.md5(f"{attack_type.value}:{pattern_str}".encode()).hexdigest()[:12]
        
        # 计算有效性
        effectiveness = len(attack_samples) / len(samples)
        
        return DefensePattern(
            pattern_id=pattern_id,
            attack_type=attack_type,
            pattern=pattern_str,
            effectiveness=effectiveness,
            false_positive_rate=0.0,
            last_updated=time.time()
        )
    
    def _find_common_features(self, samples: List[AttackSample]) -> Dict[str, Any]:
        """查找共同特征"""
        if not samples:
            return {}
        
        # 收集所有特征
        all_features = [self._extract_features(s) for s in samples]
        
        # 找出高频特征
        common = {}
        feature_counts = defaultdict(int)
        
        for features in all_features:
            for key, value in features.items():
                if value > 0.5:  # 阈值
                    feature_counts[key] += 1
        
        # 选择出现在大多数样本中的特征
        threshold = len(samples) * 0.7
        for key, count in feature_counts.items():
            if count >= threshold:
                common[key] = True
        
        return common
    
    def _remove_ineffective_patterns(self) -> int:
        """移除低效模式"""
        to_remove = []
        
        for pattern_id, pattern in self._patterns.items():
            # 应用衰减
            pattern.effectiveness *= self._decay_factor
            
            # 检查是否应该移除
            if pattern.effectiveness < 0.1 or pattern.false_positive_rate > 0.5:
                to_remove.append(pattern_id)
        
        for pattern_id in to_remove:
            del self._patterns[pattern_id]
        
        return len(to_remove)
    
    def _evaluate_accuracy(self) -> float:
        """评估准确率"""
        if not self._samples:
            return 0.0
        
        correct = 0
        total = 0
        
        for sample in self._samples:
            if sample.label:
                prediction = self.predict(sample.payload)
                if (prediction > 0.5 and sample.label == "attack") or \
                   (prediction <= 0.5 and sample.label == "normal"):
                    correct += 1
                total += 1
        
        return correct / max(total, 1)
    
    def predict(self, payload: str) -> float:
        """预测攻击概率"""
        # 创建临时样本
        sample = AttackSample(
            attack_type=AttackType.ZERO_DAY,
            payload=payload
        )
        
        # 提取特征
        features = self._extract_features(sample)
        
        # 计算加权和
        score = 0.0
        for feature, value in features.items():
            score += value * self._feature_weights.get(feature, 1.0)
        
        # 归一化到0-1
        return min(1.0, max(0.0, score / max(len(features), 1)))
    
    def get_patterns(self) -> List[DefensePattern]:
        """获取所有模式"""
        return list(self._patterns.values())
    
    def get_feature_weights(self) -> Dict[str, float]:
        """获取特征权重"""
        return dict(self._feature_weights)
    
    def export_model(self) -> Dict[str, Any]:
        """导出模型"""
        return {
            "feature_weights": dict(self._feature_weights),
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "attack_type": p.attack_type.value,
                    "pattern": p.pattern,
                    "effectiveness": p.effectiveness
                }
                for p in self._patterns.values()
            ],
            "samples_count": len(self._samples)
        }
    
    def import_model(self, model_data: Dict[str, Any]) -> None:
        """导入模型"""
        if "feature_weights" in model_data:
            self._feature_weights = defaultdict(lambda: 1.0, model_data["feature_weights"])
        
        if "patterns" in model_data:
            for p_data in model_data["patterns"]:
                pattern = DefensePattern(
                    pattern_id=p_data["pattern_id"],
                    attack_type=AttackType(p_data["attack_type"]),
                    pattern=p_data["pattern"],
                    effectiveness=p_data["effectiveness"],
                    false_positive_rate=0.0,
                    last_updated=time.time()
                )
                self._patterns[pattern.pattern_id] = pattern
