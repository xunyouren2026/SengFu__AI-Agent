"""
防御规则生成 - 自动生成安全防御规则
"""
import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class RuleType(Enum):
    """规则类型"""
    SIGNATURE = "signature"
    PATTERN = "pattern"
    BEHAVIORAL = "behavioral"
    ANOMALY = "anomaly"
    HEURISTIC = "heuristic"


class RuleSeverity(Enum):
    """规则严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class GeneratedRule:
    """生成的规则"""
    rule_id: str
    rule_type: RuleType
    name: str
    description: str
    pattern: str
    severity: RuleSeverity
    tags: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleGenerator:
    """规则生成器"""
    
    def __init__(self):
        self._generated_rules: Dict[str, GeneratedRule] = {}
        self._pattern_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """加载规则模板"""
        return {
            "sql_injection": r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table|or\s+1\s*=\s*1|'\s*or\s*'|--\s*$)",
            "xss": r"(?i)(<script|javascript:|on\w+\s*=|<iframe|<object|<embed|<img\s+[^>]*onerror)",
            "command_injection": r"(?i)(\||;|&&|\|\||\$?\(|`[^`]+`|eval\s*\(|exec\s*\()",
            "path_traversal": r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|/etc/|/proc/|/passwd|/shadow)",
            "ldap_injection": r"(?i)(\*\(\)\(|\|\(|\)\(|\(\|=",
            "xml_injection": r"(?i)(<!ENTITY|<!DOCTYPE.*\[|<\?xml)",
            "ssrf": r"(?i)(file://|gopher://|dict://|ldap://|php://filter)",
            "deserialization": r"(?i)(pickle\.loads|yaml\.load|unserialize|ObjectInputStream)",
        }
    
    def generate_from_patterns(
        self,
        attack_patterns: List[str],
        attack_type: str,
        confidence: float = 0.8
    ) -> List[GeneratedRule]:
        """从攻击模式生成规则"""
        rules = []
        
        for pattern in attack_patterns:
            # 规范化模式
            normalized = self._normalize_pattern(pattern)
            
            if not normalized:
                continue
            
            # 生成规则ID
            rule_id = self._generate_rule_id(normalized, attack_type)
            
            # 确定严重程度
            severity = self._determine_severity(attack_type, confidence)
            
            rule = GeneratedRule(
                rule_id=rule_id,
                rule_type=RuleType.PATTERN,
                name=f"Auto-generated {attack_type} rule",
                description=f"Automatically generated rule for {attack_type} attack pattern",
                pattern=normalized,
                severity=severity,
                tags=[attack_type, "auto-generated"],
                confidence=confidence,
                metadata={"original_pattern": pattern}
            )
            
            rules.append(rule)
            self._generated_rules[rule_id] = rule
        
        return rules
    
    def generate_from_samples(
        self,
        samples: List[Dict[str, Any]],
        attack_type: str
    ) -> List[GeneratedRule]:
        """从样本生成规则"""
        rules = []
        
        # 提取共同特征
        common_patterns = self._extract_common_patterns(samples)
        
        for pattern_data in common_patterns:
            pattern = pattern_data.get("pattern", "")
            frequency = pattern_data.get("frequency", 0)
            
            if frequency < 3:  # 忽略低频模式
                continue
            
            rule_id = self._generate_rule_id(pattern, attack_type)
            confidence = min(1.0, frequency / 10)
            
            rule = GeneratedRule(
                rule_id=rule_id,
                rule_type=RuleType.SIGNATURE,
                name=f"Learned {attack_type} signature",
                description=f"Learned from {frequency} attack samples",
                pattern=pattern,
                severity=self._determine_severity(attack_type, confidence),
                tags=[attack_type, "learned"],
                confidence=confidence,
                metadata={"sample_count": frequency}
            )
            
            rules.append(rule)
            self._generated_rules[rule_id] = rule
        
        return rules
    
    def generate_behavioral_rule(
        self,
        behavior_sequence: List[str],
        attack_type: str,
        time_window: int = 60
    ) -> GeneratedRule:
        """生成行为规则"""
        # 构建行为模式
        pattern = " -> ".join(behavior_sequence)
        rule_id = self._generate_rule_id(pattern, "behavioral")
        
        return GeneratedRule(
            rule_id=rule_id,
            rule_type=RuleType.BEHAVIORAL,
            name=f"Behavioral rule for {attack_type}",
            description=f"Detects {attack_type} behavior pattern",
            pattern=json.dumps({
                "sequence": behavior_sequence,
                "time_window": time_window
            }),
            severity=RuleSeverity.HIGH,
            tags=[attack_type, "behavioral"],
            confidence=0.7,
            metadata={"time_window": time_window}
        )
    
    def generate_anomaly_rule(
        self,
        baseline: Dict[str, Any],
        threshold: float = 3.0
    ) -> GeneratedRule:
        """生成异常检测规则"""
        rule_id = self._generate_rule_id(json.dumps(baseline), "anomaly")
        
        return GeneratedRule(
            rule_id=rule_id,
            rule_type=RuleType.ANOMALY,
            name="Anomaly detection rule",
            description=f"Detects anomalies with threshold {threshold} std deviations",
            pattern=json.dumps({
                "baseline": baseline,
                "threshold": threshold
            }),
            severity=RuleSeverity.MEDIUM,
            tags=["anomaly"],
            confidence=0.6,
            metadata={"threshold": threshold}
        )
    
    def _normalize_pattern(self, pattern: str) -> str:
        """规范化模式"""
        # 移除多余空白
        pattern = ' '.join(pattern.split())
        
        # 转义特殊正则字符（如果不是正则）
        if not pattern.startswith('(') and not pattern.startswith('['):
            # 检查是否已经是正则
            regex_chars = {'*', '+', '?', '|', '(', ')', '[', ']', '{', '}'}
            if not any(c in pattern for c in regex_chars):
                pattern = re.escape(pattern)
        
        return pattern
    
    def _extract_common_patterns(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取共同模式"""
        pattern_counts: Dict[str, int] = {}
        
        for sample in samples:
            payload = sample.get("payload", "")
            
            # 提取n-gram
            ngrams = self._extract_ngrams(payload, n=4)
            for ngram in ngrams:
                pattern_counts[ngram] = pattern_counts.get(ngram, 0) + 1
        
        # 排序并返回高频模式
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"pattern": p, "frequency": f}
            for p, f in sorted_patterns[:20]  # 返回前20个
        ]
    
    def _extract_ngrams(self, text: str, n: int = 4) -> List[str]:
        """提取n-gram"""
        ngrams = []
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            if ngram.isprintable() and not ngram.isspace():
                ngrams.append(ngram)
        return ngrams
    
    def _generate_rule_id(self, pattern: str, prefix: str) -> str:
        """生成规则ID"""
        hash_val = hashlib.md5(f"{prefix}:{pattern}".encode()).hexdigest()[:8]
        return f"{prefix}_{hash_val}"
    
    def _determine_severity(self, attack_type: str, confidence: float) -> RuleSeverity:
        """确定严重程度"""
        critical_types = {"sql_injection", "command_injection", "rce"}
        high_types = {"xss", "path_traversal", "ssrf", "deserialization"}
        
        if attack_type in critical_types:
            return RuleSeverity.CRITICAL if confidence > 0.8 else RuleSeverity.HIGH
        elif attack_type in high_types:
            return RuleSeverity.HIGH if confidence > 0.7 else RuleSeverity.MEDIUM
        else:
            return RuleSeverity.MEDIUM if confidence > 0.6 else RuleSeverity.LOW
    
    def to_yara_format(self, rule: GeneratedRule) -> str:
        """转换为YARA格式"""
        lines = [
            f"rule {rule.rule_id} {{",
            "    meta:",
            f'        description = "{rule.description}"',
            f'        severity = "{rule.severity.value}"',
            f'        confidence = "{rule.confidence}"',
        ]
        
        for tag in rule.tags:
            lines.append(f'        tag = "{tag}"')
        
        lines.extend([
            "    strings:",
            f'        $pattern = "{rule.pattern}" nocase',
            "    condition:",
            "        any of them",
            "}"
        ])
        
        return "\n".join(lines)
    
    def to_snort_format(self, rule: GeneratedRule) -> str:
        """转换为Snort格式"""
        sid = hash(rule.rule_id) % 1000000
        return (
            f'alert tcp any any -> any any '
            f'(msg:"{rule.name}"; '
            f'content:"{rule.pattern}"; '
            f'nocase; '
            f'sid:{sid}; '
            f'rev:1;)'
        )
    
    def to_modsecurity_format(self, rule: GeneratedRule) -> str:
        """转换为ModSecurity格式"""
        return (
            f'SecRule REQUEST_BODY '
            f'"@contains {rule.pattern}" '
            f'"id:{hash(rule.rule_id) % 1000000},'
            f'phase:2,'
            f'deny,'
            f'status:403,'
            f'msg:\'{rule.name}\'"'
        )
    
    def get_rule(self, rule_id: str) -> Optional[GeneratedRule]:
        """获取规则"""
        return self._generated_rules.get(rule_id)
    
    def get_all_rules(self) -> List[GeneratedRule]:
        """获取所有规则"""
        return list(self._generated_rules.values())
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self._generated_rules:
            del self._generated_rules[rule_id]
            return True
        return False
    
    def export_rules(self) -> str:
        """导出规则"""
        return json.dumps([
            {
                "rule_id": r.rule_id,
                "rule_type": r.rule_type.value,
                "name": r.name,
                "pattern": r.pattern,
                "severity": r.severity.value,
                "tags": r.tags,
                "confidence": r.confidence
            }
            for r in self._generated_rules.values()
        ], indent=2)
