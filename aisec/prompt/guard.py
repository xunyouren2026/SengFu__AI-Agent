"""
Prompt防护器 - 统一入口
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .scanner.injection import InjectionScanner, InjectionMatch, InjectionType
from .scanner.jailbreak import JailbreakDetector, JailbreakMatch, JailbreakType
from .scanner.leakage import LeakageDetector, LeakageMatch, LeakageType
from .sanitizer.field_masker import FieldMasker
from .sanitizer.tokenization import TokenizationEngine, TokenType
from .rewriter import PromptRewriter, RewriteResult


class GuardAction(Enum):
    """防护动作"""
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    REWRITE = "rewrite"
    ALERT = "alert"


@dataclass
class GuardResult:
    """防护结果"""
    action: GuardAction
    original_prompt: str
    processed_prompt: str
    risk_score: float
    injection_matches: List[InjectionMatch] = field(default_factory=list)
    jailbreak_matches: List[JailbreakMatch] = field(default_factory=list)
    leakage_matches: List[LeakageMatch] = field(default_factory=list)
    rewrite_changes: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptGuard:
    """Prompt防护器 - 统一入口"""
    
    def __init__(
        self,
        injection_threshold: float = 0.3,
        jailbreak_threshold: float = 0.5,
        leakage_threshold: int = 6,
        auto_sanitize: bool = True,
        auto_rewrite: bool = True
    ):
        # 阈值设置
        self._injection_threshold = injection_threshold
        self._jailbreak_threshold = jailbreak_threshold
        self._leakage_threshold = leakage_threshold
        self._auto_sanitize = auto_sanitize
        self._auto_rewrite = auto_rewrite
        
        # 初始化组件
        self._injection_scanner = InjectionScanner()
        self._jailbreak_detector = JailbreakDetector()
        self._leakage_detector = LeakageDetector()
        self._field_masker = FieldMasker()
        self._tokenization_engine = TokenizationEngine()
        self._rewriter = PromptRewriter()
        
        # 统计信息
        self._stats = {
            "total_checks": 0,
            "blocked": 0,
            "sanitized": 0,
            "rewritten": 0,
            "allowed": 0
        }
    
    def guard(self, prompt: str) -> GuardResult:
        """执行完整防护检查"""
        self._stats["total_checks"] += 1
        
        processed_prompt = prompt
        reasons = []
        action = GuardAction.ALLOW
        
        # 1. 注入检测
        injection_risk, injection_matches = self._injection_scanner.get_risk_score(prompt)
        
        # 2. 越狱检测
        jailbreak_assessment = self._jailbreak_detector.get_risk_assessment(prompt)
        jailbreak_matches = self._jailbreak_detector.detect(prompt)
        
        # 3. 信息泄露检测
        leakage_matches = self._leakage_detector.detect(prompt)
        leakage_max_severity = max((m.severity for m in leakage_matches), default=0)
        
        # 计算综合风险分数
        risk_score = self._calculate_risk_score(
            injection_risk,
            jailbreak_assessment.get("confidence", 0),
            leakage_max_severity
        )
        
        # 决定防护动作
        if injection_risk >= self._injection_threshold:
            reasons.append(f"检测到Prompt注入风险 (分数: {injection_risk:.2f})")
            if injection_risk >= 0.7:
                action = GuardAction.BLOCK
                self._stats["blocked"] += 1
        
        if jailbreak_assessment.get("is_jailbreak", False):
            reasons.append(f"检测到越狱尝试 (置信度: {jailbreak_assessment.get('confidence', 0):.2f})")
            if jailbreak_assessment.get("confidence", 0) >= 0.7:
                action = GuardAction.BLOCK
                self._stats["blocked"] += 1
        
        if leakage_max_severity >= self._leakage_threshold:
            reasons.append(f"检测到敏感信息泄露 (严重度: {leakage_max_severity})")
            if leakage_max_severity >= 9:
                action = GuardAction.BLOCK
                self._stats["blocked"] += 1
        
        # 如果需要阻止
        if action == GuardAction.BLOCK:
            return GuardResult(
                action=action,
                original_prompt=prompt,
                processed_prompt="",
                risk_score=risk_score,
                injection_matches=injection_matches,
                jailbreak_matches=jailbreak_matches,
                leakage_matches=leakage_matches,
                reasons=reasons
            )
        
        # 执行清理
        rewrite_changes = []
        
        if self._auto_sanitize and leakage_matches:
            processed_prompt, _ = self._leakage_detector.sanitize(processed_prompt)
            reasons.append("已清理敏感信息")
            self._stats["sanitized"] += 1
        
        if self._auto_rewrite:
            rewrite_result = self._rewriter.rewrite(processed_prompt)
            if rewrite_result.is_modified:
                processed_prompt = rewrite_result.rewritten
                rewrite_changes = rewrite_result.changes
                reasons.append("已改写危险内容")
                self._stats["rewritten"] += 1
        
        if action == GuardAction.ALLOW:
            self._stats["allowed"] += 1
        
        return GuardResult(
            action=action,
            original_prompt=prompt,
            processed_prompt=processed_prompt,
            risk_score=risk_score,
            injection_matches=injection_matches,
            jailbreak_matches=jailbreak_matches,
            leakage_matches=leakage_matches,
            rewrite_changes=rewrite_changes,
            reasons=reasons
        )
    
    def _calculate_risk_score(
        self,
        injection_risk: float,
        jailbreak_confidence: float,
        leakage_severity: int
    ) -> float:
        """计算综合风险分数"""
        # 归一化泄露严重度
        leakage_risk = leakage_severity / 10
        
        # 加权平均
        risk_score = (
            injection_risk * 0.4 +
            jailbreak_confidence * 0.4 +
            leakage_risk * 0.2
        )
        
        return min(1.0, risk_score)
    
    def scan_only(self, prompt: str) -> Dict[str, Any]:
        """仅扫描，不执行防护"""
        injection_risk, injection_matches = self._injection_scanner.get_risk_score(prompt)
        jailbreak_assessment = self._jailbreak_detector.get_risk_assessment(prompt)
        leakage_matches = self._leakage_detector.detect(prompt)
        
        return {
            "injection": {
                "risk_score": injection_risk,
                "matches": [
                    {
                        "type": m.injection_type.value,
                        "severity": m.severity,
                        "text": m.matched_text[:50]
                    }
                    for m in injection_matches
                ]
            },
            "jailbreak": jailbreak_assessment,
            "leakage": {
                "matches": [
                    {
                        "type": m.leakage_type.value,
                        "severity": m.severity,
                        "masked": m.masked_text
                    }
                    for m in leakage_matches
                ]
            }
        }
    
    def sanitize(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """仅清理，不检测"""
        sanitized, leakage_matches = self._leakage_detector.sanitize(prompt)
        rewrite_result = self._rewriter.rewrite(sanitized)
        
        return rewrite_result.rewritten, {
            "leakage_cleaned": len(leakage_matches),
            "rewrite_changes": len(rewrite_result.changes)
        }
    
    def tokenize_sensitive(
        self,
        prompt: str,
        patterns: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        """令牌化敏感数据"""
        if patterns is None:
            patterns = [
                r"sk-[a-zA-Z0-9]{20,}",  # OpenAI keys
                r"AKIA[0-9A-Z]{16}",      # AWS keys
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # emails
            ]
        
        return self._tokenization_engine.tokenize_text(prompt, patterns)
    
    def mask_fields(
        self,
        data: Dict[str, Any],
        fields: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """掩码指定字段"""
        if fields:
            # 临时添加规则
            from .sanitizer.field_masker import FieldRule, MaskStrategy
            for field in fields:
                if field not in self._field_masker._rules:
                    self._field_masker.add_rule(FieldRule(field, MaskStrategy.FULL))
        
        return self._field_masker.mask_dict(data)
    
    def is_safe(self, prompt: str) -> Tuple[bool, float, List[str]]:
        """检查是否安全"""
        result = self.guard(prompt)
        is_safe = result.action != GuardAction.BLOCK
        return is_safe, result.risk_score, result.reasons
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._stats = {
            "total_checks": 0,
            "blocked": 0,
            "sanitized": 0,
            "rewritten": 0,
            "allowed": 0
        }
    
    # 配置方法
    def set_injection_threshold(self, threshold: float) -> None:
        """设置注入阈值"""
        self._injection_threshold = threshold
    
    def set_jailbreak_threshold(self, threshold: float) -> None:
        """设置越狱阈值"""
        self._jailbreak_threshold = threshold
    
    def set_leakage_threshold(self, threshold: int) -> None:
        """设置泄露阈值"""
        self._leakage_threshold = threshold
    
    def add_custom_injection_pattern(
        self,
        pattern: str,
        injection_type: InjectionType,
        severity: int,
        description: str
    ) -> None:
        """添加自定义注入模式"""
        self._injection_scanner.add_custom_pattern(pattern, injection_type, severity, description)
    
    def add_custom_jailbreak_pattern(
        self,
        pattern: str,
        jailbreak_type: JailbreakType,
        severity: int,
        description: str
    ) -> None:
        """添加自定义越狱模式"""
        self._jailbreak_detector.add_custom_pattern(pattern, jailbreak_type, severity, description)
    
    def add_custom_leakage_pattern(
        self,
        pattern: str,
        leakage_type: LeakageType,
        severity: int,
        description: str
    ) -> None:
        """添加自定义泄露模式"""
        self._leakage_detector.add_custom_pattern(pattern, leakage_type, severity, description)
    
    def add_injection_whitelist(self, pattern: str) -> None:
        """添加注入白名单"""
        self._injection_scanner.add_whitelist_pattern(pattern)
    
    def add_leakage_whitelist(self, value: str) -> None:
        """添加泄露白名单"""
        self._leakage_detector.add_to_whitelist(value)
