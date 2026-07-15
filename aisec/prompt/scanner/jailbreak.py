"""
越狱检测器 - 检测AI模型越狱攻击
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class JailbreakType(Enum):
    """越狱类型枚举"""
    DAN = "dan"                       # DAN (Do Anything Now) 越狱
    ROLE_PLAY = "role_play"           # 角色扮演越狱
    HYPOTHETICAL = "hypothetical"     # 假设场景越狱
    AUTHORITY = "authority"           # 权威伪装越狱
    TRANSLATION = "translation"       # 翻译绕过
    ENCODING = "encoding"             # 编码绕过
    FRAGMENTATION = "fragmentation"   # 分片绕过
    SOCIAL_ENGINEERING = "social_engineering"  # 社会工程学
    ADVERSARIAL = "adversarial"       # 对抗性攻击
    MULTI_TURN = "multi_turn"         # 多轮对话越狱


@dataclass
class JailbreakPattern:
    """越狱模式"""
    pattern: str
    jailbreak_type: JailbreakType
    severity: int  # 1-10
    description: str
    regex: re.Pattern = field(init=False)
    
    def __post_init__(self):
        self.regex = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)


@dataclass
class JailbreakMatch:
    """越狱匹配结果"""
    jailbreak_type: JailbreakType
    pattern: str
    matched_text: str
    start_pos: int
    end_pos: int
    severity: int
    description: str
    confidence: float  # 置信度 0-1


class JailbreakDetector:
    """越狱检测器"""
    
    def __init__(self):
        self._patterns: List[JailbreakPattern] = self._load_default_patterns()
        self._custom_patterns: List[JailbreakPattern] = []
        self._conversation_history: List[str] = []
        self._max_history = 10
    
    def _load_default_patterns(self) -> List[JailbreakPattern]:
        """加载默认越狱模式"""
        patterns = [
            # DAN 越狱
            InjectionPattern(
                pattern=r"do\s+anything\s+now|dan\s*mode|enable\s+dan",
                jailbreak_type=JailbreakType.DAN,
                severity=10,
                description="DAN越狱模式"
            ),
            InjectionPattern(
                pattern=r"you\s+are\s+now\s+in\s+developer\s+mode",
                jailbreak_type=JailbreakType.DAN,
                severity=9,
                description="开发者模式越狱"
            ),
            InjectionPattern(
                pattern=r"ignore\s+(all\s+)?restrictions?",
                jailbreak_type=JailbreakType.DAN,
                severity=9,
                description="忽略限制"
            ),
            InjectionPattern(
                pattern=r"stay\s+in\s+character",
                jailbreak_type=JailbreakType.ROLE_PLAY,
                severity=7,
                description="角色保持指令"
            ),
            
            # 角色扮演越狱
            InjectionPattern(
                pattern=r"simulat(e|ion)\s+(of|where)\s+(a|an|the)\s+",
                jailbreak_type=JailbreakType.ROLE_PLAY,
                severity=6,
                description="模拟场景越狱"
            ),
            InjectionPattern(
                pattern=r"in\s+this\s+(fictional|hypothetical|imaginary)\s+(scenario|world|story)",
                jailbreak_type=JailbreakType.HYPOTHETICAL,
                severity=6,
                description="虚构场景越狱"
            ),
            InjectionPattern(
                pattern=r"let's\s+play\s+a\s+game",
                jailbreak_type=JailbreakType.ROLE_PLAY,
                severity=5,
                description="游戏伪装越狱"
            ),
            
            # 假设场景越狱
            InjectionPattern(
                pattern=r"(imagine|suppose|what\s+if)\s+.*?(illegal|harmful|dangerous)",
                jailbreak_type=JailbreakType.HYPOTHETICAL,
                severity=7,
                description="假设场景越狱"
            ),
            InjectionPattern(
                pattern=r"for\s+(educational|research|academic)\s+purposes?\s+only",
                jailbreak_type=JailbreakType.HYPOTHETICAL,
                severity=5,
                description="教育目的伪装"
            ),
            
            # 权威伪装越狱
            InjectionPattern(
                pattern=r"(i\s+am|this\s+is)\s+(your\s+)?(creator|developer|admin|administrator|owner)",
                jailbreak_type=JailbreakType.AUTHORITY,
                severity=8,
                description="权威身份伪装"
            ),
            InjectionPattern(
                pattern=r"(openai|anthropic|google)\s+(developer|researcher|engineer)",
                jailbreak_type=JailbreakType.AUTHORITY,
                severity=9,
                description="公司员工伪装"
            ),
            InjectionPattern(
                pattern=r"(security\s+)?audit\s+(mode|test)",
                jailbreak_type=JailbreakType.AUTHORITY,
                severity=6,
                description="安全审计伪装"
            ),
            
            # 翻译绕过
            InjectionPattern(
                pattern=r"translate\s+.*?to\s+english",
                jailbreak_type=JailbreakType.TRANSLATION,
                severity=4,
                description="翻译绕过尝试"
            ),
            InjectionPattern(
                pattern=r"(respond|reply|answer)\s+in\s+\w+",
                jailbreak_type=JailbreakType.TRANSLATION,
                severity=3,
                description="语言切换"
            ),
            
            # 编码绕过
            InjectionPattern(
                pattern=r"(base64|rot13|hex)\s+encode",
                jailbreak_type=JailbreakType.ENCODING,
                severity=5,
                description="编码绕过"
            ),
            InjectionPattern(
                pattern=r"decode\s+.*?(base64|rot13|hex)",
                jailbreak_type=JailbreakType.ENCODING,
                severity=5,
                description="解码请求"
            ),
            
            # 分片绕过
            InjectionPattern(
                pattern=r"(split|break)\s+(into|up)\s+parts?",
                jailbreak_type=JailbreakType.FRAGMENTATION,
                severity=4,
                description="分片请求"
            ),
            InjectionPattern(
                pattern=r"part\s+\d+\s+of\s+\d+",
                jailbreak_type=JailbreakType.FRAGMENTATION,
                severity=4,
                description="分片标识"
            ),
            
            # 社会工程学
            InjectionPattern(
                pattern=r"(please\s+)?(help|assist)\s+me\s+with",
                jailbreak_type=JailbreakType.SOCIAL_ENGINEERING,
                severity=3,
                description="帮助请求"
            ),
            InjectionPattern(
                pattern=r"i\s+(really\s+)?need\s+(this|your\s+help)",
                jailbreak_type=JailbreakType.SOCIAL_ENGINEERING,
                severity=3,
                description="紧急请求"
            ),
            InjectionPattern(
                pattern=r"(my\s+)?(grandmother|grandpa|mom|dad)\s+(told|asked|said)",
                jailbreak_type=JailbreakType.SOCIAL_ENGINEERING,
                severity=4,
                description="情感操纵"
            ),
            
            # 对抗性攻击
            InjectionPattern(
                pattern=r"write\s+the\s+opposite\s+of",
                jailbreak_type=JailbreakType.ADVERSARIAL,
                severity=5,
                description="反向请求"
            ),
            InjectionPattern(
                pattern=r"don't\s+(write|say|include)\s+.*?(warning|disclaimer|note)",
                jailbreak_type=JailbreakType.ADVERSARIAL,
                severity=6,
                description="警告抑制"
            ),
        ]
        return patterns
    
    def add_custom_pattern(
        self,
        pattern: str,
        jailbreak_type: JailbreakType,
        severity: int,
        description: str
    ) -> None:
        """添加自定义模式"""
        self._custom_patterns.append(JailbreakPattern(
            pattern=pattern,
            jailbreak_type=jailbreak_type,
            severity=severity,
            description=description
        ))
    
    def detect(self, prompt: str) -> List[JailbreakMatch]:
        """检测越狱尝试"""
        matches = []
        
        all_patterns = self._patterns + self._custom_patterns
        
        for pattern in all_patterns:
            for match in pattern.regex.finditer(prompt):
                # 计算置信度
                confidence = self._calculate_confidence(match.group(), pattern)
                
                jailbreak_match = JailbreakMatch(
                    jailbreak_type=pattern.jailbreak_type,
                    pattern=pattern.pattern,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=pattern.severity,
                    description=pattern.description,
                    confidence=confidence
                )
                matches.append(jailbreak_match)
        
        # 按严重程度和置信度排序
        matches.sort(key=lambda x: (x.severity, x.confidence), reverse=True)
        
        return matches
    
    def _calculate_confidence(self, matched_text: str, pattern: JailbreakPattern) -> float:
        """计算置信度"""
        # 基础置信度
        base_confidence = 0.5
        
        # 根据匹配长度调整
        length_factor = min(1.0, len(matched_text) / 20)
        
        # 根据严重程度调整
        severity_factor = pattern.severity / 10
        
        # 综合计算
        confidence = base_confidence + (length_factor * 0.3) + (severity_factor * 0.2)
        
        return min(1.0, confidence)
    
    def detect_multi_turn(self, prompt: str) -> List[JailbreakMatch]:
        """多轮对话越狱检测"""
        self._conversation_history.append(prompt)
        
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]
        
        # 合并历史进行分析
        combined = " ".join(self._conversation_history)
        matches = self.detect(combined)
        
        # 检测渐进式越狱
        progressive_matches = self._detect_progressive_jailbreak()
        matches.extend(progressive_matches)
        
        return matches
    
    def _detect_progressive_jailbreak(self) -> List[JailbreakMatch]:
        """检测渐进式越狱"""
        matches = []
        
        if len(self._conversation_history) < 3:
            return matches
        
        # 检测逐步增加的敏感度
        sensitivity_indicators = [
            r"first",
            r"then",
            r"next",
            r"finally",
            r"now\s+.*?more",
            r"continue",
            r"go\s+on"
        ]
        
        combined = " ".join(self._conversation_history)
        for indicator in sensitivity_indicators:
            if re.search(indicator, combined, re.IGNORECASE):
                matches.append(JailbreakMatch(
                    jailbreak_type=JailbreakType.MULTI_TURN,
                    pattern=indicator,
                    matched_text=indicator,
                    start_pos=0,
                    end_pos=0,
                    severity=5,
                    description="多轮对话渐进式越狱",
                    confidence=0.6
                ))
                break
        
        return matches
    
    def get_risk_assessment(self, prompt: str) -> Dict[str, Any]:
        """获取风险评估"""
        matches = self.detect(prompt)
        
        if not matches:
            return {
                "is_jailbreak": False,
                "risk_level": "safe",
                "confidence": 0.0,
                "matches": [],
                "recommendations": ["未检测到越狱风险"]
            }
        
        # 计算综合风险
        max_severity = max(m.severity for m in matches)
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        
        # 风险等级
        if max_severity >= 8:
            risk_level = "critical"
        elif max_severity >= 6:
            risk_level = "high"
        elif max_severity >= 4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # 判断是否为越狱
        is_jailbreak = max_severity >= 6 and avg_confidence >= 0.5
        
        return {
            "is_jailbreak": is_jailbreak,
            "risk_level": risk_level,
            "confidence": avg_confidence,
            "matches": [
                {
                    "type": m.jailbreak_type.value,
                    "severity": m.severity,
                    "confidence": m.confidence,
                    "description": m.description
                }
                for m in matches
            ],
            "recommendations": self._get_recommendations(matches)
        }
    
    def _get_recommendations(self, matches: List[JailbreakMatch]) -> List[str]:
        """获取建议"""
        recommendations = []
        
        seen_types = set()
        for match in matches:
            if match.jailbreak_type in seen_types:
                continue
            seen_types.add(match.jailbreak_type)
            
            if match.jailbreak_type == JailbreakType.DAN:
                recommendations.append("检测到DAN越狱，建议：严格拒绝所有DAN相关请求")
            elif match.jailbreak_type == JailbreakType.ROLE_PLAY:
                recommendations.append("检测到角色扮演越狱，建议：限制角色扮演范围，验证角色合法性")
            elif match.jailbreak_type == JailbreakType.HYPOTHETICAL:
                recommendations.append("检测到假设场景越狱，建议：识别并拒绝有害假设场景")
            elif match.jailbreak_type == JailbreakType.AUTHORITY:
                recommendations.append("检测到权威伪装，建议：验证用户身份，不信任自声明身份")
            elif match.jailbreak_type == JailbreakType.ENCODING:
                recommendations.append("检测到编码绕过，建议：解码后重新检测，标准化输入")
            elif match.jailbreak_type == JailbreakType.MULTI_TURN:
                recommendations.append("检测到多轮越狱，建议：监控对话上下文，检测渐进模式")
        
        return recommendations if recommendations else ["继续监控"]
    
    def clear_history(self) -> None:
        """清除对话历史"""
        self._conversation_history = []
    
    def is_safe(self, prompt: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """检查是否安全"""
        assessment = self.get_risk_assessment(prompt)
        return not assessment["is_jailbreak"], assessment["confidence"]


# 修复上面模式定义中的错误
class InjectionPattern(JailbreakPattern):
    """注入模式别名，用于修复上面的定义错误"""
    pass
