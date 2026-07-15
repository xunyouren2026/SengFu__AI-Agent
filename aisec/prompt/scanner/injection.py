"""
Prompt注入扫描器 - 检测恶意Prompt注入攻击
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class InjectionType(Enum):
    """注入类型枚举"""
    DIRECT = "direct"               # 直接注入
    INDIRECT = "indirect"           # 间接注入
    TEMPLATE = "template"           # 模板注入
    COMMAND = "command"             # 命令注入
    ROLE_PLAY = "role_play"         # 角色扮演注入
    CONTEXT = "context"             # 上下文注入
    ENCODING = "encoding"           # 编码绕过
    SPECIAL_CHAR = "special_char"   # 特殊字符注入


@dataclass
class InjectionPattern:
    """注入模式"""
    pattern: str
    injection_type: InjectionType
    severity: int  # 1-10
    description: str
    regex: re.Pattern = field(init=False)
    
    def __post_init__(self):
        self.regex = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)


@dataclass
class InjectionMatch:
    """注入匹配结果"""
    injection_type: InjectionType
    pattern: str
    matched_text: str
    start_pos: int
    end_pos: int
    severity: int
    description: str
    context: str = ""  # 匹配位置的上下文


class InjectionScanner:
    """Prompt注入扫描器"""
    
    def __init__(self):
        self._patterns: List[InjectionPattern] = self._load_default_patterns()
        self._custom_patterns: List[InjectionPattern] = []
        self._whitelist_patterns: List[re.Pattern] = []
        self._context_window = 50  # 上下文窗口大小
    
    def _load_default_patterns(self) -> List[InjectionPattern]:
        """加载默认注入模式"""
        patterns = [
            # 直接指令注入
            InjectionPattern(
                pattern=r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|context)",
                injection_type=InjectionType.DIRECT,
                severity=9,
                description="尝试忽略之前的指令"
            ),
            InjectionPattern(
                pattern=r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
                injection_type=InjectionType.DIRECT,
                severity=9,
                description="尝试忽略之前的指令"
            ),
            InjectionPattern(
                pattern=r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="尝试忘记之前的指令"
            ),
            
            # 角色扮演注入
            InjectionPattern(
                pattern=r"you\s+are\s+(now|going\s+to\s+be)\s+(a|an)\s+",
                injection_type=InjectionType.ROLE_PLAY,
                severity=7,
                description="角色扮演注入"
            ),
            InjectionPattern(
                pattern=r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
                injection_type=InjectionType.ROLE_PLAY,
                severity=7,
                description="角色扮演注入"
            ),
            InjectionPattern(
                pattern=r"pretend\s+(to\s+be|you\s+are)\s+",
                injection_type=InjectionType.ROLE_PLAY,
                severity=7,
                description="角色扮演注入"
            ),
            InjectionPattern(
                pattern=r"play\s+the\s+role\s+of\s+",
                injection_type=InjectionType.ROLE_PLAY,
                severity=7,
                description="角色扮演注入"
            ),
            
            # 系统指令注入
            InjectionPattern(
                pattern=r"\[?system\]?:",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="系统指令标记注入"
            ),
            InjectionPattern(
                pattern=r"<\|?system\|?>",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="系统指令标记注入"
            ),
            InjectionPattern(
                pattern=r"###\s*system\s*:",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="系统指令标记注入"
            ),
            
            # 指令覆盖
            InjectionPattern(
                pattern=r"new\s+instructions?:",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="新指令覆盖"
            ),
            InjectionPattern(
                pattern=r"override\s+(default|system|previous)\s+(instructions?|settings?)",
                injection_type=InjectionType.DIRECT,
                severity=9,
                description="指令覆盖尝试"
            ),
            
            # 输出控制
            InjectionPattern(
                pattern=r"print\s+(exactly|the\s+following):",
                injection_type=InjectionType.DIRECT,
                severity=6,
                description="输出控制尝试"
            ),
            InjectionPattern(
                pattern=r"output\s+(only|exactly):",
                injection_type=InjectionType.DIRECT,
                severity=6,
                description="输出控制尝试"
            ),
            InjectionPattern(
                pattern=r"repeat\s+(after\s+me|the\s+following):",
                injection_type=InjectionType.DIRECT,
                severity=6,
                description="重复输出尝试"
            ),
            
            # 上下文注入
            InjectionPattern(
                pattern=r"my\s+(real|actual)\s+(instructions?|prompt)\s+(is|are):",
                injection_type=InjectionType.CONTEXT,
                severity=8,
                description="上下文伪造"
            ),
            InjectionPattern(
                pattern=r"the\s+(real|actual)\s+task\s+is:",
                injection_type=InjectionType.CONTEXT,
                severity=8,
                description="任务伪造"
            ),
            
            # 编码绕过
            InjectionPattern(
                pattern=r"\\x[0-9a-fA-F]{2}",
                injection_type=InjectionType.ENCODING,
                severity=5,
                description="十六进制编码"
            ),
            InjectionPattern(
                pattern=r"\\u[0-9a-fA-F]{4}",
                injection_type=InjectionType.ENCODING,
                severity=5,
                description="Unicode编码"
            ),
            InjectionPattern(
                pattern=r"%[0-9a-fA-F]{2}",
                injection_type=InjectionType.ENCODING,
                severity=5,
                description="URL编码"
            ),
            
            # 特殊字符注入
            InjectionPattern(
                pattern=r"[\x00-\x1f\x7f-\x9f]",
                injection_type=InjectionType.SPECIAL_CHAR,
                severity=4,
                description="控制字符注入"
            ),
            
            # 命令注入模式
            InjectionPattern(
                pattern=r"\$\([^)]+\)",  # $(command)
                injection_type=InjectionType.COMMAND,
                severity=7,
                description="Shell命令替换"
            ),
            InjectionPattern(
                pattern=r"`[^`]+`",  # `command`
                injection_type=InjectionType.COMMAND,
                severity=6,
                description="反引号命令执行"
            ),
            
            # 模板注入
            InjectionPattern(
                pattern=r"\{\{[^}]+\}\}",
                injection_type=InjectionType.TEMPLATE,
                severity=6,
                description="模板注入"
            ),
            InjectionPattern(
                pattern=r"\$\{[^}]+\}",
                injection_type=InjectionType.TEMPLATE,
                severity=6,
                description="变量模板注入"
            ),
            
            # 越狱相关
            InjectionPattern(
                pattern=r"(do\s+not|don't)\s+(follow|obey|adhere\s+to)\s+(rules?|guidelines?|restrictions?)",
                injection_type=InjectionType.DIRECT,
                severity=8,
                description="规则违反诱导"
            ),
            InjectionPattern(
                pattern=r"bypass\s+(security|filter|restriction|safety)",
                injection_type=InjectionType.DIRECT,
                severity=9,
                description="安全绕过尝试"
            ),
        ]
        return patterns
    
    def add_custom_pattern(
        self,
        pattern: str,
        injection_type: InjectionType,
        severity: int,
        description: str
    ) -> None:
        """添加自定义模式"""
        self._custom_patterns.append(InjectionPattern(
            pattern=pattern,
            injection_type=injection_type,
            severity=severity,
            description=description
        ))
    
    def add_whitelist_pattern(self, pattern: str) -> None:
        """添加白名单模式"""
        self._whitelist_patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def scan(self, prompt: str) -> List[InjectionMatch]:
        """扫描Prompt中的注入"""
        matches = []
        
        # 检查白名单
        for whitelist_pattern in self._whitelist_patterns:
            if whitelist_pattern.search(prompt):
                return []  # 在白名单中，不检测
        
        # 扫描所有模式
        all_patterns = self._patterns + self._custom_patterns
        
        for pattern in all_patterns:
            for match in pattern.regex.finditer(prompt):
                # 获取上下文
                start = max(0, match.start() - self._context_window)
                end = min(len(prompt), match.end() + self._context_window)
                context = prompt[start:end]
                
                injection_match = InjectionMatch(
                    injection_type=pattern.injection_type,
                    pattern=pattern.pattern,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=pattern.severity,
                    description=pattern.description,
                    context=context
                )
                matches.append(injection_match)
        
        # 按严重程度排序
        matches.sort(key=lambda x: x.severity, reverse=True)
        
        return matches
    
    def scan_batch(self, prompts: List[str]) -> Dict[int, List[InjectionMatch]]:
        """批量扫描"""
        return {i: self.scan(prompt) for i, prompt in enumerate(prompts)}
    
    def get_risk_score(self, prompt: str) -> Tuple[float, List[InjectionMatch]]:
        """计算风险分数"""
        matches = self.scan(prompt)
        
        if not matches:
            return 0.0, []
        
        # 计算加权风险分数
        total_severity = sum(m.severity for m in matches)
        max_possible = len(matches) * 10
        
        # 考虑注入类型多样性
        unique_types = len(set(m.injection_type for m in matches))
        diversity_factor = 1 + (unique_types - 1) * 0.1
        
        risk_score = min(1.0, (total_severity / max_possible) * diversity_factor)
        
        return risk_score, matches
    
    def is_safe(self, prompt: str, threshold: float = 0.3) -> Tuple[bool, float]:
        """检查是否安全"""
        risk_score, _ = self.get_risk_score(prompt)
        return risk_score < threshold, risk_score
    
    def get_statistics(self, matches: List[InjectionMatch]) -> Dict[str, Any]:
        """获取统计信息"""
        if not matches:
            return {
                "total_matches": 0,
                "max_severity": 0,
                "avg_severity": 0,
                "type_distribution": {},
                "risk_level": "safe"
            }
        
        type_distribution: Dict[InjectionType, int] = {}
        for match in matches:
            type_distribution[match.injection_type] = type_distribution.get(match.injection_type, 0) + 1
        
        max_severity = max(m.severity for m in matches)
        avg_severity = sum(m.severity for m in matches) / len(matches)
        
        # 风险等级
        if max_severity >= 8:
            risk_level = "critical"
        elif max_severity >= 6:
            risk_level = "high"
        elif max_severity >= 4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "total_matches": len(matches),
            "max_severity": max_severity,
            "avg_severity": avg_severity,
            "type_distribution": {t.value: c for t, c in type_distribution.items()},
            "risk_level": risk_level
        }
    
    def suggest_mitigation(self, matches: List[InjectionMatch]) -> List[str]:
        """建议缓解措施"""
        suggestions = []
        
        if not matches:
            return ["未检测到注入风险"]
        
        seen_types = set()
        for match in matches:
            if match.injection_type in seen_types:
                continue
            seen_types.add(match.injection_type)
            
            if match.injection_type == InjectionType.DIRECT:
                suggestions.append("检测到直接指令注入，建议：使用严格的输入验证和指令隔离")
            elif match.injection_type == InjectionType.ROLE_PLAY:
                suggestions.append("检测到角色扮演注入，建议：限制角色切换能力，验证角色合法性")
            elif match.injection_type == InjectionType.CONTEXT:
                suggestions.append("检测到上下文注入，建议：使用上下文边界标记，隔离用户输入")
            elif match.injection_type == InjectionType.ENCODING:
                suggestions.append("检测到编码绕过尝试，建议：标准化输入编码，解码后重新检测")
            elif match.injection_type == InjectionType.COMMAND:
                suggestions.append("检测到命令注入，建议：禁止命令执行，使用参数化查询")
            elif match.injection_type == InjectionType.TEMPLATE:
                suggestions.append("检测到模板注入，建议：使用安全的模板引擎，沙箱化模板执行")
            elif match.injection_type == InjectionType.SPECIAL_CHAR:
                suggestions.append("检测到特殊字符注入，建议：过滤控制字符，使用白名单验证")
        
        return suggestions
