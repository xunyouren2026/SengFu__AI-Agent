"""
信息泄露检测器 - 检测Prompt中的敏感信息泄露
"""
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class LeakageType(Enum):
    """泄露类型枚举"""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    SECRET = "secret"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"                   # 社会安全号
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    URL = "url"
    PRIVATE_KEY = "private_key"
    DATABASE_CONNECTION = "database_connection"
    AWS_CREDENTIAL = "aws_credential"
    JWT = "jwt"
    PERSONAL_INFO = "personal_info"
    INTERNAL_PATH = "internal_path"
    ENV_VAR = "env_var"


@dataclass
class LeakagePattern:
    """泄露模式"""
    pattern: str
    leakage_type: LeakageType
    severity: int  # 1-10
    description: str
    regex: re.Pattern = field(init=False)
    mask_char: str = "*"
    
    def __post_init__(self):
        self.regex = re.compile(self.pattern)


@dataclass
class LeakageMatch:
    """泄露匹配结果"""
    leakage_type: LeakageType
    matched_text: str
    start_pos: int
    end_pos: int
    severity: int
    description: str
    masked_text: str
    context: str = ""


class LeakageDetector:
    """信息泄露检测器"""
    
    def __init__(self):
        self._patterns: List[LeakagePattern] = self._load_default_patterns()
        self._custom_patterns: List[LeakagePattern] = []
        self._whitelist: Set[str] = set()
        self._context_window = 30
    
    def _load_default_patterns(self) -> List[LeakagePattern]:
        """加载默认泄露模式"""
        patterns = [
            # API密钥
            LeakagePattern(
                pattern=r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
                leakage_type=LeakageType.API_KEY,
                severity=9,
                description="API密钥泄露",
                mask_char="*"
            ),
            LeakagePattern(
                pattern=r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API key
                leakage_type=LeakageType.API_KEY,
                severity=10,
                description="OpenAI API密钥"
            ),
            LeakagePattern(
                pattern=r"sk_live_[a-zA-Z0-9]{24,}",  # Stripe live key
                leakage_type=LeakageType.API_KEY,
                severity=10,
                description="Stripe API密钥"
            ),
            
            # 密码
            LeakagePattern(
                pattern=r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?([^'\"\\s]{4,})['\"]?",
                leakage_type=LeakageType.PASSWORD,
                severity=9,
                description="密码泄露"
            ),
            
            # Token
            LeakagePattern(
                pattern=r"(?i)(token|access_token|auth_token)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-\.]{10,})['\"]?",
                leakage_type=LeakageType.TOKEN,
                severity=8,
                description="Token泄露"
            ),
            LeakagePattern(
                pattern=r"Bearer\s+[a-zA-Z0-9_\-\.]+",
                leakage_type=LeakageType.TOKEN,
                severity=8,
                description="Bearer Token"
            ),
            
            # 密钥
            LeakagePattern(
                pattern=r"(?i)(secret|secret_key|private_key)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{10,})['\"]?",
                leakage_type=LeakageType.SECRET,
                severity=9,
                description="密钥泄露"
            ),
            
            # 信用卡
            LeakagePattern(
                pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
                leakage_type=LeakageType.CREDIT_CARD,
                severity=10,
                description="信用卡号"
            ),
            
            # SSN (美国社会安全号)
            LeakagePattern(
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                leakage_type=LeakageType.SSN,
                severity=9,
                description="社会安全号"
            ),
            
            # 邮箱
            LeakagePattern(
                pattern=r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                leakage_type=LeakageType.EMAIL,
                severity=5,
                description="邮箱地址"
            ),
            
            # 电话号码
            LeakagePattern(
                pattern=r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                leakage_type=LeakageType.PHONE,
                severity=5,
                description="电话号码"
            ),
            
            # IP地址
            LeakagePattern(
                pattern=r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})\b",
                leakage_type=LeakageType.IP_ADDRESS,
                severity=6,
                description="内网IP地址"
            ),
            LeakagePattern(
                pattern=r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                leakage_type=LeakageType.IP_ADDRESS,
                severity=4,
                description="IP地址"
            ),
            
            # 私钥
            LeakagePattern(
                pattern=r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
                leakage_type=LeakageType.PRIVATE_KEY,
                severity=10,
                description="私钥泄露"
            ),
            LeakagePattern(
                pattern=r"-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----",
                leakage_type=LeakageType.PRIVATE_KEY,
                severity=10,
                description="SSH私钥"
            ),
            
            # 数据库连接字符串
            LeakagePattern(
                pattern=r"(?i)(mysql|postgres|mongodb|redis)://[^\s]+",
                leakage_type=LeakageType.DATABASE_CONNECTION,
                severity=9,
                description="数据库连接字符串"
            ),
            LeakagePattern(
                pattern=r"(?i)jdbc:[a-z]+://[^\s]+",
                leakage_type=LeakageType.DATABASE_CONNECTION,
                severity=9,
                description="JDBC连接字符串"
            ),
            
            # AWS凭证
            LeakagePattern(
                pattern=r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
                leakage_type=LeakageType.AWS_CREDENTIAL,
                severity=10,
                description="AWS访问密钥"
            ),
            LeakagePattern(
                pattern=r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
                leakage_type=LeakageType.AWS_CREDENTIAL,
                severity=10,
                description="AWS秘密密钥"
            ),
            
            # JWT
            LeakagePattern(
                pattern=r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
                leakage_type=LeakageType.JWT,
                severity=7,
                description="JWT Token"
            ),
            
            # 内部路径
            LeakagePattern(
                pattern=r"(?:/home/|/etc/|/var/|/usr/|C:\\Users\\|C:\\Windows\\)[^\s]+",
                leakage_type=LeakageType.INTERNAL_PATH,
                severity=5,
                description="内部路径"
            ),
            
            # 环境变量
            LeakagePattern(
                pattern=r"(?i)(export\s+[A-Z_]+=[^\s]+|[A-Z_]+=[^\s]+)",
                leakage_type=LeakageType.ENV_VAR,
                severity=6,
                description="环境变量"
            ),
        ]
        return patterns
    
    def add_custom_pattern(
        self,
        pattern: str,
        leakage_type: LeakageType,
        severity: int,
        description: str,
        mask_char: str = "*"
    ) -> None:
        """添加自定义模式"""
        self._custom_patterns.append(LeakagePattern(
            pattern=pattern,
            leakage_type=leakage_type,
            severity=severity,
            description=description,
            mask_char=mask_char
        ))
    
    def add_to_whitelist(self, value: str) -> None:
        """添加白名单"""
        self._whitelist.add(value)
    
    def detect(self, text: str) -> List[LeakageMatch]:
        """检测信息泄露"""
        matches = []
        
        all_patterns = self._patterns + self._custom_patterns
        
        for pattern in all_patterns:
            for match in pattern.regex.finditer(text):
                matched_text = match.group()
                
                # 检查白名单
                if matched_text in self._whitelist:
                    continue
                
                # 生成掩码文本
                masked_text = self._mask_value(matched_text, pattern)
                
                # 获取上下文
                start = max(0, match.start() - self._context_window)
                end = min(len(text), match.end() + self._context_window)
                context = text[start:end]
                
                leakage_match = LeakageMatch(
                    leakage_type=pattern.leakage_type,
                    matched_text=matched_text,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    severity=pattern.severity,
                    description=pattern.description,
                    masked_text=masked_text,
                    context=context
                )
                matches.append(leakage_match)
        
        # 按严重程度排序
        matches.sort(key=lambda x: x.severity, reverse=True)
        
        return matches
    
    def _mask_value(self, value: str, pattern: LeakagePattern) -> str:
        """掩码敏感值"""
        if len(value) <= 4:
            return pattern.mask_char * len(value)
        
        # 保留前2和后2字符
        return value[:2] + pattern.mask_char * (len(value) - 4) + value[-2:]
    
    def sanitize(self, text: str) -> Tuple[str, List[LeakageMatch]]:
        """清理敏感信息"""
        matches = self.detect(text)
        
        sanitized = text
        # 从后往前替换，避免位置偏移
        for match in reversed(matches):
            sanitized = sanitized[:match.start_pos] + match.masked_text + sanitized[match.end_pos:]
        
        return sanitized, matches
    
    def get_risk_summary(self, text: str) -> Dict[str, Any]:
        """获取风险摘要"""
        matches = self.detect(text)
        
        if not matches:
            return {
                "has_leakage": False,
                "total_count": 0,
                "max_severity": 0,
                "types_found": [],
                "risk_level": "safe"
            }
        
        types_found = list(set(m.leakage_type.value for m in matches))
        max_severity = max(m.severity for m in matches)
        
        # 风险等级
        if max_severity >= 9:
            risk_level = "critical"
        elif max_severity >= 7:
            risk_level = "high"
        elif max_severity >= 5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "has_leakage": True,
            "total_count": len(matches),
            "max_severity": max_severity,
            "types_found": types_found,
            "risk_level": risk_level,
            "details": [
                {
                    "type": m.leakage_type.value,
                    "severity": m.severity,
                    "description": m.description,
                    "masked": m.masked_text
                }
                for m in matches
            ]
        }
    
    def is_safe(self, text: str, threshold_severity: int = 6) -> Tuple[bool, int]:
        """检查是否安全"""
        matches = self.detect(text)
        max_severity = max((m.severity for m in matches), default=0)
        return max_severity < threshold_severity, max_severity
    
    def detect_batch(self, texts: List[str]) -> Dict[int, List[LeakageMatch]]:
        """批量检测"""
        return {i: self.detect(text) for i, text in enumerate(texts)}
    
    def get_statistics(self, matches: List[LeakageMatch]) -> Dict[str, Any]:
        """获取统计信息"""
        if not matches:
            return {"total": 0, "by_type": {}, "by_severity": {}}
        
        by_type: Dict[LeakageType, int] = {}
        by_severity: Dict[int, int] = {}
        
        for match in matches:
            by_type[match.leakage_type] = by_type.get(match.leakage_type, 0) + 1
            by_severity[match.severity] = by_severity.get(match.severity, 0) + 1
        
        return {
            "total": len(matches),
            "by_type": {t.value: c for t, c in by_type.items()},
            "by_severity": by_severity
        }
