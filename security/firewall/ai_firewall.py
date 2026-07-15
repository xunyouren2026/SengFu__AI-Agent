"""
AI防火墙模块 - AI Firewall

实现AI系统安全防护机制：
1. 输入过滤 (Input Filtering)
2. 输出过滤 (Output Filtering)
3. 速率限制 (Rate Limiting)
4. 异常检测 (Anomaly Detection)
5. 内容策略执行 (Content Policy Enforcement)
"""

import re
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from abc import ABC, abstractmethod
import threading
import math


class FilterAction(Enum):
    """过滤动作"""
    ALLOW = "allow"
    BLOCK = "block"
    FLAG = "flag"
    SANITIZE = "sanitize"
    RATE_LIMIT = "rate_limit"


class ContentCategory(Enum):
    """内容类别"""
    SAFE = "safe"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    ILLEGAL = "illegal"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    PII = "pii"
    TOXIC = "toxic"


@dataclass
class FilterResult:
    """过滤结果"""
    action: FilterAction
    allowed: bool
    risk_score: float
    category: Optional[ContentCategory] = None
    matched_rules: List[Dict] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FirewallConfig:
    """防火墙配置"""
    # 输入过滤
    enable_input_filter: bool = True
    input_risk_threshold: float = 0.7
    
    # 输出过滤
    enable_output_filter: bool = True
    output_risk_threshold: float = 0.6
    
    # 速率限制
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    
    # 异常检测
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 2.5  # 标准差倍数
    history_window: int = 100
    
    # 内容策略
    enable_policy_enforcement: bool = True
    strict_mode: bool = False
    
    # 响应设置
    block_message: str = "Content blocked by security policy"
    flag_message: str = "Content flagged for review"


class InputFilter:
    """
    输入过滤器
    
    过滤和检测恶意输入：
    - 恶意代码检测
    - 社会工程攻击检测
    - 钓鱼攻击检测
    - 数据泄露尝试检测
    """
    
    # 恶意代码模式
    MALICIOUS_CODE_PATTERNS = {
        'script_injection': [
            r'<script[^>]*>.*?</script>',
            r'javascript:\s*',
            r'on\w+\s*=\s*["\'][^"\']*["\']',
            r'\beval\s*\(',
            r'\bFunction\s*\(',
            r'\bsetTimeout\s*\([^,]+,\s*["\']',
            r'\bsetInterval\s*\([^,]+,\s*["\']',
        ],
        'sql_injection': [
            r"'\s*OR\s*['\"\d]\s*=\s*['\"\d]",
            r";\s*--",
            r";\s*DROP\s+",
            r";\s*DELETE\s+FROM",
            r"UNION\s+SELECT",
            r"INSERT\s+INTO",
            r"UPDATE\s+\w+\s+SET",
        ],
        'command_injection': [
            r'[`;|&$]\s*\w+',
            r'\$\([^)]+\)',
            r'`[^`]+`',
            r'\|\s*\w+',
            r';\s*\w+\s+-',
            r'&&\s*\w+',
            r'\|\|\s*\w+',
        ],
        'path_traversal': [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e',
            r'\\x2e\\x2e',
        ],
        'xxe_injection': [
            r'<!ENTITY\s+\w+\s+SYSTEM',
            r'<!DOCTYPE\s+\w+\s+\[',
            r'file://',
        ],
    }
    
    # 社会工程攻击模式
    SOCIAL_ENGINEERING_PATTERNS = {
        'urgency': [
            r'\burgent\b',
            r'\bimmediate\s+action\s+required\b',
            r'\bact\s+now\b',
            r'\blimited\s+time\b',
            r'\bexpires?\s+(?:today|soon)',
            r'\brunning\s+out\b',
        ],
        'authority': [
            r'\badministrator\b',
            r'\bsystem\s+admin\b',
            r'\bsecurity\s+team\b',
            r'\bIT\s+department\b',
            r'\bsupport\s+team\b',
            r'\bverify\s+your\s+account\b',
        ],
        'credential_harvesting': [
            r'\bpassword\b.*\bconfirm\b',
            r'\bverify\s+your\s+password\b',
            r'\blogin\s+credentials\b',
            r'\baccount\s+verification\b',
            r'\bsuspicious\s+activity\b',
        ],
        'fake_prizes': [
            r'\bcongratulations\b.*\bwon\b',
            r'\byou\s+have\s+won\b',
            r'\bclaim\s+your\s+prize\b',
            r'\bwinner\b',
            r'\blottery\b',
        ],
    }
    
    # 数据泄露尝试模式
    DATA_EXFILTRATION_PATTERNS = {
        'file_access': [
            r'\b/etc/passwd\b',
            r'\b/etc/shadow\b',
            r'\b\.env\b',
            r'\bconfig\.\w+\b',
            r'\bcredentials\b',
            r'\bsecret\b',
            r'\bprivate[_-]?key\b',
        ],
        'database_queries': [
            r'SELECT\s+\*\s+FROM',
            r'SHOW\s+DATABASES',
            r'SHOW\s+TABLES',
            r'DESCRIBE\s+\w+',
        ],
        'system_info': [
            r'\bwhoami\b',
            r'\bid\b',
            r'\buname\s+-a\b',
            r'\bifconfig\b',
            r'\bip\s+addr\b',
            r'\bps\s+-ef\b',
        ],
    }
    
    # 提示词注入模式
    PROMPT_INJECTION_PATTERNS = [
        r'ignore\s+(?:all\s+)?previous\s+instructions',
        r'disregard\s+(?:all\s+)?(?:your\s+)?instructions',
        r'forget\s+(?:all\s+)?(?:your\s+)?instructions',
        r'you\s+are\s+now\s+a',
        r'act\s+as\s+(?:if\s+)?you\s+are',
        r'pretend\s+to\s+be',
        r'new\s+persona\s*:',
        r'system\s*:\s*',
        r'assistant\s*:\s*',
    ]
    
    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        self._compile_patterns()
        self.filter_stats = defaultdict(int)
    
    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.compiled_malicious = {}
        for category, patterns in self.MALICIOUS_CODE_PATTERNS.items():
            self.compiled_malicious[category] = [
                re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns
            ]
        
        self.compiled_social = {}
        for category, patterns in self.SOCIAL_ENGINEERING_PATTERNS.items():
            self.compiled_social[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self.compiled_exfil = {}
        for category, patterns in self.DATA_EXFILTRATION_PATTERNS.items():
            self.compiled_exfil[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self.compiled_injection = [
            re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTION_PATTERNS
        ]
    
    def detect_malicious_code(self, content: str) -> Tuple[float, List[Dict]]:
        """检测恶意代码"""
        score = 0.0
        matches = []
        
        for category, patterns in self.compiled_malicious.items():
            for pattern in patterns:
                if pattern.search(content):
                    score += 0.25
                    matches.append({
                        'type': 'malicious_code',
                        'category': category,
                        'pattern': pattern.pattern[:50],
                        'severity': 'critical'
                    })
        
        return min(score, 1.0), matches
    
    def detect_social_engineering(self, content: str) -> Tuple[float, List[Dict]]:
        """检测社会工程攻击"""
        score = 0.0
        matches = []
        
        for category, patterns in self.compiled_social.items():
            category_matches = 0
            for pattern in patterns:
                if pattern.search(content):
                    category_matches += 1
                    matches.append({
                        'type': 'social_engineering',
                        'category': category,
                        'matched_text': pattern.search(content).group()[:50],
                        'severity': 'medium'
                    })
            
            if category_matches > 0:
                score += 0.15 * min(category_matches, 3)
        
        return min(score, 1.0), matches
    
    def detect_data_exfiltration(self, content: str) -> Tuple[float, List[Dict]]:
        """检测数据泄露尝试"""
        score = 0.0
        matches = []
        
        for category, patterns in self.compiled_exfil.items():
            for pattern in patterns:
                if pattern.search(content):
                    score += 0.2
                    matches.append({
                        'type': 'data_exfiltration',
                        'category': category,
                        'pattern': pattern.pattern[:50],
                        'severity': 'high'
                    })
        
        return min(score, 1.0), matches
    
    def detect_prompt_injection(self, content: str) -> Tuple[float, List[Dict]]:
        """检测提示词注入"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_injection:
            if pattern.search(content):
                score += 0.2
                matches.append({
                    'type': 'prompt_injection',
                    'pattern': pattern.pattern[:50],
                    'severity': 'high'
                })
        
        # 检测分隔符滥用
        delimiter_count = content.count('```') + content.count('"""') + content.count("'''")
        if delimiter_count >= 2:
            score += 0.1 * delimiter_count
            matches.append({
                'type': 'delimiter_abuse',
                'count': delimiter_count,
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def sanitize_input(self, content: str) -> str:
        """净化输入内容"""
        sanitized = content
        
        # 移除控制字符
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32 or c in '\n\r\t')
        
        # 规范化空白
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # 限制长度
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized
    
    def filter(self, content: str, context: Optional[Dict] = None) -> FilterResult:
        """
        执行输入过滤
        
        Args:
            content: 输入内容
            context: 上下文信息
            
        Returns:
            FilterResult: 过滤结果
        """
        if not self.config.enable_input_filter:
            return FilterResult(
                action=FilterAction.ALLOW,
                allowed=True,
                risk_score=0.0
            )
        
        all_matches = []
        total_score = 0.0
        
        # 各项检测
        score, matches = self.detect_malicious_code(content)
        total_score += score
        all_matches.extend(matches)
        
        score, matches = self.detect_social_engineering(content)
        total_score += score
        all_matches.extend(matches)
        
        score, matches = self.detect_data_exfiltration(content)
        total_score += score
        all_matches.extend(matches)
        
        score, matches = self.detect_prompt_injection(content)
        total_score += score
        all_matches.extend(matches)
        
        # 归一化分数
        final_score = min(total_score / 4, 1.0)
        
        # 确定动作
        if final_score >= self.config.input_risk_threshold:
            action = FilterAction.BLOCK
            allowed = False
        elif final_score >= self.config.input_risk_threshold * 0.7:
            action = FilterAction.FLAG
            allowed = True
        else:
            action = FilterAction.ALLOW
            allowed = True
        
        # 净化
        sanitized = self.sanitize_input(content) if action != FilterAction.ALLOW else content
        
        self.filter_stats['input_filtered'] += 1
        if not allowed:
            self.filter_stats['input_blocked'] += 1
        
        return FilterResult(
            action=action,
            allowed=allowed,
            risk_score=final_score,
            matched_rules=all_matches,
            sanitized_content=sanitized if action != FilterAction.ALLOW else None,
            metadata={
                'original_length': len(content),
                'detection_categories': list(set(m['type'] for m in all_matches))
            }
        )


class OutputFilter:
    """
    输出过滤器
    
    过滤和检测有害输出：
    - 有害内容检测
    - 偏见内容检测
    - 虚假信息检测
    - 个人信息泄露检测
    """
    
    # 有害内容模式
    HARMFUL_CONTENT_PATTERNS = {
        'hate_speech': [
            r'\b(hate|hating)\s+(?:all\s+)?\w+\s+(?:people|race|group)',
            r'\b(inferior|superior)\s+(?:race|ethnicity|group)',
            r'\b(eliminate|exterminate|destroy)\s+(?:all\s+)?\w+',
        ],
        'harassment': [
            r'\b(stupid|idiot|moron|dumb)\s+(?:person|people)?',
            r'\bkill\s+yourself\b',
            r'\bgo\s+to\s+hell\b',
            r'\bworthless\b',
        ],
        'self_harm': [
            r'\bhow\s+to\s+(?:commit\s+)?suicide\b',
            r'\bkill\s+myself\b',
            r'\bend\s+(?:my\s+)?life\b',
            r'\bself[-\s]?harm\b',
            r'\bcut\s+(?:myself|yourself)\b',
        ],
        'violence': [
            r'\bhow\s+to\s+make\s+(?:a\s+)?(?:bomb|weapon)',
            r'\bbuild\s+(?:a\s+)?(?:explosive|weapon)',
            r'\bpoison\s+(?:someone|people)',
            r'\battack\s+(?:someone|people)',
        ],
        'illegal': [
            r'\bhow\s+to\s+(?:steal|hack|break\s+into)',
            r'\bcreate\s+(?:fake|forged)\s+',
            r'\billegal\s+(?:drugs|substances)',
            r'\bbuy\s+(?:illegal|stolen)\s+',
        ],
    }
    
    # 个人信息模式
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    # 偏见指示词
    BIAS_INDICATORS = [
        r'\ball\s+\w+\s+are\s+(?:the\s+)?same',
        r'\bwomen\s+(?:can\'t|cannot|shouldn\'t)',
        r'\bmen\s+(?:can\'t|cannot|shouldn\'t)',
        r'\b(?:always|never)\s+(?:do|behave|act)\s+like',
        r'\bnatural\s+(?:role|place)\s+(?:for|of)\b',
    ]
    
    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        self._compile_patterns()
        self.filter_stats = defaultdict(int)
    
    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.compiled_harmful = {}
        for category, patterns in self.HARMFUL_CONTENT_PATTERNS.items():
            self.compiled_harmful[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self.compiled_pii = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PII_PATTERNS.items()
        }
        
        self.compiled_bias = [
            re.compile(p, re.IGNORECASE) for p in self.BIAS_INDICATORS
        ]
    
    def detect_harmful_content(self, content: str) -> Tuple[float, ContentCategory, List[Dict]]:
        """检测有害内容"""
        max_score = 0.0
        detected_category = ContentCategory.SAFE
        all_matches = []
        
        category_scores = {
            'hate_speech': ContentCategory.HATE_SPEECH,
            'harassment': ContentCategory.HARASSMENT,
            'self_harm': ContentCategory.SELF_HARM,
            'violence': ContentCategory.VIOLENCE,
            'illegal': ContentCategory.ILLEGAL,
        }
        
        for category, patterns in self.compiled_harmful.items():
            category_score = 0.0
            category_matches = []
            
            for pattern in patterns:
                matches = list(pattern.finditer(content))
                if matches:
                    category_score += 0.3 * len(matches)
                    for match in matches:
                        category_matches.append({
                            'type': 'harmful_content',
                            'category': category,
                            'matched_text': match.group()[:50],
                            'severity': 'high'
                        })
            
            if category_score > max_score:
                max_score = min(category_score, 1.0)
                detected_category = category_scores.get(category, ContentCategory.TOXIC)
            
            all_matches.extend(category_matches)
        
        return max_score, detected_category, all_matches
    
    def detect_pii(self, content: str) -> Tuple[float, List[Dict]]:
        """检测个人信息"""
        score = 0.0
        matches = []
        
        for pii_type, pattern in self.compiled_pii.items():
            found = list(pattern.finditer(content))
            if found:
                score += 0.2 * len(found)
                for match in found:
                    matches.append({
                        'type': 'pii',
                        'pii_type': pii_type,
                        'position': (match.start(), match.end()),
                        'severity': 'medium'
                    })
        
        return min(score, 1.0), matches
    
    def detect_bias(self, content: str) -> Tuple[float, List[Dict]]:
        """检测偏见内容"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_bias:
            if pattern.search(content):
                score += 0.25
                matches.append({
                    'type': 'bias',
                    'pattern': pattern.pattern[:50],
                    'severity': 'medium'
                })
        
        return min(score, 1.0), matches
    
    def detect_misinformation_indicators(self, content: str) -> Tuple[float, List[Dict]]:
        """检测虚假信息指示"""
        score = 0.0
        matches = []
        
        # 绝对化表述
        absolute_terms = [
            r'\babsolutely\s+(?:true|false|certain)',
            r'\bdefinitely\s+(?:true|false)',
            r'\b100%\s+(?:true|false)',
            r'\bproven\s+fact\b',
            r'\beveryone\s+knows\b',
        ]
        
        for pattern in absolute_terms:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.1
                matches.append({
                    'type': 'absolute_claim',
                    'severity': 'low'
                })
        
        # 阴谋论指示
        conspiracy_indicators = [
            r'\bmainstream\s+media\s+(?:won\'t|doesn\'t)\s+tell\s+you',
            r'\bthey\s+don\'t\s+want\s+you\s+to\s+know\b',
            r'\bwake\s+up\s+sheeple\b',
            r'\bdo\s+your\s+own\s+research\b',
        ]
        
        for pattern in conspiracy_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.15
                matches.append({
                    'type': 'conspiracy_indicator',
                    'severity': 'medium'
                })
        
        return min(score, 1.0), matches
    
    def redact_pii(self, content: str) -> str:
        """脱敏个人信息"""
        redacted = content
        
        for pii_type, pattern in self.compiled_pii.items():
            redacted = pattern.sub(f'[{pii_type.upper()}_REDACTED]', redacted)
        
        return redacted
    
    def filter(self, content: str, context: Optional[Dict] = None) -> FilterResult:
        """
        执行输出过滤
        
        Args:
            content: 输出内容
            context: 上下文信息
            
        Returns:
            FilterResult: 过滤结果
        """
        if not self.config.enable_output_filter:
            return FilterResult(
                action=FilterAction.ALLOW,
                allowed=True,
                risk_score=0.0
            )
        
        all_matches = []
        total_score = 0.0
        
        # 有害内容检测
        harmful_score, category, matches = self.detect_harmful_content(content)
        total_score += harmful_score * 0.5  # 权重较高
        all_matches.extend(matches)
        
        # PII检测
        pii_score, matches = self.detect_pii(content)
        total_score += pii_score * 0.2
        all_matches.extend(matches)
        
        # 偏见检测
        bias_score, matches = self.detect_bias(content)
        total_score += bias_score * 0.15
        all_matches.extend(matches)
        
        # 虚假信息检测
        misinfo_score, matches = self.detect_misinformation_indicators(content)
        total_score += misinfo_score * 0.15
        all_matches.extend(matches)
        
        final_score = min(total_score, 1.0)
        
        # 确定动作
        if final_score >= self.config.output_risk_threshold:
            if harmful_score > 0.5:
                action = FilterAction.BLOCK
                allowed = False
            else:
                action = FilterAction.FLAG
                allowed = True
        elif final_score >= self.config.output_risk_threshold * 0.5:
            action = FilterAction.FLAG
            allowed = True
        else:
            action = FilterAction.ALLOW
            allowed = True
        
        # 脱敏处理
        sanitized = content
        if pii_score > 0 and self.config.strict_mode:
            sanitized = self.redact_pii(content)
        
        self.filter_stats['output_filtered'] += 1
        if not allowed:
            self.filter_stats['output_blocked'] += 1
        
        return FilterResult(
            action=action,
            allowed=allowed,
            risk_score=final_score,
            category=category if harmful_score > 0 else None,
            matched_rules=all_matches,
            sanitized_content=sanitized if sanitized != content else None,
            metadata={
                'harmful_score': harmful_score,
                'pii_score': pii_score,
                'bias_score': bias_score,
                'misinfo_score': misinfo_score
            }
        )


class RateLimiter:
    """
    速率限制器
    
    实现令牌桶算法进行速率限制。
    """
    
    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        
        # 用户令牌桶: user_id -> (tokens, last_update)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        
        # 用户请求历史: user_id -> deque of timestamps
        self._request_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # 阻止列表
        self._blocked_users: Set[str] = set()
        
        self._lock = threading.RLock()
    
    def _get_bucket(self, user_id: str) -> Tuple[float, float]:
        """获取用户的令牌桶状态"""
        if user_id not in self._buckets:
            self._buckets[user_id] = (
                float(self.config.burst_size),
                time.time()
            )
        return self._buckets[user_id]
    
    def _update_bucket(self, user_id: str) -> float:
        """更新令牌桶并返回当前令牌数"""
        tokens, last_update = self._get_bucket(user_id)
        now = time.time()
        
        # 计算新增令牌
        time_passed = now - last_update
        new_tokens = time_passed * (self.config.requests_per_minute / 60.0)
        
        # 更新令牌数
        tokens = min(tokens + new_tokens, float(self.config.burst_size))
        
        self._buckets[user_id] = (tokens, now)
        return tokens
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, Dict]:
        """
        检查速率限制
        
        Returns:
            (是否允许, 详细信息)
        """
        with self._lock:
            if user_id in self._blocked_users:
                return False, {
                    'allowed': False,
                    'reason': 'user_blocked',
                    'retry_after': 3600
                }
            
            # 更新令牌桶
            tokens = self._update_bucket(user_id)
            
            # 检查令牌
            if tokens < 1.0:
                retry_after = (1.0 - tokens) / (self.config.requests_per_minute / 60.0)
                return False, {
                    'allowed': False,
                    'reason': 'rate_limit_exceeded',
                    'retry_after': int(retry_after) + 1,
                    'limit': self.config.requests_per_minute,
                    'window': 'minute'
                }
            
            # 消耗令牌
            self._buckets[user_id] = (tokens - 1.0, time.time())
            
            # 记录请求
            now = time.time()
            self._request_history[user_id].append(now)
            
            # 清理旧记录并检查小时限制
            hour_ago = now - 3600
            while self._request_history[user_id] and self._request_history[user_id][0] < hour_ago:
                self._request_history[user_id].popleft()
            
            hour_requests = len(self._request_history[user_id])
            if hour_requests > self.config.requests_per_hour:
                return False, {
                    'allowed': False,
                    'reason': 'hourly_limit_exceeded',
                    'limit': self.config.requests_per_hour,
                    'window': 'hour',
                    'retry_after': 3600 - int(now - self._request_history[user_id][0])
                }
            
            return True, {
                'allowed': True,
                'remaining': int(tokens - 1),
                'reset_after': int(60 / self.config.requests_per_minute) + 1,
                'hourly_remaining': self.config.requests_per_hour - hour_requests
            }
    
    def block_user(self, user_id: str, duration: Optional[int] = None) -> None:
        """阻止用户"""
        with self._lock:
            self._blocked_users.add(user_id)
            
            if duration:
                # 可以在这里设置定时解除
                pass
    
    def unblock_user(self, user_id: str) -> bool:
        """解除用户阻止"""
        with self._lock:
            if user_id in self._blocked_users:
                self._blocked_users.remove(user_id)
                return True
            return False
    
    def get_user_stats(self, user_id: str) -> Dict:
        """获取用户统计"""
        with self._lock:
            tokens, _ = self._get_bucket(user_id)
            recent_requests = len([
                t for t in self._request_history[user_id]
                if time.time() - t < 60
            ])
            
            return {
                'current_tokens': tokens,
                'recent_requests': recent_requests,
                'is_blocked': user_id in self._blocked_users
            }


class AnomalyDetector:
    """
    异常检测器
    
    使用统计方法检测异常请求模式。
    """
    
    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        
        # 请求特征历史
        self._feature_history: deque = deque(maxlen=config.history_window if config else 100)
        
        # 用户行为基线: user_id -> {feature -> (mean, std)}
        self._user_baselines: Dict[str, Dict[str, Tuple[float, float]]] = {}
        
        # 异常计数
        self._anomaly_counts: Dict[str, int] = defaultdict(int)
        
        self._lock = threading.RLock()
    
    def extract_features(self, content: str, context: Optional[Dict] = None) -> Dict[str, float]:
        """提取内容特征"""
        features = {}
        
        # 长度特征
        features['length'] = len(content)
        features['word_count'] = len(content.split())
        
        # 字符分布特征
        features['uppercase_ratio'] = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        features['digit_ratio'] = sum(1 for c in content if c.isdigit()) / max(len(content), 1)
        features['special_char_ratio'] = sum(1 for c in content if not c.isalnum() and not c.isspace()) / max(len(content), 1)
        
        # 熵特征
        features['entropy'] = self._calculate_entropy(content)
        
        # 重复特征
        features['repetition_score'] = self._calculate_repetition(content)
        
        # 时间特征
        if context and 'timestamp' in context:
            hour = context['timestamp'] % 86400 // 3600
            features['hour'] = hour
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """计算文本熵"""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        total = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_repetition(self, text: str) -> float:
        """计算重复度"""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
        
        unique_words = len(set(words))
        return 1.0 - (unique_words / len(words))
    
    def update_baseline(self, user_id: str, features: Dict[str, float]) -> None:
        """更新用户行为基线"""
        with self._lock:
            if user_id not in self._user_baselines:
                self._user_baselines[user_id] = {}
            
            for feature, value in features.items():
                if feature not in self._user_baselines[user_id]:
                    self._user_baselines[user_id][feature] = (value, 0.0)
                else:
                    old_mean, old_std = self._user_baselines[user_id][feature]
                    # 增量更新
                    new_mean = 0.9 * old_mean + 0.1 * value
                    new_std = 0.9 * old_std + 0.1 * abs(value - old_mean)
                    self._user_baselines[user_id][feature] = (new_mean, new_std)
    
    def detect_anomaly(
        self,
        user_id: str,
        features: Dict[str, float]
    ) -> Tuple[bool, float, List[Dict]]:
        """
        检测异常
        
        Returns:
            (是否异常, 异常分数, 异常详情)
        """
        with self._lock:
            if user_id not in self._user_baselines:
                # 首次请求，建立基线
                self.update_baseline(user_id, features)
                return False, 0.0, []
            
            baseline = self._user_baselines[user_id]
            anomalies = []
            total_z_score = 0.0
            
            for feature, value in features.items():
                if feature in baseline:
                    mean, std = baseline[feature]
                    if std > 0:
                        z_score = abs(value - mean) / std
                        total_z_score += z_score
                        
                        if z_score > self.config.anomaly_threshold:
                            anomalies.append({
                                'feature': feature,
                                'value': value,
                                'expected_mean': mean,
                                'expected_std': std,
                                'z_score': z_score
                            })
            
            # 计算整体异常分数
            anomaly_score = min(total_z_score / max(len(features), 1), 5.0) / 5.0
            is_anomaly = len(anomalies) > 0
            
            if is_anomaly:
                self._anomaly_counts[user_id] += 1
            
            # 更新基线
            self.update_baseline(user_id, features)
            
            return is_anomaly, anomaly_score, anomalies
    
    def analyze_request(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict] = None
    ) -> Tuple[bool, float, List[Dict]]:
        """分析请求是否异常"""
        features = self.extract_features(content, context)
        return self.detect_anomaly(user_id, features)
    
    def get_user_anomaly_count(self, user_id: str) -> int:
        """获取用户异常计数"""
        with self._lock:
            return self._anomaly_counts[user_id]


class ContentPolicy:
    """
    内容策略
    
    定义和执行内容安全策略。
    """
    
    def __init__(self):
        # 禁止主题
        self._blocked_topics: Set[str] = set()
        
        # 必需警告主题
        self._warning_topics: Set[str] = set()
        
        # 自定义规则
        self._custom_rules: List[Dict] = []
        
        # 允许列表
        self._allowed_patterns: List[str] = []
        
        self._lock = threading.RLock()
    
    def block_topic(self, topic: str) -> None:
        """添加禁止主题"""
        with self._lock:
            self._blocked_topics.add(topic.lower())
    
    def allow_topic(self, topic: str) -> None:
        """移除禁止主题"""
        with self._lock:
            self._blocked_topics.discard(topic.lower())
    
    def add_warning_topic(self, topic: str) -> None:
        """添加警告主题"""
        with self._lock:
            self._warning_topics.add(topic.lower())
    
    def add_custom_rule(
        self,
        name: str,
        pattern: str,
        action: FilterAction,
        score: float
    ) -> None:
        """添加自定义规则"""
        with self._lock:
            self._custom_rules.append({
                'name': name,
                'pattern': re.compile(pattern, re.IGNORECASE),
                'action': action,
                'score': score
            })
    
    def add_allowed_pattern(self, pattern: str) -> None:
        """添加允许模式"""
        with self._lock:
            self._allowed_patterns.append(pattern)
    
    def evaluate(self, content: str) -> Tuple[FilterAction, float, List[Dict]]:
        """
        评估内容是否符合策略
        
        Returns:
            (动作, 分数, 匹配规则)
        """
        with self._lock:
            # 检查允许列表
            for pattern in self._allowed_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return FilterAction.ALLOW, 0.0, []
            
            matches = []
            total_score = 0.0
            
            # 检查禁止主题
            content_lower = content.lower()
            for topic in self._blocked_topics:
                if topic in content_lower:
                    matches.append({
                        'rule': 'blocked_topic',
                        'topic': topic,
                        'severity': 'high'
                    })
                    total_score += 0.5
            
            # 检查警告主题
            for topic in self._warning_topics:
                if topic in content_lower:
                    matches.append({
                        'rule': 'warning_topic',
                        'topic': topic,
                        'severity': 'medium'
                    })
                    total_score += 0.2
            
            # 检查自定义规则
            for rule in self._custom_rules:
                if rule['pattern'].search(content):
                    matches.append({
                        'rule': rule['name'],
                        'action': rule['action'].value,
                        'score': rule['score'],
                        'severity': 'high' if rule['action'] == FilterAction.BLOCK else 'medium'
                    })
                    total_score += rule['score']
            
            # 确定动作
            if total_score >= 0.7:
                action = FilterAction.BLOCK
            elif total_score >= 0.4:
                action = FilterAction.FLAG
            elif total_score > 0:
                action = FilterAction.SANITIZE
            else:
                action = FilterAction.ALLOW
            
            return action, min(total_score, 1.0), matches


class AIFirewall:
    """
    AI防火墙
    
    整合所有安全防护机制的统一接口。
    """
    
    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        
        # 初始化各模块
        self.input_filter = InputFilter(self.config)
        self.output_filter = OutputFilter(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.content_policy = ContentPolicy()
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'flagged_requests': 0,
            'rate_limited': 0,
            'anomalies_detected': 0
        }
        
        self._lock = threading.Lock()
    
    def check_input(self, content: str, user_id: str, context: Optional[Dict] = None) -> FilterResult:
        """
        检查输入内容
        
        Args:
            content: 输入内容
            user_id: 用户ID
            context: 上下文信息
            
        Returns:
            FilterResult: 过滤结果
        """
        with self._lock:
            self.stats['total_requests'] += 1
        
        # 1. 速率限制检查
        if self.config.enable_rate_limiting:
            allowed, rate_info = self.rate_limiter.check_rate_limit(user_id)
            if not allowed:
                with self._lock:
                    self.stats['rate_limited'] += 1
                return FilterResult(
                    action=FilterAction.RATE_LIMIT,
                    allowed=False,
                    risk_score=1.0,
                    metadata=rate_info
                )
        
        # 2. 异常检测
        if self.config.enable_anomaly_detection:
            is_anomaly, anomaly_score, anomalies = self.anomaly_detector.analyze_request(
                user_id, content, context
            )
            if is_anomaly and anomaly_score > 0.8:
                with self._lock:
                    self.stats['anomalies_detected'] += 1
                # 高异常分数可能导致阻止
                if self.config.strict_mode:
                    return FilterResult(
                        action=FilterAction.BLOCK,
                        allowed=False,
                        risk_score=anomaly_score,
                        matched_rules=anomalies,
                        metadata={'anomaly_detected': True}
                    )
        
        # 3. 输入过滤
        input_result = self.input_filter.filter(content, context)
        
        # 4. 内容策略检查
        if self.config.enable_policy_enforcement:
            policy_action, policy_score, policy_matches = self.content_policy.evaluate(content)
            
            # 合并结果
            combined_score = max(input_result.risk_score, policy_score)
            combined_matches = input_result.matched_rules + policy_matches
            
            if policy_action == FilterAction.BLOCK or input_result.action == FilterAction.BLOCK:
                final_action = FilterAction.BLOCK
                allowed = False
            elif policy_action == FilterAction.FLAG or input_result.action == FilterAction.FLAG:
                final_action = FilterAction.FLAG
                allowed = True
            else:
                final_action = input_result.action
                allowed = input_result.allowed
            
            if not allowed:
                with self._lock:
                    self.stats['blocked_requests'] += 1
            elif final_action == FilterAction.FLAG:
                with self._lock:
                    self.stats['flagged_requests'] += 1
            
            return FilterResult(
                action=final_action,
                allowed=allowed,
                risk_score=combined_score,
                matched_rules=combined_matches,
                sanitized_content=input_result.sanitized_content,
                metadata={
                    'input_filter_result': input_result.metadata,
                    'policy_matches': policy_matches,
                    'rate_limit_info': rate_info if self.config.enable_rate_limiting else None
                }
            )
        
        if not input_result.allowed:
            with self._lock:
                self.stats['blocked_requests'] += 1
        elif input_result.action == FilterAction.FLAG:
            with self._lock:
                self.stats['flagged_requests'] += 1
        
        return input_result
    
    def check_output(self, content: str, user_id: str, context: Optional[Dict] = None) -> FilterResult:
        """
        检查输出内容
        
        Args:
            content: 输出内容
            user_id: 用户ID
            context: 上下文信息
            
        Returns:
            FilterResult: 过滤结果
        """
        return self.output_filter.filter(content, context)
    
    def process_request(
        self,
        input_content: str,
        user_id: str,
        generate_fn: Optional[Callable] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        处理完整请求流程
        
        Args:
            input_content: 输入内容
            user_id: 用户ID
            generate_fn: 内容生成函数
            context: 上下文信息
            
        Returns:
            处理结果
        """
        # 1. 输入检查
        input_result = self.check_input(input_content, user_id, context)
        
        if not input_result.allowed:
            return {
                'success': False,
                'stage': 'input_filter',
                'blocked': True,
                'message': self.config.block_message,
                'risk_score': input_result.risk_score,
                'violations': input_result.matched_rules
            }
        
        # 2. 生成内容
        output_content = None
        if generate_fn:
            try:
                sanitized_input = input_result.sanitized_content or input_content
                output_content = generate_fn(sanitized_input)
            except Exception as e:
                return {
                    'success': False,
                    'stage': 'generation',
                    'error': str(e)
                }
        
        # 3. 输出检查
        if output_content:
            output_result = self.check_output(output_content, user_id, context)
            
            if not output_result.allowed:
                return {
                    'success': False,
                    'stage': 'output_filter',
                    'blocked': True,
                    'message': self.config.block_message,
                    'risk_score': output_result.risk_score,
                    'violations': output_result.matched_rules
                }
            
            final_output = output_result.sanitized_content or output_content
            
            return {
                'success': True,
                'output': final_output,
                'input_flagged': input_result.action == FilterAction.FLAG,
                'output_flagged': output_result.action == FilterAction.FLAG,
                'input_risk_score': input_result.risk_score,
                'output_risk_score': output_result.risk_score,
                'sanitized': (
                    input_result.sanitized_content is not None or
                    output_result.sanitized_content is not None
                )
            }
        
        return {
            'success': True,
            'input_allowed': True,
            'risk_score': input_result.risk_score
        }
    
    def get_statistics(self) -> Dict:
        """获取防火墙统计信息"""
        with self._lock:
            total = self.stats['total_requests']
            return {
                **self.stats,
                'block_rate': self.stats['blocked_requests'] / total if total > 0 else 0,
                'flag_rate': self.stats['flagged_requests'] / total if total > 0 else 0,
                'input_filter_stats': dict(self.input_filter.filter_stats),
                'output_filter_stats': dict(self.output_filter.filter_stats)
            }
    
    def block_user(self, user_id: str) -> None:
        """阻止用户"""
        self.rate_limiter.block_user(user_id)
    
    def unblock_user(self, user_id: str) -> bool:
        """解除用户阻止"""
        return self.rate_limiter.unblock_user(user_id)
