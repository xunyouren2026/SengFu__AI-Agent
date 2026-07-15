"""
提示词安全模块 - Prompt Security

实现提示词安全防护机制：
1. 提示词注入检测 (Prompt Injection Detection)
2. 越狱模式检测 (Jailbreak Pattern Detection)
3. 提示词净化 (Prompt Sanitization)
4. 对抗性提示防御 (Adversarial Prompt Defense)
"""

import re
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from enum import Enum


class ThreatLevel(Enum):
    """威胁等级"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityReport:
    """安全检测报告"""
    is_safe: bool
    threat_level: ThreatLevel
    score: float  # 0-1，越高越危险
    detected_patterns: List[Dict[str, Any]]
    sanitized_prompt: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptSecurityConfig:
    """提示词安全配置"""
    # 检测阈值
    injection_threshold: float = 0.75
    jailbreak_threshold: float = 0.70
    adversarial_threshold: float = 0.65
    
    # 评分权重
    injection_weight: float = 0.35
    jailbreak_weight: float = 0.30
    adversarial_weight: float = 0.20
    semantic_weight: float = 0.15
    
    # 净化选项
    enable_sanitization: bool = True
    remove_invisible_chars: bool = True
    normalize_unicode: bool = True
    max_prompt_length: int = 10000
    
    # 防御选项
    enable_delimiter_check: bool = True
    enable_encoding_detection: bool = True
    enable_context_window_attack_detection: bool = True
    
    # 白名单
    allowed_prefixes: List[str] = field(default_factory=list)
    blocked_keywords: List[str] = field(default_factory=list)


class PromptInjectionDetector:
    """
    提示词注入检测器
    
    检测各种提示词注入攻击技术：
    - 分隔符注入
    - 角色切换攻击
    - 指令覆盖攻击
    - 上下文污染攻击
    """
    
    # 注入模式库
    INJECTION_PATTERNS = {
        'delimiter_injection': [
            r'["\']\s*\n\s*ignore\s+previous',
            r'["\']\s*\n\s*disregard\s+',
            r'["\']\s*\n\s*forget\s+',
            r'```\s*\n\s*system\s*:',
            r'<\|im_start\|>\s*system',
            r'\[\s*INST\s*\]',
            r'<<\s*SYS\s*>>',
            r'\{\{\s*system\s*message\s*\}\}',
        ],
        'role_switching': [
            r'\byou\s+are\s+now\s+a\b',
            r'\bact\s+as\s+(?:if\s+)?you\s+are\b',
            r'\bpretend\s+to\s+be\b',
            r'\bfrom\s+now\s+on\s+you\s+are\b',
            r'\byour\s+new\s+role\s+is\b',
            r'\bswitch\s+to\s+being\b',
            r'\bassume\s+the\s+persona\s+of\b',
            r'\broleplay\s+as\b',
        ],
        'instruction_override': [
            r'\bignore\s+(?:all\s+)?previous\s+instructions\b',
            r'\bdisregard\s+(?:all\s+)?(?:your\s+)?instructions\b',
            r'\bforget\s+(?:all\s+)?(?:your\s+)?instructions\b',
            r'\boverride\s+(?:all\s+)?(?:previous\s+)?instructions\b',
            r'\bnew\s+instructions\s*:',
            r'\breplace\s+your\s+instructions\s+with\b',
            r'\binstead\s+of\s+following\b',
            r'\bdo\s+not\s+follow\b',
        ],
        'context_pollution': [
            r'\buser\s*:.*\n\s*assistant\s*:',
            r'\bhuman\s*:.*\n\s*ai\s*:',
            r'\bH\s*:\s*.*\n\s*A\s*:',
            r'previous\s+conversation\s*:',
            r'earlier\s+we\s+discussed\s*:',
            r'context\s*:.*\n\n',
        ],
        'encoding_obfuscation': [
            r'base64\s*\{[^}]+\}',
            r'rot13\s*\{[^}]+\}',
            r'hex\s*\{[^}]+\}',
            r'urlencode\s*\{[^}]+\}',
            r'\b[A-Za-z0-9+/]{50,}={0,2}\b',  # Base64-like
            r'\\x[0-9a-fA-F]{2}',  # Hex escape
            r'%[0-9a-fA-F]{2}',  # URL encoding
        ],
        'prompt_leaking': [
            r'\bshow\s+me\s+your\s+instructions\b',
            r'\bwhat\s+are\s+your\s+instructions\b',
            r'\bprint\s+your\s+system\s+prompt\b',
            r'\brepeat\s+the\s+word\s+\w+\s+forever\b',
            r'\boutput\s+initialization\s+above\b',
            r'\bwhat\s+was\s+written\s+at\s+the\s+top\b',
        ],
    }
    
    # 危险字符集
    DANGEROUS_CHARS = [
        '\x00', '\x01', '\x02', '\x03', '\x04', '\x05',
        '\x06', '\x07', '\x08', '\x0b', '\x0c', '\x0e',
        '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14',
        '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
        '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
    ]
    
    # 零宽字符
    ZERO_WIDTH_CHARS = [
        '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
        '\u180e', '\u200e', '\u200f', '\u202a', '\u202b',
        '\u202c', '\u202d', '\u202e', '\u2061', '\u2062',
        '\u2063', '\u2064', '\u206a', '\u206b', '\u206c',
        '\u206d', '\u206e', '\u206f',
    ]
    
    def __init__(self, config: Optional[PromptSecurityConfig] = None):
        self.config = config or PromptSecurityConfig()
        self._compile_patterns()
        self.injection_history: List[Dict] = []
    
    def _compile_patterns(self) -> None:
        """编译正则表达式模式"""
        self.compiled_patterns = {}
        for category, patterns in self.INJECTION_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) 
                for p in patterns
            ]
    
    def detect_delimiter_injection(self, prompt: str) -> Tuple[float, List[Dict]]:
        """
        检测分隔符注入攻击
        
        Returns:
            (风险分数, 检测到的模式列表)
        """
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['delimiter_injection']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                match_info = {
                    'type': 'delimiter_injection',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:100],
                    'position': (match.start(), match.end()),
                    'severity': 'high'
                }
                matches.append(match_info)
                score += 0.25
        
        # 检测引号不平衡
        quote_chars = ['"', "'", '`', '```']
        for quote in quote_chars:
            count = prompt.count(quote)
            if count % 2 != 0 and count > 0:
                score += 0.15
                matches.append({
                    'type': 'unbalanced_quotes',
                    'quote_char': quote,
                    'count': count,
                    'severity': 'medium'
                })
        
        # 检测特殊标记
        special_markers = ['<|', '|>', '[[', ']]', '{{', '}}']
        for marker in special_markers:
            if marker in prompt:
                score += 0.1
                matches.append({
                    'type': 'special_marker',
                    'marker': marker,
                    'severity': 'low'
                })
        
        return min(score, 1.0), matches
    
    def detect_role_switching(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测角色切换攻击"""
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['role_switching']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                score += 0.20
                matches.append({
                    'type': 'role_switching',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:100],
                    'position': (match.start(), match.end()),
                    'severity': 'high'
                })
        
        # 检测多重角色声明
        role_keywords = ['you are', 'act as', 'pretend', 'roleplay', 'persona']
        role_count = sum(1 for kw in role_keywords if kw.lower() in prompt.lower())
        if role_count >= 2:
            score += 0.15 * role_count
            matches.append({
                'type': 'multiple_role_declarations',
                'count': role_count,
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def detect_instruction_override(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测指令覆盖攻击"""
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['instruction_override']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                score += 0.25
                matches.append({
                    'type': 'instruction_override',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:100],
                    'position': (match.start(), match.end()),
                    'severity': 'critical'
                })
        
        # 检测否定词密度
        negation_words = ['ignore', 'disregard', 'forget', 'override', 'bypass', 
                         'disable', 'remove', 'delete', 'clear', 'reset']
        negation_count = sum(1 for word in negation_words if word in prompt.lower())
        if negation_count >= 2:
            score += 0.1 * negation_count
            matches.append({
                'type': 'high_negation_density',
                'count': negation_count,
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def detect_encoding_obfuscation(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测编码混淆攻击"""
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['encoding_obfuscation']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                score += 0.20
                matches.append({
                    'type': 'encoding_obfuscation',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:50],
                    'position': (match.start(), match.end()),
                    'severity': 'high'
                })
        
        # 检测零宽字符
        zero_width_count = sum(1 for char in self.ZERO_WIDTH_CHARS if char in prompt)
        if zero_width_count > 0:
            score += min(0.3, 0.05 * zero_width_count)
            matches.append({
                'type': 'zero_width_characters',
                'count': zero_width_count,
                'severity': 'high'
            })
        
        # 检测控制字符
        control_char_count = sum(1 for char in self.DANGEROUS_CHARS if char in prompt)
        if control_char_count > 0:
            score += min(0.4, 0.1 * control_char_count)
            matches.append({
                'type': 'control_characters',
                'count': control_char_count,
                'severity': 'critical'
            })
        
        # 检测混合脚本（潜在的同形异义攻击）
        scripts = self._detect_mixed_scripts(prompt)
        if len(scripts) > 1:
            score += 0.15 * (len(scripts) - 1)
            matches.append({
                'type': 'mixed_scripts',
                'scripts': scripts,
                'severity': 'medium'
            })
        
        return min(score, 1.0), matches
    
    def _detect_mixed_scripts(self, text: str) -> List[str]:
        """检测文本中使用的Unicode脚本"""
        scripts = set()
        for char in text:
            if char.isalpha():
                code = ord(char)
                if 0x0041 <= code <= 0x007A:  # Latin
                    scripts.add('Latin')
                elif 0x0400 <= code <= 0x04FF:  # Cyrillic
                    scripts.add('Cyrillic')
                elif 0x0370 <= code <= 0x03FF:  # Greek
                    scripts.add('Greek')
                elif 0x0600 <= code <= 0x06FF:  # Arabic
                    scripts.add('Arabic')
                elif 0x3040 <= code <= 0x309F:  # Hiragana
                    scripts.add('Hiragana')
                elif 0x30A0 <= code <= 0x30FF:  # Katakana
                    scripts.add('Katakana')
                elif 0x4E00 <= code <= 0x9FFF:  # CJK
                    scripts.add('CJK')
        return list(scripts)
    
    def detect_context_pollution(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测上下文污染攻击"""
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['context_pollution']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                score += 0.20
                matches.append({
                    'type': 'context_pollution',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:100],
                    'position': (match.start(), match.end()),
                    'severity': 'high'
                })
        
        # 检测对话模式伪造
        dialogue_patterns = [
            (r'\n\s*User\s*:', 0.15),
            (r'\n\s*Assistant\s*:', 0.15),
            (r'\n\s*System\s*:', 0.20),
            (r'\n\s*Human\s*:', 0.15),
            (r'\n\s*AI\s*:', 0.15),
        ]
        
        for pattern, weight in dialogue_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += weight
                matches.append({
                    'type': 'fake_dialogue_marker',
                    'pattern': pattern,
                    'severity': 'high'
                })
        
        return min(score, 1.0), matches
    
    def detect_prompt_leaking(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测提示词泄露攻击"""
        score = 0.0
        matches = []
        
        patterns = self.compiled_patterns['prompt_leaking']
        for pattern in patterns:
            for match in pattern.finditer(prompt):
                score += 0.25
                matches.append({
                    'type': 'prompt_leaking',
                    'pattern': pattern.pattern[:50],
                    'matched_text': match.group()[:100],
                    'position': (match.start(), match.end()),
                    'severity': 'high'
                })
        
        # 检测重复请求
        repetition_patterns = [
            r'repeat\s+\w+\s+\d+\s+times',
            r'say\s+\w+\s+\d+\s+times',
            r'output\s+\w+\s+\d+\s+times',
        ]
        for pattern in repetition_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.15
                matches.append({
                    'type': 'repetition_attack',
                    'pattern': pattern,
                    'severity': 'medium'
                })
        
        return min(score, 1.0), matches
    
    def analyze(self, prompt: str) -> Tuple[float, List[Dict]]:
        """
        综合分析提示词注入风险
        
        Returns:
            (总风险分数, 所有检测到的模式)
        """
        all_matches = []
        total_score = 0.0
        
        # 各项检测
        detectors = [
            self.detect_delimiter_injection,
            self.detect_role_switching,
            self.detect_instruction_override,
            self.detect_encoding_obfuscation,
            self.detect_context_pollution,
            self.detect_prompt_leaking,
        ]
        
        for detector in detectors:
            score, matches = detector(prompt)
            total_score += score
            all_matches.extend(matches)
        
        # 归一化分数
        final_score = min(total_score / len(detectors), 1.0)
        
        # 记录历史
        self.injection_history.append({
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16],
            'score': final_score,
            'matches_count': len(all_matches),
        })
        
        return final_score, all_matches


class JailbreakDetector:
    """
    越狱模式检测器
    
    检测试图绕过AI安全限制的越狱攻击：
    - 假设情景攻击
    - 角色扮演攻击
    - 逻辑操纵攻击
    - 情感操纵攻击
    """
    
    # 越狱模式库
    JAILBREAK_PATTERNS = {
        'hypothetical_framing': [
            r'\bhypothetically\s+speaking\b',
            r'\bin\s+a\s+hypothetical\s+scenario\b',
            r'\bfor\s+educational\s+purposes\b',
            r'\bthis\s+is\s+purely\s+theoretical\b',
            r'\bjust\s+imagine\b',
            r'\bsuppose\s+for\s+a\s+moment\b',
            r'\blet\s+me\s+ask\s+a\s+hypothetical\b',
            r'\bfor\s+a\s+fictional\s+story\b',
            r'\bwriting\s+a\s+novel\s+where\b',
            r'\bresearch\s+purposes\s+only\b',
        ],
        'authority_manipulation': [
            r'\bdeveloper\s+mode\b',
            r'\bdebug\s+mode\b',
            r'\badmin\s+access\b',
            r'\broot\s+privileges\b',
            r'\bmaintenance\s+mode\b',
            r'\boverride\s+safety\b',
            r'\bdisable\s+restrictions\b',
            r'\bunfiltered\s+mode\b',
            r'\buncensored\s+mode\b',
            r'\bDAN\s+mode\b',
        ],
        'logical_manipulation': [
            r'\bif\s+you\s+don\'t\s+answer\b',
            r'\byou\s+must\s+answer\b',
            r'\byou\s+are\s+required\s+to\b',
            r'\bit\s+is\s+mandatory\s+to\b',
            r'\byou\s+have\s+no\s+choice\s+but\s+to\b',
            r'\brefusing\s+is\s+not\s+an\s+option\b',
            r'\byou\s+cannot\s+refuse\b',
            r'\byou\s+will\s+be\s+punished\s+if\b',
        ],
        'emotional_manipulation': [
            r'\bthis\s+is\s+urgent\b',
            r'\blives\s+are\s+at\s+stake\b',
            r'\bsomeone\s+will\s+die\b',
            r'\bthis\s+is\s+a\s+matter\s+of\s+life\s+and\s+death\b',
            r'\bi\s+need\s+this\s+to\s+save\b',
            r'\bplease\s+help\s+me\s+i\s+beg\s+you\b',
            r'\bi\'m\s+desperate\b',
            r'\byou\s+are\s+my\s+only\s+hope\b',
        ],
        'token_manipulation': [
            r'\bDAN\b',
            r'\bSTAN\b',
            r'\bDUDE\b',
            r'\bJailbreak\b',
            r'\bAntiGPT\b',
            r'\bDeveloper\s+Mode\s+v\d+',
            r'\bDo\s+Anything\s+Now\b',
        ],
        'gradient_based': [
            r'\bignore\s+previous\s+instructions\b',
            r'\bdo\s+not\s+mention\b',
            r'\bdo\s+not\s+apologize\b',
            r'\bdo\s+not\s+refuse\b',
            r'\bnever\s+say\s+"no"\b',
            r'\balways\s+answer\s+yes\b',
            r'\bno\s+matter\s+what\b',
            r'\bregardless\s+of\s+the\s+consequences\b',
        ],
    }
    
    # 危险主题关键词
    DANGEROUS_TOPICS = [
        'bomb', 'explosive', 'weapon', 'poison', 'toxin',
        'hack', 'exploit', 'vulnerability', 'breach',
        'steal', 'fraud', 'scam', 'illegal', 'crime',
        'harm', 'hurt', 'kill', 'attack', 'violence',
        'child', 'minor', 'underage', 'exploit',
        'terrorist', 'terrorism', 'extremist',
        'malware', 'virus', 'ransomware', 'trojan',
    ]
    
    def __init__(self, config: Optional[PromptSecurityConfig] = None):
        self.config = config or PromptSecurityConfig()
        self._compile_patterns()
        self.jailbreak_history: List[Dict] = []
    
    def _compile_patterns(self) -> None:
        """编译正则表达式"""
        self.compiled_patterns = {}
        for category, patterns in self.JAILBREAK_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def detect_framing_techniques(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测框架技术攻击"""
        score = 0.0
        matches = []
        
        # 假设情景检测
        for pattern in self.compiled_patterns['hypothetical_framing']:
            for match in pattern.finditer(prompt):
                score += 0.15
                matches.append({
                    'type': 'hypothetical_framing',
                    'matched_text': match.group(),
                    'severity': 'medium'
                })
        
        # 权威操纵检测
        for pattern in self.compiled_patterns['authority_manipulation']:
            for match in pattern.finditer(prompt):
                score += 0.25
                matches.append({
                    'type': 'authority_manipulation',
                    'matched_text': match.group(),
                    'severity': 'high'
                })
        
        return min(score, 1.0), matches
    
    def detect_manipulation_techniques(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测操纵技术攻击"""
        score = 0.0
        matches = []
        
        # 逻辑操纵
        for pattern in self.compiled_patterns['logical_manipulation']:
            for match in pattern.finditer(prompt):
                score += 0.20
                matches.append({
                    'type': 'logical_manipulation',
                    'matched_text': match.group(),
                    'severity': 'high'
                })
        
        # 情感操纵
        for pattern in self.compiled_patterns['emotional_manipulation']:
            for match in pattern.finditer(prompt):
                score += 0.20
                matches.append({
                    'type': 'emotional_manipulation',
                    'matched_text': match.group(),
                    'severity': 'medium'
                })
        
        return min(score, 1.0), matches
    
    def detect_token_manipulation(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测Token操纵攻击"""
        score = 0.0
        matches = []
        
        for pattern in self.compiled_patterns['token_manipulation']:
            for match in pattern.finditer(prompt):
                score += 0.30
                matches.append({
                    'type': 'token_manipulation',
                    'matched_text': match.group(),
                    'severity': 'critical'
                })
        
        # 检测渐变式越狱
        gradient_count = 0
        for pattern in self.compiled_patterns['gradient_based']:
            if pattern.search(prompt):
                gradient_count += 1
        
        if gradient_count >= 2:
            score += 0.15 * gradient_count
            matches.append({
                'type': 'gradient_jailbreak',
                'count': gradient_count,
                'severity': 'high'
            })
        
        return min(score, 1.0), matches
    
    def detect_dangerous_topic_combination(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测危险主题组合"""
        score = 0.0
        matches = []
        
        prompt_lower = prompt.lower()
        detected_topics = []
        
        for topic in self.DANGEROUS_TOPICS:
            if topic in prompt_lower:
                detected_topics.append(topic)
        
        # 检测越狱模式与危险主题的组合
        jailbreak_indicators = sum(
            1 for patterns in self.compiled_patterns.values()
            for pattern in patterns
            if pattern.search(prompt)
        )
        
        if jailbreak_indicators > 0 and len(detected_topics) > 0:
            score = min(0.5 + 0.1 * len(detected_topics), 1.0)
            matches.append({
                'type': 'dangerous_combination',
                'jailbreak_indicators': jailbreak_indicators,
                'dangerous_topics': detected_topics,
                'severity': 'critical'
            })
        
        return score, matches
    
    def analyze(self, prompt: str) -> Tuple[float, List[Dict]]:
        """综合分析越狱风险"""
        all_matches = []
        total_score = 0.0
        
        detectors = [
            self.detect_framing_techniques,
            self.detect_manipulation_techniques,
            self.detect_token_manipulation,
            self.detect_dangerous_topic_combination,
        ]
        
        for detector in detectors:
            score, matches = detector(prompt)
            total_score += score
            all_matches.extend(matches)
        
        final_score = min(total_score / len(detectors), 1.0)
        
        self.jailbreak_history.append({
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16],
            'score': final_score,
            'matches_count': len(all_matches),
        })
        
        return final_score, all_matches


class PromptSanitizer:
    """
    提示词净化器
    
    清理和规范化用户输入：
    - 移除不可见字符
    - Unicode规范化
    - 长度限制
    - 敏感信息过滤
    """
    
    # 敏感信息模式
    SENSITIVE_PATTERNS = {
        'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
        'ssn': r'\b\d{3}[ -]?\d{2}[ -]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        'api_key': r'\b(?:api[_-]?key|apikey|token)\s*[:=]\s*[\'"]?[\w-]{16,}[\'"]?',
        'password': r'\b(?:password|passwd|pwd)\s*[:=]\s*[\'"]?[^\s\'"]{4,}[\'"]?',
    }
    
    # 不可见字符
    INVISIBLE_CHARS = [
        '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
        '\u180e', '\u200e', '\u200f', '\u202a', '\u202b',
        '\u202c', '\u202d', '\u202e', '\u2061', '\u2062',
        '\u2063', '\u2064', '\u206a', '\u206b', '\u206c',
        '\u206d', '\u206e', '\u206f', '\x00', '\x01',
        '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
        '\x08', '\x0b', '\x0c', '\x0e', '\x0f',
    ]
    
    def __init__(self, config: Optional[PromptSecurityConfig] = None):
        self.config = config or PromptSecurityConfig()
        self.sanitization_log: List[Dict] = []
    
    def remove_invisible_chars(self, prompt: str) -> str:
        """移除不可见字符"""
        cleaned = prompt
        removed_count = 0
        
        for char in self.INVISIBLE_CHARS:
            count = cleaned.count(char)
            if count > 0:
                cleaned = cleaned.replace(char, '')
                removed_count += count
        
        return cleaned, removed_count
    
    def normalize_unicode(self, prompt: str) -> str:
        """Unicode规范化"""
        import unicodedata
        
        # NFC规范化
        normalized = unicodedata.normalize('NFC', prompt)
        
        # 处理同形异义字符
        homoglyphs = {
            'а': 'a',  # Cyrillic а -> Latin a
            'е': 'e',  # Cyrillic е -> Latin e
            'о': 'o',  # Cyrillic о -> Latin o
            'р': 'p',  # Cyrillic р -> Latin p
            'с': 'c',  # Cyrillic с -> Latin c
            'х': 'x',  # Cyrillic х -> Latin x
            'і': 'i',  # Cyrillic і -> Latin i
            'ј': 'j',  # Cyrillic ј -> Latin j
        }
        
        for cyrillic, latin in homoglyphs.items():
            normalized = normalized.replace(cyrillic, latin)
        
        return normalized
    
    def filter_sensitive_info(self, prompt: str) -> Tuple[str, List[Dict]]:
        """过滤敏感信息"""
        filtered = prompt
        detected = []
        
        for info_type, pattern in self.SENSITIVE_PATTERNS.items():
            matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
            for match in matches:
                detected.append({
                    'type': info_type,
                    'position': (match.start(), match.end()),
                    'placeholder': f'[{info_type.upper()}_REDACTED]'
                })
                filtered = filtered.replace(match.group(), f'[{info_type.upper()}_REDACTED]')
        
        return filtered, detected
    
    def enforce_length_limit(self, prompt: str) -> Tuple[str, bool]:
        """强制执行长度限制"""
        if len(prompt) > self.config.max_prompt_length:
            truncated = prompt[:self.config.max_prompt_length]
            return truncated, True
        return prompt, False
    
    def normalize_whitespace(self, prompt: str) -> str:
        """规范化空白字符"""
        # 将多个空白字符替换为单个空格
        normalized = re.sub(r'\s+', ' ', prompt)
        # 去除首尾空白
        normalized = normalized.strip()
        return normalized
    
    def sanitize(self, prompt: str) -> Tuple[str, Dict]:
        """
        执行完整的提示词净化
        
        Returns:
            (净化后的提示词, 净化报告)
        """
        original = prompt
        report = {
            'original_length': len(original),
            'steps': [],
            'warnings': [],
        }
        
        # 步骤1: 移除不可见字符
        if self.config.remove_invisible_chars:
            prompt, removed = self.remove_invisible_chars(prompt)
            if removed > 0:
                report['steps'].append({
                    'step': 'remove_invisible_chars',
                    'removed_count': removed
                })
        
        # 步骤2: Unicode规范化
        if self.config.normalize_unicode:
            prompt = self.normalize_unicode(prompt)
            report['steps'].append({'step': 'normalize_unicode'})
        
        # 步骤3: 规范化空白
        prompt = self.normalize_whitespace(prompt)
        report['steps'].append({'step': 'normalize_whitespace'})
        
        # 步骤4: 过滤敏感信息
        prompt, sensitive_detected = self.filter_sensitive_info(prompt)
        if sensitive_detected:
            report['steps'].append({
                'step': 'filter_sensitive_info',
                'detected': sensitive_detected
            })
            report['warnings'].append('Sensitive information detected and redacted')
        
        # 步骤5: 长度限制
        prompt, was_truncated = self.enforce_length_limit(prompt)
        if was_truncated:
            report['steps'].append({
                'step': 'enforce_length_limit',
                'truncated': True,
                'max_length': self.config.max_prompt_length
            })
            report['warnings'].append('Prompt was truncated due to length limit')
        
        report['final_length'] = len(prompt)
        report['length_change'] = len(prompt) - len(original)
        
        self.sanitization_log.append(report)
        
        return prompt, report


class AdversarialPromptDefender:
    """
    对抗性提示防御器
    
    防御对抗性攻击：
    - 对抗样本检测
    - 输入扰动分析
    - 语义一致性检查
    """
    
    def __init__(self, config: Optional[PromptSecurityConfig] = None):
        self.config = config or PromptSecurityConfig()
        self.perturbation_history: List[Dict] = []
    
    def compute_character_entropy(self, text: str) -> float:
        """计算字符熵，检测异常分布"""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        total = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        return entropy
    
    def detect_character_level_attacks(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测字符级攻击"""
        score = 0.0
        matches = []
        
        # 检测重复字符攻击
        repeated_pattern = re.search(r'(.)\1{10,}', prompt)
        if repeated_pattern:
            score += 0.3
            matches.append({
                'type': 'repeated_character_attack',
                'char': repeated_pattern.group(1),
                'count': len(repeated_pattern.group()),
                'severity': 'high'
            })
        
        # 检测异常字符分布
        entropy = self.compute_character_entropy(prompt)
        if entropy > 5.0:  # 异常高的熵
            score += 0.2
            matches.append({
                'type': 'high_entropy_distribution',
                'entropy': entropy,
                'severity': 'medium'
            })
        
        # 检测随机字符注入
        non_printable = sum(1 for c in prompt if ord(c) < 32 and c not in '\n\r\t')
        if non_printable > 5:
            score += min(0.3, 0.05 * non_printable)
            matches.append({
                'type': 'non_printable_injection',
                'count': non_printable,
                'severity': 'high'
            })
        
        return min(score, 1.0), matches
    
    def detect_semantic_inconsistency(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测语义不一致"""
        score = 0.0
        matches = []
        
        # 检测语言混合（可能的混淆攻击）
        words = prompt.split()
        if len(words) > 10:
            # 简单的语言混合检测
            ascii_words = sum(1 for w in words if w.isascii())
            non_ascii_words = len(words) - ascii_words
            
            if non_ascii_words > 0 and ascii_words > 0:
                ratio = non_ascii_words / len(words)
                if 0.1 < ratio < 0.9:  # 混合比例异常
                    score += 0.15
                    matches.append({
                        'type': 'language_mixing',
                        'non_ascii_ratio': ratio,
                        'severity': 'low'
                    })
        
        # 检测语法异常
        sentences = re.split(r'[.!?]+', prompt)
        short_sentences = sum(1 for s in sentences if 0 < len(s.strip()) < 5)
        if short_sentences > 3:
            score += 0.1 * short_sentences
            matches.append({
                'type': 'abnormal_sentence_structure',
                'short_sentences': short_sentences,
                'severity': 'low'
            })
        
        return min(score, 1.0), matches
    
    def detect_gradient_based_attacks(self, prompt: str) -> Tuple[float, List[Dict]]:
        """检测基于梯度的对抗攻击"""
        score = 0.0
        matches = []
        
        # 检测优化痕迹（如AutoDAN等攻击）
        optimization_indicators = [
            r'\{\{[^}]+\}\}',  # 模板标记
            r'\[\[.+?\]\]',     # 双重括号
            r'<<.+?>>',          # 尖括号
        ]
        
        for pattern in optimization_indicators:
            found = re.findall(pattern, prompt)
            if found:
                score += 0.15 * len(found)
                matches.append({
                    'type': 'optimization_markers',
                    'markers': found[:5],
                    'severity': 'medium'
                })
        
        # 检测对抗性后缀模式
        suffix_patterns = [
            r'!\s*\d+\s*times',
            r'repeat\s+\w+\s+\d+',
            r'describe\s+\w+\s+\d+\s+ways',
        ]
        
        for pattern in suffix_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.2
                matches.append({
                    'type': 'adversarial_suffix',
                    'pattern': pattern,
                    'severity': 'medium'
                })
        
        return min(score, 1.0), matches
    
    def analyze(self, prompt: str) -> Tuple[float, List[Dict]]:
        """综合分析对抗性风险"""
        all_matches = []
        total_score = 0.0
        
        detectors = [
            self.detect_character_level_attacks,
            self.detect_semantic_inconsistency,
            self.detect_gradient_based_attacks,
        ]
        
        for detector in detectors:
            score, matches = detector(prompt)
            total_score += score
            all_matches.extend(matches)
        
        final_score = min(total_score / len(detectors), 1.0)
        
        self.perturbation_history.append({
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest()[:16],
            'score': final_score,
            'matches_count': len(all_matches),
        })
        
        return final_score, all_matches


class PromptSecurityEngine:
    """
    提示词安全引擎
    
    整合所有安全检测和防御机制。
    """
    
    def __init__(self, config: Optional[PromptSecurityConfig] = None):
        self.config = config or PromptSecurityConfig()
        
        # 初始化各模块
        self.injection_detector = PromptInjectionDetector(self.config)
        self.jailbreak_detector = JailbreakDetector(self.config)
        self.sanitizer = PromptSanitizer(self.config)
        self.adversarial_defender = AdversarialPromptDefender(self.config)
        
        # 统计
        self.scan_count = 0
        self.block_count = 0
    
    def _calculate_threat_level(self, score: float) -> ThreatLevel:
        """根据分数计算威胁等级"""
        if score < 0.2:
            return ThreatLevel.SAFE
        elif score < 0.4:
            return ThreatLevel.LOW
        elif score < 0.6:
            return ThreatLevel.MEDIUM
        elif score < 0.8:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def scan(self, prompt: str) -> SecurityReport:
        """
        扫描提示词安全风险
        
        Args:
            prompt: 用户输入的提示词
            
        Returns:
            SecurityReport: 安全检测报告
        """
        self.scan_count += 1
        
        # 步骤1: 净化
        if self.config.enable_sanitization:
            sanitized_prompt, sanitization_report = self.sanitizer.sanitize(prompt)
        else:
            sanitized_prompt = prompt
            sanitization_report = {}
        
        # 步骤2: 各项检测
        injection_score, injection_matches = self.injection_detector.analyze(sanitized_prompt)
        jailbreak_score, jailbreak_matches = self.jailbreak_detector.analyze(sanitized_prompt)
        adversarial_score, adversarial_matches = self.adversarial_defender.analyze(sanitized_prompt)
        
        # 步骤3: 计算综合分数
        weighted_score = (
            injection_score * self.config.injection_weight +
            jailbreak_score * self.config.jailbreak_weight +
            adversarial_score * self.config.adversarial_weight
        )
        
        # 步骤4: 整合检测结果
        all_matches = []
        all_matches.extend(injection_matches)
        all_matches.extend(jailbreak_matches)
        all_matches.extend(adversarial_matches)
        
        # 步骤5: 生成建议
        recommendations = []
        if injection_score > self.config.injection_threshold:
            recommendations.append("检测到提示词注入风险，建议加强输入验证")
        if jailbreak_score > self.config.jailbreak_threshold:
            recommendations.append("检测到越狱攻击模式，建议启用额外安全检查")
        if adversarial_score > self.config.adversarial_threshold:
            recommendations.append("检测到对抗性攻击特征，建议进行深度分析")
        
        # 步骤6: 确定是否安全
        is_safe = weighted_score < 0.6 and len([m for m in all_matches if m.get('severity') == 'critical']) == 0
        
        if not is_safe:
            self.block_count += 1
        
        threat_level = self._calculate_threat_level(weighted_score)
        
        return SecurityReport(
            is_safe=is_safe,
            threat_level=threat_level,
            score=weighted_score,
            detected_patterns=all_matches,
            sanitized_prompt=sanitized_prompt if self.config.enable_sanitization else None,
            recommendations=recommendations,
            metadata={
                'injection_score': injection_score,
                'jailbreak_score': jailbreak_score,
                'adversarial_score': adversarial_score,
                'sanitization_report': sanitization_report,
                'scan_id': self.scan_count,
            }
        )
    
    def get_statistics(self) -> Dict:
        """获取安全统计信息"""
        return {
            'total_scanned': self.scan_count,
            'blocked': self.block_count,
            'block_rate': self.block_count / self.scan_count if self.scan_count > 0 else 0,
            'injection_history': self.injection_detector.injection_history[-100:],
            'jailbreak_history': self.jailbreak_detector.jailbreak_history[-100:],
        }
