"""
敏感数据过滤器模块

提供日志中的敏感数据脱敏功能，支持API Key、邮箱、手机号、
信用卡号、IP地址等多种模式的自动检测和替换。
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple


# ---------------------------------------------------------------------------
# 敏感模式定义
# ---------------------------------------------------------------------------

@dataclass
class SensitivePattern:
    """敏感模式定义。

    Attributes:
        name: 模式名称（用于标识和调试）
        regex: 正则表达式模式字符串
        replacement_func: 替换函数，接收匹配对象，返回替换字符串
        enabled: 是否启用
    """
    name: str
    regex: str
    replacement_func: Callable[[re.Match], str]
    enabled: bool = True


# ---------------------------------------------------------------------------
# 敏感数据过滤器
# ---------------------------------------------------------------------------

class SensitiveDataFilter:
    """敏感数据脱敏过滤器。

    自动检测并脱敏日志消息中的敏感信息，包括：
    - API Key（sk-xxx, key-xxx 等模式）
    - 邮箱地址
    - 手机号码
    - 信用卡号
    - IP地址
    - 自定义模式

    使用示例::

        filter = SensitiveDataFilter()
        safe_msg = filter.filter("api_key=sk-abc123def456 and email=test@example.com")
        # => "api_key=sk-***456 and email=t***@example.com"
    """

    def __init__(self, enabled: bool = True):
        """初始化敏感数据过滤器。

        Args:
            enabled: 是否全局启用过滤
        """
        self._enabled = enabled
        self._patterns: List[SensitivePattern] = []
        self._lock = threading.Lock()
        self._compiled_cache: Dict[str, Pattern] = {}
        self._init_default_patterns()

    # ------------------------------------------------------------------
    # 默认模式注册
    # ------------------------------------------------------------------

    def _init_default_patterns(self) -> None:
        """注册默认的敏感数据模式。"""
        # API Key 模式: sk-xxx, key-xxx, api_key=xxx, Bearer xxx 等
        self._patterns.append(SensitivePattern(
            name="api_key_sk",
            regex=r'(sk-[a-zA-Z0-9]{8,})([a-zA-Z0-9]{4})',
            replacement_func=lambda m: f'{m.group(1)[:3]}{"*" * (len(m.group(1)) - 3)}{m.group(2)}',
        ))
        self._patterns.append(SensitivePattern(
            name="api_key_prefix",
            regex=r'((?:key|api[_-]?key|apikey|secret|token|password|passwd)\s*[=:]\s*)'
                  r'(["\']?)([a-zA-Z0-9_\-]{8,})(\2)',
            replacement_func=lambda m: (
                f'{m.group(1)}{m.group(2)}'
                f'{m.group(3)[:3]}{"*" * max(len(m.group(3)) - 6, 3)}{m.group(3)[-3:]}'
                f'{m.group(4)}'
            ),
        ))
        self._patterns.append(SensitivePattern(
            name="bearer_token",
            regex=r'(Bearer\s+)([a-zA-Z0-9_\-\.]{16,})',
            replacement_func=lambda m: f'{m.group(1)}{"*" * min(len(m.group(2)), 20)}',
        ))

        # 邮箱模式
        self._patterns.append(SensitivePattern(
            name="email",
            regex=r'([a-zA-Z0-9._%+\-])([a-zA-Z0-9._%+\-]{1,})@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})',
            replacement_func=lambda m: (
                f'{m.group(1)}{"*" * len(m.group(2))}@{m.group(3)}'
            ),
        ))

        # 中国手机号模式: 1开头的11位数字
        self._patterns.append(SensitivePattern(
            name="phone_cn",
            regex=r'(1[3-9]\d)(\d{4})(\d{4})',
            replacement_func=lambda m: f'{m.group(1)}****{m.group(3)}',
        ))

        # 国际手机号模式（带区号）
        self._patterns.append(SensitivePattern(
            name="phone_intl",
            regex=r'(\+\d{1,3}[-\s]?)\d{4,}(\d{2})',
            replacement_func=lambda m: f'{m.group(1)}{"*" * 6}{m.group(2)}',
        ))

        # 信用卡号模式: 4-6位一组，共13-19位数字
        self._patterns.append(SensitivePattern(
            name="credit_card",
            regex=r'\b([0-9]{4})[ -]?([0-9]{4})[ -]?([0-9]{4})[ -]?([0-9]{4})\b',
            replacement_func=lambda m: f'{m.group(1)}-****-****-{m.group(4)}',
        ))

        # IPv4 地址模式
        self._patterns.append(SensitivePattern(
            name="ipv4",
            regex=r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b',
            replacement_func=lambda m: self._mask_ipv4_match(m),
        ))

        # IPv6 地址模式（简化版）
        self._patterns.append(SensitivePattern(
            name="ipv6",
            regex=r'\b([0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b',
            replacement_func=lambda m: self._mask_ipv6_match(m),
        ))

    # ------------------------------------------------------------------
    # 脱敏方法
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_ipv4_match(m: re.Match) -> str:
        """脱敏IPv4地址，保留首尾段。

        Args:
            m: 正则匹配对象

        Returns:
            脱敏后的IP字符串
        """
        return f'{m.group(1)}.***.***.{m.group(4)}'

    @staticmethod
    def _mask_ipv6_match(m: re.Match) -> str:
        """脱敏IPv6地址。

        Args:
            m: 正则匹配对象

        Returns:
            脱敏后的IPv6字符串
        """
        full = m.group(0)
        parts = full.split(':')
        if len(parts) >= 4:
            return f'{parts[0]}:****:****:{parts[-1]}'
        return '****'

    def mask_api_key(self, text: str) -> str:
        """对文本中的API Key进行脱敏。

        支持的模式: sk-xxx, key-xxx, Bearer xxx, api_key=xxx 等。

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text
        for pattern in self._patterns:
            if pattern.enabled and pattern.name.startswith("api_key") or pattern.name == "bearer_token":
                text = self._apply_pattern(text, pattern)
        return text

    def mask_email(self, text: str) -> str:
        """对文本中的邮箱地址进行脱敏。

        格式: u***@example.com

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text
        for pattern in self._patterns:
            if pattern.enabled and pattern.name == "email":
                text = self._apply_pattern(text, pattern)
        return text

    def mask_phone(self, text: str) -> str:
        """对文本中的手机号进行脱敏。

        中国手机号格式: 138****1234

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text
        for pattern in self._patterns:
            if pattern.enabled and pattern.name.startswith("phone"):
                text = self._apply_pattern(text, pattern)
        return text

    def mask_credit_card(self, text: str) -> str:
        """对文本中的信用卡号进行脱敏。

        格式: 4000-****-****-1234

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text
        for pattern in self._patterns:
            if pattern.enabled and pattern.name == "credit_card":
                text = self._apply_pattern(text, pattern)
        return text

    def mask_ip(self, text: str) -> str:
        """对文本中的IP地址进行脱敏。

        IPv4格式: 192.***.***.1
        IPv6格式: 2001:****:****::1

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled:
            return text
        for pattern in self._patterns:
            if pattern.enabled and pattern.name.startswith("ip"):
                text = self._apply_pattern(text, pattern)
        return text

    # ------------------------------------------------------------------
    # 自定义模式
    # ------------------------------------------------------------------

    def add_custom_pattern(
        self,
        name: str,
        regex: str,
        replacement_func: Callable[[re.Match], str],
        enabled: bool = True,
    ) -> None:
        """添加自定义脱敏规则。

        Args:
            name: 模式名称
            regex: 正则表达式
            replacement_func: 替换函数，接收re.Match对象，返回替换字符串
            enabled: 是否启用

        Raises:
            ValueError: 如果同名模式已存在
        """
        with self._lock:
            for p in self._patterns:
                if p.name == name:
                    raise ValueError(f"Pattern '{name}' already exists")
            self._patterns.append(SensitivePattern(
                name=name,
                regex=regex,
                replacement_func=replacement_func,
                enabled=enabled,
            ))
            # 清除编译缓存
            self._compiled_cache.pop(name, None)

    def remove_pattern(self, name: str) -> bool:
        """移除指定名称的脱敏模式。

        Args:
            name: 模式名称

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, p in enumerate(self._patterns):
                if p.name == name:
                    self._patterns.pop(i)
                    self._compiled_cache.pop(name, None)
                    return True
        return False

    def enable_pattern(self, name: str, enabled: bool = True) -> bool:
        """启用或禁用指定模式。

        Args:
            name: 模式名称
            enabled: 是否启用

        Returns:
            是否成功设置
        """
        with self._lock:
            for p in self._patterns:
                if p.name == name:
                    p.enabled = enabled
                    return True
        return False

    # ------------------------------------------------------------------
    # 核心过滤方法
    # ------------------------------------------------------------------

    def filter(self, text: str) -> str:
        """对文本执行所有已启用的脱敏规则。

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        if not self._enabled or not text:
            return text
        with self._lock:
            patterns = list(self._patterns)
        for pattern in patterns:
            if pattern.enabled:
                text = self._apply_pattern(text, pattern)
        return text

    def filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """对字典中的所有字符串值执行脱敏。

        Args:
            data: 原始字典

        Returns:
            脱敏后的字典
        """
        if not self._enabled:
            return data
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.filter(value)
            elif isinstance(value, dict):
                result[key] = self.filter_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.filter(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def _apply_pattern(self, text: str, pattern: SensitivePattern) -> str:
        """应用单个脱敏模式。

        Args:
            text: 原始文本
            pattern: 敏感模式

        Returns:
            脱敏后的文本
        """
        compiled = self._get_compiled(pattern.name, pattern.regex)
        try:
            return compiled.sub(pattern.replacement_func, text)
        except re.error:
            return text

    def _get_compiled(self, name: str, regex: str) -> Pattern:
        """获取编译后的正则表达式，带缓存。

        Args:
            name: 模式名称（用作缓存键）
            regex: 正则表达式字符串

        Returns:
            编译后的正则表达式对象
        """
        if name not in self._compiled_cache:
            self._compiled_cache[name] = re.compile(regex)
        return self._compiled_cache[name]

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """是否全局启用。"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """设置全局启用状态。"""
        self._enabled = value

    def get_patterns(self) -> List[Dict[str, Any]]:
        """获取所有已注册的模式信息。

        Returns:
            模式信息列表，每个元素包含 name, regex, enabled
        """
        with self._lock:
            return [
                {"name": p.name, "regex": p.regex, "enabled": p.enabled}
                for p in self._patterns
            ]

    def reset(self) -> None:
        """重置过滤器，清除所有自定义模式，恢复默认模式。"""
        with self._lock:
            self._patterns.clear()
            self._compiled_cache.clear()
        self._init_default_patterns()
