"""
消息路由模块

提供动态消息路由功能，支持正则匹配、前缀匹配和精确匹配。
"""

import fnmatch
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern

from .interface import Message, MessageHandler

logger = logging.getLogger(__name__)


class MatchType:
    """匹配类型常量"""
    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"
    WILDCARD = "wildcard"


@dataclass
class RouteRule:
    """
    路由规则

    定义消息路由的匹配规则和处理逻辑。

    Attributes:
        rule_id: 规则唯一标识
        pattern: 匹配模式
        match_type: 匹配类型（exact/prefix/regex/wildcard）
        handler: 消息处理函数
        priority: 优先级（数值越小优先级越高）
        metadata: 规则元数据
        enabled: 是否启用
        compiled_regex: 编译后的正则表达式（仅regex类型）
    """

    rule_id: str = ""
    pattern: str = ""
    match_type: str = MatchType.EXACT
    handler: Optional[MessageHandler] = None
    priority: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    compiled_regex: Optional[Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.rule_id:
            import uuid
            self.rule_id = str(uuid.uuid4())
        if self.match_type == MatchType.REGEX and self.pattern:
            try:
                self.compiled_regex = re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(
                    f"无效的正则表达式: {self.pattern}: {exc}"
                )

    def matches(self, topic: str) -> bool:
        """
        检查主题是否匹配此规则

        Args:
            topic: 消息主题

        Returns:
            是否匹配
        """
        if not self.enabled:
            return False

        if self.match_type == MatchType.EXACT:
            return topic == self.pattern

        elif self.match_type == MatchType.PREFIX:
            return topic.startswith(self.pattern)

        elif self.match_type == MatchType.REGEX:
            if self.compiled_regex is None:
                return False
            return bool(self.compiled_regex.search(topic))

        elif self.match_type == MatchType.WILDCARD:
            return fnmatch.fnmatch(topic, self.pattern)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "pattern": self.pattern,
            "match_type": self.match_type,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "has_handler": self.handler is not None,
        }


class MessageRouter:
    """
    动态消息路由器

    根据消息的主题和头部信息，将消息路由到对应的处理函数。
    支持多种匹配方式和优先级排序。

    Usage:
        router = MessageRouter()

        # 精确匹配
        router.add_route("user.created", handle_user_created, match_type="exact")

        # 前缀匹配
        router.add_route("events.", handle_all_events, match_type="prefix")

        # 正则匹配
        router.add_route(r"order\.\d+", handle_order, match_type="regex")

        # 通配符匹配
        router.add_route("sensor.*", handle_sensor, match_type="wildcard")

        # 路由消息
        router.route_message(message)
    """

    def __init__(self, default_handler: Optional[MessageHandler] = None):
        """
        初始化消息路由器

        Args:
            default_handler: 默认处理函数（无匹配规则时使用）
        """
        self._rules: List[RouteRule] = []
        self._default_handler = default_handler
        self._lock = threading.RLock()
        self._total_routed: int = 0
        self._total_matched: int = 0
        self._total_unmatched: int = 0
        self._total_errors: int = 0

    def add_route(
        self,
        pattern: str,
        handler: MessageHandler,
        match_type: str = MatchType.EXACT,
        priority: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        rule_id: Optional[str] = None,
    ) -> str:
        """
        添加路由规则

        Args:
            pattern: 匹配模式
            handler: 处理函数
            match_type: 匹配类型
            priority: 优先级（数值越小优先级越高）
            metadata: 规则元数据
            rule_id: 自定义规则ID

        Returns:
            规则ID

        Raises:
            ValueError: 匹配类型无效或正则表达式无效
            TypeError: handler不可调用
        """
        if match_type not in (MatchType.EXACT, MatchType.PREFIX,
                              MatchType.REGEX, MatchType.WILDCARD):
            raise ValueError(
                f"无效的匹配类型: {match_type}，"
                f"可选值: {MatchType.EXACT}, {MatchType.PREFIX}, "
                f"{MatchType.REGEX}, {MatchType.WILDCARD}"
            )
        if not callable(handler):
            raise TypeError("handler 必须是可调用对象")

        rule = RouteRule(
            rule_id=rule_id or "",
            pattern=pattern,
            match_type=match_type,
            handler=handler,
            priority=priority,
            metadata=metadata or {},
        )

        with self._lock:
            self._rules.append(rule)
            # 按优先级排序（数值小的优先）
            self._rules.sort(key=lambda r: r.priority)

        logger.debug(
            "已添加路由规则: id=%s, pattern=%s, type=%s, priority=%d",
            rule.rule_id,
            pattern,
            match_type,
            priority,
        )
        return rule.rule_id

    def remove_route(self, rule_id: str) -> bool:
        """
        移除路由规则

        Args:
            rule_id: 规则ID

        Returns:
            是否成功移除
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.rule_id == rule_id:
                    self._rules.pop(i)
                    logger.debug("已移除路由规则: id=%s", rule_id)
                    return True
        return False

    def route_message(self, message: Message) -> Any:
        """
        根据路由规则分发消息

        按优先级顺序匹配规则，第一个匹配的规则将被执行。
        如果没有匹配的规则且存在默认处理器，则使用默认处理器。

        Args:
            message: 要路由的消息

        Returns:
            处理函数的返回值

        Raises:
            RuntimeError: 没有匹配的规则且没有默认处理器
        """
        with self._lock:
            self._total_routed += 1

        matched_rule = self._find_matching_rule(message)

        if matched_rule is not None:
            with self._lock:
                self._total_matched += 1
            try:
                result = matched_rule.handler(message)
                return result
            except Exception as exc:
                with self._lock:
                    self._total_errors += 1
                logger.error(
                    "路由处理错误: rule_id=%s, topic=%s, error=%s",
                    matched_rule.rule_id,
                    message.topic,
                    exc,
                )
                raise
        else:
            with self._lock:
                self._total_unmatched += 1

            if self._default_handler is not None:
                try:
                    return self._default_handler(message)
                except Exception as exc:
                    with self._lock:
                        self._total_errors += 1
                    raise

            logger.warning(
                "消息未匹配任何路由规则: topic=%s", message.topic
            )
            return None

    def _find_matching_rule(self, message: Message) -> Optional[RouteRule]:
        """
        查找匹配消息的路由规则

        首先检查消息头部中的 route_to 字段进行精确匹配，
        然后按优先级顺序检查所有规则。

        Args:
            message: 消息对象

        Returns:
            匹配的路由规则或None
        """
        # 检查消息头部中的显式路由指令
        explicit_route = message.headers.get("route_to")
        if explicit_route:
            for rule in self._rules:
                if (rule.enabled
                        and rule.match_type == MatchType.EXACT
                        and rule.pattern == explicit_route):
                    return rule

        # 按优先级匹配
        for rule in self._rules:
            if rule.matches(message.topic):
                return rule

        return None

    def get_rules(self, match_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取路由规则列表

        Args:
            match_type: 按匹配类型过滤

        Returns:
            规则信息列表
        """
        with self._lock:
            rules = self._rules
            if match_type:
                rules = [r for r in rules if r.match_type == match_type]
            return [r.to_dict() for r in rules]

    def enable_rule(self, rule_id: str) -> bool:
        """启用路由规则"""
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    rule.enabled = True
                    return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """禁用路由规则"""
        with self._lock:
            for rule in self._rules:
                if rule.rule_id == rule_id:
                    rule.enabled = False
                    return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取路由器统计信息"""
        with self._lock:
            return {
                "total_rules": len(self._rules),
                "enabled_rules": sum(1 for r in self._rules if r.enabled),
                "disabled_rules": sum(1 for r in self._rules if not r.enabled),
                "total_routed": self._total_routed,
                "total_matched": self._total_matched,
                "total_unmatched": self._total_unmatched,
                "total_errors": self._total_errors,
                "has_default_handler": self._default_handler is not None,
                "match_rate": (
                    self._total_matched / self._total_routed
                    if self._total_routed > 0 else 0.0
                ),
            }

    def clear(self) -> int:
        """
        清除所有路由规则

        Returns:
            清除的规则数量
        """
        with self._lock:
            count = len(self._rules)
            self._rules.clear()
            return count

    def __repr__(self) -> str:
        return (
            f"MessageRouter(rules={len(self._rules)}, "
            f"routed={self._total_routed})"
        )
