"""
模型路由器 (Model Router)

根据请求特征（模型名称、任务类型、参数大小等）将请求路由到最合适的后端。
支持基于规则的路由、正则匹配、标签匹配等策略。

模块路径: compat/unified/model_router.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """路由策略类型。"""

    EXACT_MATCH = "exact_match"
    REGEX_MATCH = "regex_match"
    PREFIX_MATCH = "prefix_match"
    TAG_MATCH = "tag_match"
    CUSTOM = "custom"
    DEFAULT = "default"


class TaskType(str, Enum):
    """任务类型。"""

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    IMAGE_GENERATION = "image_generation"
    ANY = "any"


@dataclass
class RoutingRule:
    """路由规则。

    Attributes:
        name: 规则名称。
        strategy: 路由策略。
        pattern: 匹配模式（模型名称、正则表达式、前缀等）。
        backend_name: 目标后端名称。
        priority: 规则优先级（数字越小越优先）。
        task_types: 适用任务类型列表。
        tags: 匹配标签列表。
        max_input_tokens: 最大输入 token 数限制。
        custom_matcher: 自定义匹配函数。
        enabled: 是否启用。
        metadata: 额外元数据。
    """

    name: str
    strategy: RoutingStrategy = RoutingStrategy.EXACT_MATCH
    pattern: str = ""
    backend_name: str = ""
    priority: int = 0
    task_types: List[TaskType] = field(default_factory=lambda: [TaskType.ANY])
    tags: Set[str] = field(default_factory=set)
    max_input_tokens: int = 0
    custom_matcher: Optional[Callable[[Dict[str, Any]], bool]] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        model_name: str = "",
        task_type: Optional[TaskType] = None,
        tags: Optional[Set[str]] = None,
        input_tokens: int = 0,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """判断请求是否匹配此规则。

        Args:
            model_name: 请求的模型名称。
            task_type: 任务类型。
            tags: 请求标签集合。
            input_tokens: 输入 token 数。
            request_context: 额外请求上下文。

        Returns:
            是否匹配。
        """
        if not self.enabled:
            return False

        if task_type is not None and TaskType.ANY not in self.task_types:
            if task_type not in self.task_types:
                return False

        if self.max_input_tokens > 0 and input_tokens > self.max_input_tokens:
            return False

        if tags and self.tags:
            if not self.tags.intersection(tags):
                return False

        if self.strategy == RoutingStrategy.EXACT_MATCH:
            return model_name == self.pattern

        if self.strategy == RoutingStrategy.PREFIX_MATCH:
            return model_name.startswith(self.pattern)

        if self.strategy == RoutingStrategy.REGEX_MATCH:
            try:
                return bool(re.search(self.pattern, model_name))
            except re.error:
                logger.warning("路由规则 '%s' 的正则表达式无效: %s", self.name, self.pattern)
                return False

        if self.strategy == RoutingStrategy.TAG_MATCH:
            return bool(tags and self.tags.issubset(tags))

        if self.strategy == RoutingStrategy.CUSTOM:
            if self.custom_matcher and request_context:
                try:
                    return self.custom_matcher(request_context)
                except Exception as exc:
                    logger.warning("自定义路由匹配器异常: %s", exc)
                    return False
            return False

        if self.strategy == RoutingStrategy.DEFAULT:
            return True

        return False


@dataclass
class RoutingResult:
    """路由结果。

    Attributes:
        backend_name: 目标后端名称。
        rule_name: 匹配的规则名称。
        model_name: 解析后的模型名称。
        task_type: 任务类型。
        confidence: 路由置信度（0-1）。
    """

    backend_name: str
    rule_name: str = ""
    model_name: str = ""
    task_type: TaskType = TaskType.ANY
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "backend_name": self.backend_name,
            "rule_name": self.rule_name,
            "model_name": self.model_name,
            "task_type": self.task_type.value,
            "confidence": self.confidence,
        }


class ModelRouter:
    """模型路由器。

    根据请求特征将推理请求路由到最合适的后端。

    Args:
        default_backend: 默认后端名称。
    """

    def __init__(
        self,
        default_backend: str = "default",
    ) -> None:
        self._default_backend = default_backend
        self._rules: List[RoutingRule] = []
        self._backend_capabilities: Dict[str, Dict[str, Any]] = {}

    @property
    def default_backend(self) -> str:
        """获取默认后端名称。"""
        return self._default_backend

    @default_backend.setter
    def default_backend(self, value: str) -> None:
        """设置默认后端名称。"""
        self._default_backend = value

    @property
    def rules(self) -> List[RoutingRule]:
        """获取所有路由规则。"""
        return list(self._rules)

    def add_rule(self, rule: RoutingRule) -> None:
        """添加路由规则。

        Args:
            rule: 路由规则。
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)
        logger.info("添加路由规则: %s -> %s (优先级=%d)", rule.name, rule.backend_name, rule.priority)

    def remove_rule(self, name: str) -> bool:
        """移除路由规则。

        Args:
            name: 规则名称。

        Returns:
            是否成功移除。
        """
        original_len = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < original_len

    def register_backend(self, name: str, capabilities: Dict[str, Any]) -> None:
        """注册后端及其能力。

        Args:
            name: 后端名称。
            capabilities: 能力描述（支持的模型、任务类型等）。
        """
        self._backend_capabilities[name] = capabilities
        logger.info("注册后端: %s, 能力: %s", name, list(capabilities.keys()))

    def unregister_backend(self, name: str) -> None:
        """取消注册后端。

        Args:
            name: 后端名称。
        """
        self._backend_capabilities.pop(name, None)
        logger.info("取消注册后端: %s", name)

    def get_backend_capabilities(self, name: str) -> Optional[Dict[str, Any]]:
        """获取后端能力。

        Args:
            name: 后端名称。

        Returns:
            能力描述，或 None。
        """
        return self._backend_capabilities.get(name)

    def route(
        self,
        model_name: str = "",
        task_type: Optional[TaskType] = None,
        tags: Optional[Set[str]] = None,
        input_tokens: int = 0,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> RoutingResult:
        """路由请求到合适的后端。

        按优先级依次匹配规则，返回第一个匹配结果。

        Args:
            model_name: 模型名称。
            task_type: 任务类型。
            tags: 请求标签。
            input_tokens: 输入 token 数。
            request_context: 额外请求上下文。

        Returns:
            路由结果。
        """
        for rule in self._rules:
            if rule.matches(model_name, task_type, tags, input_tokens, request_context):
                logger.debug(
                    "路由匹配: model=%s, rule=%s, backend=%s",
                    model_name, rule.name, rule.backend_name,
                )
                return RoutingResult(
                    backend_name=rule.backend_name,
                    rule_name=rule.name,
                    model_name=model_name,
                    task_type=task_type or TaskType.ANY,
                    confidence=1.0,
                )

        logger.debug("未匹配路由规则，使用默认后端: %s", self._default_backend)
        return RoutingResult(
            backend_name=self._default_backend,
            rule_name="default",
            model_name=model_name,
            task_type=task_type or TaskType.ANY,
            confidence=0.5,
        )

    def find_backends_for_model(self, model_name: str) -> List[str]:
        """查找支持指定模型的所有后端。

        Args:
            model_name: 模型名称。

        Returns:
            支持该模型的后端名称列表。
        """
        backends: List[str] = []
        for name, caps in self._backend_capabilities.items():
            supported_models: List[str] = caps.get("supported_models", [])
            if model_name in supported_models or not supported_models:
                backends.append(name)
        return backends

    def get_routing_table(self) -> List[Dict[str, Any]]:
        """获取当前路由表。

        Returns:
            路由规则列表。
        """
        return [
            {
                "name": rule.name,
                "strategy": rule.strategy.value,
                "pattern": rule.pattern,
                "backend": rule.backend_name,
                "priority": rule.priority,
                "task_types": [t.value for t in rule.task_types],
                "enabled": rule.enabled,
            }
            for rule in self._rules
        ]

    def validate(self) -> List[str]:
        """验证路由配置。

        Returns:
            问题列表（空列表表示配置有效）。
        """
        issues: List[str] = []
        backend_names = set(self._backend_capabilities.keys())

        for rule in self._rules:
            if rule.backend_name and rule.backend_name not in backend_names:
                issues.append(f"规则 '{rule.name}' 引用了未注册的后端 '{rule.backend_name}'")

            if rule.strategy == RoutingStrategy.REGEX_MATCH:
                try:
                    re.compile(rule.pattern)
                except re.error as exc:
                    issues.append(f"规则 '{rule.name}' 的正则表达式无效: {exc}")

        if not self._default_backend:
            issues.append("未设置默认后端")

        return issues
