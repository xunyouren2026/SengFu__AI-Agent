"""
Gateway模块 - 透明代理网关与规则引擎
"""
from .proxy import (
    TransparentProxy,
    RequestInterceptor,
    HTTPRequest,
    InterceptResult,
    InterceptAction,
    ProxyChain,
    RateLimiter
)
from .rule_engine import (
    RuleEngine,
    Rule,
    RuleBuilder,
    RuleTemplates,
    Condition,
    ConditionGroup,
    ConditionOperator,
    LogicalOperator,
    Action,
    ActionType
)

__all__ = [
    # proxy.py
    "TransparentProxy",
    "RequestInterceptor",
    "HTTPRequest",
    "InterceptResult",
    "InterceptAction",
    "ProxyChain",
    "RateLimiter",
    # rule_engine.py
    "RuleEngine",
    "Rule",
    "RuleBuilder",
    "RuleTemplates",
    "Condition",
    "ConditionGroup",
    "ConditionOperator",
    "LogicalOperator",
    "Action",
    "ActionType"
]
