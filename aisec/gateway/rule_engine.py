"""
规则引擎 - 条件匹配与规则执行
"""
import re
import json
import time
from typing import Optional, Dict, List, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import operator
from functools import reduce


class ConditionOperator(Enum):
    """条件操作符"""
    EQ = "eq"           # 等于
    NE = "ne"           # 不等于
    GT = "gt"           # 大于
    GE = "ge"           # 大于等于
    LT = "lt"           # 小于
    LE = "le"           # 小于等于
    IN = "in"           # 包含
    NOT_IN = "not_in"   # 不包含
    CONTAINS = "contains"       # 字符串包含
    NOT_CONTAINS = "not_contains"  # 字符串不包含
    MATCHES = "matches"         # 正则匹配
    STARTS_WITH = "starts_with"  # 以...开始
    ENDS_WITH = "ends_with"     # 以...结束
    IS_EMPTY = "is_empty"       # 为空
    IS_NOT_EMPTY = "is_not_empty"  # 不为空
    BETWEEN = "between"         # 在范围内
    SIZE_EQ = "size_eq"         # 大小等于
    SIZE_GT = "size_gt"         # 大于指定大小
    SIZE_LT = "size_lt"         # 小于指定大小


class LogicalOperator(Enum):
    """逻辑操作符"""
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """动作类型"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    ALERT = "alert"
    MODIFY = "modify"
    REDIRECT = "redirect"
    THROTTLE = "throttle"
    CACHE = "cache"


@dataclass
class Condition:
    """条件"""
    field: str                           # 字段路径，如 "request.headers.content-type"
    operator: ConditionOperator          # 操作符
    value: Any                           # 比较值
    case_sensitive: bool = True          # 是否区分大小写
    negate: bool = False                 # 是否取反
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """评估条件"""
        # 获取字段值
        field_value = self._get_field_value(context, self.field)
        
        # 执行比较
        result = self._compare(field_value, self.value)
        
        # 取反
        return not result if self.negate else result
    
    def _get_field_value(self, context: Dict[str, Any], field_path: str) -> Any:
        """获取字段值"""
        keys = field_path.split('.')
        value = context
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None
            if value is None:
                return None
        return value
    
    def _compare(self, field_value: Any, compare_value: Any) -> bool:
        """执行比较操作"""
        op = self.operator
        
        # 字符串处理
        if isinstance(field_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            if isinstance(compare_value, str):
                compare_value = compare_value.lower()
        
        try:
            if op == ConditionOperator.EQ:
                return field_value == compare_value
            
            elif op == ConditionOperator.NE:
                return field_value != compare_value
            
            elif op == ConditionOperator.GT:
                return field_value > compare_value
            
            elif op == ConditionOperator.GE:
                return field_value >= compare_value
            
            elif op == ConditionOperator.LT:
                return field_value < compare_value
            
            elif op == ConditionOperator.LE:
                return field_value <= compare_value
            
            elif op == ConditionOperator.IN:
                return field_value in compare_value
            
            elif op == ConditionOperator.NOT_IN:
                return field_value not in compare_value
            
            elif op == ConditionOperator.CONTAINS:
                if isinstance(field_value, str) and isinstance(compare_value, str):
                    return compare_value in field_value
                elif isinstance(field_value, (list, dict)):
                    return compare_value in field_value
                return False
            
            elif op == ConditionOperator.NOT_CONTAINS:
                if isinstance(field_value, str) and isinstance(compare_value, str):
                    return compare_value not in field_value
                elif isinstance(field_value, (list, dict)):
                    return compare_value not in field_value
                return True
            
            elif op == ConditionOperator.MATCHES:
                if isinstance(field_value, str):
                    pattern = re.compile(compare_value, 0 if self.case_sensitive else re.IGNORECASE)
                    return bool(pattern.search(field_value))
                return False
            
            elif op == ConditionOperator.STARTS_WITH:
                if isinstance(field_value, str) and isinstance(compare_value, str):
                    return field_value.startswith(compare_value)
                return False
            
            elif op == ConditionOperator.ENDS_WITH:
                if isinstance(field_value, str) and isinstance(compare_value, str):
                    return field_value.endswith(compare_value)
                return False
            
            elif op == ConditionOperator.IS_EMPTY:
                if field_value is None:
                    return True
                if isinstance(field_value, (str, list, dict)):
                    return len(field_value) == 0
                return False
            
            elif op == ConditionOperator.IS_NOT_EMPTY:
                if field_value is None:
                    return False
                if isinstance(field_value, (str, list, dict)):
                    return len(field_value) > 0
                return True
            
            elif op == ConditionOperator.BETWEEN:
                if isinstance(compare_value, (list, tuple)) and len(compare_value) == 2:
                    return compare_value[0] <= field_value <= compare_value[1]
                return False
            
            elif op == ConditionOperator.SIZE_EQ:
                if hasattr(field_value, '__len__'):
                    return len(field_value) == compare_value
                return False
            
            elif op == ConditionOperator.SIZE_GT:
                if hasattr(field_value, '__len__'):
                    return len(field_value) > compare_value
                return False
            
            elif op == ConditionOperator.SIZE_LT:
                if hasattr(field_value, '__len__'):
                    return len(field_value) < compare_value
                return False
            
        except Exception:
            return False
        
        return False


@dataclass
class ConditionGroup:
    """条件组"""
    logical_operator: LogicalOperator
    conditions: List[Union['ConditionGroup', Condition]]
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """评估条件组"""
        results = [cond.evaluate(context) for cond in self.conditions]
        
        if self.logical_operator == LogicalOperator.AND:
            return all(results)
        elif self.logical_operator == LogicalOperator.OR:
            return any(results)
        elif self.logical_operator == LogicalOperator.NOT:
            return not results[0] if results else True
        
        return False


@dataclass
class Action:
    """动作"""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行动作"""
        result = {
            "action": self.action_type.value,
            "timestamp": time.time(),
            "parameters": self.parameters
        }
        
        if self.action_type == ActionType.MODIFY:
            # 执行修改
            modifications = self.parameters.get("modifications", {})
            for path, value in modifications.items():
                self._apply_modification(context, path, value)
            result["modified"] = True
        
        elif self.action_type == ActionType.REDIRECT:
            result["redirect_url"] = self.parameters.get("url")
        
        elif self.action_type == ActionType.THROTTLE:
            result["throttle_rate"] = self.parameters.get("rate")
        
        return result
    
    def _apply_modification(self, context: Dict[str, Any], path: str, value: Any) -> None:
        """应用修改"""
        keys = path.split('.')
        obj = context
        for key in keys[:-1]:
            if key not in obj:
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value


@dataclass
class Rule:
    """规则"""
    rule_id: str
    name: str
    description: str = ""
    condition: Union[Condition, ConditionGroup] = None
    actions: List[Action] = field(default_factory=list)
    priority: int = 100                  # 优先级，数字越小优先级越高
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    last_hit_time: Optional[float] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """评估规则"""
        if not self.enabled:
            return False
        
        if self.condition is None:
            return False
        
        return self.condition.evaluate(context)
    
    def execute(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行规则动作"""
        results = []
        for action in self.actions:
            result = action.execute(context)
            results.append(result)
        
        # 更新命中统计
        self.hit_count += 1
        self.last_hit_time = time.time()
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "enabled": self.enabled,
            "tags": self.tags,
            "hit_count": self.hit_count,
            "last_hit_time": self.last_hit_time
        }


class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._rule_list: List[Rule] = []
        self._action_handlers: Dict[ActionType, Callable[[Action, Dict], Any]] = {}
        self._pre_hooks: List[Callable[[Dict], None]] = []
        self._post_hooks: List[Callable[[Dict, List], None]] = []
    
    def add_rule(self, rule: Rule) -> None:
        """添加规则"""
        self._rules[rule.rule_id] = rule
        self._rebuild_rule_list()
    
    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._rebuild_rule_list()
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self._rules.get(rule_id)
    
    def update_rule(self, rule: Rule) -> bool:
        """更新规则"""
        if rule.rule_id in self._rules:
            self._rules[rule.rule_id] = rule
            self._rebuild_rule_list()
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False
    
    def _rebuild_rule_list(self) -> None:
        """重建规则列表（按优先级排序）"""
        self._rule_list = sorted(self._rules.values(), key=lambda r: r.priority)
    
    def add_action_handler(self, action_type: ActionType, handler: Callable[[Action, Dict], Any]) -> None:
        """添加动作处理器"""
        self._action_handlers[action_type] = handler
    
    def add_pre_hook(self, hook: Callable[[Dict], None]) -> None:
        """添加前置钩子"""
        self._pre_hooks.append(hook)
    
    def add_post_hook(self, hook: Callable[[Dict, List], None]) -> None:
        """添加后置钩子"""
        self._post_hooks.append(hook)
    
    def evaluate(self, context: Dict[str, Any]) -> List[Rule]:
        """评估所有规则，返回匹配的规则列表"""
        matched_rules = []
        
        # 执行前置钩子
        for hook in self._pre_hooks:
            hook(context)
        
        # 按优先级评估规则
        for rule in self._rule_list:
            if rule.evaluate(context):
                matched_rules.append(rule)
        
        return matched_rules
    
    def execute(self, context: Dict[str, Any], stop_on_deny: bool = True) -> Dict[str, Any]:
        """执行规则引擎"""
        matched_rules = self.evaluate(context)
        all_results = []
        final_action = ActionType.ALLOW
        
        for rule in matched_rules:
            results = rule.execute(context)
            all_results.extend(results)
            
            # 检查是否有拒绝动作
            for result in results:
                if result.get("action") == ActionType.DENY.value:
                    final_action = ActionType.DENY
                    if stop_on_deny:
                        break
            
            if stop_on_deny and final_action == ActionType.DENY:
                break
        
        # 执行后置钩子
        for hook in self._post_hooks:
            hook(context, all_results)
        
        return {
            "matched_rules": [r.rule_id for r in matched_rules],
            "results": all_results,
            "final_action": final_action.value,
            "timestamp": time.time()
        }
    
    def get_rules_by_tag(self, tag: str) -> List[Rule]:
        """按标签获取规则"""
        return [r for r in self._rules.values() if tag in r.tags]
    
    def get_all_rules(self) -> List[Rule]:
        """获取所有规则"""
        return list(self._rules.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "disabled_rules": sum(1 for r in self._rules.values() if not r.enabled),
            "total_hits": sum(r.hit_count for r in self._rules.values())
        }
    
    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """导入规则"""
        imported = 0
        for rule_dict in rules_data:
            try:
                rule = self._parse_rule(rule_dict)
                self.add_rule(rule)
                imported += 1
            except Exception:
                continue
        return imported
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """导出规则"""
        return [r.to_dict() for r in self._rules.values()]
    
    def _parse_rule(self, data: Dict[str, Any]) -> Rule:
        """解析规则数据"""
        condition = self._parse_condition(data.get("condition", {}))
        
        actions = []
        for action_data in data.get("actions", []):
            action = Action(
                action_type=ActionType(action_data["type"]),
                parameters=action_data.get("parameters", {})
            )
            actions.append(action)
        
        return Rule(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data.get("description", ""),
            condition=condition,
            actions=actions,
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
    
    def _parse_condition(self, data: Dict[str, Any]) -> Union[Condition, ConditionGroup]:
        """解析条件数据"""
        if "logical_operator" in data:
            # 条件组
            logical_op = LogicalOperator(data["logical_operator"])
            conditions = [self._parse_condition(c) for c in data.get("conditions", [])]
            return ConditionGroup(logical_operator=logical_op, conditions=conditions)
        else:
            # 单个条件
            return Condition(
                field=data["field"],
                operator=ConditionOperator(data["operator"]),
                value=data["value"],
                case_sensitive=data.get("case_sensitive", True),
                negate=data.get("negate", False)
            )


class RuleBuilder:
    """规则构建器"""
    
    def __init__(self):
        self._rule_id: str = ""
        self._name: str = ""
        self._description: str = ""
        self._condition: Optional[Union[Condition, ConditionGroup]] = None
        self._actions: List[Action] = []
        self._priority: int = 100
        self._enabled: bool = True
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}
    
    def set_id(self, rule_id: str) -> 'RuleBuilder':
        self._rule_id = rule_id
        return self
    
    def set_name(self, name: str) -> 'RuleBuilder':
        self._name = name
        return self
    
    def set_description(self, description: str) -> 'RuleBuilder':
        self._description = description
        return self
    
    def set_condition(self, condition: Union[Condition, ConditionGroup]) -> 'RuleBuilder':
        self._condition = condition
        return self
    
    def add_action(self, action: Action) -> 'RuleBuilder':
        self._actions.append(action)
        return self
    
    def set_priority(self, priority: int) -> 'RuleBuilder':
        self._priority = priority
        return self
    
    def add_tag(self, tag: str) -> 'RuleBuilder':
        self._tags.append(tag)
        return self
    
    def set_metadata(self, key: str, value: Any) -> 'RuleBuilder':
        self._metadata[key] = value
        return self
    
    def build(self) -> Rule:
        """构建规则"""
        if not self._rule_id or not self._name:
            raise ValueError("Rule ID and name are required")
        
        return Rule(
            rule_id=self._rule_id,
            name=self._name,
            description=self._description,
            condition=self._condition,
            actions=self._actions,
            priority=self._priority,
            enabled=self._enabled,
            tags=self._tags,
            metadata=self._metadata
        )


# 预定义规则模板
class RuleTemplates:
    """规则模板"""
    
    @staticmethod
    def create_sql_injection_rule() -> Rule:
        """创建SQL注入检测规则"""
        condition = Condition(
            field="request.body",
            operator=ConditionOperator.MATCHES,
            value=r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table|exec\s*\(|--|;|\bor\b\s*\d+\s*=\s*\d+)"
        )
        
        return Rule(
            rule_id="sql_injection_detect",
            name="SQL注入检测",
            description="检测常见的SQL注入模式",
            condition=condition,
            actions=[Action(action_type=ActionType.DENY, parameters={"reason": "SQL注入攻击检测"})],
            priority=10,
            tags=["security", "injection", "sql"]
        )
    
    @staticmethod
    def create_xss_rule() -> Rule:
        """创建XSS检测规则"""
        condition = Condition(
            field="request.body",
            operator=ConditionOperator.MATCHES,
            value=r"(?i)(<script|javascript:|on\w+\s*=|<iframe|<object|<embed)"
        )
        
        return Rule(
            rule_id="xss_detect",
            name="XSS攻击检测",
            description="检测跨站脚本攻击模式",
            condition=condition,
            actions=[Action(action_type=ActionType.DENY, parameters={"reason": "XSS攻击检测"})],
            priority=10,
            tags=["security", "injection", "xss"]
        )
    
    @staticmethod
    def create_path_traversal_rule() -> Rule:
        """创建路径遍历检测规则"""
        condition = Condition(
            field="request.path",
            operator=ConditionOperator.MATCHES,
            value=r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/)"
        )
        
        return Rule(
            rule_id="path_traversal_detect",
            name="路径遍历检测",
            description="检测路径遍历攻击",
            condition=condition,
            actions=[Action(action_type=ActionType.DENY, parameters={"reason": "路径遍历攻击检测"})],
            priority=10,
            tags=["security", "traversal"]
        )
    
    @staticmethod
    def create_rate_limit_rule(max_requests: int = 100) -> Rule:
        """创建速率限制规则"""
        condition = Condition(
            field="client.request_count",
            operator=ConditionOperator.GT,
            value=max_requests
        )
        
        return Rule(
            rule_id="rate_limit",
            name="请求速率限制",
            description=f"限制请求速率不超过{max_requests}次/分钟",
            condition=condition,
            actions=[Action(action_type=ActionType.THROTTLE, parameters={"rate": max_requests})],
            priority=5,
            tags=["security", "rate-limit"]
        )
