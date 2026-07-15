"""
条件执行器模块

提供高级条件执行功能：
- 复杂条件评估（and/or/not逻辑组合）
- 条件分支执行
- Switch-Case模式
- 规则引擎集成

Classes:
    ConditionalExecutor: 条件执行器主类
    ConditionEvaluator: 条件评估器
    LogicalOperator: 逻辑操作符枚举
    BranchExecutor: 分支执行器
    SwitchCaseExecutor: Switch-Case执行器
    RuleEngineIntegration: 规则引擎集成
"""

import ast
import copy
import operator
import re
from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class LogicalOperator(Enum):
    """逻辑操作符枚举"""
    AND = "and"           # 逻辑与
    OR = "or"             # 逻辑或
    NOT = "not"           # 逻辑非
    XOR = "xor"           # 异或
    NAND = "nand"         # 与非
    NOR = "nor"           # 或非


class ComparisonOperator(Enum):
    """比较操作符枚举"""
    EQ = "eq"             # 等于
    NE = "ne"             # 不等于
    GT = "gt"             # 大于
    GTE = "gte"           # 大于等于
    LT = "lt"             # 小于
    LTE = "lte"           # 小于等于
    IN = "in"             # 包含于
    CONTAINS = "contains" # 包含
    STARTS_WITH = "starts_with"  # 以...开头
    ENDS_WITH = "ends_with"      # 以...结尾
    MATCHES = "matches"   # 正则匹配
    EXISTS = "exists"     # 存在
    NOT_EXISTS = "not_exists"    # 不存在
    BETWEEN = "between"   # 在...之间


class ConditionError(Exception):
    """条件执行异常"""
    pass


class InvalidConditionError(ConditionError):
    """无效条件异常"""
    pass


class BranchExecutionError(ConditionError):
    """分支执行异常"""
    pass


@dataclass
class Condition:
    """
    条件定义

    Attributes:
        field: 字段路径（支持点号分隔，如"user.age"）
        operator: 比较操作符
        value: 比较值
        negate: 是否取反
        case_sensitive: 字符串比较是否区分大小写
    """
    field: str
    operator: Union[str, ComparisonOperator]
    value: Any = None
    negate: bool = False
    case_sensitive: bool = True

    def __post_init__(self):
        if isinstance(self.operator, str):
            try:
                self.operator = ComparisonOperator(self.operator)
            except ValueError:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "field": self.field,
            "operator": self.operator.value if isinstance(self.operator, ComparisonOperator) else self.operator,
            "value": copy.deepcopy(self.value),
            "negate": self.negate,
            "case_sensitive": self.case_sensitive,
        }


@dataclass
class CompositeCondition:
    """
    复合条件定义

    Attributes:
        operator: 逻辑操作符
        conditions: 子条件列表（可以是Condition或CompositeCondition）
    """
    operator: LogicalOperator
    conditions: List[Union[Condition, "CompositeCondition"]] = dataclass_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operator": self.operator.value,
            "conditions": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.conditions
            ],
        }


@dataclass
class Branch:
    """
    分支定义

    Attributes:
        name: 分支名称
        condition: 分支条件
        priority: 优先级（数字越小优先级越高）
        action: 分支动作（可调用对象）
        metadata: 元数据
    """
    name: str
    condition: Optional[Union[Condition, CompositeCondition]] = None
    priority: int = 0
    action: Optional[Callable[[Dict[str, Any]], Any]] = None
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass
class Rule:
    """
    规则定义

    Attributes:
        id: 规则ID
        name: 规则名称
        condition: 规则条件
        actions: 规则动作列表
        priority: 优先级
        enabled: 是否启用
        metadata: 元数据
    """
    id: str
    name: str
    condition: Union[Condition, CompositeCondition]
    actions: List[Callable[[Dict[str, Any]], Any]] = dataclass_field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)


class ConditionEvaluator:
    """
    条件评估器

    评估单个条件或复合条件，支持复杂的数据路径解析和多种比较操作。

    Usage:
        evaluator = ConditionEvaluator()
        condition = Condition(field="user.age", operator="gte", value=18)
        result = evaluator.evaluate(condition, {"user": {"age": 25}})
    """

    def __init__(self):
        self._comparison_funcs: Dict[ComparisonOperator, Callable] = {
            ComparisonOperator.EQ: operator.eq,
            ComparisonOperator.NE: operator.ne,
            ComparisonOperator.GT: operator.gt,
            ComparisonOperator.GTE: operator.ge,
            ComparisonOperator.LT: operator.lt,
            ComparisonOperator.LTE: operator.le,
            ComparisonOperator.IN: lambda a, b: a in b if b is not None else False,
            ComparisonOperator.CONTAINS: lambda a, b: b in a if a is not None else False,
            ComparisonOperator.STARTS_WITH: lambda a, b: str(a).startswith(str(b)) if a is not None else False,
            ComparisonOperator.ENDS_WITH: lambda a, b: str(a).endswith(str(b)) if a is not None else False,
            ComparisonOperator.MATCHES: lambda a, b: bool(re.match(b, str(a))) if a is not None else False,
            ComparisonOperator.EXISTS: lambda a, b: a is not None,
            ComparisonOperator.NOT_EXISTS: lambda a, b: a is None,
            ComparisonOperator.BETWEEN: lambda a, b: b[0] <= a <= b[1] if isinstance(b, (list, tuple)) and len(b) >= 2 and a is not None else False,
        }

    def evaluate(
        self,
        condition: Union[Condition, CompositeCondition],
        context: Dict[str, Any],
    ) -> bool:
        """
        评估条件

        Args:
            condition: 条件定义
            context: 评估上下文

        Returns:
            条件是否满足
        """
        if isinstance(condition, Condition):
            return self._evaluate_simple(condition, context)
        elif isinstance(condition, CompositeCondition):
            return self._evaluate_composite(condition, context)
        else:
            raise InvalidConditionError(f"不支持的条件类型: {type(condition)}")

    def _evaluate_simple(self, condition: Condition, context: Dict[str, Any]) -> bool:
        """评估简单条件"""
        actual_value = self._get_value_by_path(context, condition.field)
        expected_value = condition.value

        # 获取比较函数
        op = condition.operator
        if isinstance(op, str):
            try:
                op = ComparisonOperator(op)
            except ValueError:
                raise InvalidConditionError(f"未知的比较操作符: {op}")

        compare_fn = self._comparison_funcs.get(op)
        if compare_fn is None:
            raise InvalidConditionError(f"未实现的比较操作符: {op}")

        # 执行比较
        try:
            if not condition.case_sensitive and isinstance(actual_value, str) and isinstance(expected_value, str):
                actual_value = actual_value.lower()
                expected_value = expected_value.lower()

            result = compare_fn(actual_value, expected_value)
        except Exception as e:
            raise InvalidConditionError(f"条件评估失败: {e}")

        # 应用取反
        if condition.negate:
            result = not result

        return bool(result)

    def _evaluate_composite(self, condition: CompositeCondition, context: Dict[str, Any]) -> bool:
        """评估复合条件"""
        if not condition.conditions:
            return True

        results = []
        for sub_condition in condition.conditions:
            result = self.evaluate(sub_condition, context)
            results.append(result)

        if condition.operator == LogicalOperator.AND:
            return all(results)
        elif condition.operator == LogicalOperator.OR:
            return any(results)
        elif condition.operator == LogicalOperator.NOT:
            return not results[0] if results else True
        elif condition.operator == LogicalOperator.XOR:
            return sum(results) == 1
        elif condition.operator == LogicalOperator.NAND:
            return not all(results)
        elif condition.operator == LogicalOperator.NOR:
            return not any(results)
        else:
            raise InvalidConditionError(f"未知的逻辑操作符: {condition.operator}")

    def _get_value_by_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        根据路径获取值

        支持点号分隔的路径，如 "user.address.city"
        支持数组索引，如 "users[0].name"
        """
        if not path:
            return None

        current = data
        # 解析路径：支持 "a.b.c" 和 "a[0].b" 格式
        parts = re.findall(r'[^.\[\]]+|\[\d+\]', path)

        for part in parts:
            if current is None:
                return None

            # 处理数组索引 [n]
            if part.startswith('[') and part.endswith(']'):
                index = int(part[1:-1])
                if isinstance(current, (list, tuple)) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                if isinstance(current, dict):
                    current = current.get(part)
                elif hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None

        return current

    def evaluate_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        评估Python表达式

        安全地评估简单的Python表达式，支持基本的数学和逻辑运算。

        Args:
            expression: Python表达式字符串
            context: 变量上下文

        Returns:
            表达式结果
        """
        try:
            # 使用ast模块安全解析
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body, context)
        except Exception as e:
            raise InvalidConditionError(f"表达式评估失败: {e}")

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """递归评估AST节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Str):  # Python < 3.8
            return node.s
        elif isinstance(node, ast.Name):
            return context.get(node.id)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right if right != 0 else float('inf')
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                if isinstance(op, ast.Eq):
                    if left != right:
                        return False
                elif isinstance(op, ast.NotEq):
                    if left == right:
                        return False
                elif isinstance(op, ast.Lt):
                    if not (left < right):
                        return False
                elif isinstance(op, ast.LtE):
                    if not (left <= right):
                        return False
                elif isinstance(op, ast.Gt):
                    if not (left > right):
                        return False
                elif isinstance(op, ast.GtE):
                    if not (left >= right):
                        return False
                elif isinstance(op, ast.In):
                    if left not in right:
                        return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            values = [self._eval_node(v, context) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            if isinstance(node.op, ast.Not):
                return not operand
            elif isinstance(node.op, ast.USub):
                return -operand
        elif isinstance(node, ast.Call):
            raise InvalidConditionError("函数调用不允许")
        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value, context)
            slice_val = self._eval_node(node.slice, context)
            return value[slice_val]
        elif isinstance(node, ast.Attribute):
            value = self._eval_node(node.value, context)
            return getattr(value, node.attr, None) if value else None

        raise InvalidConditionError(f"不支持的表达式类型: {type(node).__name__}")


class BranchExecutor:
    """
    分支执行器

    管理多个条件分支，根据条件选择执行路径。

    Usage:
        executor = BranchExecutor()
        executor.add_branch(Branch(name="adult", condition=age_condition))
        executor.add_branch(Branch(name="minor", condition=minor_condition))
        result = executor.execute(context)
    """

    def __init__(self):
        self._branches: List[Branch] = []
        self._evaluator = ConditionEvaluator()
        self._default_branch: Optional[Branch] = None

    def add_branch(self, branch: Branch) -> "BranchExecutor":
        """
        添加分支

        Args:
            branch: 分支定义

        Returns:
            self（支持链式调用）
        """
        self._branches.append(branch)
        # 按优先级排序
        self._branches.sort(key=lambda b: b.priority)
        return self

    def set_default_branch(self, branch: Branch) -> "BranchExecutor":
        """
        设置默认分支

        Args:
            branch: 默认分支

        Returns:
            self
        """
        self._default_branch = branch
        return self

    def remove_branch(self, name: str) -> bool:
        """
        移除分支

        Args:
            name: 分支名称

        Returns:
            是否成功移除
        """
        for i, branch in enumerate(self._branches):
            if branch.name == name:
                self._branches.pop(i)
                return True
        return False

    def execute(
        self,
        context: Dict[str, Any],
        execute_action: bool = True,
    ) -> Tuple[str, Optional[Any]]:
        """
        执行分支选择

        Args:
            context: 执行上下文
            execute_action: 是否执行分支动作

        Returns:
            (选中的分支名称, 动作执行结果)
        """
        # 按优先级评估分支
        for branch in self._branches:
            if branch.condition is None:
                # 无条件分支，直接匹配
                if execute_action and branch.action:
                    result = branch.action(context)
                    return branch.name, result
                return branch.name, None

            if self._evaluator.evaluate(branch.condition, context):
                if execute_action and branch.action:
                    result = branch.action(context)
                    return branch.name, result
                return branch.name, None

        # 没有匹配的分支，使用默认分支
        if self._default_branch:
            if execute_action and self._default_branch.action:
                result = self._default_branch.action(context)
                return self._default_branch.name, result
            return self._default_branch.name, None

        raise BranchExecutionError("没有匹配的分支且未设置默认分支")

    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """
        评估所有分支条件

        Args:
            context: 评估上下文

        Returns:
            分支名称到评估结果的映射
        """
        results = {}
        for branch in self._branches:
            if branch.condition is None:
                results[branch.name] = True
            else:
                results[branch.name] = self._evaluator.evaluate(branch.condition, context)
        return results

    def get_branch_names(self) -> List[str]:
        """获取所有分支名称"""
        return [b.name for b in self._branches]

    def clear(self) -> None:
        """清空所有分支"""
        self._branches.clear()
        self._default_branch = None


class SwitchCaseExecutor:
    """
    Switch-Case执行器

    实现类switch-case的条件分支模式。

    Usage:
        executor = SwitchCaseExecutor(switch_field="status")
        executor.add_case("active", handle_active)
        executor.add_case("inactive", handle_inactive)
        executor.set_default(handle_default)
        result = executor.execute(context)
    """

    def __init__(self, switch_field: str):
        """
        初始化

        Args:
            switch_field: 用于switch判断的字段路径
        """
        self._switch_field = switch_field
        self._cases: Dict[Any, Callable[[Dict[str, Any]], Any]] = {}
        self._default_case: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._evaluator = ConditionEvaluator()

    def add_case(
        self,
        value: Any,
        action: Callable[[Dict[str, Any]], Any],
    ) -> "SwitchCaseExecutor":
        """
        添加case

        Args:
            value: case匹配值
            action: case执行动作

        Returns:
            self
        """
        self._cases[value] = action
        return self

    def add_cases(
        self,
        values: List[Any],
        action: Callable[[Dict[str, Any]], Any],
    ) -> "SwitchCaseExecutor":
        """
        批量添加case

        Args:
            values: case匹配值列表
            action: case执行动作

        Returns:
            self
        """
        for value in values:
            self._cases[value] = action
        return self

    def set_default(self, action: Callable[[Dict[str, Any]], Any]) -> "SwitchCaseExecutor":
        """
        设置默认case

        Args:
            action: 默认执行动作

        Returns:
            self
        """
        self._default_case = action
        return self

    def execute(self, context: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        执行switch-case

        Args:
            context: 执行上下文

        Returns:
            (匹配的值, 执行结果)
        """
        switch_value = self._evaluator._get_value_by_path(context, self._switch_field)

        # 精确匹配
        if switch_value in self._cases:
            action = self._cases[switch_value]
            return switch_value, action(context)

        # 尝试字符串匹配
        for case_value, action in self._cases.items():
            if str(switch_value) == str(case_value):
                return case_value, action(context)

        # 默认case
        if self._default_case:
            return None, self._default_case(context)

        raise BranchExecutionError(f"没有匹配的case且未设置默认case，switch值: {switch_value}")

    def get_cases(self) -> List[Any]:
        """获取所有case值"""
        return list(self._cases.keys())

    def has_case(self, value: Any) -> bool:
        """检查是否存在指定case"""
        return value in self._cases

    def remove_case(self, value: Any) -> bool:
        """移除指定case"""
        if value in self._cases:
            del self._cases[value]
            return True
        return False

    def clear(self) -> None:
        """清空所有case"""
        self._cases.clear()
        self._default_case = None


class RuleEngineIntegration:
    """
    规则引擎集成

    提供规则定义、管理和执行功能，支持规则优先级和冲突解决。

    Usage:
        engine = RuleEngineIntegration()
        engine.add_rule(Rule(id="r1", name=" adult_rule", condition=cond, actions=[action]))
        results = engine.execute(context)
    """

    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._evaluator = ConditionEvaluator()
        self._conflict_resolution: str = "priority"  # priority, first_match, all

    def add_rule(self, rule: Rule) -> "RuleEngineIntegration":
        """
        添加规则

        Args:
            rule: 规则定义

        Returns:
            self
        """
        self._rules[rule.id] = rule
        return self

    def remove_rule(self, rule_id: str) -> bool:
        """
        移除规则

        Args:
            rule_id: 规则ID

        Returns:
            是否成功移除
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """获取规则"""
        return self._rules.get(rule_id)

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

    def set_conflict_resolution(self, strategy: str) -> "RuleEngineIntegration":
        """
        设置冲突解决策略

        Args:
            strategy: 策略名称（priority, first_match, all）

        Returns:
            self
        """
        if strategy not in ("priority", "first_match", "all"):
            raise ValueError(f"未知的冲突解决策略: {strategy}")
        self._conflict_resolution = strategy
        return self

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规则引擎

        Args:
            context: 执行上下文

        Returns:
            执行结果，包含匹配的规则和动作结果
        """
        # 获取启用的规则并按优先级排序
        enabled_rules = [r for r in self._rules.values() if r.enabled]
        enabled_rules.sort(key=lambda r: r.priority)

        matched_rules = []
        action_results = []

        for rule in enabled_rules:
            if self._evaluator.evaluate(rule.condition, context):
                matched_rules.append(rule)

                # 执行规则动作
                for action in rule.actions:
                    try:
                        result = action(context)
                        action_results.append({
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "result": result,
                        })
                    except Exception as e:
                        action_results.append({
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "error": str(e),
                        })

                # 根据冲突解决策略决定是否继续
                if self._conflict_resolution == "first_match":
                    break
                elif self._conflict_resolution == "priority":
                    # 只执行最高优先级的规则
                    break

        return {
            "matched_rules": [r.id for r in matched_rules],
            "action_results": action_results,
            "context": context,
        }

    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """
        评估所有规则（不执行动作）

        Args:
            context: 评估上下文

        Returns:
            规则ID到评估结果的映射
        """
        results = {}
        for rule in self._rules.values():
            if rule.enabled:
                results[rule.id] = self._evaluator.evaluate(rule.condition, context)
            else:
                results[rule.id] = False
        return results

    def get_rule_stats(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        total = len(self._rules)
        enabled = sum(1 for r in self._rules.values() if r.enabled)
        disabled = total - enabled

        return {
            "total_rules": total,
            "enabled_rules": enabled,
            "disabled_rules": disabled,
            "rule_ids": list(self._rules.keys()),
        }

    def clear(self) -> None:
        """清空所有规则"""
        self._rules.clear()


class ConditionalExecutor:
    """
    条件执行器主类

    整合条件评估、分支执行、switch-case和规则引擎功能。

    Usage:
        executor = ConditionalExecutor()
        executor.add_branch(...)
        executor.add_rule(...)
        result = executor.execute(context)
    """

    def __init__(self):
        self._branch_executor = BranchExecutor()
        self._switch_executors: Dict[str, SwitchCaseExecutor] = {}
        self._rule_engine = RuleEngineIntegration()
        self._evaluator = ConditionEvaluator()
        self._pre_conditions: List[Condition] = []
        self._post_conditions: List[Condition] = []

    def add_branch(self, branch: Branch) -> "ConditionalExecutor":
        """添加分支"""
        self._branch_executor.add_branch(branch)
        return self

    def set_default_branch(self, branch: Branch) -> "ConditionalExecutor":
        """设置默认分支"""
        self._branch_executor.set_default_branch(branch)
        return self

    def add_switch(self, name: str, switch_executor: SwitchCaseExecutor) -> "ConditionalExecutor":
        """添加switch-case执行器"""
        self._switch_executors[name] = switch_executor
        return self

    def add_rule(self, rule: Rule) -> "ConditionalExecutor":
        """添加规则"""
        self._rule_engine.add_rule(rule)
        return self

    def add_pre_condition(self, condition: Condition) -> "ConditionalExecutor":
        """添加前置条件"""
        self._pre_conditions.append(condition)
        return self

    def add_post_condition(self, condition: Condition) -> "ConditionalExecutor":
        """添加后置条件"""
        self._post_conditions.append(condition)
        return self

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行条件逻辑

        Args:
            context: 执行上下文

        Returns:
            执行结果
        """
        # 检查前置条件
        for pre_cond in self._pre_conditions:
            if not self._evaluator.evaluate(pre_cond, context):
                raise ConditionError(f"前置条件不满足: {pre_cond.field}")

        # 执行分支
        branch_name, branch_result = self._branch_executor.execute(context)

        # 执行switch-case
        switch_results = {}
        for name, switch_exec in self._switch_executors.items():
            switch_results[name] = switch_exec.execute(context)

        # 执行规则引擎
        rule_results = self._rule_engine.execute(context)

        # 检查后置条件
        for post_cond in self._post_conditions:
            if not self._evaluator.evaluate(post_cond, context):
                raise ConditionError(f"后置条件不满足: {post_cond.field}")

        return {
            "branch": {"name": branch_name, "result": branch_result},
            "switches": switch_results,
            "rules": rule_results,
        }

    def evaluate(self, condition: Union[Condition, CompositeCondition], context: Dict[str, Any]) -> bool:
        """评估条件"""
        return self._evaluator.evaluate(condition, context)

    def evaluate_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """评估表达式"""
        return self._evaluator.evaluate_expression(expression, context)

    def get_evaluator(self) -> ConditionEvaluator:
        """获取条件评估器"""
        return self._evaluator

    def get_branch_executor(self) -> BranchExecutor:
        """获取分支执行器"""
        return self._branch_executor

    def get_rule_engine(self) -> RuleEngineIntegration:
        """获取规则引擎"""
        return self._rule_engine
