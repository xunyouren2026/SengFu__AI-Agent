"""
条件分支模块

提供条件定义、评估和分支选择功能。
支持多种操作符和复杂条件组合。
仅使用Python标准库实现。
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable


class Operator(Enum):
    """条件操作符枚举。"""
    EQ = "eq"                    # 等于
    NE = "ne"                    # 不等于
    GT = "gt"                    # 大于
    GTE = "gte"                  # 大于等于
    LT = "lt"                    # 小于
    LTE = "lte"                  # 小于等于
    CONTAINS = "contains"        # 包含
    STARTS_WITH = "starts_with"  # 以...开头
    ENDS_WITH = "ends_with"      # 以...结尾
    MATCHES_REGEX = "matches_regex"  # 匹配正则
    IS_EMPTY = "is_empty"        # 为空
    IS_NOT_EMPTY = "is_not_empty"  # 不为空
    IN = "in"                    # 在列表中
    NOT_IN = "not_in"            # 不在列表中


class LogicalOperator(Enum):
    """逻辑操作符枚举。"""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class Condition:
    """
    条件数据类。
    
    Attributes:
        type: 条件类型（简单条件或复杂条件）
        left_operand: 左操作数（变量名或值）
        operator: 操作符
        right_operand: 右操作数（值）
    """
    left_operand: str
    operator: Operator
    right_operand: Any = None
    
    def __post_init__(self):
        """初始化后处理。"""
        if isinstance(self.operator, str):
            self.operator = Operator(self.operator)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "left_operand": self.left_operand,
            "operator": self.operator.value,
            "right_operand": self.right_operand,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Condition":
        """从字典创建。"""
        return cls(
            left_operand=data["left_operand"],
            operator=Operator(data.get("operator", "eq")),
            right_operand=data.get("right_operand"),
        )


@dataclass
class ComplexCondition:
    """
    复杂条件数据类（用于AND/OR/NOT组合）。
    
    Attributes:
        logical_op: 逻辑操作符
        conditions: 子条件列表
    """
    logical_op: LogicalOperator
    conditions: List[Union[Condition, "ComplexCondition"]] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理。"""
        if isinstance(self.logical_op, str):
            self.logical_op = LogicalOperator(self.logical_op)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "logical_op": self.logical_op.value,
            "conditions": [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.conditions
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComplexCondition":
        """从字典创建。"""
        conditions = []
        for c in data.get("conditions", []):
            if "logical_op" in c:
                conditions.append(ComplexCondition.from_dict(c))
            else:
                conditions.append(Condition.from_dict(c))
        
        return cls(
            logical_op=LogicalOperator(data.get("logical_op", "and")),
            conditions=conditions,
        )


class ConditionEvaluator:
    """
    条件评估器。
    
    评估单个条件和复杂条件树。
    """
    
    def __init__(self, variable_resolver: Optional[Callable[[str], Any]] = None):
        """
        初始化条件评估器。
        
        Args:
            variable_resolver: 变量解析函数，接受变量名返回变量值
        """
        self._variable_resolver = variable_resolver
        self._operator_handlers = {
            Operator.EQ: self._eval_eq,
            Operator.NE: self._eval_ne,
            Operator.GT: self._eval_gt,
            Operator.GTE: self._eval_gte,
            Operator.LT: self._eval_lt,
            Operator.LTE: self._eval_lte,
            Operator.CONTAINS: self._eval_contains,
            Operator.STARTS_WITH: self._eval_starts_with,
            Operator.ENDS_WITH: self._eval_ends_with,
            Operator.MATCHES_REGEX: self._eval_matches_regex,
            Operator.IS_EMPTY: self._eval_is_empty,
            Operator.IS_NOT_EMPTY: self._eval_is_not_empty,
            Operator.IN: self._eval_in,
            Operator.NOT_IN: self._eval_not_in,
        }
    
    def set_variable_resolver(self, resolver: Callable[[str], Any]) -> None:
        """设置变量解析函数。"""
        self._variable_resolver = resolver
    
    def evaluate_condition(
        self,
        condition: Condition,
        state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        评估单个条件。
        
        Args:
            condition: 条件对象
            state: 工作流状态（变量字典）
            
        Returns:
            条件是否成立
        """
        left_value = self._resolve_value(condition.left_operand, state)
        right_value = condition.right_operand
        
        handler = self._operator_handlers.get(condition.operator)
        if handler is None:
            raise ValueError(f"不支持的操作符: {condition.operator}")
        
        return handler(left_value, right_value)
    
    def evaluate_complex(
        self,
        condition_tree: Union[Condition, ComplexCondition],
        state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        评估复杂条件树。
        
        Args:
            condition_tree: 条件树（可以是简单条件或复杂条件）
            state: 工作流状态
            
        Returns:
            条件是否成立
        """
        if isinstance(condition_tree, Condition):
            return self.evaluate_condition(condition_tree, state)
        
        if isinstance(condition_tree, ComplexCondition):
            if condition_tree.logical_op == LogicalOperator.AND:
                return all(
                    self.evaluate_complex(c, state)
                    for c in condition_tree.conditions
                )
            elif condition_tree.logical_op == LogicalOperator.OR:
                return any(
                    self.evaluate_complex(c, state)
                    for c in condition_tree.conditions
                )
            elif condition_tree.logical_op == LogicalOperator.NOT:
                if len(condition_tree.conditions) != 1:
                    raise ValueError("NOT操作符只能有一个子条件")
                return not self.evaluate_complex(condition_tree.conditions[0], state)
        
        raise ValueError(f"无效的条件类型: {type(condition_tree)}")
    
    def _resolve_value(self, operand: str, state: Optional[Dict[str, Any]]) -> Any:
        """解析操作数值。"""
        if state is not None and operand in state:
            return state[operand]
        
        if self._variable_resolver is not None:
            return self._variable_resolver(operand)
        
        return operand
    
    def _eval_eq(self, left: Any, right: Any) -> bool:
        """评估等于。"""
        return left == right
    
    def _eval_ne(self, left: Any, right: Any) -> bool:
        """评估不等于。"""
        return left != right
    
    def _eval_gt(self, left: Any, right: Any) -> bool:
        """评估大于。"""
        try:
            return float(left) > float(right)
        except (ValueError, TypeError):
            return False
    
    def _eval_gte(self, left: Any, right: Any) -> bool:
        """评估大于等于。"""
        try:
            return float(left) >= float(right)
        except (ValueError, TypeError):
            return False
    
    def _eval_lt(self, left: Any, right: Any) -> bool:
        """评估小于。"""
        try:
            return float(left) < float(right)
        except (ValueError, TypeError):
            return False
    
    def _eval_lte(self, left: Any, right: Any) -> bool:
        """评估小于等于。"""
        try:
            return float(left) <= float(right)
        except (ValueError, TypeError):
            return False
    
    def _eval_contains(self, left: Any, right: Any) -> bool:
        """评估包含。"""
        if isinstance(left, str) and isinstance(right, str):
            return right in left
        if isinstance(left, (list, tuple)):
            return right in left
        if isinstance(left, dict):
            return right in left
        return False
    
    def _eval_starts_with(self, left: Any, right: Any) -> bool:
        """评估以...开头。"""
        if isinstance(left, str) and isinstance(right, str):
            return left.startswith(right)
        return False
    
    def _eval_ends_with(self, left: Any, right: Any) -> bool:
        """评估以...结尾。"""
        if isinstance(left, str) and isinstance(right, str):
            return left.endswith(right)
        return False
    
    def _eval_matches_regex(self, left: Any, right: Any) -> bool:
        """评估匹配正则。"""
        if isinstance(left, str) and isinstance(right, str):
            return bool(re.search(right, left))
        return False
    
    def _eval_is_empty(self, left: Any, right: Any) -> bool:
        """评估为空。"""
        if left is None:
            return True
        if isinstance(left, (str, list, tuple, dict)):
            return len(left) == 0
        return False
    
    def _eval_is_not_empty(self, left: Any, right: Any) -> bool:
        """评估不为空。"""
        return not self._eval_is_empty(left, right)
    
    def _eval_in(self, left: Any, right: Any) -> bool:
        """评估在列表中。"""
        if isinstance(right, (list, tuple)):
            return left in right
        return False
    
    def _eval_not_in(self, left: Any, right: Any) -> bool:
        """评估不在列表中。"""
        return not self._eval_in(left, right)


class ConditionalBranch:
    """
    条件分支。
    
    管理多个条件和对应的分支。
    """
    
    def __init__(self):
        """初始化条件分支。"""
        self._branches: List[Dict[str, Any]] = []
        self._default_branch: Optional[Dict[str, Any]] = None
        self._evaluator = ConditionEvaluator()
    
    def add_condition(
        self,
        branch_id: str,
        condition: Union[Condition, ComplexCondition, Dict[str, Any]],
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        添加条件分支。
        
        Args:
            branch_id: 分支ID
            condition: 条件对象或字典
            actions: 该分支对应的操作列表
        """
        if isinstance(condition, dict):
            if "logical_op" in condition:
                condition = ComplexCondition.from_dict(condition)
            else:
                condition = Condition.from_dict(condition)
        
        self._branches.append({
            "id": branch_id,
            "condition": condition,
            "actions": actions or [],
        })
    
    def add_default_branch(
        self,
        branch_id: str = "default",
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        添加默认分支。
        
        Args:
            branch_id: 分支ID
            actions: 该分支对应的操作列表
        """
        self._default_branch = {
            "id": branch_id,
            "condition": None,
            "actions": actions or [],
        }
    
    def evaluate(self, workflow_state: Dict[str, Any]) -> Optional[str]:
        """
        评估条件并返回匹配的分支ID。
        
        Args:
            workflow_state: 工作流状态
            
        Returns:
            匹配的分支ID，无匹配则返回默认分支ID或None
        """
        for branch in self._branches:
            condition = branch["condition"]
            if self._evaluator.evaluate_complex(condition, workflow_state):
                return branch["id"]
        
        if self._default_branch:
            return self._default_branch["id"]
        
        return None
    
    def evaluate_with_actions(
        self,
        workflow_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        评估条件并返回匹配的分支（包含操作）。
        
        Args:
            workflow_state: 工作流状态
            
        Returns:
            匹配的分支字典，无匹配则返回默认分支或None
        """
        for branch in self._branches:
            condition = branch["condition"]
            if self._evaluator.evaluate_complex(condition, workflow_state):
                return branch
        
        return self._default_branch
    
    def remove_branch(self, branch_id: str) -> bool:
        """
        删除分支。
        
        Args:
            branch_id: 分支ID
            
        Returns:
            是否成功删除
        """
        for i, branch in enumerate(self._branches):
            if branch["id"] == branch_id:
                self._branches.pop(i)
                return True
        
        if self._default_branch and self._default_branch["id"] == branch_id:
            self._default_branch = None
            return True
        
        return False
    
    def list_branches(self) -> List[Dict[str, Any]]:
        """
        列出所有分支。
        
        Returns:
            分支信息列表
        """
        branches = [
            {
                "id": b["id"],
                "has_condition": b["condition"] is not None,
                "action_count": len(b["actions"]),
            }
            for b in self._branches
        ]
        
        if self._default_branch:
            branches.append({
                "id": self._default_branch["id"],
                "is_default": True,
                "action_count": len(self._default_branch["actions"]),
            })
        
        return branches
    
    def clear_branches(self) -> None:
        """清除所有分支。"""
        self._branches.clear()
        self._default_branch = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "branches": [
                {
                    "id": b["id"],
                    "condition": b["condition"].to_dict() if b["condition"] else None,
                    "actions": b["actions"],
                }
                for b in self._branches
            ],
            "default_branch": {
                "id": self._default_branch["id"],
                "actions": self._default_branch["actions"],
            } if self._default_branch else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConditionalBranch":
        """从字典创建。"""
        branch = cls()
        
        for b in data.get("branches", []):
            condition_data = b.get("condition")
            if condition_data:
                if "logical_op" in condition_data:
                    condition = ComplexCondition.from_dict(condition_data)
                else:
                    condition = Condition.from_dict(condition_data)
                branch.add_condition(b["id"], condition, b.get("actions", []))
        
        default_data = data.get("default_branch")
        if default_data:
            branch.add_default_branch(default_data["id"], default_data.get("actions", []))
        
        return branch


# 便捷函数
def create_condition(
    left: str,
    operator: str,
    right: Any = None,
) -> Condition:
    """
    创建条件的便捷函数。
    
    Args:
        left: 左操作数
        operator: 操作符
        right: 右操作数
        
    Returns:
        条件对象
    """
    return Condition(
        left_operand=left,
        operator=Operator(operator),
        right_operand=right,
    )


def create_and_condition(*conditions: Condition) -> ComplexCondition:
    """
    创建AND组合的复杂条件。
    
    Args:
        *conditions: 子条件
        
    Returns:
        复杂条件对象
    """
    return ComplexCondition(
        logical_op=LogicalOperator.AND,
        conditions=list(conditions),
    )


def create_or_condition(*conditions: Condition) -> ComplexCondition:
    """
    创建OR组合的复杂条件。
    
    Args:
        *conditions: 子条件
        
    Returns:
        复杂条件对象
    """
    return ComplexCondition(
        logical_op=LogicalOperator.OR,
        conditions=list(conditions),
    )


def create_not_condition(condition: Condition) -> ComplexCondition:
    """
    创建NOT组合的复杂条件。
    
    Args:
        condition: 子条件
        
    Returns:
        复杂条件对象
    """
    return ComplexCondition(
        logical_op=LogicalOperator.NOT,
        conditions=[condition],
    )
