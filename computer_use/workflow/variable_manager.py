"""
变量管理器模块

提供变量定义、存储、获取和管理功能。
支持多种变量类型和作用域。
仅使用Python标准库实现。
"""

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class VariableType(Enum):
    """变量类型枚举。"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    SECRET = "secret"


class VariableScope(Enum):
    """变量作用域枚举。"""
    GLOBAL = "global"      # 全局作用域，所有工作流共享
    WORKFLOW = "workflow"  # 工作流作用域，单个工作流内共享
    STEP = "step"          # 步骤作用域，仅在当前步骤有效


@dataclass
class Variable:
    """
    变量数据类。
    
    Attributes:
        name: 变量名称
        type: 变量类型
        value: 当前值
        default: 默认值
        description: 变量描述
        is_secret: 是否为敏感变量（不显示值）
        scope: 变量作用域
    """
    name: str
    type: VariableType = VariableType.STRING
    value: Any = None
    default: Any = None
    description: str = ""
    is_secret: bool = False
    scope: VariableScope = VariableScope.WORKFLOW
    
    def __post_init__(self):
        """初始化后处理。"""
        if isinstance(self.type, str):
            self.type = VariableType(self.type)
        if isinstance(self.scope, str):
            self.scope = VariableScope(self.scope)
        
        # 如果没有值，使用默认值
        if self.value is None and self.default is not None:
            self.value = self.default
        
        # 根据类型设置is_secret
        if self.type == VariableType.SECRET:
            self.is_secret = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value if not self.is_secret else "***",
            "default": self.default if not self.is_secret else "***",
            "description": self.description,
            "is_secret": self.is_secret,
            "scope": self.scope.value,
        }
    
    def to_dict_full(self) -> Dict[str, Any]:
        """转换为完整字典（包含敏感值）。"""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "default": self.default,
            "description": self.description,
            "is_secret": self.is_secret,
            "scope": self.scope.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variable":
        """从字典创建。"""
        return cls(
            name=data["name"],
            type=VariableType(data.get("type", "string")),
            value=data.get("value"),
            default=data.get("default"),
            description=data.get("description", ""),
            is_secret=data.get("is_secret", False),
            scope=VariableScope(data.get("scope", "workflow")),
        )
    
    def get_display_value(self) -> str:
        """获取显示值（敏感变量会隐藏）。"""
        if self.is_secret:
            return "***"
        return str(self.value)
    
    def validate_type(self, value: Any) -> bool:
        """验证值是否符合变量类型。"""
        if self.type == VariableType.STRING:
            return isinstance(value, str)
        elif self.type == VariableType.NUMBER:
            return isinstance(value, (int, float))
        elif self.type == VariableType.BOOLEAN:
            return isinstance(value, bool)
        elif self.type == VariableType.ARRAY:
            return isinstance(value, list)
        elif self.type == VariableType.OBJECT:
            return isinstance(value, dict)
        elif self.type == VariableType.SECRET:
            return isinstance(value, str)
        return True
    
    def cast_value(self, value: Any) -> Any:
        """尝试将值转换为变量类型。"""
        try:
            if self.type == VariableType.STRING:
                return str(value)
            elif self.type == VariableType.NUMBER:
                if isinstance(value, str):
                    if "." in value:
                        return float(value)
                    return int(value)
                return float(value) if isinstance(value, (int, float)) else 0
            elif self.type == VariableType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif self.type == VariableType.ARRAY:
                if isinstance(value, str):
                    return json.loads(value)
                return list(value) if not isinstance(value, list) else value
            elif self.type == VariableType.OBJECT:
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value) if not isinstance(value, dict) else value
            elif self.type == VariableType.SECRET:
                return str(value)
        except Exception:
            return self.default
        return value


class VariableManager:
    """
    变量管理器。
    
    管理变量的定义、存储和访问。
    支持不同作用域的变量。
    """
    
    # 有效的变量名正则表达式
    VALID_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    def __init__(self):
        """初始化变量管理器。"""
        self._variables: Dict[str, Variable] = {}
        self._global_variables: Dict[str, Variable] = {}
        self._step_variables: Dict[str, Dict[str, Variable]] = {}
        self._workflow_id: Optional[str] = None
    
    def set_workflow_context(self, workflow_id: str) -> None:
        """
        设置当前工作流上下文。
        
        Args:
            workflow_id: 工作流ID
        """
        self._workflow_id = workflow_id
        if workflow_id not in self._step_variables:
            self._step_variables[workflow_id] = {}
    
    def clear_workflow_context(self) -> None:
        """清除工作流上下文。"""
        if self._workflow_id and self._workflow_id in self._step_variables:
            del self._step_variables[self._workflow_id]
        self._workflow_id = None
    
    def _validate_name(self, name: str) -> None:
        """验证变量名是否有效。"""
        if not name:
            raise ValueError("变量名不能为空")
        if not self.VALID_NAME_PATTERN.match(name):
            raise ValueError(f"无效的变量名: {name}。变量名必须以字母或下划线开头，只能包含字母、数字和下划线")
    
    def define_variable(
        self,
        name: str,
        var_type: Union[VariableType, str] = VariableType.STRING,
        default: Any = None,
        description: str = "",
        is_secret: bool = False,
        scope: Union[VariableScope, str] = VariableScope.WORKFLOW,
    ) -> Variable:
        """
        定义变量。
        
        Args:
            name: 变量名称
            var_type: 变量类型
            default: 默认值
            description: 变量描述
            is_secret: 是否为敏感变量
            scope: 变量作用域
            
        Returns:
            定义的变量
        """
        self._validate_name(name)
        
        if isinstance(var_type, str):
            var_type = VariableType(var_type)
        if isinstance(scope, str):
            scope = VariableScope(scope)
        
        variable = Variable(
            name=name,
            type=var_type,
            default=default,
            description=description,
            is_secret=is_secret or var_type == VariableType.SECRET,
            scope=scope,
        )
        
        if scope == VariableScope.GLOBAL:
            self._global_variables[name] = variable
        elif scope == VariableScope.STEP:
            if not self._workflow_id:
                raise RuntimeError("设置步骤作用域变量前必须先设置工作流上下文")
            self._step_variables[self._workflow_id][name] = variable
        else:
            self._variables[name] = variable
        
        return variable
    
    def set_variable(self, name: str, value: Any) -> Variable:
        """
        设置变量值。
        
        Args:
            name: 变量名称
            value: 变量值
            
        Returns:
            更新后的变量
        """
        self._validate_name(name)
        
        # 查找变量
        variable = self._find_variable(name)
        
        if variable is None:
            # 自动创建变量（默认为string类型）
            variable = self.define_variable(name, default=value)
        
        # 类型转换和验证
        casted_value = variable.cast_value(value)
        if not variable.validate_type(casted_value):
            raise TypeError(f"值类型与变量类型不匹配: {name} 期望 {variable.type.value}")
        
        variable.value = casted_value
        return variable
    
    def get_variable(self, name: str) -> Any:
        """
        获取变量值。
        
        Args:
            name: 变量名称
            
        Returns:
            变量值，变量不存在则返回None
        """
        variable = self._find_variable(name)
        return variable.value if variable else None
    
    def get_variable_object(self, name: str) -> Optional[Variable]:
        """
        获取变量对象。
        
        Args:
            name: 变量名称
            
        Returns:
            变量对象，不存在则返回None
        """
        return self._find_variable(name)
    
    def _find_variable(self, name: str) -> Optional[Variable]:
        """查找变量（按作用域优先级）。"""
        # 步骤作用域优先
        if self._workflow_id and self._workflow_id in self._step_variables:
            if name in self._step_variables[self._workflow_id]:
                return self._step_variables[self._workflow_id][name]
        
        # 然后是工作流作用域
        if name in self._variables:
            return self._variables[name]
        
        # 最后是全局作用域
        if name in self._global_variables:
            return self._global_variables[name]
        
        return None
    
    def list_variables(
        self,
        scope: Optional[Union[VariableScope, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        列出所有变量。
        
        Args:
            scope: 可选的作用域过滤
            
        Returns:
            变量信息列表
        """
        if isinstance(scope, str):
            scope = VariableScope(scope)
        
        variables = []
        
        # 全局变量
        if scope is None or scope == VariableScope.GLOBAL:
            for var in self._global_variables.values():
                variables.append(var.to_dict())
        
        # 工作流变量
        if scope is None or scope == VariableScope.WORKFLOW:
            for var in self._variables.values():
                variables.append(var.to_dict())
        
        # 步骤变量
        if scope is None or scope == VariableScope.STEP:
            if self._workflow_id and self._workflow_id in self._step_variables:
                for var in self._step_variables[self._workflow_id].values():
                    variables.append(var.to_dict())
        
        return variables
    
    def delete_variable(self, name: str) -> bool:
        """
        删除变量。
        
        Args:
            name: 变量名称
            
        Returns:
            是否成功删除
        """
        # 按作用域顺序查找并删除
        if self._workflow_id and self._workflow_id in self._step_variables:
            if name in self._step_variables[self._workflow_id]:
                del self._step_variables[self._workflow_id][name]
                return True
        
        if name in self._variables:
            del self._variables[name]
            return True
        
        if name in self._global_variables:
            del self._global_variables[name]
            return True
        
        return False
    
    def clear_variables(self, scope: Optional[Union[VariableScope, str]] = None) -> None:
        """
        清除变量。
        
        Args:
            scope: 可选的作用域过滤，不提供则清除当前上下文的所有变量
        """
        if isinstance(scope, str):
            scope = VariableScope(scope)
        
        if scope is None:
            # 清除当前工作流的所有变量
            self._variables.clear()
            if self._workflow_id and self._workflow_id in self._step_variables:
                self._step_variables[self._workflow_id].clear()
        elif scope == VariableScope.GLOBAL:
            self._global_variables.clear()
        elif scope == VariableScope.WORKFLOW:
            self._variables.clear()
        elif scope == VariableScope.STEP:
            if self._workflow_id and self._workflow_id in self._step_variables:
                self._step_variables[self._workflow_id].clear()
    
    def interpolate_string(self, text: str) -> str:
        """
        字符串插值，替换变量占位符。
        
        支持格式: ${variable_name} 或 $variable_name
        
        Args:
            text: 包含变量占位符的字符串
            
        Returns:
            插值后的字符串
        """
        if not text:
            return text
        
        # 处理 ${variable} 格式
        def replace_braced(match):
            var_name = match.group(1)
            value = self.get_variable(var_name)
            return str(value) if value is not None else match.group(0)
        
        # 处理 $variable 格式（变量名后紧跟非字母数字下划线）
        def replace_unbraced(match):
            var_name = match.group(1)
            value = self.get_variable(var_name)
            return str(value) if value is not None else match.group(0)
        
        result = re.sub(r'\$\{([^}]+)\}', replace_braced, text)
        result = re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)', replace_unbraced, result)
        
        return result
    
    def export_variables(self, scope: Optional[Union[VariableScope, str]] = None) -> str:
        """
        导出变量为JSON字符串。
        
        Args:
            scope: 可选的作用域过滤
            
        Returns:
            JSON字符串
        """
        variables = self.list_variables(scope)
        return json.dumps(variables, ensure_ascii=False, indent=2)
    
    def import_variables(self, json_str: str) -> List[Variable]:
        """
        从JSON字符串导入变量。
        
        Args:
            json_str: JSON字符串
            
        Returns:
            导入的变量列表
        """
        data = json.loads(json_str)
        variables = []
        
        for var_data in data:
            variable = Variable.from_dict(var_data)
            
            # 根据作用域存储
            if variable.scope == VariableScope.GLOBAL:
                self._global_variables[variable.name] = variable
            elif variable.scope == VariableScope.STEP:
                if not self._workflow_id:
                    self._workflow_id = "default"
                if self._workflow_id not in self._step_variables:
                    self._step_variables[self._workflow_id] = {}
                self._step_variables[self._workflow_id][variable.name] = variable
            else:
                self._variables[variable.name] = variable
            
            variables.append(variable)
        
        return variables
    
    def get_variables_dict(self, scope: Optional[Union[VariableScope, str]] = None) -> Dict[str, Any]:
        """
        获取变量字典（用于工作流状态）。
        
        Args:
            scope: 可选的作用域过滤
            
        Returns:
            变量名到值的字典
        """
        variables = self.list_variables(scope)
        return {var["name"]: var["value"] for var in variables}
    
    def copy_variables_from(self, other: "VariableManager") -> None:
        """
        从其他变量管理器复制变量。
        
        Args:
            other: 源变量管理器
        """
        for name, var in other._global_variables.items():
            self._global_variables[name] = Variable.from_dict(var.to_dict_full())
        
        for name, var in other._variables.items():
            self._variables[name] = Variable.from_dict(var.to_dict_full())


# 便捷函数
def create_variable(
    name: str,
    var_type: str = "string",
    default: Any = None,
    description: str = "",
) -> Variable:
    """
    创建变量的便捷函数。
    
    Args:
        name: 变量名称
        var_type: 变量类型
        default: 默认值
        description: 变量描述
        
    Returns:
        创建的变量
    """
    return Variable(
        name=name,
        type=VariableType(var_type),
        default=default,
        description=description,
    )
