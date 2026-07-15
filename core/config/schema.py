"""
配置Schema定义模块

提供配置结构的声明式定义能力，支持嵌套结构、类型检查、
范围约束、枚举约束和必填检查。
"""

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# 支持的配置值类型
VALID_TYPES = (int, float, str, bool, list, dict)


class ConfigField:
    """配置字段定义。

    描述配置中单个字段的元信息，包括类型、默认值、约束条件等。

    Args:
        name: 字段名称
        field_type: 字段类型，支持 int/float/str/bool/list/dict
        default: 默认值，默认为 None
        required: 是否必填，默认为 False
        validator: 自定义验证函数，接收值返回 (bool, str) 元组
        range: 数值范围约束，格式为 (min, max)，None 表示不限制
        choices: 枚举值列表，值必须在此列表中
        description: 字段描述信息
    """

    def __init__(
        self,
        name: str,
        field_type: type = str,
        default: Any = None,
        required: bool = False,
        validator: Optional[Callable[[Any], Tuple[bool, str]]] = None,
        range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        choices: Optional[List[Any]] = None,
        description: str = "",
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"字段名称必须是非空字符串， got: {name!r}")
        if field_type not in VALID_TYPES:
            raise ValueError(
                f"不支持的类型 '{field_type.__name__}'，"
                f"支持的类型: {[t.__name__ for t in VALID_TYPES]}"
            )

        self.name = name
        self.field_type = field_type
        self.default = default
        self.required = required
        self.validator = validator
        self.range = range
        self.choices = choices
        self.description = description

    def validate(self, value: Any) -> Tuple[bool, str]:
        """验证给定值是否符合此字段的约束。

        Args:
            value: 待验证的值

        Returns:
            (是否通过, 错误信息) 元组
        """
        # 检查 None 值
        if value is None:
            if self.required:
                return False, f"字段 '{self.name}' 是必填项，不能为 None"
            return True, ""

        # 类型检查
        if not isinstance(value, self.field_type):
            # 允许 int 赋值给 float 字段
            if self.field_type is float and isinstance(value, int):
                pass
            else:
                return False, (
                    f"字段 '{self.name}' 类型错误：期望 {self.field_type.__name__}，"
                    f"实际为 {type(value).__name__}"
                )

        # 范围检查（仅对数值类型）
        if self.range is not None and isinstance(value, (int, float)):
            min_val, max_val = self.range
            if min_val is not None and value < min_val:
                return False, (
                    f"字段 '{self.name}' 值 {value} 小于最小值 {min_val}"
                )
            if max_val is not None and value > max_val:
                return False, (
                    f"字段 '{self.name}' 值 {value} 大于最大值 {max_val}"
                )

        # 枚举检查
        if self.choices is not None and value not in self.choices:
            return False, (
                f"字段 '{self.name}' 值 '{value}' 不在允许的选项中: {self.choices}"
            )

        # 自定义验证器
        if self.validator is not None:
            try:
                ok, msg = self.validator(value)
                if not ok:
                    return False, f"字段 '{self.name}' 自定义验证失败: {msg}"
            except Exception as e:
                return False, f"字段 '{self.name}' 自定义验证器异常: {e}"

        return True, ""

    def get_default(self) -> Any:
        """获取字段的默认值（返回深拷贝以避免引用共享）。"""
        return copy.deepcopy(self.default)

    def __repr__(self) -> str:
        range_str = f", range={self.range}" if self.range else ""
        choices_str = f", choices={self.choices}" if self.choices else ""
        required_str = ", required=True" if self.required else ""
        return (
            f"ConfigField(name={self.name!r}, type={self.field_type.__name__}"
            f"{required_str}{range_str}{choices_str})"
        )


class NestedConfig:
    """嵌套配置组定义。

    将多个字段组织为一个命名的配置组，支持嵌套结构。

    Args:
        name: 配置组名称
        fields: 字段列表，可以是 ConfigField 或 NestedConfig
        description: 配置组描述信息
        required: 是否必填，默认为 False
    """

    def __init__(
        self,
        name: str,
        fields: Optional[List[Union["ConfigField", "NestedConfig"]]] = None,
        description: str = "",
        required: bool = False,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"配置组名称必须是非空字符串， got: {name!r}")

        self.name = name
        self.fields = fields or []
        self.description = description
        self.required = required

    def add_field(self, field: Union[ConfigField, "NestedConfig"]) -> None:
        """添加一个字段或嵌套配置组。"""
        if not isinstance(field, (ConfigField, NestedConfig)):
            raise TypeError(
                f"只能添加 ConfigField 或 NestedConfig，got: {type(field).__name__}"
            )
        self.fields.append(field)

    def get_field(self, name: str) -> Optional[Union[ConfigField, "NestedConfig"]]:
        """按名称查找字段。"""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_all_field_names(self) -> List[str]:
        """获取所有字段名称（包括嵌套字段，用点号分隔）。"""
        names = []
        for field in self.fields:
            if isinstance(field, NestedConfig):
                for sub_name in field.get_all_field_names():
                    names.append(f"{field.name}.{sub_name}")
            else:
                names.append(field.name)
        return names

    def validate(self, value: Any) -> Tuple[bool, str]:
        """验证给定值是否符合此嵌套配置组的约束。"""
        if value is None:
            if self.required:
                return False, f"配置组 '{self.name}' 是必填项，不能为 None"
            return True, ""

        if not isinstance(value, dict):
            return False, (
                f"配置组 '{self.name}' 必须是字典类型，"
                f"实际为 {type(value).__name__}"
            )

        for field in self.fields:
            if isinstance(field, ConfigField):
                if field.name in value:
                    ok, msg = field.validate(value[field.name])
                    if not ok:
                        return False, msg
                elif field.required:
                    return False, f"配置组 '{self.name}' 中缺少必填字段 '{field.name}'"
            elif isinstance(field, NestedConfig):
                if field.name in value:
                    ok, msg = field.validate(value[field.name])
                    if not ok:
                        return False, msg
                elif field.required:
                    return False, f"配置组 '{self.name}' 中缺少必填子组 '{field.name}'"

        return True, ""

    def __repr__(self) -> str:
        return (
            f"NestedConfig(name={self.name!r}, "
            f"fields_count={len(self.fields)})"
        )


class ConfigSchema:
    """配置Schema定义。

    顶层配置结构描述，包含多个字段和嵌套配置组。

    Args:
        name: Schema名称
        version: Schema版本号
        fields: 顶层字段列表
        description: Schema描述信息
    """

    def __init__(
        self,
        name: str = "default",
        version: str = "1.0.0",
        fields: Optional[List[Union[ConfigField, NestedConfig]]] = None,
        description: str = "",
    ):
        self.name = name
        self.version = version
        self.fields = fields or []
        self.description = description

    def add_field(self, field: Union[ConfigField, NestedConfig]) -> None:
        """添加一个顶层字段或嵌套配置组。"""
        if not isinstance(field, (ConfigField, NestedConfig)):
            raise TypeError(
                f"只能添加 ConfigField 或 NestedConfig，got: {type(field).__name__}"
            )
        self.fields.append(field)

    def get_field(self, name: str) -> Optional[Union[ConfigField, NestedConfig]]:
        """按名称查找顶层字段。"""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_all_field_names(self) -> List[str]:
        """获取所有字段名称（包括嵌套字段，用点号分隔）。"""
        names = []
        for field in self.fields:
            if isinstance(field, NestedConfig):
                for sub_name in field.get_all_field_names():
                    names.append(f"{field.name}.{sub_name}")
            else:
                names.append(field.name)
        return names

    def get_defaults(self) -> Dict[str, Any]:
        """获取所有字段的默认值组成的字典。"""
        defaults = {}
        for field in self.fields:
            if isinstance(field, ConfigField):
                defaults[field.name] = field.get_default()
            elif isinstance(field, NestedConfig):
                nested_defaults = _get_nested_defaults(field)
                defaults[field.name] = nested_defaults
        return defaults

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """根据此Schema验证配置字典。

        Args:
            config: 待验证的配置字典

        Returns:
            (是否全部通过, 错误信息列表) 元组
        """
        return validate_config(config, self)

    def __repr__(self) -> str:
        return (
            f"ConfigSchema(name={self.name!r}, version={self.version!r}, "
            f"fields_count={len(self.fields)})"
        )


def _get_nested_defaults(nested: NestedConfig) -> Dict[str, Any]:
    """递归获取嵌套配置组的默认值。"""
    defaults = {}
    for field in nested.fields:
        if isinstance(field, ConfigField):
            defaults[field.name] = field.get_default()
        elif isinstance(field, NestedConfig):
            defaults[field.name] = _get_nested_defaults(field)
    return defaults


def validate_config(
    config: Dict[str, Any],
    schema: ConfigSchema,
) -> Tuple[bool, List[str]]:
    """根据Schema验证配置字典。

    执行以下检查：
    1. 必填项检查
    2. 类型检查
    3. 范围检查
    4. 枚举检查
    5. 自定义验证器检查

    Args:
        config: 待验证的配置字典
        schema: 配置Schema定义

    Returns:
        (是否全部通过, 错误信息列表) 元组
    """
    errors = []

    if not isinstance(config, dict):
        errors.append(f"配置必须是字典类型，实际为 {type(config).__name__}")
        return False, errors

    for field in schema.fields:
        if isinstance(field, ConfigField):
            # 必填检查
            if field.name not in config:
                if field.required:
                    errors.append(
                        f"缺少必填字段 '{field.name}'"
                    )
                continue

            # 字段验证
            ok, msg = field.validate(config[field.name])
            if not ok:
                errors.append(msg)

        elif isinstance(field, NestedConfig):
            # 嵌套配置组验证
            if field.name not in config:
                if field.required:
                    errors.append(
                        f"缺少必填配置组 '{field.name}'"
                    )
                continue

            nested_value = config[field.name]
            ok, msg = field.validate(nested_value)
            if not ok:
                errors.append(msg)

    # 检查未知字段并发出警告（不作为错误）
    known_fields = set(schema.get_all_field_names())
    # 提取顶层字段名
    top_level_names = set()
    for field in schema.fields:
        top_level_names.add(field.name)

    for key in config:
        if key not in top_level_names:
            # 不报错，但可以在扩展中使用
            pass

    return len(errors) == 0, errors


def build_schema_from_dict(
    schema_dict: Dict[str, Any],
) -> ConfigSchema:
    """从字典构建ConfigSchema。

    支持的字典格式：
    {
        "name": "schema_name",
        "version": "1.0.0",
        "description": "...",
        "fields": [
            {
                "name": "field_name",
                "type": "str",
                "default": "value",
                "required": true,
                "description": "..."
            },
            {
                "name": "nested_group",
                "type": "nested",
                "fields": [...],
                "required": false
            }
        ]
    }

    Args:
        schema_dict: Schema描述字典

    Returns:
        ConfigSchema 实例
    """
    TYPE_MAP = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    name = schema_dict.get("name", "default")
    version = schema_dict.get("version", "1.0.0")
    description = schema_dict.get("description", "")
    fields = []

    for field_def in schema_dict.get("fields", []):
        field_name = field_def["name"]
        field_type_str = field_def.get("type", "str")

        if field_type_str == "nested":
            nested = NestedConfig(
                name=field_name,
                description=field_def.get("description", ""),
                required=field_def.get("required", False),
            )
            for sub_def in field_def.get("fields", []):
                sub_field = _build_field_from_dict(sub_def, TYPE_MAP)
                nested.add_field(sub_field)
            fields.append(nested)
        else:
            field = _build_field_from_dict(field_def, TYPE_MAP)
            fields.append(field)

    return ConfigSchema(
        name=name,
        version=version,
        fields=fields,
        description=description,
    )


def _build_field_from_dict(
    field_def: Dict[str, Any],
    type_map: Dict[str, type],
) -> ConfigField:
    """从字典定义构建ConfigField。"""
    field_type = type_map.get(field_def.get("type", "str"), str)
    range_val = field_def.get("range")
    choices = field_def.get("choices")

    return ConfigField(
        name=field_def["name"],
        field_type=field_type,
        default=field_def.get("default"),
        required=field_def.get("required", False),
        range=tuple(range_val) if range_val else None,
        choices=choices,
        description=field_def.get("description", ""),
    )
