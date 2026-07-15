"""
配置完整性校验器模块

提供全面的配置校验能力，包括必填项检查、类型检查、
数值范围检查、配置项间依赖关系检查和废弃配置项警告。
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .schema import ConfigField, ConfigSchema, NestedConfig


class ValidationError(Exception):
    """配置验证错误异常。

    Args:
        message: 错误描述信息
        field: 出错字段名称
        value: 导致错误的值
    """

    def __init__(
        self,
        message: str,
        field: str = "",
        value: Any = None,
    ):
        self.field = field
        self.value = value
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"ValidationError(field={self.field!r}, "
            f"value={self.value!r}, message={self.args[0]!r})"
        )


class ValidationReport:
    """校验报告。

    收集校验过程中的所有错误和警告信息。

    Args:
        errors: 错误信息列表
        warnings: 警告信息列表
    """

    def __init__(
        self,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.errors = errors or []
        self.warnings = warnings or []

    @property
    def is_valid(self) -> bool:
        """是否校验通过（无错误）。"""
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        """添加一条错误信息。"""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """添加一条警告信息。"""
        self.warnings.append(message)

    def merge(self, other: "ValidationReport") -> None:
        """合并另一个校验报告。"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def get_summary(self) -> str:
        """获取校验报告摘要。"""
        lines = []
        if self.is_valid:
            lines.append("配置校验通过。")
        else:
            lines.append(f"配置校验失败，共 {len(self.errors)} 个错误。")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"  错误 {i}: {err}")

        if self.warnings:
            lines.append(f"共 {len(self.warnings)} 个警告。")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"  警告 {i}: {warn}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ValidationReport(errors={len(self.errors)}, "
            f"warnings={len(self.warnings)})"
        )

    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """配置完整性校验器。

    提供全面的配置校验能力，可基于Schema校验，也可独立使用。

    Args:
        schema: 可选的配置Schema定义
        strict: 是否启用严格模式（未知字段报错）
    """

    def __init__(
        self,
        schema: Optional[ConfigSchema] = None,
        strict: bool = False,
    ):
        self.schema = schema
        self.strict = strict
        self._dependencies: List[Dict[str, Any]] = []
        self._deprecated_fields: Dict[str, str] = {}
        self._custom_validators: List[Dict[str, Any]] = []

    def set_schema(self, schema: ConfigSchema) -> None:
        """设置配置Schema。"""
        self.schema = schema

    def add_dependency(
        self,
        field: str,
        depends_on: str,
        condition: Optional[Callable[[Any], bool]] = None,
        message: str = "",
    ) -> None:
        """添加配置项依赖关系。

        当 field 存在且非None时，depends_on 也必须存在且非None。
        可选的 condition 函数用于更精细的条件判断。

        Args:
            field: 源字段名
            depends_on: 依赖字段名
            condition: 可选条件函数，返回True表示依赖生效
            message: 自定义错误信息
        """
        if not message:
            message = f"字段 '{field}' 需要 '{depends_on}' 同时配置"
        self._dependencies.append({
            "field": field,
            "depends_on": depends_on,
            "condition": condition,
            "message": message,
        })

    def add_deprecated(
        self,
        field: str,
        replacement: str = "",
        message: str = "",
    ) -> None:
        """标记废弃的配置字段。

        Args:
            field: 废弃字段名
            replacement: 替代字段名
            message: 自定义警告信息
        """
        if not message:
            if replacement:
                message = f"字段 '{field}' 已废弃，请使用 '{replacement}' 代替"
            else:
                message = f"字段 '{field}' 已废弃，将在未来版本中移除"
        self._deprecated_fields[field] = message

    def add_custom_validator(
        self,
        name: str,
        validator: Callable[[Dict[str, Any]], Tuple[bool, str]],
    ) -> None:
        """添加自定义校验函数。

        Args:
            name: 校验器名称
            validator: 校验函数，接收配置字典，返回 (是否通过, 错误信息)
        """
        self._custom_validators.append({
            "name": name,
            "validator": validator,
        })

    def validate(self, config: Dict[str, Any]) -> ValidationReport:
        """执行完整的配置校验。

        依次执行：
        1. 基础结构检查
        2. 必填项检查
        3. 类型检查
        4. 数值范围检查
        5. 依赖关系检查
        6. 废弃字段检查
        7. 自定义校验器

        Args:
            config: 待校验的配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        # 基础结构检查
        report.merge(self._validate_structure(config))

        # 基于Schema的校验
        if self.schema is not None:
            report.merge(self._validate_against_schema(config))

        # 必填项检查
        report.merge(self.validate_required(config))

        # 类型检查
        report.merge(self.validate_types(config))

        # 数值范围检查
        report.merge(self.validate_ranges(config))

        # 依赖关系检查
        report.merge(self.validate_dependencies(config))

        # 废弃字段检查
        report.merge(self.validate_deprecated(config))

        # 自定义校验器
        report.merge(self._validate_custom(config))

        return report

    def validate_required(self, config: Dict[str, Any]) -> ValidationReport:
        """检查必填项。

        如果设置了Schema，则根据Schema中的required字段检查。
        否则检查所有值为None的配置项。

        Args:
            config: 配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        if self.schema is not None:
            for field in self.schema.fields:
                if isinstance(field, ConfigField):
                    if field.required and field.name not in config:
                        report.add_error(
                            f"缺少必填字段 '{field.name}'"
                        )
                    elif (
                        field.required
                        and field.name in config
                        and config[field.name] is None
                    ):
                        report.add_error(
                            f"必填字段 '{field.name}' 的值不能为 None"
                        )
                elif isinstance(field, NestedConfig):
                    if field.required and field.name not in config:
                        report.add_error(
                            f"缺少必填配置组 '{field.name}'"
                        )

        return report

    def validate_types(self, config: Dict[str, Any]) -> ValidationReport:
        """类型检查。

        如果设置了Schema，则根据Schema中的字段类型检查。
        否则跳过类型检查。

        Args:
            config: 配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        if self.schema is None:
            return report

        for field in self.schema.fields:
            if isinstance(field, ConfigField):
                if field.name not in config:
                    continue
                value = config[field.name]
                if value is None:
                    continue

                expected_type = field.field_type
                actual_type = type(value)

                # 允许 int 赋值给 float 字段
                if expected_type is float and isinstance(value, int):
                    continue

                if not isinstance(value, expected_type):
                    report.add_error(
                        f"字段 '{field.name}' 类型错误："
                        f"期望 {expected_type.__name__}，"
                        f"实际为 {actual_type.__name__}（值: {value!r}）"
                    )

            elif isinstance(field, NestedConfig):
                if field.name not in config:
                    continue
                nested_value = config[field.name]
                if nested_value is None:
                    continue
                if not isinstance(nested_value, dict):
                    report.add_error(
                        f"配置组 '{field.name}' 必须是字典类型，"
                        f"实际为 {type(nested_value).__name__}"
                    )

        return report

    def validate_ranges(self, config: Dict[str, Any]) -> ValidationReport:
        """数值范围检查。

        检查所有数值类型字段是否在允许的范围内。

        Args:
            config: 配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        if self.schema is None:
            return report

        for field in self.schema.fields:
            if isinstance(field, ConfigField):
                if field.name not in config:
                    continue
                value = config[field.name]
                if value is None:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                if field.range is None:
                    continue

                min_val, max_val = field.range
                if min_val is not None and value < min_val:
                    report.add_error(
                        f"字段 '{field.name}' 值 {value} 小于最小值 {min_val}"
                    )
                if max_val is not None and value > max_val:
                    report.add_error(
                        f"字段 '{field.name}' 值 {value} 大于最大值 {max_val}"
                    )

            elif isinstance(field, NestedConfig):
                if field.name not in config:
                    continue
                nested_value = config[field.name]
                if not isinstance(nested_value, dict):
                    continue
                # 递归检查嵌套字段
                nested_report = self._validate_nested_ranges(
                    field, nested_value
                )
                report.merge(nested_report)

        return report

    def validate_dependencies(
        self, config: Dict[str, Any]
    ) -> ValidationReport:
        """配置项间依赖关系检查。

        检查已注册的依赖关系是否满足。

        Args:
            config: 配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        for dep in self._dependencies:
            field = dep["field"]
            depends_on = dep["depends_on"]
            condition = dep["condition"]
            message = dep["message"]

            # 获取字段值
            field_value = self._get_nested_value(config, field)

            # 如果字段不存在或为None，跳过依赖检查
            if field_value is None:
                continue

            # 如果有条件函数，检查条件
            if condition is not None:
                try:
                    if not condition(field_value):
                        continue
                except Exception:
                    continue

            # 检查依赖字段
            depends_value = self._get_nested_value(config, depends_on)
            if depends_value is None:
                report.add_error(message)

        return report

    def validate_deprecated(
        self, config: Dict[str, Any]
    ) -> ValidationReport:
        """废弃配置项检查。

        检查配置中是否使用了已废弃的字段，发出警告。

        Args:
            config: 配置字典

        Returns:
            ValidationReport 校验报告
        """
        report = ValidationReport()

        for field, message in self._deprecated_fields.items():
            value = self._get_nested_value(config, field)
            if value is not None:
                report.add_warning(message)

        return report

    def _validate_structure(
        self, config: Dict[str, Any]
    ) -> ValidationReport:
        """基础结构检查。"""
        report = ValidationReport()

        if not isinstance(config, dict):
            report.add_error(
                f"配置必须是字典类型，实际为 {type(config).__name__}"
            )
            return report

        return report

    def _validate_against_schema(
        self, config: Dict[str, Any]
    ) -> ValidationReport:
        """基于Schema的完整校验。"""
        report = ValidationReport()

        if self.schema is None:
            return report

        # 检查未知字段（严格模式）
        if self.strict:
            known_fields = set()
            for field in self.schema.fields:
                known_fields.add(field.name)

            for key in config:
                if key not in known_fields:
                    report.add_warning(
                        f"未知配置字段 '{key}'"
                    )

        # 使用Schema的validate方法
        ok, errors = self.schema.validate(config)
        for err in errors:
            report.add_error(err)

        return report

    def _validate_nested_ranges(
        self,
        nested: NestedConfig,
        config: Dict[str, Any],
    ) -> ValidationReport:
        """递归校验嵌套配置组的数值范围。"""
        report = ValidationReport()

        for field in nested.fields:
            if isinstance(field, ConfigField):
                if field.name not in config:
                    continue
                value = config[field.name]
                if value is None or not isinstance(value, (int, float)):
                    continue
                if field.range is None:
                    continue

                min_val, max_val = field.range
                full_name = f"{nested.name}.{field.name}"
                if min_val is not None and value < min_val:
                    report.add_error(
                        f"字段 '{full_name}' 值 {value} 小于最小值 {min_val}"
                    )
                if max_val is not None and value > max_val:
                    report.add_error(
                        f"字段 '{full_name}' 值 {value} 大于最大值 {max_val}"
                    )

            elif isinstance(field, NestedConfig):
                if field.name not in config:
                    continue
                nested_value = config[field.name]
                if isinstance(nested_value, dict):
                    sub_report = self._validate_nested_ranges(
                        field, nested_value
                    )
                    report.merge(sub_report)

        return report

    def _validate_custom(
        self, config: Dict[str, Any]
    ) -> ValidationReport:
        """执行自定义校验器。"""
        report = ValidationReport()

        for cv in self._custom_validators:
            name = cv["name"]
            validator = cv["validator"]
            try:
                ok, msg = validator(config)
                if not ok:
                    report.add_error(f"[{name}] {msg}")
            except Exception as e:
                report.add_error(
                    f"自定义校验器 '{name}' 执行异常: {e}"
                )

        return report

    @staticmethod
    def _get_nested_value(
        config: Dict[str, Any], key_path: str
    ) -> Any:
        """从嵌套字典中获取值。

        Args:
            config: 配置字典
            key_path: 用点号分隔的键路径，如 "database.host"

        Returns:
            对应的值，不存在则返回 None
        """
        parts = key_path.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def __repr__(self) -> str:
        schema_info = f"schema={self.schema!r}" if self.schema else "schema=None"
        return (
            f"ConfigValidator({schema_info}, strict={self.strict}, "
            f"dependencies={len(self._dependencies)}, "
            f"deprecated={len(self._deprecated_fields)}, "
            f"custom_validators={len(self._custom_validators)})"
        )
