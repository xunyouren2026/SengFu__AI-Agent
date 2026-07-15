"""
配置版本迁移工具模块

提供配置版本间的迁移能力，支持字段重命名、默认值设置、
值转换和字段删除等操作。
"""

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


class MigrationError(Exception):
    """配置迁移错误异常。

    Args:
        message: 错误描述
        version_from: 源版本
        version_to: 目标版本
    """

    def __init__(
        self,
        message: str,
        version_from: str = "",
        version_to: str = "",
    ):
        self.version_from = version_from
        self.version_to = version_to
        full_message = message
        if version_from and version_to:
            full_message = (
                f"迁移 {version_from} -> {version_to} 失败: {message}"
            )
        super().__init__(full_message)

    def __repr__(self) -> str:
        return (
            f"MigrationError("
            f"version_from={self.version_from!r}, "
            f"version_to={self.version_to!r}, "
            f"message={self.args[0]!r})"
        )


class MigrationRule:
    """迁移规则定义。

    描述从一个版本到另一个版本的配置迁移方式。

    Args:
        version_from: 源版本号
        version_to: 目标版本号
        field_mapping: 字段重命名映射 {旧名: 新名}
        default_values: 需要设置的默认值 {字段名: 默认值}
        transformations: 字段值转换函数 {字段名: 转换函数}
        removed_fields: 需要删除的字段列表
        description: 迁移描述
    """

    def __init__(
        self,
        version_from: str,
        version_to: str,
        field_mapping: Optional[Dict[str, str]] = None,
        default_values: Optional[Dict[str, Any]] = None,
        transformations: Optional[Dict[str, Callable[[Any], Any]]] = None,
        removed_fields: Optional[List[str]] = None,
        description: str = "",
    ):
        if not version_from or not version_to:
            raise ValueError("版本号不能为空")
        if version_from == version_to:
            raise ValueError("源版本和目标版本不能相同")

        self.version_from = version_from
        self.version_to = version_to
        self.field_mapping = field_mapping or {}
        self.default_values = default_values or {}
        self.transformations = transformations or {}
        self.removed_fields = removed_fields or []
        self.description = description

    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """将此迁移规则应用到配置字典。

        执行顺序：
        1. 字段重命名
        2. 值转换
        3. 设置默认值
        4. 删除废弃字段
        5. 更新版本号

        Args:
            config: 配置字典

        Returns:
            迁移后的配置字典
        """
        result = copy.deepcopy(config)

        # 1. 字段重命名
        for old_name, new_name in self.field_mapping.items():
            if old_name in result:
                value = result.pop(old_name)
                # 处理嵌套路径（如 "database.host" -> "db.host"）
                self._set_nested_value(result, new_name, value)

        # 2. 值转换
        for field_name, transform_fn in self.transformations.items():
            value = self._get_nested_value(result, field_name)
            if value is not None:
                try:
                    new_value = transform_fn(value)
                    self._set_nested_value(result, field_name, new_value)
                except Exception as e:
                    raise MigrationError(
                        f"字段 '{field_name}' 值转换失败: {e}",
                        self.version_from,
                        self.version_to,
                    )

        # 3. 设置默认值（仅在字段不存在时设置）
        for field_name, default_value in self.default_values.items():
            existing = self._get_nested_value(result, field_name)
            if existing is None:
                self._set_nested_value(result, field_name, default_value)

        # 4. 删除废弃字段
        for field_name in self.removed_fields:
            self._delete_nested_value(result, field_name)

        # 5. 更新版本号
        result["version"] = self.version_to

        return result

    @staticmethod
    def _get_nested_value(
        config: Dict[str, Any], key_path: str
    ) -> Any:
        """从嵌套字典中获取值。"""
        parts = key_path.split(".")
        current = config
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _set_nested_value(
        config: Dict[str, Any], key_path: str, value: Any
    ) -> None:
        """在嵌套字典中设置值。"""
        parts = key_path.split(".")
        current = config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    @staticmethod
    def _delete_nested_value(
        config: Dict[str, Any], key_path: str
    ) -> bool:
        """从嵌套字典中删除值。"""
        parts = key_path.split(".")
        current = config
        for part in parts[:-1]:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]

        last_part = parts[-1]
        if isinstance(current, dict) and last_part in current:
            del current[last_part]
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"MigrationRule("
            f"version_from={self.version_from!r}, "
            f"version_to={self.version_to!r}, "
            f"field_mappings={len(self.field_mapping)}, "
            f"defaults={len(self.default_values)}, "
            f"transforms={len(self.transformations)}, "
            f"removed={len(self.removed_fields)}"
            f")"
        )


class ConfigMigrator:
    """配置版本迁移工具。

    管理和执行配置版本间的迁移。

    Args:
        version_key: 配置中存储版本号的字段名，默认为 "version"
    """

    def __init__(self, version_key: str = "version"):
        self.version_key = version_key
        self._rules: Dict[str, MigrationRule] = {}

    def register_migration(self, rule: MigrationRule) -> None:
        """注册迁移规则。

        Args:
            rule: 迁移规则

        Raises:
            MigrationError: 如果版本范围冲突
        """
        rule_key = f"{rule.version_from}->{rule.version_to}"

        if rule_key in self._rules:
            raise MigrationError(
                f"已存在从 {rule.version_from} 到 {rule.version_to} 的迁移规则",
                rule.version_from,
                rule.version_to,
            )

        self._rules[rule_key] = rule

    def migrate(
        self,
        config: Dict[str, Any],
        target_version: str = "",
    ) -> Dict[str, Any]:
        """执行配置迁移。

        自动检测当前版本，并按顺序应用所有必要的迁移规则。

        Args:
            config: 配置字典
            target_version: 目标版本，默认迁移到最新版本

        Returns:
            迁移后的配置字典

        Raises:
            MigrationError: 迁移失败
        """
        result = copy.deepcopy(config)
        current_version = self.detect_version(result)

        if not target_version:
            target_version = self._get_latest_version()

        if current_version == target_version:
            return result

        # 构建迁移路径
        path = self._find_migration_path(current_version, target_version)

        if path is None:
            raise MigrationError(
                f"无法找到从 {current_version} 到 {target_version} 的迁移路径",
                current_version,
                target_version,
            )

        # 按顺序执行迁移
        for rule in path:
            result = rule.apply(result)

        return result

    def detect_version(self, config: Dict[str, Any]) -> str:
        """检测配置的当前版本。

        优先从配置中读取版本号，如果没有则尝试推断。

        Args:
            config: 配置字典

        Returns:
            版本号字符串
        """
        if not isinstance(config, dict):
            return "0.0.0"

        # 直接读取版本字段
        if self.version_key in config:
            version = config[self.version_key]
            if isinstance(version, str):
                return version

        # 尝试从 config_version 字段读取
        if "config_version" in config:
            version = config["config_version"]
            if isinstance(version, str):
                return version

        # 根据配置内容推断版本
        return self._infer_version(config)

    def get_registered_rules(self) -> List[MigrationRule]:
        """获取所有已注册的迁移规则。

        Returns:
            迁移规则列表
        """
        return list(self._rules.values())

    def get_available_versions(self) -> List[str]:
        """获取所有涉及的版本号。

        Returns:
            版本号列表（排序后）
        """
        versions: set = set()
        for rule in self._rules.values():
            versions.add(rule.version_from)
            versions.add(rule.version_to)
        return sorted(versions, key=self._version_sort_key)

    def _find_migration_path(
        self,
        from_version: str,
        to_version: str,
    ) -> Optional[List[MigrationRule]]:
        """查找从源版本到目标版本的迁移路径。

        使用广度优先搜索（BFS）查找最短路径。

        Args:
            from_version: 源版本
            to_version: 目标版本

        Returns:
            迁移规则列表，找不到返回 None
        """
        if from_version == to_version:
            return []

        # BFS
        visited = {from_version}
        queue: List[Tuple[str, List[MigrationRule]]] = [
            (from_version, [])
        ]

        while queue:
            current, path = queue.pop(0)

            # 查找所有从当前版本出发的规则
            for rule_key, rule in self._rules.items():
                if rule.version_from == current and rule.version_to not in visited:
                    new_path = path + [rule]

                    if rule.version_to == to_version:
                        return new_path

                    visited.add(rule.version_to)
                    queue.append((rule.version_to, new_path))

        return None

    def _get_latest_version(self) -> str:
        """获取最新版本号。"""
        versions = self.get_available_versions()
        if not versions:
            return "1.0.0"
        return versions[-1]

    def _infer_version(self, config: Dict[str, Any]) -> str:
        """根据配置内容推断版本号。

        通过检查特征字段来推断配置版本。

        Args:
            config: 配置字典

        Returns:
            推断的版本号
        """
        if not isinstance(config, dict):
            return "0.0.0"

        # 检查各版本的特征字段
        keys = set(config.keys())

        # 如果有 "config_version" 字段但值不是字符串
        if "config_version" in keys:
            return "0.1.0"

        # 如果有新版字段
        new_style_markers = {"settings", "profiles", "environments"}
        if keys & new_style_markers:
            return "2.0.0"

        # 默认为最旧版本
        return "1.0.0"

    @staticmethod
    def _version_sort_key(version: str) -> Tuple[int, ...]:
        """版本号排序键。

        将版本号字符串转换为可比较的元组。

        Args:
            version: 版本号字符串（如 "1.2.3"）

        Returns:
            版本号元组（如 (1, 2, 3)）
        """
        parts = version.split(".")
        result = []
        for part in parts:
            # 提取数字部分
            match = re.match(r"(\d+)", part)
            if match:
                result.append(int(match.group(1)))
            else:
                result.append(0)
        # 补齐到至少3位
        while len(result) < 3:
            result.append(0)
        return tuple(result)

    def __repr__(self) -> str:
        return (
            f"ConfigMigrator("
            f"version_key={self.version_key!r}, "
            f"registered_rules={len(self._rules)}, "
            f"available_versions={self.get_available_versions()}"
            f")"
        )
