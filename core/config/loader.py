"""
多源配置加载器模块

支持从多种来源加载配置并按优先级合并：
默认配置 -> 文件配置 -> 环境变量 -> 命令行参数

支持的文件格式：JSON、简易YAML
"""

import copy
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


class ConfigLoader:
    """多源配置加载器。

    支持从多种来源加载配置并按优先级深度合并。

    Args:
        env_prefix: 环境变量前缀，默认为 "APP_"
        separator: 环境变量层级分隔符，默认为 "__"
    """

    def __init__(self, env_prefix: str = "APP_", separator: str = "__"):
        self.env_prefix = env_prefix
        self.separator = separator
        self._loaded_configs: List[Dict[str, Any]] = []
        self._merged_config: Dict[str, Any] = {}

    def load_from_dict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """从字典加载配置。

        Args:
            config: 配置字典

        Returns:
            加载的配置字典
        """
        if not isinstance(config, dict):
            raise TypeError(f"配置必须是字典类型，got: {type(config).__name__}")
        loaded = copy.deepcopy(config)
        self._loaded_configs.append(loaded)
        self._rebuild_merged()
        return loaded

    def load_from_json(self, file_path: str) -> Dict[str, Any]:
        """从JSON文件加载配置。

        Args:
            file_path: JSON文件路径

        Returns:
            加载的配置字典

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError(f"JSON文件顶层必须是字典/对象，got: {type(config).__name__}")

        loaded = copy.deepcopy(config)
        self._loaded_configs.append(loaded)
        self._rebuild_merged()
        return loaded

    def load_from_yaml(self, file_path: str) -> Dict[str, Any]:
        """从YAML文件加载配置（使用内置简易YAML解析器）。

        支持的YAML特性：
        - 键值对（key: value）
        - 列表（- item）
        - 嵌套结构（缩进）
        - 字符串、数字、布尔值、null
        - 多行字符串
        - 注释（# 开头）

        Args:
            file_path: YAML文件路径

        Returns:
            加载的配置字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML格式错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        config = self._parse_yaml(raw_content)

        if not isinstance(config, dict):
            raise ValueError(f"YAML文件顶层必须是映射，got: {type(config).__name__}")

        loaded = copy.deepcopy(config)
        self._loaded_configs.append(loaded)
        self._rebuild_merged()
        return loaded

    def load_from_env(
        self,
        prefix: Optional[str] = None,
        type_hints: Optional[Dict[str, type]] = None,
    ) -> Dict[str, Any]:
        """从环境变量加载配置。

        支持层级结构：使用分隔符（默认 "__"）表示嵌套。
        例如：APP_DATABASE__HOST -> {"database": {"host": "..."}}

        Args:
            prefix: 环境变量前缀，默认使用实例的 env_prefix
            type_hints: 类型提示字典，key为环境变量名（不含前缀），
                       value为目标类型（int/float/bool/str/list）

        Returns:
            从环境变量加载的配置字典
        """
        if prefix is None:
            prefix = self.env_prefix

        config: Dict[str, Any] = {}
        type_hints = type_hints or {}

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # 去掉前缀
            key_part = env_key[len(prefix):]
            if not key_part:
                continue

            # 按分隔符拆分层级
            parts = key_part.split(self.separator)

            # 自动类型转换
            converted_value = self._convert_env_value(
                env_value, type_hints, key_part
            )

            # 构建嵌套字典
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = converted_value

        self._loaded_configs.append(config)
        self._rebuild_merged()
        return config

    def load_from_argv(
        self,
        argv: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """从命令行参数加载配置。

        支持格式：
        --key value
        --key=value
        --nested.key value

        Args:
            argv: 命令行参数列表，默认使用 sys.argv[1:]

        Returns:
            从命令行参数加载的配置字典
        """
        if argv is None:
            argv = sys.argv[1:]

        config: Dict[str, Any] = {}
        i = 0

        while i < len(argv):
            arg = argv[i]
            if arg.startswith("--"):
                # 去掉 -- 前缀
                key_part = arg[2:]

                # 检查是否是 --key=value 格式
                if "=" in key_part:
                    key_part, value_str = key_part.split("=", 1)
                    value = self._convert_env_value(value_str)
                elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    i += 1
                    value = self._convert_env_value(argv[i])
                else:
                    # 布尔标志
                    value = True

                # 按点号拆分层级
                parts = key_part.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value

            i += 1

        self._loaded_configs.append(config)
        self._rebuild_merged()
        return config

    def merge_config(
        self,
        *configs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """深度合并多个配置字典。

        后面的配置覆盖前面的配置。对于嵌套字典，执行递归合并。

        Args:
            *configs: 待合并的配置字典列表

        Returns:
            合并后的配置字典
        """
        result: Dict[str, Any] = {}
        for config in configs:
            result = self._deep_merge(result, config)
        return result

    def get_merged(self) -> Dict[str, Any]:
        """获取当前所有已加载配置的合并结果。"""
        return copy.deepcopy(self._merged_config)

    def clear(self) -> None:
        """清除所有已加载的配置。"""
        self._loaded_configs.clear()
        self._merged_config.clear()

    def _rebuild_merged(self) -> None:
        """根据加载顺序重建合并配置。"""
        result: Dict[str, Any] = {}
        for config in self._loaded_configs:
            result = self._deep_merge(result, config)
        self._merged_config = result

    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """深度合并两个字典。

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            合并后的新字典
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def _convert_env_value(
        value: str,
        type_hints: Optional[Dict[str, type]] = None,
        key: str = "",
    ) -> Any:
        """将环境变量字符串值转换为适当的Python类型。

        优先使用类型提示，否则自动推断。

        Args:
            value: 环境变量字符串值
            type_hints: 类型提示字典
            key: 配置键名（用于查找类型提示）

        Returns:
            转换后的值
        """
        type_hints = type_hints or {}

        # 如果有类型提示，使用提示的类型
        if key and key in type_hints:
            target_type = type_hints[key]
            return ConfigLoader._cast_value(value, target_type)

        # 自动推断类型
        # 布尔值
        lower_val = value.lower()
        if lower_val in ("true", "yes", "on", "1"):
            return True
        if lower_val in ("false", "no", "off", "0"):
            return False

        # null
        if lower_val in ("null", "none", "~", ""):
            return None

        # 数字
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        # 列表（逗号分隔）
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            items = [item.strip() for item in inner.split(",")]
            return items

        # 默认为字符串
        return value

    @staticmethod
    def _cast_value(value: str, target_type: type) -> Any:
        """将字符串值转换为目标类型。"""
        if target_type is bool:
            return value.lower() in ("true", "yes", "on", "1")
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        if target_type is str:
            return value
        if target_type is list:
            return [item.strip() for item in value.split(",")]
        if target_type is dict:
            # 简易 key=value,key2=value2 格式
            result = {}
            for pair in value.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    result[k.strip()] = v.strip()
            return result
        return value

    # ==================== 简易YAML解析器 ====================

    def _parse_yaml(self, content: str) -> Any:
        """解析YAML内容。

        Args:
            content: YAML文本内容

        Returns:
            解析后的Python对象
        """
        lines = content.split("\n")
        # 预处理：去掉空行和注释
        processed_lines = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue
            processed_lines.append(line)

        if not processed_lines:
            return {}

        result, _ = self._parse_yaml_block(processed_lines, 0, 0)
        return result

    def _parse_yaml_block(
        self,
        lines: List[str],
        start_index: int,
        base_indent: int,
    ) -> Tuple[Any, int]:
        """解析YAML块。

        Args:
            lines: 所有行
            start_index: 起始行索引
            base_indent: 基础缩进级别

        Returns:
            (解析结果, 下一行索引) 元组
        """
        if start_index >= len(lines):
            return None, start_index

        first_line = lines[start_index]
        stripped = first_line.lstrip()
        first_indent = len(first_line) - len(stripped)

        if first_indent < base_indent:
            return None, start_index

        # 判断是否是列表项
        if stripped.startswith("- "):
            return self._parse_yaml_list(lines, start_index, first_indent)

        # 判断是否是映射
        if self._is_mapping_line(stripped):
            return self._parse_yaml_mapping(lines, start_index, first_indent)

        # 单个值
        value = self._parse_yaml_value(stripped)
        return value, start_index + 1

    def _is_mapping_line(self, line: str) -> bool:
        """判断一行是否是映射（key: value）。"""
        # 排除列表项
        if line.startswith("- "):
            return False
        # 查找冒号，但排除在引号内的冒号
        in_single_quote = False
        in_double_quote = False
        for i, ch in enumerate(line):
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == ":" and not in_single_quote and not in_double_quote:
                # 冒号后面必须是空格、行尾或另一个冒号
                rest = line[i + 1:]
                if not rest or rest[0] in (" ", "\t", "\n"):
                    return True
                return False
        return False

    def _parse_yaml_mapping(
        self,
        lines: List[str],
        start_index: int,
        base_indent: int,
    ) -> Tuple[Dict[str, Any], int]:
        """解析YAML映射块。"""
        result: Dict[str, Any] = {}
        i = start_index

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # 缩进回退，结束当前映射
            if indent < base_indent:
                break

            # 跳过空行和注释
            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            # 必须是映射行
            if not self._is_mapping_line(stripped):
                i += 1
                continue

            # 解析 key: value
            colon_pos = self._find_colon(stripped)
            key = stripped[:colon_pos].strip()
            value_part = stripped[colon_pos + 1:].strip()

            # 去掉key的引号
            key = self._strip_quotes(key)

            if not value_part or value_part.startswith("#"):
                # 值在下一行（可能是嵌套映射或列表）
                i += 1
                if i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.lstrip()
                    next_indent = len(next_line) - len(next_stripped)

                    if next_indent > base_indent:
                        if next_stripped.startswith("- "):
                            value, i = self._parse_yaml_list(
                                lines, i, next_indent
                            )
                        elif self._is_mapping_line(next_stripped):
                            value, i = self._parse_yaml_mapping(
                                lines, i, next_indent
                            )
                        else:
                            value, i = self._parse_yaml_block(
                                lines, i, next_indent
                            )
                        result[key] = value
                        continue
                result[key] = None
            else:
                # 去掉行内注释
                value_part = self._strip_inline_comment(value_part)
                value = self._parse_yaml_value(value_part)
                result[key] = value

            i += 1

        return result, i

    def _parse_yaml_list(
        self,
        lines: List[str],
        start_index: int,
        base_indent: int,
    ) -> Tuple[List[Any], int]:
        """解析YAML列表块。"""
        result: List[Any] = []
        i = start_index

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if indent < base_indent:
                break

            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            if not stripped.startswith("- "):
                break

            item_content = stripped[2:].strip()

            if not item_content or item_content.startswith("#"):
                # 列表项的值在下一行
                i += 1
                if i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.lstrip()
                    next_indent = len(next_line) - len(next_stripped)

                    if next_indent > base_indent:
                        if next_stripped.startswith("- "):
                            # 嵌套列表
                            value, i = self._parse_yaml_list(
                                lines, i, next_indent
                            )
                        elif self._is_mapping_line(next_stripped):
                            # 嵌套映射
                            value, i = self._parse_yaml_mapping(
                                lines, i, next_indent
                            )
                        else:
                            value, i = self._parse_yaml_block(
                                lines, i, next_indent
                            )
                        result.append(value)
                        continue
                result.append(None)
            else:
                # 行内值
                item_content = self._strip_inline_comment(item_content)
                value = self._parse_yaml_value(item_content)
                result.append(value)

            i += 1

        return result, i

    def _parse_yaml_value(self, value_str: str) -> Any:
        """解析单个YAML值。"""
        value_str = value_str.strip()

        if not value_str or value_str == "~":
            return None

        # 去掉引号
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        # 布尔值
        if value_str.lower() in ("true", "yes", "on"):
            return True
        if value_str.lower() in ("false", "no", "off"):
            return False

        # null
        if value_str.lower() in ("null", "none"):
            return None

        # 列表（行内格式 [a, b, c]）
        if value_str.startswith("[") and value_str.endswith("]"):
            inner = value_str[1:-1].strip()
            if not inner:
                return []
            items = []
            for item in inner.split(","):
                item = item.strip()
                if item:
                    items.append(self._parse_yaml_value(item))
            return items

        # 字典（行内格式 {a: 1, b: 2}）
        if value_str.startswith("{") and value_str.endswith("}"):
            inner = value_str[1:-1].strip()
            if not inner:
                return {}
            result = {}
            for pair in inner.split(","):
                pair = pair.strip()
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    result[k.strip()] = self._parse_yaml_value(v.strip())
            return result

        # 整数
        try:
            return int(value_str)
        except ValueError:
            pass

        # 浮点数
        try:
            return float(value_str)
        except ValueError:
            pass

        # 默认为字符串
        return value_str

    @staticmethod
    def _find_colon(line: str) -> int:
        """查找映射行中冒号的位置（排除引号内的冒号）。"""
        in_single_quote = False
        in_double_quote = False
        for i, ch in enumerate(line):
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == ":" and not in_single_quote and not in_double_quote:
                return i
        return -1

    @staticmethod
    def _strip_quotes(s: str) -> str:
        """去掉字符串两端的引号。"""
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    @staticmethod
    def _strip_inline_comment(value: str) -> str:
        """去掉行内注释。"""
        in_single_quote = False
        in_double_quote = False
        for i, ch in enumerate(value):
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == "#" and not in_single_quote and not in_double_quote:
                # 注释前面必须有空格
                if i > 0 and value[i - 1] == " ":
                    return value[:i].rstrip()
        return value

    def load(
        self,
        default: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        use_env: bool = True,
        use_argv: bool = False,
        type_hints: Optional[Dict[str, type]] = None,
    ) -> Dict[str, Any]:
        """按优先级从多个来源加载并合并配置。

        加载顺序（后者覆盖前者）：
        1. 默认配置
        2. 文件配置（自动检测JSON/YAML格式）
        3. 环境变量
        4. 命令行参数

        Args:
            default: 默认配置字典
            file_path: 配置文件路径
            use_env: 是否加载环境变量
            use_argv: 是否加载命令行参数
            type_hints: 环境变量类型提示

        Returns:
            合并后的配置字典
        """
        self.clear()

        # 1. 默认配置
        if default is not None:
            self.load_from_dict(default)

        # 2. 文件配置
        if file_path is not None:
            if file_path.endswith(".json"):
                self.load_from_json(file_path)
            elif file_path.endswith((".yaml", ".yml")):
                self.load_from_yaml(file_path)
            else:
                # 尝试JSON，失败则尝试YAML
                try:
                    self.load_from_json(file_path)
                except (json.JSONDecodeError, ValueError):
                    self.load_from_yaml(file_path)

        # 3. 环境变量
        if use_env:
            self.load_from_env(type_hints=type_hints)

        # 4. 命令行参数
        if use_argv:
            self.load_from_argv()

        return self.get_merged()

    def __repr__(self) -> str:
        return (
            f"ConfigLoader(env_prefix={self.env_prefix!r}, "
            f"sources_count={len(self._loaded_configs)})"
        )
