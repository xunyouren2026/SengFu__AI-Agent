"""
插件规范定义模块

定义插件元数据规范、插件清单解析等功能。
仅使用 Python 标准库。
"""

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 版本号工具
# ---------------------------------------------------------------------------
def _parse_version(version_str: str) -> Tuple[int, ...]:
    """将版本号字符串解析为元组，用于比较"""
    parts = re.split(r"[.\-]", version_str)
    result = []
    for part in parts:
        match = re.match(r"(\d+)", part)
        if match:
            result.append(int(match.group(1)))
        else:
            result.append(0)
    return tuple(result)


def _check_version_compatible(version: str, constraint: str) -> bool:
    """检查版本是否满足约束

    支持的约束格式:
    - ">=1.0.0"
    - ">1.0.0"
    - ">=1.0.0,<2.0.0"
    - "~=1.4.2" (兼容版本)
    - "1.0.0" (精确匹配)
    """
    constraints = [c.strip() for c in constraint.split(",")]
    ver = _parse_version(version)

    for c in constraints:
        c = c.strip()
        if not c:
            continue

        if c.startswith("~="):
            # 兼容版本: ~=1.4.2 意味着 >=1.4.2, <1.5.0
            base = c[2:].strip()
            base_parts = _parse_version(base)
            if ver < base_parts:
                return False
            # 上一级版本号 + 1
            upper = list(base_parts[:-1])
            upper[-1] = upper[-1] + 1 if upper else 1
            if ver >= tuple(upper):
                return False

        elif c.startswith(">="):
            required = _parse_version(c[2:].strip())
            if ver < required:
                return False

        elif c.startswith(">"):
            required = _parse_version(c[1:].strip())
            if ver <= required:
                return False

        elif c.startswith("<="):
            required = _parse_version(c[2:].strip())
            if ver > required:
                return False

        elif c.startswith("<"):
            required = _parse_version(c[1:].strip())
            if ver >= required:
                return False

        elif c.startswith("=="):
            required = _parse_version(c[2:].strip())
            if ver != required:
                return False

        else:
            # 精确匹配
            required = _parse_version(c)
            if ver != required:
                return False

    return True


# ---------------------------------------------------------------------------
# PluginDependency - 插件依赖
# ---------------------------------------------------------------------------
@dataclass
class PluginDependency:
    """插件依赖定义"""
    name: str
    version: str = ""  # 版本约束
    optional: bool = False
    description: str = ""

    def satisfies(self, actual_version: str) -> bool:
        """检查给定版本是否满足此依赖"""
        if not self.version:
            return True
        return _check_version_compatible(actual_version, self.version)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "optional": self.optional,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# PluginHook - 插件钩子
# ---------------------------------------------------------------------------
@dataclass
class PluginHook:
    """插件钩子定义"""
    name: str
    description: str = ""
    priority: int = 100  # 优先级，越小越先执行

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# PluginToolSpec - 插件提供的工具
# ---------------------------------------------------------------------------
@dataclass
class PluginToolSpec:
    """插件提供的工具定义"""
    name: str
    description: str = ""
    entry_point: str = ""  # "module.path:function_name"
    version: str = "1.0.0"
    parameters: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "entry_point": self.entry_point,
            "version": self.version,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# PluginSpec - 插件规范
# ---------------------------------------------------------------------------
@dataclass
class PluginSpec:
    """插件规范定义

    定义一个插件的完整元数据，包括名称、版本、作者、依赖、工具和钩子等。
    """
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    license: str = ""
    homepage: str = ""
    tags: Set[str] = field(default_factory=set)
    category: str = "general"

    # 依赖
    dependencies: List[PluginDependency] = field(default_factory=list)
    python_version: str = ">=3.8"

    # 插件提供的工具
    tools: List[PluginToolSpec] = field(default_factory=list)

    # 插件钩子
    hooks: List[PluginHook] = field(default_factory=list)

    # 权限需求
    permissions: List[str] = field(default_factory=list)

    # 入口点
    entry_point: str = ""  # "module.path:ClassName" 或 "module.path:function"

    # 其他元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """校验规范完整性

        Returns:
            (是否合法, 错误列表)
        """
        errors: List[str] = []

        if not self.name or not self.name.strip():
            errors.append("插件名称不能为空")

        if not self.version or not self.version.strip():
            errors.append("插件版本不能为空")

        # 校验名称格式（仅允许字母、数字、下划线和连字符）
        if self.name and not re.match(r"^[a-zA-Z0-9_-]+$", self.name):
            errors.append(
                f"插件名称 '{self.name}' 格式无效，仅允许字母、数字、下划线和连字符"
            )

        # 校验版本格式
        if self.version and not re.match(r"^\d+(\.\d+)*", self.version):
            errors.append(f"插件版本 '{self.version}' 格式无效")

        # 校验依赖
        dep_names: Set[str] = set()
        for dep in self.dependencies:
            if not dep.name:
                errors.append("依赖名称不能为空")
            if dep.name in dep_names:
                errors.append(f"重复的依赖: {dep.name}")
            dep_names.add(dep.name)

        # 校验工具
        tool_names: Set[str] = set()
        for tool in self.tools:
            if not tool.name:
                errors.append("工具名称不能为空")
            if tool.name in tool_names:
                errors.append(f"重复的工具名称: {tool.name}")
            tool_names.add(tool.name)

        # 校验钩子
        hook_names: Set[str] = set()
        for hook in self.hooks:
            if not hook.name:
                errors.append("钩子名称不能为空")
            if hook.name in hook_names:
                errors.append(f"重复的钩子名称: {hook.name}")
            hook_names.add(hook.name)

        # 校验入口点格式
        if self.entry_point:
            if ":" not in self.entry_point:
                errors.append(
                    f"入口点格式无效: '{self.entry_point}'，应为 'module.path:callable'"
                )

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "license": self.license,
            "homepage": self.homepage,
            "tags": sorted(self.tags),
            "category": self.category,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "python_version": self.python_version,
            "tools": [t.to_dict() for t in self.tools],
            "hooks": [h.to_dict() for h in self.hooks],
            "permissions": self.permissions,
            "entry_point": self.entry_point,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginSpec":
        """从字典创建"""
        dependencies = []
        for d in data.get("dependencies", []):
            dependencies.append(PluginDependency(
                name=d.get("name", ""),
                version=d.get("version", ""),
                optional=d.get("optional", False),
                description=d.get("description", ""),
            ))

        tools = []
        for t in data.get("tools", []):
            tools.append(PluginToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                entry_point=t.get("entry_point", ""),
                version=t.get("version", "1.0.0"),
                parameters=t.get("parameters"),
            ))

        hooks = []
        for h in data.get("hooks", []):
            hooks.append(PluginHook(
                name=h.get("name", ""),
                description=h.get("description", ""),
                priority=h.get("priority", 100),
            ))

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            description=data.get("description", ""),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            tags=set(data.get("tags", [])),
            category=data.get("category", "general"),
            dependencies=dependencies,
            python_version=data.get("python_version", ">=3.8"),
            tools=tools,
            hooks=hooks,
            permissions=data.get("permissions", []),
            entry_point=data.get("entry_point", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# PluginManifest - 插件清单
# ---------------------------------------------------------------------------
@dataclass
class PluginManifest:
    """插件清单

    从 YAML/JSON 文件解析的插件完整描述。
    """
    spec: PluginSpec
    source_path: str = ""
    source_format: str = ""  # "json" or "yaml"

    def to_dict(self) -> dict:
        result = self.spec.to_dict()
        result["_source_path"] = self.source_path
        result["_source_format"] = self.source_format
        return result


# ---------------------------------------------------------------------------
# parse_manifest - 解析插件清单
# ---------------------------------------------------------------------------
def parse_manifest(file_path: str) -> PluginManifest:
    """解析插件清单文件

    支持 JSON 和 YAML 格式（YAML 使用标准库的 json 模块模拟解析，
    仅支持简单的 YAML 子集）。

    Args:
        file_path: 清单文件路径

    Returns:
        PluginManifest 实例

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 解析失败
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"插件清单文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".json":
        data = json.loads(content)
        spec = PluginSpec.from_dict(data)
        return PluginManifest(
            spec=spec,
            source_path=file_path,
            source_format="json",
        )

    elif ext in (".yml", ".yaml"):
        data = _parse_simple_yaml(content)
        spec = PluginSpec.from_dict(data)
        return PluginManifest(
            spec=spec,
            source_path=file_path,
            source_format="yaml",
        )

    else:
        # 尝试 JSON
        try:
            data = json.loads(content)
            spec = PluginSpec.from_dict(data)
            return PluginManifest(
                spec=spec,
                source_path=file_path,
                source_format="json",
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"无法解析插件清单文件 '{file_path}'，不支持的格式: {ext}"
            )


# ---------------------------------------------------------------------------
# 简易 YAML 解析器（仅支持标准库可实现的子集）
# ---------------------------------------------------------------------------
def _parse_simple_yaml(content: str) -> dict:
    """简易 YAML 解析器

    支持:
    - 键值对（缩进表示层级）
    - 列表（- 开头）
    - 字符串、数字、布尔值
    - 注释（# 开头）

    不支持:
    - 多行字符串
    - 复杂引用
    - 锚点和别名
    """
    lines = content.split("\n")
    result: Dict[str, Any] = {}
    stack: List[Tuple[int, dict]] = [(0, result)]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # 跳过空行和注释
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(stripped)

        # 列表项
        if stripped.startswith("- "):
            list_value = stripped[2:].strip()
            # 找到当前层级
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()

            parent = stack[-1][1]
            # 确定列表所属的键
            # 回溯找到最近的键
            if isinstance(parent, dict):
                # 找到最后一个值是列表的键
                last_key = None
                for k, v in reversed(list(parent.items())):
                    if isinstance(v, list):
                        last_key = k
                        break
                if last_key is not None:
                    parsed_val = _parse_yaml_value(list_value)
                    parent[last_key].append(parsed_val)
            i += 1
            continue

        # 键值对
        if ":" in stripped:
            colon_pos = stripped.index(":")
            key = stripped[:colon_pos].strip()
            value_str = stripped[colon_pos + 1:].strip()

            # 去除注释
            if " #" in key:
                key = key[:key.index(" #")].strip()

            # 回溯到正确的父级
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()

            parent = stack[-1][1]

            if value_str:
                # 同行值
                if value_str.startswith("#"):
                    value: Any = []
                else:
                    # 去除行尾注释
                    if " #" in value_str:
                        value_str = value_str[:value_str.index(" #")].strip()
                    value = _parse_yaml_value(value_str)
                parent[key] = value
            else:
                # 值在下一行
                # 检查下一行是否是列表
                next_indent = indent + 2
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].lstrip()
                    next_indent_actual = len(lines[i + 1]) - len(next_stripped)
                    if next_stripped.startswith("- ") and next_indent_actual > indent:
                        parent[key] = []
                        stack.append((next_indent_actual, parent[key]))
                        i += 1
                        continue
                    elif next_indent_actual > indent:
                        parent[key] = {}
                        stack.append((next_indent_actual, parent[key]))
                        i += 1
                        continue
                parent[key] = ""

        i += 1

    return result


def _parse_yaml_value(value: str) -> Any:
    """解析 YAML 值"""
    value = value.strip()

    # 布尔值
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # None
    if value.lower() in ("null", "~", ""):
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

    # 字符串（去除引号）
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    return value
