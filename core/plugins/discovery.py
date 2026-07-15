"""
插件发现模块

提供插件目录扫描、包扫描和插件信息提取功能。
仅使用 Python 标准库。
"""

import importlib
import importlib.util
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .spec import PluginSpec, parse_manifest


# ---------------------------------------------------------------------------
# PluginInfo - 插件信息
# ---------------------------------------------------------------------------
@dataclass
class PluginInfo:
    """插件信息

    Attributes:
        path: 插件路径（目录或文件）
        name: 插件名称
        version: 插件版本
        entry_point: 入口点
        is_installed: 是否已安装（可导入）
        manifest_path: 清单文件路径
        spec: 插件规范（如果已解析）
        error: 解析错误信息
    """
    path: str = ""
    name: str = ""
    version: str = ""
    entry_point: str = ""
    is_installed: bool = False
    manifest_path: str = ""
    spec: Optional[PluginSpec] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "version": self.version,
            "entry_point": self.entry_point,
            "is_installed": self.is_installed,
            "manifest_path": self.manifest_path,
            "spec": self.spec.to_dict() if self.spec else None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# PluginDiscovery - 插件发现机制
# ---------------------------------------------------------------------------
class PluginDiscovery:
    """插件发现机制

    支持从文件系统目录和 Python 包中发现插件。
    插件可通过以下方式被识别:
    1. 目录中包含 plugin.json 或 plugin.yaml 清单文件
    2. Python 包中包含 _plugin_ 模块或 plugin.py
    3. 目录中包含 __init__.py 且有特定标记
    """

    # 清单文件名（按优先级排序）
    MANIFEST_FILENAMES = ["plugin.json", "plugin.yaml", "plugin.yml"]

    # 插件目录标记文件
    MARKER_FILENAMES = ["_plugin_.py", "plugin.py"]

    def __init__(self):
        self._discovered: Dict[str, PluginInfo] = {}
        self._search_paths: List[str] = []

    @property
    def search_paths(self) -> List[str]:
        return list(self._search_paths)

    def add_search_path(self, path: str) -> None:
        """添加搜索路径"""
        abs_path = os.path.abspath(path)
        if abs_path not in self._search_paths:
            self._search_paths.append(abs_path)

    def remove_search_path(self, path: str) -> bool:
        """移除搜索路径"""
        abs_path = os.path.abspath(path)
        if abs_path in self._search_paths:
            self._search_paths.remove(abs_path)
            return True
        return False

    # ----- 目录扫描 -----

    def scan_directory(self, path: str) -> List[PluginInfo]:
        """扫描目录查找插件

        遍历指定目录的子目录，查找包含清单文件或标记文件的插件。

        Args:
            path: 要扫描的目录路径

        Returns:
            发现的插件信息列表
        """
        if not os.path.isdir(path):
            return []

        plugins: List[PluginInfo] = []

        try:
            entries = os.listdir(path)
        except PermissionError:
            return []

        for entry in sorted(entries):
            entry_path = os.path.join(path, entry)

            if not os.path.isdir(entry_path):
                continue

            # 跳过隐藏目录和特殊目录
            if entry.startswith(".") or entry.startswith("_"):
                continue
            if entry in ("__pycache__", "node_modules", "venv", ".git"):
                continue

            info = self.get_plugin_info(entry_path)
            if info is not None:
                plugins.append(info)
                self._discovered[info.name] = info

        return plugins

    def scan_all(self) -> List[PluginInfo]:
        """扫描所有搜索路径"""
        all_plugins: List[PluginInfo] = []
        for search_path in self._search_paths:
            plugins = self.scan_directory(search_path)
            all_plugins.extend(plugins)
        return all_plugins

    # ----- 包扫描 -----

    def scan_package(self, package_name: str) -> List[PluginInfo]:
        """扫描 Python 包查找插件

        在包的子模块中查找插件标记。

        Args:
            package_name: Python 包名

        Returns:
            发现的插件信息列表
        """
        plugins: List[PluginInfo] = []

        try:
            package = importlib.import_module(package_name)
            package_path = getattr(package, "__path__", None)
            if package_path is None:
                return []
            package_dir = package_path[0] if isinstance(package_path, list) else str(package_path)
        except (ImportError, ModuleNotFoundError):
            return []

        # 查找包目录中的清单文件
        for filename in self.MANIFEST_FILENAMES:
            manifest_path = os.path.join(package_dir, filename)
            if os.path.exists(manifest_path):
                info = self._load_from_manifest(manifest_path)
                if info is not None:
                    plugins.append(info)
                    self._discovered[info.name] = info
                return plugins

        # 查找包目录中的标记文件
        for filename in self.MARKER_FILENAMES:
            marker_path = os.path.join(package_dir, filename)
            if os.path.exists(marker_path):
                info = self._load_from_marker(marker_path, package_dir)
                if info is not None:
                    plugins.append(info)
                    self._discovered[info.name] = info
                return plugins

        # 扫描子包
        try:
            entries = os.listdir(package_dir)
        except PermissionError:
            return []

        for entry in sorted(entries):
            entry_path = os.path.join(package_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry.startswith(".") or entry.startswith("_"):
                continue

            init_path = os.path.join(entry_path, "__init__.py")
            if not os.path.exists(init_path):
                continue

            sub_package = f"{package_name}.{entry}"
            sub_plugins = self.scan_package(sub_package)
            plugins.extend(sub_plugins)

        return plugins

    # ----- 获取插件信息 -----

    def get_plugin_info(self, path: str) -> Optional[PluginInfo]:
        """获取插件信息

        Args:
            path: 插件目录路径

        Returns:
            PluginInfo 或 None（如果不是有效插件）
        """
        if not os.path.isdir(path):
            return None

        # 1. 查找清单文件
        for filename in self.MANIFEST_FILENAMES:
            manifest_path = os.path.join(path, filename)
            if os.path.exists(manifest_path):
                return self._load_from_manifest(manifest_path)

        # 2. 查找标记文件
        for filename in self.MARKER_FILENAMES:
            marker_path = os.path.join(path, filename)
            if os.path.exists(marker_path):
                return self._load_from_marker(marker_path, path)

        # 3. 检查是否有 __init__.py（Python 包插件）
        init_path = os.path.join(path, "__init__.py")
        if os.path.exists(init_path):
            return self._load_from_init(init_path, path)

        return None

    def get_discovered(self, name: Optional[str] = None) -> List[PluginInfo]:
        """获取已发现的插件"""
        if name is not None:
            info = self._discovered.get(name)
            return [info] if info else []
        return list(self._discovered.values())

    def clear_discovered(self) -> None:
        """清空已发现的插件"""
        self._discovered.clear()

    # ----- 内部方法 -----

    def _load_from_manifest(self, manifest_path: str) -> Optional[PluginInfo]:
        """从清单文件加载插件信息"""
        try:
            from .spec import parse_manifest
            manifest = parse_manifest(manifest_path)
            spec = manifest.spec
            plugin_dir = os.path.dirname(manifest_path)

            # 检查是否可导入
            is_installed = self._check_installation(spec, plugin_dir)

            return PluginInfo(
                path=plugin_dir,
                name=spec.name,
                version=spec.version,
                entry_point=spec.entry_point,
                is_installed=is_installed,
                manifest_path=manifest_path,
                spec=spec,
            )
        except Exception as exc:
            return PluginInfo(
                path=os.path.dirname(manifest_path),
                name=os.path.basename(os.path.dirname(manifest_path)),
                error=f"清单解析失败: {exc}",
            )

    def _load_from_marker(
        self, marker_path: str, plugin_dir: str
    ) -> Optional[PluginInfo]:
        """从标记文件加载插件信息"""
        try:
            name = os.path.basename(plugin_dir)
            spec = self._extract_spec_from_module(marker_path, name)
            if spec is None:
                spec = PluginSpec(name=name)

            is_installed = self._check_installation(spec, plugin_dir)

            return PluginInfo(
                path=plugin_dir,
                name=spec.name,
                version=spec.version,
                entry_point=spec.entry_point,
                is_installed=is_installed,
                spec=spec,
            )
        except Exception as exc:
            return PluginInfo(
                path=plugin_dir,
                name=os.path.basename(plugin_dir),
                error=f"标记文件解析失败: {exc}",
            )

    def _load_from_init(
        self, init_path: str, plugin_dir: str
    ) -> Optional[PluginInfo]:
        """从 __init__.py 加载插件信息"""
        try:
            name = os.path.basename(plugin_dir)
            spec = self._extract_spec_from_module(init_path, name)
            if spec is None:
                return None

            is_installed = self._check_installation(spec, plugin_dir)

            return PluginInfo(
                path=plugin_dir,
                name=spec.name,
                version=spec.version,
                entry_point=spec.entry_point,
                is_installed=is_installed,
                spec=spec,
            )
        except Exception as exc:
            return PluginInfo(
                path=plugin_dir,
                name=os.path.basename(plugin_dir),
                error=f"模块解析失败: {exc}",
            )

    def _extract_spec_from_module(
        self, module_path: str, default_name: str
    ) -> Optional[PluginSpec]:
        """从 Python 模块文件中提取插件规范

        查找模块中的 PLUGIN_SPEC 变量或 get_plugin_spec() 函数。
        """
        spec = self._load_module_attribute(module_path, "PLUGIN_SPEC")
        if spec is not None and isinstance(spec, PluginSpec):
            return spec

        # 尝试调用函数
        get_spec = self._load_module_attribute(module_path, "get_plugin_spec")
        if get_spec is not None and callable(get_spec):
            result = get_spec()
            if isinstance(result, PluginSpec):
                return result

        # 尝试从字典创建
        spec_dict = self._load_module_attribute(module_path, "PLUGIN_META")
        if spec_dict is not None and isinstance(spec_dict, dict):
            return PluginSpec.from_dict(spec_dict)

        return None

    @staticmethod
    def _load_module_attribute(module_path: str, attr_name: str) -> Any:
        """从模块文件加载指定属性"""
        spec = importlib.util.spec_from_file_location("_plugin_inspect", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return getattr(module, attr_name, None)
        except Exception:
            return None

    @staticmethod
    def _check_installation(spec: PluginSpec, plugin_dir: str) -> bool:
        """检查插件是否已安装（可导入）"""
        if not spec.entry_point:
            return os.path.exists(os.path.join(plugin_dir, "__init__.py"))

        module_path = spec.entry_point.split(":")[0]
        try:
            importlib.import_module(module_path)
            return True
        except (ImportError, ModuleNotFoundError):
            return False
