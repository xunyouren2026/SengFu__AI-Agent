#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin 管理器模块

本模块提供插件生命周期管理，包括插件加载、依赖解析、
版本管理和冲突检测等功能。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import importlib
import importlib.util
import subprocess
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Callable
from collections import defaultdict
import logging

from .sdk import (
    BasePlugin, PluginManifest, PluginMetadata, PluginDependency,
    PluginContext, PluginConfig, PluginState, PluginPriority,
    PluginError, PluginLoadError, PluginConfigError,
)

# 配置日志
logger = logging.getLogger(__name__)


class PluginNotFoundError(PluginError):
    """插件未找到错误"""
    pass


class PluginConflictError(PluginError):
    """插件冲突错误"""
    pass


class PluginDependencyError(PluginError):
    """插件依赖错误"""
    pass


class PluginVersionError(PluginError):
    """插件版本错误"""
    pass


@dataclass
class InstalledPlugin:
    """
    已安装插件信息
    
    属性:
        manifest: 插件清单
        install_path: 安装路径
        state: 当前状态
        loaded_at: 加载时间
        enabled_at: 启用时间
        error_message: 错误信息
    """
    manifest: PluginManifest
    install_path: Path
    state: PluginState = PluginState.UNLOADED
    loaded_at: Optional[Any] = None
    enabled_at: Optional[Any] = None
    error_message: Optional[str] = None
    
    @property
    def plugin_id(self) -> str:
        """获取插件 ID"""
        return self.manifest.metadata.id
    
    @property
    def version(self) -> str:
        """获取插件版本"""
        return self.manifest.metadata.version
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'manifest': self.manifest.to_dict(),
            'install_path': str(self.install_path),
            'state': self.state.name,
            'loaded_at': self.loaded_at,
            'enabled_at': self.enabled_at,
            'error_message': self.error_message,
        }


class VersionConstraint:
    """版本约束解析器"""
    
    def __init__(self, constraint: str):
        """
        初始化版本约束
        
        参数:
            constraint: 版本约束字符串
        """
        self.constraint = constraint
        self._parse()
    
    def _parse(self) -> None:
        """解析版本约束"""
        constraint = self.constraint.strip()
        
        # 通配符
        if constraint == "*":
            self.operator = "*"
            self.version = None
            return
        
        # 操作符约束
        import re
        match = re.match(r'^(>=|>|<=|<|=|~|\^)?\s*(.+)$', constraint)
        if match:
            self.operator = match.group(1) or "="
            self.version = match.group(2)
        else:
            self.operator = "="
            self.version = constraint
    
    def matches(self, version: str) -> bool:
        """
        检查版本是否满足约束
        
        参数:
            version: 要检查的版本
            
        返回:
            是否满足约束
        """
        if self.operator == "*":
            return True
        
        if self.operator == "=":
            return version == self.version
        
        # 简化版本比较
        v_parts = [int(p) for p in version.split('.') if p.isdigit()]
        c_parts = [int(p) for p in self.version.split('.') if p.isdigit()]
        
        # 补齐长度
        while len(v_parts) < len(c_parts):
            v_parts.append(0)
        while len(c_parts) < len(v_parts):
            c_parts.append(0)
        
        # 比较
        comparison = 0
        for v, c in zip(v_parts, c_parts):
            if v < c:
                comparison = -1
                break
            elif v > c:
                comparison = 1
                break
        
        if self.operator == ">=":
            return comparison >= 0
        elif self.operator == ">":
            return comparison > 0
        elif self.operator == "<=":
            return comparison <= 0
        elif self.operator == "<":
            return comparison < 0
        elif self.operator == "^":
            # 兼容版本
            if v_parts[0] != c_parts[0]:
                return False
            return comparison >= 0
        elif self.operator == "~":
            # 近似版本
            if v_parts[0] != c_parts[0] or v_parts[1] != c_parts[1]:
                return False
            return comparison >= 0
        
        return False


class PluginLoader:
    """插件加载器"""
    
    def __init__(self, plugin_dirs: List[Path]):
        """
        初始化加载器
        
        参数:
            plugin_dirs: 插件目录列表
        """
        self.plugin_dirs = plugin_dirs
    
    def discover(self) -> List[Path]:
        """
        发现可用插件
        
        返回:
            插件清单文件路径列表
        """
        manifests = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # 查找 manifest.json
            for manifest_path in plugin_dir.rglob("manifest.json"):
                manifests.append(manifest_path)
        
        return manifests
    
    def load_manifest(self, manifest_path: Path) -> PluginManifest:
        """
        加载插件清单
        
        参数:
            manifest_path: 清单文件路径
            
        返回:
            插件清单
        """
        return PluginManifest.from_file(manifest_path)
    
    def load_plugin_class(self, manifest: PluginManifest, 
                          install_path: Path) -> Type[BasePlugin]:
        """
        加载插件类
        
        参数:
            manifest: 插件清单
            install_path: 安装路径
            
        返回:
            插件类
        """
        entry_point = manifest.entry_point
        module_path = install_path / entry_point
        
        if not module_path.exists():
            raise PluginLoadError(f"入口文件不存在: {module_path}")
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(
            f"plugin_{manifest.metadata.id}", 
            module_path
        )
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"无法加载模块: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        
        # 查找插件类
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BasePlugin) and 
                attr is not BasePlugin):
                plugin_class = attr
                break
        
        if plugin_class is None:
            raise PluginLoadError(f"在 {module_path} 中未找到插件类")
        
        return plugin_class


class DependencyResolver:
    """依赖解析器"""
    
    def __init__(self, installed_plugins: Dict[str, InstalledPlugin]):
        """
        初始化解析器
        
        参数:
            installed_plugins: 已安装插件字典
        """
        self.installed_plugins = installed_plugins
    
    def resolve(self, plugin_id: str) -> Tuple[List[str], List[str]]:
        """
        解析插件依赖
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            (加载顺序列表, 错误信息列表)
        """
        resolved = []
        errors = []
        visiting = set()
        visited = set()
        
        def visit(pid: str) -> None:
            if pid in visited:
                return
            if pid in visiting:
                errors.append(f"检测到循环依赖: {pid}")
                return
            
            visiting.add(pid)
            
            plugin = self.installed_plugins.get(pid)
            if plugin is None:
                errors.append(f"依赖的插件未找到: {pid}")
                visiting.remove(pid)
                return
            
            # 检查依赖
            for dep in plugin.manifest.dependencies:
                dep_plugin = self.installed_plugins.get(dep.plugin_id)
                
                if dep_plugin is None:
                    if not dep.optional:
                        errors.append(f"必需依赖未找到: {dep.plugin_id}")
                    continue
                
                # 检查版本兼容性
                constraint = VersionConstraint(dep.version_constraint)
                if not constraint.matches(dep_plugin.version):
                    errors.append(
                        f"依赖版本不兼容: {dep.plugin_id} "
                        f"需要 {dep.version_constraint}, "
                        f"实际为 {dep_plugin.version}"
                    )
                    continue
                
                visit(dep.plugin_id)
            
            visiting.remove(pid)
            visited.add(pid)
            resolved.append(pid)
        
        visit(plugin_id)
        
        return resolved, errors
    
    def check_conflicts(self) -> List[str]:
        """
        检查插件冲突
        
        返回:
            冲突信息列表
        """
        conflicts = []
        
        # 检查 ID 冲突
        ids = defaultdict(list)
        for plugin_id, plugin in self.installed_plugins.items():
            ids[plugin_id].append(plugin)
        
        for pid, plugins in ids.items():
            if len(plugins) > 1:
                versions = [p.version for p in plugins]
                conflicts.append(f"插件 ID 冲突: {pid} ({', '.join(versions)})")
        
        return conflicts


class PluginManager:
    """
    插件管理器
    
    管理插件的生命周期，包括加载、启用、禁用和卸载。
    """
    
    def __init__(self, plugin_dirs: List[Path], 
                 data_dir: Optional[Path] = None,
                 temp_dir: Optional[Path] = None):
        """
        初始化管理器
        
        参数:
            plugin_dirs: 插件目录列表
            data_dir: 数据目录
            temp_dir: 临时目录
        """
        self.plugin_dirs = plugin_dirs
        self.data_dir = data_dir or Path.home() / ".agi_plugins" / "data"
        self.temp_dir = temp_dir or Path.home() / ".agi_plugins" / "temp"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = PluginLoader(plugin_dirs)
        self._plugins: Dict[str, InstalledPlugin] = {}
        self._instances: Dict[str, BasePlugin] = {}
        self._resolver = DependencyResolver(self._plugins)
    
    def scan(self) -> List[PluginManifest]:
        """
        扫描可用插件
        
        返回:
            插件清单列表
        """
        manifests = []
        
        for manifest_path in self.loader.discover():
            try:
                manifest = self.loader.load_manifest(manifest_path)
                manifests.append(manifest)
            except Exception as e:
                logger.error(f"加载清单失败 {manifest_path}: {e}")
        
        return manifests
    
    def install(self, source: Path, force: bool = False) -> InstalledPlugin:
        """
        安装插件
        
        参数:
            source: 插件源路径
            force: 是否强制重新安装
            
        返回:
            已安装插件信息
        """
        # 加载清单
        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            raise PluginLoadError(f"清单文件不存在: {manifest_path}")
        
        manifest = PluginManifest.from_file(manifest_path)
        plugin_id = manifest.metadata.id
        
        # 检查是否已安装
        if plugin_id in self._plugins and not force:
            raise PluginConflictError(f"插件 {plugin_id} 已安装")
        
        # 安装目录
        install_path = self.data_dir / "installed" / plugin_id
        if install_path.exists():
            shutil.rmtree(install_path)
        install_path.mkdir(parents=True)
        
        # 复制文件
        if source.is_dir():
            shutil.copytree(source, install_path, dirs_exist_ok=True)
        else:
            # 解压压缩包
            shutil.unpack_archive(source, install_path)
        
        # 创建已安装记录
        installed = InstalledPlugin(
            manifest=manifest,
            install_path=install_path,
        )
        
        self._plugins[plugin_id] = installed
        
        logger.info(f"插件 {plugin_id} 安装成功")
        return installed
    
    def uninstall(self, plugin_id: str) -> bool:
        """
        卸载插件
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            是否成功
        """
        if plugin_id not in self._plugins:
            return False
        
        # 先禁用
        if plugin_id in self._instances:
            self.disable(plugin_id)
        
        installed = self._plugins[plugin_id]
        
        # 删除安装目录
        if installed.install_path.exists():
            shutil.rmtree(installed.install_path)
        
        # 删除配置
        config_path = self.data_dir / "config" / f"{plugin_id}.json"
        if config_path.exists():
            config_path.unlink()
        
        # 移除记录
        del self._plugins[plugin_id]
        
        logger.info(f"插件 {plugin_id} 已卸载")
        return True
    
    def load(self, plugin_id: str) -> bool:
        """
        加载插件
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            是否成功
        """
        if plugin_id in self._instances:
            return True
        
        if plugin_id not in self._plugins:
            raise PluginNotFoundError(f"插件未找到: {plugin_id}")
        
        installed = self._plugins[plugin_id]
        
        # 解析依赖
        load_order, errors = self._resolver.resolve(plugin_id)
        if errors:
            installed.state = PluginState.ERROR
            installed.error_message = "; ".join(errors)
            raise PluginDependencyError(f"依赖解析失败: {errors}")
        
        # 按顺序加载依赖
        for dep_id in load_order:
            if dep_id == plugin_id:
                continue
            if dep_id not in self._instances:
                self._load_single(dep_id)
        
        # 加载目标插件
        return self._load_single(plugin_id)
    
    def _load_single(self, plugin_id: str) -> bool:
        """
        加载单个插件
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            是否成功
        """
        installed = self._plugins[plugin_id]
        
        try:
            # 加载插件类
            plugin_class = self.loader.load_plugin_class(
                installed.manifest, 
                installed.install_path
            )
            
            # 创建实例
            instance = plugin_class()
            
            # 加载配置
            config_path = self.data_dir / "config" / f"{plugin_id}.json"
            config = PluginConfig(
                schema=installed.manifest.config_schema,
                config_path=config_path,
            )
            if config_path.exists():
                config.load()
            
            # 创建上下文
            plugin_data_dir = self.data_dir / "data" / plugin_id
            plugin_data_dir.mkdir(parents=True, exist_ok=True)
            
            plugin_temp_dir = self.temp_dir / plugin_id
            plugin_temp_dir.mkdir(parents=True, exist_ok=True)
            
            context = PluginContext(
                plugin_id=plugin_id,
                config=config,
                data_dir=plugin_data_dir,
                temp_dir=plugin_temp_dir,
            )
            
            # 调用生命周期
            result = instance.on_load(context, installed.manifest)
            
            if result:
                self._instances[plugin_id] = instance
                installed.state = PluginState.LOADED
                from datetime import datetime
                installed.loaded_at = datetime.now().isoformat()
                return True
            else:
                installed.state = PluginState.ERROR
                return False
                
        except Exception as e:
            installed.state = PluginState.ERROR
            installed.error_message = str(e)
            logger.error(f"加载插件 {plugin_id} 失败: {e}")
            return False
    
    def enable(self, plugin_id: str) -> bool:
        """
        启用插件
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            是否成功
        """
        if plugin_id not in self._instances:
            if not self.load(plugin_id):
                return False
        
        instance = self._instances[plugin_id]
        installed = self._plugins[plugin_id]
        
        result = instance.on_enable()
        
        if result:
            installed.state = PluginState.ENABLED
            from datetime import datetime
            installed.enabled_at = datetime.now().isoformat()
        
        return result
    
    def disable(self, plugin_id: str) -> bool:
        """
        禁用插件
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            是否成功
        """
        if plugin_id not in self._instances:
            return False
        
        instance = self._instances[plugin_id]
        installed = self._plugins[plugin_id]
        
        result = instance.on_disable()
        
        return result
    
    def unload(self, plugin_id: str) -> None:
        """
        卸载插件（从内存）
        
        参数:
            plugin_id: 插件 ID
        """
        if plugin_id not in self._instances:
            return
        
        instance = self._instances[plugin_id]
        instance.on_unload()
        
        del self._instances[plugin_id]
        
        if plugin_id in self._plugins:
            self._plugins[plugin_id].state = PluginState.UNLOADED
    
    def get(self, plugin_id: str) -> Optional[BasePlugin]:
        """
        获取插件实例
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            插件实例或 None
        """
        return self._instances.get(plugin_id)
    
    def get_installed(self, plugin_id: str) -> Optional[InstalledPlugin]:
        """
        获取已安装插件信息
        
        参数:
            plugin_id: 插件 ID
            
        返回:
            已安装插件信息或 None
        """
        return self._plugins.get(plugin_id)
    
    def list_plugins(self) -> List[InstalledPlugin]:
        """
        列出所有已安装插件
        
        返回:
            已安装插件列表
        """
        return list(self._plugins.values())
    
    def list_enabled(self) -> List[BasePlugin]:
        """
        列出所有已启用插件
        
        返回:
            已启用插件实例列表
        """
        return [p for p in self._instances.values() if p.is_enabled]
    
    def check_updates(self) -> List[Tuple[str, str, str]]:
        """
        检查插件更新
        
        返回:
            [(plugin_id, current_version, latest_version), ...]
        """
        # TODO: 实现更新检查逻辑
        return []
    
    def shutdown(self) -> None:
        """关闭管理器，卸载所有插件"""
        for plugin_id in list(self._instances.keys()):
            self.unload(plugin_id)
        
        logger.info("插件管理器已关闭")


# 便捷函数
def create_plugin_manager(
    plugin_dirs: Optional[List[Path]] = None,
    data_dir: Optional[Path] = None
) -> PluginManager:
    """
    创建插件管理器
    
    参数:
        plugin_dirs: 插件目录列表
        data_dir: 数据目录
        
    返回:
        插件管理器实例
    """
    if plugin_dirs is None:
        plugin_dirs = [
            Path.home() / ".agi_plugins",
            Path("/usr/share/agi/plugins"),
        ]
    
    return PluginManager(plugin_dirs, data_dir)


# 单元测试存根
class TestPluginManager:
    """PluginManager 单元测试"""
    
    def test_version_constraint(self) -> None:
        """测试版本约束"""
        vc = VersionConstraint(">=1.0.0")
        assert vc.matches("1.0.0")
        assert vc.matches("2.0.0")
        assert not vc.matches("0.9.0")
        
        vc = VersionConstraint("^1.0.0")
        assert vc.matches("1.0.0")
        assert vc.matches("1.5.0")
        assert not vc.matches("2.0.0")
    
    def test_dependency_resolution(self, tmp_path) -> None:
        """测试依赖解析"""
        plugins = {}
        
        # 创建模拟插件
        for pid, deps in [
            ("plugin-a", ["plugin-b"]),
            ("plugin-b", ["plugin-c"]),
            ("plugin-c", []),
        ]:
            manifest = PluginManifest(
                metadata=PluginMetadata(id=pid, name=pid),
                dependencies=[PluginDependency(plugin_id=d) for d in deps],
            )
            plugins[pid] = InstalledPlugin(
                manifest=manifest,
                install_path=tmp_path / pid,
            )
        
        resolver = DependencyResolver(plugins)
        order, errors = resolver.resolve("plugin-a")
        
        assert len(errors) == 0
        assert order == ["plugin-c", "plugin-b", "plugin-a"]
