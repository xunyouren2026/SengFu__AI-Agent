"""
Plugin System
插件系统

提供插件生命周期管理、沙箱执行、API暴露等功能
"""

import asyncio
import hashlib
import importlib
import json
import logging
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """插件状态"""
    UNINSTALLED = "uninstalled"
    INSTALLED = "installed"
    ENABLED = "enabled"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


class PluginType(Enum):
    """插件类型"""
    TOOL = "tool"
    AGENT = "agent"
    MODEL = "model"
    INTEGRATION = "integration"
    UI = "ui"
    WORKFLOW = "workflow"


@dataclass
class PluginManifest:
    """插件清单"""
    id: str = ""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    plugin_type: PluginType = PluginType.TOOL
    main: str = "main.py"
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[Dict[str, Any]] = field(default_factory=list)
    hooks: Dict[str, str] = field(default_factory=dict)
    min_version: str = "1.0.0"
    max_version: str = ""
    icon: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "main": self.main,
            "dependencies": self.dependencies,
            "permissions": self.permissions,
            "config_schema": self.config_schema,
            "api_endpoints": self.api_endpoints,
            "hooks": self.hooks,
            "min_version": self.min_version,
            "max_version": self.max_version,
            "icon": self.icon,
            "homepage": self.homepage,
            "repository": self.repository,
            "license": self.license,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            plugin_type=PluginType(data.get("plugin_type", "tool")),
            main=data.get("main", "main.py"),
            dependencies=data.get("dependencies", []),
            permissions=data.get("permissions", []),
            config_schema=data.get("config_schema", {}),
            api_endpoints=data.get("api_endpoints", []),
            hooks=data.get("hooks", {}),
            min_version=data.get("min_version", "1.0.0"),
            max_version=data.get("max_version", ""),
            icon=data.get("icon", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            license=data.get("license", "MIT"),
        )


@dataclass
class PluginInstance:
    """插件实例"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    manifest: PluginManifest = None
    status: PluginStatus = PluginStatus.UNINSTALLED
    config: Dict[str, Any] = field(default_factory=dict)
    path: str = ""
    enabled: bool = False
    loaded_at: Optional[float] = None
    started_at: Optional[float] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "status": self.status.value,
            "config": self.config,
            "path": self.path,
            "enabled": self.enabled,
            "loaded_at": self.loaded_at,
            "started_at": self.started_at,
            "error": self.error,
            "metrics": self.metrics,
        }


class PluginBase(ABC):
    """插件基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._manifest: Optional[PluginManifest] = None
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化插件"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """关闭插件"""
        pass
    
    def get_manifest(self) -> Optional[PluginManifest]:
        """获取插件清单"""
        return self._manifest
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取插件提供的工具"""
        return []
    
    def get_api_routes(self) -> List[Dict[str, Any]]:
        """获取插件提供的API路由"""
        return []


class PluginSandbox:
    """
    插件沙箱
    
    提供安全的插件执行环境
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._allowed_modules = set(self.config.get("allowed_modules", [
            "json", "os", "sys", "time", "datetime", "asyncio",
            "typing", "dataclasses", "enum", "pathlib",
            "httpx", "aiohttp",  # 网络请求
        ]))
        self._blocked_modules = set(self.config.get("blocked_modules", [
            "subprocess", "socket", "ctypes", "multiprocessing",
        ]))
        self._resource_limits = self.config.get("resource_limits", {
            "max_memory_mb": 512,
            "max_cpu_percent": 50,
            "max_execution_time": 30,
        })
        
    def create_sandbox_globals(self) -> Dict[str, Any]:
        """创建沙箱全局命名空间"""
        # 受限的内置函数
        safe_builtins = {
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "isinstance": isinstance,
            "type": type,
            "None": None,
            "True": True,
            "False": False,
        }
        
        return {"__builtins__": safe_builtins}
    
    def check_permission(self, permission: str, plugin_permissions: List[str]) -> bool:
        """检查权限"""
        if permission in plugin_permissions:
            return True
        return False
    
    async def execute_in_sandbox(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """在沙箱中执行代码"""
        sandbox_globals = self.create_sandbox_globals()
        if globals_dict:
            sandbox_globals.update(globals_dict)
        
        try:
            # 使用asyncio超时
            async def run():
                exec(code, sandbox_globals, locals_dict or {})
                return sandbox_globals.get("__result__")
            
            return await asyncio.wait_for(
                run(),
                timeout=self._resource_limits["max_execution_time"]
            )
            
        except asyncio.TimeoutError:
            raise RuntimeError("Plugin execution timed out")
        except Exception as e:
            raise RuntimeError(f"Plugin execution error: {e}")


class PluginManager:
    """
    插件管理器
    
    功能：
    - 插件安装/卸载
    - 插件启用/禁用
    - 插件生命周期管理
    - 插件沙箱执行
    - 插件API暴露
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._plugins: Dict[str, PluginInstance] = {}
        self._sandbox = PluginSandbox(config)
        self._plugin_dir = self.config.get("plugin_dir", "/tmp/plugins")
        self._hooks: Dict[str, List[Callable]] = {}
        self._initialized = False
        
    async def initialize(self):
        """初始化插件管理器"""
        if self._initialized:
            return
        
        # 确保插件目录存在
        os.makedirs(self._plugin_dir, exist_ok=True)
        
        # 扫描已安装的插件
        await self._scan_plugins()
        
        self._initialized = True
        logger.info("Plugin manager initialized")
    
    async def _scan_plugins(self):
        """扫描插件目录"""
        if not os.path.exists(self._plugin_dir):
            return
        
        for item in os.listdir(self._plugin_dir):
            plugin_path = os.path.join(self._plugin_dir, item)
            manifest_path = os.path.join(plugin_path, "manifest.json")
            
            if os.path.isdir(plugin_path) and os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as f:
                        manifest_data = json.load(f)
                    
                    manifest = PluginManifest.from_dict(manifest_data)
                    
                    instance = PluginInstance(
                        manifest=manifest,
                        status=PluginStatus.INSTALLED,
                        path=plugin_path,
                    )
                    
                    self._plugins[manifest.id] = instance
                    
                    logger.info(f"Found plugin: {manifest.name} ({manifest.id})")
                    
                except Exception as e:
                    logger.error(f"Failed to load plugin from {plugin_path}: {e}")
    
    async def install(
        self,
        source: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> PluginInstance:
        """
        安装插件
        
        Args:
            source: 插件源（路径或URL）
            config: 安装配置
        """
        await self.initialize()
        
        # 检查源类型
        if source.startswith("http"):
            # 从URL下载
            plugin_path = await self._download_plugin(source)
        else:
            # 本地路径
            plugin_path = source
        
        # 加载清单
        manifest_path = os.path.join(plugin_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Plugin manifest not found: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        
        manifest = PluginManifest.from_dict(manifest_data)
        
        # 检查依赖
        await self._check_dependencies(manifest)
        
        # 创建实例
        instance = PluginInstance(
            manifest=manifest,
            status=PluginStatus.INSTALLED,
            path=plugin_path,
            config=config or {},
        )
        
        self._plugins[manifest.id] = instance
        
        logger.info(f"Installed plugin: {manifest.name} ({manifest.id})")
        
        return instance
    
    async def _download_plugin(self, url: str) -> str:
        """从URL下载插件"""
        import tempfile
        import zipfile
        
        # 下载
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        
        # 解压
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "plugin.zip")
        
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        os.unlink(zip_path)
        
        return temp_dir
    
    async def _check_dependencies(self, manifest: PluginManifest):
        """检查依赖"""
        for dep in manifest.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"Missing dependency: {dep}")
    
    async def uninstall(self, plugin_id: str) -> bool:
        """卸载插件"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False
        
        # 先停止
        if instance.status == PluginStatus.RUNNING:
            await self.stop(plugin_id)
        
        # 删除文件
        if os.path.exists(instance.path):
            import shutil
            shutil.rmtree(instance.path)
        
        del self._plugins[plugin_id]
        
        logger.info(f"Uninstalled plugin: {plugin_id}")
        
        return True
    
    async def enable(self, plugin_id: str) -> bool:
        """启用插件"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False
        
        if instance.status not in [PluginStatus.INSTALLED, PluginStatus.DISABLED]:
            return False
        
        instance.enabled = True
        instance.status = PluginStatus.ENABLED
        
        logger.info(f"Enabled plugin: {plugin_id}")
        
        return True
    
    async def disable(self, plugin_id: str) -> bool:
        """禁用插件"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False
        
        if instance.status == PluginStatus.RUNNING:
            await self.stop(plugin_id)
        
        instance.enabled = False
        instance.status = PluginStatus.DISABLED
        
        logger.info(f"Disabled plugin: {plugin_id}")
        
        return True
    
    async def start(self, plugin_id: str) -> bool:
        """启动插件"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False
        
        if instance.status not in [PluginStatus.ENABLED, PluginStatus.ERROR]:
            return False
        
        try:
            # 加载插件模块
            plugin_module = await self._load_plugin_module(instance)
            
            if plugin_module:
                instance.status = PluginStatus.RUNNING
                instance.started_at = time.time()
                
                # 注册钩子
                if instance.manifest.hooks:
                    for hook_name, hook_func in instance.manifest.hooks.items():
                        if hasattr(plugin_module, hook_func):
                            self._register_hook(hook_name, getattr(plugin_module, hook_func))
                
                logger.info(f"Started plugin: {plugin_id}")
                return True
            
        except Exception as e:
            instance.status = PluginStatus.ERROR
            instance.error = str(e)
            logger.error(f"Failed to start plugin {plugin_id}: {e}")
        
        return False
    
    async def _load_plugin_module(self, instance: PluginInstance):
        """加载插件模块"""
        # 添加插件路径到sys.path
        if instance.path not in sys.path:
            sys.path.insert(0, instance.path)
        
        # 导入主模块
        main_file = instance.manifest.main.replace(".py", "")
        
        try:
            module = importlib.import_module(main_file)
            return module
        except Exception as e:
            logger.error(f"Failed to load plugin module: {e}")
            return None
    
    async def stop(self, plugin_id: str) -> bool:
        """停止插件"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False
        
        if instance.status != PluginStatus.RUNNING:
            return False
        
        instance.status = PluginStatus.ENABLED
        
        # 清理钩子
        self._hooks.clear()
        
        logger.info(f"Stopped plugin: {plugin_id}")
        
        return True
    
    def _register_hook(self, hook_name: str, callback: Callable):
        """注册钩子"""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """触发钩子"""
        results = []
        
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        result = await callback(*args, **kwargs)
                    else:
                        result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook callback error: {e}")
        
        return results
    
    async def get_plugin(self, plugin_id: str) -> Optional[PluginInstance]:
        """获取插件"""
        return self._plugins.get(plugin_id)
    
    async def list_plugins(
        self,
        status: Optional[PluginStatus] = None,
        plugin_type: Optional[PluginType] = None,
    ) -> List[PluginInstance]:
        """列出插件"""
        plugins = list(self._plugins.values())
        
        if status:
            plugins = [p for p in plugins if p.status == status]
        
        if plugin_type:
            plugins = [p for p in plugins if p.manifest and p.manifest.plugin_type == plugin_type]
        
        return plugins
    
    async def get_plugin_tools(self, plugin_id: str) -> List[Dict[str, Any]]:
        """获取插件工具"""
        instance = self._plugins.get(plugin_id)
        if not instance or instance.status != PluginStatus.RUNNING:
            return []
        
        # TODO: 从插件模块获取工具
        return []
    
    async def execute_plugin_function(
        self,
        plugin_id: str,
        function_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """执行插件函数"""
        instance = self._plugins.get(plugin_id)
        if not instance or instance.status != PluginStatus.RUNNING:
            raise ValueError(f"Plugin not running: {plugin_id}")
        
        # 在沙箱中执行
        # TODO: 实现安全的函数调用
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_plugins": len(self._plugins),
            "by_status": {
                status.value: len([p for p in self._plugins.values() if p.status == status])
                for status in PluginStatus
            },
            "by_type": {
                ptype.value: len([p for p in self._plugins.values() if p.manifest and p.manifest.plugin_type == ptype])
                for ptype in PluginType
            },
        }


# 全局实例
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


async def init_plugin_manager(config: Optional[Dict[str, Any]] = None):
    """初始化全局插件管理器"""
    global _plugin_manager
    _plugin_manager = PluginManager(config)
    await _plugin_manager.initialize()
    return _plugin_manager
