"""
插件管理器模块

提供插件生命周期管理、依赖解析和配置管理功能。
"""

import json
import os
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .loader import PluginLoader
from .sandbox import PluginSandbox, SandboxConfig


@dataclass
class PluginInstance:
    """插件实例"""
    plugin_id: str
    name: str
    version: str
    instance: Any
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    loaded_at: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'loaded_at': self.loaded_at.isoformat(),
            'error_count': self.error_count,
        }


class PluginManager:
    """插件管理器
    
    管理插件的生命周期、依赖和配置。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 配置文件路径
        """
        self._config_path = config_path or os.path.expanduser('~/.clawhub/plugins.json')
        self._loader = PluginLoader()
        self._sandbox = PluginSandbox()
        self._plugins: Dict[str, PluginInstance] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # 加载配置
        self._load_config()
    
    def register(self, plugin_id: str, instance: Any,
                 config: Optional[Dict[str, Any]] = None) -> bool:
        """注册插件
        
        Args:
            plugin_id: 插件ID
            instance: 插件实例
            config: 配置
            
        Returns:
            是否成功
        """
        with self._lock:
            if plugin_id in self._plugins:
                return False
            
            # 获取元数据
            metadata = instance.get_metadata() if hasattr(instance, 'get_metadata') else {}
            
            plugin_instance = PluginInstance(
                plugin_id=plugin_id,
                name=metadata.get('name', plugin_id),
                version=metadata.get('version', '1.0.0'),
                instance=instance,
                config=config or {},
            )
            
            self._plugins[plugin_id] = plugin_instance
            
            # 触发钩子
            self._trigger_hook('on_register', plugin_instance)
            
            return True
    
    def unregister(self, plugin_id: str) -> bool:
        """注销插件"""
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            
            plugin = self._plugins[plugin_id]
            
            # 触发钩子
            self._trigger_hook('on_unregister', plugin)
            
            del self._plugins[plugin_id]
            return True
    
    def get(self, plugin_id: str) -> Optional[Any]:
        """获取插件实例"""
        with self._lock:
            plugin = self._plugins.get(plugin_id)
            return plugin.instance if plugin else None
    
    def list_plugins(self, enabled_only: bool = False) -> List[PluginInstance]:
        """列出插件"""
        with self._lock:
            plugins = list(self._plugins.values())
            if enabled_only:
                plugins = [p for p in plugins if p.enabled]
            return plugins
    
    def enable(self, plugin_id: str) -> bool:
        """启用插件"""
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            self._plugins[plugin_id].enabled = True
            return True
    
    def disable(self, plugin_id: str) -> bool:
        """禁用插件"""
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            self._plugins[plugin_id].enabled = False
            return True
    
    def configure(self, plugin_id: str, config: Dict[str, Any]) -> bool:
        """配置插件"""
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            self._plugins[plugin_id].config.update(config)
            self._save_config()
            return True
    
    def execute(self, plugin_id: str, method: str,
                *args, **kwargs) -> Any:
        """执行插件方法
        
        Args:
            plugin_id: 插件ID
            method: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            执行结果
        """
        with self._lock:
            plugin = self._plugins.get(plugin_id)
            if not plugin or not plugin.enabled:
                raise ValueError(f"Plugin {plugin_id} not found or disabled")
            
            func = getattr(plugin.instance, method, None)
            if not func:
                raise AttributeError(f"Method {method} not found in plugin {plugin_id}")
            
            # 在沙箱中执行
            result = self._sandbox.execute(func, *args, **kwargs)
            
            if not result.success:
                plugin.error_count += 1
            
            if result.killed:
                raise TimeoutError(f"Plugin {plugin_id} execution timed out")
            
            return result.return_value
    
    def add_hook(self, event: str, callback: Callable) -> None:
        """添加钩子"""
        with self._lock:
            if event not in self._hooks:
                self._hooks[event] = []
            self._hooks[event].append(callback)
    
    def remove_hook(self, event: str, callback: Callable) -> None:
        """移除钩子"""
        with self._lock:
            if event in self._hooks:
                if callback in self._hooks[event]:
                    self._hooks[event].remove(callback)
    
    def _trigger_hook(self, event: str, plugin: PluginInstance) -> None:
        """触发钩子"""
        if event in self._hooks:
            for callback in self._hooks[event]:
                try:
                    callback(plugin)
                except Exception:
                    pass
    
    def _load_config(self) -> None:
        """加载配置"""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    config = json.load(f)
                    # 应用配置
            except json.JSONDecodeError:
                pass
    
    def _save_config(self) -> None:
        """保存配置"""
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        
        config = {
            'plugins': {
                pid: p.to_dict()
                for pid, p in self._plugins.items()
            }
        }
        
        with open(self._config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'total': len(self._plugins),
                'enabled': sum(1 for p in self._plugins.values() if p.enabled),
                'disabled': sum(1 for p in self._plugins.values() if not p.enabled),
                'total_errors': sum(p.error_count for p in self._plugins.values()),
            }
