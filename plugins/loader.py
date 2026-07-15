"""
插件加载器模块

提供插件发现、加载和初始化功能。
"""

import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable
import threading


class PluginLoader:
    """插件加载器
    
    负责发现、加载和初始化插件。
    """
    
    def __init__(self, plugin_dirs: List[str] = None):
        """
        Args:
            plugin_dirs: 插件目录列表
        """
        self._plugin_dirs = plugin_dirs or []
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_classes: Dict[str, Type] = {}
        self._lock = threading.RLock()
    
    def add_plugin_dir(self, directory: str) -> None:
        """添加插件目录"""
        if directory not in self._plugin_dirs:
            self._plugin_dirs.append(directory)
    
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """发现可用插件
        
        Returns:
            插件信息列表
        """
        plugins = []
        
        for directory in self._plugin_dirs:
            if not os.path.exists(directory):
                continue
            
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # 检查是否是Python模块或包
                if os.path.isdir(item_path):
                    # 检查是否有__init__.py
                    init_file = os.path.join(item_path, '__init__.py')
                    if os.path.exists(init_file):
                        plugin_info = self._get_plugin_info(item_path)
                        if plugin_info:
                            plugins.append(plugin_info)
                
                elif item.endswith('.py') and not item.startswith('_'):
                    plugin_info = self._get_plugin_info(item_path)
                    if plugin_info:
                        plugins.append(plugin_info)
        
        return plugins
    
    def _get_plugin_info(self, path: str) -> Optional[Dict[str, Any]]:
        """获取插件信息"""
        # 检查是否有manifest.json
        manifest_path = os.path.join(os.path.dirname(path), 'manifest.json')
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        
        # 从文件名推断
        name = os.path.splitext(os.path.basename(path))[0]
        return {
            'name': name,
            'version': '1.0.0',
            'path': path,
        }
    
    def load_plugin(self, plugin_path: str,
                    class_name: Optional[str] = None) -> Optional[Any]:
        """加载插件
        
        Args:
            plugin_path: 插件路径
            class_name: 类名，None表示自动查找
            
        Returns:
            插件实例
        """
        with self._lock:
            # 检查是否已加载
            if plugin_path in self._loaded_plugins:
                return self._loaded_plugins[plugin_path]
            
            # 加载模块
            module = self._load_module(plugin_path)
            if not module:
                return None
            
            # 查找插件类
            if class_name:
                plugin_class = getattr(module, class_name, None)
            else:
                plugin_class = self._find_plugin_class(module)
            
            if not plugin_class:
                return None
            
            # 实例化
            try:
                instance = plugin_class()
                self._loaded_plugins[plugin_path] = instance
                self._plugin_classes[plugin_path] = plugin_class
                return instance
            except Exception as e:
                print(f"Failed to instantiate plugin: {e}")
                return None
    
    def _load_module(self, path: str) -> Optional[Any]:
        """加载模块"""
        try:
            module_name = os.path.splitext(os.path.basename(path))[0]
            
            if os.path.isdir(path):
                # 包
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    os.path.join(path, '__init__.py')
                )
            else:
                # 模块
                spec = importlib.util.spec_from_file_location(module_name, path)
            
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            print(f"Failed to load module {path}: {e}")
            return None
    
    def _find_plugin_class(self, module: Any) -> Optional[Type]:
        """在模块中查找插件类"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith('_'):
                # 检查是否有get_metadata方法
                if hasattr(obj, 'get_metadata'):
                    return obj
        
        return None
    
    def unload_plugin(self, plugin_path: str) -> bool:
        """卸载插件"""
        with self._lock:
            if plugin_path in self._loaded_plugins:
                del self._loaded_plugins[plugin_path]
                if plugin_path in self._plugin_classes:
                    del self._plugin_classes[plugin_path]
                return True
            return False
    
    def get_loaded_plugins(self) -> Dict[str, Any]:
        """获取已加载的插件"""
        with self._lock:
            return self._loaded_plugins.copy()
    
    def reload_plugin(self, plugin_path: str) -> Optional[Any]:
        """重新加载插件"""
        self.unload_plugin(plugin_path)
        return self.load_plugin(plugin_path)
