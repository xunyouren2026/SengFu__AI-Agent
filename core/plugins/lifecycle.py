"""
插件生命周期管理模块

提供插件的加载、卸载、启用、禁用以及钩子调用功能。
包含依赖解析和错误隔离机制。
仅使用 Python 标准库。
"""

import importlib
import importlib.util
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .dependency_resolver import DependencyResolver, DependencyGraph
from .discovery import PluginDiscovery, PluginInfo
from .spec import PluginSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PluginState - 插件状态枚举
# ---------------------------------------------------------------------------
class PluginState(str, Enum):
    """插件状态"""
    DISCOVERED = "discovered"   # 已发现，未加载
    LOADING = "loading"         # 正在加载
    ACTIVE = "active"           # 已激活，可用
    DISABLED = "disabled"       # 已禁用
    ERROR = "error"             # 错误状态
    UNLOADED = "unloaded"       # 已卸载


# ---------------------------------------------------------------------------
# PluginInstance - 插件实例
# ---------------------------------------------------------------------------
@dataclass
class PluginInstance:
    """插件运行时实例"""
    spec: PluginSpec
    state: PluginState = PluginState.DISCOVERED
    module: Any = None
    instance: Any = None
    error: Optional[str] = None
    loaded_at: Optional[float] = None
    enabled_at: Optional[float] = None
    hooks: Dict[str, List[Callable]] = field(default_factory=dict)
    provided_tools: List[Any] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def version(self) -> str:
        return self.spec.version

    def to_dict(self) -> dict:
        return {
            "name": self.spec.name,
            "version": self.spec.version,
            "state": self.state.value,
            "error": self.error,
            "loaded_at": self.loaded_at,
            "enabled_at": self.enabled_at,
            "hooks": {k: len(v) for k, v in self.hooks.items()},
            "tools_count": len(self.provided_tools),
        }


# ---------------------------------------------------------------------------
# PluginLifecycleManager - 插件生命周期管理
# ---------------------------------------------------------------------------
class PluginLifecycleManager:
    """插件生命周期管理器

    管理插件的完整生命周期：发现 -> 加载 -> 启用 -> 禁用 -> 卸载。
    支持:
    - 依赖解析和拓扑排序加载
    - 钩子调用 (on_load/on_enable/on_disable/on_unload)
    - 错误隔离（单个插件异常不影响其他插件）
    - 线程安全
    """

    def __init__(self):
        self._plugins: OrderedDict[str, PluginInstance] = OrderedDict()
        self._lock = threading.RLock()
        self._discovery = PluginDiscovery()
        self._dependency_resolver = DependencyResolver()
        self._global_hooks: Dict[str, List[Tuple[int, Callable]]] = {}
        self._event_listeners: Dict[str, List[Callable]] = {}

    @property
    def discovery(self) -> PluginDiscovery:
        return self._discovery

    # ----- 发现 -----

    def discover(self, path: str) -> List[PluginInfo]:
        """发现插件"""
        infos = self._discovery.scan_directory(path)
        with self._lock:
            for info in infos:
                if info.spec and info.name not in self._plugins:
                    instance = PluginInstance(
                        spec=info.spec,
                        state=PluginState.DISCOVERED,
                    )
                    self._plugins[info.name] = instance
        return infos

    def discover_all(self) -> List[PluginInfo]:
        """发现所有搜索路径中的插件"""
        infos = self._discovery.scan_all()
        with self._lock:
            for info in infos:
                if info.spec and info.name not in self._plugins:
                    instance = PluginInstance(
                        spec=info.spec,
                        state=PluginState.DISCOVERED,
                    )
                    self._plugins[info.name] = instance
        return infos

    # ----- 加载 -----

    def load_plugin(self, name: str) -> bool:
        """加载插件

        自动解析依赖，按拓扑排序加载依赖插件。

        Returns:
            是否成功
        """
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                logger.error("插件 '%s' 未发现", name)
                return False

            if instance.state in (PluginState.ACTIVE, PluginState.LOADING):
                return True

            if instance.state == PluginState.DISABLED:
                return self.enable_plugin(name)

        # 解析依赖并确定加载顺序
        load_order = self._resolve_load_order(name)
        if load_order is None:
            with self._lock:
                instance.state = PluginState.ERROR
                instance.error = "依赖解析失败"
            return False

        # 按顺序加载
        for plugin_name in load_order:
            with self._lock:
                pi = self._plugins.get(plugin_name)
                if pi is None or pi.state in (PluginState.ACTIVE, PluginState.LOADING):
                    continue

            success = self._load_single(plugin_name)
            if not success:
                logger.error("插件 '%s' 加载失败", plugin_name)
                return plugin_name == name  # 如果依赖加载失败，返回 False

        return True

    def load_all(self) -> Dict[str, bool]:
        """加载所有已发现的插件

        Returns:
            {插件名: 是否成功}
        """
        results = {}
        with self._lock:
            names = list(self._plugins.keys())

        for name in names:
            results[name] = self.load_plugin(name)
        return results

    def _load_single(self, name: str) -> bool:
        """加载单个插件（内部方法）"""
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                return False

            instance.state = PluginState.LOADING
            instance.error = None

        try:
            # 导入模块
            module, obj = self._import_plugin(instance.spec)
            instance.module = module
            instance.instance = obj

            # 调用 on_load 钩子
            self._call_plugin_hook(instance, "on_load")

            # 注册插件提供的工具
            self._register_plugin_tools(instance)

            # 注册插件钩子
            self._register_plugin_hooks(instance)

            with self._lock:
                instance.state = PluginState.ACTIVE
                instance.loaded_at = time.time()

            self._emit_event("plugin_loaded", instance)
            logger.info("插件 '%s' v%s 加载成功", name, instance.version)
            return True

        except Exception as exc:
            with self._lock:
                instance.state = PluginState.ERROR
                instance.error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "插件 '%s' 加载失败: %s\n%s",
                name, exc, traceback.format_exc(),
            )
            self._emit_event("plugin_error", instance, error=str(exc))
            return False

    # ----- 卸载 -----

    def unload_plugin(self, name: str) -> bool:
        """卸载插件

        Returns:
            是否成功
        """
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                return False

            if instance.state not in (PluginState.ACTIVE, PluginState.DISABLED, PluginState.ERROR):
                return True

        # 先卸载依赖此插件的其他插件
        dependents = self._get_dependents(name)
        for dep_name in dependents:
            self.unload_plugin(dep_name)

        try:
            # 调用 on_unload 钩子
            self._call_plugin_hook(instance, "on_unload")

            # 注销工具
            self._unregister_plugin_tools(instance)

            # 注销钩子
            self._unregister_plugin_hooks(instance)

            with self._lock:
                instance.state = PluginState.UNLOADED
                instance.module = None
                instance.instance = None
                instance.provided_tools.clear()
                instance.hooks.clear()

            self._emit_event("plugin_unloaded", instance)
            logger.info("插件 '%s' 已卸载", name)
            return True

        except Exception as exc:
            with self._lock:
                instance.state = PluginState.ERROR
                instance.error = f"卸载失败: {exc}"
            logger.error("插件 '%s' 卸载失败: %s", name, exc)
            return False

    # ----- 启用 / 禁用 -----

    def enable_plugin(self, name: str) -> bool:
        """启用插件"""
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                return False

            if instance.state == PluginState.ACTIVE:
                return True

            if instance.state == PluginState.DISCOVERED:
                return self.load_plugin(name)

            if instance.state != PluginState.DISABLED:
                return False

        try:
            self._call_plugin_hook(instance, "on_enable")

            with self._lock:
                instance.state = PluginState.ACTIVE
                instance.enabled_at = time.time()

            self._emit_event("plugin_enabled", instance)
            logger.info("插件 '%s' 已启用", name)
            return True

        except Exception as exc:
            with self._lock:
                instance.state = PluginState.ERROR
                instance.error = f"启用失败: {exc}"
            logger.error("插件 '%s' 启用失败: %s", name, exc)
            return False

    def disable_plugin(self, name: str) -> bool:
        """禁用插件"""
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                return False

            if instance.state == PluginState.DISABLED:
                return True

            if instance.state != PluginState.ACTIVE:
                return False

        try:
            self._call_plugin_hook(instance, "on_disable")

            with self._lock:
                instance.state = PluginState.DISABLED

            self._emit_event("plugin_disabled", instance)
            logger.info("插件 '%s' 已禁用", name)
            return True

        except Exception as exc:
            with self._lock:
                instance.state = PluginState.ERROR
                instance.error = f"禁用失败: {exc}"
            logger.error("插件 '%s' 禁用失败: %s", name, exc)
            return False

    # ----- 查询 -----

    def get_plugin(self, name: str) -> Optional[PluginInstance]:
        """获取插件实例"""
        with self._lock:
            return self._plugins.get(name)

    def get_plugin_status(self, name: str) -> Optional[dict]:
        """获取插件状态"""
        with self._lock:
            instance = self._plugins.get(name)
            if instance is None:
                return None
            return instance.to_dict()

    def list_plugins(
        self, state: Optional[PluginState] = None
    ) -> List[dict]:
        """列出所有插件"""
        with self._lock:
            if state is not None:
                return [
                    p.to_dict() for p in self._plugins.values()
                    if p.state == state
                ]
            return [p.to_dict() for p in self._plugins.values()]

    def get_active_plugins(self) -> List[PluginInstance]:
        """获取所有活跃插件"""
        with self._lock:
            return [
                p for p in self._plugins.values()
                if p.state == PluginState.ACTIVE
            ]

    # ----- 钩子系统 -----

    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """调用全局钩子

        按优先级排序调用所有注册的钩子函数。

        Returns:
            所有钩子的返回值列表
        """
        results = []

        # 调用全局钩子
        with self._lock:
            hooks = self._global_hooks.get(hook_name, [])

        for priority, callback in sorted(hooks, key=lambda x: x[0]):
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "全局钩子 '%s' 执行异常: %s", hook_name, exc
                )

        # 调用插件钩子
        with self._lock:
            active_plugins = [
                p for p in self._plugins.values()
                if p.state == PluginState.ACTIVE
            ]

        for plugin in active_plugins:
            plugin_hooks = plugin.hooks.get(hook_name, [])
            for callback in plugin_hooks:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as exc:
                    logger.warning(
                        "插件 '%s' 的钩子 '%s' 执行异常: %s",
                        plugin.name, hook_name, exc,
                    )

        return results

    def register_global_hook(
        self, hook_name: str, callback: Callable, priority: int = 100
    ) -> None:
        """注册全局钩子"""
        with self._lock:
            if hook_name not in self._global_hooks:
                self._global_hooks[hook_name] = []
            self._global_hooks[hook_name].append((priority, callback))

    def unregister_global_hook(
        self, hook_name: str, callback: Callable
    ) -> bool:
        """注销全局钩子"""
        with self._lock:
            hooks = self._global_hooks.get(hook_name, [])
            for i, (_, cb) in enumerate(hooks):
                if cb is callback:
                    hooks.pop(i)
                    return True
            return False

    # ----- 事件监听 -----

    def add_event_listener(
        self, event: str, callback: Callable
    ) -> None:
        """添加事件监听器"""
        with self._lock:
            if event not in self._event_listeners:
                self._event_listeners[event] = []
            self._event_listeners[event].append(callback)

    def remove_event_listener(
        self, event: str, callback: Callable
    ) -> bool:
        """移除事件监听器"""
        with self._lock:
            listeners = self._event_listeners.get(event, [])
            try:
                listeners.remove(callback)
                return True
            except ValueError:
                return False

    def _emit_event(self, event: str, *args, **kwargs) -> None:
        """触发事件"""
        with self._lock:
            listeners = self._event_listeners.get(event, [])
        for listener in listeners:
            try:
                listener(*args, **kwargs)
            except Exception:
                pass

    # ----- 内部方法 -----

    def _import_plugin(
        self, spec: PluginSpec
    ) -> Tuple[Any, Any]:
        """导入插件模块和实例"""
        if not spec.entry_point:
            raise ValueError(f"插件 '{spec.name}' 没有入口点")

        module_path, callable_name = spec.entry_point.split(":", 1)

        # 尝试从已加载的模块中获取
        if module_path in sys.modules:
            module = sys.modules[module_path]
        else:
            module = importlib.import_module(module_path)

        # 获取可调用对象
        obj = getattr(module, callable_name, None)
        if obj is None:
            raise AttributeError(
                f"插件 '{spec.name}' 入口点 '{callable_name}' 不存在于 "
                f"模块 '{module_path}'"
            )

        # 如果是类，实例化
        if isinstance(obj, type):
            obj = obj()

        return module, obj

    def _call_plugin_hook(
        self, instance: PluginInstance, hook_name: str
    ) -> None:
        """调用插件生命周期钩子"""
        obj = instance.instance
        if obj is None:
            return

        hook_fn = getattr(obj, hook_name, None)
        if hook_fn is not None and callable(hook_fn):
            hook_fn()

    def _register_plugin_tools(self, instance: PluginInstance) -> None:
        """注册插件提供的工具"""
        obj = instance.instance
        if obj is None:
            return

        # 查找 get_tools 方法
        get_tools = getattr(obj, "get_tools", None)
        if get_tools is not None and callable(get_tools):
            try:
                tools = get_tools()
                if isinstance(tools, (list, tuple)):
                    instance.provided_tools.extend(tools)
            except Exception as exc:
                logger.warning(
                    "插件 '%s' 的 get_tools() 调用失败: %s",
                    instance.name, exc,
                )

    def _unregister_plugin_tools(self, instance: PluginInstance) -> None:
        """注销插件提供的工具"""
        instance.provided_tools.clear()

    def _register_plugin_hooks(self, instance: PluginInstance) -> None:
        """注册插件钩子"""
        obj = instance.instance
        if obj is None:
            return

        # 查找 register_hooks 方法
        register_hooks = getattr(obj, "register_hooks", None)
        if register_hooks is not None and callable(register_hooks):
            try:
                hooks = register_hooks()
                if isinstance(hooks, dict):
                    for hook_name, callbacks in hooks.items():
                        if isinstance(callbacks, list):
                            instance.hooks[hook_name] = callbacks
                        elif callable(callbacks):
                            instance.hooks[hook_name] = [callbacks]
            except Exception as exc:
                logger.warning(
                    "插件 '%s' 的 register_hooks() 调用失败: %s",
                    instance.name, exc,
                )

    def _unregister_plugin_hooks(self, instance: PluginInstance) -> None:
        """注销插件钩子"""
        instance.hooks.clear()

    def _resolve_load_order(self, name: str) -> Optional[List[str]]:
        """解析依赖并确定加载顺序（拓扑排序）"""
        with self._lock:
            # 收集所有相关插件
            all_specs: Dict[str, PluginSpec] = {}
            for pname, pinstance in self._plugins.items():
                all_specs[pname] = pinstance.spec

            # 构建依赖图
            graph = DependencyGraph()
            for pname, spec in all_specs.items():
                graph.add_node(pname)
                for dep in spec.dependencies:
                    graph.add_edge(pname, dep.name)

            # 解析
            resolver = DependencyResolver()
            result = resolver.resolve(graph)

            if not result.success:
                instance = self._plugins.get(name)
                if instance:
                    instance.error = f"依赖解析失败: {result.errors}"
                return None

            # 找到目标插件在排序中的位置，返回它及其所有依赖
            if name not in result.sorted:
                instance = self._plugins.get(name)
                if instance:
                    instance.error = f"插件 '{name}' 不在依赖图中"
                return None

            idx = result.sorted.index(name)
            return result.sorted[:idx + 1]

    def _get_dependents(self, name: str) -> List[str]:
        """获取依赖此插件的其他插件"""
        dependents = []
        with self._lock:
            for pname, pinstance in self._plugins.items():
                if pname == name:
                    continue
                for dep in pinstance.spec.dependencies:
                    if dep.name == name:
                        dependents.append(pname)
                        break
        return dependents

    # ----- 清理 -----

    def shutdown(self) -> None:
        """关闭管理器，卸载所有插件"""
        with self._lock:
            names = list(self._plugins.keys())

        for name in reversed(names):
            instance = self._plugins.get(name)
            if instance and instance.state == PluginState.ACTIVE:
                self.unload_plugin(name)

        with self._lock:
            self._plugins.clear()
            self._global_hooks.clear()
            self._event_listeners.clear()
