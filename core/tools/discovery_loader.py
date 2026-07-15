"""
工具发现与动态加载模块

提供文件系统扫描、模块内省、插件发现、懒加载、热重载和依赖注入功能。
仅使用 Python 标准库。
"""

import abc
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import re
import sys
import threading
import time
import traceback
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

from .base import Tool, ToolParameter, FunctionTool


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------
T = TypeVar("T")
ToolFactory = Callable[..., Tool]


class ToolSource(Protocol):
    """工具源协议"""

    def get_tools(self) -> List[Tool]:
        ...


@dataclass
class DiscoveredTool:
    """发现的工具信息

    Attributes:
        name: 工具名称
        tool_class: 工具类
        module_path: 模块路径
        source_file: 源文件路径
        metadata: 元数据
        load_time: 加载时间戳
        checksum: 文件校验和
    """
    name: str
    tool_class: Type[Tool]
    module_path: str
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    load_time: float = field(default_factory=time.time)
    checksum: str = ""


@dataclass
class PluginInfo:
    """插件信息

    Attributes:
        name: 插件名称
        version: 版本
        entry_point: 入口点
        dependencies: 依赖列表
        config: 配置字典
        enabled: 是否启用
    """
    name: str
    version: str
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


# ---------------------------------------------------------------------------
# FilesystemScanner - 文件系统扫描器
# ---------------------------------------------------------------------------
class FilesystemScanner:
    """文件系统扫描器

    扫描指定目录下的 Python 文件，发现潜在的工具定义。
    """

    def __init__(
        self,
        root_paths: List[Union[str, Path]],
        pattern: str = r".*\.py$",
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.root_paths = [Path(p) for p in root_paths]
        self.pattern = re.compile(pattern)
        self.exclude_patterns = exclude_patterns or [
            r"__pycache__",
            r"\.git",
            r"\.venv",
            r"venv",
            r"test_.*",
            r".*_test\.py$",
        ]
        self._exclude_regex = [re.compile(p) for p in self.exclude_patterns]
        self._scan_cache: Dict[str, float] = {}

    def scan(self, force_rescan: bool = False) -> Iterator[Path]:
        """扫描文件系统，返回匹配的 Python 文件路径

        Args:
            force_rescan: 强制重新扫描，忽略缓存

        Yields:
            匹配的文件路径
        """
        for root in self.root_paths:
            if not root.exists():
                warnings.warn(f"扫描路径不存在: {root}")
                continue

            for file_path in self._walk_files(root):
                if self._should_include(file_path):
                    stat = file_path.stat()
                    cache_key = str(file_path)
                    mtime = stat.st_mtime

                    if not force_rescan and cache_key in self._scan_cache:
                        if self._scan_cache[cache_key] == mtime:
                            continue

                    self._scan_cache[cache_key] = mtime
                    yield file_path

    def _walk_files(self, root: Path) -> Iterator[Path]:
        """递归遍历目录"""
        try:
            for entry in os.scandir(root):
                if self._is_excluded(entry.name):
                    continue

                if entry.is_dir(follow_symlinks=False):
                    yield from self._walk_files(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path)
        except PermissionError:
            warnings.warn(f"权限拒绝: {root}")
        except OSError as e:
            warnings.warn(f"扫描错误 {root}: {e}")

    def _is_excluded(self, name: str) -> bool:
        """检查是否被排除"""
        for regex in self._exclude_regex:
            if regex.search(name):
                return True
        return False

    def _should_include(self, path: Path) -> bool:
        """检查文件是否应该被包含"""
        return bool(self.pattern.match(path.name))

    def compute_checksum(self, file_path: Union[str, Path]) -> str:
        """计算文件 MD5 校验和"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_module_name(self, file_path: Union[str, Path]) -> str:
        """从文件路径推导模块名称"""
        path = Path(file_path).resolve()

        # 尝试找到包根目录
        for root in self.root_paths:
            root_resolved = root.resolve()
            try:
                relative = path.relative_to(root_resolved)
                parts = list(relative.with_suffix("").parts)
                return ".".join(parts)
            except ValueError:
                continue

        # 回退：使用文件名
        return path.stem


# ---------------------------------------------------------------------------
# ModuleIntrospector - 模块内省器
# ---------------------------------------------------------------------------
class ModuleIntrospector:
    """模块内省器

    分析 Python 模块，提取工具类定义和元数据。
    """

    TOOL_BASE_CLASSES = ("Tool", "BaseTool", "FunctionTool")

    def __init__(self):
        self._ast_cache: Dict[str, ast.AST] = {}

    def introspect_file(self, file_path: Union[str, Path]) -> List[DiscoveredTool]:
        """内省单个文件，返回发现的工具列表"""
        path = Path(file_path)
        if not path.exists():
            return []

        try:
            source = path.read_text(encoding="utf-8")
            tree = self._parse_ast(str(path), source)
            module_name = self._get_module_name(path)
            return self._extract_tools_from_ast(tree, module_name, str(path))
        except SyntaxError as e:
            warnings.warn(f"语法错误 {path}: {e}")
            return []
        except UnicodeDecodeError:
            warnings.warn(f"编码错误 {path}")
            return []

    def introspect_module(self, module: types.ModuleType) -> List[DiscoveredTool]:
        """内省已加载的模块"""
        discovered: List[DiscoveredTool] = []
        module_name = getattr(module, "__name__", "unknown")
        module_file = getattr(module, "__file__", "")

        for name, obj in inspect.getmembers(module):
            if self._is_tool_class(obj):
                discovered.append(
                    DiscoveredTool(
                        name=name,
                        tool_class=obj,
                        module_path=module_name,
                        source_file=module_file,
                        metadata=self._extract_metadata(obj),
                    )
                )

        return discovered

    def _parse_ast(self, cache_key: str, source: str) -> ast.AST:
        """解析源代码为 AST"""
        if cache_key not in self._ast_cache:
            self._ast_cache[cache_key] = ast.parse(source)
        return self._ast_cache[cache_key]

    def _extract_tools_from_ast(
        self, tree: ast.AST, module_name: str, file_path: str
    ) -> List[DiscoveredTool]:
        """从 AST 提取工具类定义"""
        discovered: List[DiscoveredTool] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self._is_tool_class_def(node):
                    metadata = self._extract_metadata_from_ast(node)
                    discovered.append(
                        DiscoveredTool(
                            name=node.name,
                            tool_class=None,  # 动态加载时填充
                            module_path=module_name,
                            source_file=file_path,
                            metadata=metadata,
                        )
                    )

        return discovered

    def _is_tool_class_def(self, node: ast.ClassDef) -> bool:
        """检查 AST 节点是否是工具类定义"""
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in self.TOOL_BASE_CLASSES:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in self.TOOL_BASE_CLASSES:
                    return True
        return False

    def _is_tool_class(self, obj: Any) -> bool:
        """检查对象是否是工具类"""
        return (
            inspect.isclass(obj)
            and issubclass(obj, Tool)
            and obj is not Tool
            and not inspect.isabstract(obj)
        )

    def _extract_metadata(self, cls: Type[Tool]) -> Dict[str, Any]:
        """从工具类提取元数据"""
        metadata: Dict[str, Any] = {
            "docstring": inspect.getdoc(cls) or "",
            "line_number": getattr(cls, "__line_number__", 0),
        }

        # 提取类属性
        for attr in ["version", "category", "tags"]:
            if hasattr(cls, attr):
                metadata[attr] = getattr(cls, attr)

        return metadata

    def _extract_metadata_from_ast(self, node: ast.ClassDef) -> Dict[str, Any]:
        """从 AST 节点提取元数据"""
        metadata: Dict[str, Any] = {
            "line_number": node.lineno,
            "docstring": ast.get_docstring(node) or "",
        }

        # 提取类级别的赋值
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(item.value, ast.Constant):
                            metadata[target.id] = item.value.value
                        elif isinstance(item.value, ast.List):
                            metadata[target.id] = self._extract_list_values(item.value)

        return metadata

    def _extract_list_values(self, node: ast.List) -> List[Any]:
        """从 AST 列表节点提取值"""
        values = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                values.append(elt.value)
            elif isinstance(elt, ast.Str):  # Python < 3.8
                values.append(elt.s)
        return values

    def _get_module_name(self, path: Path) -> str:
        """获取模块名称"""
        # 尝试找到包根
        for parent in path.parents:
            if (parent / "__init__.py").exists():
                continue
            # parent 是包根
            try:
                relative = path.relative_to(parent)
                parts = list(relative.with_suffix("").parts)
                return ".".join(parts)
            except ValueError:
                pass

        return path.stem


# ---------------------------------------------------------------------------
# LazyLoader - 懒加载器
# ---------------------------------------------------------------------------
class LazyLoader(Generic[T]):
    """懒加载器

    延迟加载工具直到首次使用。
    """

    def __init__(
        self,
        factory: Callable[[], T],
        name: str = "",
        preload: bool = False,
    ):
        self._factory = factory
        self._name = name
        self._instance: Optional[T] = None
        self._loaded = False
        self._lock = threading.RLock()
        self._load_time: Optional[float] = None
        self._error: Optional[Exception] = None

        if preload:
            self._load()

    @property
    def loaded(self) -> bool:
        """是否已加载"""
        return self._loaded

    @property
    def load_time(self) -> Optional[float]:
        """加载耗时"""
        return self._load_time

    @property
    def error(self) -> Optional[Exception]:
        """加载错误"""
        return self._error

    def get(self) -> T:
        """获取实例（懒加载）"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()
        if self._error:
            raise RuntimeError(f"懒加载失败 {self._name}: {self._error}")
        return self._instance

    def _load(self) -> None:
        """执行加载"""
        start = time.monotonic()
        try:
            self._instance = self._factory()
            self._error = None
        except Exception as e:
            self._error = e
            self._instance = None
        finally:
            self._load_time = time.monotonic() - start
            self._loaded = True

    def reload(self) -> None:
        """重新加载"""
        with self._lock:
            self._loaded = False
            self._instance = None
            self._error = None
            self._load_time = None


# ---------------------------------------------------------------------------
# HotReloader - 热重载器
# ---------------------------------------------------------------------------
class HotReloader:
    """热重载器

    监视文件变化并自动重新加载工具。
    """

    def __init__(
        self,
        scanner: FilesystemScanner,
        introspector: ModuleIntrospector,
        check_interval: float = 1.0,
    ):
        self.scanner = scanner
        self.introspector = introspector
        self.check_interval = check_interval
        self._file_hashes: Dict[str, str] = {}
        self._loaded_modules: Dict[str, types.ModuleType] = {}
        self._callbacks: List[Callable[[str, str], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def add_callback(self, callback: Callable[[str, str], None]) -> None:
        """添加文件变化回调

        Args:
            callback: 接收 (file_path, event_type) 的回调函数
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, str], None]) -> None:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> None:
        """启动热重载监视"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止热重载监视"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def check_now(self) -> List[Tuple[str, str]]:
        """立即检查变化

        Returns:
            变化的文件列表 [(file_path, event_type), ...]
        """
        changes: List[Tuple[str, str]] = []

        for file_path in self.scanner.scan():
            path_str = str(file_path)
            current_hash = self.scanner.compute_checksum(file_path)

            with self._lock:
                if path_str in self._file_hashes:
                    if self._file_hashes[path_str] != current_hash:
                        changes.append((path_str, "modified"))
                        self._file_hashes[path_str] = current_hash
                else:
                    changes.append((path_str, "added"))
                    self._file_hashes[path_str] = current_hash

        # 检查删除的文件
        with self._lock:
            current_files = {str(p) for p in self.scanner.scan()}
            for path_str in list(self._file_hashes.keys()):
                if path_str not in current_files:
                    changes.append((path_str, "deleted"))
                    del self._file_hashes[path_str]

        # 触发回调
        for path_str, event in changes:
            self._notify_change(path_str, event)

        return changes

    def _watch_loop(self) -> None:
        """监视循环"""
        while self._running:
            try:
                self.check_now()
            except Exception as e:
                warnings.warn(f"热重载检查错误: {e}")
            time.sleep(self.check_interval)

    def _notify_change(self, file_path: str, event_type: str) -> None:
        """通知变化"""
        for callback in self._callbacks:
            try:
                callback(file_path, event_type)
            except Exception as e:
                warnings.warn(f"热重载回调错误: {e}")

    def reload_module(self, file_path: str) -> Optional[types.ModuleType]:
        """重新加载模块"""
        module_name = self.scanner.get_module_name(file_path)

        if module_name in sys.modules:
            try:
                module = importlib.reload(sys.modules[module_name])
                self._loaded_modules[module_name] = module
                return module
            except Exception as e:
                warnings.warn(f"重新加载模块失败 {module_name}: {e}")
                return None
        else:
            # 首次加载
            return self._load_module(file_path)

    def _load_module(self, file_path: str) -> Optional[types.ModuleType]:
        """加载模块"""
        module_name = self.scanner.get_module_name(file_path)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
            self._loaded_modules[module_name] = module
            return module
        except Exception as e:
            warnings.warn(f"加载模块失败 {module_name}: {e}")
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None


# ---------------------------------------------------------------------------
# DependencyInjector - 依赖注入器
# ---------------------------------------------------------------------------
class DependencyInjector:
    """依赖注入器

    管理工具依赖关系，支持构造函数注入和属性注入。
    """

    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_locks: Dict[str, threading.Lock] = {}
        self._dependencies: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        instance: Any = None,
        factory: Optional[Callable[[], Any]] = None,
        singleton: bool = False,
    ) -> None:
        """注册依赖

        Args:
            name: 依赖名称
            instance: 实例（直接注册）
            factory: 工厂函数
            singleton: 是否为单例
        """
        if instance is not None:
            self._registry[name] = instance
        elif factory is not None:
            self._factories[name] = factory
            if singleton:
                self._singleton_locks[name] = threading.Lock()
        else:
            raise ValueError("必须提供 instance 或 factory")

    def register_singleton(self, name: str, factory: Callable[[], Any]) -> None:
        """注册单例"""
        self.register(name, factory=factory, singleton=True)

    def register_instance(self, name: str, instance: Any) -> None:
        """注册实例"""
        self.register(name, instance=instance)

    def resolve(self, name: str) -> Any:
        """解析依赖"""
        # 1. 直接注册的实例
        if name in self._registry:
            return self._registry[name]

        # 2. 已创建的单例
        if name in self._singletons:
            return self._singletons[name]

        # 3. 工厂创建
        if name in self._factories:
            factory = self._factories[name]

            # 单例模式
            if name in self._singleton_locks:
                with self._singleton_locks[name]:
                    if name not in self._singletons:
                        self._singletons[name] = self._create_with_deps(name, factory)
                return self._singletons[name]

            return self._create_with_deps(name, factory)

        raise KeyError(f"未找到依赖: {name}")

    def _create_with_deps(
        self, name: str, factory: Callable[[], Any]
    ) -> Any:
        """创建实例并解析依赖"""
        # 检查工厂函数签名
        sig = inspect.signature(factory)
        kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # 尝试从依赖注入器解析
            try:
                kwargs[param_name] = self.resolve(param_name)
            except KeyError:
                # 使用默认值
                if param.default is not inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                else:
                    raise KeyError(
                        f"无法解析依赖 '{param_name}' 用于 '{name}'"
                    )

        return factory(**kwargs)

    def inject(self, obj: Any) -> Any:
        """向对象注入依赖

        支持构造函数注入和属性注入。
        """
        if inspect.isclass(obj):
            # 类注入 - 返回实例
            return self._inject_class(obj)
        else:
            # 实例属性注入
            self._inject_instance(obj)
            return obj

    def _inject_class(self, cls: Type[T]) -> T:
        """类构造函数注入"""
        sig = inspect.signature(cls.__init__)
        kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            try:
                kwargs[param_name] = self.resolve(param_name)
            except KeyError:
                if param.default is not inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                elif param_name in self._get_type_hints(cls):
                    # 尝试按类型解析
                    type_hint = self._get_type_hints(cls)[param_name]
                    for dep_name, dep in self._registry.items():
                        if isinstance(dep, type_hint):
                            kwargs[param_name] = dep
                            break

        return cls(**kwargs)

    def _inject_instance(self, obj: Any) -> None:
        """实例属性注入"""
        cls = type(obj)
        type_hints = self._get_type_hints(cls)

        for attr_name, type_hint in type_hints.items():
            if hasattr(obj, attr_name) and getattr(obj, attr_name) is not None:
                continue

            # 尝试按名称解析
            try:
                value = self.resolve(attr_name)
                setattr(obj, attr_name, value)
                continue
            except KeyError:
                pass

            # 尝试按类型解析
            for dep_name, dep in self._registry.items():
                if isinstance(dep, type_hint):
                    setattr(obj, attr_name, dep)
                    break

    def _get_type_hints(self, obj: Any) -> Dict[str, Any]:
        """获取类型提示"""
        try:
            return get_type_hints(obj)
        except Exception:
            return {}

    def create_scope(self) -> "DependencyInjector":
        """创建子作用域"""
        child = DependencyInjector()
        child._registry = dict(self._registry)
        child._factories = dict(self._factories)
        child._singleton_locks = dict(self._singleton_locks)
        return child


# ---------------------------------------------------------------------------
# PluginDiscovery - 插件发现器
# ---------------------------------------------------------------------------
class PluginDiscovery:
    """插件发现器

    发现和加载插件，支持入口点、配置文件和动态加载。
    """

    PLUGIN_CONFIG_FILES = ["plugin.json", "pyproject.toml", "setup.py"]

    def __init__(
        self,
        plugin_dirs: Optional[List[Union[str, Path]]] = None,
        entry_point_group: str = "agi_unified_framework.tools",
    ):
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self.entry_point_group = entry_point_group
        self._plugins: Dict[str, PluginInfo] = {}
        self._loaded_tools: Dict[str, Tool] = {}
        self._injector = DependencyInjector()

    def discover(self) -> List[PluginInfo]:
        """发现所有可用插件"""
        plugins: List[PluginInfo] = []

        # 1. 从目录发现
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                plugins.extend(self._discover_from_directory(plugin_dir))

        # 2. 从入口点发现（如果可用）
        try:
            plugins.extend(self._discover_from_entry_points())
        except ImportError:
            pass

        # 3. 从已安装包发现
        plugins.extend(self._discover_from_packages())

        # 去重并存储
        seen = set()
        for plugin in plugins:
            if plugin.name not in seen:
                seen.add(plugin.name)
                self._plugins[plugin.name] = plugin

        return list(self._plugins.values())

    def load_plugin(self, plugin_name: str) -> List[Tool]:
        """加载指定插件"""
        if plugin_name not in self._plugins:
            raise ValueError(f"插件未找到: {plugin_name}")

        plugin = self._plugins[plugin_name]
        if not plugin.enabled:
            return []

        # 检查依赖
        for dep in plugin.dependencies:
            if not self._check_dependency(dep):
                raise RuntimeError(f"插件 {plugin_name} 缺少依赖: {dep}")

        tools: List[Tool] = []
        entry_point = plugin.entry_point

        # 解析入口点
        if ":" in entry_point:
            module_path, attr_name = entry_point.split(":", 1)
        else:
            module_path = entry_point
            attr_name = None

        try:
            module = importlib.import_module(module_path)

            if attr_name:
                # 加载特定属性
                obj = getattr(module, attr_name)
                if callable(obj):
                    if inspect.isclass(obj) and issubclass(obj, Tool):
                        tool = self._injector.inject(obj)
                        tools.append(tool)
                    elif inspect.isfunction(obj):
                        tool = FunctionTool(obj)
                        tools.append(tool)
            else:
                # 自动发现模块中的工具
                introspector = ModuleIntrospector()
                discovered = introspector.introspect_module(module)
                for disc in discovered:
                    if disc.tool_class:
                        tool = self._injector.inject(disc.tool_class)
                        tools.append(tool)

        except Exception as e:
            warnings.warn(f"加载插件 {plugin_name} 失败: {e}")

        # 存储加载的工具
        for tool in tools:
            self._loaded_tools[tool.name] = tool

        return tools

    def get_loaded_tools(self) -> Dict[str, Tool]:
        """获取所有已加载的工具"""
        return dict(self._loaded_tools)

    def set_injector(self, injector: DependencyInjector) -> None:
        """设置依赖注入器"""
        self._injector = injector

    def _discover_from_directory(self, directory: Path) -> List[PluginInfo]:
        """从目录发现插件"""
        plugins: List[PluginInfo] = []

        for config_file in self.PLUGIN_CONFIG_FILES:
            config_path = directory / config_file
            if config_path.exists():
                plugin = self._parse_plugin_config(config_path)
                if plugin:
                    plugins.append(plugin)
                break

        # 扫描子目录
        for subdir in directory.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                plugins.extend(self._discover_from_directory(subdir))

        return plugins

    def _parse_plugin_config(self, config_path: Path) -> Optional[PluginInfo]:
        """解析插件配置"""
        try:
            if config_path.name == "plugin.json":
                data = json.loads(config_path.read_text())
                return PluginInfo(
                    name=data.get("name", config_path.parent.name),
                    version=data.get("version", "0.0.1"),
                    entry_point=data.get("entry_point", ""),
                    dependencies=data.get("dependencies", []),
                    config=data.get("config", {}),
                )

            elif config_path.name == "pyproject.toml":
                # 简化解析
                content = config_path.read_text()
                name_match = re.search(r'name\s*=\s*"([^"]+)"', content)
                version_match = re.search(r'version\s*=\s*"([^"]+)"', content)

                if name_match:
                    return PluginInfo(
                        name=name_match.group(1),
                        version=version_match.group(1) if version_match else "0.0.1",
                        entry_point=f"{name_match.group(1)}:tools",
                    )

        except Exception as e:
            warnings.warn(f"解析插件配置失败 {config_path}: {e}")

        return None

    def _discover_from_entry_points(self) -> List[PluginInfo]:
        """从入口点发现插件"""
        plugins: List[PluginInfo] = []

        try:
            import sys

            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points
            else:
                from importlib_metadata import entry_points

            eps = entry_points()
            if hasattr(eps, "select"):
                group_eps = eps.select(group=self.entry_point_group)
            else:
                group_eps = eps.get(self.entry_point_group, [])

            for ep in group_eps:
                plugins.append(
                    PluginInfo(
                        name=ep.name,
                        version="unknown",
                        entry_point=f"{ep.value}",
                    )
                )

        except ImportError:
            pass

        return plugins

    def _discover_from_packages(self) -> List[PluginInfo]:
        """从已安装包发现插件"""
        plugins: List[PluginInfo] = []

        try:
            for module_info in pkgutil.iter_modules():
                if module_info.name.startswith("agi_"):
                    plugins.append(
                        PluginInfo(
                            name=module_info.name,
                            version="unknown",
                            entry_point=module_info.name,
                        )
                    )
        except Exception:
            pass

        return plugins

    def _check_dependency(self, dependency: str) -> bool:
        """检查依赖是否满足"""
        try:
            importlib.import_module(dependency)
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# ToolDiscovery - 主发现类
# ---------------------------------------------------------------------------
class ToolDiscovery:
    """工具发现主类

    整合文件系统扫描、模块内省、插件发现和懒加载功能。
    """

    def __init__(
        self,
        scan_paths: Optional[List[Union[str, Path]]] = None,
        plugin_dirs: Optional[List[Union[str, Path]]] = None,
        enable_hot_reload: bool = False,
    ):
        self.scan_paths = scan_paths or [Path.cwd() / "tools"]
        self.scanner = FilesystemScanner(self.scan_paths)
        self.introspector = ModuleIntrospector()
        self.plugin_discovery = PluginDiscovery(plugin_dirs)
        self.injector = DependencyInjector()

        # 懒加载缓存
        self._lazy_tools: Dict[str, LazyLoader[Tool]] = {}
        self._discovered: Dict[str, DiscoveredTool] = {}

        # 热重载
        self._hot_reloader: Optional[HotReloader] = None
        if enable_hot_reload:
            self._hot_reloader = HotReloader(self.scanner, self.introspector)
            self._hot_reloader.add_callback(self._on_file_change)

    def discover(self) -> List[DiscoveredTool]:
        """发现所有可用工具"""
        discovered: List[DiscoveredTool] = []

        # 1. 扫描文件系统
        for file_path in self.scanner.scan():
            tools = self.introspector.introspect_file(file_path)
            for tool_info in tools:
                tool_info.checksum = self.scanner.compute_checksum(file_path)
                discovered.append(tool_info)
                self._discovered[tool_info.name] = tool_info

        # 2. 发现插件
        self.plugin_discovery.discover()

        return discovered

    def load_tool(self, name: str) -> Optional[Tool]:
        """加载指定工具"""
        if name in self._lazy_tools:
            try:
                return self._lazy_tools[name].get()
            except RuntimeError:
                return None

        if name not in self._discovered:
            self.discover()

        if name not in self._discovered:
            return None

        tool_info = self._discovered[name]
        return self._load_discovered_tool(tool_info)

    def load_all_tools(self) -> Dict[str, Tool]:
        """加载所有发现的工具"""
        self.discover()
        tools: Dict[str, Tool] = {}

        for name in self._discovered:
            tool = self.load_tool(name)
            if tool:
                tools[name] = tool

        return tools

    def register_lazy(
        self,
        name: str,
        factory: Callable[[], Tool],
        preload: bool = False,
    ) -> None:
        """注册懒加载工具"""
        self._lazy_tools[name] = LazyLoader(factory, name, preload)

    def start_hot_reload(self) -> None:
        """启动热重载"""
        if self._hot_reloader:
            self._hot_reloader.start()

    def stop_hot_reload(self) -> None:
        """停止热重载"""
        if self._hot_reloader:
            self._hot_reloader.stop()

    def get_discovered_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有发现工具的元信息"""
        return {
            name: {
                "name": info.name,
                "module_path": info.module_path,
                "source_file": info.source_file,
                "metadata": info.metadata,
                "checksum": info.checksum,
            }
            for name, info in self._discovered.items()
        }

    def _load_discovered_tool(self, tool_info: DiscoveredTool) -> Optional[Tool]:
        """加载发现的工具"""
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(
                tool_info.module_path, tool_info.source_file
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[tool_info.module_path] = module
            spec.loader.exec_module(module)

            # 获取工具类
            tool_class = getattr(module, tool_info.name)
            if not inspect.isclass(tool_class) or not issubclass(tool_class, Tool):
                return None

            # 依赖注入并实例化
            instance = self.injector.inject(tool_class)
            return instance

        except Exception as e:
            warnings.warn(f"加载工具 {tool_info.name} 失败: {e}")
            return None

    def _on_file_change(self, file_path: str, event_type: str) -> None:
        """文件变化回调"""
        if event_type in ("modified", "added"):
            # 重新发现该文件中的工具
            tools = self.introspector.introspect_file(file_path)
            for tool_info in tools:
                tool_info.checksum = self.scanner.compute_checksum(file_path)
                self._discovered[tool_info.name] = tool_info

                # 如果已懒加载，标记为需要重新加载
                if tool_info.name in self._lazy_tools:
                    self._lazy_tools[tool_info.name].reload()


__all__ = [
    "ToolDiscovery",
    "FilesystemScanner",
    "ModuleIntrospector",
    "PluginDiscovery",
    "LazyLoader",
    "HotReloader",
    "DependencyInjector",
    "DiscoveredTool",
    "PluginInfo",
    "ToolSource",
    "ToolFactory",
]
