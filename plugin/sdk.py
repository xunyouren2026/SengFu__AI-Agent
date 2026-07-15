#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin SDK 模块

本模块提供插件开发的基础 SDK，包括插件基类、生命周期钩子、
配置模式和与核心框架交互的 API。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Protocol
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)


class PluginError(Exception):
    """插件错误基类"""
    
    def __init__(self, message: str, plugin_id: Optional[str] = None, 
                 error_code: Optional[str] = None):
        self.message = message
        self.plugin_id = plugin_id
        self.error_code = error_code
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.plugin_id:
            parts.append(f"插件: {self.plugin_id}")
        if self.error_code:
            parts.append(f"错误码: {self.error_code}")
        return " | ".join(parts)


class PluginLoadError(PluginError):
    """插件加载错误"""
    pass


class PluginConfigError(PluginError):
    """插件配置错误"""
    pass


class PluginRuntimeError(PluginError):
    """插件运行时错误"""
    pass


class PluginAPIError(PluginError):
    """插件 API 错误"""
    pass


class PluginState(Enum):
    """插件状态枚举"""
    UNLOADED = auto()     # 未加载
    LOADING = auto()      # 加载中
    LOADED = auto()       # 已加载
    ENABLING = auto()     # 启用中
    ENABLED = auto()      # 已启用
    DISABLING = auto()    # 禁用中
    DISABLED = auto()     # 已禁用
    ERROR = auto()        # 错误状态


class PluginPriority(Enum):
    """插件优先级枚举"""
    CRITICAL = 0      # 关键
    HIGH = 1          # 高
    NORMAL = 2        # 正常
    LOW = 3           # 低
    BACKGROUND = 4    # 后台


@dataclass
class PluginMetadata:
    """
    插件元数据
    
    属性:
        id: 插件唯一标识
        name: 插件名称
        version: 版本号
        description: 插件描述
        author: 作者
        license: 许可证
        homepage: 主页 URL
        repository: 代码仓库 URL
        tags: 标签列表
        category: 分类
        min_platform_version: 最低平台版本
        max_platform_version: 最高平台版本
    """
    id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    min_platform_version: str = "1.0.0"
    max_platform_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginMetadata:
        """从字典创建"""
        return cls(**data)


@dataclass
class PluginDependency:
    """
    插件依赖
    
    属性:
        plugin_id: 依赖的插件 ID
        version_constraint: 版本约束
        optional: 是否为可选依赖
    """
    plugin_id: str
    version_constraint: str = "*"
    optional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginDependency:
        """从字典创建"""
        return cls(**data)


@dataclass
class PluginManifest:
    """
    插件清单
    
    属性:
        metadata: 插件元数据
        dependencies: 依赖列表
        permissions: 所需权限列表
        entry_point: 入口点
        config_schema: 配置模式
        hooks: 钩子注册信息
        assets: 资源文件列表
    """
    metadata: PluginMetadata
    dependencies: List[PluginDependency] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    entry_point: str = "plugin.py"
    config_schema: Dict[str, Any] = field(default_factory=dict)
    hooks: Dict[str, List[str]] = field(default_factory=dict)
    assets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metadata': self.metadata.to_dict(),
            'dependencies': [d.to_dict() for d in self.dependencies],
            'permissions': self.permissions,
            'entry_point': self.entry_point,
            'config_schema': self.config_schema,
            'hooks': self.hooks,
            'assets': self.assets,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PluginManifest:
        """从字典创建"""
        metadata = PluginMetadata.from_dict(data['metadata'])
        dependencies = [PluginDependency.from_dict(d) for d in data.get('dependencies', [])]
        
        return cls(
            metadata=metadata,
            dependencies=dependencies,
            permissions=data.get('permissions', []),
            entry_point=data.get('entry_point', 'plugin.py'),
            config_schema=data.get('config_schema', {}),
            hooks=data.get('hooks', {}),
            assets=data.get('assets', []),
        )
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> PluginManifest:
        """从文件加载"""
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls.from_dict(data)


@dataclass
class PluginConfig:
    """
    插件配置
    
    属性:
        values: 配置值字典
        schema: 配置模式
        auto_save: 是否自动保存
        config_path: 配置文件路径
    """
    values: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    auto_save: bool = True
    config_path: Optional[Path] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.values.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.values[key] = value
        if self.auto_save and self.config_path:
            self.save()
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        验证配置
        
        返回:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        for key, schema_def in self.schema.items():
            if schema_def.get('required', False) and key not in self.values:
                errors.append(f"缺少必需配置项: {key}")
            
            if key in self.values:
                value = self.values[key]
                value_type = schema_def.get('type')
                
                if value_type == 'string' and not isinstance(value, str):
                    errors.append(f"{key} 必须是字符串")
                elif value_type == 'integer' and not isinstance(value, int):
                    errors.append(f"{key} 必须是整数")
                elif value_type == 'boolean' and not isinstance(value, bool):
                    errors.append(f"{key} 必须是布尔值")
                
                # 检查枚举值
                if 'enum' in schema_def and value not in schema_def['enum']:
                    errors.append(f"{key} 必须是以下之一: {schema_def['enum']}")
        
        return len(errors) == 0, errors
    
    def save(self, path: Optional[Path] = None) -> None:
        """保存配置"""
        path = path or self.config_path
        if path:
            path.write_text(json.dumps(self.values, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def load(self, path: Optional[Path] = None) -> None:
        """加载配置"""
        path = path or self.config_path
        if path and path.exists():
            self.values = json.loads(path.read_text(encoding='utf-8'))


class PluginContext:
    """
    插件上下文
    
    提供插件与核心框架交互的上下文环境。
    """
    
    def __init__(self, plugin_id: str, config: PluginConfig,
                 data_dir: Path, temp_dir: Path):
        """
        初始化上下文
        
        参数:
            plugin_id: 插件 ID
            config: 插件配置
            data_dir: 数据目录
            temp_dir: 临时目录
        """
        self.plugin_id = plugin_id
        self.config = config
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self._api_client: Optional[PluginAPIClient] = None
    
    def set_api_client(self, client: PluginAPIClient) -> None:
        """设置 API 客户端"""
        self._api_client = client
    
    def call_api(self, method: str, **kwargs) -> Any:
        """
        调用核心框架 API
        
        参数:
            method: API 方法名
            **kwargs: 方法参数
            
        返回:
            API 调用结果
        """
        if self._api_client is None:
            raise PluginAPIError("API 客户端未设置")
        return self._api_client.call(method, **kwargs)
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """
        记录日志
        
        参数:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外信息
        """
        logger.log(getattr(logging, level.upper(), logging.INFO), 
                   f"[{self.plugin_id}] {message}", extra=kwargs)
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发送事件
        
        参数:
            event_type: 事件类型
            data: 事件数据
        """
        if self._api_client:
            self._api_client.emit_event(event_type, data)


class PluginAPIClient:
    """
    插件 API 客户端
    
    提供插件与核心框架交互的 API。
    """
    
    def __init__(self, plugin_id: str, core_api: Any):
        """
        初始化 API 客户端
        
        参数:
            plugin_id: 插件 ID
            core_api: 核心框架 API 对象
        """
        self.plugin_id = plugin_id
        self._core_api = core_api
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def call(self, method: str, **kwargs) -> Any:
        """
        调用核心 API
        
        参数:
            method: 方法名
            **kwargs: 参数
            
        返回:
            调用结果
        """
        if hasattr(self._core_api, method):
            return getattr(self._core_api, method)(**kwargs)
        raise PluginAPIError(f"API 方法不存在: {method}")
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发送事件到核心框架
        
        参数:
            event_type: 事件类型
            data: 事件数据
        """
        if hasattr(self._core_api, 'handle_plugin_event'):
            self._core_api.handle_plugin_event(self.plugin_id, event_type, data)
    
    def on_event(self, event_type: str, handler: Callable) -> None:
        """
        注册事件处理器
        
        参数:
            event_type: 事件类型
            handler: 处理函数
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        处理来自核心框架的事件
        
        参数:
            event_type: 事件类型
            data: 事件数据
        """
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"事件处理错误: {e}")


T = TypeVar('T', bound='BasePlugin')


class BasePlugin(ABC):
    """
    插件基类
    
    所有插件必须继承此类并实现相关方法。
    """
    
    # 类属性，子类可以覆盖
    metadata: Optional[PluginMetadata] = None
    priority: PluginPriority = PluginPriority.NORMAL
    
    def __init__(self):
        """初始化插件"""
        self._state = PluginState.UNLOADED
        self._context: Optional[PluginContext] = None
        self._manifest: Optional[PluginManifest] = None
        self._config: Optional[PluginConfig] = None
    
    @property
    def state(self) -> PluginState:
        """获取插件状态"""
        return self._state
    
    @property
    def context(self) -> Optional[PluginContext]:
        """获取插件上下文"""
        return self._context
    
    @property
    def config(self) -> Optional[PluginConfig]:
        """获取插件配置"""
        return self._config
    
    @property
    def is_enabled(self) -> bool:
        """检查插件是否已启用"""
        return self._state == PluginState.ENABLED
    
    # ========== 生命周期钩子 ==========
    
    def on_load(self, context: PluginContext, manifest: PluginManifest) -> bool:
        """
        插件加载时调用
        
        参数:
            context: 插件上下文
            manifest: 插件清单
            
        返回:
            是否加载成功
        """
        self._state = PluginState.LOADING
        try:
            self._context = context
            self._manifest = manifest
            self._config = context.config
            
            result = self.load()
            
            if result:
                self._state = PluginState.LOADED
                logger.info(f"插件 {self.get_id()} 加载成功")
            else:
                self._state = PluginState.ERROR
                logger.error(f"插件 {self.get_id()} 加载失败")
            
            return result
            
        except Exception as e:
            self._state = PluginState.ERROR
            logger.error(f"插件 {self.get_id()} 加载异常: {e}")
            return False
    
    def on_enable(self) -> bool:
        """
        插件启用时调用
        
        返回:
            是否启用成功
        """
        self._state = PluginState.ENABLING
        try:
            result = self.enable()
            
            if result:
                self._state = PluginState.ENABLED
                logger.info(f"插件 {self.get_id()} 启用成功")
            else:
                self._state = PluginState.ERROR
                logger.error(f"插件 {self.get_id()} 启用失败")
            
            return result
            
        except Exception as e:
            self._state = PluginState.ERROR
            logger.error(f"插件 {self.get_id()} 启用异常: {e}")
            return False
    
    def on_disable(self) -> bool:
        """
        插件禁用时调用
        
        返回:
            是否禁用成功
        """
        self._state = PluginState.DISABLING
        try:
            result = self.disable()
            
            self._state = PluginState.DISABLED
            logger.info(f"插件 {self.get_id()} 已禁用")
            
            return result
            
        except Exception as e:
            self._state = PluginState.ERROR
            logger.error(f"插件 {self.get_id()} 禁用异常: {e}")
            return False
    
    def on_unload(self) -> None:
        """插件卸载时调用"""
        try:
            self.unload()
            self._state = PluginState.UNLOADED
            logger.info(f"插件 {self.get_id()} 已卸载")
        except Exception as e:
            logger.error(f"插件 {self.get_id()} 卸载异常: {e}")
    
    # ========== 子类可覆盖的方法 ==========
    
    def load(self) -> bool:
        """
        加载插件
        
        子类可以覆盖此方法以实现自定义加载逻辑。
        
        返回:
            是否加载成功
        """
        return True
    
    def enable(self) -> bool:
        """
        启用插件
        
        子类可以覆盖此方法以实现自定义启用逻辑。
        
        返回:
            是否启用成功
        """
        return True
    
    def disable(self) -> bool:
        """
        禁用插件
        
        子类可以覆盖此方法以实现自定义禁用逻辑。
        
        返回:
            是否禁用成功
        """
        return True
    
    def unload(self) -> None:
        """
        卸载插件
        
        子类可以覆盖此方法以实现自定义卸载逻辑。
        """
        pass
    
    def get_id(self) -> str:
        """
        获取插件 ID
        
        返回:
            插件 ID
        """
        if self.metadata:
            return self.metadata.id
        return self.__class__.__name__
    
    def get_name(self) -> str:
        """
        获取插件名称
        
        返回:
            插件名称
        """
        if self.metadata:
            return self.metadata.name
        return self.__class__.__name__


class PluginHook:
    """
    插件钩子装饰器
    
    用于注册插件钩子函数。
    """
    
    _hooks: Dict[str, List[tuple]] = {}
    
    def __init__(self, hook_point: str, priority: int = 100):
        """
        初始化钩子装饰器
        
        参数:
            hook_point: 钩子点名称
            priority: 优先级（数字越小优先级越高）
        """
        self.hook_point = hook_point
        self.priority = priority
    
    def __call__(self, func: Callable) -> Callable:
        """
        装饰函数
        
        参数:
            func: 要装饰的函数
            
        返回:
            装饰后的函数
        """
        if self.hook_point not in self._hooks:
            self._hooks[self.hook_point] = []
        
        self._hooks[self.hook_point].append((self.priority, func))
        self._hooks[self.hook_point].sort(key=lambda x: x[0])
        
        return func
    
    @classmethod
    def get_hooks(cls, hook_point: str) -> List[Callable]:
        """
        获取钩子点的所有钩子函数
        
        参数:
            hook_point: 钩子点名称
            
        返回:
            钩子函数列表
        """
        hooks = cls._hooks.get(hook_point, [])
        return [func for _, func in hooks]
    
    @classmethod
    def execute_hooks(cls, hook_point: str, *args, **kwargs) -> List[Any]:
        """
        执行钩子点的所有钩子函数
        
        参数:
            hook_point: 钩子点名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            所有钩子函数的返回值列表
        """
        results = []
        for func in cls.get_hooks(hook_point):
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"钩子执行错误: {e}")
        return results


class PluginExtensionPoint(ABC):
    """
    插件扩展点基类
    
    定义插件可以扩展的功能点。
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """获取扩展点名称"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """获取扩展点描述"""
        pass


class ToolProvider(PluginExtensionPoint):
    """工具提供者扩展点"""
    
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取提供的工具列表"""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """执行工具"""
        pass


class CommandProvider(PluginExtensionPoint):
    """命令提供者扩展点"""
    
    @abstractmethod
    def get_commands(self) -> List[Dict[str, Any]]:
        """获取提供的命令列表"""
        pass
    
    @abstractmethod
    def execute_command(self, command_name: str, args: List[str]) -> Any:
        """执行命令"""
        pass


class EventHandler(PluginExtensionPoint):
    """事件处理器扩展点"""
    
    @abstractmethod
    def get_handled_events(self) -> List[str]:
        """获取处理的事件类型列表"""
        pass
    
    @abstractmethod
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """处理事件"""
        pass


# 便捷函数
def create_plugin_manifest(
    plugin_id: str,
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "",
    **kwargs
) -> PluginManifest:
    """
    创建插件清单
    
    参数:
        plugin_id: 插件 ID
        name: 插件名称
        version: 版本号
        description: 描述
        author: 作者
        **kwargs: 其他参数
        
    返回:
        插件清单
    """
    metadata = PluginMetadata(
        id=plugin_id,
        name=name,
        version=version,
        description=description,
        author=author,
        **{k: v for k, v in kwargs.items() if k in ['license', 'homepage', 'repository', 'tags', 'category']}
    )
    
    return PluginManifest(
        metadata=metadata,
        dependencies=kwargs.get('dependencies', []),
        permissions=kwargs.get('permissions', []),
        entry_point=kwargs.get('entry_point', 'plugin.py'),
    )


def validate_plugin_config(config: Dict[str, Any], 
                           schema: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    验证插件配置
    
    参数:
        config: 配置值
        schema: 配置模式
        
    返回:
        (是否有效, 错误列表)
    """
    plugin_config = PluginConfig(values=config, schema=schema)
    return plugin_config.validate()


# 单元测试存根
class TestPluginSDK:
    """Plugin SDK 单元测试"""
    
    def test_plugin_metadata(self) -> None:
        """测试插件元数据"""
        metadata = PluginMetadata(
            id="test-plugin",
            name="Test Plugin",
            version="1.0.0",
        )
        
        assert metadata.id == "test-plugin"
        assert metadata.name == "Test Plugin"
        
        data = metadata.to_dict()
        restored = PluginMetadata.from_dict(data)
        assert restored.id == metadata.id
    
    def test_plugin_config(self) -> None:
        """测试插件配置"""
        schema = {
            'api_key': {'type': 'string', 'required': True},
            'timeout': {'type': 'integer', 'required': False},
        }
        
        config = PluginConfig(
            values={'api_key': 'secret123'},
            schema=schema,
        )
        
        valid, errors = config.validate()
        assert valid
        
        config.values.pop('api_key')
        valid, errors = config.validate()
        assert not valid
    
    def test_plugin_lifecycle(self) -> None:
        """测试插件生命周期"""
        
        class TestPlugin(BasePlugin):
            metadata = PluginMetadata(id="test", name="Test")
            
            def load(self) -> bool:
                return True
            
            def enable(self) -> bool:
                return True
        
        plugin = TestPlugin()
        assert plugin.state == PluginState.UNLOADED
        
        # 模拟加载
        context = PluginContext(
            plugin_id="test",
            config=PluginConfig(),
            data_dir=Path("/tmp"),
            temp_dir=Path("/tmp"),
        )
        manifest = PluginManifest(metadata=PluginMetadata(id="test", name="Test"))
        
        result = plugin.on_load(context, manifest)
        assert result
        assert plugin.state == PluginState.LOADED
        
        result = plugin.on_enable()
        assert result
        assert plugin.state == PluginState.ENABLED
    
    def test_plugin_hooks(self) -> None:
        """测试插件钩子"""
        
        results = []
        
        @PluginHook("test_hook", priority=10)
        def hook1():
            results.append(1)
        
        @PluginHook("test_hook", priority=5)
        def hook2():
            results.append(2)
        
        PluginHook.execute_hooks("test_hook")
        
        # 优先级 5 的先执行
        assert results == [2, 1]
