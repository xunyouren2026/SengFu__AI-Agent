"""
渠道-模型映射器 (Channel-Model Mapper)

该模块负责将不同的业务渠道映射到合适的AI模型，支持：
- 渠道专属模型配置
- 渠道特性适配
- 默认模型绑定
- 模型优先级管理

典型使用场景：
- 客服渠道：使用对话能力强的模型
- 知识库问答：使用问答能力强的模型
- 代码助手：使用代码能力强的模型
- 营销文案：使用创意能力强的模型

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, FrozenSet
)
from collections import defaultdict

# 配置日志
logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """
    渠道类型枚举
    
    定义支持的业务渠道类型。
    """
    # 对话类渠道
    WEB_CHAT = auto()           # 网页客服
    APP_CHAT = auto()            # APP内聊天
    WECHAT = auto()              # 微信公众号/小程序
    WIDGET = auto()              # 嵌入式组件
    
    # 语音类渠道
    VOICE_ASSISTANT = auto()     # 语音助手
    CALL_CENTER = auto()         # 呼叫中心
    
    # API类渠道
    API_DIRECT = auto()          # 直接API调用
    WEBHOOK = auto()             # Webhook回调
    
    # 专业类渠道
    KNOWLEDGE_BASE = auto()      # 知识库问答
    DOCUMENT_ANALYSIS = auto()   # 文档分析
    CODE_ASSISTANT = auto()      # 代码助手
    
    # 内容类渠道
    CONTENT_GENERATION = auto()  # 内容生成
    SOCIAL_MEDIA = auto()         # 社交媒体
    EMAIL = auto()                # 邮件处理
    
    # 分析类渠道
    DATA_ANALYSIS = auto()       # 数据分析
    REPORT_GENERATION = auto()   # 报告生成
    
    # 自定义渠道
    CUSTOM = auto()              # 自定义渠道


class ChannelCapability(Enum):
    """
    渠道能力需求枚举
    
    定义渠道可能需要的能力。
    """
    STREAMING = "streaming"                    # 支持流式输出
    TOOL_USE = "tool_use"                     # 支持工具调用
    VISION = "vision"                         # 支持视觉理解
    FUNCTION_CALLING = "function_calling"     # 支持函数调用
    LONG_CONTEXT = "long_context"             # 需要长上下文
    LOW_LATENCY = "low_latency"               # 需要低延迟
    MULTI_TURN = "multi_turn"                # 支持多轮对话
    SAFETY_SENSITIVE = "safety_sensitive"    # 安全敏感
    COST_SENSITIVE = "cost_sensitive"         # 成本敏感


@dataclass(frozen=True)
class ChannelConfig:
    """
    渠道配置
    
    定义一个渠道的完整配置信息。
    
    Attributes:
        channel_id: 渠道唯一标识
        channel_type: 渠道类型
        channel_name: 渠道显示名称
        required_capabilities: 必须具备的能力列表
        preferred_capabilities: 优先考虑的能力列表
        max_latency_ms: 最大可接受延迟
        max_cost_per_request: 单次请求最大成本
        priority: 渠道优先级 (数值越高优先级越高)
        metadata: 其他配置元数据
        tags: 渠道标签
    """
    channel_id: str
    channel_type: ChannelType
    channel_name: str
    required_capabilities: FrozenSet[ChannelCapability] = field(default_factory=frozenset)
    preferred_capabilities: FrozenSet[ChannelCapability] = field(default_factory=frozenset)
    max_latency_ms: float = 10000.0
    max_cost_per_request: float = 1.0
    priority: int = 50
    metadata: FrozenSet[Tuple[str, Any]] = field(default_factory=frozenset)
    tags: FrozenSet[str] = field(default_factory=frozenset)
    
    def supports_capability(self, capability: ChannelCapability) -> bool:
        """检查渠道是否需要指定能力"""
        return capability in self.required_capabilities
    
    def prefers_capability(self, capability: ChannelCapability) -> bool:
        """检查渠道是否优先考虑指定能力"""
        return capability in self.preferred_capabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "channel_id": self.channel_id,
            "channel_type": self.channel_type.name,
            "channel_name": self.channel_name,
            "required_capabilities": [c.value for c in self.required_capabilities],
            "preferred_capabilities": [c.value for c in self.preferred_capabilities],
            "max_latency_ms": self.max_latency_ms,
            "max_cost_per_request": self.max_cost_per_request,
            "priority": self.priority,
            "tags": list(self.tags),
        }


@dataclass
class ModelBinding:
    """
    模型绑定配置
    
    定义某个渠道可用的模型绑定。
    
    Attributes:
        model_id: 模型ID
        priority: 在该渠道中的优先级
        is_default: 是否为默认模型
        max_concurrent: 最大并发数
        rate_limit: 每分钟请求限制
        cost_multiplier: 成本倍数 (用于特定渠道加价)
    """
    model_id: str
    priority: int = 0
    is_default: bool = False
    max_concurrent: int = 10
    rate_limit: int = 100  # 每分钟
    cost_multiplier: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "priority": self.priority,
            "is_default": self.is_default,
            "max_concurrent": self.max_concurrent,
            "rate_limit": self.rate_limit,
            "cost_multiplier": self.cost_multiplier,
        }


class ChannelMapper:
    """
    渠道-模型映射器
    
    根据渠道配置自动选择合适的模型。
    
    Features:
        - 渠道专属模型配置
        - 渠道特性适配
        - 默认模型绑定
        - 优先级路由
        - 流量控制
    
    Example:
        ```python
        # 创建映射器
        mapper = ChannelMapper()
        
        # 注册渠道
        mapper.register_channel(ChannelConfig(
            channel_id="web_chat",
            channel_type=ChannelType.WEB_CHAT,
            channel_name="网页客服",
            required_capabilities={ChannelCapability.STREAMING},
            priority=80
        ))
        
        # 绑定模型
        mapper.bind_model(
            "web_chat",
            ModelBinding(model_id="gpt-3.5-turbo", priority=1, is_default=True)
        )
        mapper.bind_model(
            "web_chat", 
            ModelBinding(model_id="gpt-4-turbo", priority=2)
        )
        
        # 获取渠道的推荐模型
        models = mapper.get_models_for_channel("web_chat")
        default_model = mapper.get_default_model("web_chat")
        ```
    """
    
    def __init__(self):
        """初始化渠道-模型映射器"""
        self._channels: Dict[str, ChannelConfig] = {}
        self._channel_models: Dict[str, List[ModelBinding]] = defaultdict(list)
        self._default_models: Dict[str, str] = {}
        self._model_channels: Dict[str, Set[str]] = defaultdict(set)
        self._rate_counters: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # 注册默认渠道
        self._register_default_channels()
    
    def _register_default_channels(self) -> None:
        """注册默认渠道配置"""
        default_channels = [
            ChannelConfig(
                channel_id="web_chat",
                channel_type=ChannelType.WEB_CHAT,
                channel_name="网页客服",
                required_capabilities=frozenset({ChannelCapability.STREAMING}),
                preferred_capabilities=frozenset({
                    ChannelCapability.LOW_LATENCY,
                    ChannelCapability.SAFETY_SENSITIVE
                }),
                max_latency_ms=5000,
                priority=80
            ),
            ChannelConfig(
                channel_id="wechat",
                channel_type=ChannelType.WECHAT,
                channel_name="微信渠道",
                required_capabilities=frozenset({ChannelCapability.MULTI_TURN}),
                preferred_capabilities=frozenset({ChannelCapability.LOW_LATENCY}),
                max_latency_ms=8000,
                priority=70
            ),
            ChannelConfig(
                channel_id="voice_assistant",
                channel_type=ChannelType.VOICE_ASSISTANT,
                channel_name="语音助手",
                required_capabilities=frozenset({ChannelCapability.LOW_LATENCY}),
                preferred_capabilities=frozenset({ChannelCapability.STREAMING}),
                max_latency_ms=2000,
                priority=90
            ),
            ChannelConfig(
                channel_id="api_direct",
                channel_type=ChannelType.API_DIRECT,
                channel_name="API直调",
                required_capabilities=frozenset(),
                preferred_capabilities=frozenset(),
                max_latency_ms=15000,
                priority=50
            ),
            ChannelConfig(
                channel_id="knowledge_base",
                channel_type=ChannelType.KNOWLEDGE_BASE,
                channel_name="知识库问答",
                required_capabilities=frozenset({ChannelCapability.LONG_CONTEXT}),
                preferred_capabilities=frozenset({ChannelCapability.TOOL_USE}),
                max_latency_ms=10000,
                priority=60
            ),
            ChannelConfig(
                channel_id="code_assistant",
                channel_type=ChannelType.CODE_ASSISTANT,
                channel_name="代码助手",
                required_capabilities=frozenset(),
                preferred_capabilities=frozenset({ChannelCapability.TOOL_USE}),
                max_latency_ms=8000,
                priority=75,
                tags=frozenset({"developer", "coding"})
            ),
            ChannelConfig(
                channel_id="content_generation",
                channel_type=ChannelType.CONTENT_GENERATION,
                channel_name="内容生成",
                required_capabilities=frozenset(),
                preferred_capabilities=frozenset({ChannelCapability.CREATIVITY}),
                max_latency_ms=15000,
                priority=60
            ),
            ChannelConfig(
                channel_id="social_media",
                channel_type=ChannelType.SOCIAL_MEDIA,
                channel_name="社交媒体",
                required_capabilities=frozenset(),
                preferred_capabilities=frozenset({ChannelCapability.CREATIVITY}),
                max_latency_ms=10000,
                priority=55,
                tags=frozenset({"marketing", "social"})
            ),
        ]
        
        for channel in default_channels:
            self.register_channel(channel)
    
    def register_channel(self, config: ChannelConfig) -> None:
        """
        注册一个渠道配置。
        
        Args:
            config: 渠道配置
        """
        with self._lock:
            self._channels[config.channel_id] = config
            logger.info(f"Registered channel: {config.channel_id} ({config.channel_name})")
    
    def unregister_channel(self, channel_id: str) -> bool:
        """
        注销一个渠道。
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if channel_id in self._channels:
                # 清理关联的模型绑定
                bound_models = self._channel_models.pop(channel_id, [])
                for binding in bound_models:
                    self._model_channels[binding.model_id].discard(channel_id)
                
                del self._channels[channel_id]
                self._default_models.pop(channel_id, None)
                logger.info(f"Unregistered channel: {channel_id}")
                return True
            return False
    
    def get_channel(self, channel_id: str) -> Optional[ChannelConfig]:
        """获取渠道配置"""
        return self._channels.get(channel_id)
    
    def bind_model(
        self,
        channel_id: str,
        binding: ModelBinding
    ) -> bool:
        """
        为渠道绑定模型。
        
        Args:
            channel_id: 渠道ID
            binding: 模型绑定配置
            
        Returns:
            是否绑定成功
        """
        with self._lock:
            if channel_id not in self._channels:
                logger.warning(f"Channel {channel_id} not found")
                return False
            
            # 检查是否已绑定该模型
            for existing in self._channel_models[channel_id]:
                if existing.model_id == binding.model_id:
                    # 更新现有绑定
                    existing.priority = binding.priority
                    existing.is_default = binding.is_default
                    existing.max_concurrent = binding.max_concurrent
                    existing.rate_limit = binding.rate_limit
                    existing.cost_multiplier = binding.cost_multiplier
                    return True
            
            # 添加新绑定
            self._channel_models[channel_id].append(binding)
            self._model_channels[binding.model_id].add(channel_id)
            
            # 如果是默认模型
            if binding.is_default:
                self._default_models[channel_id] = binding.model_id
            
            # 按优先级排序
            self._channel_models[channel_id].sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Bound model {binding.model_id} to channel {channel_id}")
            return True
    
    def unbind_model(self, channel_id: str, model_id: str) -> bool:
        """
        解绑模型。
        
        Args:
            channel_id: 渠道ID
            model_id: 模型ID
            
        Returns:
            是否解绑成功
        """
        with self._lock:
            if channel_id not in self._channel_models:
                return False
            
            original_len = len(self._channel_models[channel_id])
            self._channel_models[channel_id] = [
                b for b in self._channel_models[channel_id]
                if b.model_id != model_id
            ]
            
            if len(self._channel_models[channel_id]) < original_len:
                self._model_channels[model_id].discard(channel_id)
                if self._default_models.get(channel_id) == model_id:
                    del self._default_models[channel_id]
                return True
            return False
    
    def get_models_for_channel(
        self,
        channel_id: str,
        available_only: bool = True
    ) -> List[ModelBinding]:
        """
        获取渠道可用的模型列表。
        
        Args:
            channel_id: 渠道ID
            available_only: 是否只返回可用的模型
            
        Returns:
            模型绑定列表 (按优先级排序)
        """
        with self._lock:
            if channel_id not in self._channel_models:
                return []
            
            models = self._channel_models[channel_id].copy()
            
            if available_only:
                # 过滤掉超出限流的模型
                now = time.time()
                models = [
                    m for m in models
                    if self._check_rate_limit(channel_id, m.model_id, m.rate_limit, now)
                ]
            
            return models
    
    def get_default_model(self, channel_id: str) -> Optional[str]:
        """
        获取渠道的默认模型。
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            默认模型ID，如果未设置则返回优先级最高的模型
        """
        with self._lock:
            # 优先返回显式设置的默认模型
            if channel_id in self._default_models:
                return self._default_models[channel_id]
            
            # 否则返回优先级最高的模型
            models = self.get_models_for_channel(channel_id)
            if models:
                # 找到标记为默认的模型
                for m in models:
                    if m.is_default:
                        return m.model_id
                # 否则返回第一个
                return models[0].model_id
            
            return None
    
    def get_channels_for_model(self, model_id: str) -> List[str]:
        """
        获取使用特定模型的渠道列表。
        
        Args:
            model_id: 模型ID
            
        Returns:
            渠道ID列表
        """
        return list(self._model_channels.get(model_id, set()))
    
    def get_all_channels(self) -> List[ChannelConfig]:
        """获取所有已注册的渠道"""
        return list(self._channels.values())
    
    def _check_rate_limit(
        self,
        channel_id: str,
        model_id: str,
        limit: int,
        now: float
    ) -> bool:
        """
        检查速率限制。
        
        Args:
            channel_id: 渠道ID
            model_id: 模型ID
            limit: 每分钟限制
            now: 当前时间戳
            
        Returns:
            是否在限制内
        """
        key = (channel_id, model_id)
        
        # 清理过期的计数器
        self._rate_counters[key] = [
            t for t in self._rate_counters[key]
            if now - t < 60  # 保留60秒内的请求
        ]
        
        # 检查是否超限
        return len(self._rate_counters[key]) < limit
    
    def record_request(self, channel_id: str, model_id: str) -> None:
        """
        记录一次请求，用于速率限制统计。
        
        Args:
            channel_id: 渠道ID
            model_id: 模型ID
        """
        with self._lock:
            key = (channel_id, model_id)
            self._rate_counters[key].append(time.time())
    
    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """
        获取渠道统计信息。
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            统计信息字典
        """
        with self._lock:
            channel = self._channels.get(channel_id)
            if not channel:
                return {}
            
            models = self._channel_models.get(channel_id, [])
            
            return {
                "channel_id": channel_id,
                "channel_type": channel.channel_type.name,
                "channel_name": channel.channel_name,
                "priority": channel.priority,
                "bound_models_count": len(models),
                "default_model": self._default_models.get(channel_id),
                "required_capabilities": [c.value for c in channel.required_capabilities],
            }
    
    def list_channel_models_detailed(
        self,
        channel_id: str
    ) -> List[Dict[str, Any]]:
        """
        列出渠道绑定的所有模型详情。
        
        Args:
            channel_id: 渠道ID
            
        Returns:
            模型详情列表
        """
        with self._lock:
            if channel_id not in self._channel_models:
                return []
            
            now = time.time()
            result = []
            
            for binding in self._channel_models[channel_id]:
                key = (channel_id, binding.model_id)
                
                # 计算最近1分钟的请求数
                recent_requests = len([
                    t for t in self._rate_counters[key]
                    if now - t < 60
                ])
                
                # 计算剩余配额
                remaining = max(0, binding.rate_limit - recent_requests)
                
                result.append({
                    **binding.to_dict(),
                    "recent_requests_per_minute": recent_requests,
                    "remaining_quota": remaining,
                    "is_available": recent_requests < binding.rate_limit,
                })
            
            return result
    
    def update_binding(
        self,
        channel_id: str,
        model_id: str,
        **kwargs
    ) -> bool:
        """
        更新模型绑定配置。
        
        Args:
            channel_id: 渠道ID
            model_id: 模型ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        with self._lock:
            for binding in self._channel_models.get(channel_id, []):
                if binding.model_id == model_id:
                    for key, value in kwargs.items():
                        if hasattr(binding, key):
                            setattr(binding, key, value)
                    return True
            return False
    
    def set_default_model(self, channel_id: str, model_id: str) -> bool:
        """
        设置渠道的默认模型。
        
        Args:
            channel_id: 渠道ID
            model_id: 模型ID
            
        Returns:
            是否设置成功
        """
        with self._lock:
            # 验证模型已绑定
            for binding in self._channel_models.get(channel_id, []):
                if binding.model_id == model_id:
                    # 清除其他默认标记
                    for b in self._channel_models[channel_id]:
                        b.is_default = False
                    
                    # 设置新的默认模型
                    binding.is_default = True
                    self._default_models[channel_id] = model_id
                    return True
            return False
    
    def find_channels_by_tags(self, tags: Set[str]) -> List[ChannelConfig]:
        """
        根据标签查找渠道。
        
        Args:
            tags: 标签集合
            
        Returns:
            匹配的渠道列表
        """
        with self._lock:
            result = []
            for channel in self._channels.values():
                if channel.tags & tags:  # 有交集
                    result.append(channel)
            return result
    
    def find_channels_by_type(self, channel_type: ChannelType) -> List[ChannelConfig]:
        """
        根据类型查找渠道。
        
        Args:
            channel_type: 渠道类型
            
        Returns:
            匹配的渠道列表
        """
        with self._lock:
            return [
                c for c in self._channels.values()
                if c.channel_type == channel_type
            ]
    
    def export_config(self) -> Dict[str, Any]:
        """
        导出完整配置。
        
        Returns:
            配置字典
        """
        with self._lock:
            channels_data = {}
            for channel_id, channel in self._channels.items():
                channels_data[channel_id] = {
                    "config": channel.to_dict(),
                    "models": [
                        m.to_dict() for m in self._channel_models.get(channel_id, [])
                    ],
                    "default_model": self._default_models.get(channel_id),
                }
            
            return {
                "channels": channels_data,
                "total_channels": len(self._channels),
            }
    
    def import_config(self, config: Dict[str, Any]) -> int:
        """
        导入配置。
        
        Args:
            config: 配置字典
            
        Returns:
            导入的渠道数量
        """
        with self._lock:
            imported = 0
            channels_data = config.get("channels", {})
            
            for channel_id, data in channels_data.items():
                # 重建渠道配置
                config_data = data.get("config", {})
                channel_type = ChannelType[config_data.get("channel_type", "CUSTOM")]
                
                channel = ChannelConfig(
                    channel_id=config_data.get("channel_id", channel_id),
                    channel_type=channel_type,
                    channel_name=config_data.get("channel_name", channel_id),
                    required_capabilities=frozenset(
                        ChannelCapability(c) for c in config_data.get("required_capabilities", [])
                    ),
                    preferred_capabilities=frozenset(
                        ChannelCapability(c) for c in config_data.get("preferred_capabilities", [])
                    ),
                    max_latency_ms=config_data.get("max_latency_ms", 10000),
                    max_cost_per_request=config_data.get("max_cost_per_request", 1.0),
                    priority=config_data.get("priority", 50),
                )
                
                self.register_channel(channel)
                
                # 重建模型绑定
                for model_data in data.get("models", []):
                    binding = ModelBinding(
                        model_id=model_data["model_id"],
                        priority=model_data.get("priority", 0),
                        is_default=model_data.get("is_default", False),
                        max_concurrent=model_data.get("max_concurrent", 10),
                        rate_limit=model_data.get("rate_limit", 100),
                        cost_multiplier=model_data.get("cost_multiplier", 1.0),
                    )
                    self.bind_model(channel_id, binding)
                
                imported += 1
            
            return imported
