"""
LLM统一编排器 (LLM Orchestrator)

该模块是AGI多模型编排层的核心，提供统一的LLM调用接口。

Features:
- 核心调度引擎
- 请求路由
- 响应聚合
- 错误处理
- 全链路监控

Author: AGI Team
Version: 1.0.0
"""

import time
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple,
    Callable, Union, AsyncIterator
)
from collections import defaultdict
from datetime import datetime

# 导入路由模块
from .routing import (
    ModelSelector, ChannelMapper, LanguageRouter,
    ModelAggregator, LoadBalancer, CostOptimizer,
    FallbackChain, CircuitBreaker, SemanticCache,
    QuotaManager, PriorityScheduler,
    TaskType, SelectionCriteria, SelectionResult,
    ChannelConfig, ChannelType,
    AggregationStrategy, FusionResult,
    BalanceStrategy, ModelInstance,
    CacheConfig, SemanticCache,
    QuotaConfig, QuotaUsage,
    TaskPriority, ScheduledTask,
)

# 导入Providers
from .providers import (
    BaseLLMProvider, LLMConfig, LLMResponse,
    OpenAIProvider, AnthropicProvider, ZhipuAIProvider,
    DashScopeProvider, MoonshotProvider, DeepSeekProvider,
    LocalModelProvider, create_provider,
    UniversalLLMProvider, UniversalModelConfig,
    ModelRegistry,
)

logger = logging.getLogger(__name__)


class OrchestratorConfig:
    """
    编排器配置
    
    Attributes:
        enable_routing: 是否启用智能路由
        enable_caching: 是否启用缓存
        enable_fallback: 是否启用Fallback
        enable_circuit_breaker: 是否启用熔断器
        enable_cost_optimization: 是否启用成本优化
        enable_quota: 是否启用配额管理
        enable_priority_scheduling: 是否启用优先级调度
        default_timeout: 默认超时时间
        max_retries: 最大重试次数
    """
    
    def __init__(
        self,
        enable_routing: bool = True,
        enable_caching: bool = True,
        enable_fallback: bool = True,
        enable_circuit_breaker: bool = True,
        enable_cost_optimization: bool = True,
        enable_quota: bool = True,
        enable_priority_scheduling: bool = False,
        default_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.enable_routing = enable_routing
        self.enable_caching = enable_caching
        self.enable_fallback = enable_fallback
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_cost_optimization = enable_cost_optimization
        self.enable_quota = enable_quota
        self.enable_priority_scheduling = enable_priority_scheduling
        self.default_timeout = default_timeout
        self.max_retries = max_retries


@dataclass
class OrchestratorRequest:
    """
    编排器请求
    
    Attributes:
        prompt: 用户提示
        system: 系统消息
        messages: 消息历史
        model_id: 指定模型ID
        channel_id: 渠道ID
        user_id: 用户ID
        task_type: 任务类型
        priority: 优先级
        temperature: 温度参数
        max_tokens: 最大Token数
        enable_stream: 是否启用流式
        metadata: 元数据
    """
    prompt: Optional[str] = None
    system: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    model_id: Optional[str] = None
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    task_type: Optional[TaskType] = None
    priority: TaskPriority = TaskPriority.NORMAL
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    enable_stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResponse:
    """
    编排器响应
    
    Attributes:
        content: 响应内容
        model_id: 使用的模型ID
        usage: Token使用量
        latency_ms: 延迟
        from_cache: 是否来自缓存
        cost: 成本
        error: 错误信息
        metadata: 元数据
    """
    content: str
    model_id: str
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0
    from_cache: bool = False
    cost: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.error is None and self.content is not None


class LLMBridge:
    """
    LLM Bridge
    
    连接编排器和具体Provider的桥梁。
    """
    
    def __init__(self, provider: BaseLLMProvider):
        """
        初始化Bridge。
        
        Args:
            provider: LLM Provider
        """
        self._provider = provider
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> LLMResponse:
        """
        生成响应。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            LLM响应
        """
        return await self._provider.generate(messages, **kwargs)
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成。
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            响应片段
        """
        async for chunk in self._provider.stream_generate(messages, **kwargs):
            yield chunk


class Orchestrator:
    """
    LLM统一编排器
    
    Features:
        - 智能模型选择
        - 渠道路由
        - 多语言支持
        - 成本优化
        - 缓存
        - Fallback
        - 熔断器
        - 配额管理
        - 优先级调度
    
    Example:
        ```python
        # 创建编排器
        orchestrator = Orchestrator(OrchestratorConfig(
            enable_routing=True,
            enable_caching=True
        ))
        
        # 配置Provider
        orchestrator.register_provider("openai", OpenAIProvider(
            LLMConfig(model_id="gpt-4", api_key="...")
        ))
        
        # 发送请求
        response = await orchestrator.generate(
            prompt="Hello!",
            channel_id="web_chat"
        )
        
        print(response.content)
        ```
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        初始化编排器。
        
        Args:
            config: 编排器配置
        """
        self._config = config or OrchestratorConfig()
        
        # 核心组件
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._bridges: Dict[str, LLMBridge] = {}
        
        # 通用模型适配器 - 模型注册中心
        self._model_registry = ModelRegistry()
        self._universal_providers: Dict[str, UniversalLLMProvider] = {}
        
        # 路由组件
        if self._config.enable_routing:
            self._model_selector = ModelSelector()
            self._channel_mapper = ChannelMapper()
            self._language_router = LanguageRouter()
        
        # 聚合组件
        if self._config.enable_routing:
            self._aggregator = ModelAggregator()
        
        # 负载均衡
        if self._config.enable_routing:
            self._load_balancer = LoadBalancer()
        
        # 成本优化
        if self._config.enable_cost_optimization:
            self._cost_optimizer = CostOptimizer()
        
        # Fallback
        if self._config.enable_fallback:
            self._fallback_chain = FallbackChain()
        
        # 熔断器
        if self._config.enable_circuit_breaker:
            self._circuit_breaker_manager = CircuitBreakerManager()
        
        # 缓存
        if self._config.enable_caching:
            self._cache = SemanticCache(CacheConfig(
                max_size=10000,
                min_similarity=0.85
            ))
        
        # 配额管理
        if self._config.enable_quota:
            self._quota_manager = QuotaManager()
        
        # 优先级调度
        if self._config.enable_priority_scheduling:
            self._scheduler = PriorityScheduler()
        
        # 统计
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        
        self._lock = asyncio.Lock()
    
    def register_provider(
        self,
        name: str,
        provider: BaseLLMProvider
    ) -> None:
        """
        注册Provider。
        
        Args:
            name: Provider名称
            provider: Provider实例
        """
        self._providers[name] = provider
        self._bridges[name] = LLMBridge(provider)
        logger.info(f"Registered provider: {name}")
    
    def unregister_provider(self, name: str) -> bool:
        """
        注销Provider。
        
        Args:
            name: Provider名称
            
        Returns:
            是否成功
        """
        if name in self._providers:
            del self._providers[name]
            del self._bridges[name]
            logger.info(f"Unregistered provider: {name}")
            return True
        return False
    
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """获取Provider"""
        return self._providers.get(name)
    
    # ========================================================
    # 通用模型适配器集成
    # ========================================================
    
    @property
    def model_registry(self) -> ModelRegistry:
        """
        获取模型注册中心

        Returns:
            ModelRegistry实例
        """
        return self._model_registry
    
    def register_universal_model(
        self,
        model_config: UniversalModelConfig,
        auto_register_provider: bool = True,
    ) -> bool:
        """
        注册通用模型

        通过UniversalLLMProvider注册任意模型，无需单独编写适配器。

        Args:
            model_config: 通用模型配置
            auto_register_provider: 是否自动注册为Provider

        Returns:
            是否注册成功

        Example:
            ```python
            orchestrator.register_universal_model(UniversalModelConfig(
                model_id="zhipu/glm-4",
                model_name="GLM-4",
                provider="zhipu",
                api_base="https://open.bigmodel.cn/api/paas/v4",
                api_key="your-api-key",
                api_protocol="openai",
                auth_type="bearer",
            ))
            ```
        """
        # 注册到模型注册中心
        success = self._model_registry.register_model(model_config)
        if not success:
            return False

        # 自动创建Provider
        if auto_register_provider:
            provider = UniversalLLMProvider(model_config=model_config)
            provider_name = f"universal_{model_config.model_id.replace('/', '_')}"
            self._providers[provider_name] = provider
            self._bridges[provider_name] = LLMBridge(provider)
            self._universal_providers[model_config.model_id] = provider
            logger.info(
                f"已注册通用模型Provider: {model_config.model_id} "
                f"(provider_name={provider_name})"
            )

        return True
    
    def register_universal_models_from_config(
        self,
        config_path: str,
        auto_register_providers: bool = True,
    ) -> Tuple[int, int]:
        """
        从配置文件批量注册通用模型

        Args:
            config_path: 配置文件路径 (YAML/JSON)
            auto_register_providers: 是否自动注册为Provider

        Returns:
            (成功数, 失败数)

        Example:
            ```python
            # 加载国产模型预置配置
            success, failed = orchestrator.register_universal_models_from_config(
                "presets/china_models.yaml"
            )
            print(f"注册完成: 成功 {success}, 失败 {failed}")
            ```
        """
        success, failed = self._model_registry.load_from_config(config_path)

        if auto_register_providers and success > 0:
            for model_config in self._model_registry.list_models():
                if model_config.model_id not in self._universal_providers:
                    provider = UniversalLLMProvider(model_config=model_config)
                    provider_name = (
                        f"universal_{model_config.model_id.replace('/', '_')}"
                    )
                    self._providers[provider_name] = provider
                    self._bridges[provider_name] = LLMBridge(provider)
                    self._universal_providers[model_config.model_id] = provider

        return success, failed
    
    def import_universal_preset(
        self,
        preset_name: str = "china_models",
        auto_register_providers: bool = True,
    ) -> Tuple[int, int]:
        """
        导入预置模型配置

        Args:
            preset_name: 预置配置名称 (如 "china_models")
            auto_register_providers: 是否自动注册为Provider

        Returns:
            (成功数, 失败数)

        Example:
            ```python
            # 导入国产模型预置配置 (30+模型)
            success, failed = orchestrator.import_universal_preset("china_models")
            ```
        """
        success, failed = self._model_registry.import_preset(preset_name)

        if auto_register_providers and success > 0:
            for model_config in self._model_registry.list_models():
                if model_config.model_id not in self._universal_providers:
                    provider = UniversalLLMProvider(model_config=model_config)
                    provider_name = (
                        f"universal_{model_config.model_id.replace('/', '_')}"
                    )
                    self._providers[provider_name] = provider
                    self._bridges[provider_name] = LLMBridge(provider)
                    self._universal_providers[model_config.model_id] = provider

        return success, failed
    
    def get_universal_provider(
        self,
        model_id: str
    ) -> Optional[UniversalLLMProvider]:
        """
        获取通用模型Provider

        Args:
            model_id: 模型ID

        Returns:
            UniversalLLMProvider实例
        """
        return self._universal_providers.get(model_id)
    
    def list_universal_models(self) -> List[Dict[str, Any]]:
        """
        列出所有已注册的通用模型

        Returns:
            模型信息列表
        """
        return self._model_registry.list_models()
    
    async def generate(
        self,
        request: OrchestratorRequest
    ) -> OrchestratorResponse:
        """
        生成响应。
        
        Args:
            request: 编排器请求
            
        Returns:
            编排器响应
        """
        start_time = time.time()
        self._stats["total_requests"] += 1
        
        try:
            # 1. 检查配额
            if self._config.enable_quota and request.user_id:
                can_proceed, reason = self._quota_manager.check_quota(
                    request.user_id,
                    tokens=100  # 预估
                )
                if not can_proceed:
                    return OrchestratorResponse(
                        content="",
                        model_id="",
                        error=f"Quota exceeded: {reason}",
                        latency_ms=(time.time() - start_time) * 1000,
                        metadata={"stage": "quota_check"}
                    )
            
            # 2. 检查缓存
            if self._config.enable_caching:
                cache_result = self._cache.get_semantic(
                    request.prompt or str(request.messages)
                )
                if cache_result.hit:
                    self._stats["cache_hits"] += 1
                    return OrchestratorResponse(
                        content=str(cache_result.value),
                        model_id=cache_result.match.entry.metadata.get("model_id", "cached"),
                        latency_ms=(time.time() - start_time) * 1000,
                        from_cache=True,
                        metadata={
                            "similarity": cache_result.match.similarity,
                            "stage": "cache_hit"
                        }
                    )
            
            # 3. 选择模型
            model_id, routing_info = await self._select_model(request)
            
            # 4. 检查熔断器
            if self._config.enable_circuit_breaker:
                breaker = self._circuit_breaker_manager.get_breaker(model_id)
                if not breaker.can_request():
                    logger.warning(f"Circuit breaker open for {model_id}")
                    # 尝试Fallback
                    if self._config.enable_fallback:
                        model_id, routing_info = await self._select_fallback(request)
            
            # 5. 构建消息
            messages = self._build_messages(request)
            
            # 6. 调用模型
            provider = self._get_provider_for_model(model_id)
            if not provider:
                return OrchestratorResponse(
                    content="",
                    model_id=model_id,
                    error=f"No provider for model {model_id}",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            bridge = self._bridges.get(provider.provider_name)
            if not bridge:
                return OrchestratorResponse(
                    content="",
                    model_id=model_id,
                    error=f"No bridge for provider {provider.provider_name}",
                    latency_ms=(time.time() - start_time) * 1000
                )
            
            # 准备生成参数
            gen_kwargs = {
                "temperature": request.temperature,
            }
            if request.max_tokens:
                gen_kwargs["max_tokens"] = request.max_tokens
            
            # 调用
            llm_response = await bridge.generate(messages, **gen_kwargs)
            
            # 记录熔断器
            if self._config.enable_circuit_breaker:
                breaker = self._circuit_breaker_manager.get_breaker(model_id)
                if llm_response.is_success:
                    breaker.record_success()
                else:
                    breaker.record_failure(llm_response.error)
            
            # 计算成本
            cost = None
            if self._config.enable_cost_optimization:
                cost = self._cost_optimizer.cost_registry.calculate_cost(
                    model_id,
                    llm_response.input_tokens,
                    llm_response.output_tokens
                )
                self._stats["total_cost"] += cost or 0
            
            # 更新配额
            if self._config.enable_quota and request.user_id:
                self._quota_manager.consume(
                    request.user_id,
                    tokens=llm_response.total_tokens,
                    requests=1
                )
            
            # 记录成本
            if self._config.enable_cost_optimization and llm_response.is_success:
                self._cost_optimizer.record_cost(
                    model_id=model_id,
                    input_tokens=llm_response.input_tokens,
                    output_tokens=llm_response.output_tokens,
                    user_id=request.user_id,
                    channel_id=request.channel_id
                )
            
            # 更新统计
            self._stats["successful_requests"] += 1
            self._stats["total_tokens"] += llm_response.total_tokens
            
            # 存入缓存
            if self._config.enable_caching and llm_response.is_success:
                self._cache.set(
                    request.prompt or str(request.messages),
                    llm_response.content,
                    metadata={"model_id": model_id}
                )
            
            return OrchestratorResponse(
                content=llm_response.content,
                model_id=model_id,
                usage=llm_response.usage,
                latency_ms=(time.time() - start_time) * 1000,
                cost=cost,
                metadata={
                    "routing_info": routing_info,
                    "stage": "completed"
                }
            )
            
        except Exception as e:
            logger.error(f"Generate error: {e}")
            self._stats["failed_requests"] += 1
            
            return OrchestratorResponse(
                content="",
                model_id="",
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"stage": "exception"}
            )
    
    async def stream_generate(
        self,
        request: OrchestratorRequest
    ) -> AsyncIterator[str]:
        """
        流式生成。
        
        Args:
            request: 编排器请求
            
        Yields:
            响应片段
        """
        # 选择模型
        model_id, _ = await self._select_model(request)
        
        # 构建消息
        messages = self._build_messages(request)
        
        # 获取Provider
        provider = self._get_provider_for_model(model_id)
        if not provider:
            yield f"Error: No provider for model {model_id}"
            return
        
        bridge = self._bridges.get(provider.provider_name)
        if not bridge:
            yield f"Error: No bridge for provider {provider.provider_name}"
            return
        
        # 流式生成
        gen_kwargs = {"temperature": request.temperature}
        if request.max_tokens:
            gen_kwargs["max_tokens"] = request.max_tokens
        
        async for chunk in bridge.stream(messages, **gen_kwargs):
            yield chunk
    
    async def _select_model(
        self,
        request: OrchestratorRequest
    ) -> Tuple[str, Dict[str, Any]]:
        """选择模型"""
        # 如果指定了模型，直接使用
        if request.model_id:
            return request.model_id, {"source": "explicit"}
        
        # 如果启用了路由
        if self._config.enable_routing:
            # 渠道路由
            if request.channel_id:
                default_model = self._channel_mapper.get_default_model(request.channel_id)
                if default_model:
                    return default_model, {"source": "channel"}
            
            # 语言路由
            if request.prompt:
                lang_result = self._language_router.route(
                    request.prompt,
                    user_id=request.user_id,
                    channel_id=request.channel_id
                )
                if lang_result.model_id:
                    return lang_result.model_id, {
                        "source": "language",
                        "detected_lang": lang_result.detection.detected_language.value
                    }
            
            # 智能选择
            if request.task_type:
                criteria = SelectionCriteria(
                    task_type=request.task_type,
                    primary_language=request.metadata.get("language", "zh")
                )
                result = self._model_selector.select(criteria)
                return result.selected_model.model_id, {
                    "source": "selector",
                    "score": result.score,
                    "reasoning": result.reasoning
                }
        
        # 默认使用第一个Provider
        if self._providers:
            first_provider = next(iter(self._providers.values()))
            return first_provider.model_id, {"source": "default"}
        
        raise ValueError("No providers available")
    
    async def _select_fallback(
        self,
        request: OrchestratorRequest
    ) -> Tuple[str, Dict[str, Any]]:
        """选择Fallback模型"""
        if self._config.enable_fallback:
            # 使用成本优先链
            chain_id = "cost_effective"
            result = await self._fallback_chain.execute(
                prompt=request.prompt or str(request.messages),
                chain_id=chain_id,
                call_func=self._mock_call_func
            )
            if result.success:
                return result.used_model_id, {
                    "source": "fallback",
                    "chain": chain_id,
                    "attempts": result.attempts
                }
        
        # 回退到默认
        if self._providers:
            first_provider = next(iter(self._providers.values()))
            return first_provider.model_id, {"source": "fallback_default"}
        
        raise ValueError("No fallback available")
    
    async def _mock_call_func(self, model_id: str, prompt: Any) -> Dict:
        """模拟调用函数（用于Fallback测试）"""
        return {"result": f"Response from {model_id}"}
    
    def _get_provider_for_model(self, model_id: str) -> Optional[BaseLLMProvider]:
        """获取模型对应的Provider"""
        # 1. 优先检查通用模型Provider
        if model_id in self._universal_providers:
            return self._universal_providers[model_id]
        
        # 2. 检查模型注册中心是否有该模型，动态创建Provider
        model_config = self._model_registry.get_model(model_id)
        if model_config and model_id not in self._universal_providers:
            provider = UniversalLLMProvider(model_config=model_config)
            provider_name = f"universal_{model_id.replace('/', '_')}"
            self._providers[provider_name] = provider
            self._bridges[provider_name] = LLMBridge(provider)
            self._universal_providers[model_id] = provider
            return provider

        # 3. 检查传统Provider映射
        provider_mapping = {
            "gpt-4": "openai", "gpt-3.5-turbo": "openai",
            "claude-3": "anthropic", "claude-2": "anthropic",
            "glm-": "zhipuai",
            "qwen": "dashscope",
            "moonshot": "moonshot",
            "deepseek": "deepseek",
        }
        
        for prefix, provider_name in provider_mapping.items():
            if model_id.startswith(prefix) and provider_name in self._providers:
                return self._providers[provider_name]
        
        # 4. 默认返回第一个Provider
        if self._providers:
            return next(iter(self._providers.values()))
        
        return None
    
    def _build_messages(self, request: OrchestratorRequest) -> List[Dict[str, Any]]:
        """构建消息列表"""
        messages = []
        
        if request.system:
            messages.append({"role": "system", "content": request.system})
        
        if request.messages:
            messages.extend(request.messages)
        
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self._stats["total_requests"],
            "successful_requests": self._stats["successful_requests"],
            "failed_requests": self._stats["failed_requests"],
            "success_rate": (
                self._stats["successful_requests"] / max(self._stats["total_requests"], 1)
            ),
            "cache_hits": self._stats["cache_hits"],
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(self._stats["total_requests"], 1)
            ),
            "total_tokens": self._stats["total_tokens"],
            "total_cost": self._stats["total_cost"],
            "providers": {
                name: provider.get_stats()
                for name, provider in self._providers.items()
            },
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
