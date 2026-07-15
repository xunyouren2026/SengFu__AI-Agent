"""
Deep Integration Module
深度集成模块

提供自适应算法选择、智能路由、性能优化、模型管理等高级功能
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """后端类型"""
    LOCAL = "local"  # 本地模型
    API = "api"  # API服务
    HYBRID = "hybrid"  # 混合模式


class ModelCapability(Enum):
    """模型能力"""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    TTS = "tts"
    STT = "stt"
    EMBEDDING = "embedding"
    RERANK = "rerank"


@dataclass
class ModelInfo:
    """模型信息"""
    id: str
    name: str
    backend: BackendType
    capabilities: List[ModelCapability]
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_function_calling: bool = False
    cost_per_1k_tokens: float = 0.0
    latency_ms: int = 0
    quality_score: float = 0.0
    availability: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "backend": self.backend.value,
            "capabilities": [c.value for c in self.capabilities],
            "max_tokens": self.max_tokens,
            "supports_streaming": self.supports_streaming,
            "supports_vision": self.supports_vision,
            "supports_function_calling": self.supports_function_calling,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "latency_ms": self.latency_ms,
            "quality_score": self.quality_score,
            "availability": self.availability,
        }


@dataclass
class RoutingDecision:
    """路由决策"""
    selected_model: str
    backend: BackendType
    reason: str
    estimated_latency: float
    estimated_cost: float
    fallback_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveAlgorithmSelector:
    """
    自适应算法选择器
    
    功能：
    - 自动选择最优模型
    - API/本地模式自动切换
    - 成本优化
    - 性能优化
    - 故障转移
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._models: Dict[str, ModelInfo] = {}
        self._performance_history: Dict[str, List[Dict]] = {}
        self._cost_tracker: Dict[str, float] = {}
        self._initialized = False
        
        # 路由策略
        self._strategy = self.config.get("routing_strategy", "balanced")  # cost, performance, balanced
        
        # 阈值配置
        self._latency_threshold = self.config.get("latency_threshold_ms", 5000)
        self._cost_threshold = self.config.get("cost_threshold", 0.1)
        self._availability_threshold = self.config.get("availability_threshold", 0.95)
        
    async def initialize(self):
        """初始化选择器"""
        if self._initialized:
            return
        
        # 注册默认模型
        await self._register_default_models()
        
        self._initialized = True
        logger.info("Adaptive algorithm selector initialized")
    
    async def _register_default_models(self):
        """注册默认模型"""
        # 文本生成模型
        models = [
            # OpenAI
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                backend=BackendType.API,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=128000,
                supports_streaming=True,
                supports_vision=True,
                supports_function_calling=True,
                cost_per_1k_tokens=0.005,
                latency_ms=1500,
                quality_score=0.95,
            ),
            ModelInfo(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                backend=BackendType.API,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=128000,
                supports_streaming=True,
                supports_vision=True,
                supports_function_calling=True,
                cost_per_1k_tokens=0.01,
                latency_ms=2000,
                quality_score=0.93,
            ),
            ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                backend=BackendType.API,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=16385,
                supports_streaming=True,
                supports_function_calling=True,
                cost_per_1k_tokens=0.0005,
                latency_ms=500,
                quality_score=0.85,
            ),
            # Claude
            ModelInfo(
                id="claude-3-opus",
                name="Claude 3 Opus",
                backend=BackendType.API,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=200000,
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.015,
                latency_ms=3000,
                quality_score=0.96,
            ),
            ModelInfo(
                id="claude-3-sonnet",
                name="Claude 3 Sonnet",
                backend=BackendType.API,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=200000,
                supports_streaming=True,
                supports_vision=True,
                cost_per_1k_tokens=0.003,
                latency_ms=1000,
                quality_score=0.90,
            ),
            # 本地模型
            ModelInfo(
                id="llama-3-70b-local",
                name="Llama 3 70B (Local)",
                backend=BackendType.LOCAL,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=8192,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,
                latency_ms=3000,
                quality_score=0.88,
            ),
            ModelInfo(
                id="qwen-72b-local",
                name="Qwen 72B (Local)",
                backend=BackendType.LOCAL,
                capabilities=[ModelCapability.TEXT_GENERATION],
                max_tokens=32768,
                supports_streaming=True,
                cost_per_1k_tokens=0.0,
                latency_ms=2500,
                quality_score=0.87,
            ),
            # 图像生成
            ModelInfo(
                id="dall-e-3",
                name="DALL-E 3",
                backend=BackendType.API,
                capabilities=[ModelCapability.IMAGE_GENERATION],
                max_tokens=0,
                cost_per_1k_tokens=0.04,
                latency_ms=15000,
                quality_score=0.92,
            ),
            ModelInfo(
                id="sd-xl-local",
                name="Stable Diffusion XL (Local)",
                backend=BackendType.LOCAL,
                capabilities=[ModelCapability.IMAGE_GENERATION],
                max_tokens=0,
                cost_per_1k_tokens=0.0,
                latency_ms=5000,
                quality_score=0.85,
            ),
            # TTS
            ModelInfo(
                id="edge-tts",
                name="Edge TTS",
                backend=BackendType.API,
                capabilities=[ModelCapability.TTS],
                max_tokens=0,
                cost_per_1k_tokens=0.0,
                latency_ms=500,
                quality_score=0.80,
            ),
            ModelInfo(
                id="bark-local",
                name="Bark (Local)",
                backend=BackendType.LOCAL,
                capabilities=[ModelCapability.TTS],
                max_tokens=0,
                cost_per_1k_tokens=0.0,
                latency_ms=5000,
                quality_score=0.90,
            ),
        ]
        
        for model in models:
            self._models[model.id] = model
    
    def register_model(self, model: ModelInfo):
        """注册模型"""
        self._models[model.id] = model
        logger.info(f"Registered model: {model.id}")
    
    def unregister_model(self, model_id: str):
        """注销模型"""
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Unregistered model: {model_id}")
    
    async def select_model(
        self,
        capability: ModelCapability,
        constraints: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        选择最优模型
        
        Args:
            capability: 所需能力
            constraints: 约束条件（预算、延迟等）
            preferences: 偏好设置
        
        Returns:
            路由决策
        """
        await self.initialize()
        
        constraints = constraints or {}
        preferences = preferences or {}
        
        # 筛选符合条件的模型
        candidates = []
        for model_id, model in self._models.items():
            if capability in model.capabilities:
                candidates.append(model)
        
        if not candidates:
            raise ValueError(f"No models available for capability: {capability}")
        
        # 应用约束过滤
        filtered = []
        for model in candidates:
            # 检查可用性
            if model.availability < self._availability_threshold:
                continue
            
            # 检查延迟约束
            max_latency = constraints.get("max_latency_ms")
            if max_latency and model.latency_ms > max_latency:
                continue
            
            # 检查成本约束
            max_cost = constraints.get("max_cost_per_1k")
            if max_cost and model.cost_per_1k_tokens > max_cost:
                continue
            
            # 检查后端偏好
            preferred_backend = preferences.get("backend")
            if preferred_backend and model.backend.value != preferred_backend:
                continue
            
            filtered.append(model)
        
        if not filtered:
            # 如果没有符合条件的，放宽约束
            filtered = candidates
        
        # 根据策略排序
        strategy = preferences.get("strategy", self._strategy)
        
        if strategy == "cost":
            # 成本优先
            filtered.sort(key=lambda m: m.cost_per_1k_tokens)
        elif strategy == "performance":
            # 性能优先
            filtered.sort(key=lambda m: m.latency_ms)
        elif strategy == "quality":
            # 质量优先
            filtered.sort(key=lambda m: m.quality_score, reverse=True)
        else:
            # 平衡策略
            filtered.sort(key=lambda m: self._calculate_score(m, constraints), reverse=True)
        
        # 选择最优模型
        selected = filtered[0]
        fallbacks = [m.id for m in filtered[1:4]] if len(filtered) > 1 else []
        
        return RoutingDecision(
            selected_model=selected.id,
            backend=selected.backend,
            reason=f"Selected based on {strategy} strategy",
            estimated_latency=selected.latency_ms,
            estimated_cost=selected.cost_per_1k_tokens,
            fallback_models=fallbacks,
            metadata={
                "quality_score": selected.quality_score,
                "availability": selected.availability,
            }
        )
    
    def _calculate_score(self, model: ModelInfo, constraints: Dict) -> float:
        """计算综合得分"""
        # 归一化各指标
        latency_score = 1 - (model.latency_ms / 10000)  # 假设最大延迟10秒
        cost_score = 1 - min(model.cost_per_1k_tokens / 0.1, 1)  # 假设最大成本0.1/1k
        quality_score = model.quality_score
        availability_score = model.availability
        
        # 加权平均
        weights = {
            "latency": 0.25,
            "cost": 0.25,
            "quality": 0.35,
            "availability": 0.15,
        }
        
        return (
            latency_score * weights["latency"] +
            cost_score * weights["cost"] +
            quality_score * weights["quality"] +
            availability_score * weights["availability"]
        )
    
    async def record_performance(
        self,
        model_id: str,
        latency_ms: float,
        success: bool,
        tokens_used: int = 0,
        cost: float = 0.0,
    ):
        """记录性能数据"""
        if model_id not in self._performance_history:
            self._performance_history[model_id] = []
        
        record = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "success": success,
            "tokens_used": tokens_used,
            "cost": cost,
        }
        
        self._performance_history[model_id].append(record)
        
        # 更新模型可用性
        if model_id in self._models:
            recent = self._performance_history[model_id][-100:]
            success_rate = sum(1 for r in recent if r["success"]) / len(recent)
            self._models[model_id].availability = success_rate
            
            # 更新平均延迟
            avg_latency = sum(r["latency_ms"] for r in recent) / len(recent)
            self._models[model_id].latency_ms = int(avg_latency)
        
        # 更新成本追踪
        self._cost_tracker[model_id] = self._cost_tracker.get(model_id, 0) + cost
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self._models.get(model_id)
    
    def list_models(
        self,
        capability: Optional[ModelCapability] = None,
        backend: Optional[BackendType] = None,
    ) -> List[ModelInfo]:
        """列出模型"""
        models = list(self._models.values())
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        if backend:
            models = [m for m in models if m.backend == backend]
        
        return models
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_cost = sum(self._cost_tracker.values())
        
        model_stats = []
        for model_id, model in self._models.items():
            history = self._performance_history.get(model_id, [])
            recent = history[-100:] if history else []
            
            model_stats.append({
                "id": model_id,
                "total_requests": len(history),
                "success_rate": sum(1 for r in recent if r["success"]) / len(recent) if recent else 0,
                "avg_latency_ms": sum(r["latency_ms"] for r in recent) / len(recent) if recent else 0,
                "total_cost": self._cost_tracker.get(model_id, 0),
            })
        
        return {
            "total_models": len(self._models),
            "total_cost": total_cost,
            "models": model_stats,
        }


class IntelligentRouter:
    """
    智能路由器
    
    功能：
    - 请求路由
    - 负载均衡
    - 故障转移
    - 熔断器
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._selector = AdaptiveAlgorithmSelector(config)
        self._circuit_breakers: Dict[str, Dict] = {}
        self._request_counts: Dict[str, int] = {}
        self._initialized = False
        
        # 熔断器配置
        self._failure_threshold = self.config.get("failure_threshold", 5)
        self._recovery_timeout = self.config.get("recovery_timeout", 60)
        
    async def initialize(self):
        """初始化路由器"""
        if self._initialized:
            return
        
        await self._selector.initialize()
        self._initialized = True
        logger.info("Intelligent router initialized")
    
    async def route_request(
        self,
        capability: ModelCapability,
        request_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        路由请求
        
        Args:
            capability: 所需能力
            request_data: 请求数据
            constraints: 约束条件
        
        Returns:
            (模型ID, 路由信息)
        """
        await self.initialize()
        
        # 选择模型
        decision = await self._selector.select_model(capability, constraints)
        
        # 检查熔断器
        if self._is_circuit_open(decision.selected_model):
            # 使用备用模型
            for fallback in decision.fallback_models:
                if not self._is_circuit_open(fallback):
                    decision.selected_model = fallback
                    decision.reason = "Fallback due to circuit breaker"
                    break
        
        # 记录请求
        self._request_counts[decision.selected_model] = self._request_counts.get(decision.selected_model, 0) + 1
        
        return decision.selected_model, decision.to_dict()
    
    def _is_circuit_open(self, model_id: str) -> bool:
        """检查熔断器是否打开"""
        if model_id not in self._circuit_breakers:
            return False
        
        cb = self._circuit_breakers[model_id]
        
        if cb["state"] == "open":
            # 检查是否可以尝试恢复
            if time.time() - cb["last_failure"] > self._recovery_timeout:
                cb["state"] = "half-open"
                return False
            return True
        
        return False
    
    def record_success(self, model_id: str, latency_ms: float):
        """记录成功"""
        if model_id in self._circuit_breakers:
            self._circuit_breakers[model_id]["failures"] = 0
            self._circuit_breakers[model_id]["state"] = "closed"
        
        asyncio.create_task(
            self._selector.record_performance(model_id, latency_ms, True)
        )
    
    def record_failure(self, model_id: str, error: str):
        """记录失败"""
        if model_id not in self._circuit_breakers:
            self._circuit_breakers[model_id] = {
                "failures": 0,
                "state": "closed",
                "last_failure": 0,
            }
        
        cb = self._circuit_breakers[model_id]
        cb["failures"] += 1
        cb["last_failure"] = time.time()
        
        if cb["failures"] >= self._failure_threshold:
            cb["state"] = "open"
            logger.warning(f"Circuit breaker opened for model: {model_id}")
        
        asyncio.create_task(
            self._selector.record_performance(model_id, 0, False)
        )
    
    def get_selector(self) -> AdaptiveAlgorithmSelector:
        """获取算法选择器"""
        return self._selector


class PerformanceOptimizer:
    """
    性能优化器
    
    功能：
    - 缓存管理
    - 批处理
    - 并发控制
    - 预加载
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = self.config.get("cache_ttl", 3600)
        self._batch_queue: Dict[str, List] = {}
        self._batch_size = self.config.get("batch_size", 10)
        self._batch_timeout = self.config.get("batch_timeout", 0.1)
        
    def get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
            else:
                del self._cache[key]
        return None
    
    def set_cached(self, key: str, data: Any):
        """设置缓存"""
        self._cache[key] = {
            "data": data,
            "timestamp": time.time(),
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    async def batch_process(
        self,
        batch_key: str,
        item: Any,
        processor: Callable,
    ) -> Any:
        """批处理"""
        if batch_key not in self._batch_queue:
            self._batch_queue[batch_key] = []
        
        self._batch_queue[batch_key].append(item)
        
        # 检查是否达到批处理条件
        if len(self._batch_queue[batch_key]) >= self._batch_size:
            batch = self._batch_queue[batch_key]
            self._batch_queue[batch_key] = []
            return await processor(batch)
        
        # 等待超时
        await asyncio.sleep(self._batch_timeout)
        
        if self._batch_queue[batch_key]:
            batch = self._batch_queue[batch_key]
            self._batch_queue[batch_key] = []
            return await processor(batch)
        
        return None


# 全局实例
_selector: Optional[AdaptiveAlgorithmSelector] = None
_router: Optional[IntelligentRouter] = None
_optimizer: Optional[PerformanceOptimizer] = None


def get_selector() -> AdaptiveAlgorithmSelector:
    """获取全局算法选择器"""
    global _selector
    if _selector is None:
        _selector = AdaptiveAlgorithmSelector()
    return _selector


def get_router() -> IntelligentRouter:
    """获取全局路由器"""
    global _router
    if _router is None:
        _router = IntelligentRouter()
    return _router


def get_optimizer() -> PerformanceOptimizer:
    """获取全局优化器"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PerformanceOptimizer()
    return _optimizer


async def init_deep_integration(config: Optional[Dict[str, Any]] = None):
    """初始化深度集成模块"""
    global _selector, _router, _optimizer
    
    config = config or {}
    
    _selector = AdaptiveAlgorithmSelector(config)
    await _selector.initialize()
    
    _router = IntelligentRouter(config)
    await _router.initialize()
    
    _optimizer = PerformanceOptimizer(config)
    
    logger.info("Deep integration module initialized")
    
    return {
        "selector": _selector,
        "router": _router,
        "optimizer": _optimizer,
    }
