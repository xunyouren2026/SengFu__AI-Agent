"""
模型Fallback链 (Model Fallback Chain)

该模块提供模型故障自动切换和恢复机制，支持：
- 故障自动切换
- 优先级链配置
- 降级策略
- 恢复检测

核心功能：
1. 多级Fallback链
2. 故障检测与自动切换
3. 降级策略执行
4. 恢复自动熔合
5. 状态持久化

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Any, Set, Tuple, 
    Callable, Union, TypeVar, Awaitable
)
from collections import defaultdict
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

T = TypeVar('T')


class FallbackStrategy(Enum):
    """
    Fallback策略
    """
    # 简单策略
    SEQUENTIAL = auto()           # 顺序尝试
    PARALLEL = auto()            # 并行尝试
    
    # 智能策略
    QUALITY_BASED = auto()       # 质量优先
    COST_BASED = auto()          # 成本优先
    LATENCY_BASED = auto()       # 延迟优先
    
    # 降级策略
    GRACEFUL_DEGRADE = auto()    # 优雅降级
    STRICT_DEGRADE = auto()       # 严格降级


@dataclass
class FallbackConfig:
    """
    Fallback配置
    
    Attributes:
        max_retries: 最大重试次数
        retry_delay_ms: 重试延迟
        timeout_ms: 超时时间
        backoff_multiplier: 退避乘数
        max_backoff_ms: 最大退避延迟
        fallback_strategy: Fallback策略
        enable_recovery_check: 是否启用恢复检查
    """
    max_retries: int = 3
    retry_delay_ms: float = 1000.0
    timeout_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    max_backoff_ms: float = 30000.0
    fallback_strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    enable_recovery_check: bool = True
    consecutive_failure_threshold: int = 3


@dataclass
class RecoveryConfig:
    """
    恢复检测配置
    
    Attributes:
        check_interval_ms: 检查间隔
        success_threshold: 成功阈值
        recovery_timeout_ms: 恢复超时
        auto_recover: 是否自动恢复
    """
    check_interval_ms: float = 60000.0
    success_threshold: int = 3
    recovery_timeout_ms: float = 300000.0
    auto_recover: bool = True


@dataclass
class FallbackResult:
    """
    Fallback执行结果
    
    Attributes:
        final_result: 最终结果
        used_model_id: 实际使用的模型ID
        attempts: 尝试次数
        fallback_chain: 尝试的模型链
        total_time_ms: 总耗时
        success: 是否成功
        error: 错误信息
    """
    final_result: Any
    used_model_id: str
    attempts: int
    fallback_chain: List[str]
    total_time_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "final_result": str(self.final_result)[:200] if self.final_result else None,
            "used_model_id": self.used_model_id,
            "attempts": self.attempts,
            "fallback_chain": self.fallback_chain,
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class FallbackChainConfig:
    """
    Fallback链配置
    
    Attributes:
        chain_id: 链ID
        chain_name: 链名称
        models: 模型链 (按优先级排序)
        config: Fallback配置
        recovery_config: 恢复配置
    """
    chain_id: str
    chain_name: str
    models: List[str]  # 按优先级排序
    config: FallbackConfig = field(default_factory=FallbackConfig)
    recovery_config: RecoveryConfig = field(default_factory=RecoveryConfig)


class ModelFallbackChain:
    """
    模型Fallback链管理器
    
    Features:
        - 多级Fallback链
        - 故障自动切换
        - 降级策略执行
        - 恢复自动检测
        - 状态持久化
    
    Example:
        ```python
        # 创建Fallback链
        chain = ModelFallbackChain()
        
        # 定义调用函数
        async def call_model(model_id, prompt):
            # 实际调用模型
            return {"result": f"Response from {model_id}"}
        
        # 执行Fallback链
        result = await chain.execute(
            prompt="Hello",
            chain_id="default",
            call_func=call_model
        )
        
        print(f"Used model: {result.used_model_id}")
        print(f"Attempts: {result.attempts}")
        ```
    """
    
    def __init__(self):
        """初始化Fallback链管理器"""
        self._chains: Dict[str, FallbackChainConfig] = {}
        self._model_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.RLock()
        self._recovery_threads: Dict[str, threading.Thread] = {}
        
        # 设置默认链
        self._setup_default_chains()
    
    def _setup_default_chains(self) -> None:
        """设置默认Fallback链"""
        # 高质量链
        self.register_chain(FallbackChainConfig(
            chain_id="high_quality",
            chain_name="高质量链",
            models=["gpt-4", "claude-3-opus", "gpt-3.5-turbo"],
            config=FallbackConfig(
                max_retries=2,
                fallback_strategy=FallbackStrategy.QUALITY_BASED
            )
        ))
        
        # 成本优先链
        self.register_chain(FallbackChainConfig(
            chain_id="cost_effective",
            chain_name="成本优先链",
            models=["gpt-3.5-turbo", "glm-3-turbo", "deepseek-chat"],
            config=FallbackConfig(
                max_retries=2,
                fallback_strategy=FallbackStrategy.COST_BASED
            )
        ))
        
        # 快速响应链
        self.register_chain(FallbackChainConfig(
            chain_id="fast_response",
            chain_name="快速响应链",
            models=["gpt-3.5-turbo", "claude-3-haiku", "qwen-turbo"],
            config=FallbackConfig(
                max_retries=2,
                timeout_ms=10000,
                fallback_strategy=FallbackStrategy.LATENCY_BASED
            )
        ))
        
        # 国产模型链
        self.register_chain(FallbackChainConfig(
            chain_id="domestic",
            chain_name="国产模型链",
            models=["glm-4", "qwen-turbo", "moonshot-v1-8k", "deepseek-chat"],
            config=FallbackConfig(
                max_retries=2
            )
        ))
    
    def register_chain(self, chain_config: FallbackChainConfig) -> None:
        """
        注册Fallback链。
        
        Args:
            chain_config: Fallback链配置
        """
        with self._lock:
            self._chains[chain_config.chain_id] = chain_config
            logger.info(
                f"Registered fallback chain: {chain_config.chain_id} "
                f"with models: {chain_config.models}"
            )
    
    def unregister_chain(self, chain_id: str) -> bool:
        """
        注销Fallback链。
        
        Args:
            chain_id: 链ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
                logger.info(f"Unregistered fallback chain: {chain_id}")
                return True
            return False
    
    def get_chain(self, chain_id: str) -> Optional[FallbackChainConfig]:
        """获取链配置"""
        return self._chains.get(chain_id)
    
    def list_chains(self) -> List[FallbackChainConfig]:
        """列出所有链"""
        return list(self._chains.values())
    
    def update_chain(
        self,
        chain_id: str,
        models: Optional[List[str]] = None,
        config: Optional[FallbackConfig] = None
    ) -> bool:
        """
        更新链配置。
        
        Args:
            chain_id: 链ID
            models: 新的模型列表
            config: 新的配置
            
        Returns:
            是否成功
        """
        with self._lock:
            if chain_id not in self._chains:
                return False
            
            chain = self._chains[chain_id]
            if models is not None:
                chain.models = models
            if config is not None:
                chain.config = config
            
            return True
    
    async def execute(
        self,
        prompt: Any,
        chain_id: str,
        call_func: Callable[[str, Any], Awaitable[Any]],
        fallback_config: Optional[FallbackConfig] = None
    ) -> FallbackResult:
        """
        执行Fallback链。
        
        Args:
            prompt: 输入提示
            chain_id: 链ID
            call_func: 模型调用函数
            fallback_config: 临时配置覆盖
            
        Returns:
            Fallback结果
        """
        chain = self._chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        config = fallback_config or chain.config
        start_time = time.time()
        attempts = 0
        fallback_chain = []
        
        # 获取可用的模型列表
        available_models = self._filter_available_models(chain.models, config)
        
        if not available_models:
            return FallbackResult(
                final_result=None,
                used_model_id="",
                attempts=0,
                fallback_chain=[],
                total_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error="No available models in chain"
            )
        
        # 根据策略选择执行方式
        if config.fallback_strategy == FallbackStrategy.PARALLEL:
            return await self._execute_parallel(
                prompt, available_models, call_func, config, start_time
            )
        else:
            return await self._execute_sequential(
                prompt, available_models, call_func, config, start_time
            )
    
    async def _execute_sequential(
        self,
        prompt: Any,
        models: List[str],
        call_func: Callable[[str, Any], Awaitable[Any]],
        config: FallbackConfig,
        start_time: float
    ) -> FallbackResult:
        """顺序执行Fallback"""
        attempts = 0
        fallback_chain = []
        last_error = None
        current_delay = config.retry_delay_ms
        
        for model_id in models:
            attempts += 1
            fallback_chain.append(model_id)
            
            try:
                # 调用模型
                result = await asyncio.wait_for(
                    call_func(model_id, prompt),
                    timeout=config.timeout_ms / 1000.0
                )
                
                # 成功，记录状态
                self._record_success(model_id)
                
                return FallbackResult(
                    final_result=result,
                    used_model_id=model_id,
                    attempts=attempts,
                    fallback_chain=fallback_chain,
                    total_time_ms=(time.time() - start_time) * 1000,
                    success=True
                )
                
            except asyncio.TimeoutError:
                last_error = f"Timeout calling {model_id}"
                logger.warning(last_error)
                self._record_failure(model_id, "timeout")
                
            except Exception as e:
                last_error = f"Error calling {model_id}: {str(e)}"
                logger.warning(last_error)
                self._record_failure(model_id, str(e))
            
            # 如果不是最后一个模型，等待后重试
            if model_id != models[-1] and attempts < config.max_retries * len(models):
                await asyncio.sleep(current_delay / 1000.0)
                current_delay = min(
                    current_delay * config.backoff_multiplier,
                    config.max_backoff_ms
                )
        
        # 所有模型都失败
        return FallbackResult(
            final_result=None,
            used_model_id=models[-1] if models else "",
            attempts=attempts,
            fallback_chain=fallback_chain,
            total_time_ms=(time.time() - start_time) * 1000,
            success=False,
            error=last_error
        )
    
    async def _execute_parallel(
        self,
        prompt: Any,
        models: List[str],
        call_func: Callable[[str, Any], Awaitable[Any]],
        config: FallbackConfig,
        start_time: float
    ) -> FallbackResult:
        """并行执行Fallback"""
        # 并行调用所有模型
        tasks = [
            self._call_with_timeout(model_id, prompt, call_func, config.timeout_ms)
            for model_id in models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 找到第一个成功的结果
        fallback_chain = []
        attempts = len(models)
        
        for model_id, result in zip(models, results):
            fallback_chain.append(model_id)
            
            if isinstance(result, Exception):
                self._record_failure(model_id, str(result))
                continue
            
            # 成功
            self._record_success(model_id)
            
            return FallbackResult(
                final_result=result,
                used_model_id=model_id,
                attempts=attempts,
                fallback_chain=fallback_chain,
                total_time_ms=(time.time() - start_time) * 1000,
                success=True
            )
        
        return FallbackResult(
            final_result=None,
            used_model_id=models[-1] if models else "",
            attempts=attempts,
            fallback_chain=fallback_chain,
            total_time_ms=(time.time() - start_time) * 1000,
            success=False,
            error="All models failed"
        )
    
    async def _call_with_timeout(
        self,
        model_id: str,
        prompt: Any,
        call_func: Callable[[str, Any], Awaitable[Any]],
        timeout_ms: float
    ) -> Any:
        """带超时的调用"""
        return await asyncio.wait_for(
            call_func(model_id, prompt),
            timeout=timeout_ms / 1000.0
        )
    
    def _filter_available_models(
        self,
        models: List[str],
        config: FallbackConfig
    ) -> List[str]:
        """过滤可用的模型"""
        available = []
        
        for model_id in models:
            state = self._model_states.get(model_id, {})
            consecutive_failures = state.get("consecutive_failures", 0)
            
            # 跳过连续失败的模型
            if consecutive_failures >= config.consecutive_failure_threshold:
                logger.info(
                    f"Skipping {model_id} due to {consecutive_failures} consecutive failures"
                )
                continue
            
            available.append(model_id)
        
        return available
    
    def _record_success(self, model_id: str) -> None:
        """记录成功调用"""
        with self._lock:
            state = self._model_states[model_id]
            state["consecutive_failures"] = 0
            state["last_success"] = datetime.now()
            state["total_successes"] = state.get("total_successes", 0) + 1
    
    def _record_failure(self, model_id: str, error_type: str) -> None:
        """记录失败调用"""
        with self._lock:
            state = self._model_states[model_id]
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            state["last_failure"] = datetime.now()
            state["last_error"] = error_type
            state["total_failures"] = state.get("total_failures", 0) + 1
            
            # 检查是否需要触发恢复检测
            if state["consecutive_failures"] >= 3:
                self._schedule_recovery_check(model_id)
    
    def _schedule_recovery_check(self, model_id: str) -> None:
        """安排恢复检查"""
        if model_id in self._recovery_threads:
            return
        
        thread = threading.Thread(
            target=self._recovery_check_loop,
            args=(model_id,),
            daemon=True
        )
        self._recovery_threads[model_id] = thread
        thread.start()
    
    def _recovery_check_loop(self, model_id: str) -> None:
        """恢复检查循环"""
        logger.info(f"Starting recovery check for {model_id}")
        
        check_interval = 60.0  # 每分钟检查一次
        max_wait = 300.0  # 最多等待5分钟
        waited = 0.0
        
        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval
            
            state = self._model_states.get(model_id, {})
            
            # 检查连续失败数是否减少
            if state.get("consecutive_failures", 0) < 3:
                logger.info(f"Model {model_id} may have recovered")
                break
        
        # 清理线程引用
        with self._lock:
            if model_id in self._recovery_threads:
                del self._recovery_threads[model_id]
    
    def get_model_state(self, model_id: str) -> Dict[str, Any]:
        """获取模型状态"""
        with self._lock:
            state = self._model_states.get(model_id, {}).copy()
            
            # 添加计算属性
            if state:
                state["is_available"] = state.get("consecutive_failures", 0) < 3
                state["total_calls"] = (
                    state.get("total_successes", 0) + 
                    state.get("total_failures", 0)
                )
                if state.get("total_calls", 0) > 0:
                    state["success_rate"] = (
                        state.get("total_successes", 0) / state["total_calls"]
                    )
                else:
                    state["success_rate"] = 1.0
            
            return state
    
    def reset_model_state(self, model_id: str) -> bool:
        """
        重置模型状态。
        
        Args:
            model_id: 模型ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if model_id in self._model_states:
                self._model_states[model_id].clear()
                return True
            return False
    
    def get_chain_stats(self, chain_id: str) -> Dict[str, Any]:
        """获取链统计"""
        chain = self._chains.get(chain_id)
        if not chain:
            return {}
        
        stats = {
            "chain_id": chain_id,
            "chain_name": chain.chain_name,
            "models": [],
            "total_calls": 0,
            "total_successes": 0,
            "total_failures": 0,
        }
        
        for model_id in chain.models:
            state = self.get_model_state(model_id)
            stats["models"].append({
                "model_id": model_id,
                **state
            })
            
            stats["total_calls"] += state.get("total_calls", 0)
            stats["total_successes"] += state.get("total_successes", 0)
            stats["total_failures"] += state.get("total_failures", 0)
        
        if stats["total_calls"] > 0:
            stats["overall_success_rate"] = (
                stats["total_successes"] / stats["total_calls"]
            )
        else:
            stats["overall_success_rate"] = 1.0
        
        return stats
    
    def add_model_to_chain(
        self,
        chain_id: str,
        model_id: str,
        position: Optional[int] = None
    ) -> bool:
        """
        向链中添加模型。
        
        Args:
            chain_id: 链ID
            model_id: 模型ID
            position: 插入位置 (None表示追加到末尾)
            
        Returns:
            是否成功
        """
        with self._lock:
            if chain_id not in self._chains:
                return False
            
            chain = self._chains[chain_id]
            
            if model_id in chain.models:
                return False
            
            if position is None or position >= len(chain.models):
                chain.models.append(model_id)
            else:
                chain.models.insert(position, model_id)
            
            return True
    
    def remove_model_from_chain(self, chain_id: str, model_id: str) -> bool:
        """
        从链中移除模型。
        
        Args:
            chain_id: 链ID
            model_id: 模型ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if chain_id not in self._chains:
                return False
            
            chain = self._chains[chain_id]
            
            if model_id in chain.models:
                chain.models.remove(model_id)
                return True
            
            return False
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            chains_data = {}
            
            for chain_id, chain in self._chains.items():
                chains_data[chain_id] = {
                    "chain_id": chain.chain_id,
                    "chain_name": chain.chain_name,
                    "models": chain.models,
                    "config": {
                        "max_retries": chain.config.max_retries,
                        "retry_delay_ms": chain.config.retry_delay_ms,
                        "timeout_ms": chain.config.timeout_ms,
                        "backoff_multiplier": chain.config.backoff_multiplier,
                        "max_backoff_ms": chain.config.max_backoff_ms,
                        "fallback_strategy": chain.config.fallback_strategy.name,
                        "enable_recovery_check": chain.config.enable_recovery_check,
                    },
                    "recovery_config": {
                        "check_interval_ms": chain.recovery_config.check_interval_ms,
                        "success_threshold": chain.recovery_config.success_threshold,
                        "recovery_timeout_ms": chain.recovery_config.recovery_timeout_ms,
                        "auto_recover": chain.recovery_config.auto_recover,
                    }
                }
            
            return {
                "chains": chains_data,
                "model_states": {
                    model_id: state.copy()
                    for model_id, state in self._model_states.items()
                }
            }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置"""
        with self._lock:
            # 导入链配置
            for chain_id, chain_data in config.get("chains", {}).items():
                chain = FallbackChainConfig(
                    chain_id=chain_data["chain_id"],
                    chain_name=chain_data["chain_name"],
                    models=chain_data["models"],
                    config=FallbackConfig(
                        max_retries=chain_data["config"].get("max_retries", 3),
                        retry_delay_ms=chain_data["config"].get("retry_delay_ms", 1000),
                        timeout_ms=chain_data["config"].get("timeout_ms", 30000),
                        backoff_multiplier=chain_data["config"].get("backoff_multiplier", 2),
                        max_backoff_ms=chain_data["config"].get("max_backoff_ms", 30000),
                        fallback_strategy=FallbackStrategy[
                            chain_data["config"].get("fallback_strategy", "SEQUENTIAL")
                        ],
                        enable_recovery_check=chain_data["config"].get("enable_recovery_check", True),
                    ),
                    recovery_config=RecoveryConfig(
                        check_interval_ms=chain_data["recovery_config"].get("check_interval_ms", 60000),
                        success_threshold=chain_data["recovery_config"].get("success_threshold", 3),
                        recovery_timeout_ms=chain_data["recovery_config"].get("recovery_timeout_ms", 300000),
                        auto_recover=chain_data["recovery_config"].get("auto_recover", True),
                    )
                )
                self._chains[chain_id] = chain
            
            # 导入模型状态
            for model_id, state in config.get("model_states", {}).items():
                self._model_states[model_id] = state.copy()


# 别名
FallbackChain = ModelFallbackChain
