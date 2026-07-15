"""
Token配额管理 (Token Quota Manager)

该模块提供用户和渠道级别的Token配额管理，支持：
- 用户/渠道配额配置
- 实时用量追踪
- 配额预警
- 超额限制

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)


class QuotaType(Enum):
    """配额类型"""
    USER = auto()           # 用户配额
    CHANNEL = auto()        # 渠道配额
    MODEL = auto()          # 模型配额
    GLOBAL = auto()         # 全局配额


@dataclass
class QuotaConfig:
    """
    配额配置
    
    Attributes:
        quota_id: 配额ID
        name: 配额名称
        quota_type: 配额类型
        limit_tokens: Token限额
        limit_requests: 请求限额
        window_seconds: 时间窗口 (秒)
        reset_strategy: 重置策略
    """
    quota_id: str
    name: str
    quota_type: QuotaType = QuotaType.USER
    limit_tokens: int = 1000000  # 默认100万token
    limit_requests: int = 10000   # 默认1万次请求
    window_seconds: float = 2592000.0  # 30天
    reset_strategy: str = "monthly"  # monthly, daily, rolling


@dataclass
class QuotaUsage:
    """
    配额使用情况
    
    Attributes:
        quota_id: 配额ID
        used_tokens: 已用Token数
        used_requests: 已用请求数
        remaining_tokens: 剩余Token
        remaining_requests: 剩余请求
        usage_ratio: 使用比例
        last_reset: 上次重置时间
        window_start: 窗口开始时间
    """
    quota_id: str
    used_tokens: int = 0
    used_requests: int = 0
    remaining_tokens: int = 0
    remaining_requests: int = 0
    usage_ratio: float = 0.0
    last_reset: Optional[datetime] = None
    window_start: Optional[datetime] = None
    
    @property
    def tokens_exceeded(self) -> bool:
        """Token是否超限"""
        return self.remaining_tokens <= 0
    
    @property
    def requests_exceeded(self) -> bool:
        """请求数是否超限"""
        return self.remaining_requests <= 0
    
    @property
    def is_exceeded(self) -> bool:
        """是否超限"""
        return self.tokens_exceeded or self.requests_exceeded
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "quota_id": self.quota_id,
            "used_tokens": self.used_tokens,
            "used_requests": self.used_requests,
            "remaining_tokens": self.remaining_tokens,
            "remaining_requests": self.remaining_requests,
            "usage_ratio": self.usage_ratio,
            "tokens_exceeded": self.tokens_exceeded,
            "requests_exceeded": self.requests_exceeded,
            "last_reset": self.last_reset.isoformat() if self.last_reset else None,
            "window_start": self.window_start.isoformat() if self.window_start else None,
        }


class QuotaLimit:
    """
    配额限制配置
    """
    
    def __init__(
        self,
        quota_id: str,
        config: QuotaConfig
    ):
        self.quota_id = quota_id
        self.config = config
        self.usage = QuotaUsage(
            quota_id=quota_id,
            remaining_tokens=config.limit_tokens,
            remaining_requests=config.limit_requests
        )
        self._lock = threading.Lock()


class QuotaManager:
    """
    Token配额管理器
    
    Features:
        - 多类型配额管理
        - 实时用量追踪
        - 配额预警
        - 超额限制
        - 自动重置
        - 历史统计
    
    Example:
        ```python
        # 创建配额管理器
        manager = QuotaManager()
        
        # 设置用户配额
        manager.set_quota(QuotaConfig(
            quota_id="user123",
            name="用户配额",
            quota_type=QuotaType.USER,
            limit_tokens=100000
        ))
        
        # 检查配额
        can_proceed, reason = manager.check_quota("user123", tokens=100)
        
        # 消费配额
        if can_proceed:
            manager.consume("user123", tokens=100, requests=1)
        ```
    """
    
    def __init__(self):
        """初始化配额管理器"""
        self._quotas: Dict[str, QuotaLimit] = {}
        self._usage_history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.RLock()
        self._alert_callbacks: List[callable] = []
        
        # 启动重置检查
        self._reset_thread: Optional[threading.Thread] = None
        self._running = False
    
    def _start_reset_thread(self) -> None:
        """启动重置检查线程"""
        if self._reset_thread is not None:
            return
        
        self._running = True
        self._reset_thread = threading.Thread(
            target=self._reset_check_loop,
            daemon=True
        )
        self._reset_thread.start()
    
    def _reset_check_loop(self) -> None:
        """重置检查循环"""
        while self._running:
            try:
                self._check_and_reset()
            except Exception as e:
                logger.error(f"Quota reset check error: {e}")
            
            time.sleep(60)  # 每分钟检查
    
    def _check_and_reset(self) -> None:
        """检查并重置配额"""
        with self._lock:
            now = datetime.now()
            
            for quota_id, quota_limit in self._quotas.items():
                config = quota_limit.config
                usage = quota_limit.usage
                
                # 检查是否需要重置
                if self._should_reset(usage, config):
                    self._reset_quota(quota_id)
    
    def _should_reset(self, usage: QuotaUsage, config: QuotaConfig) -> bool:
        """判断是否应该重置"""
        if usage.window_start is None:
            return False
        
        window_end = usage.window_start + timedelta(seconds=config.window_seconds)
        
        if config.reset_strategy == "monthly":
            # 每月重置
            return usage.window_start.month != datetime.now().month
        
        elif config.reset_strategy == "daily":
            # 每日重置
            return usage.window_start.date() != datetime.now().date()
        
        else:  # rolling
            # 滚动重置
            return datetime.now() > window_end
    
    def _reset_quota(self, quota_id: str) -> None:
        """重置配额"""
        if quota_id not in self._quotas:
            return
        
        quota_limit = self._quotas[quota_id]
        config = quota_limit.config
        
        # 记录历史
        self._record_history(quota_id)
        
        # 重置使用量
        quota_limit.usage = QuotaUsage(
            quota_id=quota_id,
            remaining_tokens=config.limit_tokens,
            remaining_requests=config.limit_requests,
            last_reset=datetime.now(),
            window_start=datetime.now()
        )
        
        logger.info(f"Reset quota for {quota_id}")
    
    def _record_history(self, quota_id: str) -> None:
        """记录使用历史"""
        if quota_id not in self._quotas:
            return
        
        usage = self._quotas[quota_id].usage
        
        record = {
            "timestamp": datetime.now(),
            "used_tokens": usage.used_tokens,
            "used_requests": usage.used_requests,
        }
        
        self._usage_history[quota_id].append(record)
        
        # 限制历史大小
        if len(self._usage_history[quota_id]) > 100:
            self._usage_history[quota_id] = self._usage_history[quota_id][-50:]
    
    def set_quota(self, config: QuotaConfig) -> None:
        """
        设置配额。
        
        Args:
            config: 配额配置
        """
        with self._lock:
            quota_limit = QuotaLimit(config.quota_id, config)
            self._quotas[config.quota_id] = quota_limit
            
            # 确保重置检查线程运行
            self._start_reset_thread()
            
            logger.info(f"Set quota for {config.quota_id}: {config.limit_tokens} tokens, {config.limit_requests} requests")
    
    def get_quota(self, quota_id: str) -> Optional[QuotaConfig]:
        """获取配额配置"""
        with self._lock:
            if quota_id in self._quotas:
                return self._quotas[quota_id].config
            return None
    
    def update_quota(self, quota_id: str, **kwargs) -> bool:
        """
        更新配额配置。
        
        Args:
            quota_id: 配额ID
            **kwargs: 要更新的字段
            
        Returns:
            是否更新成功
        """
        with self._lock:
            if quota_id not in self._quotas:
                return False
            
            quota_limit = self._quotas[quota_id]
            
            for key, value in kwargs.items():
                if hasattr(quota_limit.config, key):
                    setattr(quota_limit.config, key, value)
            
            return True
    
    def delete_quota(self, quota_id: str) -> bool:
        """
        删除配额。
        
        Args:
            quota_id: 配额ID
            
        Returns:
            是否删除成功
        """
        with self._lock:
            if quota_id in self._quotas:
                del self._quotas[quota_id]
                return True
            return False
    
    def get_usage(self, quota_id: str) -> Optional[QuotaUsage]:
        """
        获取配额使用情况。
        
        Args:
            quota_id: 配额ID
            
        Returns:
            使用情况
        """
        with self._lock:
            if quota_id not in self._quotas:
                return None
            
            return self._quotas[quota_id].usage
    
    def check_quota(
        self,
        quota_id: str,
        tokens: int = 0,
        requests: int = 1
    ) -> Tuple[bool, str]:
        """
        检查配额是否足够。
        
        Args:
            quota_id: 配额ID
            tokens: 需要的Token数
            requests: 需要的请求数
            
        Returns:
            (是否可以执行, 原因)
        """
        with self._lock:
            # 检查配额是否存在
            if quota_id not in self._quotas:
                # 配额不存在，默认允许
                return True, "No quota configured"
            
            quota_limit = self._quotas[quota_id]
            usage = quota_limit.usage
            
            # 检查Token配额
            if tokens > 0 and usage.remaining_tokens < tokens:
                return False, f"Token quota exceeded: need {tokens}, have {usage.remaining_tokens}"
            
            # 检查请求配额
            if requests > 0 and usage.remaining_requests < requests:
                return False, f"Request quota exceeded: need {requests}, have {usage.remaining_requests}"
            
            return True, "OK"
    
    def consume(
        self,
        quota_id: str,
        tokens: int = 0,
        requests: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        消费配额。
        
        Args:
            quota_id: 配额ID
            tokens: Token使用量
            requests: 请求数
            metadata: 元数据
            
        Returns:
            是否消费成功
        """
        with self._lock:
            if quota_id not in self._quotas:
                # 配额不存在，创建默认配额
                self.set_quota(QuotaConfig(
                    quota_id=quota_id,
                    name=f"Auto-created quota for {quota_id}"
                ))
            
            quota_limit = self._quotas[quota_id]
            config = quota_limit.config
            usage = quota_limit.usage
            
            # 检查是否需要初始化
            if usage.window_start is None:
                usage.window_start = datetime.now()
            
            # 检查配额
            can_proceed, reason = self.check_quota(quota_id, tokens, requests)
            if not can_proceed:
                logger.warning(f"Quota check failed for {quota_id}: {reason}")
                return False
            
            # 更新使用量
            usage.used_tokens += tokens
            usage.used_requests += requests
            usage.remaining_tokens = max(0, config.limit_tokens - usage.used_tokens)
            usage.remaining_requests = max(0, config.limit_requests - usage.used_requests)
            
            # 计算使用比例
            if config.limit_tokens > 0:
                usage.usage_ratio = usage.used_tokens / config.limit_tokens
            
            # 检查是否需要预警
            self._check_alert(quota_id, usage, config)
            
            return True
    
    def _check_alert(
        self,
        quota_id: str,
        usage: QuotaUsage,
        config: QuotaConfig
    ) -> None:
        """检查是否需要预警"""
        alert_thresholds = [0.8, 0.9, 0.95, 1.0]
        
        for threshold in alert_thresholds:
            if usage.usage_ratio >= threshold and usage.usage_ratio < threshold + 0.01:
                logger.warning(
                    f"Quota alert for {quota_id}: "
                    f"{usage.usage_ratio:.1%} used "
                    f"({usage.used_tokens}/{config.limit_tokens} tokens)"
                )
                
                # 触发回调
                for callback in self._alert_callbacks:
                    try:
                        callback(quota_id, usage, config, threshold)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
                
                break
    
    def add_alert_callback(self, callback: callable) -> None:
        """
        添加预警回调。
        
        Args:
            callback: 回调函数
        """
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: callable) -> bool:
        """移除预警回调"""
        with self._lock:
            if callback in self._alert_callbacks:
                self._alert_callbacks.remove(callback)
                return True
            return False
    
    def refund(
        self,
        quota_id: str,
        tokens: int = 0,
        requests: int = 0
    ) -> bool:
        """
        退还配额。
        
        Args:
            quota_id: 配额ID
            tokens: 退还Token数
            requests: 退还请求数
            
        Returns:
            是否退还成功
        """
        with self._lock:
            if quota_id not in self._quotas:
                return False
            
            quota_limit = self._quotas[quota_id]
            usage = quota_limit.usage
            config = quota_limit.config
            
            # 退还
            usage.used_tokens = max(0, usage.used_tokens - tokens)
            usage.used_requests = max(0, usage.used_requests - requests)
            usage.remaining_tokens = min(config.limit_tokens, usage.remaining_tokens + tokens)
            usage.remaining_requests = min(config.limit_requests, usage.remaining_requests + requests)
            
            # 更新使用比例
            if config.limit_tokens > 0:
                usage.usage_ratio = usage.used_tokens / config.limit_tokens
            
            return True
    
    def reset_quota(self, quota_id: str) -> bool:
        """
        手动重置配额。
        
        Args:
            quota_id: 配额ID
            
        Returns:
            是否成功
        """
        with self._lock:
            self._reset_quota(quota_id)
            return True
    
    def get_all_quotas(self) -> List[QuotaConfig]:
        """获取所有配额配置"""
        with self._lock:
            return [q.config for q in self._quotas.values()]
    
    def get_all_usage(self) -> Dict[str, QuotaUsage]:
        """获取所有配额使用情况"""
        with self._lock:
            return {
                quota_id: q.usage
                for quota_id, q in self._quotas.items()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_tokens = sum(q.config.limit_tokens for q in self._quotas.values())
            total_used = sum(q.usage.used_tokens for q in self._quotas.values())
            total_requests = sum(q.usage.used_requests for q in self._quotas.values())
            
            exceeded_count = sum(
                1 for q in self._quotas.values()
                if q.usage.is_exceeded
            )
            
            return {
                "total_quotas": len(self._quotas),
                "total_tokens_quota": total_tokens,
                "total_tokens_used": total_used,
                "total_requests": total_requests,
                "exceeded_count": exceeded_count,
                "usage_distribution": {
                    quota_id: q.usage.usage_ratio
                    for quota_id, q in self._quotas.items()
                }
            }
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置"""
        with self._lock:
            return {
                "quotas": [
                    {
                        **q.config.__dict__,
                        "used_tokens": q.usage.used_tokens,
                        "used_requests": q.usage.used_requests,
                    }
                    for q in self._quotas.values()
                ],
                "history": {
                    quota_id: [
                        {"timestamp": r["timestamp"].isoformat(), "used_tokens": r["used_tokens"]}
                        for r in records
                    ]
                    for quota_id, records in self._usage_history.items()
                }
            }
    
    def import_config(self, config: Dict[str, Any]) -> None:
        """导入配置"""
        with self._lock:
            # 导入配额
            for quota_data in config.get("quotas", []):
                quota_id = quota_data["quota_id"]
                
                self.set_quota(QuotaConfig(
                    quota_id=quota_id,
                    name=quota_data.get("name", quota_id),
                    quota_type=QuotaType[quota_data.get("quota_type", "USER").upper()],
                    limit_tokens=quota_data.get("limit_tokens", 1000000),
                    limit_requests=quota_data.get("limit_requests", 10000),
                    window_seconds=quota_data.get("window_seconds", 2592000),
                    reset_strategy=quota_data.get("reset_strategy", "monthly"),
                ))
                
                # 设置已使用量
                if quota_id in self._quotas:
                    usage = self._quotas[quota_id].usage
                    usage.used_tokens = quota_data.get("used_tokens", 0)
                    usage.used_requests = quota_data.get("used_requests", 0)
                    usage.window_start = datetime.now()
    
    def __del__(self):
        """析构"""
        self._running = False
