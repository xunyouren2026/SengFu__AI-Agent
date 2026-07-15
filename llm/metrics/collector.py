"""
指标收集器 (Metrics Collector)

收集和分析LLM调用的各种指标。

Author: AGI Team
Version: 1.0.0
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Set, Tuple, Callable
)
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """
    请求指标
    
    Attributes:
        request_id: 请求ID
        model_id: 模型ID
        provider: 提供方
        start_time: 开始时间
        end_time: 结束时间
        latency_ms: 延迟
        input_tokens: 输入Token数
        output_tokens: 输出Token数
        total_tokens: 总Token数
        success: 是否成功
        error: 错误信息
        cache_hit: 是否缓存命中
        channel_id: 渠道ID
        user_id: 用户ID
        metadata: 元数据
    """
    request_id: str
    model_id: str
    provider: str
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    success: bool = True
    error: Optional[str] = None
    cache_hit: bool = False
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cost(self) -> float:
        """估算成本"""
        # 简单估算：$0.001 per 1K tokens
        return self.total_tokens * 0.000001


class MetricsCollector:
    """
    指标收集器
    
    Features:
        - 请求级指标收集
        - 模型级聚合统计
        - 渠道级聚合统计
        - 用户级聚合统计
        - 实时监控
        - 历史趋势分析
    
    Example:
        ```python
        # 创建收集器
        collector = MetricsCollector()
        
        # 记录请求
        collector.record_request(
            request_id="req_123",
            model_id="gpt-4",
            provider="openai",
            latency_ms=1500,
            input_tokens=100,
            output_tokens=200,
            success=True
        )
        
        # 获取统计
        stats = collector.get_stats()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Avg latency: {stats['avg_latency_ms']}ms")
        ```
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self._metrics: List[RequestMetrics] = []
        self._lock = threading.RLock()
        
        # 聚合统计
        self._model_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "total_tokens": 0,
            "total_cost": 0.0,
        })
        
        self._channel_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "avg_latency": 0.0,
        })
        
        self._user_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        })
        
        self._provider_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_requests": 0,
            "avg_latency": 0.0,
            "success_rate": 1.0,
        })
        
        # 配置
        self._max_stored_metrics = 100000
        self._aggregation_interval = 60  # 秒
    
    def record_request(
        self,
        request_id: str,
        model_id: str,
        provider: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: Optional[str] = None,
        cache_hit: bool = False,
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录请求指标。
        
        Args:
            request_id: 请求ID
            model_id: 模型ID
            provider: 提供方
            latency_ms: 延迟
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            success: 是否成功
            error: 错误信息
            cache_hit: 是否缓存命中
            channel_id: 渠道ID
            user_id: 用户ID
            metadata: 元数据
        """
        metrics = RequestMetrics(
            request_id=request_id,
            model_id=model_id,
            provider=provider,
            start_time=datetime.now(),
            end_time=datetime.now(),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            success=success,
            error=error,
            cache_hit=cache_hit,
            channel_id=channel_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics.append(metrics)
            
            # 限制存储大小
            if len(self._metrics) > self._max_stored_metrics:
                self._metrics = self._metrics[-self._max_stored_metrics // 2:]
            
            # 更新聚合统计
            self._update_aggregations(metrics)
    
    def _update_aggregations(self, metrics: RequestMetrics) -> None:
        """更新聚合统计"""
        # 模型统计
        model_stat = self._model_stats[metrics.model_id]
        model_stat["total_requests"] += 1
        if metrics.success:
            model_stat["successful_requests"] += 1
        else:
            model_stat["failed_requests"] += 1
        model_stat["total_latency"] += metrics.latency_ms
        model_stat["total_tokens"] += metrics.total_tokens
        model_stat["total_cost"] += metrics.cost
        
        # 渠道统计
        if metrics.channel_id:
            channel_stat = self._channel_stats[metrics.channel_id]
            channel_stat["total_requests"] += 1
        
        # 用户统计
        if metrics.user_id:
            user_stat = self._user_stats[metrics.user_id]
            user_stat["total_requests"] += 1
            user_stat["total_tokens"] += metrics.total_tokens
            user_stat["total_cost"] += metrics.cost
        
        # Provider统计
        provider_stat = self._provider_stats[metrics.provider]
        provider_stat["total_requests"] += 1
        if metrics.success:
            provider_stat["successful_requests"] += 1
        else:
            provider_stat["failed_requests"] += 1
    
    def get_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取统计信息。
        
        Args:
            start_time: 起始时间
            end_time: 结束时间
            
        Returns:
            统计信息
        """
        with self._lock:
            # 过滤数据
            filtered_metrics = self._metrics
            if start_time:
                filtered_metrics = [m for m in filtered_metrics if m.start_time >= start_time]
            if end_time:
                filtered_metrics = [m for m in filtered_metrics if m.start_time <= end_time]
            
            if not filtered_metrics:
                return self._get_empty_stats()
            
            # 计算统计
            total = len(filtered_metrics)
            successful = sum(1 for m in filtered_metrics if m.success)
            failed = total - successful
            cache_hits = sum(1 for m in filtered_metrics if m.cache_hit)
            
            latencies = [m.latency_ms for m in filtered_metrics]
            tokens = [m.total_tokens for m in filtered_metrics]
            
            return {
                "total_requests": total,
                "successful_requests": successful,
                "failed_requests": failed,
                "success_rate": successful / total if total > 0 else 0,
                "cache_hits": cache_hits,
                "cache_hit_rate": cache_hits / total if total > 0 else 0,
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p50_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": self._percentile(latencies, 99) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "total_tokens": sum(tokens),
                "avg_tokens_per_request": statistics.mean(tokens) if tokens else 0,
                "total_cost": sum(m.cost for m in filtered_metrics),
            }
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """获取空统计"""
        return {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0,
            "avg_latency_ms": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "max_latency_ms": 0,
            "min_latency_ms": 0,
            "total_tokens": 0,
            "avg_tokens_per_request": 0,
            "total_cost": 0,
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """获取模型级统计"""
        with self._lock:
            result = {}
            for model_id, stats in self._model_stats.items():
                total = stats["total_requests"]
                result[model_id] = {
                    "total_requests": total,
                    "successful_requests": stats["successful_requests"],
                    "failed_requests": stats["failed_requests"],
                    "success_rate": stats["successful_requests"] / total if total > 0 else 0,
                    "avg_latency_ms": stats["total_latency"] / total if total > 0 else 0,
                    "total_tokens": stats["total_tokens"],
                    "total_cost": stats["total_cost"],
                }
            return result
    
    def get_provider_stats(self) -> Dict[str, Dict]:
        """获取Provider级统计"""
        with self._lock:
            result = {}
            for provider, stats in self._provider_stats.items():
                total = stats["total_requests"]
                result[provider] = {
                    "total_requests": total,
                    "avg_latency_ms": stats.get("total_latency", 0) / total if total > 0 else 0,
                    "success_rate": stats["successful_requests"] / total if total > 0 else 0,
                }
            return result
    
    def get_channel_stats(self) -> Dict[str, Dict]:
        """获取渠道级统计"""
        with self._lock:
            result = {}
            for channel_id, stats in self._channel_stats.items():
                result[channel_id] = {
                    "total_requests": stats["total_requests"],
                }
            return result
    
    def get_user_stats(self) -> Dict[str, Dict]:
        """获取用户级统计"""
        with self._lock:
            result = {}
            for user_id, stats in self._user_stats.items():
                result[user_id] = {
                    "total_requests": stats["total_requests"],
                    "total_tokens": stats["total_tokens"],
                    "total_cost": stats["total_cost"],
                }
            return result
    
    def get_recent_requests(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        channel_id: Optional[str] = None
    ) -> List[Dict]:
        """获取最近的请求"""
        with self._lock:
            metrics = self._metrics[-limit:]
            
            if model_id:
                metrics = [m for m in metrics if m.model_id == model_id]
            if channel_id:
                metrics = [m for m in metrics if m.channel_id == channel_id]
            
            return [
                {
                    "request_id": m.request_id,
                    "model_id": m.model_id,
                    "provider": m.provider,
                    "latency_ms": m.latency_ms,
                    "total_tokens": m.total_tokens,
                    "success": m.success,
                    "error": m.error,
                    "cache_hit": m.cache_hit,
                    "start_time": m.start_time.isoformat(),
                }
                for m in metrics
            ]
    
    def get_trend(
        self,
        metric: str,
        interval_seconds: int = 60,
        hours: int = 24
    ) -> List[Dict]:
        """
        获取趋势数据。
        
        Args:
            metric: 指标名
            interval_seconds: 间隔秒数
            hours: 小时数
            
        Returns:
            趋势数据
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=hours)
            
            # 过滤数据
            recent = [m for m in self._metrics if m.start_time >= cutoff]
            
            if not recent:
                return []
            
            # 按时间分组
            intervals = []
            current_time = cutoff.replace(second=0, microsecond=0)
            
            while current_time <= now:
                next_time = current_time + timedelta(seconds=interval_seconds)
                
                interval_metrics = [
                    m for m in recent
                    if current_time <= m.start_time < next_time
                ]
                
                if interval_metrics:
                    if metric == "latency":
                        value = statistics.mean([m.latency_ms for m in interval_metrics])
                    elif metric == "requests":
                        value = len(interval_metrics)
                    elif metric == "tokens":
                        value = sum(m.total_tokens for m in interval_metrics)
                    elif metric == "success_rate":
                        success = sum(1 for m in interval_metrics if m.success)
                        value = success / len(interval_metrics)
                    else:
                        value = 0
                    
                    intervals.append({
                        "timestamp": current_time.isoformat(),
                        "value": value,
                        "count": len(interval_metrics),
                    })
                
                current_time = next_time
            
            return intervals
    
    def export_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict:
        """导出指标"""
        with self._lock:
            metrics = self._metrics
            
            if start_time:
                metrics = [m for m in metrics if m.start_time >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.start_time <= end_time]
            
            return {
                "export_time": datetime.now().isoformat(),
                "total_metrics": len(metrics),
                "metrics": [
                    {
                        "request_id": m.request_id,
                        "model_id": m.model_id,
                        "provider": m.provider,
                        "latency_ms": m.latency_ms,
                        "input_tokens": m.input_tokens,
                        "output_tokens": m.output_tokens,
                        "total_tokens": m.total_tokens,
                        "success": m.success,
                        "error": m.error,
                        "cache_hit": m.cache_hit,
                        "channel_id": m.channel_id,
                        "user_id": m.user_id,
                        "start_time": m.start_time.isoformat(),
                        "metadata": m.metadata,
                    }
                    for m in metrics
                ],
                "model_stats": self.get_model_stats(),
                "provider_stats": self.get_provider_stats(),
            }
    
    def clear(self) -> None:
        """清空指标"""
        with self._lock:
            self._metrics.clear()
            self._model_stats.clear()
            self._channel_stats.clear()
            self._user_stats.clear()
            self._provider_stats.clear()
    
    def reset_aggregations(self) -> None:
        """重置聚合统计"""
        with self._lock:
            self._model_stats.clear()
            self._channel_stats.clear()
            self._user_stats.clear()
            self._provider_stats.clear()
