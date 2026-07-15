"""
异步聚合器 - 处理落后客户端
"""
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from enum import Enum
import math
import copy


class UpdateStatus(Enum):
    """更新状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    APPLIED = "applied"
    STALE = "stale"
    DISCARDED = "discarded"


class ClientUpdate:
    """客户端更新"""
    
    def __init__(
        self,
        client_id: str,
        model_params: Dict[str, Any],
        num_samples: int,
        round_num: int,
        timestamp: Optional[float] = None
    ):
        self.client_id = client_id
        self.model_params = model_params
        self.num_samples = num_samples
        self.round_num = round_num
        self.timestamp = timestamp or datetime.now().timestamp()
        self.status = UpdateStatus.PENDING
        self.staleness: int = 0
        self.weight: float = 1.0
    
    def mark_stale(self, current_round: int) -> None:
        """标记为过时"""
        self.staleness = current_round - self.round_num
        self.status = UpdateStatus.STALE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'round_num': self.round_num,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'staleness': self.staleness,
            'weight': self.weight
        }


class StalenessPolicy(Enum):
    """过时策略"""
    CONSTANT = "constant"  # 常数权重
    LINEAR = "linear"  # 线性衰减
    EXPONENTIAL = "exponential"  # 指数衰减
    THRESHOLD = "threshold"  # 阈值截断


class StalenessWeighter:
    """过时权重计算器"""
    
    def __init__(
        self,
        policy: StalenessPolicy = StalenessPolicy.LINEAR,
        decay_rate: float = 0.5,
        threshold: int = 5
    ):
        self.policy = policy
        self.decay_rate = decay_rate
        self.threshold = threshold
    
    def compute_weight(self, staleness: int) -> float:
        """
        计算过时权重
        
        Args:
            staleness: 过时程度 (当前轮次 - 更新轮次)
        
        Returns:
            权重值
        """
        if staleness <= 0:
            return 1.0
        
        if self.policy == StalenessPolicy.CONSTANT:
            return 1.0
        
        elif self.policy == StalenessPolicy.LINEAR:
            return max(0.0, 1.0 - self.decay_rate * staleness)
        
        elif self.policy == StalenessPolicy.EXPONENTIAL:
            return math.exp(-self.decay_rate * staleness)
        
        elif self.policy == StalenessPolicy.THRESHOLD:
            return 0.0 if staleness > self.threshold else 1.0
        
        return 1.0


class BufferManager:
    """更新缓冲区管理器"""
    
    def __init__(
        self,
        max_buffer_size: int = 1000,
        max_staleness: int = 10,
        cleanup_interval: int = 50
    ):
        self.max_buffer_size = max_buffer_size
        self.max_staleness = max_staleness
        self.cleanup_interval = cleanup_interval
        
        self._buffer: Dict[str, ClientUpdate] = {}  # client_id -> latest update
        self._pending: List[ClientUpdate] = []
        self._processed_count: int = 0
    
    def add_update(self, update: ClientUpdate) -> bool:
        """
        添加更新到缓冲区
        
        Returns:
            是否接受更新
        """
        client_id = update.client_id
        
        # 检查是否已有更新的更新
        if client_id in self._buffer:
            existing = self._buffer[client_id]
            if existing.round_num >= update.round_num:
                update.status = UpdateStatus.DISCARDED
                return False
        
        # 添加到缓冲区
        self._buffer[client_id] = update
        self._pending.append(update)
        
        # 清理
        if len(self._pending) > self.max_buffer_size:
            self._cleanup()
        
        return True
    
    def get_pending_updates(
        self,
        current_round: int
    ) -> List[ClientUpdate]:
        """
        获取待处理的更新
        
        Args:
            current_round: 当前轮次
        
        Returns:
            待处理的更新列表
        """
        pending = []
        
        for update in self._pending:
            if update.status == UpdateStatus.PENDING:
                # 计算过时程度
                update.staleness = current_round - update.round_num
                
                if update.staleness > self.max_staleness:
                    update.status = UpdateStatus.STALE
                else:
                    pending.append(update)
        
        return pending
    
    def mark_processed(self, client_id: str) -> None:
        """标记更新已处理"""
        if client_id in self._buffer:
            self._buffer[client_id].status = UpdateStatus.APPLIED
            self._processed_count += 1
    
    def _cleanup(self) -> None:
        """清理缓冲区"""
        # 移除已处理和过时的更新
        self._pending = [
            u for u in self._pending
            if u.status in (UpdateStatus.PENDING, UpdateStatus.PROCESSING)
        ]
    
    def get_buffer_size(self) -> int:
        """获取缓冲区大小"""
        return len(self._pending)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        pending = sum(1 for u in self._pending if u.status == UpdateStatus.PENDING)
        applied = sum(1 for u in self._pending if u.status == UpdateStatus.APPLIED)
        stale = sum(1 for u in self._pending if u.status == UpdateStatus.STALE)
        
        return {
            'total_updates': len(self._pending),
            'pending': pending,
            'applied': applied,
            'stale': stale,
            'unique_clients': len(self._buffer),
            'processed_count': self._processed_count
        }


class AsyncAggregator:
    """
    异步聚合器
    
    处理客户端异步更新，支持:
    - 过时更新处理
    - 缓冲区管理
    - 部分聚合
    - 超时处理
    """
    
    def __init__(
        self,
        staleness_policy: StalenessPolicy = StalenessPolicy.LINEAR,
        staleness_decay: float = 0.5,
        max_staleness: int = 10,
        min_updates_for_aggregation: int = 1,
        aggregation_timeout: float = 60.0,
        buffer_size: int = 1000
    ):
        self._staleness_weighter = StalenessWeighter(
            policy=staleness_policy,
            decay_rate=staleness_decay,
            threshold=max_staleness
        )
        self._buffer = BufferManager(
            max_buffer_size=buffer_size,
            max_staleness=max_staleness
        )
        
        self.min_updates = min_updates_for_aggregation
        self.aggregation_timeout = aggregation_timeout
        
        self._current_round: int = 0
        self._global_params: Optional[Dict[str, Any]] = None
        self._last_aggregation_time: float = 0.0
        self._aggregation_history: List[Dict[str, Any]] = []
    
    def set_global_model(self, params: Dict[str, Any]) -> None:
        """设置全局模型"""
        self._global_params = copy.deepcopy(params)
    
    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """获取全局模型"""
        return copy.deepcopy(self._global_params) if self._global_params else None
    
    def receive_update(
        self,
        client_id: str,
        model_params: Dict[str, Any],
        num_samples: int,
        round_num: int
    ) -> bool:
        """
        接收客户端更新
        
        Args:
            client_id: 客户端ID
            model_params: 模型参数
            num_samples: 样本数
            round_num: 客户端训练时的轮次
        
        Returns:
            是否接受更新
        """
        update = ClientUpdate(
            client_id=client_id,
            model_params=model_params,
            num_samples=num_samples,
            round_num=round_num
        )
        
        return self._buffer.add_update(update)
    
    def try_aggregate(
        self,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        尝试聚合
        
        Args:
            force: 是否强制聚合（忽略最小更新数限制）
        
        Returns:
            新的全局模型，如果未聚合则返回None
        """
        pending = self._buffer.get_pending_updates(self._current_round)
        
        if len(pending) < self.min_updates and not force:
            return None
        
        # 检查超时
        current_time = datetime.now().timestamp()
        if not force and (current_time - self._last_aggregation_time) < self.aggregation_timeout:
            if len(pending) < self.min_updates * 2:  # 等待更多更新
                return None
        
        # 计算权重
        for update in pending:
            update.weight = self._staleness_weighter.compute_weight(update.staleness)
        
        # 执行聚合
        new_params = self._aggregate_updates(pending)
        
        if new_params is not None:
            self._global_params = new_params
            self._current_round += 1
            self._last_aggregation_time = current_time
            
            # 标记已处理
            for update in pending:
                self._buffer.mark_processed(update.client_id)
            
            # 记录历史
            self._aggregation_history.append({
                'round': self._current_round,
                'num_updates': len(pending),
                'avg_staleness': sum(u.staleness for u in pending) / len(pending),
                'total_samples': sum(u.num_samples for u in pending),
                'timestamp': current_time
            })
        
        return new_params
    
    def _aggregate_updates(
        self,
        updates: List[ClientUpdate]
    ) -> Optional[Dict[str, Any]]:
        """
        聚合更新
        
        使用加权平均，权重考虑样本数和过时程度
        """
        if not updates:
            return None
        
        # 计算总权重
        total_weight = sum(
            u.weight * u.num_samples for u in updates
        )
        
        if total_weight == 0:
            return None
        
        # 收集所有参数键
        all_keys: Set[str] = set()
        for update in updates:
            all_keys.update(update.model_params.keys())
        
        # 加权平均
        aggregated: Dict[str, Any] = {}
        
        for key in all_keys:
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for update in updates:
                if key not in update.model_params:
                    continue
                
                value = update.model_params[key]
                if isinstance(value, (int, float)):
                    w = update.weight * update.num_samples
                    weighted_sum += w * value
                    weight_sum += w
                elif isinstance(value, list):
                    if key not in aggregated:
                        aggregated[key] = [0.0] * len(value)
                        weight_sum = 0.0
                    
                    w = update.weight * update.num_samples
                    for i, v in enumerate(value):
                        if isinstance(v, (int, float)):
                            aggregated[key][i] += w * v
                    weight_sum += w
            
            if key not in aggregated and weight_sum > 0:
                aggregated[key] = weighted_sum / weight_sum
            elif key in aggregated and weight_sum > 0:
                aggregated[key] = [v / weight_sum for v in aggregated[key]]
        
        return aggregated
    
    def get_current_round(self) -> int:
        """获取当前轮次"""
        return self._current_round
    
    def get_pending_count(self) -> int:
        """获取待处理更新数"""
        return len(self._buffer.get_pending_updates(self._current_round))
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        buffer_stats = self._buffer.get_statistics()
        
        avg_staleness = 0.0
        if self._aggregation_history:
            recent = self._aggregation_history[-10:]
            avg_staleness = sum(h['avg_staleness'] for h in recent) / len(recent)
        
        return {
            'current_round': self._current_round,
            'buffer_stats': buffer_stats,
            'avg_staleness': avg_staleness,
            'total_aggregations': len(self._aggregation_history),
            'staleness_policy': self._staleness_weighter.policy.value
        }


class SemiSynchronousAggregator(AsyncAggregator):
    """
    半同步聚合器
    
    等待一定比例的客户端更新后再聚合
    """
    
    def __init__(
        self,
        sync_ratio: float = 0.5,
        max_wait_time: float = 30.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sync_ratio = sync_ratio
        self.max_wait_time = max_wait_time
        
        self._expected_clients: Set[str] = set()
        self._received_clients: Set[str] = set()
        self._round_start_time: float = 0.0
    
    def start_round(self, expected_clients: Set[str]) -> None:
        """开始新一轮"""
        self._expected_clients = expected_clients.copy()
        self._received_clients.clear()
        self._round_start_time = datetime.now().timestamp()
    
    def receive_update(
        self,
        client_id: str,
        model_params: Dict[str, Any],
        num_samples: int,
        round_num: int
    ) -> bool:
        """接收更新"""
        result = super().receive_update(
            client_id, model_params, num_samples, round_num
        )
        
        if result:
            self._received_clients.add(client_id)
        
        return result
    
    def should_aggregate(self) -> Tuple[bool, str]:
        """
        检查是否应该聚合
        
        Returns:
            (是否聚合, 原因)
        """
        if not self._expected_clients:
            return False, "no_expected_clients"
        
        # 检查比例
        ratio = len(self._received_clients) / len(self._expected_clients)
        
        if ratio >= self.sync_ratio:
            return True, "ratio_reached"
        
        # 检查超时
        elapsed = datetime.now().timestamp() - self._round_start_time
        if elapsed >= self.max_wait_time:
            return True, "timeout"
        
        return False, "waiting"
    
    def get_sync_progress(self) -> Dict[str, Any]:
        """获取同步进度"""
        expected = len(self._expected_clients)
        received = len(self._received_clients)
        
        return {
            'expected_clients': expected,
            'received_clients': received,
            'sync_ratio': received / expected if expected > 0 else 0.0,
            'target_ratio': self.sync_ratio,
            'missing_clients': list(self._expected_clients - self._received_clients)
        }
