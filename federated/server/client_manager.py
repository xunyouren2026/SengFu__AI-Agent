"""
客户端管理器 - 选择策略、心跳检测
"""
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime
from enum import Enum
import random
import math


class ClientStatus(Enum):
    """客户端状态"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    SUSPECTED = "suspected"


class ClientInfo:
    """客户端信息"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.status = ClientStatus.ACTIVE
        self.last_heartbeat: float = datetime.now().timestamp()
        self.last_update_round: int = 0
        self.num_samples: int = 0
        self.compute_capability: float = 1.0  # 相对计算能力
        self.network_latency: float = 0.0  # 网络延迟(ms)
        self.success_count: int = 0
        self.failure_count: int = 0
        self.total_contribution: float = 0.0
        self.metadata: Dict[str, Any] = {}
    
    @property
    def reliability(self) -> float:
        """计算可靠性分数"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.status in (ClientStatus.ACTIVE, ClientStatus.IDLE)
    
    def update_heartbeat(self) -> None:
        """更新心跳时间"""
        self.last_heartbeat = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'client_id': self.client_id,
            'status': self.status.value,
            'last_heartbeat': self.last_heartbeat,
            'last_update_round': self.last_update_round,
            'num_samples': self.num_samples,
            'compute_capability': self.compute_capability,
            'network_latency': self.network_latency,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_contribution': self.total_contribution,
            'reliability': self.reliability
        }


class SelectionStrategy(Enum):
    """选择策略"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    IMPORTANCE = "importance"
    RELIABILITY = "reliability"
    COMPUTE_AWARE = "compute_aware"
    POWER_OF_CHOICE = "power_of_choice"


class ClientSelector:
    """客户端选择器"""
    
    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.RANDOM,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        participation_ratio: float = 0.1
    ):
        self.strategy = strategy
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.participation_ratio = participation_ratio
        self._round_robin_index: int = 0
        self._power_d: int = 2  # Power of Choice参数
    
    def select(
        self,
        clients: Dict[str, ClientInfo],
        round_num: int
    ) -> List[str]:
        """
        选择参与本轮训练的客户端
        
        Args:
            clients: 所有客户端信息
            round_num: 当前轮次
        
        Returns:
            选中的客户端ID列表
        """
        # 过滤可用客户端
        available = {
            cid: info for cid, info in clients.items()
            if info.is_available
        }
        
        if len(available) < self.min_clients:
            return list(available.keys())
        
        # 计算目标数量
        target_count = max(
            self.min_clients,
            min(
                len(available),
                int(len(available) * self.participation_ratio)
            )
        )
        
        if self.max_clients:
            target_count = min(target_count, self.max_clients)
        
        # 根据策略选择
        if self.strategy == SelectionStrategy.RANDOM:
            return self._select_random(available, target_count)
        elif self.strategy == SelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(available, target_count)
        elif self.strategy == SelectionStrategy.IMPORTANCE:
            return self._select_importance(available, target_count)
        elif self.strategy == SelectionStrategy.RELIABILITY:
            return self._select_reliability(available, target_count)
        elif self.strategy == SelectionStrategy.COMPUTE_AWARE:
            return self._select_compute_aware(available, target_count)
        elif self.strategy == SelectionStrategy.POWER_OF_CHOICE:
            return self._select_power_of_choice(available, target_count)
        else:
            return self._select_random(available, target_count)
    
    def _select_random(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """随机选择"""
        client_ids = list(clients.keys())
        return random.sample(client_ids, min(count, len(client_ids)))
    
    def _select_round_robin(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """轮询选择"""
        client_ids = sorted(clients.keys())
        selected = []
        
        for _ in range(count):
            if self._round_robin_index >= len(client_ids):
                self._round_robin_index = 0
            selected.append(client_ids[self._round_robin_index])
            self._round_robin_index += 1
        
        return selected
    
    def _select_importance(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """重要性采样 - 基于数据量"""
        # 按样本数加权采样
        total_samples = sum(c.num_samples for c in clients.values())
        
        if total_samples == 0:
            return self._select_random(clients, count)
        
        weights = {
            cid: c.num_samples / total_samples
            for cid, c in clients.items()
        }
        
        return self._weighted_sample(weights, count)
    
    def _select_reliability(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """可靠性优先选择"""
        sorted_clients = sorted(
            clients.items(),
            key=lambda x: x[1].reliability,
            reverse=True
        )
        
        return [cid for cid, _ in sorted_clients[:count]]
    
    def _select_compute_aware(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """计算能力感知选择"""
        # 综合考虑计算能力和网络延迟
        def score(info: ClientInfo) -> float:
            latency_factor = 1.0 / (1.0 + info.network_latency / 100.0)
            return info.compute_capability * latency_factor * info.reliability
        
        sorted_clients = sorted(
            clients.items(),
            key=lambda x: score(x[1]),
            reverse=True
        )
        
        return [cid for cid, _ in sorted_clients[:count]]
    
    def _select_power_of_choice(
        self,
        clients: Dict[str, ClientInfo],
        count: int
    ) -> List[str]:
        """
        Power of Choice选择
        
        每次随机选择d个候选，从中选择损失最大的
        """
        selected = []
        client_ids = list(clients.keys())
        
        # 简化实现：使用贡献度作为损失代理
        for _ in range(count):
            if len(client_ids) < self._power_d:
                candidates = client_ids
            else:
                candidates = random.sample(client_ids, self._power_d)
            
            # 选择贡献最小的（假设需要更多训练）
            best = min(
                candidates,
                key=lambda cid: clients[cid].total_contribution
            )
            selected.append(best)
            client_ids.remove(best)
        
        return selected
    
    def _weighted_sample(
        self,
        weights: Dict[str, float],
        count: int
    ) -> List[str]:
        """加权采样"""
        selected = []
        remaining = dict(weights)
        
        for _ in range(min(count, len(remaining))):
            total = sum(remaining.values())
            r = random.uniform(0, total)
            
            cumulative = 0.0
            for cid, w in remaining.items():
                cumulative += w
                if r <= cumulative:
                    selected.append(cid)
                    del remaining[cid]
                    break
        
        return selected


class HeartbeatMonitor:
    """心跳监控器"""
    
    def __init__(
        self,
        timeout_seconds: float = 30.0,
        check_interval: float = 5.0,
        suspicion_threshold: int = 3
    ):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        self.suspicion_threshold = suspicion_threshold
        
        self._missed_heartbeats: Dict[str, int] = {}
        self._callbacks: List[Callable[[str, ClientStatus], None]] = []
    
    def register_callback(
        self,
        callback: Callable[[str, ClientStatus], None]
    ) -> None:
        """注册状态变化回调"""
        self._callbacks.append(callback)
    
    def check(
        self,
        clients: Dict[str, ClientInfo]
    ) -> Dict[str, ClientStatus]:
        """
        检查所有客户端心跳状态
        
        Returns:
            状态变化的客户端映射
        """
        current_time = datetime.now().timestamp()
        status_changes: Dict[str, ClientStatus] = {}
        
        for client_id, info in clients.items():
            elapsed = current_time - info.last_heartbeat
            old_status = info.status
            
            if elapsed > self.timeout_seconds:
                # 超时
                self._missed_heartbeats[client_id] = \
                    self._missed_heartbeats.get(client_id, 0) + 1
                
                if self._missed_heartbeats[client_id] >= self.suspicion_threshold:
                    new_status = ClientStatus.OFFLINE
                else:
                    new_status = ClientStatus.SUSPECTED
            else:
                # 正常
                self._missed_heartbeats[client_id] = 0
                if info.status == ClientStatus.BUSY:
                    new_status = ClientStatus.BUSY
                else:
                    new_status = ClientStatus.ACTIVE
            
            if new_status != old_status:
                info.status = new_status
                status_changes[client_id] = new_status
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        callback(client_id, new_status)
                    except Exception:
                        pass
        
        return status_changes
    
    def get_active_count(self, clients: Dict[str, ClientInfo]) -> int:
        """获取活跃客户端数量"""
        return sum(1 for c in clients.values() if c.status == ClientStatus.ACTIVE)


class ClientManager:
    """
    客户端管理器
    
    功能:
    - 客户端注册和注销
    - 心跳检测
    - 客户端选择
    - 状态跟踪
    """
    
    def __init__(
        self,
        selection_strategy: SelectionStrategy = SelectionStrategy.RANDOM,
        heartbeat_timeout: float = 30.0,
        min_clients: int = 1,
        max_clients: Optional[int] = None
    ):
        self._clients: Dict[str, ClientInfo] = {}
        self._selector = ClientSelector(
            strategy=selection_strategy,
            min_clients=min_clients,
            max_clients=max_clients
        )
        self._heartbeat_monitor = HeartbeatMonitor(
            timeout_seconds=heartbeat_timeout
        )
        
        self._round_num: int = 0
        self._selected_clients: Set[str] = set()
    
    def register(
        self,
        client_id: str,
        num_samples: int = 0,
        compute_capability: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册客户端
        
        Args:
            client_id: 客户端唯一标识
            num_samples: 本地数据样本数
            compute_capability: 计算能力
            metadata: 额外元数据
        
        Returns:
            是否注册成功
        """
        if client_id in self._clients:
            # 已存在，更新信息
            info = self._clients[client_id]
            info.num_samples = num_samples
            info.compute_capability = compute_capability
            info.update_heartbeat()
            if metadata:
                info.metadata.update(metadata)
            return True
        
        # 新客户端
        info = ClientInfo(client_id)
        info.num_samples = num_samples
        info.compute_capability = compute_capability
        if metadata:
            info.metadata = metadata
        
        self._clients[client_id] = info
        return True
    
    def unregister(self, client_id: str) -> bool:
        """注销客户端"""
        if client_id in self._clients:
            del self._clients[client_id]
            self._selected_clients.discard(client_id)
            return True
        return False
    
    def heartbeat(self, client_id: str) -> bool:
        """
        接收心跳
        
        Args:
            client_id: 客户端ID
        
        Returns:
            是否成功
        """
        if client_id not in self._clients:
            return False
        
        self._clients[client_id].update_heartbeat()
        return True
    
    def select_clients(self) -> List[str]:
        """
        为当前轮选择客户端
        
        Returns:
            选中的客户端ID列表
        """
        selected = self._selector.select(self._clients, self._round_num)
        self._selected_clients = set(selected)
        return selected
    
    def mark_client_busy(self, client_id: str) -> None:
        """标记客户端为忙碌状态"""
        if client_id in self._clients:
            self._clients[client_id].status = ClientStatus.BUSY
    
    def mark_client_done(
        self,
        client_id: str,
        success: bool = True,
        contribution: float = 0.0
    ) -> None:
        """标记客户端完成训练"""
        if client_id not in self._clients:
            return
        
        info = self._clients[client_id]
        info.status = ClientStatus.ACTIVE
        info.last_update_round = self._round_num
        
        if success:
            info.success_count += 1
        else:
            info.failure_count += 1
        
        info.total_contribution += contribution
    
    def advance_round(self) -> int:
        """推进到下一轮"""
        self._round_num += 1
        self._selected_clients.clear()
        return self._round_num
    
    def check_heartbeats(self) -> Dict[str, ClientStatus]:
        """检查心跳状态"""
        return self._heartbeat_monitor.check(self._clients)
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息"""
        return self._clients.get(client_id)
    
    def get_all_clients(self) -> Dict[str, ClientInfo]:
        """获取所有客户端"""
        return dict(self._clients)
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端列表"""
        return [
            cid for cid, info in self._clients.items()
            if info.is_available
        ]
    
    def get_selected_clients(self) -> Set[str]:
        """获取当前选中的客户端"""
        return self._selected_clients.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active = sum(1 for c in self._clients.values() if c.status == ClientStatus.ACTIVE)
        busy = sum(1 for c in self._clients.values() if c.status == ClientStatus.BUSY)
        offline = sum(1 for c in self._clients.values() if c.status == ClientStatus.OFFLINE)
        
        total_samples = sum(c.num_samples for c in self._clients.values())
        avg_reliability = (
            sum(c.reliability for c in self._clients.values()) / len(self._clients)
            if self._clients else 0
        )
        
        return {
            'total_clients': len(self._clients),
            'active_clients': active,
            'busy_clients': busy,
            'offline_clients': offline,
            'current_round': self._round_num,
            'selected_count': len(self._selected_clients),
            'total_samples': total_samples,
            'average_reliability': avg_reliability
        }
    
    def set_selection_strategy(self, strategy: SelectionStrategy) -> None:
        """设置选择策略"""
        self._selector.strategy = strategy
    
    def set_participation_ratio(self, ratio: float) -> None:
        """设置参与比例"""
        self._selector.participation_ratio = ratio
