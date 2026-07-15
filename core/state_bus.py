"""
状态总线模块 - 胜复学架构的通信中枢

状态总线(StateBus)是连接五层架构(感知层、认知层、决策层、执行层、安全层)的核心通信组件，
负责事件发布/订阅、状态系数收集和五层间消息路由。

五层架构与状态系数:
- 感知层(Perception): 收集原始输入数据
- 认知层(Cognition): 处理和理解信息
- 决策层(Decision): 制定行动计划
- 执行层(Execution): 输出动作和结果
- 安全层(Safety): 监控和保障系统安全

状态系数:
- 胜值(Swing): 执行层置信度，反映决策质量
- 郁值(Halt): 系统瓶颈程度，反映训练/推理中的阻滞
- 复值(Balance): 调节强度，反映系统自我调节能力
- 发值(Release): 爆发强度，反映系统输出能力
- 道值(Dao): 安全分数，反映系统安全状态

事件类型:
- SWING_HIGH: 胜值过高警告(可能过亢)
- HALT_WARNING: 郁值预警(系统出现瓶颈)
- HALT_ERUPT: 郁值爆发(系统严重阻滞)
- BALANCE_INTROSPECT: 复值内省(系统自我调节)
- RELEASE_ACTION: 发值释放(执行动作)
- SAFETY_VIOLATION: 安全违规

示例:
    >>> from agi_unified_framework.core.state_bus import StateBus, StateCoefficient
    >>> bus = StateBus()
    >>> 
    >>> # 订阅事件
    >>> def on_halt_warning(data):
    ...     print(f"郁值预警: {data}")
    >>> bus.subscribe("HALT_WARNING", on_halt_warning)
    >>> 
    >>> # 更新状态系数
    >>> bus.update_coefficient("swing", 0.85)
    >>> bus.update_coefficient("halt", 0.75)
    >>> 
    >>> # 发布事件
    >>> bus.publish("HALT_WARNING", {"level": "high", "value": 0.75})
    >>> 
    >>> # 获取当前状态
    >>> state = bus.get_current_state()
    >>> print(f"当前郁值: {state.halt}")
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

# ============================================================================
# 类型定义
# ============================================================================

T = TypeVar("T")
EventCallback = Callable[[str, Any], None]


class EventType(Enum):
    """
    系统事件类型枚举
    
    定义胜复学架构中的核心事件类型，用于状态总线的事件分发。
    """
    SWING_HIGH = "swing_high"           # 胜值过高警告(置信度过高可能导致过亢)
    HALT_WARNING = "halt_warning"       # 郁值预警(系统出现轻微瓶颈)
    HALT_ERUPT = "halt_erupt"           # 郁值爆发(系统严重阻滞)
    BALANCE_INTROSPECT = "balance_introspect"  # 复值内省(系统自我调节)
    RELEASE_ACTION = "release_action"   # 发值释放(执行动作)
    SAFETY_VIOLATION = "safety_violation"  # 安全违规
    STATE_CHANGE = "state_change"       # 状态变化
    LAYER_MESSAGE = "layer_message"     # 层间消息
    SYSTEM_RESET = "system_reset"       # 系统重置
    
    def __str__(self) -> str:
        return self.value


class LayerType(Enum):
    """
    五层架构层类型枚举
    
    定义胜复学五层架构的各层类型。
    """
    PERCEPTION = "perception"    # 感知层 - 输入处理
    COGNITION = "cognition"      # 认知层 - 理解推理
    DECISION = "decision"        # 决策层 - 计划制定
    EXECUTION = "execution"      # 执行层 - 动作输出
    SAFETY = "safety"            # 安全层 - 安全保障
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class StateCoefficient:
    """
    状态系数数据类
    
    存储胜复学架构的五个核心状态系数，反映系统整体健康状态。
    
    Attributes:
        swing: 胜值 (0.0-1.0) - 执行层置信度，高表示决策自信
        halt: 郁值 (0.0-1.0) - 系统瓶颈程度，高表示存在阻滞
        balance: 复值 (0.0-1.0) - 调节强度，高表示系统自适应能力强
        release: 发值 (0.0-1.0) - 爆发强度，高表示输出能力强
        dao: 道值 (0.0-1.0) - 安全分数，高表示系统安全
        timestamp: 时间戳 - 记录状态创建时间
    
    示例:
        >>> state = StateCoefficient(
        ...     swing=0.85,  # 高置信度
        ...     halt=0.2,    # 低郁值，系统流畅
        ...     balance=0.7, # 较好的自我调节
        ...     release=0.6, # 适中的输出强度
        ...     dao=0.95     # 高安全分数
        ... )
    """
    swing: float = 0.5    # 胜值 - 置信度
    halt: float = 0.5     # 郁值 - 瓶颈程度
    balance: float = 0.5  # 复值 - 调节强度
    release: float = 0.5  # 发值 - 爆发强度
    dao: float = 1.0      # 道值 - 安全分数
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """验证并限制系数范围在[0, 1]之间"""
        self.swing = max(0.0, min(1.0, self.swing))
        self.halt = max(0.0, min(1.0, self.halt))
        self.balance = max(0.0, min(1.0, self.balance))
        self.release = max(0.0, min(1.0, self.release))
        self.dao = max(0.0, min(1.0, self.dao))
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式"""
        return {
            "swing": self.swing,
            "halt": self.halt,
            "balance": self.balance,
            "release": self.release,
            "dao": self.dao,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "StateCoefficient":
        """从字典创建实例"""
        return cls(
            swing=data.get("swing", 0.5),
            halt=data.get("halt", 0.5),
            balance=data.get("balance", 0.5),
            release=data.get("release", 0.5),
            dao=data.get("dao", 1.0),
            timestamp=data.get("timestamp", time.time())
        )
    
    def get_health_score(self) -> float:
        """
        计算系统健康分数
        
        基于五值计算综合健康度，高分表示系统运行良好。
        
        Returns:
            健康分数 (0.0-1.0)
        """
        # 胜值和复值越高越好，郁值越低越好
        # 道值必须高，发值适中最好
        health = (
            self.swing * 0.2 +      # 胜值权重
            (1 - self.halt) * 0.3 + # 郁值反向权重
            self.balance * 0.2 +    # 复值权重
            (1 - abs(self.release - 0.5) * 2) * 0.1 +  # 发值适中权重
            self.dao * 0.2          # 道值权重
        )
        return max(0.0, min(1.0, health))


@dataclass
class Event:
    """
    事件数据类
    
    封装状态总线中传输的事件信息。
    
    Attributes:
        event_type: 事件类型
        data: 事件数据
        source_layer: 源层
        target_layer: 目标层(可选，None表示广播)
        timestamp: 时间戳
        priority: 优先级(0-10，数字越小优先级越高)
    """
    event_type: str
    data: Any
    source_layer: str = "unknown"
    target_layer: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    priority: int = 5
    
    def __lt__(self, other: "Event") -> bool:
        """用于优先级队列比较"""
        return self.priority < other.priority


@dataclass
class LayerState:
    """
    层状态数据类
    
    记录单个层的运行状态。
    
    Attributes:
        layer_type: 层类型
        is_active: 是否活跃
        last_update: 最后更新时间
        metrics: 层指标数据
    """
    layer_type: LayerType
    is_active: bool = True
    last_update: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 状态总线核心类
# ============================================================================

class StateBus:
    """
    状态总线 - 胜复学架构的通信中枢
    
    状态总线是连接五层架构的核心组件，提供事件发布/订阅、状态系数收集
    和五层间消息路由功能。
    
    Attributes:
        _subscribers: 事件订阅者字典
        _state_coefficient: 当前状态系数
        _history: 历史状态记录
        _layer_states: 各层状态
        _lock: 线程锁
        _running: 运行状态标志
    
    示例:
        >>> bus = StateBus(history_window=1000)
        >>> 
        >>> # 定义回调函数
        >>> def on_safety_violation(event_type, data):
        ...     print(f"安全违规: {data}")
        >>> 
        >>> # 订阅事件
        >>> bus.subscribe(EventType.SAFETY_VIOLATION, on_safety_violation)
        >>> 
        >>> # 更新状态
        >>> bus.update_coefficient("dao", 0.5)  # 降低安全分数
        >>> 
        >>> # 发布事件
        >>> bus.publish(EventType.SAFETY_VIOLATION, {"reason": "low_dao"})
    """
    
    def __init__(self, history_window: int = 1000):
        """
        初始化状态总线
        
        Args:
            history_window: 历史状态记录窗口大小
        """
        self._subscribers: Dict[str, Set[EventCallback]] = {}
        self._state_coefficient: StateCoefficient = StateCoefficient()
        self._history: deque = deque(maxlen=history_window)
        self._layer_states: Dict[LayerType, LayerState] = {
            layer: LayerState(layer_type=layer)
            for layer in LayerType
        }
        self._lock = threading.RLock()
        self._running = True
        self._event_queue: deque = deque(maxlen=10000)
        self._event_thread: Optional[threading.Thread] = None
        
        # 启动事件处理线程
        self._start_event_processor()
    
    def _start_event_processor(self) -> None:
        """启动后台事件处理线程"""
        def process_events():
            while self._running:
                try:
                    if self._event_queue:
                        with self._lock:
                            if self._event_queue:
                                event = self._event_queue.popleft()
                                self._dispatch_event(event)
                    else:
                        time.sleep(0.001)  # 1ms休眠避免CPU占用过高
                except Exception as e:
                    print(f"事件处理错误: {e}")
        
        self._event_thread = threading.Thread(target=process_events, daemon=True)
        self._event_thread.start()
    
    def subscribe(self, event_type: Union[str, EventType], 
                  callback: EventCallback) -> None:
        """
        订阅事件
        
        注册回调函数以接收特定类型的事件通知。
        
        Args:
            event_type: 事件类型(字符串或EventType枚举)
            callback: 回调函数，接收(event_type, data)参数
        
        示例:
            >>> def on_halt(data):
            ...     print(f"郁值: {data}")
            >>> bus.subscribe(EventType.HALT_WARNING, on_halt)
        """
        event_key = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            if event_key not in self._subscribers:
                self._subscribers[event_key] = set()
            self._subscribers[event_key].add(callback)
    
    def unsubscribe(self, event_type: Union[str, EventType],
                    callback: EventCallback) -> bool:
        """
        取消订阅事件
        
        Args:
            event_type: 事件类型
            callback: 要移除的回调函数
        
        Returns:
            是否成功移除
        """
        event_key = event_type.value if isinstance(event_type, EventType) else event_type
        
        with self._lock:
            if event_key in self._subscribers:
                if callback in self._subscribers[event_key]:
                    self._subscribers[event_key].remove(callback)
                    return True
        return False
    
    def publish(self, event_type: Union[str, EventType], 
                data: Any,
                source_layer: str = "unknown",
                target_layer: Optional[str] = None,
                priority: int = 5) -> None:
        """
        发布事件
        
        将事件发送到总线，由订阅者接收处理。
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source_layer: 源层标识
            target_layer: 目标层标识(可选)
            priority: 优先级(0-10，数字越小优先级越高)
        
        示例:
            >>> bus.publish(
            ...     EventType.HALT_WARNING,
            ...     {"value": 0.8, "reason": "gradient_vanishing"},
            ...     source_layer="cognition"
            ... )
        """
        event_key = event_type.value if isinstance(event_type, EventType) else event_type
        
        event = Event(
            event_type=event_key,
            data=data,
            source_layer=source_layer,
            target_layer=target_layer,
            priority=priority
        )
        
        with self._lock:
            self._event_queue.append(event)
    
    def _dispatch_event(self, event: Event) -> None:
        """
        分发事件到订阅者
        
        Args:
            event: 事件对象
        """
        if event.event_type in self._subscribers:
            for callback in list(self._subscribers[event.event_type]):
                try:
                    callback(event.event_type, event.data)
                except Exception as e:
                    print(f"回调执行错误: {e}")
        
        # 同时通知通配符订阅者
        if "*" in self._subscribers:
            for callback in list(self._subscribers["*"]):
                try:
                    callback(event.event_type, event.data)
                except Exception as e:
                    print(f"通配回调执行错误: {e}")
    
    def update_coefficient(self, coefficient_name: str, value: float) -> None:
        """
        更新状态系数
        
        更新特定的状态系数值。
        
        Args:
            coefficient_name: 系数名称(swing/halt/balance/release/dao)
            value: 新值(0.0-1.0)
        
        示例:
            >>> bus.update_coefficient("swing", 0.9)   # 更新胜值
            >>> bus.update_coefficient("halt", 0.3)    # 更新郁值
        """
        value = max(0.0, min(1.0, value))
        
        with self._lock:
            old_state = self._state_coefficient
            
            if coefficient_name == "swing":
                self._state_coefficient = StateCoefficient(
                    swing=value,
                    halt=old_state.halt,
                    balance=old_state.balance,
                    release=old_state.release,
                    dao=old_state.dao
                )
            elif coefficient_name == "halt":
                self._state_coefficient = StateCoefficient(
                    swing=old_state.swing,
                    halt=value,
                    balance=old_state.balance,
                    release=old_state.release,
                    dao=old_state.dao
                )
            elif coefficient_name == "balance":
                self._state_coefficient = StateCoefficient(
                    swing=old_state.swing,
                    halt=old_state.halt,
                    balance=value,
                    release=old_state.release,
                    dao=old_state.dao
                )
            elif coefficient_name == "release":
                self._state_coefficient = StateCoefficient(
                    swing=old_state.swing,
                    halt=old_state.halt,
                    balance=old_state.balance,
                    release=value,
                    dao=old_state.dao
                )
            elif coefficient_name == "dao":
                self._state_coefficient = StateCoefficient(
                    swing=old_state.swing,
                    halt=old_state.halt,
                    balance=old_state.balance,
                    release=old_state.release,
                    dao=value
                )
            else:
                raise ValueError(f"未知的系数名称: {coefficient_name}")
            
            # 记录历史
            self._history.append(self._state_coefficient)
            
            # 发布状态变化事件
            self.publish(
                EventType.STATE_CHANGE,
                {
                    "coefficient": coefficient_name,
                    "value": value,
                    "state": self._state_coefficient.to_dict()
                },
                source_layer="state_bus"
            )
    
    def get_current_state(self) -> StateCoefficient:
        """
        获取当前状态系数
        
        Returns:
            当前状态系数对象
        """
        with self._lock:
            return StateCoefficient(
                swing=self._state_coefficient.swing,
                halt=self._state_coefficient.halt,
                balance=self._state_coefficient.balance,
                release=self._state_coefficient.release,
                dao=self._state_coefficient.dao,
                timestamp=self._state_coefficient.timestamp
            )
    
    def get_history(self, key: Optional[str] = None, 
                    window: Optional[int] = None) -> Union[List[StateCoefficient], List[float]]:
        """
        获取历史状态记录
        
        Args:
            key: 特定系数名称(可选，None返回完整状态)
            window: 窗口大小(可选，None返回全部)
        
        Returns:
            历史记录列表
        
        示例:
            >>> # 获取所有历史状态
            >>> history = bus.get_history()
            >>> 
            >>> # 获取最近100个郁值
            >>> halt_history = bus.get_history("halt", window=100)
        """
        with self._lock:
            history_list = list(self._history)
            
            if window is not None:
                history_list = history_list[-window:]
            
            if key is None:
                return history_list
            
            # 返回特定系数的历史
            if key == "swing":
                return [s.swing for s in history_list]
            elif key == "halt":
                return [s.halt for s in history_list]
            elif key == "balance":
                return [s.balance for s in history_list]
            elif key == "release":
                return [s.release for s in history_list]
            elif key == "dao":
                return [s.dao for s in history_list]
            else:
                raise ValueError(f"未知的系数名称: {key}")
    
    def update_layer_state(self, layer: LayerType, 
                          is_active: bool,
                          metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        更新层状态
        
        Args:
            layer: 层类型
            is_active: 是否活跃
            metrics: 层指标数据(可选)
        """
        with self._lock:
            self._layer_states[layer] = LayerState(
                layer_type=layer,
                is_active=is_active,
                last_update=time.time(),
                metrics=metrics or {}
            )
    
    def get_layer_state(self, layer: LayerType) -> LayerState:
        """
        获取层状态
        
        Args:
            layer: 层类型
        
        Returns:
            层状态对象
        """
        with self._lock:
            return self._layer_states.get(layer, LayerState(layer_type=layer))
    
    def route_message(self, source: LayerType, 
                     target: LayerType,
                     message: Any,
                     priority: int = 5) -> None:
        """
        五层间消息路由
        
        在指定的源层和目标层之间路由消息。
        
        Args:
            source: 源层
            target: 目标层
            message: 消息内容
            priority: 优先级
        
        示例:
            >>> bus.route_message(
            ...     LayerType.PERCEPTION,
            ...     LayerType.COGNITION,
            ...     {"input": "sensor_data", "timestamp": time.time()}
            ... )
        """
        self.publish(
            EventType.LAYER_MESSAGE,
            {
                "source": source.value,
                "target": target.value,
                "message": message
            },
            source_layer=source.value,
            target_layer=target.value,
            priority=priority
        )
    
    def broadcast(self, event_type: Union[str, EventType],
                  data: Any,
                  exclude_layer: Optional[str] = None) -> None:
        """
        广播事件到所有层
        
        Args:
            event_type: 事件类型
            data: 事件数据
            exclude_layer: 排除的层(可选)
        """
        for layer in LayerType:
            if layer.value != exclude_layer:
                self.publish(
                    event_type,
                    data,
                    source_layer="broadcast",
                    target_layer=layer.value
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取状态总线统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                "current_state": self._state_coefficient.to_dict(),
                "health_score": self._state_coefficient.get_health_score(),
                "history_size": len(self._history),
                "subscriber_count": {
                    k: len(v) for k, v in self._subscribers.items()
                },
                "layer_states": {
                    k.value: {
                        "active": v.is_active,
                        "last_update": v.last_update,
                        "metrics": v.metrics
                    }
                    for k, v in self._layer_states.items()
                },
                "event_queue_size": len(self._event_queue)
            }
    
    def reset(self) -> None:
        """
        重置状态总线
        
        清空所有状态和订阅者，恢复到初始状态。
        """
        with self._lock:
            self._subscribers.clear()
            self._state_coefficient = StateCoefficient()
            self._history.clear()
            self._event_queue.clear()
            
            for layer in LayerType:
                self._layer_states[layer] = LayerState(layer_type=layer)
            
            self.publish(EventType.SYSTEM_RESET, {"timestamp": time.time()})
    
    def shutdown(self) -> None:
        """
        关闭状态总线
        
        停止后台线程并清理资源。
        """
        self._running = False
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=1.0)


# ============================================================================
# 便捷函数
# ============================================================================

def create_state_bus(history_window: int = 1000) -> StateBus:
    """
    创建状态总线实例的便捷函数
    
    Args:
        history_window: 历史记录窗口大小
    
    Returns:
        状态总线实例
    """
    return StateBus(history_window=history_window)


# 全局状态总线实例(单例模式)
_global_state_bus: Optional[StateBus] = None


def get_global_state_bus() -> StateBus:
    """
    获取全局状态总线实例
    
    Returns:
        全局状态总线实例
    """
    global _global_state_bus
    if _global_state_bus is None:
        _global_state_bus = StateBus()
    return _global_state_bus
