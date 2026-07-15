"""
RPC模块

基于消息总线实现远程过程调用（RPC），提供RPC客户端和服务端。
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .interface import Message, MessageBus, MessageHandler

logger = logging.getLogger(__name__)


@dataclass
class RPCRequest:
    """
    RPC请求体

    Attributes:
        id: 请求唯一标识
        service: 服务名称
        method: 方法名称
        params: 参数字典
        timestamp: 请求时间
        metadata: 附带元数据
    """

    id: str = ""
    service: str = ""
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "service": self.service,
            "method": self.method,
            "params": self.params,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RPCRequest":
        """从字典创建请求"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            id=data.get("id", ""),
            service=data.get("service", ""),
            method=data.get("method", ""),
            params=data.get("params", {}),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RPCResponse:
    """
    RPC响应体

    Attributes:
        id: 对应的请求ID
        result: 返回结果
        error: 错误信息
        duration: 处理耗时（秒）
        timestamp: 响应时间
        metadata: 附带元数据
    """

    id: str = ""
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "result": self.result,
            "error": self.error,
            "duration": round(self.duration, 6),
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RPCResponse":
        """从字典创建响应"""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            id=data.get("id", ""),
            result=data.get("result"),
            error=data.get("error"),
            duration=data.get("duration", 0.0),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )

    @property
    def is_success(self) -> bool:
        """是否成功"""
        return self.error is None


class RPCClient:
    """
    RPC客户端

    通过消息总线发送RPC请求并等待响应。

    Usage:
        bus = MemoryBackend()
        client = RPCClient(bus)

        result = client.call("user_service", "get_user", {"user_id": 123})
    """

    def __init__(
        self,
        bus: MessageBus,
        default_timeout: float = 30.0,
        prefix: str = "rpc",
    ):
        """
        初始化RPC客户端

        Args:
            bus: 消息总线实例
            default_timeout: 默认超时时间（秒）
            prefix: RPC主题前缀
        """
        self._bus = bus
        self._default_timeout = default_timeout
        self._prefix = prefix
        self._stats_lock = threading.Lock()
        self._total_calls: int = 0
        self._total_successes: int = 0
        self._total_errors: int = 0
        self._total_timeouts: int = 0

    def call(
        self,
        service_name: str,
        method_name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        远程过程调用

        构建RPC请求，通过消息总线发送，并等待响应。

        Args:
            service_name: 目标服务名称
            method_name: 方法名称
            params: 调用参数
            timeout: 超时时间（秒），None使用默认值
            metadata: 附带元数据

        Returns:
            RPC调用结果

        Raises:
            TimeoutError: 调用超时
            RuntimeError: RPC调用失败
        """
        with self._stats_lock:
            self._total_calls += 1

        request = RPCRequest(
            service=service_name,
            method=method_name,
            params=params or {},
            metadata=metadata or {},
        )

        topic = f"{self._prefix}.{service_name}.{method_name}"
        actual_timeout = timeout if timeout is not None else self._default_timeout

        headers = {
            "message_type": "rpc_request",
            "rpc_service": service_name,
            "rpc_method": method_name,
        }

        try:
            response_data = self._bus.request(
                topic=topic,
                message=request.to_dict(),
                timeout=actual_timeout,
                headers=headers,
            )

            with self._stats_lock:
                self._total_successes += 1

            if isinstance(response_data, dict):
                response = RPCResponse.from_dict(response_data)
                if not response.is_success:
                    raise RuntimeError(
                        f"RPC调用失败: service={service_name}, "
                        f"method={method_name}, error={response.error}"
                    )
                return response.result

            return response_data

        except TimeoutError:
            with self._stats_lock:
                self._total_timeouts += 1
            raise

        except RuntimeError:
            with self._stats_lock:
                self._total_errors += 1
            raise

        except Exception as exc:
            with self._stats_lock:
                self._total_errors += 1
            raise RuntimeError(
                f"RPC调用异常: service={service_name}, "
                f"method={method_name}, error={exc}"
            ) from exc

    def async_call(
        self,
        service_name: str,
        method_name: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[Any], None]] = None,
    ) -> threading.Thread:
        """
        异步RPC调用

        在新线程中执行RPC调用，可选回调处理结果。

        Args:
            service_name: 目标服务名称
            method_name: 方法名称
            params: 调用参数
            timeout: 超时时间
            callback: 结果回调函数

        Returns:
            执行线程
        """
        def _worker():
            try:
                result = self.call(
                    service_name, method_name, params, timeout
                )
                if callback:
                    callback(result)
            except Exception as exc:
                logger.error(
                    "异步RPC调用失败: service=%s, method=%s, error=%s",
                    service_name,
                    method_name,
                    exc,
                )
                if callback:
                    callback(exc)

        thread = threading.Thread(
            target=_worker,
            name=f"rpc-{service_name}-{method_name}",
            daemon=True,
        )
        thread.start()
        return thread

    def get_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        with self._stats_lock:
            return {
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_errors": self._total_errors,
                "total_timeouts": self._total_timeouts,
                "success_rate": (
                    self._total_successes / self._total_calls
                    if self._total_calls > 0 else 0.0
                ),
                "default_timeout": self._default_timeout,
            }


class RPCServer:
    """
    RPC服务端

    注册RPC方法并通过消息总线处理请求。

    Usage:
        bus = MemoryBackend()
        server = RPCServer(bus)

        @server.method("user_service", "get_user")
        def get_user(params):
            return {"name": "Alice", "age": 30}

        server.start()  # 开始监听
    """

    def __init__(
        self,
        bus: MessageBus,
        prefix: str = "rpc",
        auto_start: bool = False,
    ):
        """
        初始化RPC服务端

        Args:
            bus: 消息总线实例
            prefix: RPC主题前缀
            auto_start: 是否自动开始监听
        """
        self._bus = bus
        self._prefix = prefix
        self._methods: Dict[str, Dict[str, Callable]] = {}
        self._subscriptions: List[str] = []
        self._stats_lock = threading.Lock()
        self._total_requests: int = 0
        self._total_successes: int = 0
        self._total_errors: int = 0
        self._started = False

        if auto_start:
            self.start()

    def register_method(
        self,
        service_name: str,
        method_name: str,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        注册RPC方法

        Args:
            service_name: 服务名称
            method_name: 方法名称
            handler: 处理函数，接收参数字典，返回结果

        Raises:
            TypeError: handler不可调用
            ValueError: 方法已注册
        """
        if not callable(handler):
            raise TypeError("handler 必须是可调用对象")

        if service_name not in self._methods:
            self._methods[service_name] = {}

        if method_name in self._methods[service_name]:
            logger.warning(
                "RPC方法已存在，将被覆盖: %s.%s",
                service_name,
                method_name,
            )

        self._methods[service_name][method_name] = handler

        # 订阅对应主题
        topic = f"{self._prefix}.{service_name}.{method_name}"
        sub_id = self._bus.subscribe(topic, self._handle_request)
        self._subscriptions.append(sub_id)

        logger.info(
            "已注册RPC方法: %s.%s (subscription=%s)",
            service_name,
            method_name,
            sub_id,
        )

    def method(
        self, service_name: str, method_name: str
    ) -> Callable:
        """
        装饰器方式注册RPC方法

        Usage:
            @server.method("user_service", "get_user")
            def get_user(params):
                return {"name": "Alice"}
        """
        def decorator(func: Callable) -> Callable:
            self.register_method(service_name, method_name, func)
            return func
        return decorator

    def unregister_method(
        self, service_name: str, method_name: str
    ) -> bool:
        """
        取消注册RPC方法

        Args:
            service_name: 服务名称
            method_name: 方法名称

        Returns:
            是否成功取消
        """
        if service_name not in self._methods:
            return False

        if method_name not in self._methods[service_name]:
            return False

        del self._methods[service_name][method_name]

        # 取消订阅
        topic = f"{self._prefix}.{service_name}.{method_name}"
        subs = self._bus.get_subscriptions(topic) if hasattr(self._bus, 'get_subscriptions') else []
        for sub in subs:
            self._bus.unsubscribe(sub["subscription_id"])

        logger.info("已取消注册RPC方法: %s.%s", service_name, method_name)
        return True

    def handle_request(self, request: RPCRequest) -> RPCResponse:
        """
        处理RPC请求

        查找注册的处理函数并执行，返回RPC响应。

        Args:
            request: RPC请求

        Returns:
            RPC响应
        """
        start_time = time.monotonic()

        with self._stats_lock:
            self._total_requests += 1

        service_methods = self._methods.get(request.service)
        if service_methods is None:
            return RPCResponse(
                id=request.id,
                error=f"未知服务: {request.service}",
                duration=time.monotonic() - start_time,
            )

        handler = service_methods.get(request.method)
        if handler is None:
            available = list(service_methods.keys())
            return RPCResponse(
                id=request.id,
                error=(
                    f"未知方法: {request.method}, "
                    f"可用方法: {available}"
                ),
                duration=time.monotonic() - start_time,
            )

        try:
            result = handler(request.params)
            duration = time.monotonic() - start_time

            with self._stats_lock:
                self._total_successes += 1

            return RPCResponse(
                id=request.id,
                result=result,
                duration=duration,
            )

        except Exception as exc:
            duration = time.monotonic() - start_time

            with self._stats_lock:
                self._total_errors += 1

            logger.error(
                "RPC方法执行错误: %s.%s, error=%s",
                request.service,
                request.method,
                exc,
            )

            return RPCResponse(
                id=request.id,
                error=f"{type(exc).__name__}: {str(exc)}",
                duration=duration,
            )

    def _handle_request(self, message: Message) -> None:
        """消息总线回调：处理RPC请求消息"""
        try:
            request_data = message.payload
            if isinstance(request_data, dict):
                request = RPCRequest.from_dict(request_data)
            else:
                request = RPCRequest(
                    service=message.headers.get("rpc_service", ""),
                    method=message.headers.get("rpc_method", ""),
                    params=request_data if isinstance(request_data, dict) else {},
                )

            response = self.handle_request(request)

            # 发送回复
            reply_topic = message.headers.get("reply_to", "")
            if reply_topic:
                # 从headers获取correlation_id（与request方法设置的一致）
                corr_id = message.headers.get("correlation_id", "") or message.correlation_id or message.id
                reply_headers = {
                    "message_type": "rpc_response",
                    "correlation_id": corr_id,
                }
                if response.is_success:
                    self._bus.publish(
                        reply_topic,
                        response.to_dict(),
                        headers=reply_headers,
                    )
                else:
                    # RPC业务错误仍然使用rpc_response类型，
                    # 让客户端通过RPCResponse.error字段处理
                    self._bus.publish(
                        reply_topic,
                        response.to_dict(),
                        headers=reply_headers,
                    )

        except Exception as exc:
            logger.error("RPC请求处理异常: %s", exc)

    def start(self) -> None:
        """开始监听RPC请求（重新订阅所有已注册方法）"""
        if self._started:
            return

        # 订阅通配符主题，处理所有RPC请求（包括未注册的方法）
        wildcard_topic = f"{self._prefix}.*"
        sub_id = self._bus.subscribe(wildcard_topic, self._handle_request)
        self._subscriptions.append(sub_id)

        self._started = True
        logger.info("RPC服务端已启动，已注册 %d 个服务",
                    len(self._methods))

    def stop(self) -> None:
        """停止RPC服务端"""
        for sub_id in self._subscriptions:
            self._bus.unsubscribe(sub_id)
        self._subscriptions.clear()
        self._started = False
        logger.info("RPC服务端已停止")

    def get_methods(self) -> Dict[str, List[str]]:
        """获取所有已注册的方法"""
        return {
            service: list(methods.keys())
            for service, methods in self._methods.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取服务端统计信息"""
        with self._stats_lock:
            total_methods = sum(
                len(methods) for methods in self._methods.values()
            )
            return {
                "started": self._started,
                "total_services": len(self._methods),
                "total_methods": total_methods,
                "total_requests": self._total_requests,
                "total_successes": self._total_successes,
                "total_errors": self._total_errors,
                "success_rate": (
                    self._total_successes / self._total_requests
                    if self._total_requests > 0 else 0.0
                ),
                "methods": self.get_methods(),
            }

    def __repr__(self) -> str:
        total_methods = sum(
            len(m) for m in self._methods.values()
        )
        return (
            f"RPCServer(services={len(self._methods)}, "
            f"methods={total_methods}, started={self._started})"
        )
