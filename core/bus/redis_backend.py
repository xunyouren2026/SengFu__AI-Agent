"""
Redis消息总线后端实现

提供Redis连接池、发布/订阅、流处理、消费者组和集群支持。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

from .interface import Message, MessageBus, MessageHandler

T = TypeVar("T")
logger = logging.getLogger(__name__)


class RedisError(Exception):
    """Redis操作错误"""
    pass


class RedisConnectionError(RedisError):
    """Redis连接错误"""
    pass


class RedisTimeoutError(RedisError):
    """Redis超时错误"""
    pass


class RedisClusterError(RedisError):
    """Redis集群错误"""
    pass


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    ssl: bool = False
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, Any] = field(default_factory=dict)
    retry_on_timeout: bool = True
    max_connections: int = 50
    min_connections: int = 10
    connection_retry_delay: float = 1.0
    connection_retry_max: int = 3
    health_check_interval: float = 30.0
    encoding: str = "utf-8"
    decode_responses: bool = True


@dataclass
class RedisClusterConfig:
    """Redis集群配置"""
    startup_nodes: List[Tuple[str, int]] = field(default_factory=list)
    skip_full_coverage_check: bool = False
    nodemanager_follow_cluster: bool = True
    max_connections_per_node: int = 50
    reinitialize_steps: int = 10
    read_from_replicas: bool = False


@dataclass
class RedisSentinelConfig:
    """Redis Sentinel配置"""
    sentinels: List[Tuple[str, int]] = field(default_factory=list)
    service_name: str = "mymaster"
    sentinel_password: Optional[str] = None
    min_other_sentinels: int = 0
    sentinel_kwargs: Dict[str, Any] = field(default_factory=dict)


class RedisProtocol(Protocol):
    """Redis协议接口"""
    
    async def execute(self, command: str, *args: Any) -> Any:
        ...
    
    async def pipeline(self, commands: List[Tuple[str, ...]]) -> List[Any]:
        ...
    
    async def close(self) -> None:
        ...


class RedisConnection:
    """Redis连接实现"""
    
    def __init__(self, config: RedisConfig) -> None:
        self.config = config
        self._connected = False
        self._socket: Optional[socket.socket] = None
        self._lock = asyncio.Lock()
        self._last_used = time.time()
        self._id = uuid.uuid4().hex[:8]
        
    async def connect(self) -> None:
        """建立连接"""
        async with self._lock:
            if self._connected:
                return
                
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.config.socket_connect_timeout)
                
                if self.config.socket_keepalive:
                    self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    for opt, val in self.config.socket_keepalive_options.items():
                        self._socket.setsockopt(socket.IPPROTO_TCP, opt, val)
                
                await asyncio.get_event_loop().sock_connect(
                    self._socket, (self.config.host, self.config.port)
                )
                
                # 认证
                if self.config.password:
                    await self._send_command("AUTH", self.config.password)
                    response = await self._read_response()
                    if response != "OK":
                        raise RedisConnectionError(f"认证失败: {response}")
                
                # 选择数据库
                if self.config.db != 0:
                    await self._send_command("SELECT", str(self.config.db))
                    response = await self._read_response()
                    if response != "OK":
                        raise RedisConnectionError(f"选择数据库失败: {response}")
                
                self._connected = True
                self._last_used = time.time()
                logger.debug(f"Redis连接 {self._id} 已建立")
                
            except (socket.error, asyncio.TimeoutError) as e:
                raise RedisConnectionError(f"连接失败: {e}")
    
    async def execute(self, command: str, *args: Any) -> Any:
        """执行命令"""
        if not self._connected:
            await self.connect()
        
        async with self._lock:
            try:
                await self._send_command(command, *args)
                result = await self._read_response()
                self._last_used = time.time()
                return result
            except (socket.error, asyncio.TimeoutError) as e:
                self._connected = False
                raise RedisError(f"命令执行失败: {e}")
    
    async def _send_command(self, command: str, *args: Any) -> None:
        """发送Redis命令"""
        if not self._socket:
            raise RedisConnectionError("未连接")
        
        parts = [command.encode()]
        for arg in args:
            if isinstance(arg, str):
                parts.append(arg.encode())
            elif isinstance(arg, bytes):
                parts.append(arg)
            else:
                parts.append(str(arg).encode())
        
        # RESP协议编码
        data = f"*{len(parts)}\r\n".encode()
        for part in parts:
            data += f"${len(part)}\r\n".encode() + part + b"\r\n"
        
        await asyncio.get_event_loop().sock_sendall(self._socket, data)
    
    async def _read_response(self) -> Any:
        """读取Redis响应"""
        if not self._socket:
            raise RedisConnectionError("未连接")
        
        line = await self._read_line()
        if not line:
            raise RedisConnectionError("连接已关闭")
        
        prefix = line[0:1]
        data = line[1:-2]  # 移除\r\n
        
        if prefix == b"+":  # 简单字符串
            return data.decode() if self.config.decode_responses else data
        elif prefix == b"-":  # 错误
            raise RedisError(data.decode())
        elif prefix == b":":  # 整数
            return int(data)
        elif prefix == b"$":  # 批量字符串
            length = int(data)
            if length == -1:
                return None
            bulk = await self._read_bytes(length + 2)
            result = bulk[:-2]
            return result.decode() if self.config.decode_responses else result
        elif prefix == b"*":  # 数组
            count = int(data)
            if count == -1:
                return None
            result = []
            for _ in range(count):
                result.append(await self._read_response())
            return result
        else:
            raise RedisError(f"未知响应类型: {prefix}")
    
    async def _read_line(self) -> bytes:
        """读取一行"""
        buffer = b""
        while not buffer.endswith(b"\r\n"):
            chunk = await asyncio.get_event_loop().sock_recv(self._socket, 1)
            if not chunk:
                break
            buffer += chunk
        return buffer
    
    async def _read_bytes(self, n: int) -> bytes:
        """读取指定字节数"""
        buffer = b""
        while len(buffer) < n:
            chunk = await asyncio.get_event_loop().sock_recv(
                self._socket, n - len(buffer)
            )
            if not chunk:
                break
            buffer += chunk
        return buffer
    
    async def close(self) -> None:
        """关闭连接"""
        async with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            self._connected = False
            logger.debug(f"Redis连接 {self._id} 已关闭")
    
    @property
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected
    
    @property
    def idle_time(self) -> float:
        """获取空闲时间"""
        return time.time() - self._last_used


class RedisConnectionPool:
    """Redis连接池"""
    
    def __init__(self, config: RedisConfig) -> None:
        self.config = config
        self._pool: asyncio.Queue[RedisConnection] = asyncio.Queue(
            maxsize=config.max_connections
        )
        self._in_use: Set[RedisConnection] = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(config.max_connections)
        self._closed = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_count = 0
        
    async def initialize(self) -> None:
        """初始化连接池"""
        # 创建最小连接数
        for _ in range(self.config.min_connections):
            conn = RedisConnection(self.config)
            await conn.connect()
            await self._pool.put(conn)
            self._connection_count += 1
        
        # 启动健康检查
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Redis连接池初始化完成，初始连接数: {self.config.min_connections}")
    
    async def acquire(self) -> RedisConnection:
        """获取连接"""
        if self._closed:
            raise RedisError("连接池已关闭")
        
        async with self._semaphore:
            async with self._lock:
                # 尝试从池中获取
                if not self._pool.empty():
                    conn = await self._pool.get()
                    if conn.is_connected:
                        self._in_use.add(conn)
                        return conn
                    await conn.close()
                    self._connection_count -= 1
                
                # 创建新连接
                if self._connection_count < self.config.max_connections:
                    conn = RedisConnection(self.config)
                    await conn.connect()
                    self._in_use.add(conn)
                    self._connection_count += 1
                    return conn
            
            # 等待可用连接
            conn = await self._pool.get()
            async with self._lock:
                self._in_use.add(conn)
            return conn
    
    async def release(self, conn: RedisConnection) -> None:
        """释放连接"""
        async with self._lock:
            self._in_use.discard(conn)
            if conn.is_connected and not self._closed:
                await self._pool.put(conn)
            else:
                await conn.close()
                self._connection_count -= 1
    
    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
    
    async def _health_check(self) -> None:
        """执行健康检查"""
        to_remove = []
        
        # 检查池中的连接
        temp_list = []
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if conn.idle_time > self.config.health_check_interval:
                    try:
                        await conn.execute("PING")
                        temp_list.append(conn)
                    except Exception:
                        to_remove.append(conn)
                else:
                    temp_list.append(conn)
            except asyncio.QueueEmpty:
                break
        
        for conn in temp_list:
            await self._pool.put(conn)
        
        # 关闭失效连接
        for conn in to_remove:
            await conn.close()
            self._connection_count -= 1
            logger.debug("关闭失效连接")
    
    async def close(self) -> None:
        """关闭连接池"""
        self._closed = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        async with self._lock:
            for conn in list(self._in_use):
                await conn.close()
            self._in_use.clear()
            
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    await conn.close()
                except asyncio.QueueEmpty:
                    break
        
        logger.info("Redis连接池已关闭")
    
    @property
    def size(self) -> int:
        """获取当前连接数"""
        return self._connection_count
    
    @property
    def available(self) -> int:
        """获取可用连接数"""
        return self._pool.qsize()


class RedisPubSub:
    """Redis发布/订阅实现"""
    
    def __init__(self, pool: RedisConnectionPool) -> None:
        self.pool = pool
        self._subscribers: Dict[str, Set[Callable[[str, Any], Coroutine]]] = defaultdict(set)
        self._patterns: Dict[str, Set[Callable[[str, Any], Coroutine]]] = defaultdict(set)
        self._listening = False
        self._listen_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._pubsub_conn: Optional[RedisConnection] = None
        
    async def subscribe(self, channel: str, handler: Callable[[str, Any], Coroutine]) -> None:
        """订阅频道"""
        async with self._lock:
            is_new = len(self._subscribers[channel]) == 0
            self._subscribers[channel].add(handler)
            
            if is_new:
                if not self._pubsub_conn:
                    self._pubsub_conn = await self.pool.acquire()
                await self._pubsub_conn.execute("SUBSCRIBE", channel)
                
                if not self._listening:
                    self._start_listening()
        
        logger.debug(f"订阅频道: {channel}")
    
    async def unsubscribe(self, channel: str, handler: Callable[[str, Any], Coroutine]) -> None:
        """取消订阅"""
        async with self._lock:
            if channel in self._subscribers:
                self._subscribers[channel].discard(handler)
                if not self._subscribers[channel]:
                    del self._subscribers[channel]
                    if self._pubsub_conn:
                        await self._pubsub_conn.execute("UNSUBSCRIBE", channel)
        
        logger.debug(f"取消订阅频道: {channel}")
    
    async def psubscribe(self, pattern: str, handler: Callable[[str, Any], Coroutine]) -> None:
        """按模式订阅"""
        async with self._lock:
            is_new = len(self._patterns[pattern]) == 0
            self._patterns[pattern].add(handler)
            
            if is_new:
                if not self._pubsub_conn:
                    self._pubsub_conn = await self.pool.acquire()
                await self._pubsub_conn.execute("PSUBSCRIBE", pattern)
                
                if not self._listening:
                    self._start_listening()
        
        logger.debug(f"模式订阅: {pattern}")
    
    async def publish(self, channel: str, message: Any) -> int:
        """发布消息"""
        conn = await self.pool.acquire()
        try:
            data = json.dumps(message) if not isinstance(message, (str, bytes)) else message
            result = await conn.execute("PUBLISH", channel, data)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    def _start_listening(self) -> None:
        """开始监听"""
        if not self._listening:
            self._listening = True
            self._listen_task = asyncio.create_task(self._listen_loop())
    
    async def _listen_loop(self) -> None:
        """监听循环"""
        while self._listening:
            try:
                if not self._pubsub_conn:
                    await asyncio.sleep(0.1)
                    continue
                
                # 读取发布/订阅消息
                response = await self._pubsub_conn._read_response()
                if isinstance(response, list) and len(response) >= 3:
                    msg_type = response[0]
                    if msg_type == "message":
                        channel = response[1]
                        data = response[2]
                        await self._handle_message(channel, data, False)
                    elif msg_type == "pmessage":
                        pattern = response[1]
                        channel = response[2]
                        data = response[3]
                        await self._handle_message(channel, data, True, pattern)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监听错误: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(
        self, channel: str, data: Any, is_pattern: bool, pattern: Optional[str] = None
    ) -> None:
        """处理消息"""
        try:
            message = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError:
            message = data
        
        handlers = set()
        if is_pattern and pattern:
            handlers.update(self._patterns.get(pattern, set()))
        else:
            handlers.update(self._subscribers.get(channel, set()))
        
        for handler in handlers:
            try:
                asyncio.create_task(handler(channel, message))
            except Exception as e:
                logger.error(f"消息处理错误: {e}")
    
    async def close(self) -> None:
        """关闭发布/订阅"""
        self._listening = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._pubsub_conn:
            await self.pool.release(self._pubsub_conn)
            self._pubsub_conn = None


class RedisStream:
    """Redis Stream实现"""
    
    def __init__(self, pool: RedisConnectionPool) -> None:
        self.pool = pool
        
    async def add(
        self,
        stream: str,
        fields: Dict[str, Any],
        message_id: str = "*",
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        """添加消息到流"""
        conn = await self.pool.acquire()
        try:
            args = [stream]
            
            if maxlen is not None:
                args.append("MAXLEN")
                if approximate:
                    args.append("~")
                args.append(str(maxlen))
            
            args.append(message_id)
            
            for key, value in fields.items():
                args.append(key)
                args.append(json.dumps(value) if not isinstance(value, str) else value)
            
            result = await conn.execute("XADD", *args)
            return result if isinstance(result, str) else ""
        finally:
            await self.pool.release(conn)
    
    async def read(
        self,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """读取流消息"""
        conn = await self.pool.acquire()
        try:
            args = []
            
            if count is not None:
                args.extend(["COUNT", str(count)])
            
            if block is not None:
                args.extend(["BLOCK", str(block)])
            
            args.append("STREAMS")
            
            stream_names = list(streams.keys())
            ids = list(streams.values())
            
            args.extend(stream_names)
            args.extend(ids)
            
            result = await conn.execute("XREAD", *args)
            return self._parse_stream_result(result)
        finally:
            await self.pool.release(conn)
    
    async def range(
        self,
        stream: str,
        start: str = "-",
        end: str = "+",
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """获取流范围"""
        conn = await self.pool.acquire()
        try:
            args = [stream, start, end]
            if count is not None:
                args.extend(["COUNT", str(count)])
            
            result = await conn.execute("XRANGE", *args)
            return self._parse_entries(result)
        finally:
            await self.pool.release(conn)
    
    async def delete(self, stream: str, message_ids: List[str]) -> int:
        """删除流消息"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XDEL", stream, *message_ids)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    async def trim(self, stream: str, maxlen: int, approximate: bool = True) -> int:
        """修剪流"""
        conn = await self.pool.acquire()
        try:
            args = [stream, "MAXLEN"]
            if approximate:
                args.append("~")
            args.append(str(maxlen))
            
            result = await conn.execute("XTRIM", *args)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    async def len(self, stream: str) -> int:
        """获取流长度"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XLEN", stream)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    def _parse_stream_result(self, result: Any) -> Dict[str, List[Dict[str, Any]]]:
        """解析流结果"""
        if not result:
            return {}
        
        output = {}
        for item in result:
            if isinstance(item, list) and len(item) == 2:
                stream_name = item[0]
                entries = self._parse_entries(item[1])
                output[stream_name] = entries
        
        return output
    
    def _parse_entries(self, entries: Any) -> List[Dict[str, Any]]:
        """解析条目"""
        result = []
        if not entries:
            return result
        
        for entry in entries:
            if isinstance(entry, list) and len(entry) == 2:
                message_id = entry[0]
                fields = entry[1]
                
                field_dict = {}
                if isinstance(fields, list):
                    for i in range(0, len(fields), 2):
                        key = fields[i]
                        value = fields[i + 1] if i + 1 < len(fields) else None
                        try:
                            field_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            field_dict[key] = value
                
                result.append({
                    "id": message_id,
                    "fields": field_dict,
                })
        
        return result


class RedisConsumerGroup:
    """Redis消费者组实现"""
    
    def __init__(self, pool: RedisConnectionPool) -> None:
        self.pool = pool
        
    async def create(
        self, group: str, stream: str, message_id: str = "0", mkstream: bool = False
    ) -> bool:
        """创建消费者组"""
        conn = await self.pool.acquire()
        try:
            args = ["CREATE", stream, group, message_id]
            if mkstream:
                args.append("MKSTREAM")
            
            result = await conn.execute("XGROUP", *args)
            return result == "OK"
        finally:
            await self.pool.release(conn)
    
    async def destroy(self, group: str, stream: str) -> int:
        """销毁消费者组"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XGROUP", "DESTROY", stream, group)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    async def create_consumer(self, group: str, stream: str, consumer: str) -> bool:
        """创建消费者"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XGROUP", "CREATECONSUMER", stream, group, consumer)
            return result == 1
        finally:
            await self.pool.release(conn)
    
    async def delete_consumer(self, group: str, stream: str, consumer: str) -> int:
        """删除消费者"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XGROUP", "DELCONSUMER", stream, group, consumer)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    async def read(
        self,
        group: str,
        consumer: str,
        streams: List[str],
        count: Optional[int] = None,
        block: Optional[int] = None,
        noack: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """消费者组读取"""
        conn = await self.pool.acquire()
        try:
            args = ["GROUP", group, consumer]
            
            if count is not None:
                args.extend(["COUNT", str(count)])
            
            if block is not None:
                args.extend(["BLOCK", str(block)])
            
            if noack:
                args.append("NOACK")
            
            args.append("STREAMS")
            args.extend(streams)
            args.extend([">"] * len(streams))  # 只读新消息
            
            result = await conn.execute("XREADGROUP", *args)
            return self._parse_stream_result(result)
        finally:
            await self.pool.release(conn)
    
    async def ack(self, stream: str, group: str, message_ids: List[str]) -> int:
        """确认消息"""
        conn = await self.pool.acquire()
        try:
            result = await conn.execute("XACK", stream, group, *message_ids)
            return result if isinstance(result, int) else 0
        finally:
            await self.pool.release(conn)
    
    async def claim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        message_ids: List[str],
        idle: Optional[int] = None,
        time: Optional[int] = None,
        retrycount: Optional[int] = None,
        force: bool = False,
        justid: bool = False,
    ) -> List[Dict[str, Any]]:
        """声明消息所有权"""
        conn = await self.pool.acquire()
        try:
            args = [stream, group, consumer, str(min_idle_time)]
            args.extend(message_ids)
            
            if idle is not None:
                args.extend(["IDLE", str(idle)])
            if time is not None:
                args.extend(["TIME", str(time)])
            if retrycount is not None:
                args.extend(["RETRYCOUNT", str(retrycount)])
            if force:
                args.append("FORCE")
            if justid:
                args.append("JUSTID")
            
            result = await conn.execute("XCLAIM", *args)
            
            if justid:
                return [{"id": mid} for mid in (result or [])]
            return self._parse_entries(result)
        finally:
            await self.pool.release(conn)
    
    async def pending(
        self,
        stream: str,
        group: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        count: Optional[int] = None,
        consumer: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """获取待处理消息"""
        conn = await self.pool.acquire()
        try:
            args = [stream, group]
            
            if start is not None:
                args.extend([start, end, str(count)])
                if consumer:
                    args.append(consumer)
            
            result = await conn.execute("XPENDING", *args)
            
            if start is None:
                # 返回摘要信息
                if isinstance(result, list) and len(result) >= 3:
                    return {
                        "count": result[0],
                        "min_id": result[1],
                        "max_id": result[2],
                        "consumers": [
                            {"name": c[0], "count": c[1]} 
                            for c in (result[3] if len(result) > 3 else [])
                        ],
                    }
            else:
                # 返回详细列表
                return [
                    {
                        "id": item[0],
                        "consumer": item[1],
                        "idle": item[2],
                        "deliveries": item[3],
                    }
                    for item in (result or [])
                ]
            
            return result
        finally:
            await self.pool.release(conn)
    
    def _parse_stream_result(self, result: Any) -> Dict[str, List[Dict[str, Any]]]:
        """解析流结果"""
        if not result:
            return {}
        
        output = {}
        for item in result:
            if isinstance(item, list) and len(item) == 2:
                stream_name = item[0]
                entries = self._parse_entries(item[1])
                output[stream_name] = entries
        
        return output
    
    def _parse_entries(self, entries: Any) -> List[Dict[str, Any]]:
        """解析条目"""
        result = []
        if not entries:
            return result
        
        for entry in entries:
            if isinstance(entry, list) and len(entry) == 2:
                message_id = entry[0]
                fields = entry[1]
                
                field_dict = {}
                if isinstance(fields, list):
                    for i in range(0, len(fields), 2):
                        key = fields[i]
                        value = fields[i + 1] if i + 1 < len(fields) else None
                        try:
                            field_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            field_dict[key] = value
                
                result.append({
                    "id": message_id,
                    "fields": field_dict,
                })
        
        return result


class RedisClusterManager:
    """Redis集群管理器"""
    
    def __init__(self, config: RedisClusterConfig) -> None:
        self.config = config
        self._nodes: Dict[str, RedisConnectionPool] = {}
        self._slots: Dict[int, str] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self) -> None:
        """初始化集群连接"""
        if self._initialized:
            return
        
        async with self._lock:
            # 连接启动节点
            for host, port in self.config.startup_nodes:
                try:
                    config = RedisConfig(host=host, port=port)
                    pool = RedisConnectionPool(config)
                    await pool.initialize()
                    node_id = f"{host}:{port}"
                    self._nodes[node_id] = pool
                    
                    # 获取集群槽位信息
                    await self._discover_cluster_slots(pool)
                    break
                except Exception as e:
                    logger.warning(f"无法连接启动节点 {host}:{port}: {e}")
                    continue
            else:
                raise RedisClusterError("无法连接任何启动节点")
            
            self._initialized = True
            logger.info("Redis集群管理器初始化完成")
    
    async def _discover_cluster_slots(self, pool: RedisConnectionPool) -> None:
        """发现集群槽位"""
        conn = await pool.acquire()
        try:
            result = await conn.execute("CLUSTER", "SLOTS")
            if not result:
                return
            
            for slot_range in result:
                if len(slot_range) >= 3:
                    start_slot = slot_range[0]
                    end_slot = slot_range[1]
                    master_node = slot_range[2]
                    
                    host = master_node[0]
                    port = master_node[1]
                    node_id = f"{host}:{port}"
                    
                    # 确保节点连接
                    if node_id not in self._nodes:
                        config = RedisConfig(host=host, port=port)
                        new_pool = RedisConnectionPool(config)
                        await new_pool.initialize()
                        self._nodes[node_id] = new_pool
                    
                    # 映射槽位
                    for slot in range(start_slot, end_slot + 1):
                        self._slots[slot] = node_id
                        
        finally:
            await pool.release(conn)
    
    def _get_slot(self, key: str) -> int:
        """计算键的槽位"""
        # 处理hash tag
        if "{" in key:
            start = key.find("{")
            end = key.find("}", start)
            if end != -1 and end > start + 1:
                key = key[start + 1:end]
        
        # CRC16算法
        crc = self._crc16(key.encode())
        return crc % 16384
    
    def _crc16(self, data: bytes) -> int:
        """CRC16计算"""
        crc_table = [
            0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
            0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
        ]
        crc = 0
        for byte in data:
            crc = ((crc << 8) & 0xFF00) ^ crc_table[((crc >> 12) & 0x0F) ^ ((byte >> 4) & 0x0F)]
            crc = ((crc << 8) & 0xFF00) ^ crc_table[((crc >> 12) & 0x0F) ^ (byte & 0x0F)]
        return crc & 0xFFFF
    
    async def execute(self, command: str, *args: Any) -> Any:
        """在集群上执行命令"""
        if not args:
            # 没有键，随机选择节点
            node_id = next(iter(self._nodes))
        else:
            key = str(args[0])
            slot = self._get_slot(key)
            node_id = self._slots.get(slot)
            
            if not node_id:
                raise RedisClusterError(f"无法找到槽位 {slot} 对应的节点")
        
        pool = self._nodes.get(node_id)
        if not pool:
            raise RedisClusterError(f"节点 {node_id} 不存在")
        
        conn = await pool.acquire()
        try:
            return await conn.execute(command, *args)
        except RedisError as e:
            if "MOVED" in str(e):
                # 槽位迁移，重新发现
                await self._discover_cluster_slots(pool)
                return await self.execute(command, *args)
            raise
        finally:
            await pool.release(conn)
    
    async def close(self) -> None:
        """关闭集群连接"""
        async with self._lock:
            for pool in self._nodes.values():
                await pool.close()
            self._nodes.clear()
            self._slots.clear()
            self._initialized = False


class RedisSentinel:
    """Redis Sentinel实现"""
    
    def __init__(self, config: RedisSentinelConfig) -> None:
        self.config = config
        self._sentinel_pools: List[RedisConnectionPool] = []
        self._master_pool: Optional[RedisConnectionPool] = None
        self._replica_pools: List[RedisConnectionPool] = []
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> None:
        """初始化Sentinel连接"""
        # 连接所有Sentinel
        for host, port in self.config.sentinels:
            config = RedisConfig(host=host, port=port)
            pool = RedisConnectionPool(config)
            await pool.initialize()
            self._sentinel_pools.append(pool)
        
        # 发现主节点
        await self._discover_master()
        
        # 启动监控
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Redis Sentinel初始化完成")
    
    async def _discover_master(self) -> None:
        """发现主节点"""
        for sentinel_pool in self._sentinel_pools:
            conn = await sentinel_pool.acquire()
            try:
                result = await conn.execute(
                    "SENTINEL", "get-master-addr-by-name", self.config.service_name
                )
                if result and len(result) == 2:
                    host, port = result[0], int(result[1])
                    
                    # 创建主节点连接池
                    config = RedisConfig(
                        host=host,
                        port=port,
                        password=self.config.sentinel_password,
                    )
                    self._master_pool = RedisConnectionPool(config)
                    await self._master_pool.initialize()
                    return
            except Exception as e:
                logger.warning(f"Sentinel查询失败: {e}")
            finally:
                await sentinel_pool.release(conn)
        
        raise RedisConnectionError("无法发现主节点")
    
    async def _discover_replicas(self) -> None:
        """发现副本节点"""
        for replica_pool in self._replica_pools:
            await replica_pool.close()
        self._replica_pools.clear()
        
        for sentinel_pool in self._sentinel_pools:
            conn = await sentinel_pool.acquire()
            try:
                result = await conn.execute(
                    "SENTINEL", "replicas", self.config.service_name
                )
                for replica in result:
                    replica_info = dict(zip(replica[::2], replica[1::2]))
                    if replica_info.get("is_odown") == "0":
                        host = replica_info.get("ip")
                        port = int(replica_info.get("port", 6379))
                        
                        config = RedisConfig(
                            host=host,
                            port=port,
                            password=self.config.sentinel_password,
                        )
                        pool = RedisConnectionPool(config)
                        await pool.initialize()
                        self._replica_pools.append(pool)
                return
            except Exception as e:
                logger.warning(f"发现副本失败: {e}")
            finally:
                await sentinel_pool.release(conn)
    
    async def _monitor_loop(self) -> None:
        """监控循环"""
        while True:
            try:
                await asyncio.sleep(10)
                
                # 检查主节点
                if self._master_pool:
                    conn = await self._master_pool.acquire()
                    try:
                        await conn.execute("PING")
                    except Exception:
                        logger.warning("主节点失效，重新发现")
                        await self._discover_master()
                    finally:
                        await self._master_pool.release(conn)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控错误: {e}")
    
    async def execute(self, command: str, *args: Any, readonly: bool = False) -> Any:
        """执行命令"""
        if readonly and self._replica_pools:
            # 读操作使用副本
            import random
            pool = random.choice(self._replica_pools)
        else:
            # 写操作使用主节点
            if not self._master_pool:
                raise RedisConnectionError("无可用主节点")
            pool = self._master_pool
        
        conn = await pool.acquire()
        try:
            return await conn.execute(command, *args)
        finally:
            await pool.release(conn)
    
    async def close(self) -> None:
        """关闭Sentinel连接"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        for pool in self._sentinel_pools:
            await pool.close()
        
        if self._master_pool:
            await self._master_pool.close()
        
        for pool in self._replica_pools:
            await pool.close()


class RedisBackend(MessageBus):
    """Redis消息总线后端"""
    
    def __init__(
        self,
        config: Optional[RedisConfig] = None,
        cluster_config: Optional[RedisClusterConfig] = None,
        sentinel_config: Optional[RedisSentinelConfig] = None,
    ) -> None:
        self.config = config or RedisConfig()
        self.cluster_config = cluster_config
        self.sentinel_config = sentinel_config
        
        self._pool: Optional[RedisConnectionPool] = None
        self._cluster: Optional[RedisClusterManager] = None
        self._sentinel: Optional[RedisSentinel] = None
        
        self._pubsub: Optional[RedisPubSub] = None
        self._stream: Optional[RedisStream] = None
        self._consumer_group: Optional[RedisConsumerGroup] = None
        
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._running = False
        
    async def start(self) -> None:
        """启动后端"""
        if self._running:
            return
        
        if self.cluster_config:
            self._cluster = RedisClusterManager(self.cluster_config)
            await self._cluster.initialize()
        elif self.sentinel_config:
            self._sentinel = RedisSentinel(self.sentinel_config)
            await self._sentinel.initialize()
            self._pool = self._sentinel._master_pool
        else:
            self._pool = RedisConnectionPool(self.config)
            await self._pool.initialize()
        
        if self._pool:
            self._pubsub = RedisPubSub(self._pool)
            self._stream = RedisStream(self._pool)
            self._consumer_group = RedisConsumerGroup(self._pool)
        
        self._running = True
        logger.info("Redis后端已启动")
    
    async def stop(self) -> None:
        """停止后端"""
        if not self._running:
            return
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._cluster:
            await self._cluster.close()
        elif self._sentinel:
            await self._sentinel.close()
        elif self._pool:
            await self._pool.close()
        
        self._running = False
        logger.info("Redis后端已停止")
    
    async def publish(self, topic: str, message: Message) -> bool:
        """发布消息"""
        if not self._running:
            raise RedisError("后端未启动")
        
        try:
            data = json.dumps({
                "id": message.id,
                "topic": message.topic,
                "payload": message.payload,
                "headers": message.headers,
                "timestamp": message.timestamp,
            })
            
            if self._cluster:
                await self._cluster.execute("PUBLISH", topic, data)
            elif self._sentinel:
                await self._sentinel.execute("PUBLISH", topic, data)
            elif self._pool and self._pubsub:
                await self._pubsub.publish(topic, data)
            else:
                return False
            
            return True
        except Exception as e:
            logger.error(f"发布失败: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """订阅主题"""
        if not self._running:
            raise RedisError("后端未启动")
        
        self._handlers[topic].append(handler)
        
        if self._pubsub:
            await self._pubsub.subscribe(topic, self._on_message)
    
    async def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """取消订阅"""
        if topic in self._handlers:
            self._handlers[topic].remove(handler)
            
            if self._pubsub:
                await self._pubsub.unsubscribe(topic, self._on_message)
    
    async def _on_message(self, channel: str, data: Any) -> None:
        """处理消息"""
        try:
            if isinstance(data, str):
                msg_data = json.loads(data)
            else:
                msg_data = data
            
            message = Message(
                id=msg_data.get("id", ""),
                topic=msg_data.get("topic", channel),
                payload=msg_data.get("payload"),
                headers=msg_data.get("headers", {}),
                timestamp=msg_data.get("timestamp", 0.0),
            )
            
            handlers = self._handlers.get(channel, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"消息处理错误: {e}")
                    
        except Exception as e:
            logger.error(f"消息解析错误: {e}")
    
    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Optional[Message]:
        """请求-响应模式"""
        # 使用唯一响应主题
        response_topic = f"{topic}:response:{message.id}"
        
        future: asyncio.Future[Message] = asyncio.Future()
        
        async def response_handler(msg: Message) -> None:
            if not future.done():
                future.set_result(msg)
        
        # 订阅响应主题
        await self.subscribe(response_topic, response_handler)
        
        try:
            # 发布请求
            await self.publish(topic, message)
            
            # 等待响应
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            await self.unsubscribe(response_topic, response_handler)
    
    @property
    def pubsub(self) -> Optional[RedisPubSub]:
        """获取发布/订阅组件"""
        return self._pubsub
    
    @property
    def stream(self) -> Optional[RedisStream]:
        """获取流组件"""
        return self._stream
    
    @property
    def consumer_group(self) -> Optional[RedisConsumerGroup]:
        """获取消费者组组件"""
        return self._consumer_group
