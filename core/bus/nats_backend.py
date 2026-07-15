"""
NATS消息总线后端实现

提供NATS连接、JetStream、键值存储、对象存储、队列组和请求-响应模式支持。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import struct
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

from .interface import Message, MessageBus, MessageHandler

T = TypeVar("T")
logger = logging.getLogger(__name__)


class NATSError(Exception):
    """NATS操作错误"""
    pass


class NATSConnectionError(NATSError):
    """NATS连接错误"""
    pass


class NATSTimeoutError(NATSError):
    """NATS超时错误"""
    pass


class NATSJetStreamError(NATSError):
    """JetStream错误"""
    pass


@dataclass
class NATSConfig:
    """NATS配置"""
    servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    name: str = "nats_client"
    pedantic: bool = False
    verbose: bool = False
    allow_reconnect: bool = True
    connect_timeout: float = 2.0
    reconnect_time_wait: float = 1.0
    max_reconnect_attempts: int = 60
    ping_interval: float = 120.0
    max_outstanding_pings: int = 2
    flusher_queue_size: int = 1024
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None
    tls_ca_cert: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    nkeys_seed_str: Optional[str] = None
    inbox_prefix: str = "_INBOX"


@dataclass
class JetStreamConfig:
    """JetStream配置"""
    domain: Optional[str] = None
    prefix: str = "$JS.API"
    timeout: float = 5.0
    ack_wait: float = 30.0
    max_deliver: int = 20
    max_ack_pending: int = 1000
    replay_policy: str = "instant"  # instant, original
    retention_policy: str = "limits"  # limits, interest, workqueue
    storage_type: str = "file"  # file, memory


@dataclass
class StreamConfig:
    """流配置"""
    name: str
    subjects: List[str]
    retention: str = "limits"
    max_consumers: int = -1
    max_msgs: int = -1
    max_bytes: int = -1
    max_age: int = 0
    max_msg_size: int = -1
    storage: str = "file"
    num_replicas: int = 1
    no_ack: bool = False
    duplicate_window: float = 120.0


@dataclass
class ConsumerConfig:
    """消费者配置"""
    name: str
    stream_name: str
    durable_name: Optional[str] = None
    description: Optional[str] = None
    deliver_policy: str = "all"  # all, last, new, by_start_sequence, by_start_time
    opt_start_seq: Optional[int] = None
    opt_start_time: Optional[str] = None
    ack_policy: str = "explicit"  # explicit, all, none
    ack_wait: float = 30.0
    max_deliver: int = -1
    filter_subject: Optional[str] = None
    replay_policy: str = "instant"
    max_ack_pending: int = 1000
    max_waiting: int = 512
    max_batch: int = 0
    max_expires: float = 0.0
    inactive_threshold: float = 0.0


class NATSProtocol(Protocol):
    """NATS协议接口"""
    
    async def connect(self) -> None:
        ...
    
    async def close(self) -> None:
        ...
    
    async def publish(self, subject: str, payload: bytes, reply: Optional[str] = None) -> None:
        ...
    
    async def subscribe(
        self,
        subject: str,
        callback: Callable[[NATSMessage], Coroutine],
        queue: Optional[str] = None,
    ) -> Subscription:
        ...
    
    async def request(self, subject: str, payload: bytes, timeout: float = 0.5) -> NATSMessage:
        ...


@dataclass
class NATSMessage:
    """NATS消息"""
    subject: str
    reply: Optional[str]
    data: bytes
    sid: str
    headers: Optional[Dict[str, str]] = None
    
    @property
    def payload(self) -> Any:
        """解析后的载荷"""
        try:
            return json.loads(self.data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self.data


@dataclass
class Subscription:
    """订阅对象"""
    sid: str
    subject: str
    queue: Optional[str]
    callback: Callable[[NATSMessage], Coroutine]
    max_msgs: Optional[int] = None
    received: int = 0


class NATSConnection:
    """NATS连接实现"""
    
    def __init__(self, config: NATSConfig) -> None:
        self.config = config
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._connect_id: str = ""
        self._subscriptions: Dict[str, Subscription] = {}
        self._next_sid = 0
        self._lock = asyncio.Lock()
        self._pending: Dict[str, asyncio.Future[NATSMessage]] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._pongs: List[asyncio.Future[bool]] = []
        
    async def connect(self) -> None:
        """建立连接"""
        for server in self.config.servers:
            try:
                parsed = self._parse_server_url(server)
                host, port = parsed["host"], parsed["port"]
                
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.config.connect_timeout,
                )
                
                # 发送CONNECT命令
                connect_info = {
                    "verbose": self.config.verbose,
                    "pedantic": self.config.pedantic,
                    "tls_required": False,
                    "name": self.config.name,
                    "lang": "python",
                    "version": "1.0.0",
                    "protocol": 1,
                }
                
                if self.config.user and self.config.password:
                    connect_info["user"] = self.config.user
                    connect_info["pass"] = self.config.password
                elif self.config.token:
                    connect_info["auth_token"] = self.config.token
                
                await self._send_command(f"CONNECT {json.dumps(connect_info)}")
                
                # 等待响应
                response = await self._read_line()
                if b"+OK" not in response and b"INFO" not in response:
                    raise NATSConnectionError(f"连接失败: {response}")
                
                # 如果有INFO，读取它
                if b"INFO" in response:
                    info_data = response[5:].decode().strip()
                    server_info = json.loads(info_data)
                    self._connect_id = server_info.get("server_id", "")
                
                self._connected = True
                
                # 启动读取循环
                self._read_task = asyncio.create_task(self._read_loop())
                
                # 启动pinger
                asyncio.create_task(self._ping_loop())
                
                logger.info(f"NATS连接已建立: {server}")
                return
                
            except Exception as e:
                logger.warning(f"无法连接到 {server}: {e}")
                continue
        
        raise NATSConnectionError("无法连接到任何NATS服务器")
    
    def _parse_server_url(self, url: str) -> Dict[str, Any]:
        """解析服务器URL"""
        # 简化解析，假设格式为 nats://host:port
        url = url.replace("nats://", "").replace("tls://", "")
        if ":" in url:
            host, port_str = url.rsplit(":", 1)
            port = int(port_str)
        else:
            host = url
            port = 4222
        return {"host": host, "port": port}
    
    async def _send_command(self, command: str) -> None:
        """发送命令"""
        if not self._writer:
            raise NATSConnectionError("未连接")
        
        data = (command + "\r\n").encode()
        self._writer.write(data)
        await self._writer.drain()
    
    async def _read_line(self) -> bytes:
        """读取一行"""
        if not self._reader:
            raise NATSConnectionError("未连接")
        
        line = await self._reader.readline()
        return line
    
    async def _read_bytes(self, n: int) -> bytes:
        """读取指定字节数"""
        if not self._reader:
            raise NATSConnectionError("未连接")
        
        data = await self._reader.readexactly(n)
        return data
    
    async def _read_loop(self) -> None:
        """读取循环"""
        while self._connected:
            try:
                line = await self._read_line()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(b" ", 1)
                cmd = parts[0].decode()
                
                if cmd == "MSG":
                    await self._handle_msg(parts[1] if len(parts) > 1 else b"")
                elif cmd == "PING":
                    await self._send_command("PONG")
                elif cmd == "PONG":
                    if self._pongs:
                        future = self._pongs.pop(0)
                        if not future.done():
                            future.set_result(True)
                elif cmd == "+OK":
                    pass
                elif cmd == "-ERR":
                    error_msg = parts[1].decode() if len(parts) > 1 else "未知错误"
                    logger.error(f"NATS错误: {error_msg}")
                elif cmd == "INFO":
                    pass
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"读取错误: {e}")
                await asyncio.sleep(1)
    
    async def _handle_msg(self, data: bytes) -> None:
        """处理消息"""
        # 格式: MSG <subject> <sid> <reply-to> <#bytes> 或 MSG <subject> <sid> <#bytes>
        parts = data.split(b" ")
        
        if len(parts) >= 3:
            subject = parts[0].decode()
            sid = parts[1].decode()
            
            # 检查是否有reply-to
            if len(parts) == 4:
                reply_to = None
                num_bytes = int(parts[2])
            elif len(parts) == 5:
                reply_to = parts[2].decode()
                num_bytes = int(parts[3])
            else:
                reply_to = None
                num_bytes = int(parts[2])
            
            # 读取消息体
            payload = await self._read_bytes(num_bytes)
            await self._read_bytes(2)  # 读取\r\n
            
            # 创建消息对象
            msg = NATSMessage(
                subject=subject,
                reply=reply_to,
                data=payload,
                sid=sid,
            )
            
            # 查找订阅并调用回调
            sub = self._subscriptions.get(sid)
            if sub:
                sub.received += 1
                asyncio.create_task(sub.callback(msg))
                
                # 检查是否达到最大消息数
                if sub.max_msgs and sub.received >= sub.max_msgs:
                    await self.unsubscribe(sid)
    
    async def _ping_loop(self) -> None:
        """ping循环"""
        while self._connected:
            try:
                await asyncio.sleep(self.config.ping_interval)
                
                pong_future: asyncio.Future[bool] = asyncio.Future()
                self._pongs.append(pong_future)
                
                await self._send_command("PING")
                
                try:
                    await asyncio.wait_for(pong_future, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("PONG超时")
                    self._connected = False
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping错误: {e}")
    
    async def publish(self, subject: str, payload: bytes, reply: Optional[str] = None) -> None:
        """发布消息"""
        if not self._connected:
            raise NATSConnectionError("未连接")
        
        if reply:
            command = f"PUB {subject} {reply} {len(payload)}"
        else:
            command = f"PUB {subject} {len(payload)}"
        
        await self._send_command(command)
        self._writer.write(payload + b"\r\n")
        await self._writer.drain()
    
    async def subscribe(
        self,
        subject: str,
        callback: Callable[[NATSMessage], Coroutine],
        queue: Optional[str] = None,
    ) -> str:
        """订阅主题"""
        if not self._connected:
            raise NATSConnectionError("未连接")
        
        async with self._lock:
            self._next_sid += 1
            sid = str(self._next_sid)
        
        if queue:
            command = f"SUB {subject} {queue} {sid}"
        else:
            command = f"SUB {subject} {sid}"
        
        await self._send_command(command)
        
        self._subscriptions[sid] = Subscription(
            sid=sid,
            subject=subject,
            queue=queue,
            callback=callback,
        )
        
        logger.debug(f"订阅主题: {subject}, sid: {sid}")
        return sid
    
    async def unsubscribe(self, sid: str, max_msgs: Optional[int] = None) -> None:
        """取消订阅"""
        if max_msgs:
            command = f"UNSUB {sid} {max_msgs}"
        else:
            command = f"UNSUB {sid}"
        
        await self._send_command(command)
        
        if sid in self._subscriptions and not max_msgs:
            del self._subscriptions[sid]
    
    async def request(self, subject: str, payload: bytes, timeout: float = 0.5) -> NATSMessage:
        """请求-响应"""
        # 创建inbox
        inbox = f"{self.config.inbox_prefix}.{uuid.uuid4().hex}"
        
        future: asyncio.Future[NATSMessage] = asyncio.Future()
        
        async def callback(msg: NATSMessage) -> None:
            if not future.done():
                future.set_result(msg)
        
        # 订阅响应
        sid = await self.subscribe(inbox, callback)
        
        try:
            # 发送请求
            await self.publish(subject, payload, reply=inbox)
            
            # 等待响应
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            await self.unsubscribe(sid)
    
    async def close(self) -> None:
        """关闭连接"""
        self._connected = False
        
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        
        logger.info("NATS连接已关闭")


class NATSJetStream:
    """NATS JetStream实现"""
    
    def __init__(self, conn: NATSConnection, config: JetStreamConfig) -> None:
        self.conn = conn
        self.config = config
        self._streams: Dict[str, StreamConfig] = {}
        self._consumers: Dict[str, ConsumerConfig] = {}
        
    def _api_subject(self, suffix: str) -> str:
        """构建API主题"""
        if self.config.domain:
            return f"$JS.{self.config.domain}.API.{suffix}"
        return f"{self.config.prefix}.{suffix}"
    
    async def add_stream(self, config: StreamConfig) -> Dict[str, Any]:
        """添加流"""
        subject = self._api_subject(f"STREAM.CREATE.{config.name}")
        
        payload = {
            "name": config.name,
            "subjects": config.subjects,
            "retention": config.retention,
            "max_consumers": config.max_consumers,
            "max_msgs": config.max_msgs,
            "max_bytes": config.max_bytes,
            "max_age": config.max_age,
            "max_msg_size": config.max_msg_size,
            "storage": config.storage,
            "num_replicas": config.num_replicas,
            "no_ack": config.no_ack,
            "duplicate_window": config.duplicate_window,
        }
        
        response = await self.conn.request(
            subject,
            json.dumps(payload).encode(),
            timeout=self.config.timeout,
        )
        
        result = json.loads(response.data.decode())
        if result.get("error"):
            raise NATSJetStreamError(result["error"].get("description", "未知错误"))
        
        self._streams[config.name] = config
        return result.get("config", {})
    
    async def delete_stream(self, name: str) -> bool:
        """删除流"""
        subject = self._api_subject(f"STREAM.DELETE.{name}")
        
        response = await self.conn.request(
            subject,
            b"",
            timeout=self.config.timeout,
        )
        
        result = json.loads(response.data.decode())
        return result.get("success", False)
    
    async def add_consumer(self, config: ConsumerConfig) -> Dict[str, Any]:
        """添加消费者"""
        subject = self._api_subject(f"CONSUMER.CREATE.{config.stream_name}.{config.name}")
        
        payload: Dict[str, Any] = {
            "name": config.name,
            "durable_name": config.durable_name or config.name,
            "deliver_policy": config.deliver_policy,
            "ack_policy": config.ack_policy,
            "ack_wait": int(config.ack_wait * 1e9),  # 转换为纳秒
            "max_deliver": config.max_deliver,
            "replay_policy": config.replay_policy,
            "max_ack_pending": config.max_ack_pending,
            "max_waiting": config.max_waiting,
        }
        
        if config.description:
            payload["description"] = config.description
        if config.opt_start_seq:
            payload["opt_start_seq"] = config.opt_start_seq
        if config.opt_start_time:
            payload["opt_start_time"] = config.opt_start_time
        if config.filter_subject:
            payload["filter_subject"] = config.filter_subject
        
        response = await self.conn.request(
            subject,
            json.dumps(payload).encode(),
            timeout=self.config.timeout,
        )
        
        result = json.loads(response.data.decode())
        if result.get("error"):
            raise NATSJetStreamError(result["error"].get("description", "未知错误"))
        
        self._consumers[f"{config.stream_name}.{config.name}"] = config
        return result.get("config", {})
    
    async def delete_consumer(self, stream: str, consumer: str) -> bool:
        """删除消费者"""
        subject = self._api_subject(f"CONSUMER.DELETE.{stream}.{consumer}")
        
        response = await self.conn.request(
            subject,
            b"",
            timeout=self.config.timeout,
        )
        
        result = json.loads(response.data.decode())
        return result.get("success", False)
    
    async def publish(
        self,
        subject: str,
        payload: bytes,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """发布消息到JetStream"""
        # 使用标准发布，JetStream会拦截
        await self.conn.publish(subject, payload)
        
        return {
            "stream": "",
            "seq": 0,
        }
    
    async def subscribe(
        self,
        stream: str,
        consumer: str,
        callback: Callable[[NATSMessage], Coroutine],
        config: Optional[ConsumerConfig] = None,
    ) -> str:
        """订阅JetStream消息"""
        # 创建pull消费者
        if config:
            try:
                await self.add_consumer(config)
            except NATSJetStreamError:
                pass  # 消费者可能已存在
        
        # 构建交付主题
        deliver_subject = f"$JS.ACK.{stream}.{consumer}"
        
        # 订阅交付主题
        sid = await self.conn.subscribe(deliver_subject, callback)
        
        return sid
    
    async def pull(
        self,
        stream: str,
        consumer: str,
        batch: int = 1,
        expires: Optional[float] = None,
    ) -> List[NATSMessage]:
        """拉取消息"""
        subject = self._api_subject(f"CONSUMER.MSG.NEXT.{stream}.{consumer}")
        
        request: Dict[str, Any] = {"batch": batch}
        if expires:
            request["expires"] = int(expires * 1e9)
        
        response = await self.conn.request(
            subject,
            json.dumps(request).encode(),
            timeout=(expires or 5.0) + 1.0,
        )
        
        # 解析响应
        messages = []
        data = response.data
        
        # 简化处理，实际应该解析多个消息
        if data:
            msg = NATSMessage(
                subject=subject,
                reply=None,
                data=data,
                sid="",
            )
            messages.append(msg)
        
        return messages
    
    async def ack(self, msg: NATSMessage) -> None:
        """确认消息"""
        if msg.reply:
            await self.conn.publish(msg.reply, b"+ACK")
    
    async def nak(self, msg: NATSMessage, delay: Optional[float] = None) -> None:
        """否定确认消息"""
        if msg.reply:
            payload = f"-NAK {int(delay * 1e9)}" if delay else "-NAK"
            await self.conn.publish(msg.reply, payload.encode())
    
    async def term(self, msg: NATSMessage) -> None:
        """终止消息"""
        if msg.reply:
            await self.conn.publish(msg.reply, b"+TERM")
    
    async def in_progress(self, msg: NATSMessage) -> None:
        """标记消息处理中"""
        if msg.reply:
            await self.conn.publish(msg.reply, b"+WIP")


class NATSKeyValue:
    """NATS键值存储实现"""
    
    def __init__(self, jetstream: NATSJetStream) -> None:
        self.js = jetstream
        self._buckets: Dict[str, str] = {}  # bucket_name -> stream_name
        
    def _bucket_stream(self, bucket: str) -> str:
        """获取桶对应的流名称"""
        return f"KV_{bucket}"
    
    async def create_bucket(
        self,
        bucket: str,
        description: Optional[str] = None,
        max_value_size: int = -1,
        history: int = 1,
        ttl: float = 0,
        max_bucket_size: int = -1,
        storage: str = "file",
        replicas: int = 1,
    ) -> Dict[str, Any]:
        """创建桶"""
        stream_name = self._bucket_stream(bucket)
        
        config = StreamConfig(
            name=stream_name,
            subjects=[f"$KV.{bucket}.>"],
            retention="limits",
            max_msgs_per_subject=history,
            max_bytes=max_bucket_size,
            max_age=int(ttl) if ttl > 0 else 0,
            max_msg_size=max_value_size,
            storage=storage,
            num_replicas=replicas,
        )
        
        result = await self.js.add_stream(config)
        self._buckets[bucket] = stream_name
        
        return {
            "bucket": bucket,
            "stream": stream_name,
            "config": result,
        }
    
    async def delete_bucket(self, bucket: str) -> bool:
        """删除桶"""
        stream_name = self._bucket_stream(bucket)
        result = await self.js.delete_stream(stream_name)
        if result:
            del self._buckets[bucket]
        return result
    
    async def put(self, bucket: str, key: str, value: bytes) -> int:
        """存储键值"""
        subject = f"$KV.{bucket}.{key}"
        
        await self.js.publish(subject, value)
        
        # 返回修订号（简化）
        return 1
    
    async def get(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """获取键值"""
        # 使用直接流访问获取最新值
        stream_name = self._bucket_stream(bucket)
        
        # 这里简化处理，实际应该使用消费者
        return {
            "key": key,
            "value": None,
            "revision": 0,
        }
    
    async def delete(self, bucket: str, key: str) -> bool:
        """删除键值（逻辑删除）"""
        # 发送删除标记
        subject = f"$KV.{bucket}.{key}"
        await self.js.publish(subject, b"", headers={"KV-Operation": "DEL"})
        return True
    
    async def purge(self, bucket: str, key: str) -> bool:
        """清除键值（物理删除）"""
        # 发送清除标记
        subject = f"$KV.{bucket}.{key}"
        await self.js.publish(subject, b"", headers={"KV-Operation": "PURGE"})
        return True
    
    async def keys(self, bucket: str, filter_str: str = ">") -> List[str]:
        """列出所有键"""
        # 简化实现
        return []
    
    async def history(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        """获取键的历史"""
        # 简化实现
        return []
    
    async def watch(
        self,
        bucket: str,
        key: str,
        callback: Callable[[Dict[str, Any]], Coroutine],
    ) -> str:
        """监视键变化"""
        subject = f"$KV.{bucket}.{key}"
        
        async def handler(msg: NATSMessage) -> None:
            entry = {
                "key": key,
                "value": msg.data,
                "revision": 0,
            }
            await callback(entry)
        
        sid = await self.js.conn.subscribe(subject, handler)
        return sid


class NATSObjectStore:
    """NATS对象存储实现"""
    
    def __init__(self, jetstream: NATSJetStream) -> None:
        self.js = jetstream
        self._buckets: Dict[str, str] = {}
        
    def _bucket_stream(self, bucket: str) -> str:
        """获取桶对应的流名称"""
        return f"OBJ_{bucket}"
    
    async def create_bucket(
        self,
        bucket: str,
        description: Optional[str] = None,
        ttl: float = 0,
        max_bucket_size: int = -1,
        storage: str = "file",
        replicas: int = 1,
        chunk_size: int = 128 * 1024,  # 128KB
    ) -> Dict[str, Any]:
        """创建对象存储桶"""
        stream_name = self._bucket_stream(bucket)
        
        config = StreamConfig(
            name=stream_name,
            subjects=[f"$O.{bucket}.C.", f"$O.{bucket}.M.>"],
            retention="limits",
            max_bytes=max_bucket_size,
            max_age=int(ttl) if ttl > 0 else 0,
            storage=storage,
            num_replicas=replicas,
        )
        
        result = await self.js.add_stream(config)
        self._buckets[bucket] = stream_name
        
        return {
            "bucket": bucket,
            "stream": stream_name,
            "chunk_size": chunk_size,
            "config": result,
        }
    
    async def delete_bucket(self, bucket: str) -> bool:
        """删除对象存储桶"""
        stream_name = self._bucket_stream(bucket)
        result = await self.js.delete_stream(stream_name)
        if result:
            del self._buckets[bucket]
        return result
    
    async def put(
        self,
        bucket: str,
        name: str,
        data: bytes,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """存储对象"""
        # 分块存储
        chunk_size = 128 * 1024
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        object_id = uuid.uuid4().hex
        
        # 存储块
        for i, chunk in enumerate(chunks):
            chunk_subject = f"$O.{bucket}.C.{object_id}.{i}"
            await self.js.publish(chunk_subject, chunk)
        
        # 存储元数据
        meta = {
            "name": name,
            "id": object_id,
            "size": len(data),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "description": description,
            "metadata": metadata or {},
            "mtime": time.time(),
        }
        
        meta_subject = f"$O.{bucket}.M.{name}"
        await self.js.publish(meta_subject, json.dumps(meta).encode())
        
        return meta
    
    async def get(self, bucket: str, name: str) -> Optional[bytes]:
        """获取对象"""
        # 获取元数据
        meta_subject = f"$O.{bucket}.M.{name}"
        
        # 简化实现，实际应该从流中读取
        return None
    
    async def delete(self, bucket: str, name: str) -> bool:
        """删除对象"""
        # 发送删除标记
        meta_subject = f"$O.{bucket}.M.{name}"
        await self.js.publish(meta_subject, b"", headers={"Operation": "DELETE"})
        return True
    
    async def list(self, bucket: str) -> List[Dict[str, Any]]:
        """列出对象"""
        # 简化实现
        return []
    
    async def info(self, bucket: str, name: str) -> Optional[Dict[str, Any]]:
        """获取对象信息"""
        # 简化实现
        return None


class NATSQueueGroup:
    """NATS队列组实现"""
    
    def __init__(self, conn: NATSConnection) -> None:
        self.conn = conn
        self._groups: Dict[str, Set[str]] = defaultdict(set)  # group_name -> set of sids
        
    async def subscribe(
        self,
        subject: str,
        queue: str,
        handler: Callable[[NATSMessage], Coroutine],
    ) -> str:
        """订阅队列组"""
        sid = await self.conn.subscribe(subject, handler, queue=queue)
        self._groups[queue].add(sid)
        
        logger.debug(f"队列组订阅: {subject}, 队列: {queue}, sid: {sid}")
        return sid
    
    async def unsubscribe(self, sid: str, queue: Optional[str] = None) -> None:
        """取消订阅"""
        await self.conn.unsubscribe(sid)
        
        if queue and sid in self._groups[queue]:
            self._groups[queue].remove(sid)
    
    def get_group_size(self, queue: str) -> int:
        """获取队列组大小"""
        return len(self._groups.get(queue, set()))
    
    def list_groups(self) -> List[str]:
        """列出所有队列组"""
        return list(self._groups.keys())


class NATSRequestReply:
    """NATS请求-响应模式实现"""
    
    def __init__(self, conn: NATSConnection) -> None:
        self.conn = conn
        self._handlers: Dict[str, Callable[[NATSMessage], Coroutine]] = {}
        self._inbox_prefix = f"_INBOX.{uuid.uuid4().hex}"
        self._next_inbox = 0
        self._lock = asyncio.Lock()
        
    async def request(
        self,
        subject: str,
        payload: bytes,
        timeout: float = 1.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> NATSMessage:
        """发送请求"""
        return await self.conn.request(subject, payload, timeout=timeout)
    
    async def reply(
        self,
        subject: str,
        handler: Callable[[bytes], Coroutine[None, None, bytes]],
        queue: Optional[str] = None,
    ) -> str:
        """设置回复处理器"""
        async def wrapper(msg: NATSMessage) -> None:
            try:
                response_data = await handler(msg.data)
                if msg.reply:
                    await self.conn.publish(msg.reply, response_data)
            except Exception as e:
                logger.error(f"请求处理错误: {e}")
                if msg.reply:
                    error_payload = json.dumps({"error": str(e)}).encode()
                    await self.conn.publish(msg.reply, error_payload)
        
        sid = await self.conn.subscribe(subject, wrapper, queue=queue)
        self._handlers[subject] = wrapper
        
        logger.debug(f"设置回复处理器: {subject}, sid: {sid}")
        return sid
    
    async def cancel_reply(self, subject: str, sid: str) -> None:
        """取消回复处理器"""
        await self.conn.unsubscribe(sid)
        if subject in self._handlers:
            del self._handlers[subject]
    
    async def create_inbox(self) -> str:
        """创建唯一inbox主题"""
        async with self._lock:
            self._next_inbox += 1
            return f"{self._inbox_prefix}.{self._next_inbox}"


class NATSBackend(MessageBus):
    """NATS消息总线后端"""
    
    def __init__(
        self,
        config: Optional[NATSConfig] = None,
        jetstream_config: Optional[JetStreamConfig] = None,
    ) -> None:
        self.config = config or NATSConfig()
        self.jetstream_config = jetstream_config or JetStreamConfig()
        
        self._conn: Optional[NATSConnection] = None
        self._jetstream: Optional[NATSJetStream] = None
        self._kv: Optional[NATSKeyValue] = None
        self._obj: Optional[NATSObjectStore] = None
        self._queue_group: Optional[NATSQueueGroup] = None
        self._request_reply: Optional[NATSRequestReply] = None
        
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._subscriptions: Dict[str, str] = {}  # topic -> sid
        self._running = False
        
    async def start(self) -> None:
        """启动后端"""
        if self._running:
            return
        
        self._conn = NATSConnection(self.config)
        await self._conn.connect()
        
        # 初始化组件
        self._jetstream = NATSJetStream(self._conn, self.jetstream_config)
        self._kv = NATSKeyValue(self._jetstream)
        self._obj = NATSObjectStore(self._jetstream)
        self._queue_group = NATSQueueGroup(self._conn)
        self._request_reply = NATSRequestReply(self._conn)
        
        self._running = True
        logger.info("NATS后端已启动")
    
    async def stop(self) -> None:
        """停止后端"""
        if not self._running:
            return
        
        # 取消所有订阅
        for topic, sid in self._subscriptions.items():
            await self._conn.unsubscribe(sid)
        self._subscriptions.clear()
        
        if self._conn:
            await self._conn.close()
        
        self._running = False
        logger.info("NATS后端已停止")
    
    async def publish(self, topic: str, message: Message) -> bool:
        """发布消息"""
        if not self._running:
            raise NATSError("后端未启动")
        
        try:
            data = json.dumps({
                "id": message.id,
                "topic": message.topic,
                "payload": message.payload,
                "headers": message.headers,
                "timestamp": message.timestamp,
            }).encode()
            
            await self._conn.publish(topic, data)
            return True
        except Exception as e:
            logger.error(f"发布失败: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """订阅主题"""
        if not self._running:
            raise NATSError("后端未启动")
        
        self._handlers[topic].append(handler)
        
        # 如果还没有订阅，创建订阅
        if topic not in self._subscriptions:
            async def callback(msg: NATSMessage) -> None:
                await self._on_message(topic, msg)
            
            sid = await self._conn.subscribe(topic, callback)
            self._subscriptions[topic] = sid
    
    async def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """取消订阅"""
        if topic in self._handlers:
            self._handlers[topic].remove(handler)
            
            # 如果没有处理器了，取消订阅
            if not self._handlers[topic] and topic in self._subscriptions:
                sid = self._subscriptions.pop(topic)
                await self._conn.unsubscribe(sid)
    
    async def _on_message(self, topic: str, msg: NATSMessage) -> None:
        """处理消息"""
        try:
            data = json.loads(msg.data.decode())
            
            message = Message(
                id=data.get("id", ""),
                topic=data.get("topic", topic),
                payload=data.get("payload"),
                headers=data.get("headers", {}),
                timestamp=data.get("timestamp", 0.0),
            )
            
            handlers = self._handlers.get(topic, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"消息处理错误: {e}")
                    
        except Exception as e:
            logger.error(f"消息解析错误: {e}")
    
    async def request(self, topic: str, message: Message, timeout: float = 30.0) -> Optional[Message]:
        """请求-响应模式"""
        try:
            data = json.dumps({
                "id": message.id,
                "topic": message.topic,
                "payload": message.payload,
                "headers": message.headers,
                "timestamp": message.timestamp,
            }).encode()
            
            response = await self._conn.request(topic, data, timeout=timeout)
            
            response_data = json.loads(response.data.decode())
            return Message(
                id=response_data.get("id", ""),
                topic=response_data.get("topic", ""),
                payload=response_data.get("payload"),
                headers=response_data.get("headers", {}),
                timestamp=response_data.get("timestamp", 0.0),
            )
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"请求失败: {e}")
            return None
    
    async def subscribe_queue(
        self,
        topic: str,
        queue: str,
        handler: MessageHandler,
    ) -> str:
        """订阅队列组"""
        async def callback(msg: NATSMessage) -> None:
            await self._on_message(topic, msg)
        
        return await self._queue_group.subscribe(topic, queue, callback)
    
    async def reply(
        self,
        topic: str,
        handler: Callable[[Message], Coroutine[None, None, Message]],
        queue: Optional[str] = None,
    ) -> str:
        """设置请求回复处理器"""
        async def wrapper(data: bytes) -> bytes:
            try:
                request_data = json.loads(data.decode())
                request_msg = Message(
                    id=request_data.get("id", ""),
                    topic=request_data.get("topic", ""),
                    payload=request_data.get("payload"),
                    headers=request_data.get("headers", {}),
                    timestamp=request_data.get("timestamp", 0.0),
                )
                
                response_msg = await handler(request_msg)
                
                return json.dumps({
                    "id": response_msg.id,
                    "topic": response_msg.topic,
                    "payload": response_msg.payload,
                    "headers": response_msg.headers,
                    "timestamp": response_msg.timestamp,
                }).encode()
            except Exception as e:
                return json.dumps({"error": str(e)}).encode()
        
        return await self._request_reply.reply(topic, wrapper, queue=queue)
    
    @property
    def jetstream(self) -> Optional[NATSJetStream]:
        """获取JetStream组件"""
        return self._jetstream
    
    @property
    def kv(self) -> Optional[NATSKeyValue]:
        """获取键值存储组件"""
        return self._kv
    
    @property
    def obj(self) -> Optional[NATSObjectStore]:
        """获取对象存储组件"""
        return self._obj
    
    @property
    def queue(self) -> Optional[NATSQueueGroup]:
        """获取队列组组件"""
        return self._queue_group
    
    @property
    def request_reply(self) -> Optional[NATSRequestReply]:
        """获取请求-响应组件"""
        return self._request_reply
