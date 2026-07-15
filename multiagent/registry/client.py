"""
注册客户端
Agent启动时注册元数据并维护心跳
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from typing import Any, Callable, Dict, List, Optional

from .schema import AgentMetadata, AgentRegistration, AgentStatus, AgentAddress


class RegistryClientError(Exception):
    """注册客户端错误"""
    pass


class RegistrationFailedError(RegistryClientError):
    """注册失败错误"""
    pass


class HeartbeatFailedError(RegistryClientError):
    """心跳失败错误"""
    pass


class RegistryClient:
    """
    Agent注册客户端
    
    负责向注册中心注册Agent元数据，并定期发送心跳维护租约
    """

    def __init__(
        self,
        registry_url: str,
        agent_metadata: AgentMetadata,
        heartbeat_interval: float = 10.0,
        retry_interval: float = 5.0,
        max_retries: int = 3,
        timeout: float = 5.0
    ):
        """
        初始化注册客户端
        
        Args:
            registry_url: 注册中心地址
            agent_metadata: Agent元数据
            heartbeat_interval: 心跳间隔（秒）
            retry_interval: 重试间隔（秒）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self._registry_url = registry_url.rstrip('/')
        self._metadata = agent_metadata
        self._heartbeat_interval = heartbeat_interval
        self._retry_interval = retry_interval
        self._max_retries = max_retries
        self._timeout = timeout
        
        self._registered = False
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 状态回调
        self._status_callbacks: List[Callable[[str, Any], None]] = []
        
        # 统计信息
        self._stats = {
            "heartbeats_sent": 0,
            "heartbeats_failed": 0,
            "last_heartbeat_time": None,
            "registration_time": None
        }

    @property
    def agent_id(self) -> str:
        """获取Agent ID"""
        return self._metadata.agent_id

    @property
    def is_registered(self) -> bool:
        """检查是否已注册"""
        with self._lock:
            return self._registered

    def add_status_callback(self, callback: Callable[[str, Any], None]) -> None:
        """添加状态变更回调"""
        self._status_callbacks.append(callback)

    def remove_status_callback(self, callback: Callable[[str, Any], None]) -> None:
        """移除状态变更回调"""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)

    def _notify_status(self, event: str, data: Any) -> None:
        """通知状态变更"""
        for callback in self._status_callbacks:
            try:
                callback(event, data)
            except Exception:
                pass

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        retries: int = 0
    ) -> Dict[str, Any]:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            data: 请求数据
            retries: 当前重试次数
            
        Returns:
            响应数据
            
        Raises:
            RegistryClientError: 请求失败
        """
        url = f"{self._registry_url}{path}"
        
        try:
            if data is not None:
                body = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={'Content-Type': 'application/json'},
                    method=method
                )
            else:
                req = urllib.request.Request(url, method=method)
            
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                response_body = response.read().decode('utf-8')
                if response_body:
                    return json.loads(response_body)
                return {}
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise RegistryClientError(f"HTTP {e.code}: {error_body}")
        except urllib.error.URLError as e:
            if retries < self._max_retries:
                time.sleep(self._retry_interval)
                return self._make_request(method, path, data, retries + 1)
            raise RegistryClientError(f"Connection failed: {e.reason}")
        except Exception as e:
            raise RegistryClientError(f"Request failed: {e}")

    def register(self, endpoints: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        注册Agent到注册中心
        
        Args:
            endpoints: 服务端点列表
            
        Returns:
            注册是否成功
            
        Raises:
            RegistrationFailedError: 注册失败
        """
        registration = AgentRegistration(
            metadata=self._metadata,
            endpoints=endpoints or [],
            api_version="1.0.0"
        )
        
        try:
            response = self._make_request(
                "POST",
                "/api/v1/agents/register",
                registration.to_dict()
            )
            
            with self._lock:
                self._registered = True
                self._stats["registration_time"] = time.time()
            
            self._notify_status("registered", {"agent_id": self.agent_id})
            return True
            
        except RegistryClientError as e:
            raise RegistrationFailedError(f"Failed to register: {e}")

    def deregister(self) -> bool:
        """
        从注册中心注销Agent
        
        Returns:
            注销是否成功
        """
        with self._lock:
            self._running = False
            
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval + 1)
        
        try:
            self._make_request(
                "DELETE",
                f"/api/v1/agents/{self.agent_id}"
            )
        except RegistryClientError:
            pass  # 忽略注销错误
        
        with self._lock:
            self._registered = False
        
        self._notify_status("deregistered", {"agent_id": self.agent_id})
        return True

    def heartbeat(self) -> bool:
        """
        发送心跳到注册中心
        
        Returns:
            心跳是否成功
        """
        try:
            response = self._make_request(
                "POST",
                f"/api/v1/agents/{self.agent_id}/heartbeat"
            )
            
            with self._lock:
                self._stats["heartbeats_sent"] += 1
                self._stats["last_heartbeat_time"] = time.time()
            
            self._notify_status("heartbeat_success", {"agent_id": self.agent_id})
            return True
            
        except RegistryClientError as e:
            with self._lock:
                self._stats["heartbeats_failed"] += 1
            
            self._notify_status("heartbeat_failed", {
                "agent_id": self.agent_id,
                "error": str(e)
            })
            return False

    def start_heartbeat(self) -> None:
        """启动心跳线程"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name=f"Heartbeat-{self.agent_id}"
            )
            self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """停止心跳线程"""
        with self._lock:
            self._running = False
        
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval + 1)

    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        consecutive_failures = 0
        
        while True:
            with self._lock:
                if not self._running:
                    break
            
            success = self.heartbeat()
            
            if success:
                consecutive_failures = 0
                sleep_time = self._heartbeat_interval
            else:
                consecutive_failures += 1
                # 指数退避
                sleep_time = min(
                    self._retry_interval * (2 ** (consecutive_failures - 1)),
                    60.0  # 最大60秒
                )
                
                # 连续失败过多，可能需要重新注册
                if consecutive_failures >= 5:
                    self._notify_status("reconnect_needed", {
                        "agent_id": self.agent_id,
                        "failures": consecutive_failures
                    })
            
            # 分段睡眠以便及时响应停止信号
            slept = 0.0
            while slept < sleep_time:
                with self._lock:
                    if not self._running:
                        break
                time.sleep(0.5)
                slept += 0.5

    def update_metadata(self, **kwargs) -> bool:
        """
        更新Agent元数据
        
        Args:
            **kwargs: 要更新的字段
            
        Returns:
            更新是否成功
        """
        for key, value in kwargs.items():
            if hasattr(self._metadata, key):
                setattr(self._metadata, key, value)
        
        try:
            response = self._make_request(
                "PUT",
                f"/api/v1/agents/{self.agent_id}",
                self._metadata.to_dict()
            )
            self._notify_status("metadata_updated", {"agent_id": self.agent_id})
            return True
        except RegistryClientError:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return self._stats.copy()

    def start(self) -> bool:
        """
        启动客户端（注册并启动心跳）
        
        Returns:
            启动是否成功
        """
        if not self.register():
            return False
        self.start_heartbeat()
        return True

    def stop(self) -> None:
        """停止客户端"""
        self.stop_heartbeat()
        self.deregister()

    def __enter__(self) -> RegistryClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class SimpleRegistryClient:
    """
    简化版注册客户端
    
    适用于简单的注册场景，自动处理生命周期
    """

    def __init__(
        self,
        registry_url: str,
        agent_id: str,
        agent_name: str,
        version: str,
        address: AgentAddress,
        capabilities: Optional[List[str]] = None,
        heartbeat_interval: float = 10.0
    ):
        """
        初始化简化版客户端
        
        Args:
            registry_url: 注册中心地址
            agent_id: Agent唯一标识
            agent_name: Agent名称
            version: Agent版本
            address: Agent网络地址
            capabilities: 能力标签列表
            heartbeat_interval: 心跳间隔
        """
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=agent_name,
            version=version,
            address=address,
            capabilities=set(capabilities or [])
        )
        
        self._client = RegistryClient(
            registry_url=registry_url,
            agent_metadata=metadata,
            heartbeat_interval=heartbeat_interval
        )

    def start(self) -> bool:
        """启动客户端"""
        return self._client.start()

    def stop(self) -> None:
        """停止客户端"""
        self._client.stop()

    def __enter__(self) -> SimpleRegistryClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
