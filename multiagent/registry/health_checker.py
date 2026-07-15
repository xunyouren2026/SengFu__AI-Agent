"""
健康检查器
周期性探测Agent的/health端点
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .schema import AgentMetadata, AgentStatus, AgentAddress, HealthStatus


class HealthCheckMethod(Enum):
    """健康检查方法"""
    HTTP = "http"           # HTTP GET请求
    TCP = "tcp"             # TCP连接检查
    PING = "ping"           # ICMP ping
    CUSTOM = "custom"       # 自定义检查


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    method: HealthCheckMethod = HealthCheckMethod.HTTP
    path: str = "/health"
    port: Optional[int] = None
    timeout: float = 5.0
    interval: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    expected_status: int = 200
    expected_body: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    agent_id: str
    is_healthy: bool
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_health_status(self) -> HealthStatus:
        """转换为HealthStatus"""
        return HealthStatus(
            agent_id=self.agent_id,
            status=AgentStatus.HEALTHY if self.is_healthy else AgentStatus.UNHEALTHY,
            last_check=self.timestamp,
            response_time_ms=self.response_time_ms,
            error_message=self.error_message
        )


class HealthChecker:
    """
    健康检查器
    
    周期性探测Agent健康状态，支持：
    - HTTP健康检查
    - TCP连接检查
    - 并发批量检查
    - 状态变更回调
    - 自适应检查间隔
    """

    def __init__(
        self,
        default_config: Optional[HealthCheckConfig] = None,
        max_workers: int = 10,
        adaptive_interval: bool = True
    ):
        """
        初始化健康检查器
        
        Args:
            default_config: 默认检查配置
            max_workers: 最大并发工作线程
            adaptive_interval: 是否启用自适应检查间隔
        """
        self._default_config = default_config or HealthCheckConfig()
        self._max_workers = max_workers
        self._adaptive_interval = adaptive_interval
        
        # Agent配置: agent_id -> (metadata, config)
        self._agents: Dict[str, Tuple[AgentMetadata, HealthCheckConfig]] = {}
        
        # 检查状态
        self._check_states: Dict[str, Dict[str, Any]] = {}
        
        # 回调函数
        self._on_status_change: Optional[Callable[[str, AgentStatus, AgentStatus], None]] = None
        self._on_check_complete: Optional[Callable[[HealthCheckResult], None]] = None
        
        # 线程控制
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 统计信息
        self._stats = {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "status_changes": 0
        }

    def set_callbacks(
        self,
        on_status_change: Optional[Callable[[str, AgentStatus, AgentStatus], None]] = None,
        on_check_complete: Optional[Callable[[HealthCheckResult], None]] = None
    ) -> None:
        """设置回调函数"""
        self._on_status_change = on_status_change
        self._on_check_complete = on_check_complete

    def register_agent(
        self,
        agent: AgentMetadata,
        config: Optional[HealthCheckConfig] = None
    ) -> None:
        """
        注册Agent进行健康检查
        
        Args:
            agent: Agent元数据
            config: 健康检查配置（None使用默认配置）
        """
        config = config or self._default_config
        
        with self._lock:
            self._agents[agent.agent_id] = (agent, config)
            self._check_states[agent.agent_id] = {
                "consecutive_successes": 0,
                "consecutive_failures": 0,
                "current_status": AgentStatus.UNKNOWN,
                "last_check": None,
                "next_check": time.time()
            }

    def unregister_agent(self, agent_id: str) -> bool:
        """
        注销Agent的健康检查
        
        Args:
            agent_id: Agent ID
            
        Returns:
            是否成功注销
        """
        with self._lock:
            if agent_id not in self._agents:
                return False
            
            del self._agents[agent_id]
            if agent_id in self._check_states:
                del self._check_states[agent_id]
            return True

    def start(self) -> None:
        """启动健康检查器"""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._check_thread = threading.Thread(
                target=self._check_loop,
                daemon=True,
                name="HealthChecker"
            )
            self._check_thread.start()

    def stop(self) -> None:
        """停止健康检查器"""
        with self._lock:
            self._running = False
        
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        
        self._executor.shutdown(wait=True)

    def check_once(self, agent_id: Optional[str] = None) -> List[HealthCheckResult]:
        """
        执行一次健康检查
        
        Args:
            agent_id: 指定Agent ID（None则检查所有）
            
        Returns:
            检查结果列表
        """
        with self._lock:
            if agent_id:
                if agent_id not in self._agents:
                    return []
                agents_to_check = {agent_id: self._agents[agent_id]}
            else:
                agents_to_check = dict(self._agents)
        
        results = []
        futures = {}
        
        # 提交检查任务
        for aid, (agent, config) in agents_to_check.items():
            future = self._executor.submit(self._check_agent, agent, config)
            futures[future] = aid
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            self._process_result(result)
        
        return results

    def _check_loop(self) -> None:
        """健康检查主循环"""
        while True:
            with self._lock:
                if not self._running:
                    break
            
            now = time.time()
            agents_to_check = []
            
            with self._lock:
                for agent_id, (agent, config) in self._agents.items():
                    state = self._check_states.get(agent_id)
                    if state and now >= state.get("next_check", 0):
                        agents_to_check.append((agent, config))
            
            if agents_to_check:
                self._check_batch(agents_to_check)
            
            # 短暂休眠
            time.sleep(0.5)

    def _check_batch(self, agents: List[Tuple[AgentMetadata, HealthCheckConfig]]) -> None:
        """批量检查"""
        futures = {}
        
        for agent, config in agents:
            future = self._executor.submit(self._check_agent, agent, config)
            futures[future] = agent.agent_id
        
        for future in as_completed(futures):
            result = future.result()
            self._process_result(result)

    def _check_agent(
        self,
        agent: AgentMetadata,
        config: HealthCheckConfig
    ) -> HealthCheckResult:
        """
        检查单个Agent
        
        Args:
            agent: Agent元数据
            config: 检查配置
            
        Returns:
            检查结果
        """
        start_time = time.time()
        
        try:
            if config.method == HealthCheckMethod.HTTP:
                result = self._http_check(agent, config)
            elif config.method == HealthCheckMethod.TCP:
                result = self._tcp_check(agent, config)
            else:
                result = HealthCheckResult(
                    agent_id=agent.agent_id,
                    is_healthy=False,
                    response_time_ms=0.0,
                    error_message=f"Unsupported check method: {config.method.value}"
                )
        except Exception as e:
            result = HealthCheckResult(
                agent_id=agent.agent_id,
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
        
        return result

    def _http_check(
        self,
        agent: AgentMetadata,
        config: HealthCheckConfig
    ) -> HealthCheckResult:
        """HTTP健康检查"""
        start_time = time.time()
        
        port = config.port or agent.address.port
        url = f"{agent.address.protocol}://{agent.address.host}:{port}{config.path}"
        
        req = urllib.request.Request(
            url,
            headers=config.headers,
            method="GET"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=config.timeout) as response:
                response_time_ms = (time.time() - start_time) * 1000
                status_code = response.getcode()
                body = response.read().decode('utf-8')
                
                is_healthy = status_code == config.expected_status
                
                if config.expected_body and config.expected_body not in body:
                    is_healthy = False
                
                return HealthCheckResult(
                    agent_id=agent.agent_id,
                    is_healthy=is_healthy,
                    response_time_ms=response_time_ms,
                    status_code=status_code,
                    details={"body": body[:1024]}  # 限制body大小
                )
                
        except urllib.error.HTTPError as e:
            return HealthCheckResult(
                agent_id=agent.agent_id,
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                status_code=e.code,
                error_message=f"HTTP {e.code}"
            )
        except urllib.error.URLError as e:
            return HealthCheckResult(
                agent_id=agent.agent_id,
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Connection failed: {e.reason}"
            )

    def _tcp_check(
        self,
        agent: AgentMetadata,
        config: HealthCheckConfig
    ) -> HealthCheckResult:
        """TCP连接检查"""
        import socket
        
        start_time = time.time()
        
        port = config.port or agent.address.port
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(config.timeout)
            result = sock.connect_ex((agent.address.host, port))
            sock.close()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if result == 0:
                return HealthCheckResult(
                    agent_id=agent.agent_id,
                    is_healthy=True,
                    response_time_ms=response_time_ms
                )
            else:
                return HealthCheckResult(
                    agent_id=agent.agent_id,
                    is_healthy=False,
                    response_time_ms=response_time_ms,
                    error_message=f"TCP connection failed with code {result}"
                )
        except Exception as e:
            return HealthCheckResult(
                agent_id=agent.agent_id,
                is_healthy=False,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

    def _process_result(self, result: HealthCheckResult) -> None:
        """处理检查结果"""
        agent_id = result.agent_id
        
        with self._lock:
            self._stats["checks_total"] += 1
            
            if agent_id not in self._check_states:
                return
            
            state = self._check_states[agent_id]
            state["last_check"] = time.time()
            
            if result.is_healthy:
                self._stats["checks_passed"] += 1
                state["consecutive_successes"] += 1
                state["consecutive_failures"] = 0
            else:
                self._stats["checks_failed"] += 1
                state["consecutive_failures"] += 1
                state["consecutive_successes"] = 0
            
            # 状态转换逻辑
            old_status = state["current_status"]
            new_status = old_status
            
            if result.is_healthy:
                if state["consecutive_successes"] >= self._default_config.healthy_threshold:
                    new_status = AgentStatus.HEALTHY
            else:
                if state["consecutive_failures"] >= self._default_config.unhealthy_threshold:
                    new_status = AgentStatus.UNHEALTHY
            
            # 计算下次检查时间
            if self._adaptive_interval:
                if new_status == AgentStatus.HEALTHY:
                    # 健康时延长检查间隔
                    interval = self._default_config.interval * 1.5
                elif new_status == AgentStatus.UNHEALTHY:
                    # 不健康时缩短检查间隔
                    interval = self._default_config.interval * 0.5
                else:
                    interval = self._default_config.interval
            else:
                interval = self._default_config.interval
            
            state["next_check"] = time.time() + interval
            
            # 状态变更处理
            if new_status != old_status:
                state["current_status"] = new_status
                self._stats["status_changes"] += 1
                
                if self._on_status_change:
                    try:
                        self._on_status_change(agent_id, old_status, new_status)
                    except Exception:
                        pass
        
        # 检查完成回调
        if self._on_check_complete:
            try:
                self._on_check_complete(result)
            except Exception:
                pass

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """获取Agent当前健康状态"""
        with self._lock:
            state = self._check_states.get(agent_id)
            if state:
                return state["current_status"]
            return None

    def get_all_statuses(self) -> Dict[str, AgentStatus]:
        """获取所有Agent的健康状态"""
        with self._lock:
            return {
                aid: state["current_status"]
                for aid, state in self._check_states.items()
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats["monitored_agents"] = len(self._agents)
            return stats

    def __enter__(self) -> HealthChecker:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class PassiveHealthChecker:
    """
    被动健康检查器
    
    基于心跳和报告来跟踪健康状态，而非主动探测
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        max_missed_heartbeats: int = 3
    ):
        self._heartbeat_timeout = heartbeat_timeout
        self._max_missed_heartbeats = max_missed_heartbeats
        
        self._agent_heartbeats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def record_heartbeat(self, agent_id: str) -> None:
        """记录心跳"""
        with self._lock:
            if agent_id not in self._agent_heartbeats:
                self._agent_heartbeats[agent_id] = {
                    "last_heartbeat": time.time(),
                    "missed_count": 0,
                    "status": AgentStatus.HEALTHY
                }
            else:
                self._agent_heartbeats[agent_id]["last_heartbeat"] = time.time()
                self._agent_heartbeats[agent_id]["missed_count"] = 0
                self._agent_heartbeats[agent_id]["status"] = AgentStatus.HEALTHY

    def check_health(self, agent_id: str) -> AgentStatus:
        """检查Agent健康状态"""
        with self._lock:
            if agent_id not in self._agent_heartbeats:
                return AgentStatus.UNKNOWN
            
            info = self._agent_heartbeats[agent_id]
            elapsed = time.time() - info["last_heartbeat"]
            
            if elapsed > self._heartbeat_timeout:
                info["missed_count"] += 1
                
                if info["missed_count"] >= self._max_missed_heartbeats:
                    info["status"] = AgentStatus.UNHEALTHY
                    return AgentStatus.UNHEALTHY
                
                return AgentStatus.UNKNOWN
            
            return AgentStatus.HEALTHY

    def get_all_unhealthy(self) -> List[str]:
        """获取所有不健康的Agent"""
        unhealthy = []
        with self._lock:
            for agent_id in list(self._agent_heartbeats.keys()):
                if self.check_health(agent_id) == AgentStatus.UNHEALTHY:
                    unhealthy.append(agent_id)
        return unhealthy
