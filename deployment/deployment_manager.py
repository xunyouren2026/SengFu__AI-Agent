"""
AGI Unified Framework - 生产环境部署管理模块
Production Deployment Manager

实现完整的AI模型生产环境部署方案，包括：
- 部署配置管理
- 模型服务与负载均衡
- API网关与流量控制
- 容器编排与自动扩缩
- 模型注册与版本管理
- 监控告警与SLA管理
- CI/CD流水线
- 边缘部署与移动端部署
"""

import time
import hashlib
import json
import threading
import heapq
import random
import struct
import base64
import copy
import re
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Callable,
    Set, Union, NamedTuple
)
from datetime import datetime, timedelta
from functools import wraps, lru_cache


# ============================================================================
# 第一部分：基础数据结构与枚举
# ============================================================================

class DeploymentEnvironment(Enum):
    """部署环境枚举"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"
    MOBILE = "mobile"


class ContainerState(Enum):
    """容器状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class DeploymentStrategy(Enum):
    """部署策略"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class LoadBalanceAlgorithm(Enum):
    """负载均衡算法"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"


class RateLimitAlgorithm(Enum):
    """限流算法"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class PipelineStage(Enum):
    """流水线阶段"""
    SOURCE = "source"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    STAGING_DEPLOY = "staging_deploy"
    INTEGRATION_TEST = "integration_test"
    PRODUCTION_DEPLOY = "production_deploy"
    VERIFICATION = "verification"
    COMPLETE = "complete"
    FAILED = "failed"


class ScalingPolicy(Enum):
    """扩缩策略"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_RATE = "request_rate"
    CUSTOM_METRIC = "custom_metric"


class ModelFormat(Enum):
    """模型格式"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"
    COREML = "coreml"
    OPENVINO = "openvino"


class AlertSeverity(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# ============================================================================
# 第二部分：部署配置 (DeploymentConfig)
# ============================================================================

@dataclass
class ServerSettings:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    max_connections: int = 10000
    connection_timeout: float = 30.0
    request_timeout: float = 120.0
    keepalive_timeout: float = 75.0
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    compression_enabled: bool = True
    access_log_enabled: bool = True


@dataclass
class ModelSettings:
    """模型服务配置"""
    model_name: str = "default_model"
    model_version: str = "1.0.0"
    model_path: str = "/models/default"
    device: str = "auto"
    dtype: str = "float32"
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_kv_cache: bool = True
    kv_cache_size: int = 4096
    enable_speculative_decoding: bool = False
    enable_chunked_prefill: bool = True
    warmup_requests: int = 5


@dataclass
class ScalingSettings:
    """自动扩缩配置"""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_request_rate: float = 100.0
    scale_up_cooldown: int = 60
    scale_down_cooldown: int = 300
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    stabilization_window: int = 300


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    enabled: bool = True
    interval: float = 10.0
    timeout: float = 5.0
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    endpoint: str = "/health"
    initial_delay: float = 30.0
    grpc_enabled: bool = False


@dataclass
class RateLimitConfig:
    """限流配置"""
    enabled: bool = True
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    requests_per_second: float = 100.0
    burst_size: int = 200
    window_size: float = 60.0
    max_tokens_per_minute: int = 10000
    max_tokens_per_hour: int = 100000
    per_ip_enabled: bool = True
    per_ip_requests_per_second: float = 10.0
    retry_after_header: bool = True


@dataclass
class DeploymentConfig:
    """部署配置 - 统一管理所有部署相关配置"""
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    server: ServerSettings = field(default_factory=ServerSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    scaling: ScalingSettings = field(default_factory=ScalingSettings)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # 元数据
    deployment_name: str = "default_deployment"
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        result = {}
        for fld in self.__dataclass_fields__:
            val = getattr(self, fld)
            if isinstance(val, Enum):
                result[fld] = val.value
            elif hasattr(val, 'to_dict'):
                result[fld] = val.to_dict()
            elif isinstance(val, (list, dict, str, int, float, bool)):
                result[fld] = val
            else:
                result[fld] = str(val)
        return result

    def update(self, **kwargs) -> None:
        """更新配置字段"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()

    def validate(self) -> List[str]:
        """验证配置合法性"""
        errors = []
        if self.server.port < 1 or self.server.port > 65535:
            errors.append(f"无效端口号: {self.server.port}")
        if self.server.workers < 1:
            errors.append(f"Worker数量必须 >= 1")
        if self.scaling.min_replicas < 1:
            errors.append("最小副本数必须 >= 1")
        if self.scaling.min_replicas > self.scaling.max_replicas:
            errors.append("最小副本数不能大于最大副本数")
        if self.model.max_batch_size < 1:
            errors.append("最大批处理大小必须 >= 1")
        return errors


# ============================================================================
# 第三部分：模型服务 (ModelServer)
# ============================================================================

class Request:
    """HTTP请求模拟"""
    _id_counter = 0
    _lock = threading.Lock()

    def __init__(self, method: str = "POST", path: str = "/",
                 headers: Optional[Dict[str, str]] = None,
                 body: Optional[Dict[str, Any]] = None,
                 priority: int = 5):
        with Request._lock:
            Request._id_counter += 1
            self.id = Request._id_counter
        self.method = method.upper()
        self.path = path
        self.headers = headers or {}
        self.body = body or {}
        self.priority = priority  # 1=最高, 10=最低
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.response: Optional["Response"] = None
        self.timeout: float = self.headers.get("X-Timeout", 30.0)
        self.client_ip = self.headers.get("X-Forwarded-For", "127.0.0.1")

    @property
    def elapsed(self) -> float:
        """请求已用时间"""
        if self.completed_at:
            return self.completed_at - self.created_at
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """是否已超时"""
        return self.elapsed > self.timeout

    def __lt__(self, other):
        """优先队列比较：优先级数值越小越优先，同优先级按时间排序"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class Response:
    """HTTP响应模拟"""
    def __init__(self, status_code: int = 200, body: Any = None,
                 headers: Optional[Dict[str, str]] = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status_code": self.status_code,
            "body": self.body,
            "headers": self.headers,
        }


class BackendInstance:
    """后端服务实例"""
    def __init__(self, instance_id: str, host: str, port: int,
                 weight: int = 1, max_connections: int = 1000):
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.weight = weight
        self.max_connections = max_connections
        self.active_connections = 0
        self.total_requests = 0
        self.total_errors = 0
        self.is_healthy = True
        self.last_health_check = 0.0
        self.avg_latency_ms = 0.0
        self._latency_samples: deque = deque(maxlen=100)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_available(self) -> bool:
        return (self.is_healthy and
                self.active_connections < self.max_connections)

    def record_latency(self, latency_ms: float) -> None:
        self._latency_samples.append(latency_ms)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def get_load(self) -> float:
        """获取负载百分比"""
        return self.active_connections / max(self.max_connections, 1)


class ModelServer:
    """
    模型服务引擎
    - HTTP请求处理模拟
    - 请求路由
    - 负载均衡 (轮询、最少连接、加权)
    - 优先级请求队列
    - 超时处理
    - 优雅关闭
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.backends: List[BackendInstance] = []
        self.request_queue: List[Request] = []
        self._queue_lock = threading.Lock()
        self._rr_index = 0
        self._rr_lock = threading.Lock()
        self._running = False
        self._shutdown_event = threading.Event()
        self._worker_threads: List[threading.Thread] = []
        self._request_handlers: Dict[str, Callable] = {}
        self._middleware_chain: List[Callable] = []
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "timeout_requests": 0,
            "rejected_requests": 0,
        }
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认请求处理器"""
        self._request_handlers["/health"] = self._handle_health
        self._request_handlers["/predict"] = self._handle_predict
        self._request_handlers["/embeddings"] = self._handle_embeddings
        self._request_handlers["/metrics"] = self._handle_metrics

    def _handle_health(self, request: Request) -> Response:
        """健康检查端点"""
        healthy_backends = sum(1 for b in self.backends if b.is_healthy)
        return Response(200, {
            "status": "healthy" if healthy_backends > 0 else "degraded",
            "backends": {
                "total": len(self.backends),
                "healthy": healthy_backends,
            },
            "uptime": time.time(),
        })

    def _handle_predict(self, request: Request) -> Response:
        """推理端点"""
        body = request.body or {}
        prompt = body.get("prompt", "")
        if not prompt:
            return Response(400, {"error": "缺少 prompt 参数"})
        # 模拟推理延迟
        latency = random.uniform(10, 200)
        time.sleep(latency / 1000.0)
        return Response(200, {
            "model": self.config.model.model_name,
            "version": self.config.model.model_version,
            "output": f"[模拟推理结果] 输入: {prompt[:50]}...",
            "tokens": random.randint(10, 100),
            "latency_ms": round(latency, 2),
        })

    def _handle_embeddings(self, request: Request) -> Response:
        """嵌入向量端点"""
        body = request.body or {}
        text = body.get("text", "")
        if not text:
            return Response(400, {"error": "缺少 text 参数"})
        dim = body.get("dimension", 768)
        embedding = [random.gauss(0, 1) for _ in range(dim)]
        norm = sum(x * x for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]
        return Response(200, {
            "embedding": embedding[:10],  # 截断展示
            "dimension": dim,
            "model": self.config.model.model_name,
        })

    def _handle_metrics(self, request: Request) -> Response:
        """指标端点"""
        return Response(200, self._stats)

    def add_backend(self, instance_id: str, host: str, port: int,
                    weight: int = 1) -> BackendInstance:
        """添加后端实例"""
        instance = BackendInstance(
            instance_id=instance_id,
            host=host,
            port=port,
            weight=weight,
            max_connections=self.config.server.max_connections // max(len(self.backends) + 1, 1),
        )
        self.backends.append(instance)
        return instance

    def remove_backend(self, instance_id: str) -> bool:
        """移除后端实例"""
        for i, b in enumerate(self.backends):
            if b.instance_id == instance_id:
                self.backends.pop(i)
                return True
        return False

    def route_request(self, request: Request,
                      algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.ROUND_ROBIN
                      ) -> Optional[BackendInstance]:
        """
        请求路由 - 根据负载均衡算法选择后端
        """
        available = [b for b in self.backends if b.is_available]
        if not available:
            return None

        if algorithm == LoadBalanceAlgorithm.ROUND_ROBIN:
            return self._lb_round_robin(available)
        elif algorithm == LoadBalanceAlgorithm.LEAST_CONNECTIONS:
            return self._lb_least_connections(available)
        elif algorithm == LoadBalanceAlgorithm.WEIGHTED:
            return self._lb_weighted(available)
        elif algorithm == LoadBalanceAlgorithm.RANDOM:
            return random.choice(available)
        elif algorithm == LoadBalanceAlgorithm.CONSISTENT_HASH:
            return self._lb_consistent_hash(request, available)
        return available[0]

    def _lb_round_robin(self, available: List[BackendInstance]) -> BackendInstance:
        """轮询负载均衡"""
        with self._rr_lock:
            instance = available[self._rr_index % len(available)]
            self._rr_index += 1
        return instance

    def _lb_least_connections(self, available: List[BackendInstance]) -> BackendInstance:
        """最少连接负载均衡"""
        return min(available, key=lambda b: b.active_connections)

    def _lb_weighted(self, available: List[BackendInstance]) -> BackendInstance:
        """加权负载均衡 - 加权随机"""
        total_weight = sum(b.weight for b in available)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for b in available:
            cumulative += b.weight
            if r <= cumulative:
                return b
        return available[-1]

    def _lb_consistent_hash(self, request: Request,
                            available: List[BackendInstance]) -> BackendInstance:
        """一致性哈希负载均衡"""
        key = f"{request.client_ip}:{request.path}"
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return available[hash_val % len(available)]

    def enqueue_request(self, request: Request) -> bool:
        """将请求加入优先级队列"""
        with self._queue_lock:
            if len(self.request_queue) >= self.config.server.max_connections:
                self._stats["rejected_requests"] += 1
                return False
            heapq.heappush(self.request_queue, request)
        return True

    def dequeue_request(self, timeout: float = 1.0) -> Optional[Request]:
        """从队列中取出最高优先级请求"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._queue_lock:
                # 检查并移除超时请求
                while self.request_queue and self.request_queue[0].is_expired:
                    expired = heapq.heappop(self.request_queue)
                    self._stats["timeout_requests"] += 1
                if self.request_queue:
                    return heapq.heappop(self.request_queue)
            time.sleep(0.01)
        return None

    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self._middleware_chain.append(middleware)

    def process_request(self, request: Request) -> Response:
        """处理请求 - 执行中间件链和路由"""
        self._stats["total_requests"] += 1
        request.started_at = time.time()

        # 执行中间件
        for middleware in self._middleware_chain:
            result = middleware(request)
            if isinstance(result, Response):
                return result

        # 路由到处理器
        handler = self._request_handlers.get(request.path)
        if not handler:
            request.completed_at = time.time()
            self._stats["failed_requests"] += 1
            return Response(404, {"error": f"未知端点: {request.path}"})

        try:
            response = handler(request)
            request.completed_at = time.time()
            request.response = response
            if response.status_code < 400:
                self._stats["successful_requests"] += 1
            else:
                self._stats["failed_requests"] += 1
            return response
        except Exception as e:
            request.completed_at = time.time()
            self._stats["failed_requests"] += 1
            return Response(500, {"error": str(e)})

    def start(self, num_workers: Optional[int] = None) -> None:
        """启动模型服务"""
        if self._running:
            return
        self._running = True
        self._shutdown_event.clear()
        num_workers = num_workers or self.config.server.workers
        for i in range(num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"model-server-worker-{i}",
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

    def _worker_loop(self) -> None:
        """工作线程主循环"""
        while not self._shutdown_event.is_set():
            request = self.dequeue_request(timeout=0.5)
            if request is None:
                continue
            backend = self.route_request(request)
            if backend is None:
                response = Response(503, {"error": "无可用后端"})
                request.response = response
                request.completed_at = time.time()
                self._stats["failed_requests"] += 1
                continue
            backend.active_connections += 1
            backend.total_requests += 1
            start = time.time()
            try:
                response = self.process_request(request)
                latency = (time.time() - start) * 1000
                backend.record_latency(latency)
            except Exception:
                backend.total_errors += 1
            finally:
                backend.active_connections -= 1

    def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """优雅关闭"""
        self._running = False
        self._shutdown_event.set()
        deadline = time.time() + timeout
        for t in self._worker_threads:
            remaining = max(deadline - time.time(), 0)
            t.join(timeout=remaining)
        self._worker_threads.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计"""
        return {
            **self._stats,
            "backends": [
                {
                    "id": b.instance_id,
                    "address": b.address,
                    "healthy": b.is_healthy,
                    "connections": b.active_connections,
                    "total_requests": b.total_requests,
                    "avg_latency_ms": round(b.avg_latency_ms, 2),
                    "load": round(b.get_load() * 100, 1),
                }
                for b in self.backends
            ],
            "queue_size": len(self.request_queue),
        }


# ============================================================================
# 第四部分：API网关 (APIGateway)
# ============================================================================

class APIEndpoint:
    """API端点定义"""
    def __init__(self, path: str, method: str = "GET",
                 handler: Optional[Callable] = None,
                 version: str = "v1",
                 auth_required: bool = True,
                 rate_limit: Optional[int] = None,
                 timeout: float = 30.0,
                 description: str = ""):
        self.path = path
        self.method = method.upper()
        self.handler = handler
        self.version = version
        self.auth_required = auth_required
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.description = description
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0

    @property
    def full_path(self) -> str:
        return f"/api/{self.version}{self.path}"

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count


class TokenBucket:
    """令牌桶限流算法"""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # 令牌/秒
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌"""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate,
        )
        self.last_refill = now


class SlidingWindowCounter:
    """滑动窗口计数器"""
    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """判断是否允许请求"""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()
            if len(self._requests) < self.limit:
                self._requests.append(now)
                return True
            return False

    @property
    def current_count(self) -> int:
        with self._lock:
            cutoff = time.time() - self.window_seconds
            return sum(1 for t in self._requests if t >= cutoff)


class FixedWindowCounter:
    """固定窗口计数器"""
    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self._count = 0
        self._window_start = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            if now - self._window_start >= self.window_seconds:
                self._count = 0
                self._window_start = now
            if self._count < self.limit:
                self._count += 1
                return True
            return False


class LeakyBucket:
    """漏桶限流算法"""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # 漏出速率 (请求/秒)
        self.capacity = capacity
        self._queue: deque = deque(maxlen=capacity)
        self._last_leak = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            self._leak()
            if len(self._queue) < self.capacity:
                self._queue.append(time.time())
                return True
            return False

    def _leak(self) -> None:
        now = time.time()
        elapsed = now - self._last_leak
        leak_count = int(elapsed * self.rate)
        for _ in range(min(leak_count, len(self._queue))):
            self._queue.popleft()
        self._last_leak = now


class APIGateway:
    """
    API网关
    - REST API端点管理
    - 请求验证
    - 认证 (API Key, JWT模拟)
    - 限流 (令牌桶、滑动窗口、固定窗口)
    - 请求/响应转换
    - API版本管理
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.endpoints: Dict[str, APIEndpoint] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._jwt_secret = hashlib.sha256(
            f"secret-{time.time()}".encode()
        ).hexdigest()
        self._rate_limiters: Dict[str, Any] = {}
        self._request_transformers: List[Callable] = []
        self._response_transformers: List[Callable] = []
        self._version_aliases: Dict[str, str] = {}
        self._stats = defaultdict(int)
        self._lock = threading.Lock()

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """注册API端点"""
        key = f"{endpoint.method}:{endpoint.full_path}"
        self.endpoints[key] = endpoint
        # 为端点创建限流器
        if endpoint.rate_limit:
            rl_config = self.config.rate_limit
            if rl_config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                self._rate_limiters[key] = TokenBucket(
                    rate=endpoint.rate_limit / 60.0,
                    capacity=endpoint.rate_limit * 2,
                )
            elif rl_config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                self._rate_limiters[key] = SlidingWindowCounter(
                    limit=endpoint.rate_limit,
                )
            elif rl_config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                self._rate_limiters[key] = FixedWindowCounter(
                    limit=endpoint.rate_limit,
                )

    def register_api_key(self, key: str, name: str,
                         scopes: Optional[List[str]] = None,
                         rate_limit: Optional[int] = None) -> None:
        """注册API密钥"""
        self._api_keys[key] = {
            "name": name,
            "scopes": scopes or ["*"],
            "rate_limit": rate_limit,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
            "request_count": 0,
        }

    def authenticate(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        请求认证
        支持 API Key 和 JWT (模拟)
        """
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")

        # API Key 认证
        if api_key:
            if api_key in self._api_keys:
                self._api_keys[api_key]["last_used"] = datetime.now().isoformat()
                self._api_keys[api_key]["request_count"] += 1
                return True, "api_key"
            return False, "无效的API Key"

        # JWT模拟认证
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return self._validate_jwt(token)

        return False, "缺少认证信息"

    def _validate_jwt(self, token: str) -> Tuple[bool, str]:
        """JWT令牌验证 (模拟)"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return False, "无效的JWT格式"
            header_b64, payload_b64, signature_b64 = parts
            # 解码payload (模拟)
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            payload_json = base64.b64decode(payload_b64)
            payload = json.loads(payload_json)
            # 检查过期时间
            exp = payload.get("exp", 0)
            if exp and time.time() > exp:
                return False, "JWT已过期"
            return True, "jwt"
        except Exception as e:
            return False, f"JWT验证失败: {str(e)}"

    def generate_jwt(self, user_id: str, scopes: Optional[List[str]] = None,
                     expires_hours: float = 24.0) -> str:
        """生成JWT令牌 (模拟)"""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "scopes": scopes or ["*"],
            "iat": int(time.time()),
            "exp": int(time.time() + expires_hours * 3600),
        }
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip("=")
        signature = hashlib.sha256(
            f"{header_b64}.{payload_b64}.{self._jwt_secret}".encode()
        ).hexdigest()[:43]
        return f"{header_b64}.{payload_b64}.{signature}"

    def check_rate_limit(self, endpoint_key: str,
                         client_ip: str) -> Tuple[bool, Optional[str]]:
        """检查限流"""
        if not self.config.rate_limit.enabled:
            return True, None
        # 全局限流
        global_key = f"global:{endpoint_key}"
        if global_key not in self._rate_limiters:
            self._rate_limiters[global_key] = TokenBucket(
                rate=self.config.rate_limit.requests_per_second,
                capacity=self.config.rate_limit.burst_size,
            )
        if not self._rate_limiters[global_key].consume():
            return False, "超过全局速率限制"
        # 每IP限流
        if self.config.rate_limit.per_ip_enabled:
            ip_key = f"ip:{client_ip}:{endpoint_key}"
            if ip_key not in self._rate_limiters:
                self._rate_limiters[ip_key] = TokenBucket(
                    rate=self.config.rate_limit.per_ip_requests_per_second,
                    capacity=self.config.rate_limit.per_ip_requests_per_second * 5,
                )
            if not self._rate_limiters[ip_key].consume():
                return False, "超过IP速率限制"
        # 端点限流
        if endpoint_key in self._rate_limiters:
            limiter = self._rate_limiters[endpoint_key]
            if isinstance(limiter, (SlidingWindowCounter, FixedWindowCounter)):
                if not limiter.allow():
                    return False, "超过端点速率限制"
        return True, None

    def validate_request(self, request: Request) -> Tuple[bool, Optional[str]]:
        """请求验证"""
        if not request.method:
            return False, "缺少请求方法"
        if not request.path:
            return False, "缺少请求路径"
        if request.method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            return False, f"不支持的请求方法: {request.method}"
        # 路径安全检查
        if ".." in request.path or "//" in request.path:
            return False, "非法路径"
        return True, None

    def add_request_transformer(self, transformer: Callable) -> None:
        """添加请求转换器"""
        self._request_transformers.append(transformer)

    def add_response_transformer(self, transformer: Callable) -> None:
        """添加响应转换器"""
        self._response_transformers.append(transformer)

    def transform_request(self, request: Request) -> Request:
        """应用请求转换"""
        for transformer in self._request_transformers:
            request = transformer(request)
        return request

    def transform_response(self, response: Response) -> Response:
        """应用响应转换"""
        for transformer in self._response_transformers:
            response = transformer(response)
        return response

    def add_version_alias(self, alias: str, version: str) -> None:
        """添加版本别名"""
        self._version_aliases[alias] = version

    def resolve_version(self, version: str) -> str:
        """解析版本号"""
        return self._version_aliases.get(version, version)

    def handle_request(self, request: Request) -> Response:
        """处理网关请求"""
        self._stats["total_requests"] += 1
        start_time = time.time()

        # 1. 请求验证
        valid, error = self.validate_request(request)
        if not valid:
            self._stats["validation_errors"] += 1
            return Response(400, {"error": error})

        # 2. 请求转换
        request = self.transform_request(request)

        # 3. 查找端点
        endpoint_key = f"{request.method}:{request.path}"
        endpoint = self.endpoints.get(endpoint_key)
        if not endpoint:
            self._stats["not_found"] += 1
            return Response(404, {"error": "端点不存在"})

        # 4. 认证
        if endpoint.auth_required:
            auth_ok, auth_msg = self.authenticate(request)
            if not auth_ok:
                self._stats["auth_failures"] += 1
                return Response(401, {"error": auth_msg})

        # 5. 限流
        rate_ok, rate_msg = self.check_rate_limit(endpoint_key, request.client_ip)
        if not rate_ok:
            self._stats["rate_limited"] += 1
            resp = Response(429, {"error": rate_msg})
            resp.headers["Retry-After"] = "60"
            return resp

        # 6. 执行处理器
        endpoint.request_count += 1
        try:
            if endpoint.handler:
                response = endpoint.handler(request)
            else:
                response = Response(200, {"message": "OK"})
            # 7. 响应转换
            response = self.transform_response(response)
            latency = (time.time() - start_time) * 1000
            endpoint.total_latency_ms += latency
            self._stats["successful_requests"] += 1
            return response
        except Exception as e:
            endpoint.error_count += 1
            self._stats["server_errors"] += 1
            return Response(500, {"error": str(e)})

    def get_stats(self) -> Dict[str, Any]:
        """获取网关统计"""
        return {
            "total_requests": self._stats["total_requests"],
            "successful": self._stats["successful_requests"],
            "auth_failures": self._stats["auth_failures"],
            "rate_limited": self._stats["rate_limited"],
            "not_found": self._stats["not_found"],
            "server_errors": self._stats["server_errors"],
            "endpoints": {
                key: {
                    "requests": ep.request_count,
                    "errors": ep.error_count,
                    "avg_latency_ms": round(ep.avg_latency_ms, 2),
                }
                for key, ep in self.endpoints.items()
            },
        }


# ============================================================================
# 第五部分：容器编排 (ContainerManager)
# ============================================================================

@dataclass
class Container:
    """容器实例"""
    container_id: str
    image: str
    name: str
    state: ContainerState = ContainerState.PENDING
    host: str = "localhost"
    port: int = 8080
    cpu_limit: float = 1.0
    memory_limit_mb: int = 2048
    gpu_limit: int = 0
    env_vars: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    health_status: str = "unknown"
    restart_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 0.0, "memory_mb": 0.0, "gpu_percent": 0.0,
    })


@dataclass
class KubernetesPod:
    """Kubernetes Pod模拟"""
    pod_id: str
    name: str
    namespace: str = "default"
    containers: List[Container] = field(default_factory=list)
    service_name: str = ""
    node_name: str = ""
    phase: str = "Pending"
    replicas: int = 1
    ready_replicas: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    conditions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ServiceRecord:
    """服务发现记录"""
    service_name: str
    service_type: str = "ClusterIP"
    addresses: List[str] = field(default_factory=list)
    port: int = 8080
    metadata: Dict[str, str] = field(default_factory=dict)
    healthy: bool = True


class ContainerManager:
    """
    容器编排管理器
    - 容器生命周期管理
    - Docker容器模拟
    - Kubernetes Pod模拟
    - 服务发现
    - 健康检查
    - 自动扩缩 (HPA模拟)
    - 滚动更新
    - 蓝绿部署
    - 金丝雀部署
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.containers: Dict[str, Container] = {}
        self.pods: Dict[str, KubernetesPod] = {}
        self.services: Dict[str, ServiceRecord] = {}
        self._container_counter = 0
        self._lock = threading.Lock()
        self._health_check_running = False
        self._hpa_running = False
        self._deployment_history: List[Dict[str, Any]] = []
        self._active_deployment: Optional[Dict[str, Any]] = None

    # --- 容器生命周期 ---

    def create_container(self, image: str, name: str,
                         env_vars: Optional[Dict[str, str]] = None,
                         labels: Optional[Dict[str, str]] = None,
                         cpu: float = 1.0, memory_mb: int = 2048,
                         gpu: int = 0) -> Container:
        """创建容器"""
        with self._lock:
            self._container_counter += 1
            container_id = f"ctr-{self._container_counter:06d}"
            container = Container(
                container_id=container_id,
                image=image,
                name=name,
                cpu_limit=cpu,
                memory_limit_mb=memory_mb,
                gpu_limit=gpu,
                env_vars=env_vars or {},
                labels=labels or {},
                port=8000 + self._container_counter,
            )
            self.containers[container_id] = container
        # 模拟启动
        self._start_container(container_id)
        return container

    def _start_container(self, container_id: str) -> None:
        """启动容器"""
        container = self.containers.get(container_id)
        if not container:
            return
        container.state = ContainerState.RUNNING
        container.started_at = time.time()
        container.health_status = "starting"

    def stop_container(self, container_id: str, timeout: float = 30.0) -> bool:
        """停止容器"""
        container = self.containers.get(container_id)
        if not container:
            return False
        container.state = ContainerState.TERMINATING
        # 模拟异步停止
        def _do_stop():
            time.sleep(min(timeout, 0.1))  # 模拟中快速完成
            container.state = ContainerState.TERMINATED
        threading.Thread(target=_do_stop, daemon=True).start()
        return True

    def restart_container(self, container_id: str) -> bool:
        """重启容器"""
        container = self.containers.get(container_id)
        if not container:
            return False
        container.restart_count += 1
        container.state = ContainerState.TERMINATING
        def _do_restart():
            time.sleep(0.1)
            container.state = ContainerState.RUNNING
            container.started_at = time.time()
            container.health_status = "starting"
        threading.Thread(target=_do_restart, daemon=True).start()
        return True

    def remove_container(self, container_id: str) -> bool:
        """删除容器"""
        return self.containers.pop(container_id, None) is not None

    def list_containers(self, label_filter: Optional[Dict[str, str]] = None,
                        state_filter: Optional[ContainerState] = None
                        ) -> List[Container]:
        """列出容器"""
        result = list(self.containers.values())
        if label_filter:
            result = [
                c for c in result
                if all(c.labels.get(k) == v for k, v in label_filter.items())
            ]
        if state_filter:
            result = [c for c in result if c.state == state_filter]
        return result

    # --- Kubernetes Pod管理 ---

    def create_pod(self, name: str, image: str, replicas: int = 1,
                   namespace: str = "default",
                   labels: Optional[Dict[str, str]] = None,
                   env_vars: Optional[Dict[str, str]] = None
                   ) -> KubernetesPod:
        """创建Kubernetes Pod"""
        pod_id = f"pod-{hashlib.md5(name.encode()).hexdigest()[:8]}"
        pod = KubernetesPod(
            pod_id=pod_id,
            name=name,
            namespace=namespace,
            replicas=replicas,
            labels=labels or {},
        )
        # 为每个副本创建容器
        for i in range(replicas):
            container = self.create_container(
                image=image,
                name=f"{name}-{i}",
                labels=labels or {},
                env_vars=env_vars or {},
            )
            container.health_status = "starting"
            pod.containers.append(container)
        pod.ready_replicas = replicas
        pod.phase = "Running"
        self.pods[pod_id] = pod
        return pod

    def delete_pod(self, pod_id: str) -> bool:
        """删除Pod"""
        pod = self.pods.get(pod_id)
        if not pod:
            return False
        for container in pod.containers:
            self.stop_container(container.container_id)
        del self.pods[pod_id]
        return True

    # --- 服务发现 ---

    def register_service(self, name: str, service_type: str = "ClusterIP",
                         port: int = 8080,
                         metadata: Optional[Dict[str, str]] = None) -> ServiceRecord:
        """注册服务"""
        record = ServiceRecord(
            service_name=name,
            service_type=service_type,
            port=port,
            metadata=metadata or {},
        )
        # 自动发现后端地址
        for container in self.containers.values():
            if (container.state == ContainerState.RUNNING and
                    container.labels.get("service") == name):
                record.addresses.append(
                    f"{container.host}:{container.port}"
                )
        self.services[name] = record
        return record

    def discover_service(self, service_name: str) -> Optional[ServiceRecord]:
        """服务发现"""
        record = self.services.get(service_name)
        if record:
            # 刷新地址
            record.addresses = [
                f"{c.host}:{c.port}"
                for c in self.containers.values()
                if (c.state == ContainerState.RUNNING and
                    c.labels.get("service") == service_name)
            ]
            record.healthy = len(record.addresses) > 0
        return record

    def list_services(self) -> List[ServiceRecord]:
        return list(self.services.values())

    # --- 健康检查 ---

    def start_health_checks(self) -> None:
        """启动健康检查"""
        if self._health_check_running:
            return
        self._health_check_running = True
        t = threading.Thread(target=self._health_check_loop, daemon=True)
        t.start()

    def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._health_check_running:
            for container in list(self.containers.values()):
                if container.state != ContainerState.RUNNING:
                    continue
                # 模拟健康检查
                healthy = random.random() > 0.05  # 95%概率健康
                container.health_status = "healthy" if healthy else "unhealthy"
                container.resource_usage = {
                    "cpu_percent": random.uniform(10, 90),
                    "memory_mb": random.uniform(100, container.memory_limit_mb * 0.9),
                    "gpu_percent": random.uniform(0, 100) if container.gpu_limit > 0 else 0,
                }
            time.sleep(self.config.health_check.interval)

    def stop_health_checks(self) -> None:
        self._health_check_running = False

    # --- 自动扩缩 (HPA) ---

    def start_hpa(self) -> None:
        """启动水平Pod自动扩缩"""
        if self._hpa_running:
            return
        self._hpa_running = True
        t = threading.Thread(target=self._hpa_loop, daemon=True)
        t.start()

    def _hpa_loop(self) -> None:
        """HPA主循环"""
        last_scale_time = 0
        while self._hpa_running:
            now = time.time()
            if now - last_scale_time < 60:  # 最少60秒间隔
                time.sleep(10)
                continue
            for pod in list(self.pods.values()):
                desired = self._calculate_desired_replicas(pod)
                if desired != pod.replicas:
                    self._scale_pod(pod, desired)
                    last_scale_time = now
            time.sleep(30)

    def _calculate_desired_replicas(self, pod: KubernetesPod) -> int:
        """计算期望副本数"""
        if not pod.containers:
            return pod.replicas
        avg_cpu = sum(
            c.resource_usage["cpu_percent"] for c in pod.containers
        ) / len(pod.containers)
        target = self.config.scaling.target_cpu_utilization
        if avg_cpu > target:
            ratio = avg_cpu / target
            desired = int(pod.replicas * ratio) + 1
        elif avg_cpu < target * 0.5:
            ratio = avg_cpu / target
            desired = max(1, int(pod.replicas * ratio))
        else:
            desired = pod.replicas
        return max(
            self.config.scaling.min_replicas,
            min(desired, self.config.scaling.max_replicas),
        )

    def _scale_pod(self, pod: KubernetesPod, desired: int) -> None:
        """扩缩Pod"""
        current = pod.replicas
        if desired > current:
            # 扩容
            for i in range(current, desired):
                container = self.create_container(
                    image=pod.containers[0].image if pod.containers else "default",
                    name=f"{pod.name}-{i}",
                    labels=pod.labels,
                )
                pod.containers.append(container)
            pod.replicas = desired
            pod.ready_replicas = desired
            self._log_deployment("scale_up", pod.name, {
                "from": current, "to": desired,
            })
        elif desired < current:
            # 缩容 - 移除多余的
            for i in range(current - 1, desired - 1, -1):
                if i < len(pod.containers):
                    self.stop_container(pod.containers[i].container_id)
                    pod.containers.pop(i)
            pod.replicas = desired
            pod.ready_replicas = desired
            self._log_deployment("scale_down", pod.name, {
                "from": current, "to": desired,
            })

    # --- 部署策略 ---

    def rolling_update(self, pod_name: str, new_image: str,
                       max_unavailable: int = 1,
                       max_surge: int = 1) -> bool:
        """滚动更新"""
        pod = self._find_pod_by_name(pod_name)
        if not pod:
            return False
        self._active_deployment = {
            "strategy": "rolling_update",
            "target": pod_name,
            "new_image": new_image,
            "started_at": datetime.now().isoformat(),
        }
        old_containers = list(pod.containers)
        updated = 0
        for old_container in old_containers:
            # 创建新容器
            new_container = self.create_container(
                image=new_image,
                name=f"{pod.name}-v2-{updated}",
                labels=pod.labels,
                env_vars=old_container.env_vars,
            )
            # 等待新容器就绪 (模拟)
            time.sleep(0.05)
            # 停止旧容器
            self.stop_container(old_container.container_id)
            pod.containers = [
                c for c in pod.containers
                if c.container_id != old_container.container_id
            ]
            pod.containers.append(new_container)
            updated += 1
        self._log_deployment("rolling_update", pod_name, {
            "new_image": new_image, "updated": updated,
        })
        self._active_deployment = None
        return True

    def blue_green_deploy(self, pod_name: str, new_image: str,
                          switch_after: float = 5.0) -> bool:
        """蓝绿部署"""
        pod = self._find_pod_by_name(pod_name)
        if not pod:
            return False
        # 保存当前(蓝)容器
        blue_containers = list(pod.containers)
        # 创建绿环境
        green_containers = []
        for i, blue_c in enumerate(blue_containers):
            green_c = self.create_container(
                image=new_image,
                name=f"{pod.name}-green-{i}",
                labels={**pod.labels, "environment": "green"},
                env_vars=blue_c.env_vars,
            )
            green_containers.append(green_c)
        # 等待绿环境就绪
        time.sleep(min(switch_after, 0.1))
        # 切换流量
        pod.containers = green_containers
        # 停止蓝环境
        for blue_c in blue_containers:
            self.stop_container(blue_c.container_id)
        self._log_deployment("blue_green", pod_name, {
            "new_image": new_image,
        })
        return True

    def canary_deploy(self, pod_name: str, new_image: str,
                      canary_weight: float = 0.1,
                      step_duration: float = 30.0,
                      steps: int = 10) -> bool:
        """金丝雀部署 - 逐步增加新版本流量"""
        pod = self._find_pod_by_name(pod_name)
        if not pod:
            return False
        old_replicas = len(pod.containers)
        canary_replicas = max(1, int(old_replicas * canary_weight))
        # 创建金丝雀实例
        canary_containers = []
        for i in range(canary_replicas):
            c = self.create_container(
                image=new_image,
                name=f"{pod.name}-canary-{i}",
                labels={**pod.labels, "canary": "true"},
            )
            canary_containers.append(c)
        # 模拟逐步增加流量
        for step in range(steps):
            weight = canary_weight + (1.0 - canary_weight) * (step / steps)
            time.sleep(min(step_duration, 0.01))
            self._log_deployment("canary_step", pod_name, {
                "step": step + 1, "weight": round(weight, 2),
            })
        # 全量切换
        for old_c in list(pod.containers):
            self.stop_container(old_c.container_id)
        for i in range(old_replicas - canary_replicas):
            c = self.create_container(
                image=new_image,
                name=f"{pod.name}-canary-full-{i}",
                labels=pod.labels,
            )
            canary_containers.append(c)
        pod.containers = canary_containers
        self._log_deployment("canary_complete", pod_name, {
            "new_image": new_image,
        })
        return True

    def _find_pod_by_name(self, name: str) -> Optional[KubernetesPod]:
        for pod in self.pods.values():
            if pod.name == name:
                return pod
        return None

    def _log_deployment(self, action: str, target: str,
                        details: Dict[str, Any]) -> None:
        self._deployment_history.append({
            "action": action,
            "target": target,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        running = sum(
            1 for c in self.containers.values()
            if c.state == ContainerState.RUNNING
        )
        return {
            "containers": {
                "total": len(self.containers),
                "running": running,
                "pending": sum(1 for c in self.containers.values() if c.state == ContainerState.PENDING),
                "failed": sum(1 for c in self.containers.values() if c.state == ContainerState.FAILED),
            },
            "pods": {
                "total": len(self.pods),
                "running": sum(1 for p in self.pods.values() if p.phase == "Running"),
            },
            "services": len(self.services),
            "active_deployment": self._active_deployment is not None,
        }


# ============================================================================
# 第六部分：模型注册表 (ModelRegistry)
# ============================================================================

@dataclass
class ModelVersion:
    """模型版本"""
    version: str
    model_path: str
    format: ModelFormat = ModelFormat.PYTORCH
    stage: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_production: bool = False
    checksum: str = ""
    size_bytes: int = 0
    description: str = ""


@dataclass
class ABTestConfig:
    """A/B测试配置"""
    test_id: str
    model_a_version: str
    model_b_version: str
    traffic_split: float = 0.5  # A版本流量比例
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    total_requests: int = 0
    a_requests: int = 0
    b_requests: int = 0


class SemanticVersion:
    """语义化版本管理"""

    @staticmethod
    def parse(version_str: str) -> Tuple[int, int, int]:
        """解析版本号"""
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
        if not match:
            raise ValueError(f"无效版本号: {version_str}")
        return tuple(int(x) for x in match.groups())

    @staticmethod
    def compare(v1: str, v2: str) -> int:
        """比较版本号: -1, 0, 1"""
        p1 = SemanticVersion.parse(v1)
        p2 = SemanticVersion.parse(v2)
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
        return 0

    @staticmethod
    def next_major(version: str) -> str:
        major, minor, patch = SemanticVersion.parse(version)
        return f"{major + 1}.0.0"

    @staticmethod
    def next_minor(version: str) -> str:
        major, minor, patch = SemanticVersion.parse(version)
        return f"{major}.{minor + 1}.0"

    @staticmethod
    def next_patch(version: str) -> str:
        major, minor, patch = SemanticVersion.parse(version)
        return f"{major}.{minor}.{patch + 1}"


class ModelRegistry:
    """
    模型注册表
    - 模型注册
    - 版本管理 (语义化版本)
    - 模型分级 (dev/staging/production)
    - A/B测试
    - 影子部署
    - 模型回滚
    """

    def __init__(self):
        self.models: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._production_versions: Dict[str, str] = {}
        self._ab_tests: Dict[str, ABTestConfig] = {}
        self._shadow_deployments: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._ab_counter = 0

    def register_model(self, model_name: str, version: str,
                       model_path: str,
                       format: ModelFormat = ModelFormat.PYTORCH,
                       metrics: Optional[Dict[str, float]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       description: str = "") -> ModelVersion:
        """注册模型"""
        with self._lock:
            mv = ModelVersion(
                version=version,
                model_path=model_path,
                format=format,
                metrics=metrics or {},
                metadata=metadata or {},
                description=description,
                checksum=hashlib.sha256(
                    f"{model_name}:{version}:{model_path}".encode()
                ).hexdigest()[:16],
                size_bytes=random.randint(100_000_000, 10_000_000_000),
            )
            self.models[model_name][version] = mv
        return mv

    def get_model(self, model_name: str,
                  version: Optional[str] = None) -> Optional[ModelVersion]:
        """获取模型"""
        versions = self.models.get(model_name, {})
        if version:
            return versions.get(version)
        # 返回最新版本
        if not versions:
            return None
        latest = max(
            versions.keys(),
            key=lambda v: SemanticVersion.parse(v),
        )
        return versions[latest]

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """列出模型所有版本"""
        versions = self.models.get(model_name, {})
        return sorted(
            versions.values(),
            key=lambda v: SemanticVersion.parse(v.version),
            reverse=True,
        )

    def promote(self, model_name: str, version: str,
                target_stage: DeploymentEnvironment) -> bool:
        """提升模型到指定阶段"""
        mv = self.models.get(model_name, {}).get(version)
        if not mv:
            return False
        mv.stage = target_stage
        if target_stage == DeploymentEnvironment.PRODUCTION:
            # 取消之前的production标记
            for v in self.models[model_name].values():
                v.is_production = False
            mv.is_production = True
            self._production_versions[model_name] = version
        return True

    def rollback(self, model_name: str,
                 target_version: Optional[str] = None) -> bool:
        """回滚模型版本"""
        versions = self.models.get(model_name, {})
        if not versions:
            return False
        if target_version and target_version not in versions:
            return False
        if not target_version:
            # 回滚到上一个production版本
            prod_versions = [
                v for v in versions.values()
                if v.stage == DeploymentEnvironment.PRODUCTION
            ]
            if len(prod_versions) < 2:
                return False
            prod_versions.sort(
                key=lambda v: SemanticVersion.parse(v.version),
                reverse=True,
            )
            target_version = prod_versions[1].version
        return self.promote(
            model_name, target_version,
            DeploymentEnvironment.PRODUCTION,
        )

    def delete_version(self, model_name: str, version: str) -> bool:
        """删除模型版本"""
        if model_name not in self.models:
            return False
        mv = self.models[model_name].get(version)
        if not mv:
            return False
        if mv.is_production:
            return False  # 不能删除生产版本
        del self.models[model_name][version]
        return True

    # --- A/B测试 ---

    def create_ab_test(self, model_name: str,
                       version_a: str, version_b: str,
                       traffic_split: float = 0.5) -> Optional[ABTestConfig]:
        """创建A/B测试"""
        versions = self.models.get(model_name, {})
        if version_a not in versions or version_b not in versions:
            return None
        with self._lock:
            self._ab_counter += 1
            test = ABTestConfig(
                test_id=f"ab-{self._ab_counter:04d}",
                model_a_version=version_a,
                model_b_version=version_b,
                traffic_split=traffic_split,
            )
            self._ab_tests[test.test_id] = test
        return test

    def route_ab_request(self, test_id: str) -> Optional[str]:
        """A/B测试请求路由"""
        test = self._ab_tests.get(test_id)
        if not test or not test.is_active:
            return None
        test.total_requests += 1
        if random.random() < test.traffic_split:
            test.a_requests += 1
            return test.model_a_version
        else:
            test.b_requests += 1
            return test.model_b_version

    def stop_ab_test(self, test_id: str,
                     winner: Optional[str] = None) -> Dict[str, Any]:
        """停止A/B测试"""
        test = self._ab_tests.get(test_id)
        if not test:
            return {"error": "测试不存在"}
        test.is_active = False
        test.end_time = datetime.now().isoformat()
        result = {
            "test_id": test_id,
            "model_a": test.model_a_version,
            "model_b": test.model_b_version,
            "a_requests": test.a_requests,
            "b_requests": test.b_requests,
            "winner": winner,
        }
        return result

    # --- 影子部署 ---

    def enable_shadow(self, model_name: str, shadow_version: str) -> bool:
        """启用影子部署"""
        versions = self.models.get(model_name, {})
        if shadow_version not in versions:
            return False
        self._shadow_deployments[model_name] = shadow_version
        return True

    def disable_shadow(self, model_name: str) -> bool:
        """禁用影子部署"""
        return self._shadow_deployments.pop(model_name, None) is not None

    def get_shadow_version(self, model_name: str) -> Optional[str]:
        """获取影子版本"""
        return self._shadow_deployments.get(model_name)

    def get_registry_summary(self) -> Dict[str, Any]:
        """获取注册表摘要"""
        return {
            "models": {
                name: {
                    "versions": len(versions),
                    "production": next(
                        (v.version for v in versions.values() if v.is_production),
                        None,
                    ),
                }
                for name, versions in self.models.items()
            },
            "active_ab_tests": sum(
                1 for t in self._ab_tests.values() if t.is_active
            ),
            "shadow_deployments": len(self._shadow_deployments),
        }


# ============================================================================
# 第七部分：监控仪表盘 (MonitoringDashboard)
# ============================================================================

@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    metric: str
    condition: str  # gt, lt, eq, ne
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = 300
    enabled: bool = True
    last_triggered: Optional[float] = None


@dataclass
class SLAConfig:
    """SLA配置"""
    target_availability: float = 99.9  # %
    target_latency_p99_ms: float = 500.0
    target_error_rate: float = 0.1  # %
    target_throughput_rps: float = 1000.0
    measurement_window: float = 3600.0  # 秒


class LatencyTracker:
    """延迟跟踪器 - 计算P50/P90/P99"""
    def __init__(self, max_samples: int = 10000):
        self._samples: deque = deque(maxlen=max_samples)

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """计算百分位数"""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100.0)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p90(self) -> float:
        return self.percentile(90)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def avg(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)


class ThroughputTracker:
    """吞吐量跟踪器"""
    def __init__(self, window_seconds: float = 60.0):
        self._window = window_seconds
        self._timestamps: deque = deque()

    def record(self) -> None:
        self._timestamps.append(time.time())

    @property
    def current_rps(self) -> float:
        """当前每秒请求数"""
        cutoff = time.time() - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if not self._timestamps:
            return 0.0
        span = time.time() - self._timestamps[0]
        return len(self._timestamps) / max(span, 0.001)

    @property
    def total_count(self) -> int:
        return len(self._timestamps)


class ResourceMonitor:
    """资源使用监控"""
    def __init__(self):
        self._cpu_history: deque = deque(maxlen=3600)
        self._memory_history: deque = deque(maxlen=3600)
        self._gpu_history: deque = deque(maxlen=3600)

    def record(self, cpu: float, memory: float, gpu: float = 0.0) -> None:
        now = time.time()
        self._cpu_history.append((now, cpu))
        self._memory_history.append((now, memory))
        self._gpu_history.append((now, gpu))

    def get_current(self) -> Dict[str, float]:
        """获取当前资源使用"""
        cpu = self._cpu_history[-1][1] if self._cpu_history else 0.0
        mem = self._memory_history[-1][1] if self._memory_history else 0.0
        gpu = self._gpu_history[-1][1] if self._gpu_history else 0.0
        return {"cpu_percent": cpu, "memory_percent": mem, "gpu_percent": gpu}

    def get_average(self, window: float = 300.0) -> Dict[str, float]:
        """获取时间窗口内的平均值"""
        cutoff = time.time() - window
        cpu_vals = [v for t, v in self._cpu_history if t >= cutoff]
        mem_vals = [v for t, v in self._memory_history if t >= cutoff]
        gpu_vals = [v for t, v in self._gpu_history if t >= cutoff]
        return {
            "cpu_percent": sum(cpu_vals) / max(len(cpu_vals), 1),
            "memory_percent": sum(mem_vals) / max(len(mem_vals), 1),
            "gpu_percent": sum(gpu_vals) / max(len(gpu_vals), 1),
        }


class MonitoringDashboard:
    """
    部署监控仪表盘
    - 请求延迟跟踪 (P50/P90/P99)
    - 吞吐量测量
    - 错误率跟踪
    - 资源利用率 (CPU, 内存, GPU)
    - 告警规则
    - SLA管理
    """

    def __init__(self, sla_config: Optional[SLAConfig] = None):
        self.sla = sla_config or SLAConfig()
        self.latency = LatencyTracker()
        self.throughput = ThroughputTracker()
        self.resources = ResourceMonitor()
        self._error_timestamps: deque = deque(maxlen=100000)
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: List[Dict[str, Any]] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._alert_counter = 0
        self._lock = threading.Lock()
        self._monitoring_running = False

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        """记录请求指标"""
        self.latency.record(latency_ms)
        self.throughput.record()
        if is_error:
            self._error_timestamps.append(time.time())

    def record_resources(self, cpu: float, memory: float,
                         gpu: float = 0.0) -> None:
        """记录资源使用"""
        self.resources.record(cpu, memory, gpu)

    @property
    def error_rate(self) -> float:
        """当前错误率 (%)"""
        total = self.throughput.total_count
        if total == 0:
            return 0.0
        cutoff = time.time() - self.sla.measurement_window
        errors = sum(1 for t in self._error_timestamps if t >= cutoff)
        return (errors / total) * 100.0

    def add_alert_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self._alert_rules[rule.rule_id] = rule

    def evaluate_alerts(self) -> List[Dict[str, Any]]:
        """评估告警规则"""
        triggered = []
        metrics = {
            "latency_p50": self.latency.p50,
            "latency_p90": self.latency.p90,
            "latency_p99": self.latency.p99,
            "throughput_rps": self.throughput.current_rps,
            "error_rate": self.error_rate,
            "cpu_percent": self.resources.get_current()["cpu_percent"],
            "memory_percent": self.resources.get_current()["memory_percent"],
            "gpu_percent": self.resources.get_current()["gpu_percent"],
        }
        for rule in self._alert_rules.values():
            if not rule.enabled:
                continue
            value = metrics.get(rule.metric)
            if value is None:
                continue
            # 检查冷却时间
            if (rule.last_triggered and
                    time.time() - rule.last_triggered < rule.cooldown_seconds):
                continue
            # 评估条件
            triggered_flag = False
            if rule.condition == "gt" and value > rule.threshold:
                triggered_flag = True
            elif rule.condition == "lt" and value < rule.threshold:
                triggered_flag = True
            elif rule.condition == "eq" and abs(value - rule.threshold) < 0.01:
                triggered_flag = True
            elif rule.condition == "ne" and abs(value - rule.threshold) >= 0.01:
                triggered_flag = True
            if triggered_flag:
                rule.last_triggered = time.time()
                alert = {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "severity": rule.severity.value,
                    "metric": rule.metric,
                    "value": round(value, 2),
                    "threshold": rule.threshold,
                    "timestamp": datetime.now().isoformat(),
                }
                triggered.append(alert)
                self._alert_history.append(alert)
                if rule.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
                    self._active_alerts.append(alert)
        return triggered

    def check_sla(self) -> Dict[str, Any]:
        """检查SLA合规性"""
        availability = max(0, 100.0 - self.error_rate)
        return {
            "availability": {
                "current": round(availability, 3),
                "target": self.sla.target_availability,
                "compliant": availability >= self.sla.target_availability,
            },
            "latency_p99": {
                "current_ms": round(self.latency.p99, 2),
                "target_ms": self.sla.target_latency_p99_ms,
                "compliant": self.latency.p99 <= self.sla.target_latency_p99_ms,
            },
            "error_rate": {
                "current_percent": round(self.error_rate, 3),
                "target_percent": self.sla.target_error_rate,
                "compliant": self.error_rate <= self.sla.target_error_rate,
            },
            "throughput": {
                "current_rps": round(self.throughput.current_rps, 2),
                "target_rps": self.sla.target_throughput_rps,
                "compliant": (
                    self.throughput.current_rps >= self.sla.target_throughput_rps
                ),
            },
            "overall_compliant": all([
                availability >= self.sla.target_availability,
                self.latency.p99 <= self.sla.target_latency_p99_ms,
                self.error_rate <= self.sla.target_error_rate,
            ]),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        return {
            "latency": {
                "p50_ms": round(self.latency.p50, 2),
                "p90_ms": round(self.latency.p90, 2),
                "p99_ms": round(self.latency.p99, 2),
                "avg_ms": round(self.latency.avg, 2),
                "samples": self.latency.count,
            },
            "throughput": {
                "current_rps": round(self.throughput.current_rps, 2),
                "total_requests": self.throughput.total_count,
            },
            "error_rate": round(self.error_rate, 3),
            "resources": self.resources.get_current(),
            "active_alerts": len(self._active_alerts),
            "sla": self.check_sla(),
        }

    def start_monitoring(self, interval: float = 10.0) -> None:
        """启动后台监控"""
        if self._monitoring_running:
            return
        self._monitoring_running = True

        def _loop():
            while self._monitoring_running:
                alerts = self.evaluate_alerts()
                if alerts:
                    pass  # 告警已记录
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()

    def stop_monitoring(self) -> None:
        self._monitoring_running = False


# ============================================================================
# 第八部分：CI/CD流水线 (CICDPipeline)
# ============================================================================

@dataclass
class PipelineStep:
    """流水线步骤"""
    name: str
    stage: PipelineStage
    command: str
    timeout: float = 300.0
    on_failure: str = "abort"  # abort, continue, retry
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    output: str = ""
    artifacts: List[str] = field(default_factory=list)


@dataclass
class PipelineRun:
    """流水线运行记录"""
    run_id: str
    name: str
    trigger: str = "manual"
    branch: str = "main"
    commit: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration: float = 0.0
    environment: str = "production"


class CICDPipeline:
    """
    CI/CD流水线
    - 构建流水线
    - 测试流水线
    - 部署流水线
    - 自动回滚
    - 流水线状态跟踪
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.pipelines: Dict[str, List[PipelineRun]] = defaultdict(list)
        self._run_counter = 0
        self._lock = threading.Lock()
        self._webhooks: List[Callable] = []

    def create_pipeline(self, name: str,
                        steps: Optional[List[Dict[str, Any]]] = None
                        ) -> PipelineRun:
        """创建流水线"""
        with self._lock:
            self._run_counter += 1
            run_id = f"pipeline-{self._run_counter:06d}"
        run = PipelineRun(
            run_id=run_id,
            name=name,
        )
        if steps:
            for step_def in steps:
                step = PipelineStep(
                    name=step_def.get("name", "unnamed"),
                    stage=PipelineStage(step_def.get("stage", "build")),
                    command=step_def.get("command", ""),
                    timeout=step_def.get("timeout", 300.0),
                    on_failure=step_def.get("on_failure", "abort"),
                    max_retries=step_def.get("max_retries", 3),
                )
                run.steps.append(step)
        self.pipelines[name].append(run)
        return run

    def create_standard_ml_pipeline(self, name: str,
                                     branch: str = "main",
                                     commit: str = "") -> PipelineRun:
        """创建标准ML部署流水线"""
        steps = [
            {"name": "代码检出", "stage": "source",
             "command": "git checkout", "timeout": 60},
            {"name": "依赖安装", "stage": "build",
             "command": "pip install -r requirements.txt", "timeout": 300},
            {"name": "模型构建", "stage": "build",
             "command": "python build_model.py", "timeout": 600},
            {"name": "单元测试", "stage": "test",
             "command": "pytest tests/unit/ -v", "timeout": 300},
            {"name": "集成测试", "stage": "test",
             "command": "pytest tests/integration/ -v", "timeout": 600},
            {"name": "安全扫描", "stage": "security_scan",
             "command": "bandit -r . && safety check", "timeout": 120},
            {"name": "模型验证", "stage": "test",
             "command": "python validate_model.py", "timeout": 300},
            {"name": "Staging部署", "stage": "staging_deploy",
             "command": "kubectl apply -f k8s/staging/", "timeout": 300},
            {"name": "冒烟测试", "stage": "integration_test",
             "command": "python smoke_test.py --env staging", "timeout": 120},
            {"name": "生产部署", "stage": "production_deploy",
             "command": "kubectl apply -f k8s/production/", "timeout": 600},
            {"name": "部署验证", "stage": "verification",
             "command": "python verify_deployment.py", "timeout": 300},
        ]
        run = self.create_pipeline(name, steps)
        run.branch = branch
        run.commit = commit
        return run

    def run_pipeline(self, run_id: str) -> Dict[str, Any]:
        """执行流水线"""
        run = self._find_run(run_id)
        if not run:
            return {"error": "流水线不存在"}
        run.status = "running"
        run.started_at = time.time()
        for step in run.steps:
            step_result = self._execute_step(step)
            if not step_result["success"]:
                if step.on_failure == "abort":
                    run.status = "failed"
                    run.completed_at = time.time()
                    run.duration = run.completed_at - run.started_at
                    return {"status": "failed", "failed_at": step.name}
                elif step.on_failure == "retry" and step.retry_count < step.max_retries:
                    step.retry_count += 1
                    retry_result = self._execute_step(step)
                    if not retry_result["success"]:
                        run.status = "failed"
                        run.completed_at = time.time()
                        run.duration = run.completed_at - run.started_at
                        return {"status": "failed", "failed_at": step.name}
        run.status = "complete"
        run.completed_at = time.time()
        run.duration = run.completed_at - run.started_at
        # 触发webhook
        for webhook in self._webhooks:
            try:
                webhook({"run_id": run_id, "status": "complete"})
            except Exception:
                pass
        return {"status": "complete", "duration": round(run.duration, 2)}

    def _execute_step(self, step: PipelineStep) -> Dict[str, bool]:
        """执行流水线步骤 (模拟)"""
        step.status = "running"
        step.started_at = time.time()
        # 模拟执行
        success = random.random() > 0.05  # 95%成功率
        step.output = f"[模拟] 执行: {step.command}\n状态: {'成功' if success else '失败'}"
        step.completed_at = time.time()
        step.status = "success" if success else "failed"
        return {"success": success}

    def rollback(self, run_id: str,
                 target_version: Optional[str] = None) -> Dict[str, Any]:
        """自动回滚"""
        run = self._find_run(run_id)
        if not run:
            return {"error": "流水线不存在"}
        rollback_steps = [
            {"name": "停止当前版本", "stage": "production_deploy",
             "command": f"kubectl rollout undo deployment/{run.name}"},
            {"name": "验证回滚", "stage": "verification",
             "command": "python verify_deployment.py --rollback"},
        ]
        if target_version:
            rollback_steps[0]["command"] = (
                f"kubectl set image deployment/{run.name} "
                f"model={run.name}:{target_version}"
            )
        rollback_run = self.create_pipeline(
            f"{run.name}-rollback", rollback_steps,
        )
        rollback_run.trigger = "auto_rollback"
        result = self.run_pipeline(rollback_run.run_id)
        return {
            "original_run": run_id,
            "rollback_run": rollback_run.run_id,
            "status": result.get("status"),
        }

    def add_webhook(self, callback: Callable) -> None:
        """添加webhook回调"""
        self._webhooks.append(callback)

    def get_pipeline_history(self, name: str,
                             limit: int = 10) -> List[Dict[str, Any]]:
        """获取流水线历史"""
        runs = self.pipelines.get(name, [])
        return [
            {
                "run_id": r.run_id,
                "status": r.status,
                "trigger": r.trigger,
                "branch": r.branch,
                "duration": round(r.duration, 2),
                "steps_completed": sum(
                    1 for s in r.steps if s.status == "success"
                ),
                "steps_total": len(r.steps),
            }
            for r in runs[-limit:]
        ]

    def _find_run(self, run_id: str) -> Optional[PipelineRun]:
        for runs in self.pipelines.values():
            for run in runs:
                if run.run_id == run_id:
                    return run
        return None


# ============================================================================
# 第九部分：边缘部署 (EdgeDeployment)
# ============================================================================

@dataclass
class EdgeDevice:
    """边缘设备"""
    device_id: str
    name: str
    device_type: str  # jetson, raspberry_pi, mobile, custom
    ip_address: str
    architecture: str = "arm64"
    memory_mb: int = 4096
    storage_mb: int = 16000
    compute_power: float = 1.0  # 相对算力
    model_version: str = ""
    firmware_version: str = "1.0.0"
    is_online: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    temperature: float = 45.0
    battery_level: float = 100.0


@dataclass
class OTAUpdate:
    """OTA更新包"""
    update_id: str
    model_name: str
    version: str
    model_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    checksum: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"
    target_devices: List[str] = field(default_factory=list)
    completed_devices: List[str] = field(default_factory=list)
    failed_devices: List[str] = field(default_factory=list)


class EdgeDeployment:
    """
    边缘部署管理
    - 模型压缩
    - ONNX导出模拟
    - 模型量化
    - 边缘设备管理
    - OTA更新
    """

    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self._updates: Dict[str, OTAUpdate] = {}
        self._update_counter = 0
        self._device_heartbeat_interval = 30.0
        self._heartbeat_running = False

    def register_device(self, device: EdgeDevice) -> None:
        """注册边缘设备"""
        self.devices[device.device_id] = device

    def remove_device(self, device_id: str) -> bool:
        return self.devices.pop(device_id, None) is not None

    def list_devices(self, online_only: bool = False,
                     device_type: Optional[str] = None) -> List[EdgeDevice]:
        """列出设备"""
        result = list(self.devices.values())
        if online_only:
            result = [d for d in result if d.is_online]
        if device_type:
            result = [d for d in result if d.device_type == device_type]
        return result

    def check_device_heartbeat(self, device_id: str) -> bool:
        """检查设备心跳"""
        device = self.devices.get(device_id)
        if not device:
            return False
        elapsed = time.time() - device.last_heartbeat
        device.is_online = elapsed < self._device_heartbeat_interval * 3
        return device.is_online

    def start_heartbeat_monitor(self) -> None:
        """启动心跳监控"""
        if self._heartbeat_running:
            return
        self._heartbeat_running = True

        def _monitor():
            while self._heartbeat_running:
                for device in list(self.devices.values()):
                    self.check_device_heartbeat(device.device_id)
                time.sleep(self._device_heartbeat_interval)

        threading.Thread(target=_monitor, daemon=True).start()

    # --- 模型优化 ---

    def compress_model(self, model_path: str,
                       target_size_mb: float = 50.0) -> Dict[str, Any]:
        """
        模型压缩 (模拟)
        实现知识蒸馏 + 剪枝 + 量化组合策略
        """
        original_size = random.uniform(100, 500)
        compression_ratio = target_size_mb / original_size
        # 模拟压缩过程
        pruning_ratio = min(0.8, compression_ratio * 0.4)
        quantization_bits = 8 if compression_ratio > 0.3 else 4
        estimated_size = original_size * (1 - pruning_ratio * 0.5) * (quantization_bits / 32)
        return {
            "original_size_mb": round(original_size, 2),
            "target_size_mb": target_size_mb,
            "estimated_size_mb": round(estimated_size, 2),
            "compression_ratio": round(original_size / max(estimated_size, 0.1), 2),
            "pruning_ratio": round(pruning_ratio, 2),
            "quantization_bits": quantization_bits,
            "method": "pruning+quantization",
        }

    def export_onnx(self, model_path: str,
                    opset_version: int = 13,
                    optimize: bool = True) -> Dict[str, Any]:
        """ONNX导出 (模拟)"""
        original_size = random.uniform(100, 500)
        onnx_size = original_size * (0.7 if optimize else 0.9)
        return {
            "format": "onnx",
            "opset_version": opset_version,
            "optimized": optimize,
            "original_size_mb": round(original_size, 2),
            "onnx_size_mb": round(onnx_size, 2),
            "optimizations": (
                ["constant_folding", "dead_code_elimination",
                 "operator_fusion", "shape_inference"]
                if optimize else []
            ),
            "supported_eps": ["onnxruntime", "openvino", "tensorrt"],
        }

    def quantize_model(self, model_path: str,
                       bits: int = 8,
                       calibration_samples: int = 1000) -> Dict[str, Any]:
        """
        模型量化 (模拟)
        支持INT8/INT4量化，使用校准数据集
        """
        original_size = random.uniform(100, 500)
        if bits == 8:
            quantized_size = original_size * 0.25
            accuracy_drop = random.uniform(0.1, 0.5)
        elif bits == 4:
            quantized_size = original_size * 0.15
            accuracy_drop = random.uniform(0.5, 2.0)
        else:
            quantized_size = original_size * 0.5
            accuracy_drop = random.uniform(0.01, 0.1)
        return {
            "quantization_bits": bits,
            "method": f"INT{bits}" if bits >= 4 else "Mixed",
            "original_size_mb": round(original_size, 2),
            "quantized_size_mb": round(quantized_size, 2),
            "compression_ratio": round(original_size / quantized_size, 2),
            "estimated_accuracy_drop": round(accuracy_drop, 2),
            "calibration_samples": calibration_samples,
            "per_channel": bits <= 8,
        }

    # --- OTA更新 ---

    def create_ota_update(self, model_name: str, version: str,
                          target_devices: List[str]) -> OTAUpdate:
        """创建OTA更新"""
        with threading.Lock():
            self._update_counter += 1
            update_id = f"ota-{self._update_counter:04d}"
        model_size = random.uniform(10, 100)
        update = OTAUpdate(
            update_id=update_id,
            model_name=model_name,
            version=version,
            model_size_mb=round(model_size, 2),
            compressed_size_mb=round(model_size * 0.6, 2),
            checksum=hashlib.sha256(
                f"{model_name}:{version}".encode()
            ).hexdigest(),
            target_devices=target_devices,
            status="pending",
        )
        self._updates[update_id] = update
        return update

    def execute_ota_update(self, update_id: str) -> Dict[str, Any]:
        """执行OTA更新"""
        update = self._updates.get(update_id)
        if not update:
            return {"error": "更新不存在"}
        update.status = "in_progress"
        results = {"success": [], "failed": []}
        for device_id in update.target_devices:
            device = self.devices.get(device_id)
            if not device or not device.is_online:
                results["failed"].append(device_id)
                update.failed_devices.append(device_id)
                continue
            # 模拟更新
            success = random.random() > 0.1
            if success:
                device.model_version = update.version
                results["success"].append(device_id)
                update.completed_devices.append(device_id)
            else:
                results["failed"].append(device_id)
                update.failed_devices.append(device_id)
        update.status = (
            "complete"
            if len(results["failed"]) == 0
            else "partial"
        )
        return {
            "update_id": update_id,
            "status": update.status,
            "success_count": len(results["success"]),
            "failed_count": len(results["failed"]),
        }

    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态总览"""
        online = sum(1 for d in self.devices.values() if d.is_online)
        return {
            "total_devices": len(self.devices),
            "online_devices": online,
            "offline_devices": len(self.devices) - online,
            "pending_updates": sum(
                1 for u in self._updates.values() if u.status == "pending"
            ),
            "avg_temperature": round(
                sum(d.temperature for d in self.devices.values()) /
                max(len(self.devices), 1), 1
            ),
        }


# ============================================================================
# 第十部分：移动端部署 (MobileDeployment)
# ============================================================================

@dataclass
class MobileAppConfig:
    """移动应用配置"""
    app_name: str
    package_name: str  # com.example.app
    version_name: str = "1.0.0"
    version_code: int = 1
    min_sdk: int = 24
    target_sdk: int = 34
    model_name: str = ""
    model_version: str = ""
    model_format: ModelFormat = ModelFormat.TFLITE


@dataclass
class AppStoreSubmission:
    """应用商店提交记录"""
    submission_id: str
    platform: str  # google_play, app_store
    app_name: str
    version: str
    status: str = "pending"
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    review_notes: str = ""
    release_notes: str = ""


class MobileDeployment:
    """
    移动端部署
    - 模型转换 (TFLite模拟)
    - 模型优化
    - APK/IPA打包模拟
    - 应用商店部署模拟
    """

    def __init__(self):
        self._apps: Dict[str, MobileAppConfig] = {}
        self._submissions: Dict[str, AppStoreSubmission] = {}
        self._submission_counter = 0

    def convert_to_tflite(self, model_path: str,
                          optimize: bool = True,
                          supported_ops: Optional[List[str]] = None
                          ) -> Dict[str, Any]:
        """
        转换为TFLite格式 (模拟)
        包含图优化和算子选择
        """
        original_size = random.uniform(50, 200)
        tflite_size = original_size * (0.3 if optimize else 0.6)
        ops = supported_ops or [
            "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
            "MAX_POOL_2D", "AVERAGE_POOL_2D", "SOFTMAX",
            "RESHAPE", "CONCATENATION", "ADD", "MUL",
        ]
        optimizations = []
        if optimize:
            optimizations = [
                "DEFAULT", "OPTIMIZE_FOR_SIZE", "OPTIMIZE_FOR_LATENCY",
            ]
        return {
            "format": "tflite",
            "original_size_mb": round(original_size, 2),
            "tflite_size_mb": round(tflite_size, 2),
            "compression_ratio": round(original_size / tflite_size, 2),
            "optimizations": optimizations,
            "supported_ops": ops,
            "inference_latency_ms": round(random.uniform(5, 50), 2),
            "memory_footprint_mb": round(tflite_size * 1.5, 2),
        }

    def optimize_for_mobile(self, model_path: str,
                            target_latency_ms: float = 30.0,
                            target_size_mb: float = 20.0) -> Dict[str, Any]:
        """
        移动端模型优化 (模拟)
        多阶段优化：量化 + 剪枝 + 算子融合
        """
        original_size = random.uniform(50, 200)
        original_latency = random.uniform(100, 500)
        # 阶段1: INT8量化
        quantized_size = original_size * 0.25
        quantized_latency = original_latency * 0.4
        # 阶段2: 结构化剪枝
        pruned_size = quantized_size * 0.7
        pruned_latency = quantized_latency * 0.9
        # 阶段3: 算子融合
        fused_latency = pruned_latency * 0.85
        return {
            "stages": [
                {
                    "name": "INT8量化",
                    "size_mb": round(quantized_size, 2),
                    "latency_ms": round(quantized_latency, 2),
                    "size_reduction": f"{round((1 - quantized_size/original_size)*100, 1)}%",
                },
                {
                    "name": "结构化剪枝",
                    "size_mb": round(pruned_size, 2),
                    "latency_ms": round(pruned_latency, 2),
                    "size_reduction": f"{round((1 - pruned_size/original_size)*100, 1)}%",
                },
                {
                    "name": "算子融合",
                    "size_mb": round(pruned_size, 2),
                    "latency_ms": round(fused_latency, 2),
                    "latency_reduction": f"{round((1 - fused_latency/original_latency)*100, 1)}%",
                },
            ],
            "final_size_mb": round(pruned_size, 2),
            "final_latency_ms": round(fused_latency, 2),
            "meets_latency_target": fused_latency <= target_latency_ms,
            "meets_size_target": pruned_size <= target_size_mb,
        }

    def build_apk(self, app_config: MobileAppConfig,
                  include_model: bool = True) -> Dict[str, Any]:
        """构建APK (模拟)"""
        base_size = random.uniform(5, 15)
        model_size = random.uniform(5, 30) if include_model else 0
        apk_size = base_size + model_size
        return {
            "platform": "android",
            "format": "apk",
            "app_name": app_config.app_name,
            "package": app_config.package_name,
            "version": app_config.version_name,
            "version_code": app_config.version_code,
            "min_sdk": app_config.min_sdk,
            "target_sdk": app_config.target_sdk,
            "size_mb": round(apk_size, 2),
            "model_included": include_model,
            "model_version": app_config.model_version if include_model else None,
            "build_time": datetime.now().isoformat(),
            "checksum": hashlib.sha256(
                f"{app_config.package_name}:{app_config.version_name}".encode()
            ).hexdigest()[:16],
        }

    def build_ipa(self, app_config: MobileAppConfig,
                  include_model: bool = True) -> Dict[str, Any]:
        """构建IPA (模拟)"""
        base_size = random.uniform(8, 20)
        model_size = random.uniform(5, 30) if include_model else 0
        ipa_size = base_size + model_size
        return {
            "platform": "ios",
            "format": "ipa",
            "app_name": app_config.app_name,
            "bundle_id": app_config.package_name,
            "version": app_config.version_name,
            "build_number": app_config.version_code,
            "min_ios": "14.0",
            "size_mb": round(ipa_size, 2),
            "model_included": include_model,
            "model_version": app_config.model_version if include_model else None,
            "build_time": datetime.now().isoformat(),
            "checksum": hashlib.sha256(
                f"{app_config.package_name}:{app_config.version_name}".encode()
            ).hexdigest()[:16],
        }

    def submit_to_store(self, app_config: MobileAppConfig,
                        platform: str = "google_play",
                        release_notes: str = "") -> AppStoreSubmission:
        """提交到应用商店 (模拟)"""
        with threading.Lock():
            self._submission_counter += 1
            submission_id = f"sub-{self._submission_counter:04d}"
        submission = AppStoreSubmission(
            submission_id=submission_id,
            platform=platform,
            app_name=app_config.app_name,
            version=app_config.version_name,
            release_notes=release_notes,
        )
        # 模拟审核流程
        review_time = random.uniform(1, 5)  # 秒 (实际是小时/天)
        def _review():
            time.sleep(min(review_time, 0.1))
            submission.status = random.choice([
                "approved", "approved", "approved",  # 75%通过率
                "rejected",  # 25%拒绝
            ])
        threading.Thread(target=_review, daemon=True).start()
        self._submissions[submission_id] = submission
        return submission

    def get_submission_status(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """获取提交状态"""
        sub = self._submissions.get(submission_id)
        if not sub:
            return None
        return {
            "submission_id": sub.submission_id,
            "platform": sub.platform,
            "app_name": sub.app_name,
            "version": sub.version,
            "status": sub.status,
            "submitted_at": sub.submitted_at,
        }


# ============================================================================
# 第十一部分：部署管理器 (DeploymentManager) - 主编排器
# ============================================================================

class DeploymentManager:
    """
    部署管理器 - 统一编排所有部署组件

    整合以下模块：
    - DeploymentConfig: 部署配置
    - ModelServer: 模型服务
    - APIGateway: API网关
    - ContainerManager: 容器编排
    - ModelRegistry: 模型注册
    - MonitoringDashboard: 监控仪表盘
    - CICDPipeline: CI/CD流水线
    - EdgeDeployment: 边缘部署
    - MobileDeployment: 移动端部署
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        self.server = ModelServer(self.config)
        self.gateway = APIGateway(self.config)
        self.containers = ContainerManager(self.config)
        self.registry = ModelRegistry()
        self.monitoring = MonitoringDashboard()
        self.cicd = CICDPipeline(self.config)
        self.edge = EdgeDeployment()
        self.mobile = MobileDeployment()
        self._initialized = False
        self._running = False

    def initialize(self) -> Dict[str, Any]:
        """初始化部署环境"""
        errors = self.config.validate()
        if errors:
            return {"success": False, "errors": errors}
        # 注册默认API端点
        self.gateway.register_endpoint(APIEndpoint(
            path="/health", method="GET",
            handler=lambda r: Response(200, {"status": "ok"}),
            auth_required=False,
        ))
        self.gateway.register_endpoint(APIEndpoint(
            path="/predict", method="POST",
            handler=self._predict_handler,
            rate_limit=100,
        ))
        self.gateway.register_endpoint(APIEndpoint(
            path="/models", method="GET",
            handler=self._list_models_handler,
            auth_required=True,
        ))
        self.gateway.register_endpoint(APIEndpoint(
            path="/metrics", method="GET",
            handler=self._metrics_handler,
            auth_required=False,
        ))
        # 添加默认后端
        for i in range(self.config.scaling.min_replicas):
            self.server.add_backend(
                instance_id=f"backend-{i}",
                host="127.0.0.1",
                port=8000 + i,
                weight=1,
            )
        # 注册默认告警规则
        self._setup_default_alerts()
        # 启动健康检查
        self.containers.start_health_checks()
        self._initialized = True
        return {"success": True, "message": "部署环境初始化完成"}

    def _setup_default_alerts(self) -> None:
        """设置默认告警规则"""
        self.monitoring.add_alert_rule(AlertRule(
            rule_id="alert-latency-p99",
            name="P99延迟过高",
            metric="latency_p99",
            condition="gt",
            threshold=500.0,
            severity=AlertSeverity.WARNING,
        ))
        self.monitoring.add_alert_rule(AlertRule(
            rule_id="alert-error-rate",
            name="错误率过高",
            metric="error_rate",
            condition="gt",
            threshold=5.0,
            severity=AlertSeverity.CRITICAL,
        ))
        self.monitoring.add_alert_rule(AlertRule(
            rule_id="alert-cpu",
            name="CPU使用率过高",
            metric="cpu_percent",
            condition="gt",
            threshold=90.0,
            severity=AlertSeverity.WARNING,
        ))
        self.monitoring.add_alert_rule(AlertRule(
            rule_id="alert-memory",
            name="内存使用率过高",
            metric="memory_percent",
            condition="gt",
            threshold=85.0,
            severity=AlertSeverity.CRITICAL,
        ))

    def _predict_handler(self, request: Request) -> Response:
        """预测处理器"""
        start = time.time()
        backend = self.server.route_request(request)
        if not backend:
            return Response(503, {"error": "无可用后端"})
        body = request.body or {}
        model_name = body.get("model", self.config.model.model_name)
        model = self.registry.get_model(model_name)
        if not model:
            return Response(404, {"error": f"模型不存在: {model_name}"})
        latency = random.uniform(10, 200)
        self.monitoring.record_request(latency, is_error=False)
        return Response(200, {
            "model": model_name,
            "version": model.version,
            "output": f"[推理结果]",
            "latency_ms": round(latency, 2),
        })

    def _list_models_handler(self, request: Request) -> Response:
        """模型列表处理器"""
        models = self.registry.get_registry_summary()
        return Response(200, models)

    def _metrics_handler(self, request: Request) -> Response:
        """指标处理器"""
        dashboard = self.monitoring.get_dashboard()
        return Response(200, dashboard)

    def deploy(self, model_name: str, model_path: str,
               version: str = "1.0.0",
               strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
               environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
               ) -> Dict[str, Any]:
        """
        执行完整部署流程
        """
        if not self._initialized:
            return {"success": False, "error": "未初始化，请先调用 initialize()"}

        # 1. 注册模型
        model = self.registry.register_model(
            model_name=model_name,
            version=version,
            model_path=model_path,
        )

        # 2. 创建CI/CD流水线
        pipeline = self.cicd.create_standard_ml_pipeline(
            name=f"deploy-{model_name}-{version}",
        )

        # 3. 执行流水线
        pipeline_result = self.cicd.run_pipeline(pipeline.run_id)

        if pipeline_result.get("status") != "complete":
            return {
                "success": False,
                "error": "CI/CD流水线失败",
                "pipeline": pipeline_result,
            }

        # 4. 提升模型到目标环境
        self.registry.promote(model_name, version, environment)

        # 5. 容器部署
        pod = self.containers.create_pod(
            name=f"{model_name}-{version}",
            image=f"{model_name}:{version}",
            replicas=self.config.scaling.min_replicas,
            labels={"app": model_name, "version": version},
        )

        # 6. 注册服务
        self.containers.register_service(
            name=model_name,
            port=self.config.server.port,
        )

        # 7. 添加后端
        for container in pod.containers:
            self.server.add_backend(
                instance_id=container.container_id,
                host=container.host,
                port=container.port,
            )

        return {
            "success": True,
            "model": model_name,
            "version": version,
            "environment": environment.value,
            "strategy": strategy.value,
            "pipeline": pipeline_result,
            "pod": pod.pod_id,
            "containers": len(pod.containers),
        }

    def start(self) -> Dict[str, Any]:
        """启动所有服务"""
        if not self._initialized:
            init_result = self.initialize()
            if not init_result.get("success"):
                return init_result
        self.server.start()
        self.containers.start_hpa()
        self.monitoring.start_monitoring()
        self._running = True
        return {
            "success": True,
            "message": "所有服务已启动",
            "components": [
                "ModelServer",
                "APIGateway",
                "ContainerManager (HPA)",
                "MonitoringDashboard",
            ],
        }

    def stop(self) -> Dict[str, Any]:
        """停止所有服务"""
        self.server.graceful_shutdown()
        self.containers.stop_health_checks()
        self.containers.stop_hpa()
        self.monitoring.stop_monitoring()
        self._running = False
        return {"success": True, "message": "所有服务已停止"}

    def get_status(self) -> Dict[str, Any]:
        """获取整体部署状态"""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "config": self.config.to_dict(),
            "server": self.server.get_stats(),
            "gateway": self.gateway.get_stats(),
            "cluster": self.containers.get_cluster_status(),
            "registry": self.registry.get_registry_summary(),
            "monitoring": self.monitoring.get_dashboard(),
            "sla": self.monitoring.check_sla(),
        }

    def quick_demo(self) -> Dict[str, Any]:
        """
        快速演示 - 展示所有核心功能
        """
        results = {}

        # 1. 初始化
        results["init"] = self.initialize()

        # 2. 注册模型
        self.registry.register_model("gpt-model", "1.0.0", "/models/gpt-v1")
        self.registry.register_model("gpt-model", "1.1.0", "/models/gpt-v1.1")
        self.registry.register_model("gpt-model", "2.0.0", "/models/gpt-v2")
        self.registry.promote("gpt-model", "1.0.0", DeploymentEnvironment.PRODUCTION)
        results["models"] = self.registry.get_registry_summary()

        # 3. 创建API Key和JWT
        self.gateway.register_api_key("demo-key-123", "demo-app")
        jwt = self.gateway.generate_jwt("user-001", scopes=["predict", "read"])
        results["jwt_sample"] = jwt[:50] + "..."

        # 4. 模拟请求
        request = Request(
            method="POST",
            path="/predict",
            headers={"X-API-Key": "demo-key-123"},
            body={"model": "gpt-model", "prompt": "Hello world"},
        )
        response = self.gateway.handle_request(request)
        results["request_response"] = {
            "status": response.status_code,
            "body": response.body,
        }

        # 5. 容器部署
        pod = self.containers.create_pod(
            name="gpt-service",
            image="gpt-model:1.0.0",
            replicas=3,
            labels={"app": "gpt-model"},
        )
        results["containers"] = self.containers.get_cluster_status()

        # 6. 监控数据
        for _ in range(100):
            self.monitoring.record_request(
                random.uniform(5, 300),
                is_error=random.random() < 0.02,
            )
            self.monitoring.record_resources(
                random.uniform(20, 80),
                random.uniform(30, 70),
            )
        results["monitoring"] = self.monitoring.get_dashboard()
        results["sla"] = self.monitoring.check_sla()

        # 7. 边缘部署
        device = EdgeDevice(
            device_id="edge-001",
            name="Jetson Xavier",
            device_type="jetson",
            ip_address="192.168.1.100",
        )
        self.edge.register_device(device)
        results["edge"] = {
            "compression": self.edge.compress_model("/models/gpt", target_size_mb=30),
            "quantization": self.edge.quantize_model("/models/gpt", bits=8),
            "devices": self.edge.get_device_status(),
        }

        # 8. 移动端部署
        app_config = MobileAppConfig(
            app_name="AI Assistant",
            package_name="com.example.aiassistant",
            version_name="1.0.0",
        )
        results["mobile"] = {
            "tflite": self.mobile.convert_to_tflite("/models/gpt"),
            "optimization": self.mobile.optimize_for_mobile("/models/gpt"),
            "apk": self.mobile.build_apk(app_config),
        }

        return results
