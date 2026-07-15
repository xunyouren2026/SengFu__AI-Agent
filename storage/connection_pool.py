"""
连接池管理器

通用的连接池实现，支持线程安全的连接获取/释放、
空闲超时回收、连接健康检查和统计信息。
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Deque, Dict, List, Optional


class PoolError(Exception):
    """连接池异常"""
    pass


class PoolExhaustedError(PoolError):
    """连接池耗尽异常"""
    pass


class ConnectionExpiredError(PoolError):
    """连接已过期异常"""
    pass


@dataclass
class PoolConfig:
    """
    连接池配置

    Attributes:
        min_size: 最小连接数（预热数量）
        max_size: 最大连接数
        max_idle_time: 空闲连接最大存活时间（秒）
        max_lifetime: 连接最大生命周期（秒）
        acquire_timeout: 获取连接超时时间（秒）
        health_check_interval: 健康检查间隔（秒）
        health_check_fn: 健康检查函数，返回True表示连接健康
    """
    min_size: int = 2
    max_size: int = 10
    max_idle_time: float = 300.0
    max_lifetime: float = 3600.0
    acquire_timeout: float = 30.0
    health_check_interval: float = 60.0
    health_check_fn: Optional[Callable[[Any], bool]] = None


@dataclass
class _PooledConnection:
    """池化连接的内部包装"""
    connection: Any
    created_at: float = 0.0
    last_used_at: float = 0.0
    _in_use: bool = False
    _id: int = 0

    def __post_init__(self):
        now = time.time()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_used_at == 0.0:
            self.last_used_at = now

    @property
    def idle_time(self) -> float:
        """空闲时间"""
        return time.time() - self.last_used_at

    @property
    def age(self) -> float:
        """连接年龄"""
        return time.time() - self.created_at

    def touch(self) -> None:
        """更新最后使用时间"""
        self.last_used_at = time.time()


class PoolStats:
    """
    连接池统计信息

    Attributes:
        pool_size: 当前池中总连接数
        active_count: 活跃（使用中）连接数
        idle_count: 空闲连接数
        waiting_count: 等待获取连接的请求数
        total_created: 累计创建连接数
        total_destroyed: 累计销毁连接数
        total_acquired: 累计获取连接次数
        total_released: 累计释放连接次数
        total_timeouts: 累计超时次数
        total_errors: 累计错误次数
    """

    def __init__(self):
        self.pool_size = 0
        self.active_count = 0
        self.idle_count = 0
        self.waiting_count = 0
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_errors = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pool_size": self.pool_size,
            "active_count": self.active_count,
            "idle_count": self.idle_count,
            "waiting_count": self.waiting_count,
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "total_timeouts": self.total_timeouts,
            "total_errors": self.total_errors,
        }

    def __repr__(self) -> str:
        return (
            f"PoolStats(size={self.pool_size}, active={self.active_count}, "
            f"idle={self.idle_count}, waiting={self.waiting_count})"
        )


class ConnectionPool:
    """
    连接池管理器

    通用的线程安全连接池实现，支持：
    - 最小/最大连接数控制
    - 空闲连接超时回收
    - 连接最大生命周期
    - 健康检查
    - 获取连接超时
    - 连接预热
    - 上下文管理器支持

    Args:
        connection_factory: 连接创建工厂函数
        close_fn: 连接关闭函数
        config: 连接池配置
    """

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        close_fn: Optional[Callable[[Any], None]] = None,
        config: Optional[PoolConfig] = None,
    ):
        self._factory = connection_factory
        self._close_fn = close_fn or (lambda conn: None)
        self._config = config or PoolConfig()

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # 连接存储
        self._all_connections: Dict[int, _PooledConnection] = {}
        self._idle_connections: Deque[_PooledConnection] = deque()
        self._next_id = 0

        # 统计
        self._stats = PoolStats()

        # 后台清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

        # 健康检查线程
        self._health_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动连接池（预热连接并启动后台线程）"""
        with self._lock:
            if self._running:
                return
            self._running = True

        # 预热连接
        self._prewarm()

        # 启动清理线程
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="pool-cleanup",
        )
        self._cleanup_thread.start()

        # 启动健康检查线程
        if self._config.health_check_fn:
            self._health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="pool-health-check",
            )
            self._health_thread.start()

    def stop(self) -> None:
        """停止连接池（关闭所有连接）"""
        with self._lock:
            self._running = False
            self._condition.notify_all()

        # 关闭所有连接
        self._close_all()

    def acquire(self, timeout: Optional[float] = None) -> Any:
        """
        获取连接

        Args:
            timeout: 超时时间（秒），None使用配置中的默认值

        Returns:
            连接对象

        Raises:
            PoolExhaustedError: 连接池耗尽且超时
        """
        timeout = timeout if timeout is not None else self._config.acquire_timeout
        deadline = time.time() + timeout

        with self._condition:
            while True:
                # 尝试从空闲队列获取
                while self._idle_connections:
                    pooled = self._idle_connections.popleft()

                    # 检查连接是否过期
                    if pooled.age > self._config.max_lifetime:
                        self._destroy_connection(pooled)
                        continue

                    # 检查空闲超时
                    if pooled.idle_time > self._config.max_idle_time:
                        self._destroy_connection(pooled)
                        continue

                    # 健康检查
                    if self._config.health_check_fn:
                        try:
                            if not self._config.health_check_fn(pooled.connection):
                                self._destroy_connection(pooled)
                                continue
                        except Exception:
                            self._destroy_connection(pooled)
                            continue

                    # 使用连接
                    pooled._in_use = True
                    pooled.touch()
                    self._stats.total_acquired += 1
                    self._update_stats()
                    return pooled.connection

                # 尝试创建新连接
                if len(self._all_connections) < self._config.max_size:
                    pooled = self._create_connection()
                    pooled._in_use = True
                    self._stats.total_acquired += 1
                    self._update_stats()
                    return pooled.connection

                # 等待连接释放
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._stats.total_timeouts += 1
                    raise PoolExhaustedError(
                        f"连接池已耗尽，无法在 {timeout}s 内获取连接 "
                        f"(max_size={self._config.max_size})"
                    )

                self._stats.waiting_count += 1
                self._condition.wait(timeout=remaining)
                self._stats.waiting_count = max(0, self._stats.waiting_count - 1)

    def release(self, connection: Any) -> None:
        """
        释放连接

        Args:
            connection: 要释放的连接对象
        """
        with self._lock:
            pooled = self._find_pooled(connection)
            if pooled is None or not pooled._in_use:
                return

            pooled._in_use = False
            pooled.touch()
            self._stats.total_released += 1

            # 检查连接是否已过期
            if pooled.age > self._config.max_lifetime:
                self._destroy_connection(pooled)
            else:
                self._idle_connections.append(pooled)

            self._update_stats()
            self._condition.notify()

    def get_stats(self) -> PoolStats:
        """获取连接池统计信息"""
        with self._lock:
            self._update_stats()
            stats = PoolStats()
            stats.pool_size = self._stats.pool_size
            stats.active_count = self._stats.active_count
            stats.idle_count = self._stats.idle_count
            stats.waiting_count = self._stats.waiting_count
            stats.total_created = self._stats.total_created
            stats.total_destroyed = self._stats.total_destroyed
            stats.total_acquired = self._stats.total_acquired
            stats.total_released = self._stats.total_released
            stats.total_timeouts = self._stats.total_timeouts
            stats.total_errors = self._stats.total_errors
            return stats

    def resize(self, new_max_size: int) -> None:
        """
        调整连接池大小

        Args:
            new_max_size: 新的最大连接数
        """
        with self._lock:
            old_max = self._config.max_size
            self._config.max_size = max(1, new_max_size)

            # 如果缩小了池大小，关闭多余的空闲连接
            while (
                len(self._idle_connections) > 0
                and len(self._all_connections) > self._config.max_size
            ):
                pooled = self._idle_connections.popleft()
                self._destroy_connection(pooled)

    def __enter__(self) -> "ConnectionPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<ConnectionPool size={stats.pool_size} "
            f"active={stats.active_count} idle={stats.idle_count}>"
        )

    # ============================================================
    # 内部方法
    # ============================================================

    def _create_connection(self) -> _PooledConnection:
        """创建新连接"""
        try:
            conn = self._factory()
            self._next_id += 1
            pooled = _PooledConnection(
                connection=conn,
                _id=self._next_id,
            )
            self._all_connections[pooled._id] = pooled
            self._stats.total_created += 1
            return pooled
        except Exception as e:
            self._stats.total_errors += 1
            raise PoolError(f"创建连接失败: {e}") from e

    def _destroy_connection(self, pooled: _PooledConnection) -> None:
        """销毁连接"""
        try:
            self._close_fn(pooled.connection)
        except Exception:
            self._stats.total_errors += 1
        finally:
            self._all_connections.pop(pooled._id, None)
            self._stats.total_destroyed += 1

    def _find_pooled(self, connection: Any) -> Optional[_PooledConnection]:
        """根据连接对象查找池化连接"""
        for pooled in self._all_connections.values():
            if pooled.connection is connection:
                return pooled
        return None

    def _update_stats(self) -> None:
        """更新统计信息"""
        self._stats.pool_size = len(self._all_connections)
        self._stats.active_count = sum(
            1 for p in self._all_connections.values() if p._in_use
        )
        self._stats.idle_count = len(self._idle_connections)

    def _prewarm(self) -> None:
        """预热连接"""
        count = min(self._config.min_size, self._config.max_size)
        for _ in range(count):
            pooled = self._create_connection()
            self._idle_connections.append(pooled)
        self._update_stats()

    def _close_all(self) -> None:
        """关闭所有连接"""
        with self._lock:
            # 关闭空闲连接
            while self._idle_connections:
                pooled = self._idle_connections.popleft()
                self._destroy_connection(pooled)

            # 关闭活跃连接
            for pooled in list(self._all_connections.values()):
                self._destroy_connection(pooled)

    def _cleanup_loop(self) -> None:
        """后台清理循环"""
        while self._running:
            time.sleep(min(30.0, self._config.max_idle_time / 2))
            if not self._running:
                break
            self._cleanup_idle()

    def _cleanup_idle(self) -> None:
        """清理过期空闲连接"""
        with self._lock:
            to_remove = []
            while self._idle_connections:
                pooled = self._idle_connections[0]
                if pooled.idle_time > self._config.max_idle_time:
                    to_remove.append(self._idle_connections.popleft())
                elif pooled.age > self._config.max_lifetime:
                    to_remove.append(self._idle_connections.popleft())
                else:
                    break

            for pooled in to_remove:
                self._destroy_connection(pooled)

            # 如果连接数低于最小值，补充连接
            while (
                len(self._all_connections) < self._config.min_size
                and len(self._all_connections) < self._config.max_size
            ):
                pooled = self._create_connection()
                self._idle_connections.append(pooled)

            self._update_stats()

    def _health_check_loop(self) -> None:
        """后台健康检查循环"""
        interval = self._config.health_check_interval
        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            self._perform_health_checks()

    def _perform_health_checks(self) -> None:
        """执行健康检查"""
        with self._lock:
            to_remove = []
            remaining = deque()
            while self._idle_connections:
                pooled = self._idle_connections.popleft()
                try:
                    if not self._config.health_check_fn(pooled.connection):
                        to_remove.append(pooled)
                    else:
                        remaining.append(pooled)
                except Exception:
                    to_remove.append(pooled)

            self._idle_connections = remaining
            for pooled in to_remove:
                self._destroy_connection(pooled)
            self._update_stats()
