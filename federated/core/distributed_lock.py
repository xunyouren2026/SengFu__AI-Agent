import asyncio
import random
from typing import Optional

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class DistributedLock:
    """基于 Redis 的异步分布式锁，支持自动续期"""

    def __init__(self, redis_url: str = "redis://localhost:6379", lock_key: str = "codeai_lock", timeout: int = 10, auto_renewal: bool = True):
        self.redis_url = redis_url
        self.lock_key = lock_key
        self.timeout = timeout
        self.auto_renewal = auto_renewal
        self._redis = None
        self._lock_value = None
        self._renewal_task = None

    async def _get_redis(self):
        if self._redis is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError("redis-py (asyncio) is not installed. Install with: pip install redis")
            self._redis = await redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def acquire(self, blocking: bool = True, blocking_timeout: Optional[float] = None) -> bool:
        """获取锁，支持阻塞等待"""
        redis_client = await self._get_redis()
        self._lock_value = str(random.randint(1, 1000000))
        start = asyncio.get_event_loop().time()
        while True:
            acquired = await redis_client.setnx(self.lock_key, self._lock_value)
            if acquired:
                await redis_client.expire(self.lock_key, self.timeout)
                if self.auto_renewal:
                    self._renewal_task = asyncio.create_task(self._renew_lock())
                return True
            if not blocking:
                return False
            if blocking_timeout is not None and (asyncio.get_event_loop().time() - start) > blocking_timeout:
                return False
            await asyncio.sleep(0.1)

    async def _renew_lock(self):
        """自动续期（每 timeout/2 秒刷新）"""
        redis_client = await self._get_redis()
        while True:
            await asyncio.sleep(self.timeout / 2)
            # 检查锁是否仍属于自己
            current = await redis_client.get(self.lock_key)
            if current == self._lock_value:
                await redis_client.expire(self.lock_key, self.timeout)
            else:
                break

    async def release(self):
        """释放锁"""
        if self._renewal_task:
            self._renewal_task.cancel()
        redis_client = await self._get_redis()
        # 使用 Lua 脚本保证原子性
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        await redis_client.eval(lua_script, 1, self.lock_key, self._lock_value)
        self._lock_value = None

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


