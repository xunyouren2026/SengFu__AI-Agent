"""
distributed/evolution_coordinator.py
分布式进化协调：多机多卡并行进化
支持：参数服务器架构、任务分发、结果聚合、心跳检测、故障恢复
"""

import asyncio
import pickle
import socket
import threading
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class NodeRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"


@dataclass
class NodeInfo:
    node_id: str
    role: NodeRole
    address: str
    port: int
    status: str = "online"  # online, offline, busy
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class EvolutionTask:
    task_id: str
    code: str                    # 要执行的代码（或序列化函数）
    fitness_fn: str              # 适应度函数（模块路径或序列化字符串）
    parameters: Dict[str, Any]   # 进化参数（种群大小、变异率等）
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    status: str = "pending"      # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)


class EvolutionCoordinator:
    """
    分布式进化协调器（参数服务器架构）
    负责：Worker注册、任务分发、结果聚合、心跳检测、故障恢复
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8888):
        self.host = host
        self.port = port
        self.node_id = f"coord_{socket.gethostname()}_{port}"
        self.role = NodeRole.COORDINATOR

        self.workers: Dict[str, NodeInfo] = {}
        self.tasks: Dict[str, EvolutionTask] = {}
        self.task_results: Dict[str, Any] = {}

        self._server = None
        self._running = False
        self._heartbeat_thread = None
        self._task_queue = asyncio.Queue()

    async def start(self):
        """启动协调器服务"""
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        # 启动任务分发协程
        asyncio.create_task(self._task_dispatcher())

        # 启动TCP服务器
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        print(f"Evolution Coordinator started on {self.host}:{self.port}")
        async with self._server:
            await self._server.serve_forever()

    def _heartbeat_loop(self):
        """心跳检测线程"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self._running:
            now = time.time()
            offline_nodes = []
            for node_id, node in self.workers.items():
                if now - node.last_heartbeat > 30:  # 30秒无心跳
                    offline_nodes.append(node_id)
            for node_id in offline_nodes:
                print(f"Worker {node_id} offline, removing...")
                # 重新分配该worker正在执行的任务
                for task in self.tasks.values():
                    if task.assigned_to == node_id and task.status == "running":
                        task.status = "pending"
                        task.assigned_to = None
                        loop.call_soon_threadsafe(
                            lambda tid=task.task_id: asyncio.create_task(self._task_queue.put(tid))
                        )
                self.workers.pop(node_id, None)
            time.sleep(10)
        loop.close()

    async def _task_dispatcher(self):
        """任务分发协程"""
        while self._running:
            task_id = await self._task_queue.get()
            task = self.tasks.get(task_id)
            if not task or task.status != "pending":
                continue

            # 找到空闲Worker
            available_workers = [w for w in self.workers.values() if w.status == "online"]
            if not available_workers:
                await asyncio.sleep(1)
                await self._task_queue.put(task_id)
                continue

            # 简单轮询分配
            worker = available_workers[0]
            task.assigned_to = worker.node_id
            task.status = "running"
            print(f"Task {task_id} assigned to {worker.node_id}")

    async def _handle_client(self, reader, writer):
        """处理客户端连接"""
        addr = writer.get_extra_info('peername')
        try:
            data = await reader.read(4096)
            if not data:
                return
            message = pickle.loads(data)
            msg_type = message.get("type")

            handlers = {
                "register": self._handle_register,
                "heartbeat": self._handle_heartbeat,
                "get_task": self._handle_get_task,
                "submit_result": self._handle_submit_result,
            }
            handler = handlers.get(msg_type)
            if handler:
                response = await handler(message, addr)
            else:
                response = {"status": "error", "message": f"Unknown type: {msg_type}"}

            writer.write(pickle.dumps(response))
            await writer.drain()
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            writer.close()

    async def _handle_register(self, message: Dict, addr) -> Dict:
        worker_id = message.get("worker_id")
        worker_port = message.get("port", self.port + 1)
        worker_addr = addr[0]
        capabilities = message.get("capabilities", [])

        if worker_id not in self.workers:
            self.workers[worker_id] = NodeInfo(
                node_id=worker_id,
                role=NodeRole.WORKER,
                address=worker_addr,
                port=worker_port,
                status="online",
                capabilities=capabilities
            )
            print(f"Worker {worker_id} registered at {worker_addr}:{worker_port}")
        return {"status": "ok", "coordinator_id": self.node_id}

    async def _handle_heartbeat(self, message: Dict, addr) -> Dict:
        worker_id = message.get("worker_id")
        if worker_id in self.workers:
            self.workers[worker_id].last_heartbeat = time.time()
            return {"status": "ok"}
        return {"status": "error", "message": "Worker not registered"}

    async def _handle_get_task(self, message: Dict, addr) -> Dict:
        worker_id = message.get("worker_id")
        if worker_id not in self.workers:
            return {"status": "error", "message": "Worker not registered"}

        # 获取一个待处理任务
        for task_id, task in self.tasks.items():
            if task.status == "pending":
                task.status = "running"
                task.assigned_to = worker_id
                return {
                    "status": "ok",
                    "task_id": task_id,
                    "code": task.code,
                    "fitness_fn": task.fitness_fn,
                    "parameters": task.parameters
                }
        return {"status": "no_task"}

    async def _handle_submit_result(self, message: Dict, addr) -> Dict:
        task_id = message.get("task_id")
        result = message.get("result")
        worker_id = message.get("worker_id")

        if task_id in self.tasks:
            self.tasks[task_id].result = result
            self.tasks[task_id].status = "completed"
            self.task_results[task_id] = result
            print(f"Task {task_id} completed by {worker_id}")
            return {"status": "ok"}
        return {"status": "error", "message": "Task not found"}

    def submit_task(self, code: str, fitness_fn: str,
                    parameters: Dict[str, Any]) -> str:
        """提交进化任务"""
        task_id = str(uuid.uuid4())[:8]
        task = EvolutionTask(
            task_id=task_id,
            code=code,
            fitness_fn=fitness_fn,
            parameters=parameters
        )
        self.tasks[task_id] = task
        # 将任务加入队列
        asyncio.create_task(self._task_queue.put(task_id))
        return task_id

    def get_result(self, task_id: str) -> Optional[Any]:
        return self.task_results.get(task_id)

    def get_worker_count(self) -> int:
        return len(self.workers)

    def stop(self):
        self._running = False
        if self._server:
            self._server.close()


class EvolutionWorker:
    """进化工作节点"""

    def __init__(self, coordinator_host: str = "localhost",
                 coordinator_port: int = 8888,
                 worker_port: int = 8889,
                 capabilities: List[str] = None):
        self.coordinator_addr = (coordinator_host, coordinator_port)
        self.worker_port = worker_port
        self.worker_id = f"worker_{socket.gethostname()}_{worker_port}"
        self.capabilities = capabilities or ["python", "evolution"]
        self._running = False

    async def start(self):
        """启动Worker"""
        self._running = True
        await self._register()
        asyncio.create_task(self._heartbeat_loop())
        await self._work_loop()

    async def _register(self):
        reader, writer = await asyncio.open_connection(*self.coordinator_addr)
        message = {
            "type": "register",
            "worker_id": self.worker_id,
            "port": self.worker_port,
            "capabilities": self.capabilities
        }
        writer.write(pickle.dumps(message))
        await writer.drain()
        response = pickle.loads(await reader.read(4096))
        writer.close()
        print(f"Registered with coordinator: {response}")

    async def _heartbeat_loop(self):
        while self._running:
            try:
                reader, writer = await asyncio.open_connection(*self.coordinator_addr)
                message = {"type": "heartbeat", "worker_id": self.worker_id}
                writer.write(pickle.dumps(message))
                await writer.drain()
                response = pickle.loads(await reader.read(4096))
                writer.close()
            except Exception as e:
                print(f"Heartbeat error: {e}")
            await asyncio.sleep(15)

    async def _work_loop(self):
        while self._running:
            try:
                reader, writer = await asyncio.open_connection(*self.coordinator_addr)
                message = {"type": "get_task", "worker_id": self.worker_id}
                writer.write(pickle.dumps(message))
                await writer.drain()
                response = pickle.loads(await reader.read(4096))
                writer.close()

                if response.get("status") == "ok":
                    task_id = response["task_id"]
                    code = response["code"]
                    fitness_fn_str = response["fitness_fn"]
                    params = response["parameters"]

                    # 执行进化任务
                    result = self._execute_task(code, fitness_fn_str, params)

                    # 提交结果
                    await self._submit_result(task_id, result)
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"Work loop error: {e}")
                await asyncio.sleep(5)

    def _execute_task(self, code: str, fitness_fn_str: str,
                      params: Dict[str, Any]) -> Any:
        """执行进化任务（需根据实际实现）"""
        # 示例：返回随机适应度
        import random
        return {
            "best_fitness": random.uniform(0, 1),
            "best_individual": code[:100],
            "generations": params.get("generations", 10)
        }

    async def _submit_result(self, task_id: str, result: Any):
        reader, writer = await asyncio.open_connection(*self.coordinator_addr)
        message = {
            "type": "submit_result",
            "worker_id": self.worker_id,
            "task_id": task_id,
            "result": result
        }
        writer.write(pickle.dumps(message))
        await writer.drain()
        reader.close()

    def stop(self):
        self._running = False


if __name__ == "__main__":
    print("Distributed Evolution Coordinator module ready.")
    print("Run coordinator: coordinator = EvolutionCoordinator(); asyncio.run(coordinator.start())")
    print("Run worker: worker = EvolutionWorker(); asyncio.run(worker.start())")


