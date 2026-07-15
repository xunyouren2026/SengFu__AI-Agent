"""
Multi-Agent System
多智能体系统

提供智能体协作、任务分解、分布式决策等功能
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"  # 协调者
    WORKER = "worker"  # 执行者
    SPECIALIST = "specialist"  # 专家
    CRITIC = "critic"  # 评论者
    PLANNER = "planner"  # 规划者
    LEARNER = "learner"  # 学习者


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """智能体状态"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class AgentCapability:
    """智能体能力"""
    name: str
    description: str = ""
    proficiency: float = 0.5  # 熟练度 0-1
    cost: float = 0.0  # 执行成本
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "proficiency": self.proficiency,
            "cost": self.cost,
        }


@dataclass
class AgentMessage:
    """智能体消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    content: str = ""
    message_type: str = "text"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "content": self.content,
            "message_type": self.message_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class Task:
    """任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    assigned_to: Optional[str] = None
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "assigned_to": self.assigned_to,
            "parent_id": self.parent_id,
            "subtasks": self.subtasks,
            "dependencies": self.dependencies,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass
class Agent:
    """智能体"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: AgentRole = AgentRole.WORKER
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[AgentCapability] = field(default_factory=list)
    current_task: Optional[str] = None
    task_history: List[str] = field(default_factory=list)
    performance_score: float = 0.5
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "current_task": self.current_task,
            "task_history": self.task_history,
            "performance_score": self.performance_score,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    def has_capability(self, capability_name: str) -> bool:
        """检查是否具有某能力"""
        return any(c.name == capability_name for c in self.capabilities)
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """获取能力"""
        for c in self.capabilities:
            if c.name == capability_name:
                return c
        return None


class AgentBase(ABC):
    """智能体基类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._agent_info: Optional[Agent] = None
        self._llm_client = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._initialized = False
        
    async def initialize(self):
        """初始化智能体"""
        if self._initialized:
            return
        
        # 初始化LLM客户端
        try:
            from openai import AsyncOpenAI
            
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self._llm_client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            pass
        
        self._initialized = True
    
    @abstractmethod
    async def execute(self, task: Task) -> Any:
        """执行任务"""
        pass
    
    async def communicate(self, message: AgentMessage) -> Optional[AgentMessage]:
        """通信"""
        await self._message_queue.put(message)
        return None
    
    async def receive_message(self) -> AgentMessage:
        """接收消息"""
        return await self._message_queue.get()
    
    def get_info(self) -> Agent:
        """获取智能体信息"""
        return self._agent_info


class CoordinatorAgent(AgentBase):
    """协调者智能体"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._agent_info = Agent(
            name="Coordinator",
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability("task_decomposition", "任务分解", 0.9),
                AgentCapability("resource_allocation", "资源分配", 0.8),
                AgentCapability("conflict_resolution", "冲突解决", 0.7),
            ],
        )
        self._workers: Dict[str, Agent] = {}
        self._tasks: Dict[str, Task] = {}
        
    async def execute(self, task: Task) -> Any:
        """执行协调任务"""
        await self.initialize()
        
        # 分解任务
        subtasks = await self._decompose_task(task)
        
        # 分配任务
        assignments = await self._assign_tasks(subtasks)
        
        # 监控执行
        results = await self._monitor_execution(subtasks, assignments)
        
        # 合并结果
        final_result = await self._merge_results(results)
        
        return final_result
    
    async def _decompose_task(self, task: Task) -> List[Task]:
        """分解任务"""
        if self._llm_client:
            prompt = f"""请将以下任务分解为子任务:

任务: {task.description}

请以JSON格式返回子任务列表，每个子任务包含:
- name: 子任务名称
- description: 子任务描述
- priority: 优先级(1-5)
- dependencies: 依赖的其他子任务ID列表
"""
            
            try:
                response = await self._llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                
                content = response.choices[0].message.content
                
                # 解析子任务
                import json
                subtask_data = json.loads(content)
                
                subtasks = []
                for i, data in enumerate(subtask_data):
                    subtask = Task(
                        name=data.get("name", f"Subtask {i+1}"),
                        description=data.get("description", ""),
                        priority=data.get("priority", 1),
                        parent_id=task.id,
                        dependencies=data.get("dependencies", []),
                    )
                    subtasks.append(subtask)
                    self._tasks[subtask.id] = subtask
                    task.subtasks.append(subtask.id)
                
                return subtasks
                
            except Exception as e:
                logger.error(f"Task decomposition failed: {e}")
        
        # 简单分解
        return [Task(
            name=f"Subtask for {task.name}",
            description=task.description,
            parent_id=task.id,
        )]
    
    async def _assign_tasks(self, tasks: List[Task]) -> Dict[str, str]:
        """分配任务"""
        assignments = {}
        
        for task in tasks:
            # 找到最合适的智能体
            best_agent = await self._find_best_agent(task)
            if best_agent:
                task.assigned_to = best_agent.id
                task.status = TaskStatus.ASSIGNED
                assignments[task.id] = best_agent.id
        
        return assignments
    
    async def _find_best_agent(self, task: Task) -> Optional[Agent]:
        """找到最合适的智能体"""
        best = None
        best_score = -1
        
        for agent in self._workers.values():
            if agent.status != AgentStatus.IDLE:
                continue
            
            # 计算匹配分数
            score = 0
            for capability in agent.capabilities:
                if task.description.lower().find(capability.name.lower()) >= 0:
                    score += capability.proficiency
            
            score += agent.performance_score * 0.5
            
            if score > best_score:
                best_score = score
                best = agent
        
        return best
    
    async def _monitor_execution(
        self,
        tasks: List[Task],
        assignments: Dict[str, str],
    ) -> Dict[str, Any]:
        """监控执行"""
        results = {}
        
        # 按依赖关系排序
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            # 等待依赖完成
            for dep_id in task.dependencies:
                if dep_id in results:
                    while results[dep_id] is None:
                        await asyncio.sleep(0.1)
            
            # 执行任务
            agent_id = assignments.get(task.id)
            if agent_id and agent_id in self._workers:
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                # TODO: 实际执行
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                results[task.id] = task.result
        
        return results
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """拓扑排序"""
        # 简单实现
        return tasks
    
    async def _merge_results(self, results: Dict[str, Any]) -> Any:
        """合并结果"""
        return results
    
    def register_worker(self, agent: Agent):
        """注册工作智能体"""
        self._workers[agent.id] = agent
    
    def unregister_worker(self, agent_id: str):
        """注销工作智能体"""
        if agent_id in self._workers:
            del self._workers[agent_id]


class WorkerAgent(AgentBase):
    """工作智能体"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._agent_info = Agent(
            name="Worker",
            role=AgentRole.WORKER,
            capabilities=[
                AgentCapability("execution", "任务执行", 0.8),
                AgentCapability("reporting", "结果报告", 0.7),
            ],
        )
        
    async def execute(self, task: Task) -> Any:
        """执行任务"""
        await self.initialize()
        
        self._agent_info.status = AgentStatus.BUSY
        self._agent_info.current_task = task.id
        
        try:
            # 使用LLM执行
            if self._llm_client:
                response = await self._llm_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": task.description}],
                    temperature=0.7,
                )
                
                result = response.choices[0].message.content
            else:
                result = f"[Mock] Executed: {task.description}"
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return None
            
        finally:
            self._agent_info.status = AgentStatus.IDLE
            self._agent_info.current_task = None
            self._agent_info.task_history.append(task.id)


class SpecialistAgent(AgentBase):
    """专家智能体"""
    
    def __init__(
        self,
        specialty: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self._specialty = specialty
        self._agent_info = Agent(
            name=f"{specialty} Specialist",
            role=AgentRole.SPECIALIST,
            capabilities=[
                AgentCapability(specialty, f"{specialty} expertise", 0.9),
            ],
        )
        
    async def execute(self, task: Task) -> Any:
        """执行专家任务"""
        await self.initialize()
        
        # 专家特定执行逻辑
        if self._llm_client:
            prompt = f"""作为{self._specialty}领域的专家，请完成以下任务:

{task.description}

请提供专业的分析和建议。
"""
            
            response = await self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            return response.choices[0].message.content
        
        return f"[Mock] {self._specialty} specialist response for: {task.description}"


class CriticAgent(AgentBase):
    """评论者智能体"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._agent_info = Agent(
            name="Critic",
            role=AgentRole.CRITIC,
            capabilities=[
                AgentCapability("evaluation", "结果评估", 0.9),
                AgentCapability("feedback", "反馈提供", 0.8),
            ],
        )
        
    async def execute(self, task: Task) -> Any:
        """执行评论任务"""
        await self.initialize()
        
        if self._llm_client:
            prompt = f"""请评估以下任务执行结果:

任务: {task.description}
结果: {task.result}

请提供:
1. 优点
2. 缺点
3. 改进建议
4. 总体评分(1-10)
"""
            
            response = await self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            
            return response.choices[0].message.content
        
        return f"[Mock] Critic evaluation for: {task.name}"


class MultiAgentSystem:
    """
    多智能体系统
    
    功能：
    - 智能体管理
    - 任务调度
    - 协作协调
    - 结果聚合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._agents: Dict[str, AgentBase] = {}
        self._agent_infos: Dict[str, Agent] = {}
        self._tasks: Dict[str, Task] = {}
        self._coordinator: Optional[CoordinatorAgent] = None
        self._initialized = False
        
    async def initialize(self):
        """初始化多智能体系统"""
        if self._initialized:
            return
        
        # 创建协调者
        self._coordinator = CoordinatorAgent(self.config)
        await self._coordinator.initialize()
        
        self._initialized = True
        logger.info("Multi-agent system initialized")
    
    async def create_agent(
        self,
        role: AgentRole,
        name: str = "",
        capabilities: Optional[List[AgentCapability]] = None,
    ) -> Agent:
        """创建智能体"""
        await self.initialize()
        
        # 创建智能体实例
        if role == AgentRole.COORDINATOR:
            agent = CoordinatorAgent(self.config)
        elif role == AgentRole.WORKER:
            agent = WorkerAgent(self.config)
        elif role == AgentRole.SPECIALIST:
            agent = SpecialistAgent(name or "General", self.config)
        elif role == AgentRole.CRITIC:
            agent = CriticAgent(self.config)
        else:
            agent = WorkerAgent(self.config)
        
        await agent.initialize()
        
        # 更新信息
        info = agent.get_info()
        if name:
            info.name = name
        if capabilities:
            info.capabilities = capabilities
        
        self._agents[info.id] = agent
        self._agent_infos[info.id] = info
        
        # 注册到协调者
        if role != AgentRole.COORDINATOR and self._coordinator:
            self._coordinator.register_worker(info)
        
        logger.info(f"Created agent: {info.name} ({info.id})")
        
        return info
    
    async def submit_task(
        self,
        name: str,
        description: str,
        priority: int = 1,
        parent_id: Optional[str] = None,
    ) -> Task:
        """提交任务"""
        await self.initialize()
        
        task = Task(
            name=name,
            description=description,
            priority=priority,
            parent_id=parent_id,
        )
        
        self._tasks[task.id] = task
        
        logger.info(f"Submitted task: {task.name} ({task.id})")
        
        return task
    
    async def execute_task(self, task_id: str) -> Any:
        """执行任务"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        # 使用协调者执行
        if self._coordinator:
            return await self._coordinator.execute(task)
        
        # 直接执行
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        # 找到合适的智能体
        for agent_id, agent in self._agents.items():
            info = self._agent_infos[agent_id]
            if info.status == AgentStatus.IDLE:
                result = await agent.execute(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                return result
        
        task.status = TaskStatus.FAILED
        task.error = "No available agent"
        return None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self._tasks.get(task_id)
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取智能体"""
        return self._agent_infos.get(agent_id)
    
    async def list_agents(self, role: Optional[AgentRole] = None) -> List[Agent]:
        """列出智能体"""
        agents = list(self._agent_infos.values())
        
        if role:
            agents = [a for a in agents if a.role == role]
        
        return agents
    
    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """列出任务"""
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    async def communicate(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
    ) -> Optional[AgentMessage]:
        """智能体通信"""
        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
        )
        
        if receiver_id in self._agents:
            receiver = self._agents[receiver_id]
            return await receiver.communicate(message)
        
        return None
    
    async def broadcast(
        self,
        sender_id: str,
        content: str,
        exclude: Optional[List[str]] = None,
    ) -> List[AgentMessage]:
        """广播消息"""
        exclude = exclude or []
        responses = []
        
        for agent_id, agent in self._agents.items():
            if agent_id != sender_id and agent_id not in exclude:
                response = await self.communicate(sender_id, agent_id, content)
                if response:
                    responses.append(response)
        
        return responses
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_agents": len(self._agents),
            "agents_by_role": {
                role.value: len([a for a in self._agent_infos.values() if a.role == role])
                for role in AgentRole
            },
            "agents_by_status": {
                status.value: len([a for a in self._agent_infos.values() if a.status == status])
                for status in AgentStatus
            },
            "total_tasks": len(self._tasks),
            "tasks_by_status": {
                status.value: len([t for t in self._tasks.values() if t.status == status])
                for status in TaskStatus
            },
        }


# 全局实例
_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """获取全局多智能体系统"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


async def init_multi_agent_system(config: Optional[Dict[str, Any]] = None):
    """初始化全局多智能体系统"""
    global _multi_agent_system
    _multi_agent_system = MultiAgentSystem(config)
    await _multi_agent_system.initialize()
    return _multi_agent_system
