"""
仿真世界引擎 - 离散时间步推进，管理所有Agent状态
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid


class AgentState(Enum):
    """Agent状态枚举"""
    ACTIVE = auto()
    INACTIVE = auto()
    TERMINATED = auto()


@dataclass
class Position:
    """2D位置"""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: Position) -> float:
        """计算到另一个位置的距离"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __add__(self, other: Position) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> Position:
        return Position(self.x * scalar, self.y * scalar)


@dataclass
class Agent:
    """基础智能体类"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    position: Position = field(default_factory=Position)
    state: AgentState = AgentState.ACTIVE
    energy: float = 100.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    max_memory_size: int = 100

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())[:8]

    def update(self, world: World, dt: float) -> None:
        """Agent更新逻辑，子类可重写"""
        self.energy -= dt * 0.1  # 基础能量消耗
        if self.energy <= 0:
            self.state = AgentState.INACTIVE

    def perceive(self, world: World) -> Dict[str, Any]:
        """感知周围环境"""
        nearby = world.get_agents_near(self.position, radius=10.0)
        return {
            "nearby_agents": [a.agent_id for a in nearby if a.agent_id != self.agent_id],
            "position": (self.position.x, self.position.y),
            "energy": self.energy,
            "world_time": world.current_time
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """基于感知做出决策"""
        return {"action": "idle", "params": {}}

    def act(self, decision: Dict[str, Any], world: World) -> None:
        """执行决策"""
        action = decision.get("action", "idle")
        if action == "move":
            dx = decision.get("params", {}).get("dx", 0)
            dy = decision.get("params", {}).get("dy", 0)
            self.position.x += dx
            self.position.y += dy
            self.energy -= 1.0
        elif action == "rest":
            self.energy = min(100.0, self.energy + 5.0)

    def add_memory(self, event: Dict[str, Any]) -> None:
        """添加记忆"""
        event["timestamp"] = random.random()  # 简化时间戳
        self.memory.append(event)
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)


class World:
    """
    仿真世界引擎
    管理所有Agent，按离散时间步推进仿真
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        time_step: float = 1.0,
        max_steps: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        self.width = width
        self.height = height
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_time: float = 0.0
        self.current_step: int = 0
        self.agents: Dict[str, Agent] = {}
        self.terminated_agents: Dict[str, Agent] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.step_callbacks: List[Callable[[World], None]] = []
        self.interaction_callbacks: List[Callable[[Agent, Agent, World], None]] = []

        if random_seed is not None:
            random.seed(random_seed)

    def add_agent(self, agent: Agent) -> str:
        """添加Agent到世界"""
        self.agents[agent.agent_id] = agent
        self._log_event("agent_added", {"agent_id": agent.agent_id})
        return agent.agent_id

    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        """从世界移除Agent"""
        agent = self.agents.pop(agent_id, None)
        if agent:
            agent.state = AgentState.TERMINATED
            self.terminated_agents[agent_id] = agent
            self._log_event("agent_removed", {"agent_id": agent_id})
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """获取指定Agent"""
        return self.agents.get(agent_id)

    def get_all_agents(self) -> List[Agent]:
        """获取所有活跃Agent"""
        return list(self.agents.values())

    def get_agents_near(
        self,
        position: Position,
        radius: float,
        agent_type: Optional[type] = None
    ) -> List[Agent]:
        """获取指定位置附近的Agent"""
        nearby = []
        for agent in self.agents.values():
            if agent.position.distance_to(position) <= radius:
                if agent_type is None or isinstance(agent, agent_type):
                    nearby.append(agent)
        return nearby

    def get_agents_in_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ) -> List[Agent]:
        """获取区域内的Agent"""
        return [
            agent for agent in self.agents.values()
            if x_min <= agent.position.x <= x_max and y_min <= agent.position.y <= y_max
        ]

    def step(self) -> bool:
        """
        执行一个仿真时间步
        返回: 是否继续仿真
        """
        if self.max_steps and self.current_step >= self.max_steps:
            return False

        # 1. 收集所有Agent的感知
        perceptions: Dict[str, Dict[str, Any]] = {}
        for agent_id, agent in list(self.agents.items()):
            if agent.state == AgentState.ACTIVE:
                perceptions[agent_id] = agent.perceive(self)

        # 2. 所有Agent决策
        decisions: Dict[str, Dict[str, Any]] = {}
        for agent_id, perception in perceptions.items():
            agent = self.agents[agent_id]
            decisions[agent_id] = agent.decide(perception)

        # 3. 执行决策
        for agent_id, decision in decisions.items():
            agent = self.agents[agent_id]
            agent.act(decision, self)
            agent.update(self, self.time_step)

        # 4. 处理交互
        self._process_interactions()

        # 5. 边界检查
        self._enforce_boundaries()

        # 6. 清理无效Agent
        self._cleanup_agents()

        # 7. 执行回调
        for callback in self.step_callbacks:
            callback(self)

        self.current_time += self.time_step
        self.current_step += 1

        self._log_event("step_completed", {
            "step": self.current_step,
            "active_agents": len(self.agents)
        })

        return True

    def _process_interactions(self) -> None:
        """处理Agent间的交互"""
        agent_list = list(self.agents.values())
        for i, agent1 in enumerate(agent_list):
            for agent2 in agent_list[i + 1:]:
                distance = agent1.position.distance_to(agent2.position)
                if distance < 2.0:  # 交互距离阈值
                    for callback in self.interaction_callbacks:
                        callback(agent1, agent2, self)

    def _enforce_boundaries(self) -> None:
        """强制执行世界边界"""
        for agent in self.agents.values():
            agent.position.x = max(0, min(self.width, agent.position.x))
            agent.position.y = max(0, min(self.height, agent.position.y))

    def _cleanup_agents(self) -> None:
        """清理无效Agent"""
        to_remove = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.state == AgentState.INACTIVE
        ]
        for agent_id in to_remove:
            self.remove_agent(agent_id)

    def run(self, steps: Optional[int] = None) -> None:
        """运行仿真"""
        if steps:
            for _ in range(steps):
                if not self.step():
                    break
        else:
            while self.step():
                pass

    def register_step_callback(self, callback: Callable[[World], None]) -> None:
        """注册每步回调"""
        self.step_callbacks.append(callback)

    def register_interaction_callback(
        self,
        callback: Callable[[Agent, Agent, World], None]
    ) -> None:
        """注册交互回调"""
        self.interaction_callbacks.append(callback)

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """记录事件"""
        self.event_log.append({
            "step": self.current_step,
            "time": self.current_time,
            "type": event_type,
            "data": data
        })

    def get_statistics(self) -> Dict[str, Any]:
        """获取仿真统计信息"""
        return {
            "total_steps": self.current_step,
            "simulation_time": self.current_time,
            "active_agents": len(self.agents),
            "terminated_agents": len(self.terminated_agents),
            "total_events": len(self.event_log)
        }

    def reset(self) -> None:
        """重置世界"""
        self.current_time = 0.0
        self.current_step = 0
        self.agents.clear()
        self.terminated_agents.clear()
        self.event_log.clear()


class SpatialGrid:
    """
    空间网格索引，加速邻近查询
    """

    def __init__(self, world_width: float, world_height: float, cell_size: float):
        self.cell_size = cell_size
        self.cols = int(world_width / cell_size) + 1
        self.rows = int(world_height / cell_size) + 1
        self.grid: Dict[Tuple[int, int], Set[str]] = {}

    def _get_cell(self, position: Position) -> Tuple[int, int]:
        """获取位置对应的网格单元"""
        col = int(position.x / self.cell_size)
        row = int(position.y / self.cell_size)
        return (max(0, min(col, self.cols - 1)), max(0, min(row, self.rows - 1)))

    def add_agent(self, agent: Agent) -> None:
        """添加Agent到网格"""
        cell = self._get_cell(agent.position)
        if cell not in self.grid:
            self.grid[cell] = set()
        self.grid[cell].add(agent.agent_id)

    def remove_agent(self, agent: Agent) -> None:
        """从网格移除Agent"""
        cell = self._get_cell(agent.position)
        if cell in self.grid:
            self.grid[cell].discard(agent.agent_id)

    def update_agent(self, agent: Agent, old_position: Position) -> None:
        """更新Agent位置"""
        old_cell = self._get_cell(old_position)
        new_cell = self._get_cell(agent.position)
        if old_cell != new_cell:
            if old_cell in self.grid:
                self.grid[old_cell].discard(agent.agent_id)
            if new_cell not in self.grid:
                self.grid[new_cell] = set()
            self.grid[new_cell].add(agent.agent_id)

    def get_nearby(self, position: Position, radius: float) -> Set[str]:
        """获取附近的Agent ID"""
        cell_radius = int(radius / self.cell_size) + 1
        center_cell = self._get_cell(position)
        nearby = set()

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid:
                    nearby.update(self.grid[cell])

        return nearby
