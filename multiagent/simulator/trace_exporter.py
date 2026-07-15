"""
仿真轨迹导出 - 保存每一步的所有Agent决策
"""
from __future__ import annotations
import json
import csv
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

from .world import Agent, World, Position, AgentState


@dataclass
class AgentSnapshot:
    """Agent状态快照"""
    agent_id: str
    step: int
    timestamp: float
    position: Tuple[float, float]
    state: str
    energy: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    decision: Optional[Dict[str, Any]] = None
    perception: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_id": self.agent_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "position": self.position,
            "state": self.state,
            "energy": self.energy,
            "attributes": self.attributes,
            "decision": self.decision,
            "perception": self.perception
        }


@dataclass
class WorldSnapshot:
    """世界状态快照"""
    step: int
    timestamp: float
    agent_count: int
    event_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "agent_count": self.agent_count,
            "event_count": self.event_count,
            "metadata": self.metadata
        }


@dataclass
class InteractionRecord:
    """交互记录"""
    step: int
    timestamp: float
    agent1_id: str
    agent2_id: str
    interaction_type: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "agent1_id": self.agent1_id,
            "agent2_id": self.agent2_id,
            "interaction_type": self.interaction_type,
            "details": self.details
        }


class TraceExporter:
    """
    仿真轨迹导出器
    记录和导出仿真过程中的所有决策和状态
    """

    def __init__(self, world: World, output_dir: str = "./simulation_traces"):
        self.world = world
        self.output_dir = output_dir
        self.agent_snapshots: List[AgentSnapshot] = []
        self.world_snapshots: List[WorldSnapshot] = []
        self.interactions: List[InteractionRecord] = []
        self.decisions: List[Dict[str, Any]] = []
        self.custom_events: List[Dict[str, Any]] = []
        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        """设置世界回调"""
        self.world.register_step_callback(self._on_step)
        self.world.register_interaction_callback(self._on_interaction)

    def _on_step(self, world: World) -> None:
        """每步回调"""
        # 记录世界状态
        world_snapshot = WorldSnapshot(
            step=world.current_step,
            timestamp=world.current_time,
            agent_count=len(world.agents),
            event_count=len(world.event_log)
        )
        self.world_snapshots.append(world_snapshot)

        # 记录每个Agent状态
        for agent in world.get_all_agents():
            snapshot = AgentSnapshot(
                agent_id=agent.agent_id,
                step=world.current_step,
                timestamp=world.current_time,
                position=(agent.position.x, agent.position.y),
                state=agent.state.name,
                energy=agent.energy,
                attributes=dict(agent.attributes)
            )
            self.agent_snapshots.append(snapshot)

    def _on_interaction(self, agent1: Agent, agent2: Agent, world: World) -> None:
        """交互回调"""
        record = InteractionRecord(
            step=world.current_step,
            timestamp=world.current_time,
            agent1_id=agent1.agent_id,
            agent2_id=agent2.agent_id,
            interaction_type="proximity",
            details={
                "distance": agent1.position.distance_to(agent2.position),
                "agent1_energy": agent1.energy,
                "agent2_energy": agent2.energy
            }
        )
        self.interactions.append(record)

    def record_decision(self, agent_id: str, decision: Dict[str, Any],
                       perception: Optional[Dict[str, Any]] = None) -> None:
        """记录Agent决策"""
        decision_record = {
            "step": self.world.current_step,
            "timestamp": self.world.current_time,
            "agent_id": agent_id,
            "decision": decision,
            "perception": perception
        }
        self.decisions.append(decision_record)

    def record_custom_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """记录自定义事件"""
        event = {
            "step": self.world.current_step,
            "timestamp": self.world.current_time,
            "event_type": event_type,
            "data": data
        }
        self.custom_events.append(event)

    def export_json(self, filename: str = "simulation_trace.json") -> str:
        """导出为JSON格式"""
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        trace_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_steps": self.world.current_step,
                "total_agents": len(self.world.agents) + len(self.world.terminated_agents),
                "world_size": {"width": self.world.width, "height": self.world.height}
            },
            "world_snapshots": [s.to_dict() for s in self.world_snapshots],
            "agent_snapshots": [s.to_dict() for s in self.agent_snapshots],
            "interactions": [i.to_dict() for i in self.interactions],
            "decisions": self.decisions,
            "custom_events": self.custom_events
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, default=str)

        return filepath

    def export_csv(self, filename_prefix: str = "simulation") -> List[str]:
        """导出为CSV格式"""
        os.makedirs(self.output_dir, exist_ok=True)
        files_created = []

        # 导出Agent快照
        if self.agent_snapshots:
            agent_file = os.path.join(self.output_dir, f"{filename_prefix}_agents.csv")
            with open(agent_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "timestamp", "agent_id", "pos_x", "pos_y",
                               "state", "energy"])
                for snapshot in self.agent_snapshots:
                    writer.writerow([
                        snapshot.step,
                        snapshot.timestamp,
                        snapshot.agent_id,
                        snapshot.position[0],
                        snapshot.position[1],
                        snapshot.state,
                        snapshot.energy
                    ])
            files_created.append(agent_file)

        # 导出交互记录
        if self.interactions:
            interaction_file = os.path.join(self.output_dir, f"{filename_prefix}_interactions.csv")
            with open(interaction_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "timestamp", "agent1_id", "agent2_id", "type"])
                for record in self.interactions:
                    writer.writerow([
                        record.step,
                        record.timestamp,
                        record.agent1_id,
                        record.agent2_id,
                        record.interaction_type
                    ])
            files_created.append(interaction_file)

        # 导出世界快照
        if self.world_snapshots:
            world_file = os.path.join(self.output_dir, f"{filename_prefix}_world.csv")
            with open(world_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "timestamp", "agent_count", "event_count"])
                for snapshot in self.world_snapshots:
                    writer.writerow([
                        snapshot.step,
                        snapshot.timestamp,
                        snapshot.agent_count,
                        snapshot.event_count
                    ])
            files_created.append(world_file)

        return files_created

    def export_trajectory(self, agent_id: str, filename: Optional[str] = None) -> str:
        """导出单个Agent的轨迹"""
        if filename is None:
            filename = f"trajectory_{agent_id}.json"

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        # 筛选该Agent的快照
        agent_snapshots = [s for s in self.agent_snapshots if s.agent_id == agent_id]

        trajectory = {
            "agent_id": agent_id,
            "total_points": len(agent_snapshots),
            "path": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "position": s.position,
                    "state": s.state,
                    "energy": s.energy
                }
                for s in agent_snapshots
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, indent=2)

        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """获取导出统计信息"""
        return {
            "total_agent_snapshots": len(self.agent_snapshots),
            "total_world_snapshots": len(self.world_snapshots),
            "total_interactions": len(self.interactions),
            "total_decisions": len(self.decisions),
            "total_custom_events": len(self.custom_events),
            "unique_agents": len(set(s.agent_id for s in self.agent_snapshots)),
            "simulation_steps": self.world.current_step,
            "simulation_time": self.world.current_time
        }

    def clear(self) -> None:
        """清除所有记录"""
        self.agent_snapshots.clear()
        self.world_snapshots.clear()
        self.interactions.clear()
        self.decisions.clear()
        self.custom_events.clear()

    def get_agent_decision_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """获取Agent的决策历史"""
        return [d for d in self.decisions if d.get("agent_id") == agent_id]

    def get_step_summary(self, step: int) -> Dict[str, Any]:
        """获取某一步的摘要"""
        agent_snapshots = [s for s in self.agent_snapshots if s.step == step]
        interactions = [i for i in self.interactions if i.step == step]
        decisions = [d for d in self.decisions if d.get("step") == step]

        return {
            "step": step,
            "agent_count": len(agent_snapshots),
            "interactions": len(interactions),
            "decisions": len(decisions),
            "agents": [s.agent_id for s in agent_snapshots]
        }


class TrajectoryAnalyzer:
    """轨迹分析器"""

    def __init__(self, trace_exporter: TraceExporter):
        self.trace = trace_exporter

    def analyze_movement_patterns(self, agent_id: str) -> Dict[str, Any]:
        """分析移动模式"""
        snapshots = [s for s in self.trace.agent_snapshots if s.agent_id == agent_id]
        if len(snapshots) < 2:
            return {"error": "Not enough data"}

        total_distance = 0
        speeds = []

        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            dx = curr.position[0] - prev.position[0]
            dy = curr.position[1] - prev.position[1]
            distance = (dx ** 2 + dy ** 2) ** 0.5
            total_distance += distance

            dt = curr.timestamp - prev.timestamp
            if dt > 0:
                speeds.append(distance / dt)

        return {
            "total_distance": total_distance,
            "avg_speed": sum(speeds) / len(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "path_length": len(snapshots)
        }

    def analyze_interaction_frequency(self) -> Dict[str, int]:
        """分析交互频率"""
        frequency = defaultdict(int)
        for interaction in self.trace.interactions:
            key = f"{interaction.agent1_id}-{interaction.agent2_id}"
            frequency[key] += 1
        return dict(frequency)

    def get_activity_heatmap(self, grid_size: float = 10.0) -> Dict[Tuple[int, int], int]:
        """生成活动热力图"""
        heatmap = defaultdict(int)
        for snapshot in self.trace.agent_snapshots:
            x = int(snapshot.position[0] / grid_size)
            y = int(snapshot.position[1] / grid_size)
            heatmap[(x, y)] += 1
        return dict(heatmap)

    def export_analysis(self, filename: str = "analysis.json") -> str:
        """导出分析结果"""
        os.makedirs(self.trace.output_dir, exist_ok=True)
        filepath = os.path.join(self.trace.output_dir, filename)

        analysis = {
            "interaction_frequency": self.analyze_interaction_frequency(),
            "activity_heatmap": {f"{k[0]},{k[1]}": v for k, v in self.get_activity_heatmap().items()},
            "agent_movements": {
                agent_id: self.analyze_movement_patterns(agent_id)
                for agent_id in set(s.agent_id for s in self.trace.agent_snapshots)
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)

        return filepath
