"""
智能体生成器 - 按配置批量初始化异构Agent
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Callable, Any, Type, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from .world import Agent, Position, World


class AgentType(Enum):
    """预定义Agent类型"""
    EXPLORER = auto()      # 探索者
    DEFENDER = auto()      # 防御者
    COOPERATOR = auto()    # 合作者
    COMPETITOR = auto()    # 竞争者
    ADAPTOR = auto()       # 适应者


@dataclass
class AgentTemplate:
    """Agent模板配置"""
    agent_class: Type[Agent] = Agent
    name_prefix: str = "agent"
    initial_energy_range: tuple[float, float] = (80.0, 120.0)
    attributes: Dict[str, Any] = field(default_factory=dict)
    behavior_weights: Dict[str, float] = field(default_factory=dict)
    spawn_region: Optional[tuple[float, float, float, float]] = None  # x_min, x_max, y_min, y_max


@dataclass
class SpawnConfig:
    """生成配置"""
    count: int = 10
    agent_type: AgentType = AgentType.EXPLORER
    template: Optional[AgentTemplate] = None
    distribution: str = "random"  # random, cluster, grid, gaussian
    distribution_params: Dict[str, Any] = field(default_factory=dict)


class AgentSpawner:
    """
    智能体生成器
    支持批量生成、异构配置、空间分布控制
    """

    def __init__(self, world: World, random_seed: Optional[int] = None):
        self.world = world
        self.agents_created: List[str] = []
        self.spawn_history: List[Dict[str, Any]] = []
        self._type_registry: Dict[AgentType, AgentTemplate] = {}
        self._custom_creators: Dict[str, Callable[[], Agent]] = {}

        if random_seed is not None:
            random.seed(random_seed)

        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """注册默认Agent模板"""
        self._type_registry[AgentType.EXPLORER] = AgentTemplate(
            agent_class=Agent,
            name_prefix="explorer",
            initial_energy_range=(90.0, 110.0),
            attributes={"speed": 1.5, "vision_range": 15.0, "curiosity": 0.8},
            behavior_weights={"explore": 0.7, "rest": 0.2, "interact": 0.1}
        )

        self._type_registry[AgentType.DEFENDER] = AgentTemplate(
            agent_class=Agent,
            name_prefix="defender",
            initial_energy_range=(100.0, 150.0),
            attributes={"speed": 0.8, "defense": 1.5, "territory_radius": 10.0},
            behavior_weights={"defend": 0.6, "patrol": 0.3, "rest": 0.1}
        )

        self._type_registry[AgentType.COOPERATOR] = AgentTemplate(
            agent_class=Agent,
            name_prefix="cooperator",
            initial_energy_range=(80.0, 100.0),
            attributes={"cooperation_bias": 0.8, "trust_threshold": 0.5, "sharing_rate": 0.6},
            behavior_weights={"cooperate": 0.6, "share": 0.3, "explore": 0.1}
        )

        self._type_registry[AgentType.COMPETITOR] = AgentTemplate(
            agent_class=Agent,
            name_prefix="competitor",
            initial_energy_range=(85.0, 115.0),
            attributes={"aggression": 0.7, "competitiveness": 0.9, "resource_hoarding": 0.8},
            behavior_weights={"compete": 0.6, "hoard": 0.3, "explore": 0.1}
        )

        self._type_registry[AgentType.ADAPTOR] = AgentTemplate(
            agent_class=Agent,
            name_prefix="adaptor",
            initial_energy_range=(80.0, 120.0),
            attributes={"adaptability": 0.9, "learning_rate": 0.1, "strategy_flexibility": 0.8},
            behavior_weights={"adapt": 0.5, "learn": 0.3, "explore": 0.2}
        )

    def register_template(self, agent_type: AgentType, template: AgentTemplate) -> None:
        """注册自定义Agent模板"""
        self._type_registry[agent_type] = template

    def register_custom_creator(self, name: str, creator: Callable[[], Agent]) -> None:
        """注册自定义Agent创建器"""
        self._custom_creators[name] = creator

    def spawn(
        self,
        config: SpawnConfig,
        override_attributes: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        根据配置批量生成Agent
        返回: 生成的Agent ID列表
        """
        agent_ids = []
        template = config.template or self._type_registry.get(
            config.agent_type,
            AgentTemplate()
        )

        positions = self._generate_positions(config)

        for i, position in enumerate(positions):
            agent = self._create_agent(
                template=template,
                index=i,
                position=position,
                override_attributes=override_attributes
            )
            agent_id = self.world.add_agent(agent)
            self.agents_created.append(agent_id)
            agent_ids.append(agent_id)

        spawn_record = {
            "count": config.count,
            "type": config.agent_type.name,
            "distribution": config.distribution,
            "agent_ids": agent_ids.copy()
        }
        self.spawn_history.append(spawn_record)

        return agent_ids

    def spawn_heterogeneous(
        self,
        configs: List[SpawnConfig],
        randomize_order: bool = True
    ) -> Dict[str, List[str]]:
        """
        生成异构Agent群体
        返回: {Agent类型: Agent ID列表}
        """
        results: Dict[str, List[str]] = {}
        all_spawns: List[tuple[str, str]] = []  # (type_name, agent_id)

        for config in configs:
            agent_ids = self.spawn(config)
            type_name = config.agent_type.name
            if type_name not in results:
                results[type_name] = []
            results[type_name].extend(agent_ids)
            for aid in agent_ids:
                all_spawns.append((type_name, aid))

        if randomize_order:
            # 重新随机排列所有Agent的生成顺序（影响ID分配）
            random.shuffle(all_spawns)

        return results

    def spawn_from_distribution(
        self,
        template: AgentTemplate,
        count: int,
        attribute_distributions: Dict[str, tuple[float, float]]
    ) -> List[str]:
        """
        从概率分布生成具有连续属性的Agent
        attribute_distributions: {属性名: (均值, 标准差)}
        """
        agent_ids = []

        for i in range(count):
            # 从分布采样属性
            sampled_attrs = {}
            for attr_name, (mean, std) in attribute_distributions.items():
                sampled_attrs[attr_name] = random.gauss(mean, std)

            position = self._random_position()
            agent = self._create_agent(
                template=template,
                index=i,
                position=position,
                override_attributes=sampled_attrs
            )
            agent_id = self.world.add_agent(agent)
            agent_ids.append(agent_id)

        return agent_ids

    def _create_agent(
        self,
        template: AgentTemplate,
        index: int,
        position: Position,
        override_attributes: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """创建单个Agent"""
        # 生成能量值
        energy_min, energy_max = template.initial_energy_range
        initial_energy = random.uniform(energy_min, energy_max)

        # 合并属性
        attributes = template.attributes.copy()
        if override_attributes:
            attributes.update(override_attributes)

        # 创建Agent实例
        agent = template.agent_class(
            agent_id=f"{template.name_prefix}_{index:04d}",
            position=position,
            energy=initial_energy,
            attributes=attributes
        )

        # 添加行为权重到属性
        agent.attributes["_behavior_weights"] = template.behavior_weights.copy()

        return agent

    def _generate_positions(self, config: SpawnConfig) -> List[Position]:
        """根据分布策略生成位置"""
        distribution = config.distribution
        params = config.distribution_params
        count = config.count

        # 确定生成区域
        region = params.get("region")
        if region:
            x_min, x_max, y_min, y_max = region
        else:
            x_min, y_min = 0, 0
            x_max, y_max = self.world.width, self.world.height

        positions = []

        if distribution == "random":
            for _ in range(count):
                positions.append(self._random_position_in_region(x_min, x_max, y_min, y_max))

        elif distribution == "cluster":
            # 聚类分布
            num_clusters = params.get("num_clusters", 3)
            cluster_radius = params.get("cluster_radius", 10.0)

            # 生成聚类中心
            cluster_centers = [
                Position(
                    random.uniform(x_min, x_max),
                    random.uniform(y_min, y_max)
                )
                for _ in range(num_clusters)
            ]

            # 在聚类周围生成Agent
            for i in range(count):
                center = cluster_centers[i % num_clusters]
                angle = random.uniform(0, 2 * 3.14159)
                distance = random.uniform(0, cluster_radius)
                pos = Position(
                    center.x + distance * random.uniform(-1, 1),
                    center.y + distance * random.uniform(-1, 1)
                )
                positions.append(pos)

        elif distribution == "grid":
            # 网格分布
            cols = int(params.get("cols", (count ** 0.5)))
            rows = (count + cols - 1) // cols

            dx = (x_max - x_min) / max(cols, 1)
            dy = (y_max - y_min) / max(rows, 1)

            for i in range(count):
                col = i % cols
                row = i // cols
                pos = Position(
                    x_min + col * dx + dx / 2,
                    y_min + row * dy + dy / 2
                )
                positions.append(pos)

        elif distribution == "gaussian":
            # 高斯分布
            center_x = params.get("center_x", (x_min + x_max) / 2)
            center_y = params.get("center_y", (y_min + y_max) / 2)
            sigma_x = params.get("sigma_x", (x_max - x_min) / 6)
            sigma_y = params.get("sigma_y", (y_max - y_min) / 6)

            for _ in range(count):
                pos = Position(
                    random.gauss(center_x, sigma_x),
                    random.gauss(center_y, sigma_y)
                )
                positions.append(pos)

        elif distribution == "circle":
            # 圆形分布
            center_x = params.get("center_x", (x_min + x_max) / 2)
            center_y = params.get("center_y", (y_min + y_max) / 2)
            radius = params.get("radius", min(x_max - x_min, y_max - y_min) / 4)

            for i in range(count):
                angle = 2 * 3.14159 * i / count
                pos = Position(
                    center_x + radius * random.uniform(0.5, 1.0) * (random.choice([-1, 1])),
                    center_y + radius * random.uniform(0.5, 1.0) * (random.choice([-1, 1]))
                )
                positions.append(pos)

        else:
            # 默认随机
            for _ in range(count):
                positions.append(self._random_position_in_region(x_min, x_max, y_min, y_max))

        return positions

    def _random_position(self) -> Position:
        """生成随机位置"""
        return Position(
            random.uniform(0, self.world.width),
            random.uniform(0, self.world.height)
        )

    def _random_position_in_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ) -> Position:
        """在指定区域内生成随机位置"""
        return Position(
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max)
        )

    def create_spawn_wave(
        self,
        configs: List[SpawnConfig],
        interval_steps: int,
        world: World
    ) -> None:
        """
        创建分波次生成计划
        在每interval_steps步时生成一波Agent
        """
        wave_index = [0]

        def spawn_callback(w: World):
            if w.current_step > 0 and w.current_step % interval_steps == 0:
                if wave_index[0] < len(configs):
                    config = configs[wave_index[0]]
                    self.spawn(config)
                    wave_index[0] += 1

        world.register_step_callback(spawn_callback)

    def get_spawn_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        type_counts: Dict[str, int] = {}
        for record in self.spawn_history:
            agent_type = record["type"]
            type_counts[agent_type] = type_counts.get(agent_type, 0) + record["count"]

        return {
            "total_agents_created": len(self.agents_created),
            "spawn_events": len(self.spawn_history),
            "type_distribution": type_counts
        }

    def clear_history(self) -> None:
        """清除生成历史"""
        self.spawn_history.clear()
        self.agents_created.clear()
