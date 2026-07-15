"""
领地竞争模拟 - 多Agent争夺有限资源
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from .world import Agent, Position, World


class TerritoryType(Enum):
    """领地类型"""
    RESOURCE_RICH = auto()
    STRATEGIC = auto()
    DEFENSIVE = auto()
    FERTILE = auto()
    INDUSTRIAL = auto()


class ConflictResult(Enum):
    """冲突结果"""
    ATTACKER_WINS = auto()
    DEFENDER_WINS = auto()
    DRAW = auto()
    AVOIDED = auto()


@dataclass
class Territory:
    """领地"""
    territory_id: str
    center: Position
    radius: float
    territory_type: TerritoryType
    owner_id: Optional[str] = None
    resource_value: float = 100.0
    defensibility: float = 1.0
    contested_by: Set[str] = field(default_factory=set)
    control_history: List[Tuple[float, Optional[str]]] = field(default_factory=list)

    @property
    def area(self) -> float:
        return 3.14159 * self.radius ** 2

    def contains(self, position: Position) -> bool:
        return self.center.distance_to(position) <= self.radius

    def set_owner(self, agent_id: Optional[str], timestamp: float) -> None:
        self.owner_id = agent_id
        self.control_history.append((timestamp, agent_id))


@dataclass
class Conflict:
    """冲突事件"""
    attacker_id: str
    defender_id: str
    territory: Territory
    timestamp: float
    attacker_strength: float = 0.0
    defender_strength: float = 0.0
    result: Optional[ConflictResult] = None
    casualties_attacker: float = 0.0
    casualties_defender: float = 0.0


@dataclass
class AgentTerritoryProfile:
    """Agent领地属性"""
    agent_id: str
    aggression: float = 0.5
    expansion_drive: float = 0.5
    defensive_focus: float = 0.5
    territory_value_weight: float = 1.0
    military_strength: float = 100.0
    controlled_territories: Set[str] = field(default_factory=set)
    border_conflicts: List[Conflict] = field(default_factory=list)


class TerritoryMap:
    """领地地图"""

    def __init__(self, world: World):
        self.world = world
        self.territories: Dict[str, Territory] = {}
        self.spatial_index: Dict[Tuple[int, int], Set[str]] = {}
        self.grid_size: float = 10.0

    def create_territory(self, territory_id: str, center: Position, radius: float,
                         territory_type: TerritoryType, resource_value: float = 100.0,
                         defensibility: float = 1.0) -> Territory:
        territory = Territory(territory_id=territory_id, center=center, radius=radius,
                              territory_type=territory_type, resource_value=resource_value,
                              defensibility=defensibility)
        self.territories[territory_id] = territory
        self._add_to_spatial_index(territory)
        return territory

    def _add_to_spatial_index(self, territory: Territory) -> None:
        min_x = int((territory.center.x - territory.radius) / self.grid_size)
        max_x = int((territory.center.x + territory.radius) / self.grid_size)
        min_y = int((territory.center.y - territory.radius) / self.grid_size)
        max_y = int((territory.center.y + territory.radius) / self.grid_size)
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                key = (x, y)
                if key not in self.spatial_index:
                    self.spatial_index[key] = set()
                self.spatial_index[key].add(territory.territory_id)

    def get_territories_at(self, position: Position) -> List[Territory]:
        grid_x = int(position.x / self.grid_size)
        grid_y = int(position.y / self.grid_size)
        candidates = self.spatial_index.get((grid_x, grid_y), set())
        return [self.territories[tid] for tid in candidates
                if self.territories[tid].contains(position)]

    def get_territories_owned_by(self, agent_id: str) -> List[Territory]:
        return [t for t in self.territories.values() if t.owner_id == agent_id]

    def get_border_territories(self, agent_id: str) -> List[Territory]:
        owned = set(t.territory_id for t in self.get_territories_owned_by(agent_id))
        borders = []
        for tid in owned:
            territory = self.territories[tid]
            for neighbor in self._get_neighboring_territories(territory):
                if neighbor.owner_id != agent_id:
                    borders.append(territory)
                    break
        return borders

    def _get_neighboring_territories(self, territory: Territory) -> List[Territory]:
        return [other for other in self.territories.values()
                if other.territory_id != territory.territory_id
                and territory.center.distance_to(other.center) <= territory.radius + other.radius]

    def calculate_territory_value(self, territory: Territory, agent_id: str) -> float:
        base_value = territory.resource_value
        owned = self.get_territories_owned_by(agent_id)
        proximity_bonus = 0
        if owned:
            avg_distance = sum(t.center.distance_to(territory.center) for t in owned) / len(owned)
            proximity_bonus = max(0, 100 - avg_distance)
        return base_value + proximity_bonus - territory.defensibility * 10


class ConflictResolver:
    """冲突解决器"""

    def __init__(self, territory_map: TerritoryMap):
        self.territory_map = territory_map
        self.conflict_history: List[Conflict] = []
        self.casualty_rate: float = 0.1

    def resolve_conflict(self, attacker_id: str, defender_id: str, territory: Territory,
                         profiles: Dict[str, AgentTerritoryProfile]) -> Conflict:
        attacker = profiles.get(attacker_id)
        defender = profiles.get(defender_id)
        if not attacker or not defender:
            raise ValueError("Agent profiles not found")

        conflict = Conflict(attacker_id=attacker_id, defender_id=defender_id,
                           territory=territory, timestamp=self.territory_map.world.current_time)

        base_attack = attacker.military_strength * (0.5 + attacker.aggression * 0.5)
        base_defense = defender.military_strength * (0.5 + defender.defensive_focus * 0.5)
        defense_bonus = territory.defensibility * 20
        effective_defense = base_defense + defense_bonus

        attacker_roll = random.gauss(0, base_attack * 0.1)
        defender_roll = random.gauss(0, effective_defense * 0.1)

        conflict.attacker_strength = base_attack + attacker_roll
        conflict.defender_strength = effective_defense + defender_roll

        if conflict.attacker_strength > conflict.defender_strength * 1.1:
            conflict.result = ConflictResult.ATTACKER_WINS
            conflict.casualties_attacker = base_attack * self.casualty_rate * random.uniform(0.5, 1.5)
            conflict.casualties_defender = effective_defense * self.casualty_rate * random.uniform(1.0, 2.0)
            territory.set_owner(attacker_id, self.territory_map.world.current_time)
            attacker.controlled_territories.add(territory.territory_id)
            defender.controlled_territories.discard(territory.territory_id)
        elif conflict.defender_strength > conflict.attacker_strength * 1.1:
            conflict.result = ConflictResult.DEFENDER_WINS
            conflict.casualties_attacker = base_attack * self.casualty_rate * random.uniform(1.0, 2.0)
            conflict.casualties_defender = effective_defense * self.casualty_rate * random.uniform(0.5, 1.0)
        else:
            conflict.result = ConflictResult.DRAW
            conflict.casualties_attacker = base_attack * self.casualty_rate
            conflict.casualties_defender = effective_defense * self.casualty_rate

        attacker.military_strength -= conflict.casualties_attacker
        defender.military_strength -= conflict.casualties_defender

        self.conflict_history.append(conflict)
        attacker.border_conflicts.append(conflict)
        defender.border_conflicts.append(conflict)

        return conflict


class TerritoryCompetitionSimulator:
    """领地竞争模拟器"""

    def __init__(self, world: World):
        self.world = world
        self.territory_map = TerritoryMap(world)
        self.conflict_resolver = ConflictResolver(self.territory_map)
        self.agent_profiles: Dict[str, AgentTerritoryProfile] = {}
        self.alliances: Dict[str, Set[str]] = defaultdict(set)

    def register_agent(self, agent_id: str, aggression: float = 0.5,
                       expansion_drive: float = 0.5, defensive_focus: float = 0.5,
                       military_strength: float = 100.0) -> AgentTerritoryProfile:
        profile = AgentTerritoryProfile(agent_id=agent_id, aggression=aggression,
                                        expansion_drive=expansion_drive,
                                        defensive_focus=defensive_focus,
                                        military_strength=military_strength)
        self.agent_profiles[agent_id] = profile
        return profile

    def create_territory_grid(self, rows: int = 5, cols: int = 5,
                              territory_radius: float = 8.0) -> List[Territory]:
        territories = []
        spacing_x = self.world.width / (cols + 1)
        spacing_y = self.world.height / (rows + 1)
        types = list(TerritoryType)
        for row in range(rows):
            for col in range(cols):
                tid = f"territory_{row}_{col}"
                center = Position(x=spacing_x * (col + 1), y=spacing_y * (row + 1))
                t_type = types[(row + col) % len(types)]
                resource = random.uniform(50, 150)
                defense = random.uniform(0.5, 2.0)
                territory = self.territory_map.create_territory(tid, center, territory_radius,
                                                                t_type, resource, defense)
                territories.append(territory)
        return territories

    def simulate_step(self) -> Dict[str, Any]:
        conflicts = []
        for agent_id, profile in self.agent_profiles.items():
            if profile.military_strength <= 0:
                continue
            if random.random() > profile.expansion_drive * 0.3:
                continue
            target = self._select_target(agent_id)
            if target and target.owner_id and target.owner_id != agent_id:
                if target.owner_id not in self.alliances.get(agent_id, set()):
                    conflict = self.conflict_resolver.resolve_conflict(agent_id, target.owner_id,
                                                                       target, self.agent_profiles)
                    conflicts.append(conflict)
        return {"conflicts": conflicts, "total_conflicts": len(self.conflict_resolver.conflict_history)}

    def _select_target(self, agent_id: str) -> Optional[Territory]:
        borders = self.territory_map.get_border_territories(agent_id)
        if not borders:
            unowned = [t for t in self.territory_map.territories.values() if t.owner_id is None]
            return random.choice(unowned) if unowned else None
        best = max(borders, key=lambda t: self.territory_map.calculate_territory_value(t, agent_id))
        neighbors = self.territory_map._get_neighboring_territories(best)
        enemy_neighbors = [n for n in neighbors if n.owner_id and n.owner_id != agent_id]
        return random.choice(enemy_neighbors) if enemy_neighbors else None

    def form_alliance(self, agent1: str, agent2: str) -> None:
        self.alliances[agent1].add(agent2)
        self.alliances[agent2].add(agent1)

    def get_territory_statistics(self) -> Dict[str, Any]:
        stats = {"total_territories": len(self.territory_map.territories),
                 "ownership_distribution": defaultdict(int),
                 "total_conflicts": len(self.conflict_resolver.conflict_history)}
        for t in self.territory_map.territories.values():
            owner = t.owner_id or "unclaimed"
            stats["ownership_distribution"][owner] += 1
        return stats
