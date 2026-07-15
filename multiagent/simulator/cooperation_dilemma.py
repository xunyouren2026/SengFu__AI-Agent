"""
囚徒困境模拟 - 分析合作与背叛演化
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

from .world import Agent, Position, World


class Action(Enum):
    """行动选择"""
    COOPERATE = auto()
    DEFECT = auto()


class StrategyType(Enum):
    """策略类型"""
    ALWAYS_COOPERATE = auto()
    ALWAYS_DEFECT = auto()
    TIT_FOR_TAT = auto()
    TIT_FOR_TWO_TATS = auto()
    GRIM_TRIGGER = auto()
    PAVLOV = auto()
    RANDOM = auto()
    PROBABILISTIC = auto()


@dataclass
class PayoffMatrix:
    """收益矩阵"""
    R: float = 3.0  # 奖励
    S: float = 0.0  # 受骗
    T: float = 5.0  # 诱惑
    P: float = 1.0  # 惩罚

    def get_payoff(self, action1: Action, action2: Action) -> Tuple[float, float]:
        if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
            return (self.R, self.R)
        elif action1 == Action.COOPERATE and action2 == Action.DEFECT:
            return (self.S, self.T)
        elif action1 == Action.DEFECT and action2 == Action.COOPERATE:
            return (self.T, self.S)
        else:
            return (self.P, self.P)


@dataclass
class GameHistory:
    """游戏历史"""
    opponent_id: str
    my_actions: List[Action] = field(default_factory=list)
    opponent_actions: List[Action] = field(default_factory=list)
    my_payoffs: List[float] = field(default_factory=list)

    def last_opponent_action(self) -> Optional[Action]:
        return self.opponent_actions[-1] if self.opponent_actions else None

    def cooperation_rate(self) -> float:
        if not self.opponent_actions:
            return 0.5
        coop_count = sum(1 for a in self.opponent_actions if a == Action.COOPERATE)
        return coop_count / len(self.opponent_actions)


@dataclass
class AgentStrategy:
    """Agent策略"""
    agent_id: str
    strategy_type: StrategyType
    history: Dict[str, GameHistory] = field(default_factory=dict)
    total_score: float = 0.0
    games_played: int = 0
    cooperation_count: int = 0
    defection_count: int = 0
    cooperation_prob: float = 0.5
    forgiveness_threshold: int = 2

    def decide(self, opponent_id: str) -> Action:
        game_hist = self.history.get(opponent_id, GameHistory(opponent_id=opponent_id))

        if self.strategy_type == StrategyType.ALWAYS_COOPERATE:
            return Action.COOPERATE
        elif self.strategy_type == StrategyType.ALWAYS_DEFECT:
            return Action.DEFECT
        elif self.strategy_type == StrategyType.TIT_FOR_TAT:
            last = game_hist.last_opponent_action()
            return Action.COOPERATE if last is None or last == Action.COOPERATE else Action.DEFECT
        elif self.strategy_type == StrategyType.TIT_FOR_TWO_TATS:
            if len(game_hist.opponent_actions) >= 2:
                if (game_hist.opponent_actions[-1] == Action.DEFECT and
                    game_hist.opponent_actions[-2] == Action.DEFECT):
                    return Action.DEFECT
            return Action.COOPERATE
        elif self.strategy_type == StrategyType.GRIM_TRIGGER:
            if any(a == Action.DEFECT for a in game_hist.opponent_actions):
                return Action.DEFECT
            return Action.COOPERATE
        elif self.strategy_type == StrategyType.PAVLOV:
            if len(game_hist.my_actions) == 0:
                return Action.COOPERATE
            my_last = game_hist.my_actions[-1]
            opp_last = game_hist.opponent_actions[-1]
            if (my_last == Action.COOPERATE and opp_last == Action.COOPERATE) or \
               (my_last == Action.DEFECT and opp_last == Action.COOPERATE):
                return my_last
            else:
                return Action.COOPERATE if my_last == Action.DEFECT else Action.DEFECT
        elif self.strategy_type == StrategyType.RANDOM:
            return Action.COOPERATE if random.random() < 0.5 else Action.DEFECT
        elif self.strategy_type == StrategyType.PROBABILISTIC:
            return Action.COOPERATE if random.random() < self.cooperation_prob else Action.DEFECT
        return Action.COOPERATE

    def record_game(self, opponent_id: str, my_action: Action,
                    opponent_action: Action, my_payoff: float) -> None:
        if opponent_id not in self.history:
            self.history[opponent_id] = GameHistory(opponent_id=opponent_id)
        game_hist = self.history[opponent_id]
        game_hist.my_actions.append(my_action)
        game_hist.opponent_actions.append(opponent_action)
        game_hist.my_payoffs.append(my_payoff)
        self.total_score += my_payoff
        self.games_played += 1
        if my_action == Action.COOPERATE:
            self.cooperation_count += 1
        else:
            self.defection_count += 1

    def get_cooperation_rate(self) -> float:
        total = self.cooperation_count + self.defection_count
        return self.cooperation_count / total if total > 0 else 0.5


class PrisonersDilemmaSimulator:
    """囚徒困境模拟器"""

    def __init__(self, world: World, payoff_matrix: Optional[PayoffMatrix] = None):
        self.world = world
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.strategies: Dict[str, AgentStrategy] = {}
        self.round_robin_results: List[Dict[str, Any]] = []
        self.population_dynamics: List[Dict[str, Any]] = []

    def register_agent(self, agent_id: str, strategy_type: StrategyType,
                       cooperation_prob: float = 0.5) -> AgentStrategy:
        strategy = AgentStrategy(agent_id=agent_id, strategy_type=strategy_type,
                                 cooperation_prob=cooperation_prob)
        self.strategies[agent_id] = strategy
        return strategy

    def play_game(self, agent1_id: str, agent2_id: str, rounds: int = 1) -> Tuple[float, float]:
        strategy1 = self.strategies.get(agent1_id)
        strategy2 = self.strategies.get(agent2_id)
        if not strategy1 or not strategy2:
            return (0.0, 0.0)
        total_payoff1, total_payoff2 = 0.0, 0.0
        for _ in range(rounds):
            action1 = strategy1.decide(agent2_id)
            action2 = strategy2.decide(agent1_id)
            payoff1, payoff2 = self.payoff_matrix.get_payoff(action1, action2)
            strategy1.record_game(agent2_id, action1, action2, payoff1)
            strategy2.record_game(agent1_id, action2, action1, payoff2)
            total_payoff1 += payoff1
            total_payoff2 += payoff2
        return (total_payoff1, total_payoff2)

    def round_robin_tournament(self, rounds_per_game: int = 10) -> Dict[str, float]:
        """循环赛锦标赛"""
        agent_ids = list(self.strategies.keys())
        scores = {aid: 0.0 for aid in agent_ids}
        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i+1:]:
                payoff1, payoff2 = self.play_game(aid1, aid2, rounds_per_game)
                scores[aid1] += payoff1
                scores[aid2] += payoff2
        self.round_robin_results.append({"time": self.world.current_time, "scores": scores.copy()})
        return scores

    def evolutionary_step(self, selection_pressure: float = 0.1) -> Dict[str, Any]:
        """演化步骤 - 低分策略被高分策略取代"""
        if len(self.strategies) < 2:
            return {"message": "Not enough agents"}
        scores = {aid: s.total_score for aid, s in self.strategies.items()}
        if not scores:
            return {"message": "No scores available"}
        avg_score = sum(scores.values()) / len(scores)
        replacements = 0
        for aid, score in list(scores.items()):
            if score < avg_score * 0.8:
                best_aid = max(scores, key=scores.get)
                if best_aid != aid:
                    best_strategy = self.strategies[best_aid]
                    self.strategies[aid] = AgentStrategy(
                        agent_id=aid,
                        strategy_type=best_strategy.strategy_type,
                        cooperation_prob=best_strategy.cooperation_prob
                    )
                    replacements += 1
        for s in self.strategies.values():
            s.total_score = 0
            s.games_played = 0
        self.population_dynamics.append({
            "time": self.world.current_time,
            "replacements": replacements,
            "avg_score": avg_score
        })
        return {"replacements": replacements, "avg_score": avg_score}

    def get_strategy_distribution(self) -> Dict[str, int]:
        """获取策略分布"""
        distribution: Dict[str, int] = defaultdict(int)
        for s in self.strategies.values():
            distribution[s.strategy_type.name] += 1
        return dict(distribution)

    def get_cooperation_statistics(self) -> Dict[str, float]:
        """获取合作统计"""
        rates = [s.get_cooperation_rate() for s in self.strategies.values()]
        return {
            "avg_cooperation_rate": sum(rates) / len(rates) if rates else 0,
            "max_cooperation_rate": max(rates) if rates else 0,
            "min_cooperation_rate": min(rates) if rates else 0
        }
