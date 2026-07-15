"""
TestReputation - 智能体单元测试：声誉系统

模块路径: testing/unit/agent/test_reputation.py
"""

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

import pytest
import numpy as np

pytestmark = pytest.mark.unit


@dataclass
class ReputationEntry:
    from_agent: str
    to_agent: str
    score: float
    comment: str = ""
    timestamp: float = field(default_factory=time.time)


class MockReputationSystem:
    def __init__(self):
        self.reputations: Dict[str, List[ReputationEntry]] = {}
        self.trust_scores: Dict[str, float] = {}

    def give_rating(self, from_agent: str, to_agent: str, score: float, comment: str = ""):
        entry = ReputationEntry(from_agent=from_agent, to_agent=to_agent, score=score, comment=comment)
        if to_agent not in self.reputations:
            self.reputations[to_agent] = []
        self.reputations[to_agent].append(entry)
        self._update_trust_score(to_agent)

    def _update_trust_score(self, agent_id: str):
        entries = self.reputations.get(agent_id, [])
        if not entries:
            return
        self.trust_scores[agent_id] = float(np.mean([e.score for e in entries]))

    def get_trust_score(self, agent_id: str) -> float:
        return self.trust_scores.get(agent_id, 0.5)

    def get_reputation_history(self, agent_id: str) -> List[ReputationEntry]:
        return self.reputations.get(agent_id, [])

    def get_top_agents(self, n: int = 10) -> List[tuple]:
        sorted_agents = sorted(self.trust_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:n]

    def is_trusted(self, agent_id: str, threshold: float = 0.7) -> bool:
        return self.get_trust_score(agent_id) >= threshold

    def penalize(self, agent_id: str, penalty: float = 0.1):
        current = self.trust_scores.get(agent_id, 0.5)
        self.trust_scores[agent_id] = max(0.0, current - penalty)


class TestReputation:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.system = MockReputationSystem()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_give_rating(self):
        self.system.give_rating("a1", "a2", 0.9, "Great work")
        assert len(self.system.reputations["a2"]) == 1

    def test_trust_score(self):
        self.system.give_rating("a1", "a2", 0.8)
        self.system.give_rating("a3", "a2", 0.9)
        score = self.system.get_trust_score("a2")
        assert abs(score - 0.85) < 1e-5

    def test_default_trust_score(self):
        assert self.system.get_trust_score("unknown") == 0.5

    def test_reputation_history(self):
        for i in range(5):
            self.system.give_rating(f"rater_{i}", "target", 0.5 + i * 0.1)
        history = self.system.get_reputation_history("target")
        assert len(history) == 5

    def test_get_top_agents(self):
        for i in range(10):
            self.system.give_rating("rater", f"agent_{i}", 0.5 + i * 0.05)
        top = self.system.get_top_agents(n=3)
        assert len(top) == 3
        assert top[0][0] == "agent_9"

    def test_is_trusted(self):
        self.system.give_rating("a1", "a2", 0.9)
        assert self.system.is_trusted("a2", threshold=0.7)
        assert not self.system.is_trusted("a2", threshold=0.95)

    def test_penalize(self):
        self.system.give_rating("a1", "a2", 0.8)
        self.system.penalize("a2", 0.3)
        assert abs(self.system.get_trust_score("a2") - 0.5) < 1e-5

    def test_penalize_floor(self):
        self.system.give_rating("a1", "a2", 0.1)
        self.system.penalize("a2", 0.5)
        assert self.system.get_trust_score("a2") >= 0.0

    def test_multiple_ratings_same_agent(self):
        for _ in range(10):
            self.system.give_rating("a1", "a2", 0.8)
        history = self.system.get_reputation_history("a2")
        assert len(history) == 10

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_various_scores(self, score):
        self.system.give_rating("a1", "a2", score)
        assert abs(self.system.get_trust_score("a2") - score) < 1e-5
