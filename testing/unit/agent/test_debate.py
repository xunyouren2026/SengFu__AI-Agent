"""
TestDebate - 智能体单元测试：辩论

模块路径: testing/unit/agent/test_debate.py
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
class DebateArgument:
    speaker_id: str
    content: str
    round_num: int
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0


@dataclass
class DebateTopic:
    topic: str
    description: str
    pro_side: str
    con_side: str


class MockDebateArena:
    def __init__(self):
        self.arguments: List[DebateArgument] = []
        self.scores: Dict[str, float] = {}
        self.current_round = 0
        self.max_rounds = 5

    def submit_argument(self, speaker_id: str, content: str) -> DebateArgument:
        arg = DebateArgument(speaker_id=speaker_id, content=content, round_num=self.current_round)
        self.arguments.append(arg)
        return arg

    def score_argument(self, argument: DebateArgument, score: float):
        argument.score = score
        self.scores[argument.speaker_id] = self.scores.get(argument.speaker_id, 0) + score

    def next_round(self):
        self.current_round += 1

    def is_finished(self) -> bool:
        return self.current_round >= self.max_rounds

    def get_round_arguments(self, round_num: int) -> List[DebateArgument]:
        return [a for a in self.arguments if a.round_num == round_num]

    def get_winner(self) -> Optional[str]:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def get_debate_summary(self) -> Dict[str, Any]:
        return {
            "total_rounds": self.current_round,
            "total_arguments": len(self.arguments),
            "scores": dict(self.scores),
            "winner": self.get_winner()
        }


class TestDebate:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.temp_dir = tmp_path
        self.arena = MockDebateArena()
        self.test_data = []
        yield
        self.test_data.clear()

    def test_submit_argument(self):
        arg = self.arena.submit_argument("agent1", "AI is beneficial")
        assert arg.speaker_id == "agent1"
        assert arg.round_num == 0

    def test_score_argument(self):
        arg = self.arena.submit_argument("agent1", "Point 1")
        self.arena.score_argument(arg, 8.5)
        assert self.arena.scores["agent1"] == 8.5

    def test_multiple_scores(self):
        for i in range(3):
            arg = self.arena.submit_argument("agent1", f"Argument {i}")
            self.arena.score_argument(arg, 7.0)
        assert self.arena.scores["agent1"] == 21.0

    def test_next_round(self):
        self.arena.next_round()
        assert self.arena.current_round == 1
        arg = self.arena.submit_argument("agent1", "New point")
        assert arg.round_num == 1

    def test_is_finished(self):
        arena = MockDebateArena()
        arena.max_rounds = 3
        assert not arena.is_finished()
        arena.next_round()
        arena.next_round()
        arena.next_round()
        assert arena.is_finished()

    def test_get_round_arguments(self):
        self.arena.submit_argument("a1", "Round 0 arg")
        self.arena.next_round()
        self.arena.submit_argument("a1", "Round 1 arg")
        r0 = self.arena.get_round_arguments(0)
        r1 = self.arena.get_round_arguments(1)
        assert len(r0) == 1 and len(r1) == 1

    def test_get_winner(self):
        self.arena.submit_argument("a1", "arg")
        self.arena.score_argument(self.arena.arguments[0], 5.0)
        self.arena.submit_argument("a2", "arg")
        self.arena.score_argument(self.arena.arguments[1], 8.0)
        assert self.arena.get_winner() == "a2"

    def test_get_winner_no_scores(self):
        assert self.arena.get_winner() is None

    def test_debate_summary(self):
        self.arena.submit_argument("a1", "arg1")
        self.arena.score_argument(self.arena.arguments[0], 7.0)
        summary = self.arena.get_debate_summary()
        assert summary["total_arguments"] == 1
        assert summary["winner"] == "a1"

    def test_full_debate(self):
        for r in range(3):
            self.arena.submit_argument("pro", f"Pro argument round {r}")
            self.arena.score_argument(self.arena.arguments[-1], 6.0)
            self.arena.submit_argument("con", f"Con argument round {r}")
            self.arena.score_argument(self.arena.arguments[-1], 7.0)
            if r < 2:
                self.arena.next_round()
        assert self.arena.get_winner() == "con"
