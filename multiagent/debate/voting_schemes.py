"""
投票机制模块
实现多种投票机制：多数票、加权票、Borda计数等
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from uuid import uuid4
from collections import defaultdict
import statistics
import math

from .protocol import Argument, Stance


class VotingMethod(Enum):
    """投票方法枚举"""
    PLURALITY = "plurality"             # 简单多数票
    MAJORITY = "majority"               # 绝对多数票
    BORDA = "borda"                     # Borda计数
    INSTANT_RUNOFF = "instant_runoff"   # 瞬时决选
    CONDORCET = "condorcet"             # 孔多塞方法
    WEIGHTED = "weighted"               # 加权投票
    APPROVAL = "approval"               # 批准投票
    RANKED_CHOICE = "ranked_choice"     # 排序选择


class VoteType(Enum):
    """投票类型枚举"""
    SINGLE = "single"           # 单选
    MULTIPLE = "multiple"       # 多选
    RANKED = "ranked"           # 排序
    SCORE = "score"             # 评分
    APPROVAL = "approval"       # 批准


@dataclass
class Voter:
    """投票者"""
    voter_id: str
    weight: float = 1.0
    expertise: float = 0.5
    credibility: float = 0.5
    preferences: List[str] = field(default_factory=list)  # 偏好排序
    approved: Set[str] = field(default_factory=set)       # 批准的选项
    scores: Dict[str, float] = field(default_factory=dict) # 评分
    
    def effective_weight(self) -> float:
        """计算有效权重"""
        return self.weight * (0.5 + self.expertise * 0.3 + self.credibility * 0.2)


@dataclass
class Ballot:
    """选票"""
    ballot_id: str = field(default_factory=lambda: str(uuid4())[:8])
    voter_id: str = ""
    vote_type: VoteType = VoteType.SINGLE
    selections: List[str] = field(default_factory=list)  # 选择的选项
    rankings: Dict[str, int] = field(default_factory=dict)  # 排序 (选项 -> 排名)
    scores: Dict[str, float] = field(default_factory=dict)  # 评分
    timestamp: datetime = field(default_factory=datetime.now)
    weight: float = 1.0
    
    def is_valid(self, options: Set[str]) -> bool:
        """验证选票有效性"""
        if self.vote_type == VoteType.SINGLE:
            return len(self.selections) == 1 and self.selections[0] in options
        
        elif self.vote_type == VoteType.MULTIPLE:
            return all(s in options for s in self.selections)
        
        elif self.vote_type == VoteType.RANKED:
            return all(o in options for o in self.rankings.keys())
        
        elif self.vote_type == VoteType.SCORE:
            return all(o in options for o in self.scores.keys())
        
        elif self.vote_type == VoteType.APPROVAL:
            return all(s in options for s in self.selections)
        
        return False


@dataclass
class VotingResult:
    """投票结果"""
    result_id: str = field(default_factory=lambda: str(uuid4())[:8])
    method: VotingMethod = VotingMethod.PLURALITY
    winner: Optional[str] = None
    winners: List[str] = field(default_factory=list)  # 可能有多个获胜者
    scores: Dict[str, float] = field(default_factory=dict)  # 各选项得分
    rankings: List[Tuple[str, float]] = field(default_factory=list)  # 排名结果
    total_votes: int = 0
    total_weight: float = 0.0
    is_tie: bool = False
    tie_breaker: Optional[str] = None
    rounds: List[Dict[str, Any]] = field(default_factory=list)  # 多轮投票记录
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_rank(self, option: str) -> int:
        """获取选项排名"""
        for i, (opt, _) in enumerate(self.rankings):
            if opt == option:
                return i + 1
        return -1


class PluralityVoting:
    """
    简单多数票制
    得票最多的选项获胜
    """
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """
        计票
        
        Args:
            ballots: 选票列表
            options: 有效选项集合
            
        Returns:
            投票结果
        """
        vote_counts: Dict[str, float] = defaultdict(float)
        
        for ballot in ballots:
            if not ballot.is_valid(options):
                continue
            
            for selection in ballot.selections:
                vote_counts[selection] += ballot.weight
        
        # 排序
        sorted_results = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 判断是否有平局
        is_tie = False
        winner = None
        
        if len(sorted_results) >= 2:
            if sorted_results[0][1] == sorted_results[1][1]:
                is_tie = True
                winner = sorted_results[0][0]  # 默认选第一个
        
        if not is_tie and sorted_results:
            winner = sorted_results[0][0]
        
        return VotingResult(
            method=VotingMethod.PLURALITY,
            winner=winner,
            scores=dict(vote_counts),
            rankings=sorted_results,
            total_votes=len(ballots),
            total_weight=sum(b.weight for b in ballots),
            is_tie=is_tie,
        )


class MajorityVoting:
    """
    绝对多数票制
    需要超过半数票才能获胜
    """
    
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """计票"""
        vote_counts: Dict[str, float] = defaultdict(float)
        total_weight = 0.0
        
        for ballot in ballots:
            if not ballot.is_valid(options):
                continue
            
            for selection in ballot.selections:
                vote_counts[selection] += ballot.weight
                total_weight += ballot.weight
        
        sorted_results = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 检查是否达到阈值
        winner = None
        if sorted_results:
            top_vote = sorted_results[0][1]
            if top_vote > total_weight * self.threshold:
                winner = sorted_results[0][0]
        
        return VotingResult(
            method=VotingMethod.MAJORITY,
            winner=winner,
            scores=dict(vote_counts),
            rankings=sorted_results,
            total_votes=len(ballots),
            total_weight=total_weight,
            metadata={"threshold": self.threshold},
        )


class BordaCount:
    """
    Borda计数法
    排序投票，按排名分配分数
    """
    
    def __init__(
        self,
        scoring_rule: str = "standard"
    ) -> None:
        """
        初始化
        
        Args:
            scoring_rule: 计分规则
                - "standard": n-1, n-2, ..., 1, 0
                - "dowdall": 1, 1/2, 1/3, ..., 1/n
                - "custom": 自定义
        """
        self.scoring_rule = scoring_rule
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """Borda计数"""
        n = len(options)
        if n == 0:
            return VotingResult(method=VotingMethod.BORDA)
        
        borda_scores: Dict[str, float] = defaultdict(float)
        
        for ballot in ballots:
            if not ballot.is_valid(options):
                continue
            
            # 获取排序
            if ballot.rankings:
                # 将排序转换为分数
                for option, rank in ballot.rankings.items():
                    score = self._calculate_score(rank, n)
                    borda_scores[option] += score * ballot.weight
            
            elif ballot.scores:
                # 直接使用评分
                for option, score in ballot.scores.items():
                    borda_scores[option] += score * ballot.weight
        
        # 排序结果
        sorted_results = sorted(
            borda_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 判断平局
        is_tie = False
        winner = None
        
        if len(sorted_results) >= 2:
            if sorted_results[0][1] == sorted_results[1][1]:
                is_tie = True
        
        if sorted_results:
            winner = sorted_results[0][0]
        
        return VotingResult(
            method=VotingMethod.BORDA,
            winner=winner,
            scores=dict(borda_scores),
            rankings=sorted_results,
            total_votes=len(ballots),
            is_tie=is_tie,
            metadata={"scoring_rule": self.scoring_rule},
        )
    
    def _calculate_score(self, rank: int, n: int) -> float:
        """计算Borda分数"""
        if self.scoring_rule == "standard":
            # 标准Borda: n-rank
            return n - rank
        
        elif self.scoring_rule == "dowdall":
            # Dowdall规则: 1/rank
            return 1.0 / rank if rank > 0 else 0
        
        else:
            return n - rank


class InstantRunoffVoting:
    """
    瞬时决选法（IRV）
    多轮淘汰，直到有候选人获得多数票
    """
    
    def __init__(self, majority_threshold: float = 0.5) -> None:
        self.majority_threshold = majority_threshold
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """瞬时决选"""
        rounds: List[Dict[str, Any]] = []
        remaining_options = set(options)
        active_ballots = list(ballots)
        
        while len(remaining_options) > 1:
            # 计算当前轮次的票数
            round_votes: Dict[str, float] = defaultdict(float)
            
            for ballot in active_ballots:
                # 找到最高排名的有效选项
                top_choice = self._get_top_choice(
                    ballot, remaining_options
                )
                if top_choice:
                    round_votes[top_choice] += ballot.weight
            
            total_weight = sum(round_votes.values())
            
            # 记录本轮结果
            round_result = {
                "round": len(rounds) + 1,
                "votes": dict(round_votes),
                "remaining": list(remaining_options),
            }
            rounds.append(round_result)
            
            # 检查是否有人达到多数
            if round_votes:
                max_votes = max(round_votes.values())
                if max_votes > total_weight * self.majority_threshold:
                    winner = max(round_votes, key=round_votes.get)
                    return VotingResult(
                        method=VotingMethod.INSTANT_RUNOFF,
                        winner=winner,
                        scores=dict(round_votes),
                        total_votes=len(ballots),
                        rounds=rounds,
                    )
            
            # 淘汰得票最少的
            if round_votes:
                min_votes = min(round_votes.values())
                to_eliminate = [
                    opt for opt, votes in round_votes.items()
                    if votes == min_votes
                ]
                
                # 如果只剩一个选项或所有选项票数相同，结束
                if len(to_eliminate) == len(remaining_options):
                    break
                
                for opt in to_eliminate:
                    remaining_options.remove(opt)
            else:
                break
        
        # 返回最后剩余的选项
        winner = list(remaining_options)[0] if remaining_options else None
        
        return VotingResult(
            method=VotingMethod.INSTANT_RUNOFF,
            winner=winner,
            total_votes=len(ballots),
            rounds=rounds,
        )
    
    def _get_top_choice(
        self,
        ballot: Ballot,
        remaining: Set[str]
    ) -> Optional[str]:
        """获取选票中最高排名的有效选项"""
        if ballot.rankings:
            # 按排名排序
            sorted_choices = sorted(
                ballot.rankings.items(),
                key=lambda x: x[1]
            )
            for option, _ in sorted_choices:
                if option in remaining:
                    return option
        
        elif ballot.selections:
            for selection in ballot.selections:
                if selection in remaining:
                    return selection
        
        return None


class CondorcetMethod:
    """
    孔多塞方法
    两两比较，找出能击败所有其他候选人的选项
    """
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """孔多塞计票"""
        options_list = list(options)
        n = len(options_list)
        
        if n == 0:
            return VotingResult(method=VotingMethod.CONDORCET)
        
        # 构建两两比较矩阵
        # pairwise[i][j] = i击败j的票数
        pairwise: Dict[str, Dict[str, float]] = {
            opt1: {opt2: 0.0 for opt2 in options}
            for opt1 in options
        }
        
        for ballot in ballots:
            preferences = self._get_preferences(ballot, options)
            
            for i, opt1 in enumerate(options_list):
                for j, opt2 in enumerate(options_list):
                    if i != j:
                        # 如果opt1排名高于opt2，opt1获得一票
                        if preferences.get(opt1, n) < preferences.get(opt2, n):
                            pairwise[opt1][opt2] += ballot.weight
        
        # 找孔多塞赢家
        condorcet_winner = None
        for candidate in options:
            is_winner = True
            for opponent in options:
                if candidate != opponent:
                    if pairwise[candidate][opponent] <= pairwise[opponent][candidate]:
                        is_winner = False
                        break
            
            if is_winner:
                condorcet_winner = candidate
                break
        
        # 计算每个候选人的胜利次数
        wins: Dict[str, int] = {}
        for candidate in options:
            win_count = sum(
                1 for opponent in options
                if candidate != opponent and
                pairwise[candidate][opponent] > pairwise[opponent][candidate]
            )
            wins[candidate] = win_count
        
        sorted_results = sorted(
            wins.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return VotingResult(
            method=VotingMethod.CONDORCET,
            winner=condorcet_winner,
            scores=dict(wins),
            rankings=sorted_results,
            total_votes=len(ballots),
            metadata={"pairwise_matrix": {
                k: dict(v) for k, v in pairwise.items()
            }},
        )
    
    def _get_preferences(
        self,
        ballot: Ballot,
        options: Set[str]
    ) -> Dict[str, int]:
        """获取偏好排序"""
        if ballot.rankings:
            return dict(ballot.rankings)
        
        elif ballot.scores:
            # 分数转换为排名（分数高=排名靠前）
            sorted_by_score = sorted(
                ballot.scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return {
                option: rank + 1
                for rank, (option, _) in enumerate(sorted_by_score)
            }
        
        return {}


class WeightedVoting:
    """
    加权投票
    根据投票者的权重进行加权计票
    """
    
    def __init__(
        self,
        weight_calculator: Optional[Callable[[Voter], float]] = None
    ) -> None:
        """
        初始化
        
        Args:
            weight_calculator: 权重计算函数
        """
        self.weight_calculator = weight_calculator
    
    def count(
        self,
        ballots: List[Ballot],
        voters: Dict[str, Voter],
        options: Set[str]
    ) -> VotingResult:
        """加权计票"""
        vote_counts: Dict[str, float] = defaultdict(float)
        total_weight = 0.0
        
        for ballot in ballots:
            if not ballot.is_valid(options):
                continue
            
            voter = voters.get(ballot.voter_id)
            if not voter:
                continue
            
            # 计算有效权重
            if self.weight_calculator:
                effective_weight = self.weight_calculator(voter)
            else:
                effective_weight = voter.effective_weight()
            
            effective_weight *= ballot.weight
            
            for selection in ballot.selections:
                vote_counts[selection] += effective_weight
            
            total_weight += effective_weight
        
        sorted_results = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        winner = sorted_results[0][0] if sorted_results else None
        
        return VotingResult(
            method=VotingMethod.WEIGHTED,
            winner=winner,
            scores=dict(vote_counts),
            rankings=sorted_results,
            total_votes=len(ballots),
            total_weight=total_weight,
        )


class ApprovalVoting:
    """
    批准投票
    每个投票者可以批准多个选项
    """
    
    def __init__(self, max_approvals: Optional[int] = None) -> None:
        self.max_approvals = max_approvals
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """批准投票计票"""
        approval_counts: Dict[str, float] = defaultdict(float)
        
        for ballot in ballots:
            # 检查批准数量限制
            if self.max_approvals and len(ballot.selections) > self.max_approvals:
                continue
            
            if not ballot.is_valid(options):
                continue
            
            for selection in ballot.selections:
                approval_counts[selection] += ballot.weight
        
        sorted_results = sorted(
            approval_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 判断平局
        is_tie = False
        winners = []
        
        if sorted_results:
            max_approval = sorted_results[0][1]
            winners = [
                opt for opt, count in sorted_results
                if count == max_approval
            ]
            is_tie = len(winners) > 1
        
        return VotingResult(
            method=VotingMethod.APPROVAL,
            winner=winners[0] if winners else None,
            winners=winners,
            scores=dict(approval_counts),
            rankings=sorted_results,
            total_votes=len(ballots),
            is_tie=is_tie,
        )


class RankedChoiceVoting:
    """
    排序选择投票
    类似IRV但支持更复杂的排序
    """
    
    def count(
        self,
        ballots: List[Ballot],
        options: Set[str]
    ) -> VotingResult:
        """排序选择计票"""
        # 使用Borda计数作为基础
        borda = BordaCount()
        return borda.count(ballots, options)


class VotingScheme:
    """
    投票机制
    主类，提供统一的投票接口
    """
    
    def __init__(
        self,
        default_method: VotingMethod = VotingMethod.PLURALITY
    ) -> None:
        self.default_method = default_method
        
        # 初始化各种投票方法
        self.methods: Dict[VotingMethod, Any] = {
            VotingMethod.PLURALITY: PluralityVoting(),
            VotingMethod.MAJORITY: MajorityVoting(),
            VotingMethod.BORDA: BordaCount(),
            VotingMethod.INSTANT_RUNOFF: InstantRunoffVoting(),
            VotingMethod.CONDORCET: CondorcetMethod(),
            VotingMethod.APPROVAL: ApprovalVoting(),
            VotingMethod.RANKED_CHOICE: RankedChoiceVoting(),
        }
        
        self.voters: Dict[str, Voter] = {}
        self.ballots: List[Ballot] = []
        self.results: List[VotingResult] = []
    
    def register_voter(
        self,
        voter_id: str,
        weight: float = 1.0,
        expertise: float = 0.5,
        credibility: float = 0.5
    ) -> Voter:
        """注册投票者"""
        voter = Voter(
            voter_id=voter_id,
            weight=weight,
            expertise=expertise,
            credibility=credibility
        )
        self.voters[voter_id] = voter
        return voter
    
    def create_ballot(
        self,
        voter_id: str,
        selections: Optional[List[str]] = None,
        rankings: Optional[Dict[str, int]] = None,
        scores: Optional[Dict[str, float]] = None,
        vote_type: VoteType = VoteType.SINGLE
    ) -> Ballot:
        """创建选票"""
        voter = self.voters.get(voter_id)
        weight = voter.effective_weight() if voter else 1.0
        
        ballot = Ballot(
            voter_id=voter_id,
            vote_type=vote_type,
            selections=selections or [],
            rankings=rankings or {},
            scores=scores or {},
            weight=weight
        )
        
        self.ballots.append(ballot)
        return ballot
    
    def vote(
        self,
        options: Set[str],
        method: Optional[VotingMethod] = None
    ) -> VotingResult:
        """
        执行投票
        
        Args:
            options: 选项集合
            method: 投票方法（默认使用default_method）
            
        Returns:
            投票结果
        """
        voting_method = method or self.default_method
        voting_impl = self.methods.get(voting_method)
        
        if not voting_impl:
            raise ValueError(f"不支持的投票方法: {voting_method}")
        
        # 根据方法类型调用相应的计票函数
        if voting_method == VotingMethod.WEIGHTED:
            result = voting_impl.count(
                self.ballots, self.voters, options
            )
        else:
            result = voting_impl.count(self.ballots, options)
        
        self.results.append(result)
        return result
    
    def vote_on_arguments(
        self,
        arguments: List[Argument],
        method: Optional[VotingMethod] = None
    ) -> VotingResult:
        """
        对论点进行投票
        
        Args:
            arguments: 论点列表
            method: 投票方法
            
        Returns:
            投票结果
        """
        # 将论点ID作为选项
        options = {arg.argument_id for arg in arguments}
        return self.vote(options, method)
    
    def compare_methods(
        self,
        options: Set[str],
        methods: Optional[List[VotingMethod]] = None
    ) -> Dict[VotingMethod, VotingResult]:
        """
        比较不同投票方法的结果
        
        Args:
            options: 选项集合
            methods: 要比较的方法列表
            
        Returns:
            各方法的结果
        """
        if methods is None:
            methods = [
                VotingMethod.PLURALITY,
                VotingMethod.BORDA,
                VotingMethod.CONDORCET,
                VotingMethod.APPROVAL,
            ]
        
        results = {}
        for method in methods:
            try:
                results[method] = self.vote(options, method)
            except ValueError:
                continue
        
        return results
    
    def clear_ballots(self) -> None:
        """清空选票"""
        self.ballots.clear()
    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """获取投票统计"""
        return {
            "total_voters": len(self.voters),
            "total_ballots": len(self.ballots),
            "total_votes_conducted": len(self.results),
            "voter_weights": {
                vid: v.effective_weight()
                for vid, v in self.voters.items()
            },
        }
    
    def detect_strategic_voting(
        self,
        ballots: List[Ballot],
        true_preferences: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        检测策略性投票
        
        Args:
            ballots: 实际选票
            true_preferences: 真实偏好
            
        Returns:
            可能的策略性投票检测结果
        """
        strategic_cases = []
        
        for ballot in ballots:
            voter_id = ballot.voter_id
            true_pref = true_preferences.get(voter_id, [])
            
            if not true_pref:
                continue
            
            # 检查是否与真实偏好一致
            actual_first = ballot.selections[0] if ballot.selections else None
            true_first = true_pref[0] if true_pref else None
            
            if actual_first and true_first and actual_first != true_first:
                strategic_cases.append({
                    "voter_id": voter_id,
                    "true_first_choice": true_first,
                    "actual_first_choice": actual_first,
                    "potential_reason": "可能为了防止更不喜欢的选项获胜",
                })
        
        return strategic_cases


# 便捷函数
def create_ranked_ballot(
    voter_id: str,
    preferences: List[str]
) -> Ballot:
    """
    创建排序选票
    
    Args:
        voter_id: 投票者ID
        preferences: 偏好列表（从高到低）
        
    Returns:
        排序选票
    """
    rankings = {
        option: rank + 1
        for rank, option in enumerate(preferences)
    }
    
    return Ballot(
        voter_id=voter_id,
        vote_type=VoteType.RANKED,
        selections=preferences,
        rankings=rankings,
    )


def create_score_ballot(
    voter_id: str,
    scores: Dict[str, float]
) -> Ballot:
    """
    创建评分选票
    
    Args:
        voter_id: 投票者ID
        scores: 评分映射
        
    Returns:
        评分选票
    """
    return Ballot(
        voter_id=voter_id,
        vote_type=VoteType.SCORE,
        scores=scores,
        selections=list(scores.keys()),
    )


__all__ = [
    "VotingMethod",
    "VoteType",
    "Voter",
    "Ballot",
    "VotingResult",
    "PluralityVoting",
    "MajorityVoting",
    "BordaCount",
    "InstantRunoffVoting",
    "CondorcetMethod",
    "WeightedVoting",
    "ApprovalVoting",
    "RankedChoiceVoting",
    "VotingScheme",
    "create_ranked_ballot",
    "create_score_ballot",
]
