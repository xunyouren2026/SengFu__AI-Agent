"""
共识质量测试模块

测试辩论协议、投票方案、谬误检测、质量评分、
多方辩论和共识达成检测。
"""

import unittest
import time
from typing import Dict, List, Set, Optional, Any, Tuple
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from multiagent.debate.protocol import (
    DebatePhase,
    ArgumentType,
    Stance,
    Evidence,
    Argument,
    Rebuttal,
    Revision,
    Verdict,
    DebateState,
    DebateProtocol,
    ProtocolValidator
)


class MockDebateHelpers:
    """辩论测试辅助类"""

    @staticmethod
    def create_evidence(
        evidence_id: str = "ev_001",
        source: str = "test_source",
        content: str = "Test evidence content",
        credibility: float = 0.8,
        relevance: float = 0.9
    ) -> Evidence:
        """创建测试用证据"""
        return Evidence(
            evidence_id=evidence_id,
            source=source,
            content=content,
            credibility=credibility,
            relevance=relevance
        )

    @staticmethod
    def create_argument(
        argument_id: str = "arg_001",
        speaker_id: str = "speaker_1",
        content: str = "This is a test argument",
        argument_type: ArgumentType = ArgumentType.FACTUAL,
        stance: Stance = Stance.PRO
    ) -> Argument:
        """创建测试用论点"""
        return Argument(
            argument_id=argument_id,
            speaker_id=speaker_id,
            content=content,
            argument_type=argument_type,
            stance=stance
        )

    @staticmethod
    def create_rebuttal(
        rebuttal_id: str = "reb_001",
        speaker_id: str = "speaker_2",
        target_argument_id: str = "arg_001",
        content: str = "Counter argument content"
    ) -> Rebuttal:
        """创建测试用反驳"""
        return Rebuttal(
            rebuttal_id=rebuttal_id,
            speaker_id=speaker_id,
            target_argument_id=target_argument_id,
            content=content,
            rebuttal_type="logical"
        )

    @staticmethod
    def create_debate_protocol(debate_id: str = "debate_001") -> DebateProtocol:
        """创建测试用辩论协议"""
        return DebateProtocol(debate_id=debate_id)


class TestDebateProtocol(unittest.TestCase):
    """测试辩论协议"""

    def test_protocol_initialization(self):
        """测试协议初始化"""
        protocol = DebateProtocol()
        self.assertIsNotNone(protocol)
        self.assertEqual(protocol.state.current_phase, DebatePhase.CLAIM)

    def test_set_topic(self):
        """测试设置辩论主题"""
        protocol = create_debate_protocol()
        protocol.set_topic("Should AI be regulated?")
        self.assertEqual(protocol.state.topic, "Should AI be regulated?")

    def test_valid_phase_transition(self):
        """测试有效的阶段转换"""
        protocol = DebateProtocol()

        # CLAIM -> REBUTTAL 应该是有效的
        result = protocol.can_transition(DebatePhase.CLAIM, DebatePhase.REBUTTAL)
        self.assertTrue(result)

    def test_invalid_phase_transition(self):
        """测试无效的阶段转换"""
        protocol = DebateProtocol()

        # CLAIM -> VERDICT 应该是无效的
        result = protocol.can_transition(DebatePhase.CLAIM, DebatePhase.VERDICT)
        self.assertFalse(result)

    def test_transition_success(self):
        """测试成功转换阶段"""
        protocol = DebateProtocol()

        result = protocol.transition(DebatePhase.REBUTTAL)
        self.assertTrue(result)
        self.assertEqual(protocol.state.current_phase, DebatePhase.REBUTTAL)

    def test_transition_failure(self):
        """测试转换阶段失败"""
        protocol = DebateProtocol()

        # 尝试无效转换
        result = protocol.transition(DebatePhase.VERDICT)
        self.assertFalse(result)
        self.assertEqual(protocol.state.current_phase, DebatePhase.CLAIM)

    def test_phase_history_tracking(self):
        """测试阶段历史跟踪"""
        protocol = DebateProtocol()

        protocol.transition(DebatePhase.REBUTTAL)
        protocol.transition(DebatePhase.COUNTER_CLAIM)

        self.assertEqual(len(protocol.state.phase_history), 2)

    def test_add_argument(self):
        """测试添加论点"""
        protocol = DebateProtocol()
        argument = MockDebateHelpers.create_argument()

        protocol.state.add_argument(argument)

        self.assertIn(argument.argument_id, protocol.state.arguments)
        self.assertIn(argument.speaker_id, protocol.state.participants)

    def test_add_rebuttal(self):
        """测试添加反驳"""
        protocol = DebateProtocol()
        rebuttal = MockDebateHelpers.create_rebuttal()

        protocol.state.add_rebuttal(rebuttal)

        self.assertIn(rebuttal.rebuttal_id, protocol.state.rebuttals)

    def test_get_arguments_by_stance(self):
        """测试按立场获取论点"""
        protocol = DebateProtocol()

        protocol.state.add_argument(
            MockDebateHelpers.create_argument("arg_1", stance=Stance.PRO)
        )
        protocol.state.add_argument(
            MockDebateHelpers.create_argument("arg_2", stance=Stance.CON)
        )
        protocol.state.add_argument(
            MockDebateHelpers.create_argument("arg_3", stance=Stance.PRO)
        )

        pro_arguments = protocol.state.get_arguments_by_stance(Stance.PRO)
        self.assertEqual(len(pro_arguments), 2)

    def test_get_rebuttals_for_argument(self):
        """测试获取针对特定论点的反驳"""
        protocol = DebateProtocol()

        protocol.state.add_rebuttal(
            MockDebateHelpers.create_rebuttal("reb_1", target_argument_id="arg_001")
        )
        protocol.state.add_rebuttal(
            MockDebateHelpers.create_rebuttal("reb_2", target_argument_id="arg_001")
        )
        protocol.state.add_rebuttal(
            MockDebateHelpers.create_rebuttal("reb_3", target_argument_id="arg_002")
        )

        rebuttals = protocol.state.get_rebuttals_for_argument("arg_001")
        self.assertEqual(len(rebuttals), 2)


class TestVotingSchemes(unittest.TestCase):
    """测试投票方案"""

    def setUp(self):
        """测试初始化"""
        from multiagent.debate.voting_schemes import (
            VotingScheme,
            Ballot,
            Vote,
            PluralityVoting,
            BordaCountVoting,
            ApprovalVoting,
            RangeVoting
        )
        self.voting_module = __import__('multiagent.debate.voting_schemes', fromlist=['VotingSchemes'])
        self.VotingScheme = self.voting_module.VotingScheme
        self.Ballot = self.voting_module.Ballot
        self.Vote = self.voting_module.Vote
        self.PluralityVoting = self.voting_module.PluralityVoting
        self.BordaCountVoting = self.voting_module.BordaCountVoting
        self.ApprovalVoting = self.voting_module.ApprovalVoting
        self.RangeVoting = self.voting_module.RangeVoting

    def test_plurality_voting(self):
        """测试多数投票制"""
        voting = self.PluralityVoting()

        # 模拟投票
        votes = [
            self.Vote(voter_id="v1", choice="A"),
            self.Vote(voter_id="v2", choice="A"),
            self.Vote(voter_id="v3", choice="B"),
        ]

        winner = voting.determine_winner(votes)
        self.assertEqual(winner, "A")

    def test_plurality_no_majority(self):
        """测试无绝对多数情况"""
        voting = self.PluralityVoting()

        votes = [
            self.Vote(voter_id="v1", choice="A"),
            self.Vote(voter_id="v2", choice="B"),
            self.Vote(voter_id="v3", choice="C"),
        ]

        winner = voting.determine_winner(votes)
        # 没有绝对多数，应该返回得票最多的
        self.assertIn(winner, ["A", "B", "C"])

    def test_borda_count_voting(self):
        """测试Borda计数投票制"""
        voting = self.BordaCountVoting(num_candidates=4)

        votes = [
            self.Vote(voter_id="v1", ranking=["A", "B", "C", "D"]),
            self.Vote(voter_id="v2", ranking=["B", "A", "C", "D"]),
            self.Vote(voter_id="v3", ranking=["A", "C", "B", "D"]),
        ]

        winner = voting.determine_winner(votes)
        self.assertIsNotNone(winner)

    def test_approval_voting(self):
        """测试赞成投票制"""
        voting = self.ApprovalVoting()

        votes = [
            self.Vote(voter_id="v1", approvals=["A", "B"]),
            self.Vote(voter_id="v2", approvals=["A"]),
            self.Vote(voter_id="v3", approvals=["A", "C"]),
        ]

        winner = voting.determine_winner(votes)
        # A应该获得最多赞成票
        self.assertEqual(winner, "A")

    def test_range_voting(self):
        """测试范围投票制"""
        voting = self.RangeVoting(min_score=0, max_score=5)

        votes = [
            self.Vote(voter_id="v1", scores={"A": 5, "B": 3}),
            self.Vote(voter_id="v2", scores={"A": 4, "B": 4}),
            self.Vote(voter_id="v3", scores={"A": 5, "B": 2}),
        ]

        winner = voting.determine_winner(votes)
        self.assertEqual(winner, "A")

    def test_voting_tie(self):
        """测试投票平局"""
        voting = self.PluralityVoting()

        votes = [
            self.Vote(voter_id="v1", choice="A"),
            self.Vote(voter_id="v2", choice="B"),
        ]

        winner = voting.determine_winner(votes)
        # 平局时应该有处理机制
        self.assertIn(winner, ["A", "B", None])

    def test_empty_votes(self):
        """测试空投票列表"""
        voting = self.PluralityVoting()

        winner = voting.determine_winner([])
        self.assertIsNone(winner)


class TestFallacyDetection(unittest.TestCase):
    """测试谬误检测"""

    def setUp(self):
        """测试初始化"""
        from multiagent.debate.fallacy_detector import FallacyDetector, FallacyType
        self.fallacy_module = __import__('multiagent.debate.fallacy_detector', fromlist=['FallacyDetector'])
        self.FallacyDetector = self.fallacy_module.FallacyDetector
        self.FallacyType = self.fallacy_module.FallacyType

    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = self.FallacyDetector()
        self.assertIsNotNone(detector)

    def test_detect_ad_hominem(self):
        """测试人身攻击谬误检测"""
        detector = self.FallacyDetector()

        text = "You can't trust John's argument because he's a liar."
        fallacies = detector.detect(text)

        self.assertTrue(len(fallacies) > 0)
        fallacy_types = [f.type for f in fallacies]
        self.assertIn(self.FallacyType.AD_HOMINEM, fallacy_types)

    def test_detect_straw_man(self):
        """测试稻草人谬误检测"""
        detector = self.FallacyDetector()

        text = "Person A suggests we need better healthcare. Person B says A wants to destroy the economy."
        fallacies = detector.detect(text)

        self.assertTrue(len(fallacies) > 0)

    def test_detect_false_dilemma(self):
        """测试虚假两难谬误检测"""
        detector = self.FallacyDetector()

        text = "You are either with us or against us."
        fallacies = detector.detect(text)

        self.assertTrue(len(fallacies) > 0)

    def test_detect_circular_reasoning(self):
        """测试循环论证谬误检测"""
        detector = self.FallacyDetector()

        text = "The Bible is true because it says so in the Bible."
        fallacies = detector.detect(text)

        self.assertTrue(len(fallacies) > 0)

    def test_no_fallacy_detected(self):
        """测试无谬误情况"""
        detector = self.FallacyDetector()

        text = "The experiment results show that the hypothesis is supported by the data."
        fallacies = detector.detect(text)

        self.assertEqual(len(fallacies), 0)

    def test_detect_multiple_fallacies(self):
        """测试检测多个谬误"""
        detector = self.FallacyDetector()

        text = "You should ignore Smith's proposal. He's from a failing department. Either we do it my way or we fail completely."
        fallacies = detector.detect(text)

        self.assertGreaterEqual(len(fallacies), 2)

    def test_fallacy_confidence_score(self):
        """测试谬误置信度"""
        detector = self.FallacyDetector()

        text = "This is obviously wrong because it comes from a known liar."
        fallacies = detector.detect(text)

        for fallacy in fallacies:
            self.assertGreaterEqual(fallacy.confidence, 0.0)
            self.assertLessEqual(fallacy.confidence, 1.0)


class TestQualityScoring(unittest.TestCase):
    """测试质量评分"""

    def setUp(self):
        """测试初始化"""
        from multiagent.debate.quality_scorer import QualityScorer, ArgumentQuality
        self.quality_module = __import__('multiagent.debate.quality_scorer', fromlist=['QualityScorer'])
        self.QualityScorer = self.quality_module.QualityScorer
        self.ArgumentQuality = self.quality_module.ArgumentQuality

    def test_scorer_initialization(self):
        """测试评分器初始化"""
        scorer = self.QualityScorer()
        self.assertIsNotNone(scorer)

    def test_score_argument_with_evidence(self):
        """测试有证据的论点评分"""
        scorer = self.QualityScorer()

        argument = MockDebateHelpers.create_argument(
            content="The study shows X causes Y",
            argument_type=ArgumentType.FACTUAL
        )
        argument.add_evidence(
            MockDebateHelpers.create_evidence(credibility=0.9, relevance=0.8)
        )

        score = scorer.score_argument(argument)
        self.assertGreater(score.overall_score, 0)

    def test_score_argument_without_evidence(self):
        """测试无证据的论点评分"""
        scorer = self.QualityScorer()

        argument = MockDebateHelpers.create_argument(
            content="I think X is true",
            argument_type=ArgumentType.NORMATIVE
        )

        score = scorer.score_argument(argument)
        self.assertGreaterEqual(score.overall_score, 0)

    def test_score_coherence(self):
        """测试连贯性评分"""
        scorer = self.QualityScorer()

        coherent_text = "First, we need to identify the problem. Second, we analyze the root cause. Finally, we implement a solution."
        incoherent_text = "The sky is blue. Dogs bark. Mathematics is hard."

        coherent_score = scorer.score_coherence(coherent_text)
        incoherent_score = scorer.score_coherence(incoherent_text)

        self.assertGreater(coherent_score, incoherent_score)

    def test_score_relevance(self):
        """测试相关性评分"""
        scorer = self.QualityScorer()

        argument = MockDebateHelpers.create_argument(
            content="We should reduce carbon emissions because..."
        )

        topic = "climate policy"
        relevance = scorer.score_relevance(argument, topic)

        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)

    def test_calculate_factuality_score(self):
        """测试事实性评分"""
        scorer = self.QualityScorer()

        factual_argument = MockDebateHelpers.create_argument(
            content="According to the 2020 census, the population is 331 million.",
            argument_type=ArgumentType.FACTUAL
        )
        factual_argument.add_evidence(
            Evidence(evidence_id="e1", source="census.gov", content="2020 census data", credibility=1.0, relevance=1.0)
        )

        score = scorer.score_argument(factual_argument)
        self.assertGreater(score.factuality, 0.5)


class TestMultiPartyDebate(unittest.TestCase):
    """测试多方辩论"""

    def setUp(self):
        """测试初始化"""
        from multiagent.debate.multi_party_debate import MultiPartyDebate, DebateParticipant
        self.debate_module = __import__('multiagent.debate.multi_party_debate', fromlist=['MultiPartyDebate'])
        self.MultiPartyDebate = self.debate_module.MultiPartyDebate
        self.DebateParticipant = self.debate_module.DebateParticipant

    def test_debate_initialization(self):
        """测试辩论初始化"""
        debate = self.MultiPartyDebate(topic="Test topic")
        self.assertIsNotNone(debate)
        self.assertEqual(debate.topic, "Test topic")

    def test_add_participant(self):
        """测试添加参与者"""
        debate = self.MultiPartyDebate(topic="Test topic")

        participant = self.DebateParticipant(
            participant_id="p1",
            name="Participant 1",
            stance=Stance.PRO
        )

        debate.add_participant(participant)
        self.assertEqual(len(debate.participants), 1)

    def test_remove_participant(self):
        """测试移除参与者"""
        debate = self.MultiPartyDebate(topic="Test topic")

        participant = self.DebateParticipant(
            participant_id="p1",
            name="Participant 1"
        )

        debate.add_participant(participant)
        debate.remove_participant("p1")

        self.assertEqual(len(debate.participants), 0)

    def test_participant_stances(self):
        """测试参与者立场"""
        debate = self.MultiPartyDebate(topic="Test topic")

        debate.add_participant(self.DebateParticipant("p1", "Pro 1", stance=Stance.PRO))
        debate.add_participant(self.DebateParticipant("p2", "Con 1", stance=Stance.CON))
        debate.add_participant(self.DebateParticipant("p3", "Neutral 1", stance=Stance.NEUTRAL))

        self.assertEqual(len(debate.get_participants_by_stance(Stance.PRO)), 1)
        self.assertEqual(len(debate.get_participants_by_stance(Stance.CON)), 1)
        self.assertEqual(len(debate.get_participants_by_stance(Stance.NEUTRAL)), 1)

    def test_record_argument(self):
        """测试记录论点"""
        debate = self.MultiPartyDebate(topic="Test topic")

        participant = self.DebateParticipant("p1", "Speaker 1")
        debate.add_participant(participant)

        argument = MockDebateHelpers.create_argument(
            speaker_id="p1",
            content="My argument"
        )

        debate.record_argument(argument)
        self.assertEqual(len(debate.arguments), 1)

    def test_debate_duration(self):
        """测试辩论持续时间"""
        debate = self.MultiPartyDebate(topic="Test topic")
        debate.start()

        time.sleep(0.1)

        duration = debate.get_duration()
        self.assertGreater(duration, 0)

    def test_max_arguments_limit(self):
        """测试最大论点限制"""
        debate = self.MultiPartyDebate(topic="Test topic", max_arguments=5)

        for i in range(10):
            argument = MockDebateHelpers.create_argument(
                argument_id=f"arg_{i}",
                content=f"Argument {i}"
            )
            debate.record_argument(argument)

        # 应该只保留最后5个论点
        self.assertLessEqual(len(debate.arguments), 5)


class TestConsensusReach(unittest.TestCase):
    """测试共识达成检测"""

    def setUp(self):
        """测试初始化"""
        from multiagent.debate.consensus_reach import ConsensusDetector, ConsensusState
        self.consensus_module = __import__('multiagent.debate.consensus_reach', fromlist=['ConsensusDetector'])
        self.ConsensusDetector = self.consensus_module.ConsensusDetector
        self.ConsensusState = self.consensus_module.ConsensusState

    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = self.ConsensusDetector()
        self.assertIsNotNone(detector)

    def test_detect_consensus_unanimous(self):
        """测试全票共识检测"""
        detector = self.ConsensusDetector()

        votes = {"A": 10, "B": 0}

        state = detector.detect_consensus(votes, threshold=1.0)
        self.assertEqual(state, self.ConsensusState.UNANIMOUS)

    def test_detect_consensus_supermajority(self):
        """测试超多数共识检测"""
        detector = self.ConsensusDetector(threshold=0.75)

        votes = {"A": 8, "B": 2}

        state = detector.detect_consensus(votes, threshold=0.75)
        self.assertEqual(state, self.ConsensusState.SUPERMAJORITY)

    def test_detect_no_consensus(self):
        """测试无共识检测"""
        detector = self.ConsensusDetector(threshold=0.75)

        votes = {"A": 5, "B": 5}

        state = detector.detect_consensus(votes, threshold=0.75)
        self.assertEqual(state, self.ConsensusState.NO_CONSENSUS)

    def test_detect_plurality(self):
        """测试相对多数情况"""
        detector = self.ConsensusDetector(threshold=0.75)

        votes = {"A": 4, "B": 3, "C": 3}

        state = detector.detect_consensus(votes, threshold=0.5)
        self.assertEqual(state, self.ConsensusState.PLURALITY)

    def test_convergence_tracking(self):
        """测试收敛跟踪"""
        detector = self.ConsensusDetector()

        rounds = [
            {"A": 3, "B": 7},
            {"A": 5, "B": 5},
            {"A": 7, "B": 3},
            {"A": 10, "B": 0},
        ]

        convergence = detector.track_convergence(rounds)
        self.assertTrue(convergence["is_converging"])
        self.assertGreater(convergence["convergence_score"], 0)

    def test_stability_check(self):
        """测试稳定性检查"""
        detector = self.ConsensusDetector()

        # 连续多轮相同结果
        states = [
            self.ConsensusState.UNANIMOUS,
            self.ConsensusState.UNANIMOUS,
            self.ConsensusState.UNANIMOUS,
        ]

        is_stable = detector.check_stability(states, required_rounds=3)
        self.assertTrue(is_stable)


class TestEvidenceHandling(unittest.TestCase):
    """测试证据处理"""

    def test_evidence_quality_score(self):
        """测试证据质量评分"""
        evidence = MockDebateHelpers.create_evidence(
            credibility=0.9,
            relevance=0.8
        )

        score = evidence.quality_score()
        expected = 0.9 * 0.6 + 0.8 * 0.4
        self.assertAlmostEqual(score, expected, places=2)

    def test_argument_evidence_strength(self):
        """测试论点证据强度"""
        argument = MockDebateHelpers.create_argument()

        argument.add_evidence(MockDebateHelpers.create_evidence(credibility=0.8, relevance=0.9))
        argument.add_evidence(MockDebateHelpers.create_evidence(credibility=0.6, relevance=0.7))

        strength = argument.get_evidence_strength()
        self.assertGreater(strength, 0)

    def test_evidence_without_source(self):
        """测试无来源证据"""
        evidence = Evidence(evidence_id="e1", content="Some claim")
        self.assertEqual(evidence.source, "")

    def test_evidence_timestamp(self):
        """测试证据时间戳"""
        before = time.time()
        evidence = MockDebateHelpers.create_evidence()
        after = time.time()

        self.assertGreaterEqual(evidence.timestamp.timestamp(), before)
        self.assertLessEqual(evidence.timestamp.timestamp(), after)


class TestArgumentCreation(unittest.TestCase):
    """测试论点创建"""

    def test_create_argument(self):
        """测试创建论点"""
        argument = MockDebateHelpers.create_argument(
            argument_id="custom_arg",
            content="Custom content"
        )

        self.assertEqual(argument.argument_id, "custom_arg")
        self.assertEqual(argument.content, "Custom content")

    def test_argument_defaults(self):
        """测试论点默认值"""
        argument = MockDebateHelpers.create_argument()

        self.assertEqual(argument.argument_type, ArgumentType.FACTUAL)
        self.assertEqual(argument.stance, Stance.PRO)
        self.assertEqual(len(argument.evidence_list), 0)

    def test_argument_with_target(self):
        """测试针对特定论点的论点"""
        argument = MockDebateHelpers.create_argument(
            target_argument_id="arg_001"
        )

        self.assertEqual(argument.target_argument_id, "arg_001")

    def test_argument_confidence(self):
        """测试论点置信度"""
        argument = MockDebateHelpers.create_argument()
        self.assertEqual(argument.confidence, 0.5)

        argument.confidence = 0.9
        self.assertEqual(argument.confidence, 0.9)


class TestVerdict(unittest.TestCase):
    """测试裁决"""

    def test_create_verdict(self):
        """测试创建裁决"""
        protocol = DebateProtocol()
        verdict = protocol.create_verdict(
            arbitrator_id="arb_001",
            winning_stance=Stance.PRO,
            reasoning="The PRO arguments were more convincing.",
            confidence=0.85
        )

        self.assertEqual(verdict.arbitrator_id, "arb_001")
        self.assertEqual(verdict.winning_stance, Stance.PRO)
        self.assertEqual(verdict.confidence, 0.85)

    def test_verdict_argument_scores(self):
        """测试裁决中的论点评分"""
        verdict = Verdict(
            verdict_id="v_001",
            arbitrator_id="arb_001",
            topic="Test",
            argument_scores={
                "arg_1": 0.9,
                "arg_2": 0.6,
                "arg_3": 0.3
            }
        )

        self.assertEqual(len(verdict.argument_scores), 3)
        self.assertEqual(verdict.argument_scores["arg_1"], 0.9)

    def test_verdict_with_recommendations(self):
        """测试带建议的裁决"""
        verdict = Verdict(
            verdict_id="v_001",
            arbitrator_id="arb_001",
            topic="Test",
            recommendations=[
                "Consider additional evidence",
                "Address counterarguments"
            ]
        )

        self.assertEqual(len(verdict.recommendations), 2)


class TestProtocolValidator(unittest.TestCase):
    """测试协议验证器"""

    def test_validate_valid_argument(self):
        """测试验证有效论点"""
        argument = MockDebateHelpers.create_argument(
            content="This is a valid argument with sufficient length."
        )
        argument.add_evidence(MockDebateHelpers.create_evidence())

        is_valid, errors = ProtocolValidator.validate_argument(
            argument,
            DebatePhase.CLAIM
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_short_argument(self):
        """测试验证过短论点"""
        argument = MockDebateHelpers.create_argument(
            content="Short"
        )

        is_valid, errors = ProtocolValidator.validate_argument(
            argument,
            DebatePhase.CLAIM
        )

        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)

    def test_validate_evidence_phase_requires_evidence(self):
        """测试证据阶段需要证据"""
        argument = MockDebateHelpers.create_argument(
            content="Argument without evidence for evidence phase."
        )

        is_valid, errors = ProtocolValidator.validate_argument(
            argument,
            DebatePhase.EVIDENCE
        )

        self.assertFalse(is_valid)
        self.assertIn("证据阶段需要提供证据", errors)

    def test_validate_rebuttal_requires_target(self):
        """测试反驳需要目标"""
        argument = MockDebateHelpers.create_argument(
            content="This is a rebuttal without target."
        )

        is_valid, errors = ProtocolValidator.validate_argument(
            argument,
            DebatePhase.REBUTTAL
        )

        self.assertFalse(is_valid)
        self.assertIn("反驳需要指定目标论点", errors)

    def test_validate_evidence_with_source(self):
        """测试验证有来源的证据"""
        evidence = MockDebateHelpers.create_evidence(
            source="Scientific Journal",
            content="Research findings"
        )

        is_valid, errors = ProtocolValidator.validate_evidence(evidence)
        self.assertTrue(is_valid)

    def test_validate_evidence_without_source(self):
        """测试验证无来源的证据"""
        evidence = Evidence(evidence_id="e1", content="Some claim")

        is_valid, errors = ProtocolValidator.validate_evidence(evidence)
        self.assertFalse(is_valid)

    def test_validate_evidence_invalid_credibility(self):
        """测试验证无效可信度"""
        evidence = Evidence(
            evidence_id="e1",
            source="Test",
            content="Test",
            credibility=1.5  # 无效应大于1
        )

        is_valid, errors = ProtocolValidator.validate_evidence(evidence)
        self.assertFalse(is_valid)


class TestDebateState(unittest.TestCase):
    """测试辩论状态"""

    def test_state_initialization(self):
        """测试状态初始化"""
        state = DebateState(topic="Test debate")
        self.assertEqual(state.topic, "Test debate")
        self.assertEqual(state.current_phase, DebatePhase.CLAIM)
        self.assertEqual(len(state.arguments), 0)

    def test_state_transition(self):
        """测试状态转换"""
        state = DebateState()
        state.transition_to(DebatePhase.REBUTTAL)

        self.assertEqual(state.current_phase, DebatePhase.REBUTTAL)
        self.assertEqual(len(state.phase_history), 1)

    def test_state_end_time(self):
        """测试结束时间"""
        state = DebateState()
        state.transition_to(DebatePhase.VERDICT)

        self.assertIsNotNone(state.end_time)

    def test_state_max_rounds(self):
        """测试最大轮次"""
        state = DebateState(max_rounds=3)
        self.assertEqual(state.max_rounds, 3)

    def test_state_consensus_threshold(self):
        """测试共识阈值"""
        state = DebateState(consensus_threshold=0.8)
        self.assertEqual(state.consensus_threshold, 0.8)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_empty_debate(self):
        """测试空辩论"""
        protocol = DebateProtocol()
        self.assertEqual(len(protocol.state.arguments), 0)
        self.assertEqual(len(protocol.state.participants), 0)

    def test_single_participant(self):
        """测试单一参与者"""
        debate = self.MultiPartyDebate(topic="Test")
        participant = self.DebateParticipant("p1", "Solo")
        debate.add_participant(participant)

        self.assertEqual(len(debate.participants), 1)

    def test_all_arguments_same_stance(self):
        """测试所有论点同一立场"""
        protocol = DebateProtocol()

        for i in range(5):
            argument = MockDebateHelpers.create_argument(
                argument_id=f"arg_{i}",
                stance=Stance.PRO
            )
            protocol.state.add_argument(argument)

        pro_args = protocol.state.get_arguments_by_stance(Stance.PRO)
        self.assertEqual(len(pro_args), 5)

    def test_tie_votes(self):
        """测试平局投票"""
        voting = self.PluralityVoting()

        votes = [
            self.Vote(voter_id="v1", choice="A"),
            self.Vote(voter_id="v2", choice="B"),
        ]

        winner = voting.determine_winner(votes)
        # 应该处理平局
        self.assertIn(winner, ["A", "B", None])

    def test_debate_timeout(self):
        """测试辩论超时"""
        detector = self.ConsensusDetector(timeout_seconds=1.0)

        time.sleep(1.1)

        self.assertTrue(detector.check_timeout())


if __name__ == "__main__":
    unittest.main()
