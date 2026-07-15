"""
辩论模块
提供多Agent辩论框架，支持论证、反驳、裁决等完整辩论流程
"""

from .protocol import (
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
    ProtocolValidator,
)

from .participant import (
    ParticipantType,
    StrategyType,
    ParticipantProfile,
    ArgumentGenerator,
    RebuttalGenerator,
    DebateParticipant,
    create_participant,
)

from .arbitrator import (
    ArbitrationMethod,
    ArgumentEvaluation,
    ArbitrationConfig,
    ArgumentEvaluator,
    Arbitrator,
    ArbitrationPanel,
)

from .quality_scorer import (
    QualityDimension,
    DimensionScore,
    QualityAssessment,
    NoveltyScorer,
    LogicalConsistencyScorer,
    EvidenceSufficiencyScorer,
    ClarityScorer,
    RelevanceScorer,
    QualityScorer,
)

from .fallacy_detector import (
    FallacyType,
    FallacyDetection,
    ArgumentAnalysis,
    FallacyPattern,
    FallacyPatternDatabase,
    FallacyDetector,
    FallacyPreventionAdvisor,
)

from .evidence_retriever import (
    EvidenceSourceType,
    EvidenceReliability,
    KnowledgeEntry,
    RetrievalQuery,
    RetrievalResult,
    KnowledgeBase,
    RelevanceScorer as EvidenceRelevanceScorer,
    EvidenceRetriever,
    create_default_knowledge_base,
)

from .rules_of_order import (
    SpeakingRight,
    RuleViolationType,
    SpeakingSlot,
    SpeakingRecord,
    RulesConfig,
    RulesOfOrder,
    Moderator,
)

from .multi_party_debate import (
    DebateFormat,
    TeamAlignment,
    Team,
    DebateConfiguration,
    MultiPartyDebate,
)

from .consensus_reach import (
    ConsensusMetric,
    ConsensusState,
    ConsensusConfig,
    OpinionAnalyzer,
    ConsensusDetector,
    ConsensusBuilder,
)

from .radicalization_mitigation import (
    PolarizationLevel,
    MitigationStrategy,
    PolarizationMetrics,
    ModerationAction,
    StanceAnalyzer,
    ArgumentExtremityDetector,
    EchoChamberDetector,
    EmotionalIntensityAnalyzer,
    ModerationInterventionGenerator,
    RadicalizationMitigator,
)

from .transcript_logger import (
    LogFormat,
    LogLevel,
    LogEntry,
    DebateSummary,
    ArgumentRecord,
    EventLogger,
    DebateTranscript,
    TranscriptExporter,
    TranscriptLogger,
)

from .learning_from_debate import (
    KnowledgeType,
    ExtractionMethod,
    KnowledgePattern,
    ExtractedKnowledge,
    LearningInsight,
    WinnerAnalyzer,
    PatternMiner,
    ContrastiveAnalyzer,
    KnowledgeSynthesizer,
    LearningFromDebate,
)

from .voting_schemes import (
    VotingMethod,
    VoteType,
    Voter,
    Ballot,
    VotingResult,
    PluralityVoting,
    MajorityVoting,
    BordaCount,
    InstantRunoffVoting,
    CondorcetMethod,
    WeightedVoting,
    ApprovalVoting,
    RankedChoiceVoting,
    VotingScheme,
    create_ranked_ballot,
    create_score_ballot,
)


__all__ = [
    # Protocol
    "DebatePhase",
    "ArgumentType",
    "Stance",
    "Evidence",
    "Argument",
    "Rebuttal",
    "Revision",
    "Verdict",
    "DebateState",
    "DebateProtocol",
    "ProtocolValidator",
    
    # Participant
    "ParticipantType",
    "StrategyType",
    "ParticipantProfile",
    "ArgumentGenerator",
    "RebuttalGenerator",
    "DebateParticipant",
    "create_participant",
    
    # Arbitrator
    "ArbitrationMethod",
    "ArgumentEvaluation",
    "ArbitrationConfig",
    "ArgumentEvaluator",
    "Arbitrator",
    "ArbitrationPanel",
    
    # Quality Scorer
    "QualityDimension",
    "DimensionScore",
    "QualityAssessment",
    "NoveltyScorer",
    "LogicalConsistencyScorer",
    "EvidenceSufficiencyScorer",
    "ClarityScorer",
    "RelevanceScorer",
    "QualityScorer",
    
    # Fallacy Detector
    "FallacyType",
    "FallacyDetection",
    "ArgumentAnalysis",
    "FallacyPattern",
    "FallacyPatternDatabase",
    "FallacyDetector",
    "FallacyPreventionAdvisor",
    
    # Evidence Retriever
    "EvidenceSourceType",
    "EvidenceReliability",
    "KnowledgeEntry",
    "RetrievalQuery",
    "RetrievalResult",
    "KnowledgeBase",
    "EvidenceRelevanceScorer",
    "EvidenceRetriever",
    "create_default_knowledge_base",
    
    # Rules of Order
    "SpeakingRight",
    "RuleViolationType",
    "SpeakingSlot",
    "SpeakingRecord",
    "RulesConfig",
    "RulesOfOrder",
    "Moderator",
    
    # Multi-Party Debate
    "DebateFormat",
    "TeamAlignment",
    "Team",
    "DebateConfiguration",
    "MultiPartyDebate",
    
    # Consensus Reach
    "ConsensusMetric",
    "ConsensusState",
    "ConsensusConfig",
    "OpinionAnalyzer",
    "ConsensusDetector",
    "ConsensusBuilder",
    
    # Radicalization Mitigation
    "PolarizationLevel",
    "MitigationStrategy",
    "PolarizationMetrics",
    "ModerationAction",
    "StanceAnalyzer",
    "ArgumentExtremityDetector",
    "EchoChamberDetector",
    "EmotionalIntensityAnalyzer",
    "ModerationInterventionGenerator",
    "RadicalizationMitigator",
    
    # Transcript Logger
    "LogFormat",
    "LogLevel",
    "LogEntry",
    "DebateSummary",
    "ArgumentRecord",
    "EventLogger",
    "DebateTranscript",
    "TranscriptExporter",
    "TranscriptLogger",
    
    # Learning from Debate
    "KnowledgeType",
    "ExtractionMethod",
    "KnowledgePattern",
    "ExtractedKnowledge",
    "LearningInsight",
    "WinnerAnalyzer",
    "PatternMiner",
    "ContrastiveAnalyzer",
    "KnowledgeSynthesizer",
    "LearningFromDebate",
    
    # Voting Schemes
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
