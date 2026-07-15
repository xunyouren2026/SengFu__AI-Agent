"""
多Agent学习模块

提供多种学习算法:
- 模仿学习 (Imitation Learning)
- DAgger在线学习
- 行为克隆 (Behavior Cloning)
- 多任务元学习 (MAML)
- 技能涌现分析
- 联邦知识学习
- 课程学习
- 经验回放
"""

from .imitation_learning import (
    Trajectory,
    ExpertDemonstration,
    ExpertPolicy,
    ImitationLearner,
    DAggerAlgorithm,
    InverseImitationLearning,
    TrajectoryAugmentation
)

from .dagger_loop import (
    StateActionPair,
    DAggerConfig,
    ExpertOracle,
    LearnerPolicy,
    DAggerLoop,
    DAggerWithReset,
    ActiveDAgger
)

from .multi_task_meta_agent import (
    TaskSpecification,
    AdaptationResult,
    MAMLConfig,
    MetaParameter,
    TaskEnvironment,
    CollaborativeTaskEnvironment,
    MAMLMetaLearner,
    MultiTaskMetaAgent,
    StyleAwareMetaAgent
)

from .skill_emergence import (
    SkillCategory,
    EmergenceStage,
    Skill,
    SkillObservation,
    EmergenceEvent,
    SkillGraph,
    SkillEmergenceDetector,
    SkillEmergenceAnalyzer
)

from .behaviour_cloning import (
    BehaviorCloning,
    PolicyNetwork,
    Demonstration
)

from .cross_agent_fine_tune import (
    CrossAgentFineTuner,
    TrainingSample,
    ModelAdapter,
    DomainAdaptation,
    FederatedDataAggregator,
    TransferLearningManager,
    DataPrivacyFilter
)

from .curriculum_for_agent import (
    Task,
    SkillAssessment,
    LearningObjective,
    AgentCapabilityProfile,
    CurriculumGenerator,
    CurriculumRecommender,
    SpacedRepetitionScheduler
)

from .experience_replay_shared import (
    Experience,
    ExperienceSummary,
    ExperienceBuffer,
    SharedExperiencePool,
    ExperienceTransfer
)

from .federated_knowledge import (
    SoftLabel,
    KnowledgePacket,
    LocalModel,
    FederatedKnowledgeDistiller,
    FederatedKnowledgeServer,
    ConsensusBasedDistillation,
    PersonalizedDistillation,
    SecureKnowledgeExchange
)

from .skill_graph import (
    SkillNode,
    LearningPath,
    SkillGraph,
    SkillPathPlanner
)

from .teacher_student import (
    KnowledgeState,
    TeachingSession,
    ExpertAgent,
    NoviceAgent,
    TeacherStudentFramework
)


__all__ = [
    # 模仿学习
    'Trajectory',
    'ExpertDemonstration',
    'ExpertPolicy',
    'ImitationLearner',
    'DAggerAlgorithm',
    'InverseImitationLearning',
    'TrajectoryAugmentation',
    
    # DAgger在线学习
    'StateActionPair',
    'DAggerConfig',
    'ExpertOracle',
    'LearnerPolicy',
    'DAggerLoop',
    'DAggerWithReset',
    'ActiveDAgger',
    
    # 多任务元学习
    'TaskSpecification',
    'AdaptationResult',
    'MAMLConfig',
    'MetaParameter',
    'TaskEnvironment',
    'CollaborativeTaskEnvironment',
    'MAMLMetaLearner',
    'MultiTaskMetaAgent',
    'StyleAwareMetaAgent',
    
    # 技能涌现
    'SkillCategory',
    'EmergenceStage',
    'Skill',
    'SkillObservation',
    'EmergenceEvent',
    'SkillGraph',
    'SkillEmergenceDetector',
    'SkillEmergenceAnalyzer',
    
    # 行为克隆
    'BehaviorCloning',
    'PolicyNetwork',
    'Demonstration',
    
    # 跨Agent微调
    'CrossAgentFineTuner',
    'TrainingSample',
    'ModelAdapter',
    'DomainAdaptation',
    'FederatedDataAggregator',
    'TransferLearningManager',
    'DataPrivacyFilter',
    
    # 课程学习
    'Task',
    'SkillAssessment',
    'LearningObjective',
    'AgentCapabilityProfile',
    'CurriculumGenerator',
    'CurriculumRecommender',
    'SpacedRepetitionScheduler',
    
    # 经验回放
    'Experience',
    'ExperienceSummary',
    'ExperienceBuffer',
    'SharedExperiencePool',
    'ExperienceTransfer',
    
    # 联邦学习
    'SoftLabel',
    'KnowledgePacket',
    'LocalModel',
    'FederatedKnowledgeDistiller',
    'FederatedKnowledgeServer',
    'ConsensusBasedDistillation',
    'PersonalizedDistillation',
    'SecureKnowledgeExchange',
    
    # 技能图
    'SkillNode',
    'LearningPath',
    'SkillGraph',
    'SkillPathPlanner',
    
    # 教师学生
    'KnowledgeState',
    'TeachingSession',
    'ExpertAgent',
    'NoviceAgent',
    'TeacherStudentFramework'
]
