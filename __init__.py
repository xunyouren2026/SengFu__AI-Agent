"""
AGI Unified Framework - 生产级企业智能框架

一个完整的AGI（通用人工智能）框架，支持多模型、多渠道、技能系统和插件扩展。

主要模块:
    - llm: 大语言模型集成层，支持18+主流模型提供商
    - channel: 多渠道适配器，支持19+通讯和存储平台
    - skill: 技能系统，支持声明式技能定义和执行
    - plugin: 插件系统，支持安全隔离的插件扩展
    - security: 安全层，包含双授权、权限边界、能力最小化等
    - web: 前端管理界面

版本: 2.0.0
作者: AGI Framework Team
许可证: MIT
"""

__version__ = "2.0.0"
__author__ = "AGI Framework Team"
__license__ = "MIT"

# 核心框架导出
from agi_unified_framework.core import (
    StateBus,
    Gate,
    Expert,
    MoE,
    MPCPlanner,
    GoalLayer,
    Reflexion,
    HaltDetector,
    IntrinsicMotivation,
    DifficultyScheduler,
    Confidence,
    Uncertainty,
    WarmMemory,
    ColdMemory,
    ReleaseTrigger,
    SwingEngine,
    BalanceRegulator,
    OnlineDistill,
)

# LLM层导出
from agi_unified_framework.llm import (
    LLMOrchestrator,
    BaseProvider,
    ModelRegistry,
)

# 所有支持的模型提供商
from agi_unified_framework.llm.providers import (
    # OpenAI系列
    OpenAIProvider,
    AnthropicProvider,
    # 国内主流
    ZhipuAIProvider,
    WenxinProvider,
    DashScopeProvider,
    BaichuanProvider,
    MoonshotProvider,
    YiProvider,
    SparkProvider,
    DeepSeekProvider,
    # 新增7个模型
    GLMFlashProvider,
    DoubaoProvider,
    MiniMaxProvider,
    SkyworkProvider,
    WPSLingxiProvider,
    MimoProvider,
    WuDaoProvider,
    # 本地模型
    LocalProvider,
)

# Channel层导出
from agi_unified_framework.channel import (
    ChannelGateway,
    BaseChannelAdapter,
    UniversalMessage,
    SessionManager,
    RateLimiter,
    HealthChecker,
    ChannelMetrics,
    WebhookHandler,
    ChannelRouter,
)

# 所有支持的Channel适配器
from agi_unified_framework.channel.adapters import (
    # 即时通讯
    DingTalkAdapter,
    WeChatWorkAdapter,
    FeishuAdapter,
    SlackAdapter,
    TelegramAdapter,
    DiscordAdapter,
    # 腾讯生态
    TencentDocsAdapter,
    TuyaIoTAdapter,
    # 邮件
    EmailAdapter,
    # 新增10个适配器
    AliyunOSSAdapter,
    TencentCOSAdapter,
    TencentVectorDBAdapter,
    DingTalkYidaAdapter,
    TencentWeDaAdapter,
    BaiduPanAdapter,
    TencentExmailAdapter,
    MingdaoAdapter,
    QingflowAdapter,
    TiDBCloudAdapter,
)

# 技能系统导出
from agi_unified_framework.skill import (
    SkillParser,
    SkillExecutor,
    SkillMarket,
    SkillMetadata,
    SkillRuntime,
)

# 插件系统导出
from agi_unified_framework.plugin import (
    PluginSDK,
    PluginManager,
    PluginSandbox,
    PluginSecurity,
    PluginLifecycle,
    PluginManifest,
    PluginContext,
)

# 安全层导出
from agi_unified_framework.security import (
    DoubleAuthorization,
    PermissionBoundary,
    CapabilityMinimizer,
    KeyVault,
    ActionGuard,
    ComplianceAudit,
)

# RAG系统导出
from agi_unified_framework.rag import (
    RAGEmbedder,
    RAGRetriever,
    RAGPipeline,
    VectorStore,
)

# 工作流引擎导出
from agi_unified_framework.workflow import (
    WorkflowEngine,
    WorkflowExecutor,
    DSLParser,
    StateManager,
    GraphEngine,
)

# 多智能体系统导出
from agi_unified_framework.multiagent import (
    AllianceGenetic,
)

# 联邦学习导出
from agi_unified_framework.federated import (
    FederatedServer,
    MPC,
    EvolutionCoordinator,
)

# 沙箱系统导出
from agi_unified_framework.sandbox import (
    SandboxInterface,
    DockerExecutor,
    NsjailExecutor,
    VirtualenvExecutor,
)

# AISec安全模块导出
from agi_unified_framework.aisec import (
    ZeroTrustExecutor,
    CommandInterceptor,
)

# 数据管道导出
from agi_unified_framework.data_pipeline import (
    DataDownloader,
    DataPreprocessor,
    DataCache,
    DataLoader,
)

# 存储层导出
from agi_unified_framework.storage import (
    FileRepository,
    MemoryRepository,
    RedisRepository,
    SQLAlchemyRepository,
    ConnectionPool,
    ShardingManager,
    BackupManager,
)

# 数据库导出
from agi_unified_framework.database import (
    DatabaseConnection,
    CRUDBase,
    BaseModel,
)

# 遥测和监控导出
from agi_unified_framework.telemetry import (
    TelemetryTracer,
    CostAnalyzer,
)

# 评估系统导出
from agi_unified_framework.evaluation import (
    BenchmarkRunner,
    MetricsCollector,
)

# 边缘计算导出
from agi_unified_framework.edge import (
    ONNXOptimizer,
    TFLiteConverter,
    JetsonOptimizer,
)

# IoT导出
from agi_unified_framework.iot import (
    BLEScanner,
    ModbusRTUClient,
    OPCUAClient,
)

# 机器人控制导出
from agi_unified_framework.robot import (
    RobotController,
    DigitalTwin,
    EmergencyStop,
    EnergyManager,
    FaultDiagnosis,
    SafetyFence,
    Teleoperation,
)

# 视频生成导出
from agi_unified_framework.video_gen import (
    VideoGenAPI,
    VideoInferencer,
    ScriptGenerator,
)

# 多模态导出
from agi_unified_framework.multimodal import (
    MultimodalCore,
    MultimodalProcessor,
)

# 本地AI导出
from agi_unified_framework.local import (
    LocalAI,
)

# 持续学习导出
from agi_unified_framework.continual import (
    ReplayBuffer,
    SDFD,
)

# 群智系统导出
from agi_unified_framework.swarm import (
    AgentRegistry,
    Alliance,
    Communication,
    Consensus,
    Coordination,
    Debate,
    Incentive,
    KnowledgeTransfer,
    Reputation,
    TaskAllocation,
)

# 计算机使用导出
from agi_unified_framework.computer_use import (
    AgentBrain,
    ScreenEngine,
    WindowEngine,
    ElementEngine,
    InputEngine,
    ClipboardEngine,
    VisionEngine,
    OCREngine,
    FileOps,
    WorkflowRecorder,
)

# API服务导出
try:
    from agi_unified_framework.api import (
        APIService,
        create_app,
    )
except ImportError:
    # API依赖可能未安装
    pass

# CLI导出
try:
    from agi_unified_framework.cli import (
        main as cli_main,
    )
except ImportError:
    # CLI依赖可能未安装
    pass


# 便捷函数
def get_version() -> str:
    """获取框架版本"""
    return __version__


def list_supported_providers() -> list:
    """列出所有支持的LLM提供商"""
    from agi_unified_framework.llm.providers import providers
    return list(providers.keys())


def list_supported_channels() -> list:
    """列出所有支持的Channel适配器"""
    from agi_unified_framework.channel.adapters import _ADAPTER_FACTORIES
    return list(_ADAPTER_FACTORIES.keys())


def create_orchestrator(config: dict = None):
    """
    创建LLM编排器实例
    
    Args:
        config: 配置字典
        
    Returns:
        LLMOrchestrator实例
    """
    return LLMOrchestrator(config)


def create_gateway(config: dict = None):
    """
    创建Channel网关实例
    
    Args:
        config: 配置字典
        
    Returns:
        ChannelGateway实例
    """
    return ChannelGateway(config)


def init_security():
    """初始化安全层组件"""
    return {
        'double_auth': DoubleAuthorization(),
        'permission_boundary': PermissionBoundary(),
        'capability_minimizer': CapabilityMinimizer(),
        'key_vault': KeyVault(),
        'action_guard': ActionGuard(),
        'compliance_audit': ComplianceAudit(),
    }


__all__ = [
    # 元信息
    '__version__',
    '__author__',
    '__license__',
    
    # 核心
    'StateBus',
    'Gate',
    'Expert',
    'MoE',
    'MPCPlanner',
    'GoalLayer',
    'Reflexion',
    'HaltDetector',
    'IntrinsicMotivation',
    'DifficultyScheduler',
    'Confidence',
    'Uncertainty',
    'WarmMemory',
    'ColdMemory',
    'ReleaseTrigger',
    'SwingEngine',
    'BalanceRegulator',
    'OnlineDistill',
    
    # LLM
    'LLMOrchestrator',
    'BaseProvider',
    'ModelRegistry',
    'OpenAIProvider',
    'AnthropicProvider',
    'ZhipuAIProvider',
    'WenxinProvider',
    'DashScopeProvider',
    'BaichuanProvider',
    'MoonshotProvider',
    'YiProvider',
    'SparkProvider',
    'DeepSeekProvider',
    'GLMFlashProvider',
    'DoubaoProvider',
    'MiniMaxProvider',
    'SkyworkProvider',
    'WPSLingxiProvider',
    'MimoProvider',
    'WuDaoProvider',
    'LocalProvider',
    
    # Channel
    'ChannelGateway',
    'BaseChannelAdapter',
    'UniversalMessage',
    'SessionManager',
    'RateLimiter',
    'HealthChecker',
    'ChannelMetrics',
    'WebhookHandler',
    'ChannelRouter',
    'DingTalkAdapter',
    'WeChatWorkAdapter',
    'FeishuAdapter',
    'SlackAdapter',
    'TelegramAdapter',
    'DiscordAdapter',
    'TencentDocsAdapter',
    'TuyaIoTAdapter',
    'EmailAdapter',
    'AliyunOSSAdapter',
    'TencentCOSAdapter',
    'TencentVectorDBAdapter',
    'DingTalkYidaAdapter',
    'TencentWeDaAdapter',
    'BaiduPanAdapter',
    'TencentExmailAdapter',
    'MingdaoAdapter',
    'QingflowAdapter',
    'TiDBCloudAdapter',
    
    # 技能系统
    'SkillParser',
    'SkillExecutor',
    'SkillMarket',
    'SkillMetadata',
    'SkillRuntime',
    
    # 插件系统
    'PluginSDK',
    'PluginManager',
    'PluginSandbox',
    'PluginSecurity',
    'PluginLifecycle',
    'PluginManifest',
    'PluginContext',
    
    # 安全
    'DoubleAuthorization',
    'PermissionBoundary',
    'CapabilityMinimizer',
    'KeyVault',
    'ActionGuard',
    'ComplianceAudit',
    
    # RAG
    'RAGEmbedder',
    'RAGRetriever',
    'RAGPipeline',
    'VectorStore',
    
    # 工作流
    'WorkflowEngine',
    'WorkflowExecutor',
    'DSLParser',
    'StateManager',
    'GraphEngine',
    
    # 多智能体
    'AllianceGenetic',
    
    # 联邦学习
    'FederatedServer',
    'MPC',
    'EvolutionCoordinator',
    
    # 沙箱
    'SandboxInterface',
    'DockerExecutor',
    'NsjailExecutor',
    'VirtualenvExecutor',
    
    # AISec
    'ZeroTrustExecutor',
    'CommandInterceptor',
    
    # 数据管道
    'DataDownloader',
    'DataPreprocessor',
    'DataCache',
    'DataLoader',
    
    # 存储
    'FileRepository',
    'MemoryRepository',
    'RedisRepository',
    'SQLAlchemyRepository',
    'ConnectionPool',
    'ShardingManager',
    'BackupManager',
    
    # 数据库
    'DatabaseConnection',
    'CRUDBase',
    'BaseModel',
    
    # 遥测
    'TelemetryTracer',
    'CostAnalyzer',
    
    # 评估
    'BenchmarkRunner',
    'MetricsCollector',
    
    # 边缘计算
    'ONNXOptimizer',
    'TFLiteConverter',
    'JetsonOptimizer',
    
    # IoT
    'BLEScanner',
    'ModbusRTUClient',
    'OPCUAClient',
    
    # 机器人
    'RobotController',
    'DigitalTwin',
    'EmergencyStop',
    'EnergyManager',
    'FaultDiagnosis',
    'SafetyFence',
    'Teleoperation',
    
    # 视频生成
    'VideoGenAPI',
    'VideoInferencer',
    'ScriptGenerator',
    
    # 多模态
    'MultimodalCore',
    'MultimodalProcessor',
    
    # 本地AI
    'LocalAI',
    
    # 持续学习
    'ReplayBuffer',
    'SDFD',
    
    # 群智
    'AgentRegistry',
    'Alliance',
    'Communication',
    'Consensus',
    'Coordination',
    'Debate',
    'Incentive',
    'KnowledgeTransfer',
    'Reputation',
    'TaskAllocation',
    
    # 计算机使用
    'AgentBrain',
    'ScreenEngine',
    'WindowEngine',
    'ElementEngine',
    'InputEngine',
    'ClipboardEngine',
    'VisionEngine',
    'OCREngine',
    'FileOps',
    'WorkflowRecorder',
    
    # 便捷函数
    'get_version',
    'list_supported_providers',
    'list_supported_channels',
    'create_orchestrator',
    'create_gateway',
    'init_security',
]
