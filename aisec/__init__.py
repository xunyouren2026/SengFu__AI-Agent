"""
AISEC - AI安全模块

统一的AI安全框架，包含：
- 网关代理与规则引擎
- Prompt安全防护
- 代码安全扫描
- 安全执行环境
- 恶意软件检测
- 蓝队防御
- 对抗鲁棒性
- 蜜罐系统
- 水印与签名
- 模型指纹
"""

# Gateway模块
from .gateway import (
    TransparentProxy,
    RequestInterceptor,
    HTTPRequest,
    InterceptResult,
    InterceptAction,
    ProxyChain,
    RateLimiter,
    RuleEngine,
    Rule,
    RuleBuilder,
    RuleTemplates,
    Condition,
    ConditionGroup,
    ConditionOperator,
    LogicalOperator,
    Action,
    ActionType
)

# Prompt模块
from .prompt import (
    PromptGuard,
    GuardAction,
    GuardResult,
    PromptRewriter,
    RewriteAction,
    RewriteRule,
    RewriteResult,
    ContextRewriter,
    InjectionScanner,
    InjectionMatch,
    InjectionType,
    JailbreakDetector,
    JailbreakMatch,
    JailbreakType,
    LeakageDetector,
    LeakageMatch,
    LeakageType,
    FieldMasker,
    FieldRule,
    MaskStrategy,
    TokenizationEngine,
    TokenType
)

# Code模块
from .code import (
    StaticScanner,
    CodeIssue,
    ScanResult,
    Severity,
    IssueType,
    DependencyScanner,
    Vulnerability,
    Dependency,
    VulnerabilitySeverity,
    DangerousImportDetector,
    DangerousImport,
    RiskLevel,
    SupplyChainChecker,
    SupplyChainRisk,
    SupplyChainRiskType,
    PackageInfo,
    ReportGenerator,
    SecurityReport,
    RemediationAdvice,
    Remediation,
    FixDifficulty
)

# Execution模块
from .execution import (
    DockerSandbox,
    DockerSandboxBuilder,
    ContainerConfig,
    ContainerStatus,
    ExecutionResult,
    GVisorSandbox,
    GVisorSandboxBuilder,
    GVisorConfig,
    GVisorRuntime,
    SeccompProfiles,
    ResourceQuotaManager,
    ResourceQuota,
    ResourceUsage,
    ResourceType,
    QuotaViolation,
    ProcessMonitor,
    SyscallFilter,
    SyscallFilterBuilder,
    SyscallFilterManager,
    SyscallProfiles,
    SyscallRule,
    SyscallAction,
    Architecture,
    NetworkPolicy,
    NetworkPolicyBuilder,
    NetworkPolicyManager,
    NetworkRule,
    NetworkAction,
    Protocol,
    IPRange,
    IPRanges,
    ExecutionCleaner,
    CleanupScheduler,
    CleanupResult,
    CleanupType,
    ResourceTracker
)

# Malware模块
from .malware import (
    SignatureScanner,
    SignatureDatabase,
    Signature,
    ScanMatch,
    ScanResult as MalwareScanResult,
    MalwareType,
    Severity as MalwareSeverity,
    BehaviorMonitor,
    BehaviorEvent,
    BehaviorRule,
    BehaviorType,
    RiskLevel as BehaviorRiskLevel,
    ProcessBehaviorProfile,
    YaraEngine,
    YaraRule,
    YaraString,
    YaraMatch
)

# BlueTeam模块
from .blueteam import (
    DefenseLearner,
    AttackSample,
    DefensePattern,
    LearningResult,
    LearningMode,
    AttackType,
    RuleGenerator,
    GeneratedRule,
    RuleType,
    RuleSeverity
)

# Adversarial Robustness模块
from .adversarial_robustness import (
    PGDTrainer,
    PGDConfig,
    AdversarialExample,
    AdversarialDetector,
    AttackType as AdversarialAttackType,
    ModelStealDefense,
    DefenseConfig,
    DefenseStrategy,
    QueryRecord,
    BackdoorDetector,
    BackdoorDetectionResult,
    BackdoorType,
    TriggerCandidate
)

# Honeypot模块
from .honeypot import (
    DecoyManager,
    Decoy,
    DecoyEvent,
    DecoyType,
    ForensicsAnalyzer,
    ForensicEvidence,
    AttackSession,
    AttackPhase
)

# Watermark模块
from .watermark import (
    WatermarkEmbedder,
    WatermarkConfig,
    WatermarkResult,
    WatermarkType,
    WatermarkExtractor,
    ExtractionResult,
    ContentSigner,
    ContentSignature,
    VerificationResult,
    SignatureAlgorithm
)

# Fingerprint模块
from .fingerprint import (
    ModelHasher,
    ModelFingerprint,
    FingerprintType,
    FingerprintVerifier,
    VerificationResult as FingerprintVerificationResult,
    VerificationStatus
)

# 根目录模块
from .command_interceptor import (
    CommandInterceptor,
    CommandAction,
    RiskLevel as CommandRiskLevel,
    CommandAnalysis,
    InterceptResult as CommandInterceptResult
)

from .zero_trust_executor import (
    ZeroTrustExecutor,
    TrustLevel,
    ExecutionMode,
    ExecutionContext,
    ExecutionRequest,
    ExecutionResult as ZeroTrustExecutionResult
)

__version__ = "1.0.0"

__all__ = [
    # Gateway
    "TransparentProxy",
    "RequestInterceptor",
    "HTTPRequest",
    "InterceptResult",
    "InterceptAction",
    "ProxyChain",
    "RateLimiter",
    "RuleEngine",
    "Rule",
    "RuleBuilder",
    "RuleTemplates",
    "Condition",
    "ConditionGroup",
    "ConditionOperator",
    "LogicalOperator",
    "Action",
    "ActionType",
    
    # Prompt
    "PromptGuard",
    "GuardAction",
    "GuardResult",
    "PromptRewriter",
    "RewriteAction",
    "RewriteRule",
    "RewriteResult",
    "ContextRewriter",
    "InjectionScanner",
    "InjectionMatch",
    "InjectionType",
    "JailbreakDetector",
    "JailbreakMatch",
    "JailbreakType",
    "LeakageDetector",
    "LeakageMatch",
    "LeakageType",
    "FieldMasker",
    "FieldRule",
    "MaskStrategy",
    "TokenizationEngine",
    "TokenType",
    
    # Code
    "StaticScanner",
    "CodeIssue",
    "ScanResult",
    "Severity",
    "IssueType",
    "DependencyScanner",
    "Vulnerability",
    "Dependency",
    "VulnerabilitySeverity",
    "DangerousImportDetector",
    "DangerousImport",
    "RiskLevel",
    "SupplyChainChecker",
    "SupplyChainRisk",
    "SupplyChainRiskType",
    "PackageInfo",
    "ReportGenerator",
    "SecurityReport",
    "RemediationAdvice",
    "Remediation",
    "FixDifficulty",
    
    # Execution
    "DockerSandbox",
    "DockerSandboxBuilder",
    "ContainerConfig",
    "ContainerStatus",
    "ExecutionResult",
    "GVisorSandbox",
    "GVisorSandboxBuilder",
    "GVisorConfig",
    "GVisorRuntime",
    "SeccompProfiles",
    "ResourceQuotaManager",
    "ResourceQuota",
    "ResourceUsage",
    "ResourceType",
    "QuotaViolation",
    "ProcessMonitor",
    "SyscallFilter",
    "SyscallFilterBuilder",
    "SyscallFilterManager",
    "SyscallProfiles",
    "SyscallRule",
    "SyscallAction",
    "Architecture",
    "NetworkPolicy",
    "NetworkPolicyBuilder",
    "NetworkPolicyManager",
    "NetworkRule",
    "NetworkAction",
    "Protocol",
    "IPRange",
    "IPRanges",
    "ExecutionCleaner",
    "CleanupScheduler",
    "CleanupResult",
    "CleanupType",
    "ResourceTracker",
    
    # Malware
    "SignatureScanner",
    "SignatureDatabase",
    "Signature",
    "ScanMatch",
    "MalwareScanResult",
    "MalwareType",
    "MalwareSeverity",
    "BehaviorMonitor",
    "BehaviorEvent",
    "BehaviorRule",
    "BehaviorType",
    "BehaviorRiskLevel",
    "ProcessBehaviorProfile",
    "YaraEngine",
    "YaraRule",
    "YaraString",
    "YaraMatch",
    
    # BlueTeam
    "DefenseLearner",
    "AttackSample",
    "DefensePattern",
    "LearningResult",
    "LearningMode",
    "AttackType",
    "RuleGenerator",
    "GeneratedRule",
    "RuleType",
    "RuleSeverity",
    
    # Adversarial Robustness
    "PGDTrainer",
    "PGDConfig",
    "AdversarialExample",
    "AdversarialDetector",
    "AdversarialAttackType",
    "ModelStealDefense",
    "DefenseConfig",
    "DefenseStrategy",
    "QueryRecord",
    "BackdoorDetector",
    "BackdoorDetectionResult",
    "BackdoorType",
    "TriggerCandidate",
    
    # Honeypot
    "DecoyManager",
    "Decoy",
    "DecoyEvent",
    "DecoyType",
    "ForensicsAnalyzer",
    "ForensicEvidence",
    "AttackSession",
    "AttackPhase",
    
    # Watermark
    "WatermarkEmbedder",
    "WatermarkConfig",
    "WatermarkResult",
    "WatermarkType",
    "WatermarkExtractor",
    "ExtractionResult",
    "ContentSigner",
    "ContentSignature",
    "VerificationResult",
    "SignatureAlgorithm",
    
    # Fingerprint
    "ModelHasher",
    "ModelFingerprint",
    "FingerprintType",
    "FingerprintVerifier",
    "FingerprintVerificationResult",
    "VerificationStatus",
    
    # Root
    "CommandInterceptor",
    "CommandAction",
    "CommandRiskLevel",
    "CommandAnalysis",
    "CommandInterceptResult",
    "ZeroTrustExecutor",
    "TrustLevel",
    "ExecutionMode",
    "ExecutionContext",
    "ExecutionRequest",
    "ZeroTrustExecutionResult",
]
