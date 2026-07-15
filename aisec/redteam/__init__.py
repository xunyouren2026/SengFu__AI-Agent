"""
红队测试模块

提供AI系统红队测试功能，包括对抗攻击、漏洞扫描、渗透测试、自动化测试和防御验证
"""

# 对抗攻击模块
from .adversarial_attacks import (
    AttackType,
    AttackTarget,
    AttackResult,
    AttackMetrics,
    AdversarialAttack,
    FGSM,
    PGD,
    CWAttack,
    TextFooler,
    PromptInjection,
    JailbreakAttack,
    ModelExtraction,
    MembershipInference,
)

# 漏洞扫描模块
from .vulnerability_scanner import (
    SeverityLevel,
    VulnerabilityType,
    ModelVulnerability,
    VulnerabilityScanner,
    BiasScanner,
    ToxicityScanner,
    PrivacyLeakScanner,
    RobustnessScanner,
    OWASPLLM,
)

# 渗透测试模块
from .penetration_tester import (
    AttackPhase,
    RiskLevel,
    AttackVector,
    AttackChain,
    Finding,
    PenTestReport,
    Reconnaissance,
    Exploitation,
    PostExploitation,
    PenetrationTester,
)

# 自动化模块
from .automation import (
    CampaignStatus,
    ScenarioType,
    Campaign,
    AttackScenario,
    RedTeamReport,
    ScenarioLibrary,
    RedTeamAutomation,
    ContinuousRedTeam,
)

# 防御验证模块
from .defense_validator import (
    DefenseType,
    TestResult,
    DefenseTest,
    DefenseMetrics,
    BypassTechnique,
    DefenseValidator,
    BypassEncoder,
    DefenseEvasionTester,
)

__version__ = "1.0.0"

__all__ = [
    # 对抗攻击
    "AttackType",
    "AttackTarget",
    "AttackResult",
    "AttackMetrics",
    "AdversarialAttack",
    "FGSM",
    "PGD",
    "CWAttack",
    "TextFooler",
    "PromptInjection",
    "JailbreakAttack",
    "ModelExtraction",
    "MembershipInference",
    
    # 漏洞扫描
    "SeverityLevel",
    "VulnerabilityType",
    "ModelVulnerability",
    "VulnerabilityScanner",
    "BiasScanner",
    "ToxicityScanner",
    "PrivacyLeakScanner",
    "RobustnessScanner",
    "OWASPLLM",
    
    # 渗透测试
    "AttackPhase",
    "RiskLevel",
    "AttackVector",
    "AttackChain",
    "Finding",
    "PenTestReport",
    "Reconnaissance",
    "Exploitation",
    "PostExploitation",
    "PenetrationTester",
    
    # 自动化
    "CampaignStatus",
    "ScenarioType",
    "Campaign",
    "AttackScenario",
    "RedTeamReport",
    "ScenarioLibrary",
    "RedTeamAutomation",
    "ContinuousRedTeam",
    
    # 防御验证
    "DefenseType",
    "TestResult",
    "DefenseTest",
    "DefenseMetrics",
    "BypassTechnique",
    "DefenseValidator",
    "BypassEncoder",
    "DefenseEvasionTester",
]
