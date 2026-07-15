"""
Code模块 - 代码安全扫描
"""
from .static_scanner import (
    StaticScanner,
    CodeIssue,
    ScanResult,
    Severity,
    IssueType
)
from .dependency_scanner import (
    DependencyScanner,
    Vulnerability,
    Dependency,
    VulnerabilitySeverity
)
from .dangerous_imports import (
    DangerousImportDetector,
    DangerousImport,
    RiskLevel
)
from .supply_chain_check import (
    SupplyChainChecker,
    SupplyChainRisk,
    SupplyChainRiskType,
    PackageInfo
)
from .report_generator import (
    ReportGenerator,
    SecurityReport
)
from .remediation_advice import (
    RemediationAdvice,
    Remediation,
    FixDifficulty
)

__all__ = [
    # static_scanner.py
    "StaticScanner",
    "CodeIssue",
    "ScanResult",
    "Severity",
    "IssueType",
    # dependency_scanner.py
    "DependencyScanner",
    "Vulnerability",
    "Dependency",
    "VulnerabilitySeverity",
    # dangerous_imports.py
    "DangerousImportDetector",
    "DangerousImport",
    "RiskLevel",
    # supply_chain_check.py
    "SupplyChainChecker",
    "SupplyChainRisk",
    "SupplyChainRiskType",
    "PackageInfo",
    # report_generator.py
    "ReportGenerator",
    "SecurityReport",
    # remediation_advice.py
    "RemediationAdvice",
    "Remediation",
    "FixDifficulty"
]
