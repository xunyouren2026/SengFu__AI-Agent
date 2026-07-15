"""
AGI Unified Framework - Security Module
========================================

安全合规模块,实现以下安全原则:

1. 双重授权 (Double Authorization)
   - 平台授权 + 用户授权
   - 支持 AND/OR/XOR 授权模式
   
2. 权限边界与RBAC (Permission Boundary)
   - 基于角色的访问控制
   - 权限最小化原则
   
3. 能力最小化 (Capability Minimization)
   - 会话级能力管理
   - 危险操作自动降级
   
4. API密钥安全存储 (Key Vault)
   - 多层加密(Fernet/AES/XOR)
   - 机器绑定,防泄露
   
5. 操作确认机制 (Action Guard)
   - 四级敏感度分级
   - 白名单/黑名单策略
   - 防篡改审计日志
   
6. 合规审计 (Compliance Audit)
   - 统一审计日志聚合
   - GDPR/SOC2/ISO27001合规检查
   - 实时告警和报告生成

Author: AGI Framework Security Team
Version: 1.1.0
"""

# Double Authorization
from .double_auth import (
    DoubleAuthorization,
    PlatformAuthorization,
    UserAuthorization,
    AuthorizationContext,
    AuthorizationResult,
    ConsentRecord,
    AgentAction,
    ActionSensitivity,
    AuthorizationMode,
    ConsentScope,
    create_authorization_context
)

# Permission Boundary & RBAC
from .permission_boundary import (
    PermissionEnforcer,
    PermissionBoundary,
    PermissionCondition,
    PermissionEnforcerConfig,
    Role,
    ActionType,
    DataCategory,
    CheckResult,
    ExecutionContext,
    create_execution_context,
    check_permission_before_action_guard
)

# Capability Minimization
from .capability_minimizer import (
    CapabilityMinimizer,
    CapabilityRegistry,
    CapabilityBoundary,
    DangerousOperationReducer,
    CapabilityEscalation,
    Capability,
    ToolCategory,
    CapabilityContext,
    check_capability_before_action_guard
)

# API Key Vault
from .key_vault import (
    KeyVault,
    KeyValidator,
    KeyRotationManager,
    VaultConfig
)

# Action Guard
from .action_guard import (
    ActionGuard,
    GuardConfig,
    ActionPolicy,
    AuditLogger,
    Sensitivity,
    DEFAULT_SENSITIVITY
)

# Compliance Audit
from .compliance_audit import (
    ComplianceAuditSystem,
    ComplianceChecker,
    ReportGenerator,
    ComplianceExporter,
    AlertDispatcher,
    UnifiedAuditEvent,
    SecurityAlert,
    ComplianceReport
)

__all__ = [
    # Double Authorization
    'DoubleAuthorization',
    'PlatformAuthorization', 
    'UserAuthorization',
    'AuthorizationContext',
    'AuthorizationResult',
    'ConsentRecord',
    'AgentAction',
    'ActionSensitivity',
    'AuthorizationMode',
    'ConsentScope',
    'create_authorization_context',
    
    # Permission Boundary
    'PermissionEnforcer',
    'PermissionBoundary',
    'PermissionCondition',
    'PermissionEnforcerConfig',
    'Role',
    'ActionType',
    'DataCategory',
    'CheckResult',
    'ExecutionContext',
    'create_execution_context',
    'check_permission_before_action_guard',
    
    # Capability Minimization
    'CapabilityMinimizer',
    'CapabilityRegistry',
    'CapabilityBoundary',
    'DangerousOperationReducer',
    'CapabilityEscalation',
    'Capability',
    'ToolCategory',
    'CapabilityContext',
    'check_capability_before_action_guard',
    
    # Key Vault
    'KeyVault',
    'KeyValidator',
    'KeyRotationManager',
    'VaultConfig',
    
    # Action Guard
    'ActionGuard',
    'GuardConfig',
    'ActionPolicy',
    'AuditLogger',
    'Sensitivity',
    'DEFAULT_SENSITIVITY',
    
    # Compliance Audit
    'ComplianceAuditSystem',
    'ComplianceChecker',
    'ReportGenerator',
    'ComplianceExporter',
    'AlertDispatcher',
    'UnifiedAuditEvent',
    'SecurityAlert',
    'ComplianceReport'
]

__version__ = '1.1.0'
