"""
权限边界与RBAC框架 (Permission Boundary and RBAC Framework)
===========================================================

本模块实现了"权限最小化"安全原则,通过以下机制确保用户和代理
只能访问和执行其角色所需的最小权限集:

1. PermissionBoundary - 权限边界定义
   - 限制用户可执行的操作类型
   - 限制用户可访问的资源
   - 定义条件性访问规则

2. RBAC (基于角色的访问控制)
   - 预定义角色: ADMIN, OPERATOR, VIEWER, GUEST
   - 角色-权限映射
   - 用户-角色分配

3. PermissionEnforcer - 权限执行器
   - 在操作执行前进行权限检查
   - 支持动态权限调整
   - 集成条件性访问规则

Author: AGI Framework Security Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Tuple

# 配置日志记录器
logger = logging.getLogger(__name__)


class Role(Enum):
    """
    角色枚举
    
    定义系统中的预定义角色,按权限从高到低排列:
    - ADMIN: 管理员,拥有全部权限
    - OPERATOR: 操作员,中等至高权限操作,工作时间内有效
    - VIEWER: 查看者,仅低权限操作
    - GUEST: 访客,仅截图操作
    """
    ADMIN = auto()
    OPERATOR = auto()
    VIEWER = auto()
    GUEST = auto()
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def priority(self) -> int:
        """获取角色优先级,数字越大权限越高"""
        priority_map = {
            Role.ADMIN: 100,
            Role.OPERATOR: 50,
            Role.VIEWER: 25,
            Role.GUEST: 10
        }
        return priority_map.get(self, 0)


class ActionType(Enum):
    """
    操作类型枚举
    
    定义系统支持的操作类型,按敏感度分组:
    - LOW: 低敏感度操作,几乎所有角色都可执行
    - MEDIUM: 中等敏感度操作,需要OPERATOR及以上角色
    - HIGH: 高敏感度操作,需要ADMIN角色
    - CRITICAL: 极高敏感度操作,需要ADMIN且记录审计日志
    """
    # 低敏感度操作
    SCREENSHOT = ("LOW", "截图")
    FIND_ELEMENT = ("LOW", "查找元素")
    WAIT = ("LOW", "等待")
    SCROLL = ("LOW", "滚动")
    GET_TEXT = ("LOW", "获取文本")
    GET_ATTRIBUTE = ("LOW", "获取属性")
    
    # 中等敏感度操作
    CLICK = ("MEDIUM", "点击")
    TYPE = ("MEDIUM", "输入文本")
    COPY = ("MEDIUM", "复制")
    PASTE = ("MEDIUM", "粘贴")
    NAVIGATE = ("MEDIUM", "导航")
    REFRESH = ("MEDIUM", "刷新")
    
    # 高敏感度操作
    DELETE_FILES = ("HIGH", "删除文件")
    WRITE_FILE = ("HIGH", "写入文件")
    EXECUTE_SCRIPT = ("HIGH", "执行脚本")
    SEND_MESSAGE = ("HIGH", "发送消息")
    API_CALL_EXTERNAL = ("HIGH", "外部API调用")
    
    # 极高敏感度操作
    FORMAT_DISK = ("CRITICAL", "格式化磁盘")
    DELETE_DATABASE = ("CRITICAL", "删除数据库")
    MODIFY_SYSTEM = ("CRITICAL", "修改系统配置")
    PRIVILEGE_ESCALATION = ("CRITICAL", "权限提升")
    
    def __init__(self, sensitivity: str, description: str):
        self.sensitivity = sensitivity
        self.description = description
    
    @property
    def sensitivity_level(self) -> str:
        """获取敏感度级别"""
        return self.sensitivity


class DataCategory(Enum):
    """
    数据分类枚举
    
    用于条件性权限控制:
    - PUBLIC: 公开数据,所有角色可访问
    - INTERNAL: 内部数据,需要OPERATOR及以上角色
    - CONFIDENTIAL: 机密数据,需要ADMIN角色
    - RESTRICTED: 限制数据,需要特殊授权
    """
    PUBLIC = auto()
    INTERNAL = auto()
    CONFIDENTIAL = auto()
    RESTRICTED = auto()


class CheckResult(Enum):
    """
    权限检查结果枚举
    
    - ALLOWED: 允许执行
    - DENIED: 明确拒绝
    - CONDITIONAL: 条件性允许(需满足某些条件)
    - REQUIRES_CONFIRMATION: 需要用户确认
    """
    ALLOWED = auto()
    DENIED = auto()
    CONDITIONAL = auto()
    REQUIRES_CONFIRMATION = auto()


@dataclass
class PermissionCondition:
    """
    权限条件
    
    定义条件性访问规则,只有满足所有条件才允许操作
    """
    time_range: Optional[Tuple[str, str]] = None  # 时间范围,格式: ("HH:MM", "HH:MM")
    ip_whitelist: List[str] = field(default_factory=list)  # IP白名单
    device_trusted: bool = False  # 是否需要可信设备
    data_classification: Optional[DataCategory] = None  # 最低数据分类要求
    require_mfa: bool = False  # 是否需要多因素认证
    
    def check(self, context: 'ExecutionContext') -> Tuple[bool, str]:
        """
        检查是否满足此权限条件
        
        Args:
            context: 执行上下文
            
        Returns:
            Tuple[bool, str]: (是否满足, 原因说明)
        """
        # 检查时间范围
        if self.time_range:
            current_time = datetime.now().strftime("%H:%M")
            start, end = self.time_range
            if not (start <= current_time <= end):
                return False, f"当前时间 {current_time} 不在工作时间 {start}-{end} 内"
        
        # 检查IP白名单
        if self.ip_whitelist and context.ip_address:
            if context.ip_address not in self.ip_whitelist:
                return False, f"IP {context.ip_address} 不在白名单内"
        
        # 检查设备可信性
        if self.device_trusted and not context.device_trusted:
            return False, "需要使用可信设备"
        
        # 检查数据分类
        if self.data_classification and context.data_category:
            required_level = self.data_classification.value
            actual_level = context.data_category.value
            if actual_level < required_level:
                return False, f"数据分类级别不足: 需要 {self.data_classification.name}"
        
        # 检查MFA
        if self.require_mfa and not context.mfa_verified:
            return False, "需要多因素认证"
        
        return True, "所有条件满足"


@dataclass
class PermissionBoundary:
    """
    权限边界
    
    定义用户可以执行的操作和可以访问的资源的边界
    """
    user_id: str                                              # 用户ID
    allowed_actions: Set[ActionType] = field(default_factory=set)  # 允许的操作集合
    allowed_resources: Set[str] = field(default_factory=set)  # 允许访问的资源(文件路径、API端点等)
    conditions: List[PermissionCondition] = field(default_factory=list)  # 条件性规则
    expires_at: Optional[datetime] = None                    # 过期时间
    
    def is_expired(self) -> bool:
        """检查权限边界是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at
    
    def is_action_allowed(self, action: ActionType) -> bool:
        """检查操作是否在允许列表中"""
        return action in self.allowed_actions
    
    def is_resource_allowed(self, resource: str) -> bool:
        """检查资源是否在允许列表中"""
        # 支持通配符匹配
        for allowed in self.allowed_resources:
            if allowed == "*":
                return True
            if allowed.endswith("*"):
                prefix = allowed[:-1]
                if resource.startswith(prefix):
                    return True
            if resource == allowed:
                return True
        return False
    
    def check_conditions(self, context: 'ExecutionContext') -> Tuple[bool, str]:
        """
        检查所有条件是否满足
        
        Args:
            context: 执行上下文
            
        Returns:
            Tuple[bool, str]: (是否满足, 原因说明)
        """
        for condition in self.conditions:
            allowed, reason = condition.check(context)
            if not allowed:
                return False, reason
        return True, "所有条件满足"


@dataclass
class ExecutionContext:
    """
    执行上下文
    
    包含权限检查所需的环境信息
    """
    session_id: str                                    # 会话ID
    user_id: str                                        # 用户ID
    ip_address: Optional[str] = None                   # 来源IP地址
    device_trusted: bool = False                       # 设备是否可信
    data_category: Optional[DataCategory] = None        # 当前数据分类
    mfa_verified: bool = False                         # MFA是否已验证
    current_time: datetime = field(default_factory=datetime.now)  # 当前时间
    
    def __post_init__(self):
        """确保current_time是datetime对象"""
        if isinstance(self.current_time, str):
            self.current_time = datetime.fromisoformat(self.current_time)


@dataclass
class RolePermissionMapping:
    """
    角色-权限映射
    
    定义每个预定义角色的默认权限边界
    """
    role: Role                                          # 角色
    permission_boundary: PermissionBoundary             # 权限边界
    
    @classmethod
    def create_default_mappings(cls) -> Dict[Role, 'RolePermissionMapping']:
        """
        创建默认的角色-权限映射
        
        Returns:
            Dict[Role, RolePermissionMapping]: 角色映射字典
        """
        mappings = {}
        
        # ADMIN: 全部权限
        admin_boundary = PermissionBoundary(
            user_id="*",
            allowed_actions=set(ActionType),  # 所有操作
            allowed_resources={"*"},          # 所有资源
            conditions=[]                     # 无条件
        )
        mappings[Role.ADMIN] = cls(Role.ADMIN, admin_boundary)
        
        # OPERATOR: MEDIUM+HIGH操作,工作时间限制
        operator_actions = {
            ActionType.SCREENSHOT, ActionType.FIND_ELEMENT, ActionType.WAIT,
            ActionType.SCROLL, ActionType.GET_TEXT, ActionType.GET_ATTRIBUTE,
            ActionType.CLICK, ActionType.TYPE, ActionType.COPY, ActionType.PASTE,
            ActionType.NAVIGATE, ActionType.REFRESH,
            ActionType.DELETE_FILES, ActionType.WRITE_FILE,
            ActionType.EXECUTE_SCRIPT, ActionType.SEND_MESSAGE, ActionType.API_CALL_EXTERNAL
        }
        operator_boundary = PermissionBoundary(
            user_id="*",
            allowed_actions=operator_actions,
            allowed_resources={"~/work/*", "~/documents/*", "/tmp/*"},
            conditions=[
                PermissionCondition(
                    time_range=("09:00", "18:00"),  # 工作时间
                    require_mfa=False
                )
            ]
        )
        mappings[Role.OPERATOR] = cls(Role.OPERATOR, operator_boundary)
        
        # VIEWER: 仅LOW操作
        viewer_actions = {
            ActionType.SCREENSHOT, ActionType.FIND_ELEMENT,
            ActionType.WAIT, ActionType.SCROLL,
            ActionType.GET_TEXT, ActionType.GET_ATTRIBUTE
        }
        viewer_boundary = PermissionBoundary(
            user_id="*",
            allowed_actions=viewer_actions,
            allowed_resources={"~/documents/view/*"},
            conditions=[]
        )
        mappings[Role.VIEWER] = cls(Role.VIEWER, viewer_boundary)
        
        # GUEST: 仅截图
        guest_boundary = PermissionBoundary(
            user_id="*",
            allowed_actions={ActionType.SCREENSHOT},
            allowed_resources={"*"},
            conditions=[]
        )
        mappings[Role.GUEST] = cls(Role.GUEST, guest_boundary)
        
        return mappings


@dataclass
class PermissionEnforcerConfig:
    """
    权限执行器配置
    """
    default_role: Role = Role.VIEWER                    # 默认角色
    require_explicit_grant: bool = True                # 是否需要显式授权
    session_timeout_minutes: int = 30                   # 会话超时时间(分钟)
    ip_whitelist_enabled: bool = False                 # 是否启用IP白名单
    enable_audit_log: bool = True                      # 是否启用审计日志
    max_failed_attempts: int = 3                       # 最大失败尝试次数
    lockout_duration_minutes: int = 15                 # 锁定持续时间(分钟)


@dataclass
class PermissionCheckRequest:
    """
    权限检查请求
    """
    action: ActionType                                  # 待检查的操作
    context: ExecutionContext                           # 执行上下文
    resource: Optional[str] = None                      # 目标资源
    reason: Optional[str] = None                        # 执行原因


@dataclass
class PermissionCheckResponse:
    """
    权限检查响应
    """
    result: CheckResult                                 # 检查结果
    reason: str                                         # 原因说明
    boundary: Optional[PermissionBoundary] = None     # 匹配的权限边界
    requires_action: Optional[str] = None              # 需要采取的行动(如"confirm")
    warnings: List[str] = field(default_factory=list)  # 警告信息


class PermissionEnforcer:
    """
    权限执行器
    
    核心组件,负责:
    - 检查用户权限
    - 分配和撤销角色
    - 执行权限边界检查
    - 记录审计日志
    """
    
    def __init__(
        self,
        config: Optional[PermissionEnforcerConfig] = None,
        storage_path: Optional[str] = None
    ):
        """
        初始化权限执行器
        
        Args:
            config: 执行器配置
            storage_path: 权限数据存储路径
        """
        self._lock = threading.RLock()
        self._config = config or PermissionEnforcerConfig()
        self._storage_path = storage_path or "/tmp/permission_enforcer.json"
        
        # 角色-权限映射(默认)
        self._role_mappings = RolePermissionMapping.create_default_mappings()
        
        # 用户ID -> 角色分配
        self._user_roles: Dict[str, Role] = {}
        
        # 用户ID -> 自定义权限边界(覆盖角色默认)
        self._custom_boundaries: Dict[str, PermissionBoundary] = {}
        
        # 会话 -> 权限边界(临时)
        self._session_boundaries: Dict[str, PermissionBoundary] = {}
        
        # 用户 -> 失败尝试计数
        self._failed_attempts: Dict[str, int] = {}
        
        # 用户 -> 锁定截止时间
        self._lockouts: Dict[str, datetime] = {}
        
        # 审计日志
        self._audit_log: List[Dict[str, Any]] = []
        
        # 加载已存储的数据
        self._load_data()
        
        logger.info("权限执行器初始化完成")
    
    def _load_data(self) -> None:
        """从存储文件加载数据"""
        if Path(self._storage_path).exists():
            try:
                with open(self._storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 加载用户角色
                    for user_id, role_name in data.get('user_roles', {}).items():
                        self._user_roles[user_id] = Role[role_name]
                    
                    # 加载自定义边界
                    for user_id, boundary_data in data.get('custom_boundaries', {}).items():
                        self._custom_boundaries[user_id] = self._deserialize_boundary(boundary_data)
                    
                    logger.info("已从存储加载权限数据")
            except Exception as e:
                logger.warning(f"加载权限数据失败: {e}")
    
    def _save_data(self) -> None:
        """保存数据到存储文件"""
        try:
            data = {
                'user_roles': {uid: role.name for uid, role in self._user_roles.items()},
                'custom_boundaries': {
                    uid: self._serialize_boundary(boundary)
                    for uid, boundary in self._custom_boundaries.items()
                }
            }
            
            Path(self._storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug("已保存权限数据")
        except Exception as e:
            logger.error(f"保存权限数据失败: {e}")
    
    def _serialize_boundary(self, boundary: PermissionBoundary) -> Dict[str, Any]:
        """序列化权限边界"""
        return {
            'user_id': boundary.user_id,
            'allowed_actions': [a.name for a in boundary.allowed_actions],
            'allowed_resources': list(boundary.allowed_resources),
            'conditions': [
                {
                    'time_range': c.time_range,
                    'ip_whitelist': c.ip_whitelist,
                    'device_trusted': c.device_trusted,
                    'data_classification': c.data_classification.name if c.data_classification else None,
                    'require_mfa': c.require_mfa
                }
                for c in boundary.conditions
            ],
            'expires_at': boundary.expires_at.isoformat() if boundary.expires_at else None
        }
    
    def _deserialize_boundary(self, data: Dict[str, Any]) -> PermissionBoundary:
        """反序列化权限边界"""
        conditions = []
        for c_data in data.get('conditions', []):
            conditions.append(PermissionCondition(
                time_range=tuple(c_data['time_range']) if c_data.get('time_range') else None,
                ip_whitelist=c_data.get('ip_whitelist', []),
                device_trusted=c_data.get('device_trusted', False),
                data_classification=DataCategory[c_data['data_classification']] 
                    if c_data.get('data_classification') else None,
                require_mfa=c_data.get('require_mfa', False)
            ))
        
        return PermissionBoundary(
            user_id=data['user_id'],
            allowed_actions={ActionType[a] for a in data.get('allowed_actions', [])},
            allowed_resources=set(data.get('allowed_resources', [])),
            conditions=conditions,
            expires_at=datetime.fromisoformat(data['expires_at']) 
                if data.get('expires_at') else None
        )
    
    def _check_lockout(self, user_id: str) -> bool:
        """
        检查用户是否被锁定
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: 如果被锁定返回True
        """
        lockout_until = self._lockouts.get(user_id)
        if lockout_until and datetime.now() < lockout_until:
            return True
        
        # 锁定已过期,清除
        if lockout_until:
            del self._lockouts[user_id]
            self._failed_attempts[user_id] = 0
        
        return False
    
    def _record_failed_attempt(self, user_id: str) -> None:
        """记录失败尝试"""
        self._failed_attempts[user_id] = self._failed_attempts.get(user_id, 0) + 1
        
        if self._failed_attempts[user_id] >= self._config.max_failed_attempts:
            self._lockouts[user_id] = datetime.now() + timedelta(
                minutes=self._config.lockout_duration_minutes
            )
            logger.warning(f"用户 {user_id} 已达到最大失败尝试次数,已被锁定")
    
    def _record_audit(self, event_type: str, user_id: str, details: Dict[str, Any]) -> None:
        """记录审计日志"""
        if not self._config.enable_audit_log:
            return
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            **details
        }
        self._audit_log.append(entry)
        logger.info(f"审计日志: {event_type} - {user_id}")
    
    def get_user_permissions(self, user_id: str) -> PermissionBoundary:
        """
        获取用户的权限边界
        
        优先返回自定义边界,然后是角色默认边界
        
        Args:
            user_id: 用户ID
            
        Returns:
            PermissionBoundary: 用户权限边界
        """
        with self._lock:
            # 首先检查自定义边界
            if user_id in self._custom_boundaries:
                return self._custom_boundaries[user_id]
            
            # 然后检查角色默认边界
            role = self._user_roles.get(user_id, self._config.default_role)
            role_mapping = self._role_mappings.get(role)
            
            if role_mapping:
                boundary = role_mapping.permission_boundary
                # 创建副本并设置正确的user_id
                return PermissionBoundary(
                    user_id=user_id,
                    allowed_actions=boundary.allowed_actions.copy(),
                    allowed_resources=boundary.allowed_resources.copy(),
                    conditions=boundary.conditions.copy(),
                    expires_at=boundary.expires_at
                )
            
            # 使用默认VIEWER边界
            viewer_mapping = self._role_mappings.get(Role.VIEWER)
            boundary = viewer_mapping.permission_boundary
            return PermissionBoundary(
                user_id=user_id,
                allowed_actions=boundary.allowed_actions.copy(),
                allowed_resources=boundary.allowed_resources.copy(),
                conditions=[],
                expires_at=None
            )
    
    def get_session_permissions(self, session_id: str) -> Optional[PermissionBoundary]:
        """
        获取会话的临时权限边界
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[PermissionBoundary]: 会话权限边界
        """
        with self._lock:
            return self._session_boundaries.get(session_id)
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """
        分配角色给用户
        
        Args:
            user_id: 用户ID
            role: 目标角色
            
        Returns:
            bool: 分配是否成功
        """
        with self._lock:
            old_role = self._user_roles.get(user_id)
            self._user_roles[user_id] = role
            self._save_data()
            
            self._record_audit('ROLE_ASSIGNED', user_id, {
                'old_role': old_role.name if old_role else None,
                'new_role': role.name
            })
            
            logger.info(f"用户 {user_id} 角色已分配: {old_role} -> {role}")
            return True
    
    def revoke_role(self, user_id: str) -> bool:
        """
        撤销用户的角色分配(恢复为默认角色)
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: 撤销是否成功
        """
        with self._lock:
            if user_id in self._user_roles:
                old_role = self._user_roles[user_id]
                del self._user_roles[user_id]
                self._save_data()
                
                self._record_audit('ROLE_REVOKED', user_id, {
                    'old_role': old_role.name
                })
                
                logger.info(f"用户 {user_id} 的角色 {old_role} 已撤销")
                return True
            
            return False
    
    def get_user_role(self, user_id: str) -> Role:
        """
        获取用户的当前角色
        
        Args:
            user_id: 用户ID
            
        Returns:
            Role: 用户角色
        """
        return self._user_roles.get(user_id, self._config.default_role)
    
    def add_permission_rule(
        self,
        user_id: str,
        action: ActionType,
        resource: str
    ) -> bool:
        """
        添加自定义权限规则
        
        为用户添加对特定操作和资源的权限
        
        Args:
            user_id: 用户ID
            action: 操作类型
            resource: 资源路径
            
        Returns:
            bool: 添加是否成功
        """
        with self._lock:
            # 获取或创建自定义边界
            if user_id not in self._custom_boundaries:
                base_boundary = self.get_user_permissions(user_id)
                self._custom_boundaries[user_id] = PermissionBoundary(
                    user_id=user_id,
                    allowed_actions=base_boundary.allowed_actions.copy(),
                    allowed_resources=base_boundary.allowed_resources.copy(),
                    conditions=base_boundary.conditions.copy()
                )
            
            boundary = self._custom_boundaries[user_id]
            boundary.allowed_actions.add(action)
            boundary.allowed_resources.add(resource)
            
            self._save_data()
            
            self._record_audit('PERMISSION_RULE_ADDED', user_id, {
                'action': action.name,
                'resource': resource
            })
            
            logger.info(f"为用户 {user_id} 添加权限规则: {action.name} @ {resource}")
            return True
    
    def remove_permission_rule(
        self,
        user_id: str,
        action: ActionType,
        resource: str
    ) -> bool:
        """
        移除自定义权限规则
        
        Args:
            user_id: 用户ID
            action: 操作类型
            resource: 资源路径
            
        Returns:
            bool: 移除是否成功
        """
        with self._lock:
            if user_id not in self._custom_boundaries:
                return False
            
            boundary = self._custom_boundaries[user_id]
            
            # 移除规则(只在完全匹配时移除)
            if action in boundary.allowed_actions:
                boundary.allowed_actions.discard(action)
            if resource in boundary.allowed_resources:
                boundary.allowed_resources.discard(resource)
            
            # 如果自定义边界与默认边界相同,则删除自定义边界
            default_boundary = self._role_mappings.get(
                self._user_roles.get(user_id, self._config.default_role)
            ).permission_boundary
            
            if (boundary.allowed_actions == default_boundary.allowed_actions and
                boundary.allowed_resources == default_boundary.allowed_resources):
                del self._custom_boundaries[user_id]
            
            self._save_data()
            
            self._record_audit('PERMISSION_RULE_REMOVED', user_id, {
                'action': action.name,
                'resource': resource
            })
            
            logger.info(f"从用户 {user_id} 移除权限规则: {action.name} @ {resource}")
            return True
    
    def enforce(self, request: PermissionCheckRequest) -> PermissionCheckResponse:
        """
        执行权限检查
        
        这是权限执行的核心方法,所有操作执行前都应调用此方法
        
        Args:
            request: 权限检查请求
            
        Returns:
            PermissionCheckResponse: 检查结果
        """
        with self._lock:
            user_id = request.context.user_id
            
            # 检查是否被锁定
            if self._check_lockout(user_id):
                self._record_audit('ACCESS_DENIED_LOCKED', user_id, {
                    'action': request.action.name,
                    'reason': '用户被锁定'
                })
                return PermissionCheckResponse(
                    result=CheckResult.DENIED,
                    reason=f"用户已被锁定,请 {self._config.lockout_duration_minutes} 分钟后重试"
                )
            
            # 获取权限边界
            boundary = self.get_user_permissions(user_id)
            
            # 检查是否过期
            if boundary.is_expired():
                self._record_audit('ACCESS_DENIED_EXPIRED', user_id, {
                    'action': request.action.name,
                    'reason': '权限边界已过期'
                })
                return PermissionCheckResponse(
                    result=CheckResult.DENIED,
                    reason="您的权限边界已过期,请联系管理员续期",
                    boundary=boundary
                )
            
            # 检查操作是否允许
            if not boundary.is_action_allowed(request.action):
                self._record_audit('ACCESS_DENIED_ACTION', user_id, {
                    'action': request.action.name,
                    'reason': '操作不在允许列表中'
                })
                return PermissionCheckResponse(
                    result=CheckResult.DENIED,
                    reason=f"操作 {request.action.description} 不在您的权限范围内",
                    boundary=boundary
                )
            
            # 检查资源是否允许
            if request.resource and not boundary.is_resource_allowed(request.resource):
                self._record_audit('ACCESS_DENIED_RESOURCE', user_id, {
                    'action': request.action.name,
                    'resource': request.resource,
                    'reason': '资源不在允许列表中'
                })
                return PermissionCheckResponse(
                    result=CheckResult.DENIED,
                    reason=f"资源 {request.resource} 不在您的权限范围内",
                    boundary=boundary
                )
            
            # 检查条件性规则
            conditions_ok, condition_reason = boundary.check_conditions(request.context)
            if not conditions_ok:
                self._record_audit('ACCESS_DENIED_CONDITION', user_id, {
                    'action': request.action.name,
                    'reason': condition_reason
                })
                return PermissionCheckResponse(
                    result=CheckResult.CONDITIONAL,
                    reason=condition_reason,
                    boundary=boundary,
                    requires_action='satisfy_conditions'
                )
            
            # 检查是否需要确认
            if request.action.sensitivity in ('HIGH', 'CRITICAL'):
                self._record_audit('ACCESS_REQUIRES_CONFIRMATION', user_id, {
                    'action': request.action.name,
                    'sensitivity': request.action.sensitivity
                })
                return PermissionCheckResponse(
                    result=CheckResult.REQUIRES_CONFIRMATION,
                    reason=f"高敏感度操作 {request.action.description} 需要用户确认",
                    boundary=boundary,
                    requires_action='confirm',
                    warnings=["此操作需要二次确认"]
                )
            
            # 所有检查通过
            self._record_audit('ACCESS_ALLOWED', user_id, {
                'action': request.action.name,
                'resource': request.resource
            })
            
            return PermissionCheckResponse(
                result=CheckResult.ALLOWED,
                reason="权限检查通过",
                boundary=boundary
            )
    
    def check_action(self, user_id: str, action: ActionType) -> PermissionCheckResponse:
        """
        便捷方法: 检查用户是否有权限执行操作
        
        Args:
            user_id: 用户ID
            action: 操作类型
            
        Returns:
            PermissionCheckResponse: 检查结果
        """
        context = ExecutionContext(
            session_id="check_session",
            user_id=user_id
        )
        
        request = PermissionCheckRequest(
            action=action,
            context=context
        )
        
        return self.enforce(request)
    
    def create_session_boundary(
        self,
        session_id: str,
        user_id: str,
        actions: Set[ActionType],
        resources: Set[str],
        timeout_minutes: int = 30
    ) -> PermissionBoundary:
        """
        为会话创建临时权限边界
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            actions: 允许的操作集合
            resources: 允许的资源集合
            timeout_minutes: 超时时间(分钟)
            
        Returns:
            PermissionBoundary: 创建的权限边界
        """
        with self._lock:
            boundary = PermissionBoundary(
                user_id=user_id,
                allowed_actions=actions,
                allowed_resources=resources,
                expires_at=datetime.now() + timedelta(minutes=timeout_minutes)
            )
            
            self._session_boundaries[session_id] = boundary
            
            logger.info(f"为会话 {session_id} 创建临时权限边界,超时: {timeout_minutes}分钟")
            
            return boundary
    
    def revoke_session_boundary(self, session_id: str) -> bool:
        """
        撤销会话的临时权限边界
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 撤销是否成功
        """
        with self._lock:
            if session_id in self._session_boundaries:
                del self._session_boundaries[session_id]
                logger.info(f"已撤销会话 {session_id} 的临时权限边界")
                return True
            return False
    
    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            user_id: 用户ID(可选,过滤特定用户)
            event_type: 事件类型(可选)
            limit: 返回的最大记录数
            
        Returns:
            List[Dict[str, Any]]: 审计日志条目列表
        """
        with self._lock:
            logs = self._audit_log
            
            if user_id:
                logs = [l for l in logs if l.get('user_id') == user_id]
            
            if event_type:
                logs = [l for l in logs if l.get('event_type') == event_type]
            
            return logs[-limit:]
    
    def get_role_permissions(self, role: Role) -> PermissionBoundary:
        """
        获取角色的默认权限边界
        
        Args:
            role: 角色
            
        Returns:
            PermissionBoundary: 角色权限边界
        """
        mapping = self._role_mappings.get(role)
        if mapping:
            return mapping.permission_boundary
        return PermissionBoundary(user_id="*", allowed_actions=set(), allowed_resources=set())
    
    def export_permissions(self, user_id: str) -> Dict[str, Any]:
        """
        导出用户的权限配置
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 权限配置信息
        """
        boundary = self.get_user_permissions(user_id)
        role = self.get_user_role(user_id)
        
        return {
            'user_id': user_id,
            'role': role.name,
            'allowed_actions': [a.name for a in boundary.allowed_actions],
            'allowed_resources': list(boundary.allowed_resources),
            'conditions': [
                {
                    'time_range': c.time_range,
                    'ip_whitelist': c.ip_whitelist,
                    'device_trusted': c.device_trusted,
                    'data_classification': c.data_classification.name if c.data_classification else None
                }
                for c in boundary.conditions
            ],
            'expires_at': boundary.expires_at.isoformat() if boundary.expires_at else None,
            'is_custom': user_id in self._custom_boundaries
        }


def create_execution_context(
    session_id: str,
    user_id: str,
    **kwargs
) -> ExecutionContext:
    """
    便捷函数: 创建执行上下文
    
    Args:
        session_id: 会话ID
        user_id: 用户ID
        **kwargs: 其他可选参数
        
    Returns:
        ExecutionContext: 执行上下文对象
    """
    return ExecutionContext(
        session_id=session_id,
        user_id=user_id,
        ip_address=kwargs.get('ip_address'),
        device_trusted=kwargs.get('device_trusted', False),
        data_category=kwargs.get('data_category'),
        mfa_verified=kwargs.get('mfa_verified', False)
    )


# =============================================================================
# 与 action_guard 集成
# =============================================================================

def check_permission_before_action_guard(
    action: ActionType,
    context: ExecutionContext,
    enforcer: PermissionEnforcer
) -> PermissionCheckResponse:
    """
    在 action_guard 检查敏感度之前先检查权限边界
    
    如果权限边界明确拒绝,则操作永远不会到达确认对话框
    
    Args:
        action: 操作类型
        context: 执行上下文
        enforcer: 权限执行器
        
    Returns:
        PermissionCheckResponse: 检查结果
    """
    request = PermissionCheckRequest(
        action=action,
        context=context
    )
    
    response = enforcer.enforce(request)
    
    # 只有在非DENIED状态下才继续到action_guard的敏感度检查
    if response.result == CheckResult.DENIED:
        logger.info(f"权限边界拒绝: {action.name} @ {context.user_id}")
    
    return response


# =============================================================================
# CLI 接口
# =============================================================================

import argparse
import sys


def main():
    """
    权限管理CLI主入口
    
    支持以下命令:
    - assign: 分配角色给用户
    - revoke: 撤销用户角色
    - check: 检查用户权限
    - list: 列出角色权限
    - add-rule: 添加自定义权限规则
    - export: 导出用户权限配置
    """
    parser = argparse.ArgumentParser(
        description="权限边界与RBAC管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # assign 命令
    assign_parser = subparsers.add_parser('assign', help='分配角色给用户')
    assign_parser.add_argument('--user-id', required=True, help='用户ID')
    assign_parser.add_argument('--role', required=True, 
                              choices=['ADMIN', 'OPERATOR', 'VIEWER', 'GUEST'],
                              help='目标角色')
    
    # revoke 命令
    revoke_parser = subparsers.add_parser('revoke', help='撤销用户角色')
    revoke_parser.add_argument('--user-id', required=True, help='用户ID')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查用户权限')
    check_parser.add_argument('--user-id', required=True, help='用户ID')
    check_parser.add_argument('--action', required=True, 
                             choices=[a.name for a in ActionType],
                             help='操作类型')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出角色权限')
    list_parser.add_argument('--role', required=True,
                            choices=['ADMIN', 'OPERATOR', 'VIEWER', 'GUEST'],
                            help='角色名称')
    
    # add-rule 命令
    add_rule_parser = subparsers.add_parser('add-rule', help='添加自定义权限规则')
    add_rule_parser.add_argument('--user-id', required=True, help='用户ID')
    add_rule_parser.add_argument('--action', required=True,
                                choices=[a.name for a in ActionType],
                                help='操作类型')
    add_rule_parser.add_argument('--resource', required=True, help='资源路径')
    
    # export 命令
    export_parser = subparsers.add_parser('export', help='导出用户权限配置')
    export_parser.add_argument('--user-id', required=True, help='用户ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 初始化执行器
    enforcer = PermissionEnforcer()
    
    if args.command == 'assign':
        role = Role[args.role]
        success = enforcer.assign_role(args.user_id, role)
        print(json.dumps({
            'success': success,
            'user_id': args.user_id,
            'role': role.name,
            'message': f'角色 {role.name} 已分配给 {args.user_id}'
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'revoke':
        success = enforcer.revoke_role(args.user_id)
        print(json.dumps({
            'success': success,
            'user_id': args.user_id,
            'message': f'{args.user_id} 的角色已撤销' if success else '撤销失败'
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'check':
        action = ActionType[args.action]
        response = enforcer.check_action(args.user_id, action)
        print(json.dumps({
            'result': response.result.name,
            'reason': response.reason,
            'action': action.name,
            'action_description': action.description,
            'warnings': response.warnings
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'list':
        role = Role[args.role]
        boundary = enforcer.get_role_permissions(role)
        print(json.dumps({
            'role': role.name,
            'allowed_actions': [
                {'name': a.name, 'description': a.description, 'sensitivity': a.sensitivity}
                for a in boundary.allowed_actions
            ],
            'allowed_resources': list(boundary.allowed_resources),
            'conditions': [
                {
                    'time_range': c.time_range,
                    'ip_whitelist': c.ip_whitelist,
                    'device_trusted': c.device_trusted
                }
                for c in boundary.conditions
            ]
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'add-rule':
        action = ActionType[args.action]
        success = enforcer.add_permission_rule(args.user_id, action, args.resource)
        print(json.dumps({
            'success': success,
            'user_id': args.user_id,
            'action': action.name,
            'resource': args.resource,
            'message': '权限规则已添加' if success else '添加失败'
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'export':
        config = enforcer.export_permissions(args.user_id)
        print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
