"""
双重授权框架 (Double Authorization Framework)
============================================

本模块实现了"双重授权"安全原则,确保关键操作需要同时获得:
1. 平台授权 - 来自系统平台的官方许可和配额验证
2. 用户授权 - 来自终端用户的明确同意和身份验证

双重授权模式:
- AND模式: 平台授权和用户授权都必须通过
- OR模式: 平台授权或用户授权任一通过即可
- XOR模式: 一次性授权,使用后失效

Author: AGI Framework Security Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
import threading

# 配置日志记录器
logger = logging.getLogger(__name__)


class AuthorizationMode(Enum):
    """
    授权模式枚举
    
    定义双重授权的验证策略:
    - AND: 两个授权都必须成功 (最严格)
    - OR: 任一授权成功即可 (较宽松)
    - XOR: 一次性授权,使用后自动失效
    """
    AND = auto()  # 双重必须
    OR = auto()   # 任一即可
    XOR = auto()  # 一次性授权


class ActionSensitivity(Enum):
    """
    操作敏感度级别
    
    用于判断操作需要哪种级别的授权:
    - LOW: 无需额外授权
    - MEDIUM: 需要用户授权
    - HIGH: 需要双重授权(AND模式)
    - CRITICAL: 需要双重授权并记录审计日志
    """
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ConsentScope(Enum):
    """
    同意书有效期范围
    
    - ONCE: 仅本次操作有效
    - SESSION: 当前会话期间有效
    - ALWAYS: 永久有效,直至手动撤销
    """
    ONCE = auto()
    SESSION = auto()
    ALWAYS = auto()


@dataclass
class AgentAction:
    """
    代理操作数据结构
    
    描述一个需要授权的代理操作
    """
    action_type: str              # 操作类型 (如 "delete_files", "click", "send_message")
    params: Dict[str, Any]       # 操作参数字典
    sensitivity: ActionSensitivity = ActionSensitivity.LOW  # 敏感度级别
    resource_path: Optional[str] = None  # 涉及的资源路径
    platform: str = "unknown"     # 目标平台标识
    
    def get_params_hash(self) -> str:
        """
        获取参数字典的哈希值
        
        用于唯一标识此操作的具体参数组合
        
        Returns:
            str: SHA256哈希值的十六进制字符串
        """
        params_str = json.dumps(self.params, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(params_str.encode('utf-8')).hexdigest()


@dataclass
class ConsentRecord:
    """
    用户同意记录
    
    存储用户对特定操作的授权同意信息
    """
    consent_id: str                           # 同意记录唯一标识
    action_type: str                          # 被授权的操作类型
    params_hash: str                          # 操作参数的哈希值
    user_id: str                              # 授权用户ID
    granted_at: datetime                      # 授权时间
    expires_at: Optional[datetime] = None    # 过期时间(可选)
    scope: ConsentScope = ConsentScope.ONCE   # 授权范围
    revoked: bool = False                     # 是否已被撤销
    use_count: int = 0                        # 已使用次数
    max_uses: int = 1                         # 最大使用次数
    
    def is_valid(self) -> bool:
        """
        检查同意记录是否仍然有效
        
        Returns:
            bool: 如果有效返回True,否则返回False
        """
        if self.revoked:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        
        if self.scope != ConsentScope.ALWAYS and self.use_count >= self.max_uses:
            return False
        
        return True
    
    def mark_used(self) -> None:
        """标记同意记录已被使用一次"""
        self.use_count += 1
        if self.scope == ConsentScope.ONCE:
            self.max_uses = self.use_count


@dataclass
class AuthorizationContext:
    """
    授权上下文
    
    包含执行授权检查所需的完整上下文信息
    """
    action: AgentAction                        # 待授权的操作
    user_id: str                               # 请求授权的用户ID
    session_id: str                            # 当前会话ID
    platform: str                              # 目标平台标识
    requested_at: datetime = field(default_factory=datetime.now)  # 请求时间
    consent_record: Optional[ConsentRecord] = None  # 已有同意记录(如有)
    ip_address: Optional[str] = None           # 请求来源IP地址
    device_id: Optional[str] = None            # 请求设备标识


@dataclass
class AuthorizationResult:
    """
    授权结果
    
    返回授权检查的最终结果
    """
    granted: bool                              # 是否授予授权
    reason: str                                # 结果原因说明
    platform_auth_id: Optional[str] = None     # 平台授权ID(如有)
    user_auth_id: Optional[str] = None         # 用户授权ID(如有)
    expires_at: Optional[datetime] = None     # 授权过期时间
    required_mode: AuthorizationMode = AuthorizationMode.AND  # 需要的授权模式
    warnings: List[str] = field(default_factory=list)  # 警告信息列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'granted': self.granted,
            'reason': self.reason,
            'platform_auth_id': self.platform_auth_id,
            'user_auth_id': self.user_auth_id,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'required_mode': self.required_mode.name,
            'warnings': self.warnings
        }


@dataclass
class TokenInfo:
    """
    OAuth2令牌信息
    
    存储平台OAuth令牌的详细信息
    """
    token: str                                 # 访问令牌
    token_type: str = "Bearer"                  # 令牌类型
    expires_at: Optional[datetime] = None      # 过期时间
    scopes: Set[str] = field(default_factory=set)  # 授权范围列表
    refresh_token: Optional[str] = None        # 刷新令牌
    platform: str = "unknown"                  # 所属平台
    
    def is_expired(self) -> bool:
        """检查令牌是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


@dataclass 
class RateLimitInfo:
    """
    速率限制信息
    
    跟踪API调用的速率限制
    """
    action_type: str                           # 操作类型
    platform: str                              # 平台标识
    remaining: int                             # 剩余调用次数
    limit: int                                 # 总限制次数
    reset_at: datetime                         # 重置时间
    window_seconds: int = 3600                # 时间窗口秒数


class PlatformAuthorization:
    """
    平台授权验证器
    
    负责验证来自各平台的官方授权:
    - OAuth2令牌和作用域验证
    - API速率限制检查
    - 平台权限清单管理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化平台授权验证器
        
        Args:
            config_path: 配置文件路径(可选)
        """
        self._lock = threading.RLock()
        self._permission_manifest: Dict[str, Set[str]] = {}
        self._rate_limits: Dict[str, RateLimitInfo] = {}
        self._tokens: Dict[str, TokenInfo] = {}
        self._config_path = config_path
        
        # 加载配置
        self._load_config()
        
        # 注册默认权限清单
        self._register_default_permissions()
        
        logger.info("平台授权验证器初始化完成")
    
    def _load_config(self) -> None:
        """从配置文件加载设置"""
        if self._config_path and Path(self._config_path).exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._permission_manifest = {
                        platform: set(scopes) 
                        for platform, scopes in config.get('permissions', {}).items()
                    }
                    logger.info(f"已从 {self._config_path} 加载权限配置")
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}, 将使用默认配置")
    
    def _register_default_permissions(self) -> None:
        """注册默认的平台权限清单"""
        # 通用平台权限
        default_permissions = {
            'generic': {
                'screenshot', 'find_element', 'click', 'type',
                'scroll', 'wait', 'get_text', 'get_attribute'
            },
            'browser': {
                'navigate', 'refresh', 'back', 'forward',
                'switch_tab', 'close_tab', 'execute_script'
            },
            'file_system': {
                'read_file', 'write_file', 'delete_file', 
                'list_directory', 'create_directory'
            },
            'network': {
                'http_get', 'http_post', 'http_put', 'http_delete',
                'download', 'upload'
            }
        }
        
        for platform, permissions in default_permissions.items():
            if platform not in self._permission_manifest:
                self._permission_manifest[platform] = permissions
    
    def register_platform_permission(self, platform: str, permissions: Set[str]) -> None:
        """
        注册平台权限清单
        
        Args:
            platform: 平台标识符
            permissions: 该平台允许的操作集合
        """
        with self._lock:
            self._permission_manifest[platform] = permissions
            logger.info(f"已注册平台 {platform} 的权限清单: {permissions}")
    
    def check_platform_scope(
        self, 
        action: AgentAction, 
        platform: str = "generic"
    ) -> bool:
        """
        检查操作是否在平台授权范围内
        
        Args:
            action: 待检查的操作
            platform: 平台标识符
            
        Returns:
            bool: 如果操作在授权范围内返回True
        """
        with self._lock:
            allowed_actions = self._permission_manifest.get(platform, set())
            
            # 检查操作类型是否在允许列表中
            if action.action_type not in allowed_actions:
                logger.warning(
                    f"平台 {platform} 未授权操作: {action.action_type}, "
                    f"允许的操作: {allowed_actions}"
                )
                return False
            
            logger.debug(f"平台 {platform} 授权检查通过: {action.action_type}")
            return True
    
    def check_rate_limit(
        self, 
        action: AgentAction, 
        platform: str = "generic"
    ) -> bool:
        """
        检查操作是否超过平台速率限制
        
        Args:
            action: 待检查的操作
            platform: 平台标识符
            
        Returns:
            bool: 如果在速率限制内返回True
        """
        with self._lock:
            key = f"{platform}:{action.action_type}"
            rate_info = self._rate_limits.get(key)
            
            if rate_info is None:
                # 未配置速率限制,默认允许
                return True
            
            if datetime.now() >= rate_info.reset_at:
                # 时间窗口已过,重置计数
                rate_info.remaining = rate_info.limit
                rate_info.reset_at = datetime.now() + timedelta(
                    seconds=rate_info.window_seconds
                )
            
            if rate_info.remaining <= 0:
                logger.warning(
                    f"平台 {platform} 操作 {action.action_type} "
                    f"已达到速率限制 ({rate_info.limit}/{rate_info.window_seconds}s)"
                )
                return False
            
            rate_info.remaining -= 1
            return True
    
    def set_rate_limit(
        self,
        action_type: str,
        platform: str,
        limit: int,
        window_seconds: int = 3600
    ) -> None:
        """
        设置操作速率限制
        
        Args:
            action_type: 操作类型
            platform: 平台标识符
            limit: 时间窗口内允许的最大调用次数
            window_seconds: 时间窗口秒数,默认3600(1小时)
        """
        with self._lock:
            key = f"{platform}:{action_type}"
            self._rate_limits[key] = RateLimitInfo(
                action_type=action_type,
                platform=platform,
                remaining=limit,
                limit=limit,
                reset_at=datetime.now() + timedelta(seconds=window_seconds),
                window_seconds=window_seconds
            )
            logger.info(
                f"已设置平台 {platform} 操作 {action_type} "
                f"速率限制: {limit}/{window_seconds}s"
            )
    
    def validate_token(self, token: str, platform: str = "generic") -> bool:
        """
        验证OAuth2访问令牌
        
        Args:
            token: 待验证的令牌
            platform: 平台标识符
            
        Returns:
            bool: 如果令牌有效返回True
        """
        with self._lock:
            token_info = self._tokens.get(platform)
            
            if token_info is None:
                logger.warning(f"平台 {platform} 未注册令牌")
                return False
            
            if token_info.token != token:
                logger.warning(f"平台 {platform} 令牌不匹配")
                return False
            
            if token_info.is_expired():
                logger.warning(f"平台 {platform} 令牌已过期")
                return False
            
            return True
    
    def store_token(self, token_info: TokenInfo) -> str:
        """
        存储OAuth2令牌信息
        
        Args:
            token_info: 令牌信息对象
            
        Returns:
            str: 生成的令牌存储ID
        """
        with self._lock:
            token_id = str(uuid.uuid4())
            self._tokens[token_info.platform] = token_info
            logger.info(f"已存储平台 {token_info.platform} 的令牌, ID: {token_id}")
            return token_id
    
    def get_permission_manifest(self, platform: Optional[str] = None) -> Dict[str, Set[str]]:
        """
        获取平台权限清单
        
        Args:
            platform: 平台标识符,为None时返回所有平台的清单
            
        Returns:
            Dict[str, Set[str]]: 平台权限映射
        """
        with self._lock:
            if platform:
                return {platform: self._permission_manifest.get(platform, set())}
            return self._permission_manifest.copy()
    
    def authorize(self, action: AgentAction, context: AuthorizationContext) -> AuthorizationResult:
        """
        执行平台授权检查
        
        Args:
            action: 待授权的操作
            context: 授权上下文
            
        Returns:
            AuthorizationResult: 授权结果
        """
        platform = context.platform or "generic"
        
        # 检查平台范围
        scope_ok = self.check_platform_scope(action, platform)
        
        # 检查速率限制
        rate_ok = self.check_rate_limit(action, platform)
        
        if scope_ok and rate_ok:
            auth_id = str(uuid.uuid4())
            logger.info(f"平台授权成功: {action.action_type} @ {platform}, ID: {auth_id}")
            return AuthorizationResult(
                granted=True,
                reason=f"平台 {platform} 授权检查通过",
                platform_auth_id=auth_id
            )
        else:
            reasons = []
            if not scope_ok:
                reasons.append(f"操作 {action.action_type} 未在平台 {platform} 授权范围内")
            if not rate_ok:
                reasons.append(f"操作 {action.action_type} 超过平台 {platform} 速率限制")
            
            logger.warning(f"平台授权失败: {', '.join(reasons)}")
            return AuthorizationResult(
                granted=False,
                reason="; ".join(reasons),
                required_mode=AuthorizationMode.AND
            )


class UserAuthorization:
    """
    用户授权验证器
    
    负责用户身份验证和同意书管理:
    - 本地PIN/密码验证
    - 生物特征识别
    - 操作同意书捕获和管理
    - 同意书有效期管理
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化用户授权验证器
        
        Args:
            storage_path: 同意书存储路径(可选)
        """
        self._lock = threading.RLock()
        self._storage_path = storage_path or "/tmp/consent_records.json"
        self._consent_records: Dict[str, ConsentRecord] = {}
        self._user_credentials: Dict[str, Dict[str, Any]] = {}
        self._session_consents: Dict[str, Set[str]] = {}  # session_id -> consent_ids
        
        # 加载已存储的同意书
        self._load_consent_records()
        
        # 注册默认用户(测试用)
        self._register_default_users()
        
        logger.info("用户授权验证器初始化完成")
    
    def _load_consent_records(self) -> None:
        """从存储文件加载同意书记录"""
        if Path(self._storage_path).exists():
            try:
                with open(self._storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for record_data in data.get('records', []):
                        record = ConsentRecord(
                            consent_id=record_data['consent_id'],
                            action_type=record_data['action_type'],
                            params_hash=record_data['params_hash'],
                            user_id=record_data['user_id'],
                            granted_at=datetime.fromisoformat(record_data['granted_at']),
                            expires_at=datetime.fromisoformat(record_data['expires_at']) 
                                if record_data.get('expires_at') else None,
                            scope=ConsentScope[record_data.get('scope', 'ONCE')],
                            revoked=record_data.get('revoked', False),
                            use_count=record_data.get('use_count', 0),
                            max_uses=record_data.get('max_uses', 1)
                        )
                        self._consent_records[record.consent_id] = record
                logger.info(f"已加载 {len(self._consent_records)} 条同意书记录")
            except Exception as e:
                logger.warning(f"加载同意书记录失败: {e}")
    
    def _save_consent_records(self) -> None:
        """保存同意书记录到存储文件"""
        try:
            data = {
                'records': [
                    {
                        'consent_id': r.consent_id,
                        'action_type': r.action_type,
                        'params_hash': r.params_hash,
                        'user_id': r.user_id,
                        'granted_at': r.granted_at.isoformat(),
                        'expires_at': r.expires_at.isoformat() if r.expires_at else None,
                        'scope': r.scope.name,
                        'revoked': r.revoked,
                        'use_count': r.use_count,
                        'max_uses': r.max_uses
                    }
                    for r in self._consent_records.values()
                ]
            }
            
            Path(self._storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"已保存 {len(self._consent_records)} 条同意书记录")
        except Exception as e:
            logger.error(f"保存同意书记录失败: {e}")
    
    def _register_default_users(self) -> None:
        """注册默认用户(仅用于测试)"""
        # 在生产环境中应从安全的用户数据库加载
        self._user_credentials['admin'] = {
            'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
            'pin': '1234',
            'role': 'admin'
        }
        self._user_credentials['test_user'] = {
            'password_hash': hashlib.sha256('test123'.encode()).hexdigest(),
            'pin': '0000',
            'role': 'user'
        }
    
    def verify_user(
        self, 
        user_id: str, 
        credential: str, 
        auth_type: str = "password"
    ) -> bool:
        """
        验证用户身份
        
        Args:
            user_id: 用户标识符
            credential: 凭据(密码/PIN)
            auth_type: 认证类型 ("password", "pin", "biometric")
            
        Returns:
            bool: 如果验证通过返回True
        """
        with self._lock:
            user_creds = self._user_credentials.get(user_id)
            
            if user_creds is None:
                logger.warning(f"用户不存在: {user_id}")
                return False
            
            if auth_type == "password":
                cred_hash = hashlib.sha256(credential.encode()).hexdigest()
                if cred_hash != user_creds.get('password_hash'):
                    logger.warning(f"用户 {user_id} 密码验证失败")
                    return False
            elif auth_type == "pin":
                if credential != user_creds.get('pin'):
                    logger.warning(f"用户 {user_id} PIN验证失败")
                    return False
            elif auth_type == "biometric":
                # 生物特征验证 - 简化实现
                if credential != "biometric_verified":
                    logger.warning(f"用户 {user_id} 生物特征验证失败")
                    return False
            else:
                logger.warning(f"不支持的认证类型: {auth_type}")
                return False
            
            logger.info(f"用户 {user_id} 身份验证通过 (类型: {auth_type})")
            return True
    
    def register_user(
        self, 
        user_id: str, 
        password: str, 
        pin: str = "0000",
        role: str = "user"
    ) -> bool:
        """
        注册新用户
        
        Args:
            user_id: 用户标识符
            password: 密码
            pin: PIN码
            role: 用户角色
            
        Returns:
            bool: 注册是否成功
        """
        with self._lock:
            if user_id in self._user_credentials:
                logger.warning(f"用户已存在: {user_id}")
                return False
            
            self._user_credentials[user_id] = {
                'password_hash': hashlib.sha256(password.encode()).hexdigest(),
                'pin': pin,
                'role': role
            }
            
            logger.info(f"已注册新用户: {user_id}, 角色: {role}")
            return True
    
    def capture_consent(
        self,
        action: AgentAction,
        user_id: str,
        scope: ConsentScope = ConsentScope.ONCE,
        max_uses: int = 1,
        expires_in_hours: Optional[float] = None
    ) -> ConsentRecord:
        """
        捕获用户同意
        
        记录用户对特定操作的授权同意
        
        Args:
            action: 被同意的操作
            user_id: 授权用户ID
            scope: 同意范围
            max_uses: 最大使用次数(仅ONCE和SESSION模式有效)
            expires_in_hours: 过期时间(小时),None表示不过期
            
        Returns:
            ConsentRecord: 创建的同意记录
        """
        with self._lock:
            consent_id = str(uuid.uuid4())
            
            expires_at = None
            if expires_in_hours is not None:
                expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            
            consent = ConsentRecord(
                consent_id=consent_id,
                action_type=action.action_type,
                params_hash=action.get_params_hash(),
                user_id=user_id,
                granted_at=datetime.now(),
                expires_at=expires_at,
                scope=scope,
                max_uses=max_uses
            )
            
            self._consent_records[consent_id] = consent
            
            # 如果是SESSION范围,记录到会话
            if scope == ConsentScope.SESSION:
                session_key = f"{user_id}_current"
                if session_key not in self._session_consents:
                    self._session_consents[session_key] = set()
                self._session_consents[session_key].add(consent_id)
            
            self._save_consent_records()
            
            logger.info(
                f"已捕获用户 {user_id} 对操作 {action.action_type} 的同意, "
                f"同意ID: {consent_id}, 范围: {scope.name}"
            )
            
            return consent
    
    def check_consent_validity(
        self,
        action: AgentAction,
        user_id: str,
        consent_id: Optional[str] = None
    ) -> Optional[ConsentRecord]:
        """
        检查用户同意是否有效
        
        Args:
            action: 待检查的操作
            user_id: 用户ID
            consent_id: 同意记录ID(可选,如果知道则直接验证)
            
        Returns:
            Optional[ConsentRecord]: 有效的同意记录,如果没有则返回None
        """
        with self._lock:
            # 如果提供了同意ID,直接验证
            if consent_id:
                record = self._consent_records.get(consent_id)
                if record and record.is_valid() and record.user_id == user_id:
                    return record
                return None
            
            # 否则查找匹配的同意记录
            action_hash = action.get_params_hash()
            
            for record in self._consent_records.values():
                if (record.user_id == user_id and 
                    record.action_type == action.action_type and
                    record.is_valid()):
                    
                    # 对于精确匹配的检查参数哈希
                    if record.params_hash == action_hash:
                        return record
                    
                    # 或者检查是否是通配符(空哈希或通配符标记)
                    if record.params_hash == "" or record.params_hash == "*":
                        return record
            
            return None
    
    def revoke_consent(self, consent_id: str, user_id: Optional[str] = None) -> bool:
        """
        撤销用户同意
        
        Args:
            consent_id: 同意记录ID
            user_id: 用户ID(可选,用于验证权限)
            
        Returns:
            bool: 撤销是否成功
        """
        with self._lock:
            record = self._consent_records.get(consent_id)
            
            if record is None:
                logger.warning(f"同意记录不存在: {consent_id}")
                return False
            
            if user_id and record.user_id != user_id:
                logger.warning(f"用户 {user_id} 无权撤销 {record.user_id} 的同意")
                return False
            
            record.revoked = True
            self._save_consent_records()
            
            logger.info(f"已撤销同意记录: {consent_id}")
            return True
    
    def list_user_consents(
        self, 
        user_id: str, 
        include_expired: bool = False
    ) -> List[ConsentRecord]:
        """
        列出用户的所有同意记录
        
        Args:
            user_id: 用户ID
            include_expired: 是否包含已过期的记录
            
        Returns:
            List[ConsentRecord]: 同意记录列表
        """
        with self._lock:
            records = [
                r for r in self._consent_records.values()
                if r.user_id == user_id
            ]
            
            if not include_expired:
                records = [r for r in records if r.is_valid()]
            
            return records
    
    def use_consent(self, consent_id: str) -> bool:
        """
        使用一次同意记录
        
        用于XOR模式和ONCE模式的计数
        
        Args:
            consent_id: 同意记录ID
            
        Returns:
            bool: 使用是否成功
        """
        with self._lock:
            record = self._consent_records.get(consent_id)
            
            if record is None:
                return False
            
            record.mark_used()
            self._save_consent_records()
            
            logger.debug(f"同意记录已使用: {consent_id}, 使用次数: {record.use_count}")
            return True
    
    def authorize(self, action: AgentAction, context: AuthorizationContext) -> AuthorizationResult:
        """
        执行用户授权检查
        
        Args:
            action: 待授权的操作
            context: 授权上下文
            
        Returns:
            AuthorizationResult: 授权结果
        """
        # 首先检查是否已有有效的同意记录
        existing_consent = self.check_consent_validity(
            action, 
            context.user_id,
            context.consent_record.consent_id if context.consent_record else None
        )
        
        if existing_consent:
            # 使用同意记录(如果是一次性的)
            if existing_consent.scope == ConsentScope.ONCE:
                self.use_consent(existing_consent.consent_id)
            
            logger.info(
                f"用户授权成功: {context.user_id} @ {action.action_type}, "
                f"同意ID: {existing_consent.consent_id}"
            )
            
            return AuthorizationResult(
                granted=True,
                reason=f"用户 {context.user_id} 已授权",
                user_auth_id=existing_consent.consent_id,
                expires_at=existing_consent.expires_at
            )
        
        logger.warning(
            f"用户授权失败: {context.user_id} @ {action.action_type}, "
            f"未找到有效同意记录"
        )
        
        return AuthorizationResult(
            granted=False,
            reason=f"用户 {context.user_id} 未授权操作 {action.action_type}",
            required_mode=AuthorizationMode.AND
        )


class DoubleAuthorization:
    """
    双重授权管理器
    
    整合平台授权和用户授权,提供统一的双重授权接口
    
    支持三种授权模式:
    - AND: 平台授权和用户授权都必须通过
    - OR: 平台授权或用户授权任一通过即可
    - XOR: 一次性授权,使用后自动失效
    """
    
    # 敏感度与授权模式的映射
    SENSITIVITY_MODE_MAP = {
        ActionSensitivity.LOW: None,          # 无需双重授权
        ActionSensitivity.MEDIUM: AuthorizationMode.OR,  # 任一即可
        ActionSensitivity.HIGH: AuthorizationMode.AND,   # 必须双重
        ActionSensitivity.CRITICAL: AuthorizationMode.AND  # 必须双重+审计
    }
    
    def __init__(
        self,
        platform_auth: Optional[PlatformAuthorization] = None,
        user_auth: Optional[UserAuthorization] = None,
        config_path: Optional[str] = None
    ):
        """
        初始化双重授权管理器
        
        Args:
            platform_auth: 平台授权验证器实例
            user_auth: 用户授权验证器实例
            config_path: 配置文件路径
        """
        self._platform_auth = platform_auth or PlatformAuthorization(config_path)
        self._user_auth = user_auth or UserAuthorization()
        self._lock = threading.RLock()
        
        # 授权审计日志
        self._audit_log: List[Dict[str, Any]] = []
        
        logger.info("双重授权管理器初始化完成")
    
    def authorize(
        self,
        action: AgentAction,
        context: AuthorizationContext,
        mode: Optional[AuthorizationMode] = None
    ) -> AuthorizationResult:
        """
        执行双重授权检查
        
        这是双重授权的核心入口方法
        
        Args:
            action: 待授权的操作
            context: 授权上下文
            mode: 授权模式(可选,默认根据敏感度自动确定)
            
        Returns:
            AuthorizationResult: 授权结果
        """
        with self._lock:
            # 自动确定授权模式
            if mode is None:
                mode = self.SENSITIVITY_MODE_MAP.get(
                    action.sensitivity, 
                    AuthorizationMode.AND
                )
            
            # 对于LOW敏感度操作,直接通过
            if mode is None:
                return AuthorizationResult(
                    granted=True,
                    reason="低敏感度操作,无需额外授权",
                    required_mode=AuthorizationMode.AND
                )
            
            # 执行平台授权
            platform_result = self._platform_auth.authorize(action, context)
            
            # 执行用户授权
            user_result = self._user_auth.authorize(action, context)
            
            # 根据模式确定最终结果
            granted = False
            reason = ""
            warnings = []
            
            if mode == AuthorizationMode.AND:
                granted = platform_result.granted and user_result.granted
                if not granted:
                    reasons = []
                    if not platform_result.granted:
                        reasons.append(f"平台授权失败: {platform_result.reason}")
                    if not user_result.granted:
                        reasons.append(f"用户授权失败: {user_result.reason}")
                    reason = "; ".join(reasons)
            elif mode == AuthorizationMode.OR:
                granted = platform_result.granted or user_result.granted
                if not granted:
                    reason = f"平台和用户授权均未通过"
                elif platform_result.granted and not user_result.granted:
                    warnings.append("仅平台授权通过,用户授权失败")
                elif not platform_result.granted and user_result.granted:
                    warnings.append("仅用户授权通过,平台授权失败")
            elif mode == AuthorizationMode.XOR:
                granted = platform_result.granted != user_result.granted
                if not granted:
                    reason = "XOR模式要求只有一个授权通过"
                else:
                    reason = "一次性授权条件满足"
            
            # 记录审计日志
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action.action_type,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'mode': mode.name,
                'granted': granted,
                'platform_granted': platform_result.granted,
                'user_granted': user_result.granted,
                'reason': reason or "授权成功"
            }
            self._audit_log.append(audit_entry)
            
            if granted:
                logger.info(
                    f"双重授权成功: {action.action_type} @ {context.user_id}, "
                    f"模式: {mode.name}"
                )
            else:
                logger.warning(
                    f"双重授权失败: {action.action_type} @ {context.user_id}, "
                    f"原因: {reason}"
                )
            
            return AuthorizationResult(
                granted=granted,
                reason=reason or "双重授权检查通过",
                platform_auth_id=platform_result.platform_auth_id,
                user_auth_id=user_result.user_auth_id,
                expires_at=user_result.expires_at,
                required_mode=mode,
                warnings=warnings
            )
    
    def request_authorization(
        self,
        action: AgentAction,
        context: AuthorizationContext
    ) -> AuthorizationResult:
        """
        请求授权(如果需要用户同意)
        
        当缺少用户同意时,此方法返回需要用户同意的提示
        
        Args:
            action: 待授权的操作
            context: 授权上下文
            
        Returns:
            AuthorizationResult: 授权结果
        """
        result = self.authorize(action, context)
        
        if not result.granted and "用户授权失败" in result.reason:
            result.warnings.append("需要用户同意此操作")
        
        return result
    
    def grant_user_consent(
        self,
        action: AgentAction,
        user_id: str,
        scope: ConsentScope = ConsentScope.ONCE,
        max_uses: int = 1,
        expires_in_hours: Optional[float] = None
    ) -> ConsentRecord:
        """
        授予用户同意
        
        用于CLI界面或确认对话框后的同意授予
        
        Args:
            action: 被同意的操作
            user_id: 用户ID
            scope: 同意范围
            max_uses: 最大使用次数
            expires_in_hours: 过期时间(小时)
            
        Returns:
            ConsentRecord: 创建的同意记录
        """
        consent = self._user_auth.capture_consent(
            action=action,
            user_id=user_id,
            scope=scope,
            max_uses=max_uses,
            expires_in_hours=expires_in_hours
        )
        
        logger.info(f"用户 {user_id} 授予了操作 {action.action_type} 的同意")
        
        return consent
    
    def revoke_authorization(self, consent_id: str) -> bool:
        """
        撤销授权
        
        Args:
            consent_id: 同意记录ID
            
        Returns:
            bool: 撤销是否成功
        """
        result = self._user_auth.revoke_consent(consent_id)
        
        if result:
            logger.info(f"已撤销授权: {consent_id}")
        
        return result
    
    def get_audit_log(
        self, 
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取授权审计日志
        
        Args:
            user_id: 用户ID(可选,过滤特定用户)
            limit: 返回的最大记录数
            
        Returns:
            List[Dict[str, Any]]: 审计日志条目列表
        """
        with self._lock:
            logs = self._audit_log
            
            if user_id:
                logs = [l for l in logs if l.get('user_id') == user_id]
            
            return logs[-limit:]
    
    def get_user_permissions_summary(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户权限摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 权限摘要信息
        """
        consents = self._user_auth.list_user_consents(user_id, include_expired=False)
        
        return {
            'user_id': user_id,
            'active_consents': len(consents),
            'consent_summary': [
                {
                    'action_type': c.action_type,
                    'scope': c.scope.name,
                    'expires_at': c.expires_at.isoformat() if c.expires_at else None,
                    'remaining_uses': c.max_uses - c.use_count if c.scope != ConsentScope.ALWAYS else None
                }
                for c in consents
            ]
        }


def create_authorization_context(
    action: AgentAction,
    user_id: str,
    session_id: str,
    platform: str = "generic",
    **kwargs
) -> AuthorizationContext:
    """
    便捷函数: 创建授权上下文
    
    Args:
        action: 代理操作
        user_id: 用户ID
        session_id: 会话ID
        platform: 平台标识符
        **kwargs: 其他可选参数
        
    Returns:
        AuthorizationContext: 授权上下文对象
    """
    return AuthorizationContext(
        action=action,
        user_id=user_id,
        session_id=session_id,
        platform=platform,
        requested_at=datetime.now(),
        ip_address=kwargs.get('ip_address'),
        device_id=kwargs.get('device_id')
    )


# =============================================================================
# CLI 接口
# =============================================================================

import argparse
import sys


def main():
    """
    双重授权CLI主入口
    
    支持以下命令:
    - grant: 授予用户同意
    - revoke: 撤销授权
    - list: 列出用户的授权记录
    - check: 检查操作是否已授权
    """
    parser = argparse.ArgumentParser(
        description="双重授权管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 授予用户对 delete_files 操作的会话级同意
  python -m security.double_auth grant --action delete_files --scope session --user-id admin

  # 撤销特定同意记录
  python -m security.double_auth revoke --consent-id xxx

  # 列出用户的所有同意记录
  python -m security.double_auth list --user-id admin

  # 检查操作是否已授权
  python -m security.double_auth check --action click --params '{"x":100,"y":200}'
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # grant 命令
    grant_parser = subparsers.add_parser('grant', help='授予用户同意')
    grant_parser.add_argument('--action', required=True, help='操作类型')
    grant_parser.add_argument('--scope', default='once', 
                             choices=['once', 'session', 'always'],
                             help='同意范围')
    grant_parser.add_argument('--user-id', required=True, help='用户ID')
    grant_parser.add_argument('--max-uses', type=int, default=1, help='最大使用次数')
    grant_parser.add_argument('--expires-hours', type=float, help='过期时间(小时)')
    grant_parser.add_argument('--params', default='{}', help='操作参数字典(JSON)')
    
    # revoke 命令
    revoke_parser = subparsers.add_parser('revoke', help='撤销授权')
    revoke_parser.add_argument('--consent-id', required=True, help='同意记录ID')
    revoke_parser.add_argument('--user-id', help='用户ID(可选,用于验证)')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出授权记录')
    list_parser.add_argument('--user-id', required=True, help='用户ID')
    list_parser.add_argument('--include-expired', action='store_true',
                            help='包含已过期的记录')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查授权状态')
    check_parser.add_argument('--action', required=True, help='操作类型')
    check_parser.add_argument('--user-id', default='cli_user', help='用户ID')
    check_parser.add_argument('--session-id', default='cli_session', help='会话ID')
    check_parser.add_argument('--params', default='{}', help='操作参数字典(JSON)')
    check_parser.add_argument('--platform', default='generic', help='平台标识符')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 初始化管理器
    auth_manager = DoubleAuthorization()
    
    if args.command == 'grant':
        # 授予同意
        action = AgentAction(
            action_type=args.action,
            params=json.loads(args.params)
        )
        
        scope_map = {
            'once': ConsentScope.ONCE,
            'session': ConsentScope.SESSION,
            'always': ConsentScope.ALWAYS
        }
        
        consent = auth_manager.grant_user_consent(
            action=action,
            user_id=args.user_id,
            scope=scope_map[args.scope],
            max_uses=args.max_uses,
            expires_in_hours=args.expires_hours
        )
        
        print(json.dumps({
            'success': True,
            'consent_id': consent.consent_id,
            'action': consent.action_type,
            'scope': consent.scope.name,
            'granted_at': consent.granted_at.isoformat()
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'revoke':
        # 撤销授权
        success = auth_manager.revoke_authorization(args.consent_id)
        
        print(json.dumps({
            'success': success,
            'consent_id': args.consent_id,
            'message': '授权已撤销' if success else '撤销失败'
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'list':
        # 列出授权记录
        user_auth = UserAuthorization()
        consents = user_auth.list_user_consents(
            args.user_id, 
            include_expired=args.include_expired
        )
        
        print(json.dumps({
            'user_id': args.user_id,
            'total': len(consents),
            'consents': [
                {
                    'consent_id': c.consent_id,
                    'action_type': c.action_type,
                    'scope': c.scope.name,
                    'granted_at': c.granted_at.isoformat(),
                    'expires_at': c.expires_at.isoformat() if c.expires_at else None,
                    'is_valid': c.is_valid(),
                    'remaining_uses': c.max_uses - c.use_count if c.scope != ConsentScope.ALWAYS else None
                }
                for c in consents
            ]
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'check':
        # 检查授权状态
        params = json.loads(args.params)
        action = AgentAction(
            action_type=args.action,
            params=params,
            platform=args.platform
        )
        
        context = create_authorization_context(
            action=action,
            user_id=args.user_id,
            session_id=args.session_id,
            platform=args.platform
        )
        
        result = auth_manager.authorize(action, context)
        
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
