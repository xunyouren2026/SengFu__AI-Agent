"""
能力最小化控制框架 (Capability Minimization Control Framework)
=============================================================

本模块实现了"能力最小化"安全原则,通过以下机制确保代理和用户
只能使用其任务所需的最小功能集:

1. Capability - 能力级别枚举
   - READ_ONLY: 只读操作
   - LIMITED_WRITE: 有限的写入操作
   - FULL_ACCESS: 完全访问
   - ADMIN: 管理员权限

2. CapabilityContext - 能力上下文
   - 会话级别的能力配置
   - 允许/禁止的工具列表
   - API调用限制
   - 数据访问范围

3. CapabilityRegistry - 能力注册表
   - 管理会话的能力分配
   - 能力降级和撤销

4. CapabilityBoundary - 能力边界定义
   - 工具限制
   - API限流
   - 数据范围控制

5. DangerousOperationReducer - 危险操作降级器
   - 将危险操作转换为安全替代方案
   - 添加额外的审批步骤

6. CapabilityEscalation - 能力提升管理
   - 临时能力提升请求和审批
   - 关键操作自动提升

Author: AGI Framework Security Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

# 配置日志记录器
logger = logging.getLogger(__name__)


class Capability(Enum):
    """
    能力级别枚举
    
    定义代理和用户的能力级别,从低到高排列:
    - READ_ONLY: 只读模式,仅允许查看和截图操作
    - LIMITED_WRITE: 有限写入,允许基本交互操作
    - FULL_ACCESS: 完全访问,允许大部分操作但有限流
    - ADMIN: 管理员权限,允许所有操作且无限制
    
    遵循"能力最小化"原则:应始终使用完成任务所需的最低能力级别
    """
    READ_ONLY = auto()      # 只读
    LIMITED_WRITE = auto()  # 有限写入
    FULL_ACCESS = auto()    # 完全访问
    ADMIN = auto()          # 管理员
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def priority(self) -> int:
        """获取能力优先级,数字越大权限越高"""
        priority_map = {
            Capability.READ_ONLY: 10,
            Capability.LIMITED_WRITE: 30,
            Capability.FULL_ACCESS: 60,
            Capability.ADMIN: 100
        }
        return priority_map.get(self, 0)
    
    def can_escalate_to(self, target: 'Capability') -> bool:
        """
        检查是否可以从当前能力提升到目标能力
        
        Args:
            target: 目标能力级别
            
        Returns:
            bool: 如果可以提升返回True
        """
        # 只能逐级提升,不能跳级
        current_level = self.priority
        target_level = target.priority
        
        if target_level <= current_level:
            return False
        
        # READ_ONLY -> LIMITED_WRITE -> FULL_ACCESS -> ADMIN
        allowed_escalations = {
            Capability.READ_ONLY: {Capability.LIMITED_WRITE},
            Capability.LIMITED_WRITE: {Capability.FULL_ACCESS},
            Capability.FULL_ACCESS: {Capability.ADMIN},
            Capability.ADMIN: set()
        }
        
        return target in allowed_escalations.get(self, set())


class ToolCategory(Enum):
    """
    工具类别枚举
    
    用于按类别管理工具访问权限
    """
    READ = auto()        # 读取类工具
    INTERACT = auto()     # 交互类工具
    WRITE = auto()       # 写入类工具
    SYSTEM = auto()      # 系统类工具
    DANGEROUS = auto()   # 危险类工具


@dataclass
class ToolInfo:
    """
    工具信息
    
    描述单个工具的详细信息
    """
    name: str                                      # 工具名称
    category: ToolCategory                         # 工具类别
    capability_required: Capability                 # 最低所需能力
    rate_limit_per_hour: int                        # 每小时调用限制
    requires_approval: bool                        # 是否需要额外审批
    is_dangerous: bool                             # 是否为危险工具
    description: str = ""                          # 工具描述
    
    def is_read_only(self) -> bool:
        """判断工具是否为只读类型"""
        return self.category == ToolCategory.READ


@dataclass
class CapabilityContext:
    """
    能力上下文
    
    存储会话的能力配置信息
    """
    session_id: str                                           # 会话ID
    user_id: str                                               # 用户ID
    current_capability: Capability                            # 当前能力级别
    allowed_tools: Set[str] = field(default_factory=set)     # 允许使用的工具
    forbidden_tools: Set[str] = field(default_factory=set)  # 禁止使用的工具
    max_api_calls_per_hour: int = 50                          # 每小时最大API调用次数
    data_access_scope: Set[str] = field(default_factory=set) # 数据访问范围
    granted_at: datetime = field(default_factory=datetime.now)  # 授权时间
    expires_at: Optional[datetime] = None                    # 过期时间
    
    def is_expired(self) -> bool:
        """检查能力上下文是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        检查工具是否允许使用
        
        Args:
            tool_name: 工具名称
            
        Returns:
            bool: 如果允许返回True
        """
        if tool_name in self.forbidden_tools:
            return False
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'current_capability': self.current_capability.name,
            'allowed_tools': list(self.allowed_tools),
            'forbidden_tools': list(self.forbidden_tools),
            'max_api_calls_per_hour': self.max_api_calls_per_hour,
            'data_access_scope': list(self.data_access_scope),
            'granted_at': self.granted_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class CapabilityBoundary:
    """
    能力边界定义
    
    定义每种能力级别的边界配置
    """
    capability: Capability                              # 能力级别
    
    # 允许的工具列表
    allowed_tools: Set[str] = field(default_factory=set)
    
    # API调用限制
    api_rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # 数据访问范围
    data_scope: Set[str] = field(default_factory=set)
    
    @classmethod
    def create_default_boundaries(cls) -> Dict[Capability, 'CapabilityBoundary']:
        """
        创建默认的能力边界配置
        
        Returns:
            Dict[Capability, CapabilityBoundary]: 能力边界字典
        """
        boundaries = {}
        
        # READ_ONLY: 仅截图、查找元素、等待、滚动
        boundaries[Capability.READ_ONLY] = cls(
            capability=Capability.READ_ONLY,
            allowed_tools={
                'screenshot', 'find_element', 'wait', 'scroll',
                'get_text', 'get_attribute', 'get_screenshot'
            },
            api_rate_limits={
                'total': 50,           # 总调用次数
                'screenshot': 100,     # 截图次数
                'find_element': 200    # 查找元素次数
            },
            data_scope={'read_only'}
        )
        
        # LIMITED_WRITE: + 点击、输入、复制、粘贴
        boundaries[Capability.LIMITED_WRITE] = cls(
            capability=Capability.LIMITED_WRITE,
            allowed_tools={
                'screenshot', 'find_element', 'wait', 'scroll',
                'get_text', 'get_attribute', 'get_screenshot',
                'click', 'type', 'copy', 'paste', 'hotkey_press'
            },
            api_rate_limits={
                'total': 20,
                'click': 50,
                'type': 30,
                'screenshot': 50
            },
            data_scope={'read_only', 'write_temp'}
        )
        
        # FULL_ACCESS: + 启动/关闭应用、快捷键
        boundaries[Capability.FULL_ACCESS] = cls(
            capability=Capability.FULL_ACCESS,
            allowed_tools={
                'screenshot', 'find_element', 'wait', 'scroll',
                'get_text', 'get_attribute', 'get_screenshot',
                'click', 'type', 'copy', 'paste', 'hotkey_press',
                'launch_app', 'close_app', 'hotkey', 'execute_script'
            },
            api_rate_limits={
                'total': 10,
                'click': 30,
                'type': 20,
                'launch_app': 10,
                'execute_script': 5
            },
            data_scope={'read_only', 'write_temp', 'write_persistent'}
        )
        
        # ADMIN: 所有操作,无限制
        boundaries[Capability.ADMIN] = cls(
            capability=Capability.ADMIN,
            allowed_tools={'*'},  # 所有工具
            api_rate_limits={},   # 无限制
            data_scope={'*'}       # 所有数据
        )
        
        return boundaries


@dataclass
class DangerousOperationRule:
    """
    危险操作降级规则
    
    定义如何将危险操作降级为安全替代方案
    """
    original_action: str                               # 原始危险操作
    safe_action: str                                   # 安全替代操作
    requires_approval: bool                           # 是否需要审批
    add_to_trash: bool = True                         # 是否移动到回收站
    sandboxed: bool = True                            # 是否沙箱执行
    description: str = ""                             # 规则描述


class DangerousOperationReducer:
    """
    危险操作降级器
    
    将危险操作转换为安全的替代方案,实现"能力最小化"
    """
    
    def __init__(self):
        """初始化危险操作降级器"""
        self._rules: Dict[str, DangerousOperationRule] = {}
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """注册默认的降级规则"""
        # DELETE_FILES -> 移动到 ~/.trash/
        self._rules['DELETE_FILES'] = DangerousOperationRule(
            original_action='DELETE_FILES',
            safe_action='MOVE_TO_TRASH',
            requires_approval=True,
            add_to_trash=True,
            description='将删除操作转换为移动到回收站'
        )
        
        # FORMAT_DISK -> 完全阻止
        self._rules['FORMAT_DISK'] = DangerousOperationRule(
            original_action='FORMAT_DISK',
            safe_action='BLOCKED',
            requires_approval=False,
            description='格式化磁盘操作被完全阻止'
        )
        
        # SEND_MESSAGES_BULK -> 添加审批步骤
        self._rules['SEND_MESSAGES_BULK'] = DangerousOperationRule(
            original_action='SEND_MESSAGES_BULK',
            safe_action='SEND_MESSAGES_SINGLE',
            requires_approval=True,
            description='批量消息发送需要逐条审批'
        )
        
        # API_CALL_EXTERNAL -> 沙箱执行
        self._rules['API_CALL_EXTERNAL'] = DangerousOperationRule(
            original_action='API_CALL_EXTERNAL',
            safe_action='API_CALL_SANDBOXED',
            requires_approval=False,
            sandboxed=True,
            description='外部API调用必须在沙箱中执行'
        )
        
        # EXECUTE_ADMIN_SCRIPT -> 需要审批
        self._rules['EXECUTE_ADMIN_SCRIPT'] = DangerousOperationRule(
            original_action='EXECUTE_ADMIN_SCRIPT',
            safe_action='EXECUTE_ADMIN_SCRIPT_SANDBOXED',
            requires_approval=True,
            sandboxed=True,
            description='管理员脚本需要沙箱执行和审批'
        )
        
        # MODIFY_SYSTEM_CONFIG -> 需要审批
        self._rules['MODIFY_SYSTEM_CONFIG'] = DangerousOperationRule(
            original_action='MODIFY_SYSTEM_CONFIG',
            safe_action='CREATE_BACKUP_BEFORE_MODIFY',
            requires_approval=True,
            description='系统配置修改前需要创建备份'
        )
        
        # DELETE_DATABASE -> 完全阻止
        self._rules['DELETE_DATABASE'] = DangerousOperationRule(
            original_action='DELETE_DATABASE',
            safe_action='BLOCKED',
            requires_approval=False,
            description='数据库删除操作被完全阻止'
        )
        
        logger.info(f"已注册 {len(self._rules)} 条危险操作降级规则")
    
    def add_rule(self, rule: DangerousOperationRule) -> None:
        """
        添加降级规则
        
        Args:
            rule: 危险操作降级规则
        """
        self._rules[rule.original_action] = rule
        logger.info(f"已添加降级规则: {rule.original_action} -> {rule.safe_action}")
    
    def get_rule(self, action: str) -> Optional[DangerousOperationRule]:
        """
        获取操作对应的降级规则
        
        Args:
            action: 操作名称
            
        Returns:
            Optional[DangerousOperationRule]: 降级规则
        """
        return self._rules.get(action)
    
    def reduce(self, action: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any], bool]:
        """
        降级危险操作
        
        Args:
            action: 操作名称
            params: 操作参数
            
        Returns:
            Tuple[str, Dict[str, Any], bool]: (降级后操作, 新参数, 是否需要审批)
        """
        rule = self._rules.get(action)
        
        if rule is None:
            # 无降级规则,保持原样
            return action, params, False
        
        if rule.safe_action == 'BLOCKED':
            logger.warning(f"操作 {action} 被危险操作降级器阻止")
            raise PermissionError(f"危险操作 {action} 被安全策略阻止")
        
        # 应用降级规则
        new_params = params.copy()
        
        if rule.add_to_trash:
            # 添加回收站路径参数
            new_params['trash_path'] = '~/.trash'
            new_params['original_path'] = params.get('path', '')
            new_params['path'] = '~/.trash/' + params.get('path', '').split('/')[-1]
        
        if rule.sandboxed:
            # 标记为沙箱执行
            new_params['sandboxed'] = True
        
        if action == 'SEND_MESSAGES_BULK':
            # 批量操作降级为单条
            new_params['single_mode'] = True
        
        logger.info(f"操作已降级: {action} -> {rule.safe_action}")
        
        return rule.safe_action, new_params, rule.requires_approval
    
    def is_action_dangerous(self, action: str) -> bool:
        """
        检查操作是否为危险操作
        
        Args:
            action: 操作名称
            
        Returns:
            bool: 如果为危险操作返回True
        """
        rule = self._rules.get(action)
        return rule is not None and rule.safe_action != 'BLOCKED'
    
    def is_action_blocked(self, action: str) -> bool:
        """
        检查操作是否被完全阻止
        
        Args:
            action: 操作名称
            
        Returns:
            bool: 如果被阻止返回True
        """
        rule = self._rules.get(action)
        return rule is not None and rule.safe_action == 'BLOCKED'


@dataclass
class EscalationRequest:
    """
    能力提升请求
    """
    escalation_id: str                                 # 请求ID
    session_id: str                                     # 会话ID
    user_id: str                                        # 用户ID
    current_capability: Capability                     # 当前能力
    requested_capability: Capability                    # 请求的能力
    reason: str                                         # 请求原因
    requested_at: datetime = field(default_factory=datetime.now)  # 请求时间
    status: str = "pending"                             # 状态: pending, approved, rejected
    approved_by: Optional[str] = None                   # 审批人
    approved_at: Optional[datetime] = None              # 审批时间


class CapabilityRegistry:
    """
    能力注册表
    
    管理会话的能力分配和查询
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化能力注册表
        
        Args:
            storage_path: 存储路径
        """
        self._lock = threading.RLock()
        self._storage_path = storage_path or "/tmp/capability_registry.json"
        self._contexts: Dict[str, CapabilityContext] = {}  # session_id -> context
        self._user_sessions: Dict[str, Set[str]] = {}       # user_id -> session_ids
        self._default_capabilities = CapabilityBoundary.create_default_boundaries()
        
        # API调用计数器
        self._api_call_counts: Dict[str, List[datetime]] = {}  # session_id -> 调用时间列表
        
        # 加载数据
        self._load_data()
        
        logger.info("能力注册表初始化完成")
    
    def _load_data(self) -> None:
        """从存储文件加载数据"""
        if Path(self._storage_path).exists():
            try:
                with open(self._storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 加载能力上下文
                    for ctx_data in data.get('contexts', []):
                        ctx = CapabilityContext(
                            session_id=ctx_data['session_id'],
                            user_id=ctx_data['user_id'],
                            current_capability=Capability[ctx_data['current_capability']],
                            allowed_tools=set(ctx_data.get('allowed_tools', [])),
                            forbidden_tools=set(ctx_data.get('forbidden_tools', [])),
                            max_api_calls_per_hour=ctx_data.get('max_api_calls_per_hour', 50),
                            data_access_scope=set(ctx_data.get('data_access_scope', [])),
                            granted_at=datetime.fromisoformat(ctx_data['granted_at']),
                            expires_at=datetime.fromisoformat(ctx_data['expires_at']) 
                                if ctx_data.get('expires_at') else None
                        )
                        self._contexts[ctx.session_id] = ctx
                    logger.info(f"已从存储加载 {len(self._contexts)} 个能力上下文")
            except Exception as e:
                logger.warning(f"加载能力数据失败: {e}")
    
    def _save_data(self) -> None:
        """保存数据到存储文件"""
        try:
            data = {
                'contexts': [ctx.to_dict() for ctx in self._contexts.values()]
            }
            Path(self._storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug("已保存能力上下文数据")
        except Exception as e:
            logger.error(f"保存能力数据失败: {e}")
    
    def define_tool_restrictions(self, capability: Capability) -> Set[str]:
        """
        获取能力对应的工具限制
        
        Args:
            capability: 能力级别
            
        Returns:
            Set[str]: 允许的工具集合
        """
        boundary = self._default_capabilities.get(capability)
        if boundary:
            return boundary.allowed_tools.copy()
        return set()
    
    def define_api_limits(self, capability: Capability) -> Dict[str, int]:
        """
        获取能力对应的API限制
        
        Args:
            capability: 能力级别
            
        Returns:
            Dict[str, int]: API限制字典
        """
        boundary = self._default_capabilities.get(capability)
        if boundary:
            return boundary.api_rate_limits.copy()
        return {}
    
    def define_data_scope(self, capability: Capability) -> Set[str]:
        """
        获取能力对应的数据访问范围
        
        Args:
            capability: 能力级别
            
        Returns:
            Set[str]: 数据访问范围集合
        """
        boundary = self._default_capabilities.get(capability)
        if boundary:
            return boundary.data_scope.copy()
        return set()
    
    def register_capabilities(
        self,
        session_id: str,
        user_id: str,
        capability: Capability = Capability.READ_ONLY,
        allowed_tools: Optional[Set[str]] = None,
        forbidden_tools: Optional[Set[str]] = None,
        max_api_calls: Optional[int] = None,
        data_scope: Optional[Set[str]] = None,
        expires_at: Optional[datetime] = None
    ) -> CapabilityContext:
        """
        注册会话的能力
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            capability: 能力级别
            allowed_tools: 允许的工具集合(可选)
            forbidden_tools: 禁止的工具集合(可选)
            max_api_calls: 每小时最大API调用次数
            data_scope: 数据访问范围
            expires_at: 过期时间
            
        Returns:
            CapabilityContext: 创建的能力上下文
        """
        with self._lock:
            # 获取能力默认边界
            boundary = self._default_capabilities.get(
                capability, 
                self._default_capabilities[Capability.READ_ONLY]
            )
            
            # 创建能力上下文
            context = CapabilityContext(
                session_id=session_id,
                user_id=user_id,
                current_capability=capability,
                allowed_tools=allowed_tools or boundary.allowed_tools.copy(),
                forbidden_tools=forbidden_tools or set(),
                max_api_calls_per_hour=max_api_calls or boundary.api_rate_limits.get('total', 50),
                data_access_scope=data_scope or boundary.data_scope.copy(),
                expires_at=expires_at
            )
            
            self._contexts[session_id] = context
            
            # 更新用户会话映射
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session_id)
            
            self._save_data()
            
            logger.info(
                f"已为会话 {session_id} (用户: {user_id}) "
                f"注册能力: {capability.name}"
            )
            
            return context
    
    def get_capabilities(self, session_id: str) -> Optional[CapabilityContext]:
        """
        获取会话的能力上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[CapabilityContext]: 能力上下文
        """
        with self._lock:
            ctx = self._contexts.get(session_id)
            
            if ctx and ctx.is_expired():
                # 能力已过期,降级为READ_ONLY
                logger.warning(f"会话 {session_id} 能力已过期,降级为READ_ONLY")
                ctx.current_capability = Capability.READ_ONLY
                ctx.allowed_tools = self.define_tool_restrictions(Capability.READ_ONLY)
                ctx.max_api_calls_per_hour = 10
            
            return ctx
    
    def reduce_capability(
        self,
        session_id: str,
        new_capability: Capability
    ) -> bool:
        """
        降级会话的能力
        
        Args:
            session_id: 会话ID
            new_capability: 新的能力级别
            
        Returns:
            bool: 降级是否成功
        """
        with self._lock:
            ctx = self._contexts.get(session_id)
            
            if ctx is None:
                logger.warning(f"会话 {session_id} 不存在")
                return False
            
            if new_capability.priority >= ctx.current_capability.priority:
                logger.warning(
                    f"能力降级失败: {ctx.current_capability.name} -> {new_capability.name}, "
                    f"新能力不能高于或等于当前能力"
                )
                return False
            
            ctx.current_capability = new_capability
            ctx.allowed_tools = self.define_tool_restrictions(new_capability)
            ctx.max_api_calls_per_hour = self.define_api_limits(new_capability).get(
                'total', 10
            )
            
            self._save_data()
            
            logger.info(
                f"会话 {session_id} 能力已降级: {ctx.current_capability.name}"
            )
            
            return True
    
    def revoke_capability(self, session_id: str) -> bool:
        """
        撤销会话的能力
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 撤销是否成功
        """
        with self._lock:
            if session_id in self._contexts:
                ctx = self._contexts[session_id]
                del self._contexts[session_id]
                
                # 更新用户会话映射
                if ctx.user_id in self._user_sessions:
                    self._user_sessions[ctx.user_id].discard(session_id)
                
                self._save_data()
                
                logger.info(f"会话 {session_id} 的能力已被撤销")
                return True
            
            return False
    
    def check_api_limit(self, session_id: str) -> Tuple[bool, int]:
        """
        检查API调用是否超过限制
        
        Args:
            session_id: 会话ID
            
        Returns:
            Tuple[bool, int]: (是否允许, 剩余调用次数)
        """
        with self._lock:
            ctx = self._contexts.get(session_id)
            
            if ctx is None:
                return True, 0
            
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            
            # 获取或初始化调用记录
            if session_id not in self._api_call_counts:
                self._api_call_counts[session_id] = []
            
            # 清理超过1小时的记录
            self._api_call_counts[session_id] = [
                t for t in self._api_call_counts[session_id]
                if t > hour_ago
            ]
            
            # 检查限制
            remaining = ctx.max_api_calls_per_hour - len(self._api_call_counts[session_id])
            
            if remaining <= 0:
                logger.warning(
                    f"会话 {session_id} API调用已达上限: "
                    f"{ctx.max_api_calls_per_hour}/小时"
                )
                return False, 0
            
            # 记录本次调用
            self._api_call_counts[session_id].append(now)
            
            return True, remaining - 1
    
    def record_api_call(self, session_id: str, api_name: str) -> None:
        """
        记录API调用
        
        Args:
            session_id: 会话ID
            api_name: API名称
        """
        with self._lock:
            if session_id not in self._api_call_counts:
                self._api_call_counts[session_id] = []
            
            self._api_call_counts[session_id].append(datetime.now())
    
    def get_user_sessions(self, user_id: str) -> List[CapabilityContext]:
        """
        获取用户的所有会话能力上下文
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[CapabilityContext]: 能力上下文列表
        """
        with self._lock:
            session_ids = self._user_sessions.get(user_id, set())
            return [
                self._contexts[sid] 
                for sid in session_ids 
                if sid in self._contexts
            ]


class CapabilityEscalation:
    """
    能力提升管理器
    
    管理临时能力提升请求和审批流程
    """
    
    def __init__(self, registry: Optional[CapabilityRegistry] = None):
        """
        初始化能力提升管理器
        
        Args:
            registry: 能力注册表
        """
        self._lock = threading.RLock()
        self._registry = registry or CapabilityRegistry()
        self._escalation_requests: Dict[str, EscalationRequest] = {}
        
        # 自动提升配置
        self._auto_escalate_actions: Set[str] = {
            'CRITICAL_SYSTEM_OPERATION',
            'EMERGENCY_BREAK',
            'SECURITY_PATCH'
        }
        
        logger.info("能力提升管理器初始化完成")
    
    def request_escalation(
        self,
        session_id: str,
        user_id: str,
        new_capability: Capability,
        reason: str
    ) -> EscalationRequest:
        """
        请求能力提升
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            new_capability: 请求的能力级别
            reason: 请求原因
            
        Returns:
            EscalationRequest: 创建的提升请求
        """
        with self._lock:
            # 获取当前能力
            current_ctx = self._registry.get_capabilities(session_id)
            current_capability = current_ctx.current_capability if current_ctx else Capability.READ_ONLY
            
            # 检查是否允许提升
            if not current_capability.can_escalate_to(new_capability):
                raise ValueError(
                    f"能力提升不被允许: {current_capability.name} -> {new_capability.name}"
                )
            
            escalation_id = str(uuid.uuid4())
            request = EscalationRequest(
                escalation_id=escalation_id,
                session_id=session_id,
                user_id=user_id,
                current_capability=current_capability,
                requested_capability=new_capability,
                reason=reason
            )
            
            self._escalation_requests[escalation_id] = request
            
            logger.info(
                f"能力提升请求已创建: {escalation_id}, "
                f"{current_capability.name} -> {new_capability.name}"
            )
            
            return request
    
    def approve_escalation(self, escalation_id: str, approver: str) -> bool:
        """
        审批能力提升请求
        
        Args:
            escalation_id: 请求ID
            approver: 审批人
            
        Returns:
            bool: 审批是否成功
        """
        with self._lock:
            request = self._escalation_requests.get(escalation_id)
            
            if request is None:
                logger.warning(f"能力提升请求不存在: {escalation_id}")
                return False
            
            if request.status != 'pending':
                logger.warning(f"请求已被处理: {escalation_id}, 状态: {request.status}")
                return False
            
            # 更新请求状态
            request.status = 'approved'
            request.approved_by = approver
            request.approved_at = datetime.now()
            
            # 应用新能力
            self._registry.register_capabilities(
                session_id=request.session_id,
                user_id=request.user_id,
                capability=request.requested_capability,
                expires_at=datetime.now() + timedelta(hours=1)  # 默认1小时后过期
            )
            
            logger.info(
                f"能力提升已批准: {escalation_id}, "
                f"审批人: {approver}"
            )
            
            return True
    
    def reject_escalation(self, escalation_id: str, rejector: str, reason: str) -> bool:
        """
        拒绝能力提升请求
        
        Args:
            escalation_id: 请求ID
            rejector: 拒绝人
            reason: 拒绝原因
            
        Returns:
            bool: 拒绝是否成功
        """
        with self._lock:
            request = self._escalation_requests.get(escalation_id)
            
            if request is None:
                return False
            
            if request.status != 'pending':
                return False
            
            request.status = 'rejected'
            request.approved_by = rejector
            request.approved_at = datetime.now()
            
            logger.info(f"能力提升已拒绝: {escalation_id}, 原因: {reason}")
            
            return True
    
    def auto_escalation_for_critical(
        self,
        session_id: str,
        action: str
    ) -> Tuple[bool, Optional[Capability]]:
        """
        为关键操作自动提升能力
        
        当检测到需要ADMIN权限的关键操作时,自动给予临时提升
        
        Args:
            session_id: 会话ID
            action: 操作名称
            
        Returns:
            Tuple[bool, Optional[Capability]]: (是否已自动提升, 提升到的能力级别)
        """
        with self._lock:
            if action not in self._auto_escalate_actions:
                return False, None
            
            ctx = self._registry.get_capabilities(session_id)
            if ctx is None:
                return False, None
            
            # 如果已经是ADMIN,则无需提升
            if ctx.current_capability == Capability.ADMIN:
                return True, Capability.ADMIN
            
            # 临时提升到ADMIN
            self._registry.register_capabilities(
                session_id=session_id,
                user_id=ctx.user_id,
                capability=Capability.ADMIN,
                expires_at=datetime.now() + timedelta(minutes=5)  # 仅5分钟
            )
            
            logger.warning(
                f"关键操作 {action} 触发自动能力提升: "
                f"会话 {session_id} 临时提升到ADMIN (5分钟)"
            )
            
            return True, Capability.ADMIN
    
    def get_pending_requests(self) -> List[EscalationRequest]:
        """
        获取所有待审批的请求
        
        Returns:
            List[EscalationRequest]: 待审批请求列表
        """
        with self._lock:
            return [
                r for r in self._escalation_requests.values()
                if r.status == 'pending'
            ]
    
    def get_request(self, escalation_id: str) -> Optional[EscalationRequest]:
        """
        获取提升请求详情
        
        Args:
            escalation_id: 请求ID
            
        Returns:
            Optional[EscalationRequest]: 提升请求
        """
        return self._escalation_requests.get(escalation_id)


class CapabilityMinimizer:
    """
    能力最小化主控制器
    
    整合所有能力最小化组件,提供统一的接口
    """
    
    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        reducer: Optional[DangerousOperationReducer] = None,
        escalation: Optional[CapabilityEscalation] = None
    ):
        """
        初始化能力最小化控制器
        
        Args:
            registry: 能力注册表
            reducer: 危险操作降级器
            escalation: 能力提升管理器
        """
        self._registry = registry or CapabilityRegistry()
        self._reducer = reducer or DangerousOperationReducer()
        self._escalation = escalation or CapabilityEscalation(self._registry)
        
        logger.info("能力最小化控制器初始化完成")
    
    def register_session(
        self,
        session_id: str,
        user_id: str,
        capability: Capability = Capability.READ_ONLY,
        **kwargs
    ) -> CapabilityContext:
        """
        注册新会话的能力
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            capability: 能力级别
            **kwargs: 其他可选参数
            
        Returns:
            CapabilityContext: 能力上下文
        """
        return self._registry.register_capabilities(
            session_id=session_id,
            user_id=user_id,
            capability=capability,
            allowed_tools=kwargs.get('allowed_tools'),
            forbidden_tools=kwargs.get('forbidden_tools'),
            max_api_calls=kwargs.get('max_api_calls'),
            data_scope=kwargs.get('data_scope'),
            expires_at=kwargs.get('expires_at')
        )
    
    def check_capability(
        self,
        session_id: str,
        tool_name: str,
        api_call: bool = False
    ) -> Tuple[bool, str]:
        """
        检查会话是否有能力使用工具
        
        Args:
            session_id: 会话ID
            tool_name: 工具名称
            api_call: 是否为API调用
            
        Returns:
            Tuple[bool, str]: (是否允许, 原因说明)
        """
        ctx = self._registry.get_capabilities(session_id)
        
        if ctx is None:
            return False, "会话能力未注册"
        
        # 检查工具是否允许
        if not ctx.is_tool_allowed(tool_name):
            return False, f"工具 {tool_name} 不在允许列表中"
        
        # 如果是API调用,检查限流
        if api_call:
            allowed, remaining = self._registry.check_api_limit(session_id)
            if not allowed:
                return False, f"API调用已达上限 (剩余: {remaining})"
        
        return True, "能力检查通过"
    
    def process_action(
        self,
        session_id: str,
        action: str,
        params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        处理操作,应用能力最小化和危险降级
        
        Args:
            session_id: 会话ID
            action: 操作名称
            params: 操作参数
            
        Returns:
            Tuple[str, Dict[str, Any], bool]: (处理后操作, 新参数, 是否需要审批)
        """
        # 首先检查是否为危险操作
        if self._reducer.is_action_blocked(action):
            raise PermissionError(f"危险操作 {action} 被安全策略阻止")
        
        # 应用危险操作降级
        reduced_action, new_params, requires_approval = self._reducer.reduce(
            action, params
        )
        
        return reduced_action, new_params, requires_approval
    
    def request_escalation(
        self,
        session_id: str,
        user_id: str,
        new_capability: Capability,
        reason: str
    ) -> EscalationRequest:
        """
        请求能力提升
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            new_capability: 目标能力级别
            reason: 请求原因
            
        Returns:
            EscalationRequest: 提升请求
        """
        return self._escalation.request_escalation(
            session_id=session_id,
            user_id=user_id,
            new_capability=new_capability,
            reason=reason
        )
    
    def approve_escalation(self, escalation_id: str, approver: str) -> bool:
        """
        审批能力提升请求
        
        Args:
            escalation_id: 请求ID
            approver: 审批人
            
        Returns:
            bool: 审批是否成功
        """
        return self._escalation.approve_escalation(escalation_id, approver)
    
    def get_capability_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取能力摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict[str, Any]: 能力摘要信息
        """
        ctx = self._registry.get_capabilities(session_id)
        
        if ctx is None:
            return {'session_id': session_id, 'registered': False}
        
        return {
            'session_id': session_id,
            'user_id': ctx.user_id,
            'registered': True,
            'current_capability': ctx.current_capability.name,
            'allowed_tools_count': len(ctx.allowed_tools),
            'forbidden_tools_count': len(ctx.forbidden_tools),
            'max_api_calls_per_hour': ctx.max_api_calls_per_hour,
            'data_access_scope': list(ctx.data_access_scope),
            'granted_at': ctx.granted_at.isoformat(),
            'expires_at': ctx.expires_at.isoformat() if ctx.expires_at else None,
            'is_expired': ctx.is_expired()
        }


# =============================================================================
# 与 action_guard 集成
# =============================================================================

def check_capability_before_action_guard(
    session_id: str,
    tool_name: str,
    minimizer: CapabilityMinimizer
) -> Tuple[bool, str]:
    """
    在 action_guard 检查敏感度之前先检查能力
    
    如果工具不在允许列表中,则永远不会到达确认对话框
    
    Args:
        session_id: 会话ID
        tool_name: 工具名称
        minimizer: 能力最小化控制器
        
    Returns:
        Tuple[bool, str]: (是否允许, 原因说明)
    """
    allowed, reason = minimizer.check_capability(session_id, tool_name)
    
    if not allowed:
        logger.info(f"能力检查拒绝: {tool_name} @ {session_id}")
    
    return allowed, reason


# =============================================================================
# CLI 接口
# =============================================================================

import argparse
import sys


def main():
    """
    能力最小化CLI主入口
    
    支持以下命令:
    - register: 注册会话能力
    - check: 检查工具能力
    - process: 处理操作(应用降级)
    - escalate: 请求能力提升
    - approve: 审批能力提升请求
    - summary: 获取能力摘要
    """
    parser = argparse.ArgumentParser(
        description="能力最小化控制工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # register 命令
    reg_parser = subparsers.add_parser('register', help='注册会话能力')
    reg_parser.add_argument('--session-id', required=True, help='会话ID')
    reg_parser.add_argument('--user-id', required=True, help='用户ID')
    reg_parser.add_argument('--capability', required=True,
                          choices=['READ_ONLY', 'LIMITED_WRITE', 'FULL_ACCESS', 'ADMIN'],
                          help='能力级别')
    reg_parser.add_argument('--expires-minutes', type=int, help='过期时间(分钟)')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查工具能力')
    check_parser.add_argument('--session-id', required=True, help='会话ID')
    check_parser.add_argument('--tool', required=True, help='工具名称')
    check_parser.add_argument('--api-call', action='store_true', help='是否为API调用')
    
    # process 命令
    proc_parser = subparsers.add_parser('process', help='处理操作')
    proc_parser.add_argument('--session-id', required=True, help='会话ID')
    proc_parser.add_argument('--action', required=True, help='操作名称')
    proc_parser.add_argument('--params', default='{}', help='参数字典(JSON)')
    
    # escalate 命令
    esc_parser = subparsers.add_parser('escalate', help='请求能力提升')
    esc_parser.add_argument('--session-id', required=True, help='会话ID')
    esc_parser.add_argument('--user-id', required=True, help='用户ID')
    esc_parser.add_argument('--capability', required=True,
                          choices=['READ_ONLY', 'LIMITED_WRITE', 'FULL_ACCESS', 'ADMIN'],
                          help='目标能力级别')
    esc_parser.add_argument('--reason', required=True, help='请求原因')
    
    # approve 命令
    appr_parser = subparsers.add_parser('approve', help='审批能力提升请求')
    appr_parser.add_argument('--escalation-id', required=True, help='请求ID')
    appr_parser.add_argument('--approver', required=True, help='审批人')
    
    # summary 命令
    sum_parser = subparsers.add_parser('summary', help='获取能力摘要')
    sum_parser.add_argument('--session-id', required=True, help='会话ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 初始化控制器
    minimizer = CapabilityMinimizer()
    
    if args.command == 'register':
        capability = Capability[args.capability]
        expires_at = None
        if args.expires_minutes:
            expires_at = datetime.now() + timedelta(minutes=args.expires_minutes)
        
        ctx = minimizer.register_session(
            session_id=args.session_id,
            user_id=args.user_id,
            capability=capability,
            expires_at=expires_at
        )
        
        print(json.dumps({
            'success': True,
            **ctx.to_dict()
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'check':
        allowed, reason = minimizer.check_capability(
            args.session_id, 
            args.tool,
            api_call=args.api_call
        )
        
        print(json.dumps({
            'session_id': args.session_id,
            'tool': args.tool,
            'allowed': allowed,
            'reason': reason
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'process':
        params = json.loads(args.params)
        
        try:
            reduced_action, new_params, requires_approval = minimizer.process_action(
                args.session_id,
                args.action,
                params
            )
            
            print(json.dumps({
                'success': True,
                'original_action': args.action,
                'reduced_action': reduced_action,
                'new_params': new_params,
                'requires_approval': requires_approval
            }, indent=2, ensure_ascii=False))
        except PermissionError as e:
            print(json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2, ensure_ascii=False))
    
    elif args.command == 'escalate':
        capability = Capability[args.capability]
        
        try:
            request = minimizer.request_escalation(
                session_id=args.session_id,
                user_id=args.user_id,
                new_capability=capability,
                reason=args.reason
            )
            
            print(json.dumps({
                'success': True,
                'escalation_id': request.escalation_id,
                'status': request.status,
                'current_capability': request.current_capability.name,
                'requested_capability': request.requested_capability.name
            }, indent=2, ensure_ascii=False))
        except ValueError as e:
            print(json.dumps({
                'success': False,
                'error': str(e)
            }, indent=2, ensure_ascii=False))
    
    elif args.command == 'approve':
        success = minimizer.approve_escalation(args.escalation_id, args.approver)
        
        print(json.dumps({
            'success': success,
            'escalation_id': args.escalation_id,
            'message': '审批成功' if success else '审批失败'
        }, indent=2, ensure_ascii=False))
    
    elif args.command == 'summary':
        summary = minimizer.get_capability_summary(args.session_id)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
