"""
访问控制列表
限制哪些服务可发现哪些Agent
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union

from ..schema import AgentMetadata, AgentRole


class Permission(Enum):
    """权限类型"""
    DISCOVER = "discover"     # 发现服务
    REGISTER = "register"     # 注册服务
    DEREGISTER = "deregister" # 注销服务
    UPDATE = "update"         # 更新元数据
    WATCH = "watch"           # 监听事件
    ADMIN = "admin"           # 管理权限


class AccessDecision(Enum):
    """访问决策"""
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass
class AccessControlEntry:
    """
    访问控制条目
    
    定义特定主体对特定资源的访问权限
    """
    subject: str           # 主体（服务ID或通配符）
    resource: str          # 资源（Agent ID或通配符）
    permissions: Set[Permission] = field(default_factory=set)
    decision: AccessDecision = AccessDecision.ALLOW
    priority: int = 0      # 优先级，数字越大优先级越高
    conditions: Dict[str, Any] = field(default_factory=dict)

    def matches_subject(self, subject: str) -> bool:
        """检查主体是否匹配"""
        return self._match_pattern(subject, self.subject)

    def matches_resource(self, resource: str) -> bool:
        """检查资源是否匹配"""
        return self._match_pattern(resource, self.resource)

    def has_permission(self, permission: Permission) -> bool:
        """检查是否有指定权限"""
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        for key, expected_value in self.conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        return True

    @staticmethod
    def _match_pattern(value: str, pattern: str) -> bool:
        """使用通配符匹配"""
        if pattern == "*":
            return True
        if pattern == value:
            return True
        return fnmatch.fnmatch(value, pattern)


class AccessControlList:
    """
    访问控制列表
    
    管理访问控制条目，支持优先级和规则匹配
    """

    def __init__(self):
        self._entries: List[AccessControlEntry] = []
        self._default_decision = AccessDecision.DENY

    def add_entry(self, entry: AccessControlEntry) -> None:
        """添加访问控制条目"""
        self._entries.append(entry)
        # 按优先级排序（高优先级在前）
        self._entries.sort(key=lambda e: e.priority, reverse=True)

    def remove_entry(self, subject: str, resource: str) -> bool:
        """移除访问控制条目"""
        for i, entry in enumerate(self._entries):
            if entry.subject == subject and entry.resource == resource:
                del self._entries[i]
                return True
        return False

    def check_access(
        self,
        subject: str,
        resource: str,
        permission: Permission,
        context: Optional[Dict[str, Any]] = None
    ) -> AccessDecision:
        """
        检查访问权限
        
        Args:
            subject: 访问主体
            resource: 访问资源
            permission: 请求权限
            context: 上下文信息
            
        Returns:
            访问决策
        """
        context = context or {}
        
        for entry in self._entries:
            # 检查主体匹配
            if not entry.matches_subject(subject):
                continue
            
            # 检查资源匹配
            if not entry.matches_resource(resource):
                continue
            
            # 检查权限
            if not entry.has_permission(permission):
                continue
            
            # 检查条件
            if not entry.check_conditions(context):
                continue
            
            # 匹配成功，返回决策
            return entry.decision
        
        # 无匹配规则，返回默认决策
        return self._default_decision

    def is_allowed(
        self,
        subject: str,
        resource: str,
        permission: Permission,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检查是否允许访问"""
        decision = self.check_access(subject, resource, permission, context)
        return decision == AccessDecision.ALLOW

    def get_entries_for_subject(self, subject: str) -> List[AccessControlEntry]:
        """获取主体的所有条目"""
        return [e for e in self._entries if e.matches_subject(subject)]

    def get_entries_for_resource(self, resource: str) -> List[AccessControlEntry]:
        """获取资源的所有条目"""
        return [e for e in self._entries if e.matches_resource(resource)]

    def clear(self) -> None:
        """清空所有条目"""
        self._entries.clear()

    def set_default_decision(self, decision: AccessDecision) -> None:
        """设置默认决策"""
        self._default_decision = decision


class ServiceACLManager:
    """
    服务ACL管理器
    
    管理服务间的访问控制
    """

    def __init__(self):
        self._acl = AccessControlList()
        self._service_roles: Dict[str, Set[str]] = {}  # service -> roles
        self._role_permissions: Dict[str, Set[Permission]] = {}  # role -> permissions

    def register_service(self, service_id: str, roles: Optional[List[str]] = None) -> None:
        """注册服务"""
        self._service_roles[service_id] = set(roles or [])

    def assign_role(self, service_id: str, role: str) -> None:
        """为服务分配角色"""
        if service_id not in self._service_roles:
            self._service_roles[service_id] = set()
        self._service_roles[service_id].add(role)

    def define_role_permissions(self, role: str, permissions: Set[Permission]) -> None:
        """定义角色权限"""
        self._role_permissions[role] = permissions

    def can_discover(
        self,
        caller_id: str,
        agent: AgentMetadata,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        检查调用方是否可以发现Agent
        
        Args:
            caller_id: 调用方服务ID
            agent: 目标Agent
            context: 上下文信息
            
        Returns:
            是否允许发现
        """
        # 首先检查ACL
        if self._acl.is_allowed(caller_id, agent.agent_id, Permission.DISCOVER, context):
            return True
        
        # 检查角色权限
        return self._check_role_permission(caller_id, Permission.DISCOVER)

    def can_register(self, service_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """检查是否可以注册"""
        if self._acl.is_allowed(service_id, "*", Permission.REGISTER, context):
            return True
        return self._check_role_permission(service_id, Permission.REGISTER)

    def can_deregister(
        self,
        service_id: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检查是否可以注销"""
        # 服务可以注销自己
        if service_id == agent_id:
            return True
        
        if self._acl.is_allowed(service_id, agent_id, Permission.DEREGISTER, context):
            return True
        
        return self._check_role_permission(service_id, Permission.DEREGISTER)

    def can_watch(self, service_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """检查是否可以监听事件"""
        if self._acl.is_allowed(service_id, "*", Permission.WATCH, context):
            return True
        return self._check_role_permission(service_id, Permission.WATCH)

    def _check_role_permission(self, service_id: str, permission: Permission) -> bool:
        """检查服务角色权限"""
        roles = self._service_roles.get(service_id, set())
        for role in roles:
            permissions = self._role_permissions.get(role, set())
            if permission in permissions or Permission.ADMIN in permissions:
                return True
        return False

    def add_acl_entry(self, entry: AccessControlEntry) -> None:
        """添加ACL条目"""
        self._acl.add_entry(entry)

    def remove_acl_entry(self, subject: str, resource: str) -> bool:
        """移除ACL条目"""
        return self._acl.remove_entry(subject, resource)

    def filter_discoverable_agents(
        self,
        caller_id: str,
        agents: List[AgentMetadata],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AgentMetadata]:
        """
        过滤可发现的Agent列表
        
        Args:
            caller_id: 调用方服务ID
            agents: Agent列表
            context: 上下文信息
            
        Returns:
            可发现的Agent列表
        """
        return [
            agent for agent in agents
            if self.can_discover(caller_id, agent, context)
        ]


class NamespaceIsolation:
    """
    命名空间隔离
    
    实现基于命名空间的服务隔离
    """

    def __init__(self):
        self._namespaces: Dict[str, Set[str]] = {}  # namespace -> set of services
        self._service_namespace: Dict[str, str] = {}  # service -> namespace
        self._isolation_enabled: bool = True

    def create_namespace(self, namespace: str) -> None:
        """创建命名空间"""
        if namespace not in self._namespaces:
            self._namespaces[namespace] = set()

    def delete_namespace(self, namespace: str) -> bool:
        """删除命名空间"""
        if namespace in self._namespaces:
            # 移除该命名空间下的所有服务
            for service_id in list(self._namespaces[namespace]):
                del self._service_namespace[service_id]
            del self._namespaces[namespace]
            return True
        return False

    def assign_to_namespace(self, service_id: str, namespace: str) -> None:
        """将服务分配到命名空间"""
        self.create_namespace(namespace)
        
        # 从原命名空间移除
        old_namespace = self._service_namespace.get(service_id)
        if old_namespace and old_namespace in self._namespaces:
            self._namespaces[old_namespace].discard(service_id)
        
        # 添加到新命名空间
        self._namespaces[namespace].add(service_id)
        self._service_namespace[service_id] = namespace

    def get_namespace(self, service_id: str) -> Optional[str]:
        """获取服务所属命名空间"""
        return self._service_namespace.get(service_id)

    def can_communicate(self, service_a: str, service_b: str) -> bool:
        """
        检查两个服务是否可以通信
        
        同一命名空间的服务可以通信
        """
        if not self._isolation_enabled:
            return True
        
        ns_a = self._service_namespace.get(service_a)
        ns_b = self._service_namespace.get(service_b)
        
        # 如果任一服务不在命名空间中，允许通信
        if ns_a is None or ns_b is None:
            return True
        
        return ns_a == ns_b

    def get_services_in_namespace(self, namespace: str) -> Set[str]:
        """获取命名空间中的所有服务"""
        return self._namespaces.get(namespace, set()).copy()

    def enable_isolation(self) -> None:
        """启用隔离"""
        self._isolation_enabled = True

    def disable_isolation(self) -> None:
        """禁用隔离"""
        self._isolation_enabled = False


class SecureRegistryWrapper:
    """
    安全注册表包装器
    
    为注册表操作添加访问控制
    """

    def __init__(
        self,
        acl_manager: ServiceACLManager,
        namespace_isolation: Optional[NamespaceIsolation] = None
    ):
        self._acl = acl_manager
        self._ns = namespace_isolation

    def check_discover(
        self,
        caller_id: str,
        agent: AgentMetadata,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检查发现权限"""
        # 检查ACL
        if not self._acl.can_discover(caller_id, agent, context):
            return False
        
        # 检查命名空间隔离
        if self._ns and not self._ns.can_communicate(caller_id, agent.agent_id):
            return False
        
        return True

    def check_register(
        self,
        service_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检查注册权限"""
        return self._acl.can_register(service_id, context)

    def check_deregister(
        self,
        service_id: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """检查注销权限"""
        return self._acl.can_deregister(service_id, agent_id, context)


# 预定义的ACL规则构建器

class ACLBuilder:
    """ACL规则构建器"""

    @staticmethod
    def allow_all(subject: str = "*") -> AccessControlEntry:
        """允许所有权限"""
        return AccessControlEntry(
            subject=subject,
            resource="*",
            permissions=set(Permission),
            decision=AccessDecision.ALLOW,
            priority=0
        )

    @staticmethod
    def deny_all(subject: str = "*") -> AccessControlEntry:
        """拒绝所有权限"""
        return AccessControlEntry(
            subject=subject,
            resource="*",
            permissions=set(Permission),
            decision=AccessDecision.DENY,
            priority=100
        )

    @staticmethod
    def allow_discover(subject: str, resource: str = "*", priority: int = 0) -> AccessControlEntry:
        """允许发现"""
        return AccessControlEntry(
            subject=subject,
            resource=resource,
            permissions={Permission.DISCOVER},
            decision=AccessDecision.ALLOW,
            priority=priority
        )

    @staticmethod
    def allow_admin(subject: str, priority: int = 100) -> AccessControlEntry:
        """允许管理权限"""
        return AccessControlEntry(
            subject=subject,
            resource="*",
            permissions={Permission.ADMIN},
            decision=AccessDecision.ALLOW,
            priority=priority
        )
