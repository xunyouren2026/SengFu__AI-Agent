"""
工具安全管理模块

提供工具权限验证、参数安全检查（路径遍历、命令注入、XSS 等）以及 ACL 实现。
仅使用 Python 标准库。
"""

import enum
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# PermissionLevel - 权限级别
# ---------------------------------------------------------------------------
class PermissionLevel(enum.IntEnum):
    """工具权限级别"""
    DENY = 0       # 拒绝访问
    READ = 1       # 只读（可查看工具信息，不可执行）
    EXECUTE = 2    # 可执行
    ADMIN = 3      # 管理权限（可修改工具配置）


# ---------------------------------------------------------------------------
# ACLEntry - 访问控制条目
# ---------------------------------------------------------------------------
@dataclass
class ACLEntry:
    """访问控制条目"""
    agent_id: str
    tool_name: str
    level: PermissionLevel = PermissionLevel.DENY
    granted_at: float = 0.0
    expires_at: Optional[float] = None
    granted_by: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "level": self.level.name,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "granted_by": self.granted_by,
        }


# ---------------------------------------------------------------------------
# ToolSecurityManager - 工具权限验证器
# ---------------------------------------------------------------------------
class ToolSecurityManager:
    """工具权限验证器

    管理工具访问控制列表 (ACL)，提供权限检查、授予、撤销功能，
    以及参数安全检查（路径遍历、命令注入、XSS 等）。
    """

    # 危险命令模式
    _COMMAND_INJECTION_PATTERNS: List[re.Pattern] = [
        re.compile(r";\s*\w", re.IGNORECASE),
        re.compile(r"\|\s*\w", re.IGNORECASE),
        re.compile(r"`[^`]+`"),
        re.compile(r"\$\([^)]+\)"),
        re.compile(r"\$\{[^}]+\}"),
        re.compile(r"&&\s*\w", re.IGNORECASE),
        re.compile(r"\|\|\s*\w", re.IGNORECASE),
        re.compile(r">\s*/dev/"),
        re.compile(r"<\s*/dev/"),
        re.compile(r"\b(eval|exec|system|popen|shell|spawn|import)\b\s*\(", re.IGNORECASE),
        re.compile(r"\b(rm|del|format|mkfs|dd)\b\s+-[rf]", re.IGNORECASE),
        re.compile(r"\b(sudo|su|chmod|chown)\b", re.IGNORECASE),
        re.compile(r"\b(curl|wget|nc|ncat|telnet)\b", re.IGNORECASE),
        re.compile(r"\b(python|perl|ruby|bash|sh|zsh)\b\s+-[ce]", re.IGNORECASE),
    ]

    # XSS 模式
    _XSS_PATTERNS: List[re.Pattern] = [
        re.compile(r"<\s*script\b", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<\s*iframe\b", re.IGNORECASE),
        re.compile(r"<\s*object\b", re.IGNORECASE),
        re.compile(r"<\s*embed\b", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),
        re.compile(r"url\s*\(", re.IGNORECASE),
    ]

    # 路径遍历模式
    _PATH_TRAVERSAL_PATTERNS: List[re.Pattern] = [
        re.compile(r"\.\./"),
        re.compile(r"\.\.\\"),
        re.compile(r"\.\.%2[fF]"),
        re.compile(r"\.\.%5[cC]"),
        re.compile(r"%2[eE]%2[eE]"),
        re.compile(r"\.\.%0[0-9a-fA-F]"),
    ]

    def __init__(self):
        self._acl: Dict[str, ACLEntry] = {}  # key: "agent_id:tool_name"
        self._default_level: PermissionLevel = PermissionLevel.EXECUTE
        self._lock = threading.RLock()
        self._audit_log: List[dict] = []
        self._max_audit_log_size: int = 10000

    # ----- 权限管理 -----

    def grant_permission(
        self,
        agent_id: str,
        tool_name: str,
        level: PermissionLevel,
        expires_at: Optional[float] = None,
        granted_by: str = "",
    ) -> None:
        """授予权限"""
        key = self._make_key(agent_id, tool_name)
        with self._lock:
            self._acl[key] = ACLEntry(
                agent_id=agent_id,
                tool_name=tool_name,
                level=level,
                granted_at=time.time(),
                expires_at=expires_at,
                granted_by=granted_by,
            )
            self._log_audit("grant", agent_id, tool_name, level=level.name)

    def revoke_permission(self, agent_id: str, tool_name: str) -> bool:
        """撤销权限，返回是否成功"""
        key = self._make_key(agent_id, tool_name)
        with self._lock:
            if key not in self._acl:
                return False
            del self._acl[key]
            self._log_audit("revoke", agent_id, tool_name)
            return True

    def check_permission(
        self, agent_id: str, tool_name: str
    ) -> PermissionLevel:
        """检查权限，返回权限级别"""
        key = self._make_key(agent_id, tool_name)
        with self._lock:
            entry = self._acl.get(key)
            if entry is None:
                return self._default_level
            if entry.is_expired:
                del self._acl[key]
                self._log_audit("expired", agent_id, tool_name)
                return self._default_level
            self._log_audit("check", agent_id, tool_name, level=entry.level.name)
            return entry.level

    def has_permission(
        self,
        agent_id: str,
        tool_name: str,
        required_level: PermissionLevel = PermissionLevel.EXECUTE,
    ) -> bool:
        """检查是否具有指定级别的权限"""
        actual = self.check_permission(agent_id, tool_name)
        return actual >= required_level

    def set_default_level(self, level: PermissionLevel) -> None:
        """设置默认权限级别"""
        with self._lock:
            self._default_level = level

    # ----- ACL 查询 -----

    def get_acl_entry(
        self, agent_id: str, tool_name: str
    ) -> Optional[ACLEntry]:
        """获取 ACL 条目"""
        key = self._make_key(agent_id, tool_name)
        with self._lock:
            entry = self._acl.get(key)
            if entry and entry.is_expired:
                del self._acl[key]
                return None
            return entry

    def list_permissions(self, agent_id: Optional[str] = None) -> List[dict]:
        """列出权限"""
        with self._lock:
            results = []
            for entry in self._acl.values():
                if entry.is_expired:
                    continue
                if agent_id is not None and entry.agent_id != agent_id:
                    continue
                results.append(entry.to_dict())
            return results

    def get_agent_tools(self, agent_id: str) -> Dict[str, PermissionLevel]:
        """获取代理的所有工具权限"""
        with self._lock:
            result = {}
            for entry in self._acl.values():
                if entry.agent_id == agent_id and not entry.is_expired:
                    result[entry.tool_name] = entry.level
            return result

    def clear_expired(self) -> int:
        """清理过期条目，返回清理数量"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._acl.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._acl[key]
            return len(expired_keys)

    # ----- 参数安全检查 -----

    def validate_params_safety(
        self, params: dict
    ) -> Tuple[bool, Optional[str]]:
        """参数安全检查

        检查路径遍历、命令注入、XSS 等。

        Returns:
            (是否安全, 错误信息)
        """
        if not isinstance(params, dict):
            return False, "参数必须是字典类型"

        for key, value in params.items():
            if isinstance(value, str):
                # 路径遍历检查
                safe, err = self.check_path_traversal(key, value)
                if not safe:
                    return False, err

                # 命令注入检查
                safe, err = self.check_command_injection(key, value)
                if not safe:
                    return False, err

                # XSS 检查
                safe, err = self.check_xss(key, value)
                if not safe:
                    return False, err

            elif isinstance(value, dict):
                safe, err = self.validate_params_safety(value)
                if not safe:
                    return False, f"在嵌套参数 '{key}' 中: {err}"

            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        for check_fn in (
                            self.check_path_traversal,
                            self.check_command_injection,
                            self.check_xss,
                        ):
                            safe, err = check_fn(f"{key}[{i}]", item)
                            if not safe:
                                return False, err
                    elif isinstance(item, dict):
                        safe, err = self.validate_params_safety(item)
                        if not safe:
                            return False, f"在 '{key}[{i}]' 中: {err}"

        return True, None

    def check_path_traversal(
        self, param_name: str, value: str
    ) -> Tuple[bool, Optional[str]]:
        """路径遍历检测"""
        for pattern in self._PATH_TRAVERSAL_PATTERNS:
            if pattern.search(value):
                return False, (
                    f"参数 '{param_name}' 包含路径遍历字符: '{value}'"
                )
        return True, None

    def check_command_injection(
        self, param_name: str, value: str
    ) -> Tuple[bool, Optional[str]]:
        """命令注入检测"""
        for pattern in self._COMMAND_INJECTION_PATTERNS:
            if pattern.search(value):
                return False, (
                    f"参数 '{param_name}' 可能包含命令注入: '{value}'"
                )
        return True, None

    def check_xss(
        self, param_name: str, value: str
    ) -> Tuple[bool, Optional[str]]:
        """XSS 检测"""
        for pattern in self._XSS_PATTERNS:
            if pattern.search(value):
                return False, (
                    f"参数 '{param_name}' 可能包含 XSS 攻击载荷: '{value}'"
                )
        return True, None

    # ----- 路径安全化 -----

    def sanitize_path(self, path: str, base_dir: Optional[str] = None) -> str:
        """路径安全化（防止 ../ 遍历）

        Args:
            path: 待安全化的路径
            base_dir: 基础目录（如果提供，结果将限制在该目录下）

        Returns:
            安全化后的绝对路径
        """
        # 展开用户目录
        expanded = os.path.expanduser(path)
        # 规范化路径（消除 .. 和 .）
        normalized = os.path.normpath(expanded)

        if not os.path.isabs(normalized):
            if base_dir:
                normalized = os.path.join(base_dir, normalized)
            else:
                normalized = os.path.abspath(normalized)

        # 再次规范化（处理 base_dir 中的 ..）
        normalized = os.path.normpath(normalized)

        # 如果指定了 base_dir，确保路径在其下
        if base_dir:
            base_abs = os.path.abspath(base_dir)
            if not normalized.startswith(base_abs + os.sep) and normalized != base_abs:
                raise ValueError(
                    f"路径 '{path}' 超出了允许的基础目录 '{base_dir}'"
                )

        return normalized

    # ----- 审计日志 -----

    def get_audit_log(
        self,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """获取审计日志"""
        with self._lock:
            result = []
            for entry in reversed(self._audit_log):
                if agent_id and entry.get("agent_id") != agent_id:
                    continue
                if tool_name and entry.get("tool_name") != tool_name:
                    continue
                if action and entry.get("action") != action:
                    continue
                result.append(entry)
                if len(result) >= limit:
                    break
            return result

    def clear_audit_log(self) -> None:
        """清空审计日志"""
        with self._lock:
            self._audit_log.clear()

    # ----- 内部方法 -----

    def _make_key(self, agent_id: str, tool_name: str) -> str:
        return f"{agent_id}:{tool_name}"

    def _log_audit(
        self,
        action: str,
        agent_id: str,
        tool_name: str,
        level: str = "",
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "action": action,
            "agent_id": agent_id,
            "tool_name": tool_name,
            "level": level,
        }
        self._audit_log.append(entry)
        # 限制日志大小
        if len(self._audit_log) > self._max_audit_log_size:
            self._audit_log = self._audit_log[-self._max_audit_log_size:]
