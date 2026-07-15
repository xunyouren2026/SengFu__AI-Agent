"""
异常恢复模块

提供错误处理和恢复功能。
支持多种恢复类型和策略。
仅使用Python标准库实现。
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class RecoveryType(Enum):
    """恢复类型枚举。"""
    SKIP = "skip"              # 跳过当前步骤
    RETRY = "retry"            # 重试当前步骤
    GOTO = "goto"              # 跳转到指定步骤
    FALLBACK = "fallback"      # 执行备用操作
    ABORT = "abort"            # 中止工作流


@dataclass
class RecoveryAction:
    """
    恢复动作数据类。
    
    Attributes:
        type: 恢复类型
        target_step: 目标步骤ID（GOTO类型使用）
        params: 恢复参数
        description: 恢复描述
    """
    type: RecoveryType
    target_step: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        """初始化后处理。"""
        if isinstance(self.type, str):
            self.type = RecoveryType(self.type)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "type": self.type.value,
            "target_step": self.target_step,
            "params": self.params,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryAction":
        """从字典创建。"""
        return cls(
            type=RecoveryType(data.get("type", "abort")),
            target_step=data.get("target_step"),
            params=data.get("params", {}),
            description=data.get("description", ""),
        )


@dataclass
class RecoveryPolicy:
    """
    恢复策略数据类。
    
    Attributes:
        max_retries: 最大重试次数
        backoff: 退避策略（fixed, linear, exponential）
        fallback_step: 备用步骤ID
        retry_delay: 重试间隔（秒）
        on_max_retries_exceeded: 超过最大重试次数时的处理
    """
    max_retries: int = 3
    backoff: str = "fixed"  # fixed, linear, exponential
    fallback_step: Optional[str] = None
    retry_delay: float = 1.0
    on_max_retries_exceeded: str = "abort"  # skip, abort, fallback
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "max_retries": self.max_retries,
            "backoff": self.backoff,
            "fallback_step": self.fallback_step,
            "retry_delay": self.retry_delay,
            "on_max_retries_exceeded": self.on_max_retries_exceeded,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryPolicy":
        """从字典创建。"""
        return cls(
            max_retries=data.get("max_retries", 3),
            backoff=data.get("backoff", "fixed"),
            fallback_step=data.get("fallback_step"),
            retry_delay=data.get("retry_delay", 1.0),
            on_max_retries_exceeded=data.get("on_max_retries_exceeded", "abort"),
        )
    
    def calculate_delay(self, attempt: int) -> float:
        """
        计算重试延迟。
        
        Args:
            attempt: 当前尝试次数（从1开始）
            
        Returns:
            延迟时间（秒）
        """
        if self.backoff == "fixed":
            return self.retry_delay
        elif self.backoff == "linear":
            return self.retry_delay * attempt
        elif self.backoff == "exponential":
            return self.retry_delay * (2 ** (attempt - 1))
        return self.retry_delay


class ErrorRecovery:
    """
    异常恢复。
    
    管理错误恢复策略和执行。
    """
    
    def __init__(self):
        """初始化异常恢复。"""
        self._recovery_actions: Dict[str, RecoveryAction] = {}
        self._recovery_policies: Dict[str, RecoveryPolicy] = {}
        self._retry_counts: Dict[str, int] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._builtin_actions: Dict[str, Callable] = {
            "screenshot_on_error": self._screenshot_on_error,
            "notify_on_error": self._notify_on_error,
            "log_state": self._log_state,
        }
        self._custom_actions: Dict[str, Callable] = {}
    
    def register_recovery_action(
        self,
        step_id: str,
        action: Union[RecoveryAction, Dict[str, Any]],
    ) -> None:
        """
        注册恢复动作。
        
        Args:
            step_id: 步骤ID
            action: 恢复动作对象或字典
        """
        if isinstance(action, dict):
            action = RecoveryAction.from_dict(action)
        self._recovery_actions[step_id] = action
    
    def register_recovery_policy(
        self,
        step_id: str,
        policy: Union[RecoveryPolicy, Dict[str, Any]],
    ) -> None:
        """
        注册恢复策略。
        
        Args:
            step_id: 步骤ID
            policy: 恢复策略对象或字典
        """
        if isinstance(policy, dict):
            policy = RecoveryPolicy.from_dict(policy)
        self._recovery_policies[step_id] = policy
    
    def execute_recovery(
        self,
        step_id: str,
        error: Exception,
        workflow_state: Optional[Dict[str, Any]] = None,
    ) -> RecoveryAction:
        """
        执行恢复。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            workflow_state: 工作流状态
            
        Returns:
            执行的恢复动作
        """
        # 记录错误
        self._record_error(step_id, error)
        
        # 获取恢复动作
        action = self._recovery_actions.get(step_id)
        if action is None:
            action = RecoveryAction(type=RecoveryType.ABORT)
        
        # 获取恢复策略
        policy = self._recovery_policies.get(step_id, RecoveryPolicy())
        
        # 检查是否需要重试
        if action.type == RecoveryType.RETRY:
            if self.should_retry(step_id, error):
                self._retry_counts[step_id] = self._retry_counts.get(step_id, 0) + 1
                delay = policy.calculate_delay(self._retry_counts[step_id])
                time.sleep(delay)
                return action
            else:
                # 超过最大重试次数
                if policy.on_max_retries_exceeded == "skip":
                    action = RecoveryAction(type=RecoveryType.SKIP)
                elif policy.on_max_retries_exceeded == "fallback" and policy.fallback_step:
                    action = RecoveryAction(
                        type=RecoveryType.FALLBACK,
                        target_step=policy.fallback_step,
                    )
                else:
                    action = RecoveryAction(type=RecoveryType.ABORT)
        
        # 执行内置恢复动作
        self._execute_builtin_actions(step_id, error, workflow_state)
        
        return action
    
    def should_retry(self, step_id: str, error: Exception) -> bool:
        """
        判断是否重试。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            
        Returns:
            是否重试
        """
        policy = self._recovery_policies.get(step_id, RecoveryPolicy())
        retry_count = self._retry_counts.get(step_id, 0)
        return retry_count < policy.max_retries
    
    def reset_retry_count(self, step_id: Optional[str] = None) -> None:
        """
        重置重试计数。
        
        Args:
            step_id: 步骤ID，不提供则重置所有
        """
        if step_id:
            self._retry_counts.pop(step_id, None)
        else:
            self._retry_counts.clear()
    
    def _record_error(self, step_id: str, error: Exception) -> None:
        """记录错误到历史。"""
        self._error_history.append({
            "step_id": step_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
        })
        
        # 限制历史记录数量
        if len(self._error_history) > 100:
            self._error_history.pop(0)
    
    def _execute_builtin_actions(
        self,
        step_id: str,
        error: Exception,
        workflow_state: Optional[Dict[str, Any]],
    ) -> None:
        """执行内置恢复动作。"""
        action = self._recovery_actions.get(step_id)
        if action is None:
            return
        
        # 检查是否有内置动作需要执行
        for name, handler in self._builtin_actions.items():
            if action.params.get(name):
                handler(step_id, error, workflow_state)
    
    def _screenshot_on_error(
        self,
        step_id: str,
        error: Exception,
        workflow_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        错误时截图的内置动作。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            workflow_state: 工作流状态
            
        Returns:
            截图信息
        """
        screenshot_info = {
            "step_id": step_id,
            "error": str(error),
            "timestamp": time.time(),
            "workflow_state": workflow_state,
            "action": "screenshot_on_error",
        }
        
        # 实际截图功能需要集成屏幕捕获模块
        # 这里只是记录信息
        return screenshot_info
    
    def _notify_on_error(
        self,
        step_id: str,
        error: Exception,
        workflow_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        错误时通知的内置动作。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            workflow_state: 工作流状态
            
        Returns:
            通知信息
        """
        notification = {
            "step_id": step_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "action": "notify_on_error",
        }
        
        # 实际通知功能需要集成通知模块
        # 这里只是记录信息
        return notification
    
    def _log_state(
        self,
        step_id: str,
        error: Exception,
        workflow_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        记录状态的内置动作。
        
        Args:
            step_id: 步骤ID
            error: 异常对象
            workflow_state: 工作流状态
            
        Returns:
            日志信息
        """
        log_entry = {
            "step_id": step_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "workflow_state": workflow_state,
            "action": "log_state",
        }
        
        # 记录到日志
        return log_entry
    
    def register_custom_action(
        self,
        name: str,
        handler: Callable[[str, Exception, Optional[Dict[str, Any]]], Any],
    ) -> None:
        """
        注册自定义恢复动作。
        
        Args:
            name: 动作名称
            handler: 处理函数
        """
        self._custom_actions[name] = handler
    
    def get_recovery_action(self, step_id: str) -> Optional[RecoveryAction]:
        """
        获取恢复动作。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            恢复动作，不存在则返回None
        """
        return self._recovery_actions.get(step_id)
    
    def get_recovery_policy(self, step_id: str) -> Optional[RecoveryPolicy]:
        """
        获取恢复策略。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            恢复策略，不存在则返回None
        """
        return self._recovery_policies.get(step_id)
    
    def remove_recovery_action(self, step_id: str) -> bool:
        """
        删除恢复动作。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            是否成功删除
        """
        if step_id in self._recovery_actions:
            del self._recovery_actions[step_id]
            return True
        return False
    
    def remove_recovery_policy(self, step_id: str) -> bool:
        """
        删除恢复策略。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            是否成功删除
        """
        if step_id in self._recovery_policies:
            del self._recovery_policies[step_id]
            return True
        return False
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        获取错误历史。
        
        Returns:
            错误历史列表
        """
        return list(self._error_history)
    
    def clear_error_history(self) -> None:
        """清空错误历史。"""
        self._error_history.clear()
    
    def get_retry_count(self, step_id: str) -> int:
        """
        获取重试次数。
        
        Args:
            step_id: 步骤ID
            
        Returns:
            当前重试次数
        """
        return self._retry_counts.get(step_id, 0)
    
    def list_recovery_configs(self) -> List[Dict[str, Any]]:
        """
        列出所有恢复配置。
        
        Returns:
            恢复配置列表
        """
        configs = []
        
        for step_id in set(list(self._recovery_actions.keys()) + list(self._recovery_policies.keys())):
            config = {"step_id": step_id}
            
            if step_id in self._recovery_actions:
                config["action"] = self._recovery_actions[step_id].to_dict()
            
            if step_id in self._recovery_policies:
                config["policy"] = self._recovery_policies[step_id].to_dict()
            
            configs.append(config)
        
        return configs
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "recovery_actions": {
                k: v.to_dict() for k, v in self._recovery_actions.items()
            },
            "recovery_policies": {
                k: v.to_dict() for k, v in self._recovery_policies.items()
            },
            "retry_counts": self._retry_counts,
            "error_history": self._error_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorRecovery":
        """从字典创建。"""
        recovery = cls()
        
        for step_id, action_data in data.get("recovery_actions", {}).items():
            recovery.register_recovery_action(step_id, RecoveryAction.from_dict(action_data))
        
        for step_id, policy_data in data.get("recovery_policies", {}).items():
            recovery.register_recovery_policy(step_id, RecoveryPolicy.from_dict(policy_data))
        
        recovery._retry_counts = data.get("retry_counts", {})
        recovery._error_history = data.get("error_history", [])
        
        return recovery


# 便捷函数
def create_retry_action(
    max_retries: int = 3,
    backoff: str = "fixed",
    retry_delay: float = 1.0,
) -> tuple:
    """
    创建重试恢复配置的便捷函数。
    
    Args:
        max_retries: 最大重试次数
        backoff: 退避策略
        retry_delay: 重试延迟
        
    Returns:
        (RecoveryAction, RecoveryPolicy) 元组
    """
    action = RecoveryAction(
        type=RecoveryType.RETRY,
        description=f"重试最多{max_retries}次",
    )
    
    policy = RecoveryPolicy(
        max_retries=max_retries,
        backoff=backoff,
        retry_delay=retry_delay,
    )
    
    return action, policy


def create_skip_action() -> RecoveryAction:
    """
    创建跳过恢复动作的便捷函数。
    
    Returns:
        RecoveryAction对象
    """
    return RecoveryAction(
        type=RecoveryType.SKIP,
        description="跳过当前步骤",
    )


def create_goto_action(target_step: str) -> RecoveryAction:
    """
    创建跳转恢复动作的便捷函数。
    
    Args:
        target_step: 目标步骤ID
        
    Returns:
        RecoveryAction对象
    """
    return RecoveryAction(
        type=RecoveryType.GOTO,
        target_step=target_step,
        description=f"跳转到步骤 {target_step}",
    )


def create_fallback_action(fallback_step: str) -> RecoveryAction:
    """
    创建备用恢复动作的便捷函数。
    
    Args:
        fallback_step: 备用步骤ID
        
    Returns:
        RecoveryAction对象
    """
    return RecoveryAction(
        type=RecoveryType.FALLBACK,
        target_step=fallback_step,
        description=f"执行备用步骤 {fallback_step}",
    )
