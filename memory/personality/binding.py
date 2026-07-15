"""
Agent Personality Binding - Agent人格绑定

该模块提供Agent与人格的关联和生命周期管理功能。

核心功能:
- Agent与人格关联
- 绑定生命周期管理
- 多Agent人格隔离
- 人格状态同步

使用示例:
    binder = AgentBinder()
    
    # 绑定人格到Agent
    binder.bind("agent_001", personality_config)
    
    # 获取Agent的人格
    config = binder.get_bound_personality("agent_001")
    
    # 解除绑定
    binder.unbind("agent_001")
"""

import threading
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from . import (
    PersonalityConfig, BindingError,
    PersonalityEngine, InjectionStrategy
)
from .personality_engine import PersonalityState

logger = logging.getLogger(__name__)


class BindingStatus(Enum):
    """绑定状态"""
    UNBOUND = "unbound"
    BOUND = "bound"
    SUSPENDED = "suspended"
    RELEASED = "released"


class BindingScope(Enum):
    """绑定作用域"""
    SHARED = "shared"         # 共享
    ISOLATED = "isolated"     # 隔离
    INHERITED = "inherited"   # 继承


@dataclass
class BindingInfo:
    """绑定信息"""
    agent_id: str
    personality_id: str
    bound_at: datetime
    status: BindingStatus
    scope: BindingScope
    version: str
    fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonalityContext:
    """
    人格上下文
    
    在Agent运行时维护人格相关状态。
    
    Attributes:
        agent_id: Agent ID
        personality_id: 人格ID
        current_config: 当前配置
        config_stack: 配置栈（用于嵌套）
        history: 历史配置
    """
    agent_id: str
    personality_id: str
    current_config: PersonalityConfig
    config_stack: List[PersonalityConfig] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def push_config(self, config: PersonalityConfig) -> None:
        """压入新配置"""
        self.config_stack.append(self.current_config)
        self.current_config = config
        self._record_history("push")
    
    def pop_config(self) -> Optional[PersonalityConfig]:
        """弹出配置"""
        if self.config_stack:
            old_config = self.current_config
            self.current_config = self.config_stack.pop()
            self._record_history("pop")
            return old_config
        return None
    
    def _record_history(self, action: str) -> None:
        """记录历史"""
        self.history.append({
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "config_name": self.current_config.name,
            "config_version": self.current_config.version
        })


@dataclass
class PromptTemplate:
    """
    提示词模板
    
    Attributes:
        name: 模板名称
        system_template: 系统提示词模板
        user_template: 用户提示词模板
        variables: 变量定义
    """
    name: str
    system_template: str
    user_template: str = "{user_input}"
    variables: Dict[str, str] = field(default_factory=dict)


class AgentBinder:
    """
    Agent人格绑定器
    
    管理Agent与人格之间的关联关系，提供：
    - 人格绑定/解绑
    - 多Agent人格隔离
    - 人格状态同步
    - 运行时人格切换
    
    Attributes:
        enable_isolation: 是否启用隔离
        allow_shared: 是否允许共享人格
        auto_validate: 是否自动验证
    """
    
    def __init__(
        self,
        enable_isolation: bool = True,
        allow_shared: bool = False,
        auto_validate: bool = True
    ):
        """
        初始化绑定器
        
        Args:
            enable_isolation: 是否启用隔离
            allow_shared: 是否允许共享人格
            auto_validate: 是否自动验证
        """
        self.enable_isolation = enable_isolation
        self.allow_shared = allow_shared
        self.auto_validate = auto_validate
        
        self._bindings: Dict[str, BindingInfo] = {}
        self._personality_refs: Dict[str, Set[str]] = {}  # 人格ID -> Agent集合
        self._contexts: Dict[str, PersonalityContext] = {}
        self._lock = threading.RLock()
        self._engine = PersonalityEngine()
        self._hooks: Dict[str, List[Callable]] = {
            'on_bind': [],
            'on_unbind': [],
            'on_switch': [],
            'on_suspend': [],
            'on_resume': []
        }
    
    def bind(
        self,
        agent_id: str,
        config: PersonalityConfig,
        scope: BindingScope = BindingScope.ISOLATED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BindingInfo:
        """
        绑定人格到Agent
        
        Args:
            agent_id: Agent ID
            config: 人格配置
            scope: 绑定作用域
            metadata: 额外元数据
            
        Returns:
            BindingInfo对象
            
        Raises:
            BindingError: 绑定失败
        """
        with self._lock:
            # 检查是否已绑定
            if agent_id in self._bindings:
                raise BindingError(f"Agent {agent_id} is already bound")
            
            # 检查人格是否已被其他Agent使用（隔离模式）
            if self.enable_isolation and scope == BindingScope.ISOLATED:
                personality_id = self._get_personality_id(config)
                if personality_id in self._personality_refs:
                    raise BindingError(
                        f"Personality '{config.name}' is already bound to another agent "
                        f"in isolated mode"
                    )
            
            # 验证配置
            if self.auto_validate:
                from .validator import Validator
                validator = Validator()
                result = validator.validate(config)
                if not result.is_valid:
                    errors = [str(e) for e in result.errors]
                    raise BindingError(f"Validation failed: {'; '.join(errors)}")
            
            # 创建绑定信息
            binding = BindingInfo(
                agent_id=agent_id,
                personality_id=self._get_personality_id(config),
                bound_at=datetime.now(),
                status=BindingStatus.BOUND,
                scope=scope,
                version=config.version,
                fingerprint=config.get_fingerprint(),
                metadata=metadata or {}
            )
            
            # 保存绑定
            self._bindings[agent_id] = binding
            
            # 更新人格引用
            if binding.personality_id not in self._personality_refs:
                self._personality_refs[binding.personality_id] = set()
            self._personality_refs[binding.personality_id].add(agent_id)
            
            # 创建上下文
            self._contexts[agent_id] = PersonalityContext(
                agent_id=agent_id,
                personality_id=binding.personality_id,
                current_config=config
            )
            
            # 触发钩子
            self._trigger_hook('on_bind', agent_id, config)
            
            logger.info(f"Bound personality '{config.name}' to agent '{agent_id}'")
            
            return binding
    
    def unbind(self, agent_id: str, force: bool = False) -> bool:
        """
        解除绑定
        
        Args:
            agent_id: Agent ID
            force: 是否强制解除
            
        Returns:
            是否成功
            
        Raises:
            BindingError: 解除失败
        """
        with self._lock:
            if agent_id not in self._bindings:
                if not force:
                    raise BindingError(f"Agent {agent_id} is not bound")
                return False
            
            binding = self._bindings[agent_id]
            
            # 检查是否可以解除
            if binding.status == BindingStatus.SUSPENDED and not force:
                raise BindingError(
                    f"Agent {agent_id} is suspended, use force=True to unbind"
                )
            
            # 移除引用
            if binding.personality_id in self._personality_refs:
                self._personality_refs[binding.personality_id].discard(agent_id)
                if not self._personality_refs[binding.personality_id]:
                    del self._personality_refs[binding.personality_id]
            
            # 更新状态
            binding.status = BindingStatus.RELEASED
            
            # 触发钩子
            config = self._contexts[agent_id].current_config
            self._trigger_hook('on_unbind', agent_id, config)
            
            # 清理
            del self._bindings[agent_id]
            if agent_id in self._contexts:
                del self._contexts[agent_id]
            
            logger.info(f"Unbound agent '{agent_id}'")
            
            return True
    
    def suspend(self, agent_id: str) -> None:
        """
        暂停绑定
        
        Args:
            agent_id: Agent ID
        """
        with self._lock:
            if agent_id not in self._bindings:
                raise BindingError(f"Agent {agent_id} is not bound")
            
            binding = self._bindings[agent_id]
            binding.status = BindingStatus.SUSPENDED
            
            self._trigger_hook('on_suspend', agent_id, binding)
            logger.info(f"Suspended binding for agent '{agent_id}'")
    
    def resume(self, agent_id: str) -> None:
        """
        恢复绑定
        
        Args:
            agent_id: Agent ID
        """
        with self._lock:
            if agent_id not in self._bindings:
                raise BindingError(f"Agent {agent_id} is not bound")
            
            binding = self._bindings[agent_id]
            
            if binding.status != BindingStatus.SUSPENDED:
                raise BindingError(
                    f"Agent {agent_id} is not suspended (status: {binding.status.value})"
                )
            
            binding.status = BindingStatus.BOUND
            
            self._trigger_hook('on_resume', agent_id, binding)
            logger.info(f"Resumed binding for agent '{agent_id}'")
    
    def get_bound_personality(self, agent_id: str) -> Optional[PersonalityConfig]:
        """
        获取Agent绑定的人格
        
        Args:
            agent_id: Agent ID
            
        Returns:
            PersonalityConfig或None
        """
        context = self._contexts.get(agent_id)
        if context:
            return context.current_config
        return None
    
    def get_binding_info(self, agent_id: str) -> Optional[BindingInfo]:
        """
        获取绑定信息
        
        Args:
            agent_id: Agent ID
            
        Returns:
            BindingInfo或None
        """
        return self._bindings.get(agent_id)
    
    def get_context(self, agent_id: str) -> Optional[PersonalityContext]:
        """
        获取人格上下文
        
        Args:
            agent_id: Agent ID
            
        Returns:
            PersonalityContext或None
        """
        return self._contexts.get(agent_id)
    
    def switch_personality(
        self,
        agent_id: str,
        new_config: PersonalityConfig,
        preserve_history: bool = True
    ) -> None:
        """
        切换人格
        
        Args:
            agent_id: Agent ID
            new_config: 新配置
            preserve_history: 是否保留历史
            
        Raises:
            BindingError: 切换失败
        """
        with self._lock:
            if agent_id not in self._contexts:
                raise BindingError(f"Agent {agent_id} is not bound")
            
            context = self._contexts[agent_id]
            old_config = context.current_config
            
            # 验证新配置
            if self.auto_validate:
                from .validator import Validator
                validator = Validator()
                result = validator.validate(new_config)
                if not result.is_valid:
                    errors = [str(e) for e in result.errors]
                    raise BindingError(f"Validation failed: {'; '.join(errors)}")
            
            # 切换配置
            context.push_config(new_config)
            
            # 更新绑定信息
            binding = self._bindings[agent_id]
            binding.personality_id = self._get_personality_id(new_config)
            binding.version = new_config.version
            binding.fingerprint = new_config.get_fingerprint()
            
            # 触发钩子
            self._trigger_hook('on_switch', agent_id, old_config, new_config)
            
            logger.info(
                f"Switched agent '{agent_id}' from '{old_config.name}' to '{new_config.name}'"
            )
    
    def apply_to_prompt(
        self,
        agent_id: str,
        user_input: str,
        **kwargs
    ) -> str:
        """
        为Agent生成应用人格的提示词
        
        Args:
            agent_id: Agent ID
            user_input: 用户输入
            **kwargs: 额外参数
            
        Returns:
            应用人格后的提示词
            
        Raises:
            BindingError: Agent未绑定
        """
        config = self.get_bound_personality(agent_id)
        if not config:
            raise BindingError(f"Agent {agent_id} is not bound")
        
        return self._engine.apply_to_prompt(
            config=config,
            user_input=user_input,
            **kwargs
        )
    
    def generate_system_prompt(
        self,
        agent_id: str,
        **kwargs
    ) -> str:
        """
        为Agent生成系统提示词
        
        Args:
            agent_id: Agent ID
            **kwargs: 额外参数
            
        Returns:
            系统提示词
            
        Raises:
            BindingError: Agent未绑定
        """
        config = self.get_bound_personality(agent_id)
        if not config:
            raise BindingError(f"Agent {agent_id} is not bound")
        
        return self._engine.generate_system_prompt(config=config, **kwargs)
    
    def list_bound_agents(self) -> List[str]:
        """
        列出所有绑定的Agent
        
        Returns:
            Agent ID列表
        """
        return list(self._bindings.keys())
    
    def list_agents_for_personality(self, personality_id: str) -> List[str]:
        """
        列出使用特定人格的所有Agent
        
        Args:
            personality_id: 人格ID
            
        Returns:
            Agent ID列表
        """
        refs = self._personality_refs.get(personality_id, set())
        return list(refs)
    
    def get_all_bindings(self) -> Dict[str, BindingInfo]:
        """
        获取所有绑定信息
        
        Returns:
            绑定信息字典
        """
        return self._bindings.copy()
    
    def register_hook(
        self,
        event: str,
        callback: Callable
    ) -> None:
        """
        注册钩子回调
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self._hooks:
            self._hooks[event].append(callback)
        else:
            logger.warning(f"Unknown hook event: {event}")
    
    def _trigger_hook(
        self,
        event: str,
        agent_id: str,
        *args
    ) -> None:
        """触发钩子"""
        if event not in self._hooks:
            return
        
        for callback in self._hooks[event]:
            try:
                callback(agent_id, *args)
            except Exception as e:
                logger.warning(f"Hook callback error: {e}")
    
    def _get_personality_id(self, config: PersonalityConfig) -> str:
        """获取人格ID"""
        data = f"{config.name}:{config.version}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def is_bound(self, agent_id: str) -> bool:
        """检查Agent是否已绑定"""
        return agent_id in self._bindings
    
    def is_active(self, agent_id: str) -> bool:
        """检查Agent绑定是否活跃"""
        binding = self._bindings.get(agent_id)
        return binding is not None and binding.status == BindingStatus.BOUND
    
    def export_binding_state(self) -> str:
        """
        导出绑定状态
        
        Returns:
            JSON格式的状态
        """
        state = {
            "bindings": {
                agent_id: {
                    "personality_id": info.personality_id,
                    "bound_at": info.bound_at.isoformat(),
                    "status": info.status.value,
                    "scope": info.scope.value,
                    "version": info.version
                }
                for agent_id, info in self._bindings.items()
            },
            "personality_refs": {
                pid: list(agents)
                for pid, agents in self._personality_refs.items()
            },
            "stats": {
                "total_bindings": len(self._bindings),
                "isolated_count": sum(
                    1 for b in self._bindings.values() 
                    if b.scope == BindingScope.ISOLATED
                ),
                "shared_count": sum(
                    1 for b in self._bindings.values() 
                    if b.scope == BindingScope.SHARED
                )
            }
        }
        
        return json.dumps(state, indent=2)


def create_binder(
    enable_isolation: bool = True,
    allow_shared: bool = False
) -> AgentBinder:
    """
    工厂函数：创建绑定器
    
    Args:
        enable_isolation: 是否启用隔离
        allow_shared: 是否允许共享
        
    Returns:
        AgentBinder实例
    """
    return AgentBinder(
        enable_isolation=enable_isolation,
        allow_shared=allow_shared
    )
