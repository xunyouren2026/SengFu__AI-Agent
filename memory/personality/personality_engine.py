"""
Personality Engine - 人格应用引擎

该模块负责将人格配置应用到LLM提示词，生成符合人格特质的输出。

核心功能:
- 系统提示词模板引擎
- 运行时人格切换
- 人格一致性维护
- 对话上下文注入
- 多轮对话人格保持

使用示例:
    engine = PersonalityEngine()
    
    # 应用人格到提示词
    prompt = engine.apply_to_prompt(
        config=personality_config,
        user_input="Hello, how are you?"
    )
    
    # 生成系统提示词
    system_prompt = engine.generate_system_prompt(config)
    
    # 处理对话
    response = engine.process_message(
        config=personality_config,
        messages=[...],
        user_input="..."
    )
"""

import re
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from copy import deepcopy
import logging

from . import (
    PersonalityConfig, PersonalityTrait, BehaviorPattern,
    CommunicationStyle, TraitDimension, BehaviorTrigger,
    CommunicationTone, ResponseLength, PersonalityError, InjectionStrategy
)

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """
    提示词上下文
    
    Attributes:
        user_input: 用户输入
        conversation_history: 对话历史
        system_context: 系统上下文
        agent_state: Agent状态
        metadata: 额外元数据
    """
    user_input: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    system_context: Dict[str, Any] = field(default_factory=dict)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, str]]:
        """获取最近的N条消息"""
        return self.conversation_history[-count:]
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息到历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_input": self.user_input,
            "conversation_history": self.conversation_history,
            "system_context": self.system_context,
            "agent_state": self.agent_state,
            "metadata": self.metadata
        }


@dataclass
class PromptComponents:
    """提示词组件"""
    prefix: str = ""           # 前缀
    personality_section: str = ""  # 人格部分
    context_section: str = ""  # 上下文部分
    constraints_section: str = ""  # 约束部分
    values_section: str = ""   # 价值观部分
    suffix: str = ""          # 后缀
    examples_section: str = ""  # 示例部分
    
    def combine(
        self, 
        order: List[str] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        组合所有组件
        
        Args:
            order: 组件顺序
            separator: 分隔符
            
        Returns:
            组合后的提示词
        """
        if order is None:
            order = [
                "prefix", "personality_section", "values_section",
                "constraints_section", "context_section", 
                "examples_section", "suffix"
            ]
        
        parts = []
        for key in order:
            value = getattr(self, key, "")
            if value:
                parts.append(value)
        
        return separator.join(parts)


@dataclass 
class PersonalityState:
    """人格状态追踪"""
    config: PersonalityConfig
    applied_count: int = 0
    last_applied_at: Optional[datetime] = None
    consistency_score: float = 1.0  # 0.0-1.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_consistency(self, score: float) -> None:
        """更新一致性分数"""
        self.consistency_score = score
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "applied_count": self.applied_count
        })


class TemplateEngine:
    """
    提示词模板引擎
    
    提供模板渲染和变量替换功能。
    """
    
    # 内置模板变量模式
    VARIABLE_PATTERN = re.compile(r'\{\{(\w+)(?::([^}]*))?\}\}')
    
    # 条件块模式
    CONDITIONAL_PATTERN = re.compile(
        r'\{% if (\w+) %\}(.*?)\{% endif %\}',
        re.DOTALL
    )
    
    def __init__(self):
        """初始化模板引擎"""
        self._custom_filters: Dict[str, Callable] = {}
        self._register_builtin_filters()
    
    def _register_builtin_filters(self) -> None:
        """注册内置过滤器"""
        self._custom_filters = {
            'upper': str.upper,
            'lower': str.lower,
            'capitalize': str.capitalize,
            'title': str.title,
            'strip': str.strip,
            'length': len,
            'first': lambda x: x[0] if x else "",
            'last': lambda x: x[-1] if x else "",
            'join': lambda x, sep=', ': sep.join(x) if isinstance(x, list) else str(x),
            'indent': lambda x, n=2: '\n'.join(' ' * n + line for line in x.split('\n')),
        }
    
    def register_filter(self, name: str, func: Callable) -> None:
        """
        注册自定义过滤器
        
        Args:
            name: 过滤器名称
            func: 过滤器函数
        """
        self._custom_filters[name] = func
    
    def render(
        self, 
        template: str, 
        context: Dict[str, Any],
        strict: bool = True
    ) -> str:
        """
        渲染模板
        
        Args:
            template: 模板字符串
            context: 渲染上下文
            strict: 是否严格模式（未找到变量时报错）
            
        Returns:
            渲染后的字符串
        """
        result = template
        
        # 处理条件块
        for match in self.CONDITIONAL_PATTERN.finditer(template):
            var_name = match.group(1)
            content = match.group(2)
            
            if context.get(var_name):
                result = result.replace(match.group(0), content)
            else:
                result = result.replace(match.group(0), "")
        
        # 处理变量替换
        for match in self.VARIABLE_PATTERN.finditer(template):
            full_match = match.group(0)
            var_name = match.group(1)
            filter_str = match.group(2)
            
            if var_name not in context:
                if strict:
                    raise ValueError(f"Undefined variable: {var_name}")
                continue
            
            value = context[var_name]
            
            # 应用过滤器
            if filter_str:
                value = self._apply_filters(value, filter_str)
            
            result = result.replace(full_match, str(value))
        
        return result
    
    def _apply_filters(self, value: Any, filter_str: str) -> Any:
        """应用过滤器链"""
        filters = filter_str.split('|')
        for f in filters:
            f = f.strip()
            if not f:
                continue
            
            # 解析带参数的过滤器
            if '(' in f and ')' in f:
                func_name, args_str = f.split('(', 1)
                args_str = args_str.rstrip(')')
                args = [a.strip() for a in args_str.split(',')]
            else:
                func_name = f
                args = []
            
            if func_name in self._custom_filters:
                func = self._custom_filters[func_name]
                if args:
                    value = func(value, *args)
                else:
                    value = func(value)
            elif hasattr(value, func_name):
                value = getattr(value, func_name)()
        
        return value


class PersonalityEngine:
    """
    人格应用引擎
    
    核心功能:
    - 将人格配置应用到LLM提示词
    - 生成符合人格特质的系统提示词
    - 维护人格一致性
    - 处理对话上下文
    
    Attributes:
        default_strategy: 默认注入策略
        enable_caching: 是否启用缓存
        template_engine: 模板引擎实例
    """
    
    # 默认系统提示词模板
    DEFAULT_SYSTEM_TEMPLATE = """{{# 角色定义 #}}
你是一个{{ personality.name }}。

{{# 性格特质 #}}
## 核心性格特质
{% if show_traits %}
{% for trait in traits %}
- {{ trait.dimension.value | title }}: {{ trait.intensity }}/5 - {{ trait.description }}
{% endfor %}
{% endif %}

{{# 价值观 #}}
## 核心价值观
{% if values %}
{% for value in values %}
- {{ value }}
{% endfor %}
{% else %}
- 诚实、友善、专业
{% endif %}

{{# 行为准则 #}}
## 行为准则
{% if behaviors %}
{% for behavior in behaviors %}
- {{ behavior.name }}: {{ behavior.description }}
{% endfor %}
{% endif %}

{{# 约束规则 #}}
## 必须遵守的约束
{% if constraints %}
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

{{# 沟通风格 #}}
## 沟通风格
- 语气: {{ communication.tone.value }}
- 详细程度: {{ communication.length.value }}
- 正式程度: {{ communication.formality_level }}/10
- 词汇水平: {{ communication.vocabulary_level }}

{{# 专业领域 #}}
{% if domain_expertise %}
## 专业领域
{{ domain_expertise | join: ", " }}
{% endif %}

{{# 用户输入 #}}
{{ user_input }}
"""
    
    def __init__(
        self,
        default_strategy: InjectionStrategy = InjectionStrategy.PREPEND,
        enable_caching: bool = True,
        cache_size: int = 100
    ):
        """
        初始化人格应用引擎
        
        Args:
            default_strategy: 默认注入策略
            enable_caching: 是否启用缓存
            cache_size: 缓存大小
        """
        self.default_strategy = default_strategy
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        
        self._template_engine = TemplateEngine()
        self._cache: Dict[str, str] = {}
        self._states: Dict[str, PersonalityState] = {}
        
        # 注册自定义过滤器
        self._setup_custom_filters()
    
    def _setup_custom_filters(self) -> None:
        """设置自定义过滤器"""
        self._template_engine.register_filter(
            'personality_tense',
            lambda intensity: "非常" if intensity >= 4 else ("略微" if intensity <= 2 else "")
        )
        
        self._template_engine.register_filter(
            'formal_level',
            lambda level: "正式" if level >= 7 else ("非正式" if level <= 3 else "中等正式")
        )
    
    def apply_to_prompt(
        self,
        config: PersonalityConfig,
        user_input: str,
        context: Optional[PromptContext] = None,
        template: Optional[str] = None,
        strategy: Optional[InjectionStrategy] = None,
        include_traits: bool = True,
        include_values: bool = True,
        include_constraints: bool = True,
        include_behaviors: bool = True,
        **template_vars
    ) -> str:
        """
        将人格配置应用到提示词
        
        Args:
            config: 人格配置
            user_input: 用户输入
            context: 可选的提示词上下文
            template: 可选的自定义模板
            strategy: 注入策略
            include_traits: 是否包含特质
            include_values: 是否包含价值观
            include_constraints: 是否包含约束
            include_behaviors: 是否包含行为模式
            **template_vars: 额外的模板变量
            
        Returns:
            应用人格后的完整提示词
        """
        strategy = strategy or self.default_strategy
        
        # 生成或获取缓存的提示词基础部分
        cache_key = self._generate_cache_key(config)
        
        # 构建渲染上下文
        render_context = self._build_render_context(
            config=config,
            user_input=user_input,
            context=context,
            include_traits=include_traits,
            include_values=include_values,
            include_constraints=include_constraints,
            include_behaviors=include_behaviors,
            **template_vars
        )
        
        # 生成人格部分
        if cache_key in self._cache and self.enable_caching:
            personality_section = self._cache[cache_key]
        else:
            personality_section = self._generate_personality_section(
                config=config,
                template=template,
                include_traits=include_traits,
                include_values=include_values,
                include_constraints=include_constraints,
                include_behaviors=include_behaviors
            )
            if self.enable_caching:
                self._update_cache(cache_key, personality_section)
        
        # 处理上下文注入
        final_prompt = self._inject_personality(
            user_input=user_input,
            personality_section=personality_section,
            context=context,
            strategy=strategy
        )
        
        # 更新状态
        self._update_state(config)
        
        return final_prompt
    
    def generate_system_prompt(
        self,
        config: PersonalityConfig,
        template: Optional[str] = None,
        include_traits: bool = True,
        include_values: bool = True,
        include_constraints: bool = True,
        **kwargs
    ) -> str:
        """
        生成系统提示词
        
        Args:
            config: 人格配置
            template: 可选的自定义模板
            include_traits: 是否包含特质
            include_values: 是否包含价值观
            include_constraints: 是否包含约束
            **kwargs: 额外的模板变量
            
        Returns:
            系统提示词
        """
        if template is None:
            template = self.DEFAULT_SYSTEM_TEMPLATE
        
        context = {
            'personality': config,
            'traits': config.traits,
            'values': config.values,
            'behaviors': config.behaviors,
            'constraints': config.constraints,
            'communication': config.communication_style,
            'domain_expertise': config.domain_expertise,
            'show_traits': include_traits,
            **kwargs
        }
        
        return self._template_engine.render(template, context)
    
    def _generate_personality_section(
        self,
        config: PersonalityConfig,
        template: Optional[str] = None,
        include_traits: bool = True,
        include_values: bool = True,
        include_constraints: bool = True,
        include_behaviors: bool = True
    ) -> str:
        """
        生成人格部分内容
        
        Args:
            config: 人格配置
            template: 可选的自定义模板
            include_traits: 是否包含特质
            include_values: 是否包含价值观
            include_constraints: 是否包含约束
            include_behaviors: 是否包含行为模式
            
        Returns:
            人格部分的字符串
        """
        if template:
            return self.generate_system_prompt(
                config,
                template=template,
                include_traits=include_traits,
                include_values=include_values,
                include_constraints=include_constraints
            )
        
        sections = []
        
        # 角色定义
        sections.append(f"# {config.name}")
        
        # 性格特质
        if include_traits and config.traits:
            sections.append("\n## 性格特质")
            for trait in config.traits:
                intensity_word = self._get_intensity_word(trait.intensity)
                sections.append(
                    f"- **{trait.dimension.value.capitalize()}** "
                    f"({intensity_word}, {trait.intensity}/5): {trait.description}"
                )
        
        # 价值观
        if include_values and config.values:
            sections.append("\n## 核心价值观")
            for value in config.values:
                sections.append(f"- {value}")
        
        # 行为模式
        if include_behaviors and config.behaviors:
            sections.append("\n## 行为准则")
            for behavior in config.behaviors:
                if behavior.enabled:
                    sections.append(
                        f"- **{behavior.name}**: {behavior.description}"
                    )
        
        # 约束规则
        if include_constraints and config.constraints:
            sections.append("\n## 必须遵守的约束")
            for constraint in config.constraints:
                sections.append(f"- {constraint}")
        
        # 沟通风格
        sections.append("\n## 沟通风格")
        sections.append(f"- 语气: {config.communication_style.tone.value}")
        sections.append(f"- 回复长度: {config.communication_style.length.value}")
        sections.append(
            f"- 正式程度: {config.communication_style.formality_level}/10"
        )
        sections.append(
            f"- 词汇水平: {config.communication_style.vocabulary_level}"
        )
        
        # 专业领域
        if config.domain_expertise:
            sections.append("\n## 专业领域")
            sections.append(", ".join(config.domain_expertise))
        
        return "\n".join(sections)
    
    def _get_intensity_word(self, intensity: int) -> str:
        """获取强度描述词"""
        words = {
            1: "极低",
            2: "低",
            3: "中等",
            4: "高",
            5: "极高"
        }
        return words.get(intensity, "中等")
    
    def _build_render_context(
        self,
        config: PersonalityConfig,
        user_input: str,
        context: Optional[PromptContext],
        include_traits: bool,
        include_values: bool,
        include_constraints: bool,
        include_behaviors: bool,
        **extra_vars
    ) -> Dict[str, Any]:
        """构建渲染上下文"""
        render_ctx = {
            'personality': config,
            'traits': config.traits if include_traits else [],
            'values': config.values if include_values else [],
            'behaviors': config.behaviors if include_behaviors else [],
            'constraints': config.constraints if include_constraints else [],
            'communication': config.communication_style,
            'domain_expertise': config.domain_expertise,
            'user_input': user_input,
            'timestamp': datetime.now().isoformat(),
            'show_traits': include_traits,
        }
        
        # 添加上下文信息
        if context:
            render_ctx['conversation_history'] = context.conversation_history
            render_ctx['system_context'] = context.system_context
            render_ctx['agent_state'] = context.agent_state
        
        # 合并额外变量
        render_ctx.update(extra_vars)
        
        return render_ctx
    
    def _inject_personality(
        self,
        user_input: str,
        personality_section: str,
        context: Optional[PromptContext],
        strategy: InjectionStrategy
    ) -> str:
        """
        根据策略注入人格
        
        Args:
            user_input: 用户输入
            personality_section: 人格部分内容
            context: 上下文
            strategy: 注入策略
            
        Returns:
            处理后的输入
        """
        if strategy == InjectionStrategy.PREPEND:
            return f"{personality_section}\n\n## 用户输入\n\n{user_input}"
        
        elif strategy == InjectionStrategy.APPEND:
            return f"{user_input}\n\n---\n\n{personality_section}"
        
        elif strategy == InjectionStrategy.REPLACE:
            return personality_section
        
        elif strategy == InjectionStrategy.MERGE:
            # 合并用户输入到人格部分
            merged = self._merge_with_context(
                personality_section, 
                user_input, 
                context
            )
            return merged
        
        elif strategy == InjectionStrategy.CONDITIONAL:
            return self._conditional_injection(
                user_input, 
                personality_section, 
                context
            )
        
        return user_input
    
    def _merge_with_context(
        self,
        personality_section: str,
        user_input: str,
        context: Optional[PromptContext]
    ) -> str:
        """合并人格与上下文"""
        parts = [personality_section]
        
        if context:
            # 添加对话历史摘要
            if context.conversation_history:
                history_summary = self._summarize_history(
                    context.conversation_history
                )
                parts.append(f"\n## 对话历史\n\n{history_summary}")
            
            # 添加系统上下文
            if context.system_context:
                context_str = json.dumps(
                    context.system_context, 
                    ensure_ascii=False
                )
                parts.append(f"\n## 系统上下文\n\n{context_str}")
        
        parts.append(f"\n## 当前输入\n\n{user_input}")
        
        return "\n".join(parts)
    
    def _summarize_history(
        self, 
        history: List[Dict[str, str]],
        max_messages: int = 5
    ) -> str:
        """生成对话历史摘要"""
        if not history:
            return "无对话历史"
        
        recent = history[-max_messages:]
        lines = []
        
        for msg in recent:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # 截断长消息
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"**{role}**: {content}")
        
        return "\n".join(lines)
    
    def _conditional_injection(
        self,
        user_input: str,
        personality_section: str,
        context: Optional[PromptContext]
    ) -> str:
        """条件注入"""
        if not context:
            return f"{personality_section}\n\n{user_input}"
        
        # 根据上下文特征决定注入内容
        input_lower = user_input.lower()
        
        # 检测是否涉及特定主题
        conditional_sections = []
        
        if any(word in input_lower for word in ['代码', '编程', '开发', 'code', 'programming']):
            conditional_sections.append("## 代码相关指南\n- 使用清晰、简洁的代码风格\n- 添加必要的注释说明")
        
        if any(word in input_lower for word in ['分析', '数据', '统计', 'analyze', 'data']):
            conditional_sections.append("## 分析指南\n- 提供数据支持的观点\n- 列出分析方法和假设")
        
        if any(word in input_lower for word in ['创意', '设计', 'creative', 'design']):
            conditional_sections.append("## 创意指南\n- 提供多个方案选项\n- 鼓励创新思维")
        
        if conditional_sections:
            personality_section += "\n" + "\n".join(conditional_sections)
        
        return f"{personality_section}\n\n## 用户输入\n\n{user_input}"
    
    def _generate_cache_key(self, config: PersonalityConfig) -> str:
        """生成缓存键"""
        data = {
            'name': config.name,
            'version': config.version,
            'fingerprint': config.get_fingerprint()
        }
        return hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def _update_cache(self, key: str, value: str) -> None:
        """更新缓存"""
        if len(self._cache) >= self.cache_size:
            # 简单的FIFO缓存清理
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[key] = value
    
    def _update_state(self, config: PersonalityConfig) -> None:
        """更新人格状态"""
        key = config.name
        
        if key not in self._states:
            self._states[key] = PersonalityState(config=config)
        
        state = self._states[key]
        state.applied_count += 1
        state.last_applied_at = datetime.now()
    
    def get_state(self, name: str) -> Optional[PersonalityState]:
        """获取人格状态"""
        return self._states.get(name)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
    
    def switch_personality(
        self,
        current_config: PersonalityConfig,
        new_config: PersonalityConfig,
        context: Optional[PromptContext] = None,
        transition_prompt: bool = True
    ) -> str:
        """
        切换人格
        
        Args:
            current_config: 当前人格配置
            new_config: 新人格配置
            context: 上下文
            transition_prompt: 是否生成过渡提示
            
        Returns:
            过渡提示（如果需要）
        """
        # 记录切换
        if current_config.name in self._states:
            old_state = self._states[current_config.name]
            logger.info(
                f"Switching personality from {current_config.name} "
                f"(applied {old_state.applied_count} times) to {new_config.name}"
            )
        
        # 初始化新状态
        self._states[new_config.name] = PersonalityState(config=new_config)
        
        # 生成过渡提示
        if transition_prompt:
            return self._generate_transition_prompt(current_config, new_config)
        
        return ""
    
    def _generate_transition_prompt(
        self,
        old_config: PersonalityConfig,
        new_config: PersonalityConfig
    ) -> str:
        """生成人格切换过渡提示"""
        differences = self._identify_differences(old_config, new_config)
        
        lines = [
            "[人格切换]",
            f"从 {old_config.name} 切换到 {new_config.name}",
        ]
        
        if differences:
            lines.append("\n主要变化:")
            for diff in differences:
                lines.append(f"- {diff}")
        
        return "\n".join(lines)
    
    def _identify_differences(
        self,
        old_config: PersonalityConfig,
        new_config: PersonalityConfig
    ) -> List[str]:
        """识别人格配置差异"""
        differences = []
        
        # 检查特质变化
        old_traits = {t.dimension: t.intensity for t in old_config.traits}
        new_traits = {t.dimension: t.intensity for t in new_config.traits}
        
        for dimension, old_intensity in old_traits.items():
            new_intensity = new_traits.get(dimension, old_intensity)
            if old_intensity != new_intensity:
                diff = new_intensity - old_intensity
                direction = "提高" if diff > 0 else "降低"
                differences.append(
                    f"{dimension.value}: {old_intensity} -> {new_intensity} ({direction})"
                )
        
        # 检查沟通风格变化
        if old_config.communication_style.tone != new_config.communication_style.tone:
            differences.append(
                f"语气: {old_config.communication_style.tone.value} -> "
                f"{new_config.communication_style.tone.value}"
            )
        
        return differences
    
    def maintain_consistency(
        self,
        config: PersonalityConfig,
        conversation_history: List[Dict[str, str]]
    ) -> float:
        """
        维护人格一致性
        
        Args:
            config: 人格配置
            conversation_history: 对话历史
            
        Returns:
            一致性分数 (0.0-1.0)
        """
        if not conversation_history:
            return 1.0
        
        # 简单的启发式一致性检查
        inconsistencies = 0
        total = len(conversation_history)
        
        # 检查回复是否符合人格
        for msg in conversation_history:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                
                # 检查语气一致性
                if config.communication_style.tone == CommunicationTone.FORMAL:
                    # 正式语气应该避免俚语
                    informal_words = ['超', '太', '牛', '酷']
                    if any(word in content for word in informal_words):
                        inconsistencies += 0.5
                
                # 检查长度一致性
                expected_length = self._get_expected_length(
                    config.communication_style.length
                )
                actual_length = len(content)
                if actual_length > expected_length * 2:
                    inconsistencies += 0.3
        
        score = max(0.0, 1.0 - (inconsistencies / total))
        
        # 更新状态
        if config.name in self._states:
            self._states[config.name].update_consistency(score)
        
        return score
    
    def _get_expected_length(self, length_pref: ResponseLength) -> int:
        """获取期望的回复长度"""
        length_map = {
            ResponseLength.CONCISE: 50,
            ResponseLength.BRIEF: 100,
            ResponseLength.MODERATE: 300,
            ResponseLength.DETAILED: 500,
            ResponseLength.COMPREHENSIVE: 1000
        }
        return length_map.get(length_pref, 300)
    
    def apply_behavior(
        self,
        config: PersonalityConfig,
        behavior_name: str,
        context: Optional[PromptContext] = None
    ) -> Optional[str]:
        """
        应用特定行为模式
        
        Args:
            config: 人格配置
            behavior_name: 行为名称
            context: 上下文
            
        Returns:
            行为指令（如果有）
        """
        for behavior in config.behaviors:
            if behavior.name.lower() == behavior_name.lower():
                if context:
                    ctx_dict = context.to_dict() if context else {}
                    if not behavior.should_execute(ctx_dict):
                        return None
                
                # 生成行为指令
                instructions = []
                instructions.append(f"## 执行行为: {behavior.name}")
                instructions.append(behavior.description)
                
                if behavior.actions:
                    instructions.append("\n### 操作步骤:")
                    for i, action in enumerate(behavior.actions, 1):
                        instructions.append(f"{i}. {action}")
                
                return "\n".join(instructions)
        
        return None
    
    def get_applicable_behaviors(
        self,
        config: PersonalityConfig,
        context: Dict[str, Any]
    ) -> List[BehaviorPattern]:
        """
        获取适用的行为模式
        
        Args:
            config: 人格配置
            context: 执行上下文
            
        Returns:
            适用的行为列表
        """
        applicable = []
        
        for behavior in config.behaviors:
            if behavior.enabled and behavior.should_execute(context):
                applicable.append(behavior)
        
        # 按优先级排序
        applicable.sort(key=lambda b: b.priority, reverse=True)
        
        return applicable
    
    def format_response(
        self,
        content: str,
        config: PersonalityConfig,
        target_format: Optional[str] = None
    ) -> str:
        """
        格式化响应内容
        
        Args:
            content: 原始内容
            config: 人格配置
            target_format: 目标格式
            
        Returns:
            格式化后的内容
        """
        formatted = content
        
        # 应用词汇水平
        vocab_level = config.communication_style.vocabulary_level
        
        # 根据正式程度调整
        if config.communication_style.formality_level >= 7:
            # 高正式程度：确保完整句子
            if not formatted.endswith(('.', '!', '?', '。', '！', '？')):
                formatted += '。'
        
        # 根据回复长度偏好调整
        length_pref = config.communication_style.length
        
        if length_pref == ResponseLength.CONCISE and len(formatted) > 100:
            # 截断为更短的形式
            sentences = re.split(r'[。!?]', formatted)
            if len(sentences) > 1:
                formatted = sentences[0] + '。'
        
        elif length_pref == ResponseLength.DETAILED and len(formatted) < 100:
            # 添加解释性内容
            formatted += f"\n\n[基于您的人格配置({config.name})进行回复]"
        
        # 应用目标格式
        if target_format == 'markdown':
            formatted = self._apply_markdown_formatting(formatted)
        elif target_format == 'plain':
            formatted = self._remove_markdown_formatting(formatted)
        
        return formatted
    
    def _apply_markdown_formatting(self, content: str) -> str:
        """应用Markdown格式化"""
        # 确保使用适当的标题层级
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                # 首个无标题段落作为引言
                if not formatted_lines:
                    line = f"> {line}"
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _remove_markdown_formatting(self, content: str) -> str:
        """移除Markdown格式化"""
        # 移除粗体
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        # 移除斜体
        content = re.sub(r'\*(.+?)\*', r'\1', content)
        # 移除标题
        content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
        # 移除链接
        content = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', content)
        
        return content


def create_engine(
    strategy: InjectionStrategy = InjectionStrategy.PREPEND,
    enable_caching: bool = True
) -> PersonalityEngine:
    """
    工厂函数：创建人格应用引擎
    
    Args:
        strategy: 注入策略
        enable_caching: 是否启用缓存
        
    Returns:
        PersonalityEngine实例
    """
    return PersonalityEngine(
        default_strategy=strategy,
        enable_caching=enable_caching
    )
