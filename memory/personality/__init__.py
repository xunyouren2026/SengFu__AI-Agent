"""
AGI Unified Framework - SOUL.md Personality Engine Module

该模块提供完整的人格引擎实现，支持SOUL.md格式的人格配置解析、
应用、验证和版本管理。

核心组件:
- SoulParser: SOUL.md语法解析器
- PersonalityEngine: 人格应用引擎
- Validator: 人格配置验证器
- Loader: 统一人格加载器
- VersionManager: 版本管理器
- AgentBinder: Agent人格绑定器

使用示例:
    from agi_unified_framework.memory.personality import (
        SoulParser, PersonalityEngine, Loader, PersonalityConfig
    )
    
    # 加载人格配置
    loader = Loader()
    config = loader.load_from_file("path/to/personality.md")
    
    # 解析并验证
    parser = SoulParser()
    parsed = parser.parse(config.raw_content)
    validator = Validator()
    validator.validate(parsed)
    
    # 应用到提示词
    engine = PersonalityEngine()
    prompt = engine.apply_to_prompt(parsed, user_input="Hello")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from enum import Enum
import re
import hashlib
import json
from datetime import datetime


# =============================================================================
# 基础数据结构和枚举
# =============================================================================

class TraitDimension(Enum):
    """大五人格特质维度"""
    OPENNESS = "openness"           # 开放性
    CONSCIENTIOUSNESS = "conscientiousness"  # 尽责性
    EXTRAVERSION = "extraversion"   # 外向性
    AGREEABLENESS = "agreeableness" # 宜人性
    NEUROTICISM = "neuroticism"    # 神经质


class TraitIntensity(Enum):
    """特质强度级别"""
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    MAXIMAL = 5


class BehaviorTrigger(Enum):
    """行为触发类型"""
    ON_REQUEST = "on_request"           # 用户请求时触发
    ON_CONTEXT = "on_context"          # 特定上下文触发
    ON_EMOTION = "on_emotion"           # 情绪相关触发
    ALWAYS = "always"                  # 始终执行
    CONDITIONAL = "conditional"         # 条件执行


class CommunicationTone(Enum):
    """沟通语气风格"""
    FORMAL = "formal"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    EMPATHETIC = "empathetic"


class ResponseLength(Enum):
    """回复长度偏好"""
    CONCISE = "concise"         # 简洁
    BRIEF = "brief"             # 简短
    MODERATE = "moderate"       # 适中
    DETAILED = "detailed"       # 详细
    COMPREHENSIVE = "comprehensive"  # 全面


class InjectionStrategy(Enum):
    """注入策略"""
    PREPEND = "prepend"      # 前置
    APPEND = "append"        # 追加
    REPLACE = "replace"      # 替换
    MERGE = "merge"          # 合并
    CONDITIONAL = "conditional"  # 条件注入


# =============================================================================
# 核心数据类
# =============================================================================

@dataclass
class PersonalityTrait:
    """
    人格特质定义
    
    Attributes:
        dimension: 特质维度（大五人格）
        intensity: 强度级别 (1-5)
        description: 特质描述
        examples: 具体示例列表
        manifestations: 行为表现
    """
    dimension: TraitDimension
    intensity: int  # 1-5
    description: str
    examples: List[str] = field(default_factory=list)
    manifestations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """验证强度值"""
        if not 1 <= self.intensity <= 5:
            raise ValueError(f"Trait intensity must be between 1-5, got {self.intensity}")
        if not isinstance(self.dimension, TraitDimension):
            raise TypeError(f"dimension must be TraitDimension enum, got {type(self.dimension)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension": self.dimension.value,
            "intensity": self.intensity,
            "description": self.description,
            "examples": self.examples,
            "manifestations": self.manifestations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityTrait":
        """从字典创建"""
        return cls(
            dimension=TraitDimension(data["dimension"]),
            intensity=data["intensity"],
            description=data["description"],
            examples=data.get("examples", []),
            manifestations=data.get("manifestations", [])
        )


@dataclass
class BehaviorPattern:
    """
    行为模式定义
    
    Attributes:
        name: 行为名称
        description: 行为描述
        trigger: 触发条件
        conditions: 触发条件详情
        actions: 执行动作
        priority: 优先级
        examples: 示例
    """
    name: str
    description: str
    trigger: BehaviorTrigger
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    priority: int = 0
    examples: List[str] = field(default_factory=list)
    enabled: bool = True
    
    def __post_init__(self):
        """验证触发器类型"""
        if not isinstance(self.trigger, BehaviorTrigger):
            raise TypeError(f"trigger must be BehaviorTrigger enum")
    
    def should_execute(self, context: Dict[str, Any]) -> bool:
        """
        判断是否应执行此行为
        
        Args:
            context: 执行上下文
            
        Returns:
            是否应该执行
        """
        if not self.enabled:
            return False
            
        if self.trigger == BehaviorTrigger.ALWAYS:
            return True
            
        if self.trigger == BehaviorTrigger.ON_REQUEST:
            return context.get("type") == "request"
            
        if self.trigger == BehaviorTrigger.ON_CONTEXT:
            ctx_key = self.conditions.get("context_key")
            ctx_value = self.conditions.get("context_value")
            return context.get(ctx_key) == ctx_value
            
        if self.trigger == BehaviorTrigger.ON_EMOTION:
            emotions = self.conditions.get("emotions", [])
            return context.get("detected_emotion") in emotions
            
        if self.trigger == BehaviorTrigger.CONDITIONAL:
            cond_expr = self.conditions.get("expression", "")
            try:
                return eval(cond_expr, {"context": context})
            except Exception:
                return False
                
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger.value,
            "conditions": self.conditions,
            "actions": self.actions,
            "priority": self.priority,
            "examples": self.examples,
            "enabled": self.enabled
        }


@dataclass
class CommunicationStyle:
    """
    沟通风格定义
    
    Attributes:
        tone: 语气风格
        length: 回复长度
        format_preferences: 格式偏好
        vocabulary_level: 词汇水平
        emoji_usage: 表情使用
        humor_level: 幽默程度
        formality_level: 正式程度 (1-10)
    """
    tone: CommunicationTone = CommunicationTone.PROFESSIONAL
    length: ResponseLength = ResponseLength.MODERATE
    format_preferences: Dict[str, Any] = field(default_factory=dict)
    vocabulary_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    emoji_usage: float = 0.0  # 0.0-1.0
    humor_level: float = 0.0  # 0.0-1.0
    formality_level: int = 5  # 1-10
    
    def __post_init__(self):
        """验证数值范围"""
        if not 0.0 <= self.emoji_usage <= 1.0:
            raise ValueError(f"emoji_usage must be 0.0-1.0, got {self.emoji_usage}")
        if not 0.0 <= self.humor_level <= 1.0:
            raise ValueError(f"humor_level must be 0.0-1.0, got {self.humor_level}")
        if not 1 <= self.formality_level <= 10:
            raise ValueError(f"formality_level must be 1-10, got {self.formality_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tone": self.tone.value,
            "length": self.length.value,
            "format_preferences": self.format_preferences,
            "vocabulary_level": self.vocabulary_level,
            "emoji_usage": self.emoji_usage,
            "humor_level": self.humor_level,
            "formality_level": self.formality_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunicationStyle":
        """从字典创建"""
        return cls(
            tone=CommunicationTone(data.get("tone", "professional")),
            length=ResponseLength(data.get("length", "moderate")),
            format_preferences=data.get("format_preferences", {}),
            vocabulary_level=data.get("vocabulary_level", "intermediate"),
            emoji_usage=data.get("emoji_usage", 0.0),
            humor_level=data.get("humor_level", 0.0),
            formality_level=data.get("formality_level", 5)
        )


@dataclass
class PersonalityMetadata:
    """人格配置元数据"""
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    license: Optional[str] = None
    source: Optional[str] = None  # 文件路径或URL
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "description": self.description,
            "tags": self.tags,
            "license": self.license,
            "source": self.source
        }


@dataclass
class PersonalityConfig:
    """
    完整的人格配置
    
    这是人格系统的核心数据结构，包含所有人格相关配置。
    
    Attributes:
        name: 人格名称
        version: SemVer版本号
        traits: 大五人格特质列表
        values: 核心价值观列表
        behaviors: 行为模式列表
        constraints: 约束规则列表
        communication_style: 沟通风格
        domain_expertise: 专业领域
        metadata: 元数据
        compatibility: 兼容性配置
    """
    name: str
    version: str
    traits: List[PersonalityTrait] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    behaviors: List[BehaviorPattern] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    communication_style: CommunicationStyle = field(default_factory=CommunicationStyle)
    domain_expertise: List[str] = field(default_factory=list)
    metadata: PersonalityMetadata = field(default_factory=PersonalityMetadata)
    compatibility: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证版本格式"""
        if not re.match(r'^\d+\.\d+\.\d+$', self.version):
            raise ValueError(f"Invalid SemVer format: {self.version}")
    
    def get_fingerprint(self) -> str:
        """
        生成人格配置的指纹
        
        Returns:
            配置的SHA256哈希
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_trait_profile(self) -> Dict[str, int]:
        """获取特质配置"""
        return {trait.dimension.value: trait.intensity for trait in self.traits}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "traits": [t.to_dict() for t in self.traits],
            "values": self.values,
            "behaviors": [b.to_dict() for b in self.behaviors],
            "constraints": self.constraints,
            "communication_style": self.communication_style.to_dict(),
            "domain_expertise": self.domain_expertise,
            "metadata": self.metadata.to_dict(),
            "compatibility": self.compatibility
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityConfig":
        """从字典创建"""
        metadata = PersonalityMetadata()
        if "metadata" in data:
            m = data["metadata"]
            metadata = PersonalityMetadata(
                author=m.get("author"),
                created_at=datetime.fromisoformat(m["created_at"]) if m.get("created_at") else None,
                updated_at=datetime.fromisoformat(m["updated_at"]) if m.get("updated_at") else None,
                description=m.get("description"),
                tags=m.get("tags", []),
                license=m.get("license"),
                source=m.get("source")
            )
        
        communication = CommunicationStyle()
        if "communication_style" in data:
            communication = CommunicationStyle.from_dict(data["communication_style"])
        
        return cls(
            name=data["name"],
            version=data["version"],
            traits=[PersonalityTrait.from_dict(t) for t in data.get("traits", [])],
            values=data.get("values", []),
            behaviors=[BehaviorPattern(
                name=b["name"],
                description=b["description"],
                trigger=BehaviorTrigger(b.get("trigger", "always")),
                conditions=b.get("conditions", {}),
                actions=b.get("actions", []),
                priority=b.get("priority", 0),
                examples=b.get("examples", []),
                enabled=b.get("enabled", True)
            ) for b in data.get("behaviors", [])],
            constraints=data.get("constraints", []),
            communication_style=communication,
            domain_expertise=data.get("domain_expertise", []),
            metadata=metadata,
            compatibility=data.get("compatibility", {})
        )
    
    def to_markdown(self) -> str:
        """转换为SOUL.md格式"""
        lines = [
            "---",
            f"name: {self.name}",
            f"version: {self.version}",
            f"author: {self.metadata.author or 'unknown'}",
            f"created: {self.metadata.created_at.isoformat() if self.metadata.created_at else ''}",
            f"updated: {self.metadata.updated_at.isoformat() if self.metadata.updated_at else ''}",
            "---",
            "",
            f"# {self.name}",
            "",
        ]
        
        if self.metadata.description:
            lines.extend([self.metadata.description, ""])
        
        if self.traits:
            lines.extend(["## Traits", ""])
            for trait in self.traits:
                lines.extend([
                    f"- **{trait.dimension.value}** (intensity: {trait.intensity}/5)",
                    f"  - {trait.description}"
                ])
                if trait.examples:
                    lines.append("  - Examples:")
                    for ex in trait.examples:
                        lines.append(f"    - {ex}")
            lines.append("")
        
        if self.values:
            lines.extend(["## Values", ""])
            for value in self.values:
                lines.append(f"- {value}")
            lines.append("")
        
        if self.behaviors:
            lines.extend(["## Behaviors", ""])
            for behavior in self.behaviors:
                lines.append(f"### {behavior.name}")
                lines.append(f"{behavior.description}")
                lines.append(f"- Trigger: {behavior.trigger.value}")
                if behavior.actions:
                    lines.append("- Actions:")
                    for action in behavior.actions:
                        lines.append(f"  - {action}")
                lines.append("")
        
        if self.constraints:
            lines.extend(["## Constraints", ""])
            for constraint in self.constraints:
                lines.append(f"- {constraint}")
            lines.append("")
        
        if self.domain_expertise:
            lines.extend(["## Domain Expertise", ""])
            lines.append(", ".join(self.domain_expertise))
            lines.append("")
        
        lines.extend([
            "## Communication Style",
            f"- Tone: {self.communication_style.tone.value}",
            f"- Length: {self.communication_style.length.value}",
            f"- Formality: {self.communication_style.formality_level}/10",
            f"- Vocabulary: {self.communication_style.vocabulary_level}",
            ""
        ])
        
        return "\n".join(lines)


# =============================================================================
# 异常类
# =============================================================================

class PersonalityError(Exception):
    """人格相关错误基类"""
    pass


class ParseError(PersonalityError):
    """解析错误"""
    pass


class ValidationError(PersonalityError):
    """验证错误"""
    pass


class CompatibilityError(PersonalityError):
    """兼容性错误"""
    pass


class LoadingError(PersonalityError):
    """加载错误"""
    pass


class VersionError(PersonalityError):
    """版本错误"""
    pass


class BindingError(PersonalityError):
    """绑定错误"""
    pass


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    # 枚举
    "TraitDimension",
    "TraitIntensity", 
    "BehaviorTrigger",
    "CommunicationTone",
    "ResponseLength",
    "InjectionStrategy",
    # 数据类
    "PersonalityTrait",
    "BehaviorPattern",
    "CommunicationStyle",
    "PersonalityMetadata",
    "PersonalityConfig",
    # 异常
    "PersonalityError",
    "ParseError",
    "ValidationError",
    "CompatibilityError",
    "LoadingError",
    "VersionError",
    "BindingError",
]


# =============================================================================
# 延迟导入核心组件 - 避免循环依赖
# =============================================================================

# 在文件末尾进行导入，避免循环依赖
from .soul_parser import SoulParser
from .personality_engine import PersonalityEngine
from .validator import Validator, ValidationLevel
from .loader import Loader
from .versioning import VersionManager, CompatibilityResult
from .binding import AgentBinder, PersonalityContext, PromptTemplate, BindingScope


# 添加组件到__all__
__all__.extend([
    "SoulParser",
    "PersonalityEngine",
    "Validator",
    "ValidationLevel",
    "Loader",
    "VersionManager",
    "CompatibilityResult",
    "AgentBinder",
    "PersonalityContext",
    "PromptTemplate",
    "BindingScope",
])
