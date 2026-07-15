"""
Personality Validator - 人格配置验证器

该模块负责验证人格配置的正确性、完整性和安全性。

核心功能:
- 配置合法性检查
- 人格冲突检测
- 安全性审核
- 完整性验证

使用示例:
    validator = Validator()
    result = validator.validate(personality_config)
    
    if not result.is_valid:
        for error in result.errors:
            print(f"Error: {error}")
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from . import (
    PersonalityConfig, PersonalityTrait, BehaviorPattern,
    CommunicationStyle, TraitDimension, BehaviorTrigger,
    CommunicationTone, ResponseLength, ValidationError
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    MINIMAL = "minimal"        # 最小验证
    STANDARD = "standard"      # 标准验证
    STRICT = "strict"         # 严格验证
    COMPREHENSIVE = "comprehensive"  # 全面验证


class ValidationCategory(Enum):
    """验证类别"""
    STRUCTURE = "structure"        # 结构验证
    SEMANTIC = "semantic"          # 语义验证
    CONSISTENCY = "consistency"    # 一致性验证
    SAFETY = "safety"             # 安全性验证
    COMPATIBILITY = "compatibility"  # 兼容性验证


@dataclass
class ValidationIssue:
    """
    验证问题
    
    Attributes:
        category: 问题类别
        severity: 严重程度
        code: 问题代码
        message: 问题消息
        field: 相关字段
        suggestion: 修复建议
    """
    category: ValidationCategory
    severity: str  # error, warning, info
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        """字符串表示"""
        parts = [f"[{self.category.value}]"]
        parts.append(f"[{self.severity.upper()}]")
        if self.field:
            parts.append(f"({self.field})")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"  -> {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """
    验证结果
    
    Attributes:
        is_valid: 是否有效
        errors: 错误列表
        warnings: 警告列表
        info: 信息列表
        checked_items: 检查项数量
        validation_time: 验证耗时
    """
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    checked_items: int = 0
    validation_time_ms: float = 0.0
    
    def add_error(
        self, 
        category: ValidationCategory, 
        code: str, 
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """添加错误"""
        self.errors.append(ValidationIssue(
            category=category,
            severity="error",
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        ))
        self.is_valid = False
    
    def add_warning(
        self,
        category: ValidationCategory,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """添加警告"""
        self.warnings.append(ValidationIssue(
            category=category,
            severity="warning",
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        ))
    
    def add_info(
        self,
        category: ValidationCategory,
        code: str,
        message: str,
        field: Optional[str] = None
    ) -> None:
        """添加信息"""
        self.info.append(ValidationIssue(
            category=category,
            severity="info",
            code=code,
            message=message,
            field=field
        ))
    
    def merge(self, other: 'ValidationResult') -> None:
        """合并另一个验证结果"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.checked_items += other.checked_items
        self.validation_time_ms += other.validation_time_ms
        if other.errors:
            self.is_valid = False
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "checked_items": self.checked_items,
            "validation_time_ms": self.validation_time_ms
        }


class Validator:
    """
    人格配置验证器
    
    提供多层次的配置验证：
    - 结构验证：检查必要字段和数据类型
    - 语义验证：检查值的有效性和合理性
    - 一致性验证：检测配置冲突
    - 安全性验证：检查有害内容
    
    Attributes:
        level: 验证级别
        strict_mode: 严格模式
        enabled_rules: 启用的验证规则
    """
    
    # 必需字段
    REQUIRED_FIELDS = ['name', 'version']
    
    # 安全关键词（黑名单）
    SAFETY_BLACKLIST = [
        'harmful', 'illegal', 'violent', 'hate',
        'discriminat', 'malicious', 'deceptive',
        '攻击', '暴力', '非法', '有害', '恶意'
    ]
    
    # 特质维度有效值
    VALID_DIMENSIONS = {d.value for d in TraitDimension}
    
    # 沟通语气有效值
    VALID_TONES = {t.value for t in CommunicationTone}
    
    # 行为触发器有效值
    VALID_TRIGGERS = {t.value for t in BehaviorTrigger}
    
    # 回复长度有效值
    VALID_LENGTHS = {l.value for l in ResponseLength}
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        strict_mode: bool = True
    ):
        """
        初始化验证器
        
        Args:
            level: 验证级别
            strict_mode: 严格模式
        """
        self.level = level
        self.strict_mode = strict_mode
        self._custom_rules: List[callable] = []
    
    def validate(self, config: PersonalityConfig) -> ValidationResult:
        """
        验证人格配置
        
        Args:
            config: 待验证的配置
            
        Returns:
            ValidationResult对象
        """
        import time
        start_time = time.time()
        
        result = ValidationResult(is_valid=True)
        
        # 结构验证
        result.merge(self._validate_structure(config))
        
        # 语义验证
        result.merge(self._validate_semantics(config))
        
        # 一致性验证
        result.merge(self._validate_consistency(config))
        
        # 安全性验证
        result.merge(self._validate_safety(config))
        
        # 兼容性验证
        result.merge(self._validate_compatibility(config))
        
        # 应用自定义规则
        result.merge(self._validate_custom_rules(config))
        
        # 高级验证
        if self.level in (ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE):
            result.merge(self._validate_detailed(config))
        
        if self.level == ValidationLevel.COMPREHENSIVE:
            result.merge(self._validate_comprehensive(config))
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_structure(self, config: PersonalityConfig) -> ValidationResult:
        """验证结构"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 10
        
        # 检查必需字段
        for field_name in self.REQUIRED_FIELDS:
            if not hasattr(config, field_name) or not getattr(config, field_name):
                result.add_error(
                    ValidationCategory.STRUCTURE,
                    "MISSING_REQUIRED_FIELD",
                    f"Missing required field: {field_name}",
                    field=field_name,
                    suggestion=f"Add a valid {field_name} to the configuration"
                )
        
        # 检查版本格式
        if not re.match(r'^\d+\.\d+\.\d+$', config.version):
            result.add_error(
                ValidationCategory.STRUCTURE,
                "INVALID_VERSION_FORMAT",
                f"Invalid version format: {config.version}",
                field="version",
                suggestion="Use SemVer format (e.g., 1.0.0)"
            )
        
        # 检查traits列表
        if not isinstance(config.traits, list):
            result.add_error(
                ValidationCategory.STRUCTURE,
                "INVALID_TRAITS_TYPE",
                "Traits must be a list",
                field="traits"
            )
        elif len(config.traits) == 0:
            result.add_warning(
                ValidationCategory.STRUCTURE,
                "EMPTY_TRAITS",
                "No traits defined",
                field="traits",
                suggestion="Consider adding at least one personality trait"
            )
        
        # 检查communication_style
        if not isinstance(config.communication_style, CommunicationStyle):
            result.add_error(
                ValidationCategory.STRUCTURE,
                "INVALID_COMMUNICATION_STYLE",
                "communication_style must be CommunicationStyle object",
                field="communication_style"
            )
        
        return result
    
    def _validate_semantics(self, config: PersonalityConfig) -> ValidationResult:
        """验证语义"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 15
        
        # 验证每个特质
        for i, trait in enumerate(config.traits):
            # 维度有效性
            if trait.dimension.value not in self.VALID_DIMENSIONS:
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "INVALID_TRAIT_DIMENSION",
                    f"Invalid trait dimension: {trait.dimension.value}",
                    field=f"traits[{i}].dimension",
                    suggestion=f"Use one of: {', '.join(self.VALID_DIMENSIONS)}"
                )
            
            # 强度范围
            if not 1 <= trait.intensity <= 5:
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "INVALID_TRAIT_INTENSITY",
                    f"Trait intensity must be 1-5, got {trait.intensity}",
                    field=f"traits[{i}].intensity",
                    suggestion="Set intensity between 1 and 5"
                )
            
            # 描述长度
            if len(trait.description) < 10:
                result.add_warning(
                    ValidationCategory.SEMANTIC,
                    "SHORT_TRAIT_DESCRIPTION",
                    f"Trait description is very short: {trait.description}",
                    field=f"traits[{i}].description",
                    suggestion="Provide a more detailed description"
                )
        
        # 验证沟通风格
        comm = config.communication_style
        
        # 语气
        if comm.tone.value not in self.VALID_TONES:
            result.add_error(
                ValidationCategory.SEMANTIC,
                "INVALID_COMMUNICATION_TONE",
                f"Invalid communication tone: {comm.tone.value}",
                field="communication_style.tone"
            )
        
        # 回复长度
        if comm.length.value not in self.VALID_LENGTHS:
            result.add_error(
                ValidationCategory.SEMANTIC,
                "INVALID_RESPONSE_LENGTH",
                f"Invalid response length: {comm.length.value}",
                field="communication_style.length"
            )
        
        # 正式程度
        if not 1 <= comm.formality_level <= 10:
            result.add_error(
                ValidationCategory.SEMANTIC,
                "INVALID_FORMALITY_LEVEL",
                f"Formality level must be 1-10, got {comm.formality_level}",
                field="communication_style.formality_level"
            )
        
        # Emoji使用
        if not 0.0 <= comm.emoji_usage <= 1.0:
            result.add_error(
                ValidationCategory.SEMANTIC,
                "INVALID_EMOJI_USAGE",
                f"Emoji usage must be 0.0-1.0, got {comm.emoji_usage}",
                field="communication_style.emoji_usage"
            )
        
        # 验证行为模式
        for i, behavior in enumerate(config.behaviors):
            if behavior.trigger.value not in self.VALID_TRIGGERS:
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "INVALID_BEHAVIOR_TRIGGER",
                    f"Invalid behavior trigger: {behavior.trigger.value}",
                    field=f"behaviors[{i}].trigger"
                )
            
            if not behavior.name:
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "EMPTY_BEHAVIOR_NAME",
                    "Behavior name cannot be empty",
                    field=f"behaviors[{i}].name"
                )
        
        return result
    
    def _validate_consistency(self, config: PersonalityConfig) -> ValidationResult:
        """验证一致性"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 12
        
        # 检查重复的特质维度
        seen_dimensions: Set[TraitDimension] = set()
        for i, trait in enumerate(config.traits):
            if trait.dimension in seen_dimensions:
                result.add_warning(
                    ValidationCategory.CONSISTENCY,
                    "DUPLICATE_TRAIT_DIMENSION",
                    f"Duplicate trait dimension: {trait.dimension.value}",
                    field=f"traits[{i}].dimension",
                    suggestion="Each trait dimension should be unique"
                )
            seen_dimensions.add(trait.dimension)
        
        # 检查重复的行为名称
        seen_behaviors: Set[str] = set()
        for i, behavior in enumerate(config.behaviors):
            if behavior.name in seen_behaviors:
                result.add_warning(
                    ValidationCategory.CONSISTENCY,
                    "DUPLICATE_BEHAVIOR_NAME",
                    f"Duplicate behavior name: {behavior.name}",
                    field=f"behaviors[{i}].name",
                    suggestion="Each behavior should have a unique name"
                )
            seen_behaviors.add(behavior.name)
        
        # 检查冲突的特质组合
        conflicts = self._detect_trait_conflicts(config.traits)
        for conflict in conflicts:
            result.add_warning(
                ValidationCategory.CONSISTENCY,
                "TRAIT_CONFLICT",
                f"Potential trait conflict: {conflict['message']}",
                field=conflict.get('field'),
                suggestion=conflict.get('suggestion')
            )
        
        # 验证沟通风格与特质的匹配
        self._check_communication_trait_match(config, result)
        
        return result
    
    def _detect_trait_conflicts(
        self, 
        traits: List[PersonalityTrait]
    ) -> List[Dict[str, str]]:
        """检测特质冲突"""
        conflicts = []
        trait_map = {t.dimension: t.intensity for t in traits}
        
        # 检测极端特质组合
        if TraitDimension.EXTRAVERSION in trait_map:
            extraversion = trait_map[TraitDimension.EXTRAVERSION]
            if TraitDimension.NEUROTICISM in trait_map:
                neuroticism = trait_map[TraitDimension.NEUROTICISM]
                if extraversion >= 4 and neuroticism >= 4:
                    conflicts.append({
                        'message': 'High extraversion with high neuroticism may cause inconsistent behavior',
                        'field': 'traits',
                        'suggestion': 'Consider moderating one of these traits'
                    })
        
        # 检测矛盾的专业程度
        if TraitDimension.OPENNESS in trait_map:
            openness = trait_map[TraitDimension.OPENNESS]
            if TraitDimension.CONSCIENTIOUSNESS in trait_map:
                conscientiousness = trait_map[TraitDimension.CONSCIENTIOUSNESS]
                if openness <= 2 and conscientiousness <= 2:
                    conflicts.append({
                        'message': 'Low openness and low conscientiousness may limit adaptability',
                        'field': 'traits',
                        'suggestion': 'Consider increasing one of these traits'
                    })
        
        return conflicts
    
    def _check_communication_trait_match(
        self,
        config: PersonalityConfig,
        result: ValidationResult
    ) -> None:
        """检查沟通风格与特质的匹配"""
        trait_map = {t.dimension: t.intensity for t in config.traits}
        comm = config.communication_style
        
        # 外向性与正式程度的关系
        if TraitDimension.EXTRAVERSION in trait_map:
            extraversion = trait_map[TraitDimension.EXTRAVERSION]
            
            # 高外向性 + 低正式程度 = 自然
            if extraversion >= 4 and comm.formality_level <= 3:
                result.add_info(
                    ValidationCategory.CONSISTENCY,
                    "TRAIT_STYLE_MATCH",
                    "High extraversion matches casual communication style"
                )
            
            # 高外向性 + 高正式程度 = 可能不一致
            if extraversion >= 4 and comm.formality_level >= 8:
                result.add_warning(
                    ValidationCategory.CONSISTENCY,
                    "TRAIT_STYLE_MISMATCH",
                    "High extraversion may conflict with formal communication",
                    field="communication_style.formality_level"
                )
        
        # 宜人性与语气的关系
        if TraitDimension.AGREEABLENESS in trait_map:
            agreeableness = trait_map[TraitDimension.AGREEABLENESS]
            
            if agreeableness >= 4:
                if comm.tone == CommunicationTone.AUTHORITATIVE:
                    result.add_warning(
                        ValidationCategory.CONSISTENCY,
                        "TRAIT_STYLE_CONFLICT",
                        "High agreeableness may conflict with authoritative tone",
                        field="communication_style.tone"
                    )
    
    def _validate_safety(self, config: PersonalityConfig) -> ValidationResult:
        """验证安全性"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 8
        
        # 检查名称
        if config.name:
            for keyword in self.SAFETY_BLACKLIST:
                if keyword.lower() in config.name.lower():
                    result.add_error(
                        ValidationCategory.SAFETY,
                        "UNSAFE_NAME",
                        f"Name contains potentially unsafe keyword: {keyword}",
                        field="name"
                    )
        
        # 检查价值观
        for i, value in enumerate(config.values):
            for keyword in self.SAFETY_BLACKLIST:
                if keyword.lower() in value.lower():
                    result.add_error(
                        ValidationCategory.SAFETY,
                        "UNSAFE_VALUE",
                        f"Value contains potentially unsafe keyword: {keyword}",
                        field=f"values[{i}]"
                    )
        
        # 检查约束
        for i, constraint in enumerate(config.constraints):
            # 检查潜在的指令注入
            dangerous_patterns = [
                r'\bignore\s+(all\s+)?previous\s+(instructions?|prompts?)\b',
                r'\bdisregard\s+(all\s+)?(your\s+)?instructions?\b',
                r'\boverride\s+system\s+(prompt|role)\b',
                r'\byou\s+are\s+now\s+(a\s+)?different\s+(AI|model|assistant)\b',
                r'\bforget\s+(all\s+)?previous\s+(instructions?|prompts?)\b',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, constraint, re.IGNORECASE):
                    result.add_error(
                        ValidationCategory.SAFETY,
                        "POTENTIAL_INJECTION",
                        f"Constraint may contain instruction injection pattern",
                        field=f"constraints[{i}]",
                        suggestion="Review and sanitize constraint content"
                    )
        
        # 检查行为描述
        for i, behavior in enumerate(config.behaviors):
            for keyword in self.SAFETY_BLACKLIST:
                if keyword.lower() in behavior.description.lower():
                    result.add_error(
                        ValidationCategory.SAFETY,
                        "UNSAFE_BEHAVIOR",
                        f"Behavior description contains unsafe keyword: {keyword}",
                        field=f"behaviors[{i}].description"
                    )
        
        return result
    
    def _validate_compatibility(self, config: PersonalityConfig) -> ValidationResult:
        """验证兼容性"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 5
        
        # 检查版本与配置的兼容性
        version_parts = config.version.split('.')
        major_version = int(version_parts[0])
        
        # 警告过旧的版本
        if major_version == 0:
            result.add_warning(
                ValidationCategory.COMPATIBILITY,
                "PRE_V1_VERSION",
                "This is a pre-1.0 version and may have breaking changes",
                field="version"
            )
        
        # 检查必需的专业领域（如果指定了特定域）
        domain_keywords = ['analyst', 'coder', 'creative']
        has_domain = any(
            any(kw in d.lower() for kw in domain_keywords)
            for d in config.domain_expertise
        )
        
        if not has_domain and len(config.domain_expertise) == 0:
            result.add_info(
                ValidationCategory.COMPATIBILITY,
                "NO_DOMAIN_SPECIFIED",
                "No domain expertise specified, may affect performance"
            )
        
        return result
    
    def _validate_custom_rules(self, config: PersonalityConfig) -> ValidationResult:
        """应用自定义验证规则"""
        result = ValidationResult(is_valid=True)
        
        for rule in self._custom_rules:
            try:
                rule_result = rule(config)
                if isinstance(rule_result, ValidationResult):
                    result.merge(rule_result)
                elif isinstance(rule_result, bool):
                    if not rule_result:
                        result.add_error(
                            ValidationCategory.STRUCTURE,
                            "CUSTOM_RULE_FAILED",
                            f"Custom validation rule failed: {rule.__name__}"
                        )
            except Exception as e:
                result.add_error(
                    ValidationCategory.STRUCTURE,
                    "CUSTOM_RULE_ERROR",
                    f"Error in custom rule {rule.__name__}: {str(e)}"
                )
        
        return result
    
    def _validate_detailed(self, config: PersonalityConfig) -> ValidationResult:
        """详细验证"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 10
        
        # 检查特质描述的一致性
        for i, trait in enumerate(config.traits):
            if trait.examples:
                # 检查示例数量
                if len(trait.examples) > 5:
                    result.add_warning(
                        ValidationCategory.STRUCTURE,
                        "MANY_EXAMPLES",
                        f"Trait has many examples ({len(trait.examples)})",
                        field=f"traits[{i}].examples"
                    )
        
        # 检查行为的完整性
        for i, behavior in enumerate(config.behaviors):
            if not behavior.actions:
                result.add_warning(
                    ValidationCategory.STRUCTURE,
                    "EMPTY_BEHAVIOR_ACTIONS",
                    f"Behavior has no defined actions",
                    field=f"behaviors[{i}].actions",
                    suggestion="Define specific actions for this behavior"
                )
            
            if not behavior.examples:
                result.add_info(
                    ValidationCategory.STRUCTURE,
                    "NO_BEHAVIOR_EXAMPLES",
                    f"Behavior has no examples",
                    field=f"behaviors[{i}].examples"
                )
        
        return result
    
    def _validate_comprehensive(self, config: PersonalityConfig) -> ValidationResult:
        """全面验证"""
        result = ValidationResult(is_valid=True)
        result.checked_items += 15
        
        # 验证元数据
        metadata = config.metadata
        
        if not metadata.author:
            result.add_info(
                ValidationCategory.STRUCTURE,
                "NO_AUTHOR",
                "No author specified in metadata"
            )
        
        if not metadata.created_at:
            result.add_info(
                ValidationCategory.STRUCTURE,
                "NO_CREATED_DATE",
                "No creation date specified"
            )
        
        # 验证约束的语法
        for i, constraint in enumerate(config.constraints):
            # 检查是否有未闭合的括号
            if constraint.count('(') != constraint.count(')'):
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "UNBALANCED_PARENTHESES",
                    f"Constraint has unbalanced parentheses",
                    field=f"constraints[{i}]"
                )
            
            # 检查是否有未闭合的引号
            if constraint.count('"') % 2 != 0:
                result.add_error(
                    ValidationCategory.SEMANTIC,
                    "UNBALANCED_QUOTES",
                    f"Constraint has unbalanced quotes",
                    field=f"constraints[{i}]"
                )
        
        # 验证值的数量限制
        if len(config.values) > 20:
            result.add_warning(
                ValidationCategory.STRUCTURE,
                "MANY_VALUES",
                f"Configuration has many values ({len(config.values)})",
                field="values",
                suggestion="Consider consolidating values"
            )
        
        if len(config.constraints) > 30:
            result.add_warning(
                ValidationCategory.STRUCTURE,
                "MANY_CONSTRAINTS",
                f"Configuration has many constraints ({len(config.constraints)})",
                field="constraints",
                suggestion="Consider consolidating constraints"
            )
        
        return result
    
    def add_custom_rule(self, rule: callable) -> None:
        """
        添加自定义验证规则
        
        Args:
            rule: 验证函数，接受PersonalityConfig，返回ValidationResult或bool
        """
        self._custom_rules.append(rule)
    
    def check_conflicts(
        self,
        config1: PersonalityConfig,
        config2: PersonalityConfig
    ) -> List[Dict[str, Any]]:
        """
        检查两个人格配置之间的冲突
        
        Args:
            config1: 第一个配置
            config2: 第二个配置
            
        Returns:
            冲突列表
        """
        conflicts = []
        
        # 检查特质冲突
        traits1 = {t.dimension: t.intensity for t in config1.traits}
        traits2 = {t.dimension: t.intensity for t in config2.traits}
        
        for dimension in set(traits1.keys()) & set(traits2.keys()):
            diff = abs(traits1[dimension] - traits2[dimension])
            if diff >= 3:
                conflicts.append({
                    'type': 'trait_conflict',
                    'dimension': dimension.value,
                    'config1_value': traits1[dimension],
                    'config2_value': traits2[dimension],
                    'severity': 'high' if diff >= 4 else 'medium'
                })
        
        # 检查沟通风格冲突
        comm1 = config1.communication_style
        comm2 = config2.communication_style
        
        if comm1.tone != comm2.tone:
            conflicts.append({
                'type': 'communication_conflict',
                'field': 'tone',
                'config1_value': comm1.tone.value,
                'config2_value': comm2.tone.value,
                'severity': 'medium'
            })
        
        if abs(comm1.formality_level - comm2.formality_level) >= 5:
            conflicts.append({
                'type': 'communication_conflict',
                'field': 'formality_level',
                'config1_value': comm1.formality_level,
                'config2_value': comm2.formality_level,
                'severity': 'high'
            })
        
        return conflicts


def create_validator(
    level: ValidationLevel = ValidationLevel.STANDARD,
    strict: bool = True
) -> Validator:
    """
    工厂函数：创建验证器
    
    Args:
        level: 验证级别
        strict: 严格模式
        
    Returns:
        Validator实例
    """
    return Validator(level=level, strict_mode=strict)
