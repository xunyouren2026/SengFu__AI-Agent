"""
SOUL.md语法解析器

该模块负责解析SOUL.md格式的人格配置文件，将其转换为结构化的PersonalityConfig对象。

SOUL.md格式支持:
- YAML frontmatter元数据
- Markdown结构化内容
- 人格特质定义 (Traits)
- 价值观列表 (Values)
- 行为模式定义 (Behaviors)
- 约束规则 (Constraints)
- 沟通风格配置 (Communication Style)

示例:
    parser = SoulParser()
    config = parser.parse(content)
    config = parser.parse_file("path/to/personality.md")
"""

import re
import yaml
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from . import (
    PersonalityConfig, PersonalityTrait, BehaviorPattern, 
    CommunicationStyle, PersonalityMetadata, TraitDimension,
    BehaviorTrigger, CommunicationTone, ResponseLength,
    ParseError
)

logger = logging.getLogger(__name__)


class SectionType(Enum):
    """文档章节类型"""
    FRONTMATTER = "frontmatter"
    TRAITS = "traits"
    VALUES = "values"
    BEHAVIORS = "behaviors"
    CONSTRAINTS = "constraints"
    COMMUNICATION = "communication"
    DOMAIN = "domain"
    UNKNOWN = "unknown"


@dataclass
class ParsedSection:
    """解析的章节"""
    section_type: SectionType
    raw_content: str
    parsed_data: Any = None
    line_number: int = 0


@dataclass
class ParseResult:
    """解析结果"""
    config: PersonalityConfig
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SoulParser:
    """
    SOUL.md格式解析器
    
    提供完整的SOUL.md文件解析能力，支持：
    - YAML frontmatter解析
    - Markdown结构化内容解析
    - 人格特质提取
    - 行为模式识别
    - 约束规则解析
    - 沟通风格配置
    
    Attributes:
        strict_mode: 严格模式，严格检查语法错误
        allow_incomplete: 允许不完整的配置
    """
    
    # 正则表达式模式
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL | re.MULTILINE
    )
    
    HEADER_PATTERN = re.compile(
        r'^(#{1,6})\s+(.+)$',
        re.MULTILINE
    )
    
    TRAIT_PATTERN = re.compile(
        r'\*\*(\w+)\*\*\s*\(intensity:\s*(\d+)/5\)',
        re.IGNORECASE
    )
    
    LIST_ITEM_PATTERN = re.compile(
        r'^[\s]*[-*+]\s+(.+)$',
        re.MULTILINE
    )
    
    NUMBERED_LIST_PATTERN = re.compile(
        r'^[\s]*\d+\.\s+(.+)$',
        re.MULTILINE
    )
    
    KEY_VALUE_PATTERN = re.compile(
        r'^[\s]*([^:]+):\s*(.+)$',
        re.MULTILINE
    )
    
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )
    
    INLINE_CODE_PATTERN = re.compile(
        r'`([^`]+)`'
    )
    
    EMPHASIS_PATTERN = re.compile(
        r'\*\*([^*]+)\*\*'
    )
    
    BEHAVIOR_SECTION_PATTERN = re.compile(
        r'^###\s+(.+)$',
        re.MULTILINE
    )
    
    def __init__(
        self, 
        strict_mode: bool = True,
        allow_incomplete: bool = False
    ):
        """
        初始化解析器
        
        Args:
            strict_mode: 是否启用严格模式
            allow_incomplete: 是否允许不完整配置
        """
        self.strict_mode = strict_mode
        self.allow_incomplete = allow_incomplete
        self._section_cache: Dict[str, ParsedSection] = {}
        
    def parse(self, content: str) -> PersonalityConfig:
        """
        解析SOUL.md内容
        
        Args:
            content: SOUL.md格式的原始内容
            
        Returns:
            PersonalityConfig对象
            
        Raises:
            ParseError: 解析错误
        """
        if not content or not content.strip():
            raise ParseError("Empty content provided")
        
        # 清理内容
        content = self._normalize_content(content)
        
        # 提取frontmatter
        frontmatter, body = self._extract_frontmatter(content)
        
        # 解析frontmatter
        metadata = self._parse_frontmatter(frontmatter)
        
        # 解析主体内容
        sections = self._parse_sections(body)
        
        # 构建配置对象
        config = self._build_config(metadata, sections)
        
        # 验证配置
        self._validate_parsed_config(config)
        
        return config
    
    def parse_file(self, file_path: str) -> PersonalityConfig:
        """
        从文件解析SOUL.md
        
        Args:
            file_path: 文件路径
            
        Returns:
            PersonalityConfig对象
            
        Raises:
            ParseError: 解析错误
            FileNotFoundError: 文件不存在
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            config = self.parse(content)
            if config.metadata.source is None:
                config.metadata.source = file_path
            
            return config
        except FileNotFoundError:
            raise ParseError(f"File not found: {file_path}")
        except Exception as e:
            raise ParseError(f"Error reading file: {e}")
    
    def parse_with_result(self, content: str) -> ParseResult:
        """
        解析并返回完整结果（包括警告和错误）
        
        Args:
            content: SOUL.md格式的原始内容
            
        Returns:
            ParseResult对象
        """
        warnings = []
        errors = []
        
        try:
            config = self.parse(content)
        except ParseError as e:
            errors.append(str(e))
            # 返回空配置
            config = PersonalityConfig(
                name="INVALID",
                version="0.0.0"
            )
        
        return ParseResult(
            config=config,
            warnings=warnings,
            errors=errors
        )
    
    def _normalize_content(self, content: str) -> str:
        """
        规范化内容
        
        Args:
            content: 原始内容
            
        Returns:
            规范化后的内容
        """
        # 移除BOM
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # 统一行尾符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 移除尾部空白行
        content = content.rstrip()
        
        return content
    
    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        提取YAML frontmatter
        
        Args:
            content: 规范化后的内容
            
        Returns:
            (frontmatter字典, 剩余内容)
        """
        match = self.FRONTMATTER_PATTERN.match(content)
        
        if match:
            yaml_content = match.group(1)
            body = content[match.end():]
            
            try:
                data = yaml.safe_load(yaml_content) or {}
            except yaml.YAMLError as e:
                if self.strict_mode:
                    raise ParseError(f"Invalid YAML in frontmatter: {e}")
                else:
                    data = {}
            
            return data, body
        
        # 没有frontmatter
        return {}, content
    
    def _parse_frontmatter(self, frontmatter: Dict[str, Any]) -> PersonalityMetadata:
        """
        解析frontmatter为元数据
        
        Args:
            frontmatter: frontmatter字典
            
        Returns:
            PersonalityMetadata对象
        """
        metadata = PersonalityMetadata()
        
        # 映射字段
        field_mapping = {
            'author': 'author',
            'created': 'created_at',
            'created_at': 'created_at',
            'updated': 'updated_at',
            'updated_at': 'updated_at',
            'description': 'description',
            'tags': 'tags',
            'license': 'license',
            'name': None,  # 特殊处理
            'version': None,  # 特殊处理
        }
        
        for key, value in frontmatter.items():
            if key in field_mapping and field_mapping[key]:
                setattr(metadata, field_mapping[key], value)
            elif key == 'tags' and isinstance(value, list):
                metadata.tags = value
            elif key == 'tags' and isinstance(value, str):
                metadata.tags = [t.strip() for t in value.split(',')]
        
        # 解析日期
        for date_field in ['created_at', 'updated_at']:
            date_value = getattr(metadata, date_field)
            if isinstance(date_value, str):
                try:
                    setattr(metadata, date_field, datetime.fromisoformat(date_value))
                except (ValueError, TypeError):
                    pass  # 忽略无效日期
        
        metadata.source = frontmatter.get('source', None)
        
        return metadata
    
    def _parse_sections(self, body: str) -> List[ParsedSection]:
        """
        解析文档主体章节
        
        Args:
            body: 去除frontmatter后的内容
            
        Returns:
            解析后的章节列表
        """
        sections = []
        lines = body.split('\n')
        
        current_section = None
        current_content = []
        current_line = 0
        
        for i, line in enumerate(lines):
            # 检查是否是章节标题
            header_match = self.HEADER_PATTERN.match(line)
            
            if header_match:
                # 保存之前的章节
                if current_section and current_content:
                    content_str = '\n'.join(current_content).strip()
                    if content_str:
                        sections.append(ParsedSection(
                            section_type=current_section,
                            raw_content=content_str,
                            parsed_data=self._parse_section_content(current_section, content_str),
                            line_number=current_line
                        ))
                
                # 开始新章节
                level = len(header_match.group(1))
                title = header_match.group(2).strip().lower()
                current_section = self._detect_section_type(title)
                current_content = []
                current_line = i + 1
            else:
                if current_section is not None:
                    current_content.append(line)
        
        # 保存最后一个章节
        if current_section and current_content:
            content_str = '\n'.join(current_content).strip()
            if content_str:
                sections.append(ParsedSection(
                    section_type=current_section,
                    raw_content=content_str,
                    parsed_data=self._parse_section_content(current_section, content_str),
                    line_number=current_line
                ))
        
        return sections
    
    def _detect_section_type(self, title: str) -> SectionType:
        """
        根据标题检测章节类型
        
        Args:
            title: 章节标题（小写）
            
        Returns:
            SectionType枚举值
        """
        title_lower = title.lower().strip()
        
        # 映射章节标题
        section_mapping = {
            'traits': SectionType.TRAITS,
            'personality traits': SectionType.TRAITS,
            'personality': SectionType.TRAITS,
            'character traits': SectionType.TRAITS,
            'values': SectionType.VALUES,
            'core values': SectionType.VALUES,
            'principles': SectionType.VALUES,
            'behaviors': SectionType.BEHAVIORS,
            'behavior patterns': SectionType.BEHAVIORS,
            'actions': SectionType.BEHAVIORS,
            'constraints': SectionType.CONSTRAINTS,
            'rules': SectionType.CONSTRAINTS,
            'limitations': SectionType.CONSTRAINTS,
            'communication': SectionType.COMMUNICATION,
            'communication style': SectionType.COMMUNICATION,
            'style': SectionType.COMMUNICATION,
            'domain': SectionType.DOMAIN,
            'expertise': SectionType.DOMAIN,
            'skills': SectionType.DOMAIN,
            'knowledge': SectionType.DOMAIN,
        }
        
        for key, section_type in section_mapping.items():
            if key in title_lower:
                return section_type
        
        return SectionType.UNKNOWN
    
    def _parse_section_content(
        self, 
        section_type: SectionType, 
        content: str
    ) -> Any:
        """
        根据章节类型解析内容
        
        Args:
            section_type: 章节类型
            content: 原始内容
            
        Returns:
            解析后的数据
        """
        parsers = {
            SectionType.TRAITS: self._parse_traits_section,
            SectionType.VALUES: self._parse_values_section,
            SectionType.BEHAVIORS: self._parse_behaviors_section,
            SectionType.CONSTRAINTS: self._parse_constraints_section,
            SectionType.COMMUNICATION: self._parse_communication_section,
            SectionType.DOMAIN: self._parse_domain_section,
        }
        
        parser = parsers.get(section_type)
        if parser:
            return parser(content)
        
        return content
    
    def _parse_traits_section(self, content: str) -> List[PersonalityTrait]:
        """
        解析人格特质章节
        
        Args:
            content: 章节内容
            
        Returns:
            PersonalityTrait列表
        """
        traits = []
        lines = content.split('\n')
        
        current_trait = None
        current_description = []
        current_examples = []
        current_manifestations = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 检查特质标题
            trait_match = re.match(r'\*\*(\w+)\*\*\s*\(intensity:\s*(\d+)/5\)', line)
            
            if trait_match:
                # 保存之前的特质
                if current_trait:
                    try:
                        trait_obj = self._create_trait(
                            current_trait,
                            '\n'.join(current_description).strip(),
                            current_examples,
                            current_manifestations
                        )
                        traits.append(trait_obj)
                    except Exception as e:
                        logger.warning(f"Failed to parse trait: {e}")
                
                # 开始新特质
                dimension_name = trait_match.group(1)
                intensity = int(trait_match.group(2))
                
                try:
                    current_trait = TraitDimension(dimension_name.lower())
                except ValueError:
                    # 尝试模糊匹配
                    current_trait = self._fuzzy_match_dimension(dimension_name)
                
                current_description = []
                current_examples = []
                current_manifestations = []
                intensity = intensity  # 用于下一轮
                
                # 获取描述（可能在同一行或下一行）
                desc_part = line.split(')', 1)
                if len(desc_part) > 1 and desc_part[1].strip():
                    current_description.append(desc_part[1].strip())
                
            elif line.startswith('- '):
                # 检查是否是示例
                if 'example' in line.lower():
                    current_examples.append(line.lstrip('- *').strip())
                elif current_description:
                    current_description.append(line.lstrip('- *'))
                else:
                    current_description.append(line.lstrip('- *'))
            elif line.startswith('  - '):
                # 子列表项
                clean_line = line.lstrip('  -').strip()
                if 'example' in '\n'.join(lines[max(0, i-3):i+1]).lower():
                    current_examples.append(clean_line)
                else:
                    current_manifestations.append(clean_line)
            elif current_trait and line:
                if current_description:
                    current_description.append(line)
            
            i += 1
        
        # 保存最后一个特质
        if current_trait:
            try:
                trait_obj = self._create_trait(
                    current_trait,
                    '\n'.join(current_description).strip(),
                    current_examples,
                    current_manifestations
                )
                traits.append(trait_obj)
            except Exception as e:
                logger.warning(f"Failed to parse final trait: {e}")
        
        # 如果没有解析到特质，尝试备用解析
        if not traits:
            traits = self._fallback_parse_traits(content)
        
        return traits
    
    def _create_trait(
        self,
        dimension: TraitDimension,
        description: str,
        examples: List[str],
        manifestations: List[str]
    ) -> PersonalityTrait:
        """创建PersonalityTrait对象"""
        # 确定强度
        intensity = 3  # 默认中等
        
        # 从描述中提取强度
        intensity_match = re.search(r'intensity[:\s]+(\d+)', description.lower())
        if intensity_match:
            intensity = int(intensity_match.group(1))
        
        # 清理描述中的强度标记
        description = re.sub(r'\(?intensity[:\s]+\d+\)?', '', description).strip()
        
        return PersonalityTrait(
            dimension=dimension,
            intensity=intensity,
            description=description,
            examples=examples,
            manifestations=manifestations
        )
    
    def _fuzzy_match_dimension(self, name: str) -> TraitDimension:
        """
        模糊匹配特质维度
        
        Args:
            name: 特质名称
            
        Returns:
            匹配的TraitDimension
        """
        name_lower = name.lower()
        
        # 映射关系
        mapping = {
            'open': TraitDimension.OPENNESS,
            'openness': TraitDimension.OPENNESS,
            'creativity': TraitDimension.OPENNESS,
            'imagination': TraitDimension.OPENNESS,
            'conscientious': TraitDimension.CONSCIENTIOUSNESS,
            'conscientiousness': TraitDimension.CONSCIENTIOUSNESS,
            'dependability': TraitDimension.CONSCIENTIOUSNESS,
            'extravert': TraitDimension.EXTRAVERSION,
            'extraversion': TraitDimension.EXTRAVERSION,
            'outgoing': TraitDimension.EXTRAVERSION,
            'agreeable': TraitDimension.AGREEABLENESS,
            'agreeableness': TraitDimension.AGREEABLENESS,
            'cooperation': TraitDimension.AGREEABLENESS,
            'neurotic': TraitDimension.NEUROTICISM,
            'neuroticism': TraitDimension.NEUROTICISM,
            'emotional': TraitDimension.NEUROTICISM,
        }
        
        for key, dimension in mapping.items():
            if key in name_lower:
                return dimension
        
        # 默认返回开放性
        return TraitDimension.OPENNESS
    
    def _fallback_parse_traits(self, content: str) -> List[PersonalityTrait]:
        """
        备用特质解析（基于关键词匹配）
        
        Args:
            content: 内容
            
        Returns:
            PersonalityTrait列表
        """
        traits = []
        
        # 关键词映射
        keywords = {
            'open': TraitDimension.OPENNESS,
            'creative': TraitDimension.OPENNESS,
            'conscientious': TraitDimension.CONSCIENTIOUSNESS,
            'organized': TraitDimension.CONSCIENTIOUSNESS,
            'extravert': TraitDimension.EXTRAVERSION,
            'outgoing': TraitDimension.EXTRAVERSION,
            'agreeable': TraitDimension.AGREEABLENESS,
            'cooperative': TraitDimension.AGREEABLENESS,
            'neurotic': TraitDimension.NEUROTICISM,
            'emotional': TraitDimension.NEUROTICISM,
        }
        
        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword, dimension in keywords.items():
                if keyword in line_lower:
                    # 检查是否已存在
                    if not any(t.dimension == dimension for t in traits):
                        traits.append(PersonalityTrait(
                            dimension=dimension,
                            intensity=3,
                            description=line.strip(),
                            examples=[],
                            manifestations=[]
                        ))
                    break
        
        return traits
    
    def _parse_values_section(self, content: str) -> List[str]:
        """
        解析价值观章节
        
        Args:
            content: 章节内容
            
        Returns:
            价值观列表
        """
        values = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # 列表项
            if line.startswith('-') or line.startswith('*'):
                value = line.lstrip('-*').strip()
                if value:
                    values.append(value)
            # 编号列表
            elif re.match(r'^\d+\.', line):
                value = re.sub(r'^\d+\.\s*', '', line).strip()
                if value:
                    values.append(value)
            # 普通文本行
            elif line and not line.startswith('#'):
                values.append(line)
        
        return values
    
    def _parse_behaviors_section(self, content: str) -> List[BehaviorPattern]:
        """
        解析行为模式章节
        
        Args:
            content: 章节内容
            
        Returns:
            BehaviorPattern列表
        """
        behaviors = []
        
        # 分割行为块
        blocks = re.split(r'^###\s+', content, flags=re.MULTILINE)
        
        for block in blocks:
            if not block.strip():
                continue
            
            lines = block.split('\n')
            if not lines:
                continue
            
            # 提取行为名称
            name = lines[0].strip()
            if not name:
                continue
            
            # 解析行为内容
            behavior_content = '\n'.join(lines[1:])
            
            # 提取触发器
            trigger = BehaviorTrigger.ALWAYS
            trigger_match = re.search(
                r'trigger[:\s]+(\w+)',
                behavior_content.lower()
            )
            if trigger_match:
                trigger_name = trigger_match.group(1)
                try:
                    trigger = BehaviorTrigger(trigger_name.lower().replace(' ', '_'))
                except ValueError:
                    pass
            
            # 提取描述
            description = behavior_content.split('\n')[0].strip()
            if description.startswith('-'):
                description = name
            
            # 提取动作
            actions = []
            for line in behavior_content.split('\n'):
                line = line.strip()
                if line.startswith('-') and 'action' not in line.lower():
                    action = line.lstrip('-').strip()
                    if action:
                        actions.append(action)
            
            # 提取优先级
            priority = 0
            priority_match = re.search(r'priority[:\s]+(\d+)', behavior_content.lower())
            if priority_match:
                priority = int(priority_match.group(1))
            
            behaviors.append(BehaviorPattern(
                name=name,
                description=description,
                trigger=trigger,
                actions=actions,
                priority=priority,
                enabled=True
            ))
        
        # 备用解析
        if not behaviors:
            behaviors = self._fallback_parse_behaviors(content)
        
        return behaviors
    
    def _fallback_parse_behaviors(self, content: str) -> List[BehaviorPattern]:
        """备用行为解析"""
        behaviors = []
        
        lines = content.split('\n')
        current_behavior = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('-'):
                action = line.lstrip('-').strip()
                if current_behavior:
                    current_behavior.actions.append(action)
            elif line and not line.startswith('#'):
                current_behavior = BehaviorPattern(
                    name=line,
                    description=line,
                    trigger=BehaviorTrigger.ALWAYS,
                    actions=[],
                    enabled=True
                )
                behaviors.append(current_behavior)
        
        return behaviors
    
    def _parse_constraints_section(self, content: str) -> List[str]:
        """
        解析约束规则章节
        
        Args:
            content: 章节内容
            
        Returns:
            约束规则列表
        """
        constraints = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # 列表项
            if line.startswith('-') or line.startswith('*'):
                constraint = line.lstrip('-*').strip()
                if constraint and not constraint.lower().startswith('constraint'):
                    constraints.append(constraint)
            elif re.match(r'^\d+\.', line):
                constraint = re.sub(r'^\d+\.\s*', '', line).strip()
                if constraint:
                    constraints.append(constraint)
            elif line and not line.startswith('#'):
                # 尝试提取约束
                if len(line) > 10:  # 避免太短的行
                    constraints.append(line)
        
        return constraints
    
    def _parse_communication_section(self, content: str) -> CommunicationStyle:
        """
        解析沟通风格章节
        
        Args:
            content: 章节内容
            
        Returns:
            CommunicationStyle对象
        """
        tone = CommunicationTone.PROFESSIONAL
        length = ResponseLength.MODERATE
        formality = 5
        vocabulary = "intermediate"
        emoji_usage = 0.0
        humor_level = 0.0
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            # 解析键值对
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('-', '')
                value = value.strip()
                
                if 'tone' in key:
                    try:
                        tone = CommunicationTone(value)
                    except ValueError:
                        pass
                elif 'length' in key:
                    try:
                        length = ResponseLength(value)
                    except ValueError:
                        pass
                elif 'formal' in key:
                    match = re.search(r'(\d+)', value)
                    if match:
                        formality = int(match.group(1))
                elif 'vocab' in key:
                    vocabulary = value
                elif 'emoji' in key:
                    match = re.search(r'(\d+\.?\d*)', value)
                    if match:
                        emoji_usage = float(match.group(1))
                elif 'humor' in key:
                    match = re.search(r'(\d+\.?\d*)', value)
                    if match:
                        humor_level = float(match.group(1))
        
        return CommunicationStyle(
            tone=tone,
            length=length,
            vocabulary_level=vocabulary,
            emoji_usage=emoji_usage,
            humor_level=humor_level,
            formality_level=formality
        )
    
    def _parse_domain_section(self, content: str) -> List[str]:
        """
        解析专业领域章节
        
        Args:
            content: 章节内容
            
        Returns:
            专业领域列表
        """
        domains = []
        
        # 逗号分隔或列表格式
        if ',' in content:
            domains = [d.strip() for d in content.split(',')]
        else:
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    domain = line.lstrip('-*').strip()
                    if domain:
                        domains.append(domain)
                elif line and not line.startswith('#'):
                    domains.append(line)
        
        return domains
    
    def _build_config(
        self,
        metadata: PersonalityMetadata,
        sections: List[ParsedSection]
    ) -> PersonalityConfig:
        """
        构建PersonalityConfig对象
        
        Args:
            metadata: 元数据
            sections: 解析后的章节
            
        Returns:
            PersonalityConfig对象
        """
        # 提取名称和版本
        name = metadata.description or "Unnamed Personality"
        version = "1.0.0"
        
        # 从标签或源提取名称
        if metadata.tags:
            name = metadata.tags[0]
        
        # 默认特质
        traits: List[PersonalityTrait] = []
        values: List[str] = []
        behaviors: List[BehaviorPattern] = []
        constraints: List[str] = []
        communication = CommunicationStyle(
            tone=CommunicationTone.PROFESSIONAL,
            length=ResponseLength.MODERATE
        )
        domain_expertise: List[str] = []
        
        # 处理各章节
        for section in sections:
            if section.section_type == SectionType.TRAITS:
                traits = section.parsed_data or []
            elif section.section_type == SectionType.VALUES:
                values = section.parsed_data or []
            elif section.section_type == SectionType.BEHAVIORS:
                behaviors = section.parsed_data or []
            elif section.section_type == SectionType.CONSTRAINTS:
                constraints = section.parsed_data or []
            elif section.section_type == SectionType.COMMUNICATION:
                communication = section.parsed_data
            elif section.section_type == SectionType.DOMAIN:
                domain_expertise = section.parsed_data or []
        
        # 应用默认特质（如果没有指定）
        if not traits:
            traits = self._get_default_traits()
        
        return PersonalityConfig(
            name=name,
            version=version,
            traits=traits,
            values=values,
            behaviors=behaviors,
            constraints=constraints,
            communication_style=communication,
            domain_expertise=domain_expertise,
            metadata=metadata
        )
    
    def _get_default_traits(self) -> List[PersonalityTrait]:
        """获取默认特质"""
        return [
            PersonalityTrait(
                dimension=TraitDimension.OPENNESS,
                intensity=3,
                description="中等开放性，愿意尝试新方法",
                examples=[],
                manifestations=[]
            ),
            PersonalityTrait(
                dimension=TraitDimension.CONSCIENTIOUSNESS,
                intensity=3,
                description="中等尽责性，追求平衡",
                examples=[],
                manifestations=[]
            ),
        ]
    
    def _validate_parsed_config(self, config: PersonalityConfig) -> None:
        """
        验证解析后的配置
        
        Args:
            config: 待验证的配置
            
        Raises:
            ParseError: 验证失败
        """
        errors = []
        
        if not config.name:
            errors.append("Personality name is required")
        
        if not config.version:
            errors.append("Version is required")
        elif not re.match(r'^\d+\.\d+\.\d+$', config.version):
            errors.append(f"Invalid version format: {config.version}")
        
        if not config.traits and not self.allow_incomplete:
            errors.append("At least one trait is recommended")
        
        if errors:
            error_msg = "; ".join(errors)
            if self.strict_mode:
                raise ParseError(error_msg)
            else:
                logger.warning(f"Parse warnings: {error_msg}")
    
    def validate_syntax(self, content: str) -> Tuple[bool, List[str]]:
        """
        验证语法正确性
        
        Args:
            content: SOUL.md内容
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 检查frontmatter格式
        fm_match = self.FRONTMATTER_PATTERN.match(content)
        if fm_match:
            yaml_content = fm_match.group(1)
            try:
                yaml.safe_load(yaml_content)
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML: {e}")
        
        # 检查标题层级
        lines = content.split('\n')
        max_level = 0
        for line in lines:
            header_match = self.HEADER_PATTERN.match(line)
            if header_match:
                level = len(header_match.group(1))
                if level > max_level + 1:
                    errors.append(f"Header level jump at line {lines.index(line) + 1}")
                max_level = level
        
        # 检查未闭合的代码块
        code_blocks = self.CODE_BLOCK_PATTERN.findall(content)
        if len(code_blocks) % 2 != 0:
            errors.append("Unclosed code block")
        
        return len(errors) == 0, errors


def create_parser(
    strict: bool = True,
    allow_incomplete: bool = False
) -> SoulParser:
    """
    工厂函数：创建解析器
    
    Args:
        strict: 严格模式
        allow_incomplete: 允许不完整
        
    Returns:
        SoulParser实例
    """
    return SoulParser(strict_mode=strict, allow_incomplete=allow_incomplete)
