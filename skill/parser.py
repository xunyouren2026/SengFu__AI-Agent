#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKILL.md 解析器模块

本模块提供 SKILL.md 文件的解析功能，支持 YAML frontmatter + Markdown body 格式。
负责解析技能元数据、验证技能模式、处理技能依赖关系。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import re
import json
import yaml
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import logging

# 配置日志
logger = logging.getLogger(__name__)


class SkillParseError(Exception):
    """技能解析错误基类"""
    
    def __init__(self, message: str, line_number: Optional[int] = None, 
                 column: Optional[int] = None, context: Optional[str] = None):
        self.message = message
        self.line_number = line_number
        self.column = column
        self.context = context
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        parts = [self.message]
        if self.line_number is not None:
            parts.append(f"行: {self.line_number}")
        if self.column is not None:
            parts.append(f"列: {self.column}")
        if self.context:
            parts.append(f"上下文: {self.context}")
        return " | ".join(parts)


class SkillValidationError(SkillParseError):
    """技能验证错误"""
    pass


class SkillDependencyError(SkillParseError):
    """技能依赖错误"""
    pass


class SkillVersionError(SkillParseError):
    """技能版本错误"""
    pass


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    ARRAY = auto()
    OBJECT = auto()
    FILE = auto()
    DIRECTORY = auto()
    URL = auto()
    EMAIL = auto()
    DATE = auto()
    DATETIME = auto()
    ANY = auto()


@dataclass
class SkillParameter:
    """
    技能参数定义
    
    属性:
        name: 参数名称
        type: 参数类型
        description: 参数描述
        required: 是否必需
        default: 默认值
        validation: 验证规则
        options: 可选值列表（用于枚举类型）
        min_value: 最小值（用于数值类型）
        max_value: 最大值（用于数值类型）
        min_length: 最小长度（用于字符串/数组）
        max_length: 最大长度（用于字符串/数组）
        pattern: 正则表达式模式（用于字符串）
    """
    name: str
    type: ParameterType
    description: str = ""
    required: bool = False
    default: Any = None
    validation: Dict[str, Any] = field(default_factory=dict)
    options: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    
    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        验证参数值
        
        参数:
            value: 待验证的值
            
        返回:
            (是否有效, 错误信息)
        """
        # 检查必需参数
        if self.required and value is None:
            return False, f"参数 '{self.name}' 是必需的"
        
        # 如果值为 None 且有默认值，使用默认值
        if value is None and self.default is not None:
            value = self.default
        
        # 如果值为 None 且不是必需的，验证通过
        if value is None and not self.required:
            return True, None
        
        # 类型验证
        type_valid, type_error = self._validate_type(value)
        if not type_valid:
            return False, type_error
        
        # 枚举验证
        if self.options is not None and value not in self.options:
            return False, f"参数 '{self.name}' 的值必须是以下之一: {self.options}"
        
        # 数值范围验证
        if self.type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 '{self.name}' 不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 '{self.name}' 不能大于 {self.max_value}"
        
        # 长度验证
        if self.type in (ParameterType.STRING, ParameterType.ARRAY):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False, f"参数 '{self.name}' 的长度不能小于 {self.min_length}"
            if self.max_length is not None and length > self.max_length:
                return False, f"参数 '{self.name}' 的长度不能大于 {self.max_length}"
        
        # 正则模式验证
        if self.pattern is not None and self.type == ParameterType.STRING:
            if not re.match(self.pattern, value):
                return False, f"参数 '{self.name}' 格式不匹配模式 '{self.pattern}'"
        
        return True, None
    
    def _validate_type(self, value: Any) -> Tuple[bool, Optional[str]]:
        """验证值类型"""
        type_map = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: list,
            ParameterType.OBJECT: dict,
        }
        
        if self.type == ParameterType.ANY:
            return True, None
        
        if self.type in type_map:
            expected_type = type_map[self.type]
            if not isinstance(value, expected_type):
                return False, f"参数 '{self.name}' 类型错误，期望 {self.type.name}，实际为 {type(value).__name__}"
        
        # 特殊类型验证
        if self.type == ParameterType.FILE:
            if not isinstance(value, (str, Path)):
                return False, f"参数 '{self.name}' 必须是文件路径"
        
        if self.type == ParameterType.DIRECTORY:
            if not isinstance(value, (str, Path)):
                return False, f"参数 '{self.name}' 必须是目录路径"
        
        if self.type == ParameterType.URL:
            if not isinstance(value, str):
                return False, f"参数 '{self.name}' 必须是 URL 字符串"
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, value, re.IGNORECASE):
                return False, f"参数 '{self.name}' 不是有效的 URL"
        
        if self.type == ParameterType.EMAIL:
            if not isinstance(value, str):
                return False, f"参数 '{self.name}' 必须是邮箱字符串"
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                return False, f"参数 '{self.name}' 不是有效的邮箱地址"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type.name,
            'description': self.description,
            'required': self.required,
            'default': self.default,
            'validation': self.validation,
            'options': self.options,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'pattern': self.pattern,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillParameter:
        """从字典创建"""
        param_data = data.copy()
        if 'type' in param_data and isinstance(param_data['type'], str):
            param_data['type'] = ParameterType[param_data['type'].upper()]
        return cls(**param_data)


@dataclass
class SkillDependency:
    """
    技能依赖定义
    
    属性:
        skill_id: 依赖的技能 ID
        version_constraint: 版本约束（如 ">=1.0.0", "^2.0.0"）
        optional: 是否为可选依赖
        reason: 依赖原因说明
    """
    skill_id: str
    version_constraint: str = "*"
    optional: bool = False
    reason: str = ""
    
    def is_version_compatible(self, version: str) -> bool:
        """
        检查版本是否兼容
        
        参数:
            version: 要检查的版本
            
        返回:
            是否兼容
        """
        return VersionConstraint(self.version_constraint).matches(version)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'skill_id': self.skill_id,
            'version_constraint': self.version_constraint,
            'optional': self.optional,
            'reason': self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillDependency:
        """从字典创建"""
        return cls(**data)


class VersionConstraint:
    """版本约束解析器"""
    
    def __init__(self, constraint: str):
        """
        初始化版本约束
        
        参数:
            constraint: 版本约束字符串（如 ">=1.0.0", "^2.0.0", "~1.2.0"）
        """
        self.constraint = constraint
        self._parse_constraint()
    
    def _parse_constraint(self) -> None:
        """解析版本约束"""
        constraint = self.constraint.strip()
        
        # 通配符
        if constraint == "*":
            self.min_version = None
            self.max_version = None
            self.operator = "*"
            return
        
        # 精确版本
        if re.match(r'^\d', constraint):
            self.min_version = self._parse_version(constraint)
            self.max_version = self.min_version
            self.operator = "="
            return
        
        # 操作符约束 (>=, >, <=, <, =)
        match = re.match(r'^(>=|>|<=|<|=)\s*(.+)$', constraint)
        if match:
            self.operator = match.group(1)
            version_str = match.group(2)
            version = self._parse_version(version_str)
            
            if self.operator in (">=", ">"):
                self.min_version = version
                self.max_version = None
            elif self.operator in ("<=", "<"):
                self.min_version = None
                self.max_version = version
            else:  # =
                self.min_version = version
                self.max_version = version
            return
        
        # 语义化版本约束 (^, ~)
        if constraint.startswith("^"):
            version_str = constraint[1:]
            version = self._parse_version(version_str)
            self.min_version = version
            self.max_version = (version[0] + 1, 0, 0)
            self.operator = "^"
            return
        
        if constraint.startswith("~"):
            version_str = constraint[1:]
            version = self._parse_version(version_str)
            self.min_version = version
            self.max_version = (version[0], version[1] + 1, 0)
            self.operator = "~"
            return
        
        # 范围约束 (1.0.0 - 2.0.0)
        match = re.match(r'^(\d+\.\d+(?:\.\d+)?)\s*-\s*(\d+\.\d+(?:\.\d+)?)$', constraint)
        if match:
            self.min_version = self._parse_version(match.group(1))
            self.max_version = self._parse_version(match.group(2))
            self.operator = "-"
            return
        
        raise SkillVersionError(f"无法解析版本约束: {constraint}")
    
    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """解析版本字符串为元组"""
        parts = version_str.split(".")
        if len(parts) < 2:
            raise SkillVersionError(f"无效的版本格式: {version_str}")
        
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        return (major, minor, patch)
    
    def matches(self, version: str) -> bool:
        """
        检查版本是否满足约束
        
        参数:
            version: 要检查的版本字符串
            
        返回:
            是否满足约束
        """
        try:
            v = self._parse_version(version)
        except SkillVersionError:
            return False
        
        if self.operator == "*":
            return True
        
        if self.min_version is not None:
            if self.operator == ">" and v <= self.min_version:
                return False
            if self.operator in (">=", "=", "^", "~", "-") and v < self.min_version:
                return False
        
        if self.max_version is not None:
            if self.operator == "<" and v >= self.max_version:
                return False
            if self.operator in ("<=", "=", "-") and v > self.max_version:
                return False
            if self.operator in ("^", "~") and v >= self.max_version:
                return False
        
        return True


@dataclass
class SkillMetadata:
    """
    技能元数据
    
    属性:
        id: 技能唯一标识
        name: 技能名称
        description: 技能描述
        version: 版本号
        author: 作者信息
        license: 许可证
        homepage: 主页 URL
        repository: 代码仓库 URL
        tags: 标签列表
        category: 分类
        icon: 图标 URL 或 emoji
        created_at: 创建时间
        updated_at: 更新时间
    """
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    icon: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'license': self.license,
            'homepage': self.homepage,
            'repository': self.repository,
            'tags': self.tags,
            'category': self.category,
            'icon': self.icon,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillMetadata:
        """从字典创建"""
        data = data.copy()
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class SkillExecution:
    """
    技能执行配置
    
    属性:
        command: 执行命令
        script: 脚本路径
        interpreter: 解释器（python, bash, node 等）
        working_dir: 工作目录
        env: 环境变量
        timeout: 超时时间（秒）
        parallel: 是否支持并行执行
        retry: 重试次数
    """
    command: Optional[str] = None
    script: Optional[str] = None
    interpreter: str = "python"
    working_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    parallel: bool = False
    retry: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'command': self.command,
            'script': self.script,
            'interpreter': self.interpreter,
            'working_dir': self.working_dir,
            'env': self.env,
            'timeout': self.timeout,
            'parallel': self.parallel,
            'retry': self.retry,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillExecution:
        """从字典创建"""
        return cls(**data)


@dataclass
class ParsedSkill:
    """
    解析后的技能对象
    
    属性:
        metadata: 技能元数据
        parameters: 参数定义列表
        dependencies: 依赖列表
        allowed_tools: 允许使用的工具列表
        execution: 执行配置
        body: Markdown 正文内容
        raw_content: 原始文件内容
        file_path: 源文件路径
        checksum: 内容校验和
    """
    metadata: SkillMetadata
    parameters: List[SkillParameter] = field(default_factory=list)
    dependencies: List[SkillDependency] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    execution: Optional[SkillExecution] = None
    body: str = ""
    raw_content: str = ""
    file_path: Optional[Path] = None
    checksum: str = ""
    
    def __post_init__(self):
        """初始化后计算校验和"""
        if self.raw_content and not self.checksum:
            self.checksum = hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]
    
    def validate_parameters(self, values: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证参数值
        
        参数:
            values: 参数值字典
            
        返回:
            (是否全部有效, 错误信息列表)
        """
        errors = []
        
        # 检查未知参数
        param_names = {p.name for p in self.parameters}
        for key in values:
            if key not in param_names:
                errors.append(f"未知参数: '{key}'")
        
        # 验证每个参数
        for param in self.parameters:
            value = values.get(param.name)
            valid, error = param.validate_value(value)
            if not valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def get_parameter(self, name: str) -> Optional[SkillParameter]:
        """获取参数定义"""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def get_required_parameters(self) -> List[SkillParameter]:
        """获取所有必需参数"""
        return [p for p in self.parameters if p.required]
    
    def get_optional_parameters(self) -> List[SkillParameter]:
        """获取所有可选参数"""
        return [p for p in self.parameters if not p.required]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'metadata': self.metadata.to_dict(),
            'parameters': [p.to_dict() for p in self.parameters],
            'dependencies': [d.to_dict() for d in self.dependencies],
            'allowed_tools': self.allowed_tools,
            'execution': self.execution.to_dict() if self.execution else None,
            'body': self.body,
            'checksum': self.checksum,
            'file_path': str(self.file_path) if self.file_path else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ParsedSkill:
        """从字典创建"""
        metadata = SkillMetadata.from_dict(data['metadata'])
        parameters = [SkillParameter.from_dict(p) for p in data.get('parameters', [])]
        dependencies = [SkillDependency.from_dict(d) for d in data.get('dependencies', [])]
        execution = SkillExecution.from_dict(data['execution']) if data.get('execution') else None
        
        return cls(
            metadata=metadata,
            parameters=parameters,
            dependencies=dependencies,
            allowed_tools=data.get('allowed_tools', []),
            execution=execution,
            body=data.get('body', ''),
            checksum=data.get('checksum', ''),
            file_path=Path(data['file_path']) if data.get('file_path') else None,
        )


class SkillSchemaValidator:
    """技能模式验证器"""
    
    # 必需字段
    REQUIRED_FIELDS = ['id', 'name', 'version']
    
    # 有效许可证列表
    VALID_LICENSES = [
        'MIT', 'Apache-2.0', 'GPL-3.0', 'GPL-2.0', 'BSD-3-Clause',
        'BSD-2-Clause', 'ISC', 'MPL-2.0', 'LGPL-3.0', 'LGPL-2.1',
        'Unlicense', 'Proprietary', 'CC0-1.0', 'CC-BY-4.0',
    ]
    
    # 有效分类列表
    VALID_CATEGORIES = [
        'general', 'development', 'productivity', 'communication',
        'data', 'security', 'media', 'system', 'ai', 'automation',
        'analytics', 'integration', 'utility',
    ]
    
    # 有效解释器列表
    VALID_INTERPRETERS = [
        'python', 'python3', 'bash', 'sh', 'node', 'nodejs',
        'ruby', 'perl', 'php', 'deno', 'bun',
    ]
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, skill: ParsedSkill) -> Tuple[bool, List[str], List[str]]:
        """
        验证技能对象
        
        参数:
            skill: 要验证的技能对象
            
        返回:
            (是否有效, 错误列表, 警告列表)
        """
        self.errors = []
        self.warnings = []
        
        self._validate_metadata(skill.metadata)
        self._validate_parameters(skill.parameters)
        self._validate_dependencies(skill.dependencies)
        self._validate_execution(skill.execution)
        self._validate_allowed_tools(skill.allowed_tools)
        self._validate_body(skill.body)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_metadata(self, metadata: SkillMetadata) -> None:
        """验证元数据"""
        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            value = getattr(metadata, field, None)
            if not value:
                self.errors.append(f"元数据缺少必需字段: '{field}'")
        
        # 验证 ID 格式
        if metadata.id and not re.match(r'^[a-z][a-z0-9-]*$', metadata.id):
            self.errors.append(
                f"技能 ID '{metadata.id}' 格式无效，只能包含小写字母、数字和连字符，且必须以字母开头"
            )
        
        # 验证版本号格式
        if metadata.version and not re.match(r'^\d+\.\d+(?:\.\d+)?', metadata.version):
            self.errors.append(f"版本号 '{metadata.version}' 格式无效，应为语义化版本格式")
        
        # 验证许可证
        if metadata.license and metadata.license not in self.VALID_LICENSES:
            self.warnings.append(
                f"许可证 '{metadata.license}' 不是常见开源许可证，请确认是否正确"
            )
        
        # 验证分类
        if metadata.category and metadata.category not in self.VALID_CATEGORIES:
            self.warnings.append(
                f"分类 '{metadata.category}' 不是标准分类"
            )
        
        # 验证 URL 格式
        for url_field in ['homepage', 'repository']:
            url = getattr(metadata, url_field, '')
            if url and not re.match(r'^https?://', url):
                self.warnings.append(f"{url_field} 不是有效的 HTTP(S) URL")
    
    def _validate_parameters(self, parameters: List[SkillParameter]) -> None:
        """验证参数定义"""
        param_names = set()
        
        for param in parameters:
            # 检查参数名唯一性
            if param.name in param_names:
                self.errors.append(f"参数名 '{param.name}' 重复定义")
            param_names.add(param.name)
            
            # 验证参数名格式
            if not re.match(r'^[a-z][a-z0-9_]*$', param.name):
                self.errors.append(
                    f"参数名 '{param.name}' 格式无效，只能包含小写字母、数字和下划线，且必须以字母开头"
                )
            
            # 验证数值约束
            if param.min_value is not None and param.max_value is not None:
                if param.min_value > param.max_value:
                    self.errors.append(
                        f"参数 '{param.name}' 的最小值不能大于最大值"
                    )
            
            # 验证长度约束
            if param.min_length is not None and param.max_length is not None:
                if param.min_length > param.max_length:
                    self.errors.append(
                        f"参数 '{param.name}' 的最小长度不能大于最大长度"
                    )
            
            # 验证正则表达式
            if param.pattern:
                try:
                    re.compile(param.pattern)
                except re.error as e:
                    self.errors.append(
                        f"参数 '{param.name}' 的正则表达式无效: {e}"
                    )
    
    def _validate_dependencies(self, dependencies: List[SkillDependency]) -> None:
        """验证依赖"""
        for dep in dependencies:
            # 验证技能 ID 格式
            if not re.match(r'^[a-z][a-z0-9-]*$', dep.skill_id):
                self.errors.append(
                    f"依赖技能 ID '{dep.skill_id}' 格式无效"
                )
            
            # 验证版本约束
            try:
                VersionConstraint(dep.version_constraint)
            except SkillVersionError as e:
                self.errors.append(
                    f"依赖 '{dep.skill_id}' 的版本约束无效: {e}"
                )
    
    def _validate_execution(self, execution: Optional[SkillExecution]) -> None:
        """验证执行配置"""
        if execution is None:
            return
        
        # 验证解释器
        if execution.interpreter not in self.VALID_INTERPRETERS:
            self.warnings.append(
                f"解释器 '{execution.interpreter}' 不是标准解释器"
            )
        
        # 验证超时时间
        if execution.timeout < 0:
            self.errors.append("超时时间不能为负数")
        
        # 验证重试次数
        if execution.retry < 0:
            self.errors.append("重试次数不能为负数")
        
        # 检查 command 和 script 至少有一个
        if not execution.command and not execution.script:
            self.warnings.append("执行配置中既没有 command 也没有 script")
    
    def _validate_allowed_tools(self, allowed_tools: List[str]) -> None:
        """验证允许的工具列表"""
        valid_tools = {
            'Read', 'Write', 'SearchReplace', 'DeleteFile',
            'Glob', 'Grep', 'Read', 'Bash', 'WebSearch',
            'WebFetch', 'Skill', 'GenerateImage', 'TodoWrite',
        }
        
        for tool in allowed_tools:
            if tool not in valid_tools:
                self.warnings.append(f"工具 '{tool}' 不是标准工具名称")
    
    def _validate_body(self, body: str) -> None:
        """验证正文内容"""
        if not body or not body.strip():
            self.warnings.append("技能正文内容为空")


class SkillParser:
    """
    SKILL.md 文件解析器
    
    解析 YAML frontmatter + Markdown body 格式的技能文件。
    """
    
    # Frontmatter 分隔符
    FRONTMATTER_DELIMITER = '---'
    
    def __init__(self, validate: bool = True):
        """
        初始化解析器
        
        参数:
            validate: 是否进行模式验证
        """
        self.validate = validate
        self.validator = SkillSchemaValidator()
    
    def parse(self, content: str, file_path: Optional[Path] = None) -> ParsedSkill:
        """
        解析技能内容
        
        参数:
            content: SKILL.md 文件内容
            file_path: 源文件路径（用于错误报告）
            
        返回:
            解析后的技能对象
            
        抛出:
            SkillParseError: 解析失败
        """
        try:
            frontmatter, body = self._split_content(content)
            metadata_dict = self._parse_frontmatter(frontmatter)
            skill = self._build_skill(metadata_dict, body, content, file_path)
            
            if self.validate:
                valid, errors, warnings = self.validator.validate(skill)
                if not valid:
                    raise SkillValidationError(
                        f"技能验证失败: {'; '.join(errors)}"
                    )
                if warnings:
                    logger.warning(f"技能 '{skill.metadata.id}' 验证警告: {warnings}")
            
            return skill
            
        except yaml.YAMLError as e:
            raise SkillParseError(
                f"YAML 解析错误: {e}",
                context=str(file_path) if file_path else None
            )
        except Exception as e:
            if isinstance(e, SkillParseError):
                raise
            raise SkillParseError(
                f"解析错误: {e}",
                context=str(file_path) if file_path else None
            )
    
    def parse_file(self, file_path: Union[str, Path]) -> ParsedSkill:
        """
        解析技能文件
        
        参数:
            file_path: 文件路径
            
        返回:
            解析后的技能对象
        """
        path = Path(file_path)
        if not path.exists():
            raise SkillParseError(f"文件不存在: {path}")
        
        if not path.is_file():
            raise SkillParseError(f"路径不是文件: {path}")
        
        content = path.read_text(encoding='utf-8')
        return self.parse(content, path)
    
    def _split_content(self, content: str) -> Tuple[str, str]:
        """
        分割 frontmatter 和 body
        
        参数:
            content: 完整内容
            
        返回:
            (frontmatter, body)
        """
        lines = content.split('\n')
        
        # 检查是否以分隔符开始
        if not lines or lines[0].strip() != self.FRONTMATTER_DELIMITER:
            raise SkillParseError(
                "文件必须以 '---' 开头",
                line_number=1
            )
        
        # 查找结束分隔符
        end_idx = None
        for i, line in enumerate(lines[1:], start=2):
            if line.strip() == self.FRONTMATTER_DELIMITER:
                end_idx = i
                break
        
        if end_idx is None:
            raise SkillParseError(
                "找不到 frontmatter 结束分隔符 '---'"
            )
        
        frontmatter = '\n'.join(lines[1:end_idx - 1])
        body = '\n'.join(lines[end_idx:])
        
        return frontmatter, body
    
    def _parse_frontmatter(self, frontmatter: str) -> Dict[str, Any]:
        """
        解析 frontmatter YAML
        
        参数:
            frontmatter: YAML 内容
            
        返回:
            解析后的字典
        """
        return yaml.safe_load(frontmatter) or {}
    
    def _build_skill(self, metadata_dict: Dict[str, Any], body: str,
                     raw_content: str, file_path: Optional[Path]) -> ParsedSkill:
        """
        构建技能对象
        
        参数:
            metadata_dict: 元数据字典
            body: Markdown 正文
            raw_content: 原始内容
            file_path: 文件路径
            
        返回:
            技能对象
        """
        # 提取元数据
        metadata = SkillMetadata(
            id=metadata_dict.get('id', ''),
            name=metadata_dict.get('name', ''),
            description=metadata_dict.get('description', ''),
            version=metadata_dict.get('version', '1.0.0'),
            author=metadata_dict.get('author', ''),
            license=metadata_dict.get('license', 'MIT'),
            homepage=metadata_dict.get('homepage', ''),
            repository=metadata_dict.get('repository', ''),
            tags=metadata_dict.get('tags', []),
            category=metadata_dict.get('category', 'general'),
            icon=metadata_dict.get('icon', ''),
            created_at=datetime.now() if 'created_at' not in metadata_dict else None,
            updated_at=datetime.now(),
        )
        
        # 解析参数
        parameters = []
        for param_dict in metadata_dict.get('parameters', []):
            param = self._parse_parameter(param_dict)
            parameters.append(param)
        
        # 解析依赖
        dependencies = []
        for dep_dict in metadata_dict.get('dependencies', []):
            if isinstance(dep_dict, str):
                # 简写格式: "skill-id>=1.0.0"
                dep = self._parse_dependency_string(dep_dict)
            else:
                dep = SkillDependency.from_dict(dep_dict)
            dependencies.append(dep)
        
        # 解析执行配置
        execution = None
        exec_dict = metadata_dict.get('execution')
        if exec_dict:
            execution = SkillExecution.from_dict(exec_dict)
        
        # 允许的工具
        allowed_tools = metadata_dict.get('allowed_tools', [])
        
        return ParsedSkill(
            metadata=metadata,
            parameters=parameters,
            dependencies=dependencies,
            allowed_tools=allowed_tools,
            execution=execution,
            body=body,
            raw_content=raw_content,
            file_path=file_path,
        )
    
    def _parse_parameter(self, param_dict: Dict[str, Any]) -> SkillParameter:
        """解析参数字典"""
        param_data = param_dict.copy()
        
        # 转换类型字符串为枚举
        if 'type' in param_data and isinstance(param_data['type'], str):
            try:
                param_data['type'] = ParameterType[param_data['type'].upper()]
            except KeyError:
                raise SkillParseError(f"未知的参数类型: {param_data['type']}")
        
        return SkillParameter.from_dict(param_data)
    
    def _parse_dependency_string(self, dep_str: str) -> SkillDependency:
        """
        解析依赖字符串
        
        支持格式:
        - "skill-id"
        - "skill-id>=1.0.0"
        - "skill-id@^2.0.0"
        """
        # 尝试匹配带版本的格式
        match = re.match(r'^([a-z][a-z0-9-]*)(?:[@>=<^~]+(.+))?$', dep_str)
        if not match:
            raise SkillParseError(f"无法解析依赖字符串: {dep_str}")
        
        skill_id = match.group(1)
        version_constraint = match.group(2) or "*"
        
        return SkillDependency(
            skill_id=skill_id,
            version_constraint=version_constraint,
        )


class SkillDependencyResolver:
    """技能依赖解析器"""
    
    def __init__(self, skill_registry: Dict[str, ParsedSkill]):
        """
        初始化解析器
        
        参数:
            skill_registry: 技能注册表
        """
        self.skill_registry = skill_registry
    
    def resolve(self, skill: ParsedSkill) -> Tuple[List[ParsedSkill], List[str]]:
        """
        解析技能的所有依赖
        
        参数:
            skill: 要解析依赖的技能
            
        返回:
            (解析后的依赖技能列表, 错误信息列表)
        """
        resolved: Dict[str, ParsedSkill] = {}
        errors: List[str] = []
        
        self._resolve_recursive(skill, resolved, set(), errors)
        
        return list(resolved.values()), errors
    
    def _resolve_recursive(self, skill: ParsedSkill, 
                          resolved: Dict[str, ParsedSkill],
                          resolving: Set[str],
                          errors: List[str]) -> None:
        """
        递归解析依赖
        
        参数:
            skill: 当前技能
            resolved: 已解析的技能字典
            resolving: 正在解析中的技能 ID 集合（用于检测循环依赖）
            errors: 错误列表
        """
        if skill.metadata.id in resolved:
            return
        
        if skill.metadata.id in resolving:
            errors.append(f"检测到循环依赖: {skill.metadata.id}")
            return
        
        resolving.add(skill.metadata.id)
        
        for dep in skill.dependencies:
            dep_skill = self.skill_registry.get(dep.skill_id)
            
            if dep_skill is None:
                if dep.optional:
                    logger.warning(f"可选依赖 '{dep.skill_id}' 未找到")
                else:
                    errors.append(f"必需依赖 '{dep.skill_id}' 未找到")
                continue
            
            # 检查版本兼容性
            if not dep.is_version_compatible(dep_skill.metadata.version):
                errors.append(
                    f"依赖 '{dep.skill_id}' 版本不兼容: "
                    f"需要 {dep.version_constraint}, "
                    f"实际为 {dep_skill.metadata.version}"
                )
                continue
            
            # 递归解析
            self._resolve_recursive(dep_skill, resolved, resolving, errors)
            
            if dep.skill_id not in resolved:
                resolved[dep.skill_id] = dep_skill
        
        resolving.remove(skill.metadata.id)
    
    def check_circular_dependencies(self, skill: ParsedSkill) -> Optional[List[str]]:
        """
        检查循环依赖
        
        参数:
            skill: 要检查的技能
            
        返回:
            如果存在循环依赖，返回循环路径；否则返回 None
        """
        visited: Set[str] = set()
        path: List[str] = []
        
        def dfs(current_id: str) -> Optional[List[str]]:
            if current_id in path:
                cycle_start = path.index(current_id)
                return path[cycle_start:] + [current_id]
            
            if current_id in visited:
                return None
            
            visited.add(current_id)
            path.append(current_id)
            
            current_skill = self.skill_registry.get(current_id)
            if current_skill:
                for dep in current_skill.dependencies:
                    result = dfs(dep.skill_id)
                    if result:
                        return result
            
            path.pop()
            return None
        
        return dfs(skill.metadata.id)


class SkillTemplateEngine:
    """技能模板引擎"""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
    
    def register_template(self, name: str, template: str) -> None:
        """
        注册模板
        
        参数:
            name: 模板名称
            template: 模板内容
        """
        self.templates[name] = template
    
    def render(self, template_name: str, **kwargs) -> str:
        """
        渲染模板
        
        参数:
            template_name: 模板名称
            **kwargs: 模板变量
            
        返回:
            渲染后的内容
        """
        if template_name not in self.templates:
            raise SkillParseError(f"模板不存在: {template_name}")
        
        template = self.templates[template_name]
        
        # 简单的变量替换
        for key, value in kwargs.items():
            placeholder = f"{{{{ {key} }}}}"
            template = template.replace(placeholder, str(value))
        
        return template
    
    def create_skill_from_template(self, template_name: str, 
                                    metadata: Dict[str, Any]) -> str:
        """
        从模板创建技能文件
        
        参数:
            template_name: 模板名称
            metadata: 元数据字典
            
        返回:
            技能文件内容
        """
        return self.render(template_name, **metadata)


# 内置模板
DEFAULT_SKILL_TEMPLATE = '''---
id: {{ id }}
name: {{ name }}
description: {{ description }}
version: "1.0.0"
author: {{ author }}
license: MIT
category: general
parameters:
  - name: input
    type: string
    description: 输入参数
    required: true
allowed_tools:
  - Read
  - Write
  - Bash
---

# {{ name }}

{{ description }}

## 使用方法

```bash
# 使用示例
```

## 参数说明

- `input`: 输入参数

## 注意事项

- 请确保输入参数正确
'''


# 单元测试存根
class TestSkillParser:
    """SkillParser 单元测试"""
    
    def test_parse_valid_skill(self) -> None:
        """测试解析有效技能文件"""
        content = '''---
id: test-skill
name: Test Skill
description: A test skill
version: "1.0.0"
author: Test Author
parameters:
  - name: input
    type: string
    description: Input parameter
    required: true
allowed_tools:
  - Read
  - Write
---

# Test Skill

This is a test skill.
'''
        parser = SkillParser()
        skill = parser.parse(content)
        
        assert skill.metadata.id == "test-skill"
        assert skill.metadata.name == "Test Skill"
        assert len(skill.parameters) == 1
        assert skill.parameters[0].name == "input"
    
    def test_parse_invalid_yaml(self) -> None:
        """测试解析无效 YAML"""
        content = '''---
invalid: yaml: [
---

Body
'''
        parser = SkillParser()
        try:
            parser.parse(content)
            assert False, "应该抛出异常"
        except SkillParseError:
            pass
    
    def test_validate_parameters(self) -> None:
        """测试参数验证"""
        content = '''---
id: test-skill
name: Test Skill
description: A test skill
version: "1.0.0"
parameters:
  - name: count
    type: integer
    description: Count parameter
    required: true
    min_value: 0
    max_value: 100
---

# Test Skill
'''
        parser = SkillParser()
        skill = parser.parse(content)
        
        # 有效值
        valid, errors = skill.validate_parameters({'count': 50})
        assert valid
        
        # 超出范围
        valid, errors = skill.validate_parameters({'count': 150})
        assert not valid
        
        # 缺少必需参数
        valid, errors = skill.validate_parameters({})
        assert not valid
    
    def test_version_constraint(self) -> None:
        """测试版本约束"""
        # 精确版本
        vc = VersionConstraint("1.0.0")
        assert vc.matches("1.0.0")
        assert not vc.matches("1.0.1")
        
        # >= 约束
        vc = VersionConstraint(">=1.0.0")
        assert vc.matches("1.0.0")
        assert vc.matches("2.0.0")
        assert not vc.matches("0.9.0")
        
        # ^ 约束
        vc = VersionConstraint("^1.2.0")
        assert vc.matches("1.2.0")
        assert vc.matches("1.5.0")
        assert not vc.matches("2.0.0")
        assert not vc.matches("1.1.0")
    
    def test_dependency_resolution(self) -> None:
        """测试依赖解析"""
        # 创建测试技能
        skill_a = ParsedSkill(
            metadata=SkillMetadata(id="skill-a", name="Skill A", version="1.0.0"),
            dependencies=[SkillDependency(skill_id="skill-b", version_constraint=">=1.0.0")]
        )
        skill_b = ParsedSkill(
            metadata=SkillMetadata(id="skill-b", name="Skill B", version="1.0.0"),
            dependencies=[SkillDependency(skill_id="skill-c", version_constraint=">=1.0.0")]
        )
        skill_c = ParsedSkill(
            metadata=SkillMetadata(id="skill-c", name="Skill C", version="1.0.0"),
            dependencies=[]
        )
        
        registry = {
            "skill-a": skill_a,
            "skill-b": skill_b,
            "skill-c": skill_c,
        }
        
        resolver = SkillDependencyResolver(registry)
        resolved, errors = resolver.resolve(skill_a)
        
        assert len(errors) == 0
        assert len(resolved) == 2
        assert skill_b in resolved
        assert skill_c in resolved


def create_default_parser() -> SkillParser:
    """创建默认解析器实例"""
    return SkillParser(validate=True)


def parse_skill_file(file_path: Union[str, Path]) -> ParsedSkill:
    """
    便捷函数：解析技能文件
    
    参数:
        file_path: 文件路径
        
    返回:
        解析后的技能对象
    """
    parser = create_default_parser()
    return parser.parse_file(file_path)


def parse_skill_content(content: str) -> ParsedSkill:
    """
    便捷函数：解析技能内容
    
    参数:
        content: 技能内容
        
    返回:
        解析后的技能对象
    """
    parser = create_default_parser()
    return parser.parse(content)
