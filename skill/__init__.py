#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI Framework Skill 系统

本模块提供技能系统的核心功能，包括技能解析、执行和市场管理。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

# 版本信息
__version__ = "1.0.0"
__author__ = "AGI Framework Team"

# 从 parser 模块导出
from .parser import (
    # 核心类
    SkillParser,
    ParsedSkill,
    SkillMetadata,
    SkillParameter,
    SkillDependency,
    SkillExecution,
    SkillSchemaValidator,
    SkillDependencyResolver,
    SkillTemplateEngine,
    
    # 枚举
    ParameterType,
    
    # 异常
    SkillParseError,
    SkillValidationError,
    SkillDependencyError,
    SkillVersionError,
    
    # 便捷函数
    parse_skill_file,
    parse_skill_content,
    create_default_parser,
    
    # 版本约束
    VersionConstraint,
)

# 从 executor 模块导出
from .executor import (
    # 核心类
    SkillExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionTask,
    ExecutionMonitor,
    ExecutionQueue,
    
    # 执行器
    ScriptExecutor,
    BashExecutor,
    PythonExecutor,
    NodeExecutor,
    
    # 枚举
    ExecutionStatus,
    
    # 异常
    ExecutionError,
    ExecutionTimeoutError,
    ExecutionCancelledError,
    ExecutionResourceError,
    
    # 便捷函数
    execute_bash,
    execute_python,
    execute_node,
    get_default_executor,
)

# 从 market 模块导出
from .market import (
    # 核心类
    SkillRegistry,
    SkillMarket,
    SkillSearchEngine,
    SkillVersionManager,
    
    # 数据类
    SkillEntry,
    SkillVersion,
    SkillReview,
    InstalledSkill,
    
    # 枚举
    SkillStatus,
    
    # 异常
    MarketError,
    SkillNotFoundError,
    SkillAlreadyExistsError,
    VersionConflictError,
    InstallationError,
    SearchError,
    
    # 便捷函数
    create_market,
    search_skills,
    install_skill,
    uninstall_skill,
    get_default_market,
)

# 公开 API
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # Parser 模块
    "SkillParser",
    "ParsedSkill",
    "SkillMetadata",
    "SkillParameter",
    "SkillDependency",
    "SkillExecution",
    "SkillSchemaValidator",
    "SkillDependencyResolver",
    "SkillTemplateEngine",
    "ParameterType",
    "SkillParseError",
    "SkillValidationError",
    "SkillDependencyError",
    "SkillVersionError",
    "VersionConstraint",
    "parse_skill_file",
    "parse_skill_content",
    "create_default_parser",
    
    # Executor 模块
    "SkillExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionTask",
    "ExecutionMonitor",
    "ExecutionQueue",
    "ScriptExecutor",
    "BashExecutor",
    "PythonExecutor",
    "NodeExecutor",
    "ExecutionStatus",
    "ExecutionError",
    "ExecutionTimeoutError",
    "ExecutionCancelledError",
    "ExecutionResourceError",
    "execute_bash",
    "execute_python",
    "execute_node",
    "get_default_executor",
    
    # Market 模块
    "SkillRegistry",
    "SkillMarket",
    "SkillSearchEngine",
    "SkillVersionManager",
    "SkillEntry",
    "SkillVersion",
    "SkillReview",
    "InstalledSkill",
    "SkillStatus",
    "MarketError",
    "SkillNotFoundError",
    "SkillAlreadyExistsError",
    "VersionConflictError",
    "InstallationError",
    "SearchError",
    "create_market",
    "search_skills",
    "install_skill",
    "uninstall_skill",
    "get_default_market",
]


def get_skill_system_info() -> dict:
    """
    获取技能系统信息
    
    返回:
        包含系统信息的字典
    """
    return {
        "version": __version__,
        "author": __author__,
        "modules": ["parser", "executor", "market"],
        "features": [
            "SKILL.md parsing",
            "Multi-interpreter execution",
            "Skill marketplace",
            "Dependency resolution",
            "Version management",
        ],
    }


def init_skill_system(storage_path: str = None, install_dir: str = None) -> SkillMarket:
    """
    初始化技能系统
    
    参数:
        storage_path: 注册表存储路径
        install_dir: 技能安装目录
        
    返回:
        技能市场实例
    """
    from pathlib import Path
    
    if storage_path is None:
        storage_path = Path.home() / ".agi_skills" / "registry.json"
    else:
        storage_path = Path(storage_path)
    
    if install_dir is None:
        install_dir = Path.home() / ".agi_skills" / "installed"
    else:
        install_dir = Path(install_dir)
    
    return create_market(storage_path, install_dir)
