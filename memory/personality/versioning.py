"""
Personality Versioning - 人格版本管理

该模块提供人格配置的版本控制、迁移和兼容性检查功能。

核心功能:
- SemVer版本控制
- 版本迁移
- 兼容性检查
- 版本历史管理

使用示例:
    version_manager = VersionManager()
    
    # 检查兼容性
    is_compatible = version_manager.check_compatibility(
        current_version="1.0.0",
        target_version="1.1.0"
    )
    
    # 执行迁移
    migrated_config = version_manager.migrate(config, "1.0.0", "2.0.0")
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from . import (
    PersonalityConfig, PersonalityTrait, BehaviorPattern,
    CommunicationStyle, TraitDimension, VersionError
)

logger = logging.getLogger(__name__)


class VersionComponent(Enum):
    """版本组件类型"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


class CompatibilityLevel(Enum):
    """兼容性级别"""
    FULL = "full"          # 完全兼容
    BACKWARD = "backward"   # 向后兼容
    FORWARD = "forward"     # 向前兼容
    INCOMPATIBLE = "incompatible"  # 不兼容


@dataclass
class Version:
    """
    语义化版本
    
    Attributes:
        major: 主版本号
        minor: 次版本号
        patch: 补丁版本号
        prerelease: 预发布标签
        build: 构建元数据
    """
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """字符串表示"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """
        解析版本字符串
        
        Args:
            version_str: 版本字符串
            
        Returns:
            Version对象
        """
        # 移除前缀v
        version_str = version_str.lstrip('v')
        
        # 解析构建元数据
        build = None
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)
        
        # 解析预发布标签
        prerelease = None
        if '-' in version_str:
            version_str, prerelease = version_str.split('-', 1)
        
        # 解析主版本号
        match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?', version_str)
        if not match:
            raise VersionError(f"Invalid version format: {version_str}")
        
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        patch = int(match.group(3)) if match.group(3) else 0
        
        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build
        )
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """转换为元组"""
        return (self.major, self.minor, self.patch)
    
    def __lt__(self, other: "Version") -> bool:
        """小于比较"""
        return self.to_tuple() < other.to_tuple()
    
    def __le__(self, other: "Version") -> bool:
        """小于等于比较"""
        return self.to_tuple() <= other.to_tuple()
    
    def __gt__(self, other: "Version") -> bool:
        """大于比较"""
        return self.to_tuple() > other.to_tuple()
    
    def __ge__(self, other: "Version") -> bool:
        """大于等于比较"""
        return self.to_tuple() >= other.to_tuple()
    
    def __eq__(self, other: object) -> bool:
        """等于比较"""
        if not isinstance(other, Version):
            return False
        return self.to_tuple() == other.to_tuple()
    
    def bump_major(self) -> "Version":
        """增加主版本号"""
        return Version(
            major=self.major + 1,
            minor=0,
            patch=0
        )
    
    def bump_minor(self) -> "Version":
        """增加次版本号"""
        return Version(
            major=self.major,
            minor=self.minor + 1,
            patch=0
        )
    
    def bump_patch(self) -> "Version":
        """增加补丁版本号"""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1
        )


@dataclass
class MigrationStep:
    """迁移步骤"""
    from_version: str
    to_version: str
    description: str
    migrate_func: Callable[[PersonalityConfig], PersonalityConfig]
    rollback_func: Optional[Callable[[PersonalityConfig], PersonalityConfig]] = None


@dataclass
class VersionHistory:
    """版本历史"""
    version: str
    changed_at: datetime
    changed_by: Optional[str] = None
    change_description: Optional[str] = None
    migration_path: Optional[str] = None


@dataclass
class CompatibilityResult:
    """兼容性检查结果"""
    is_compatible: bool
    compatibility_level: CompatibilityLevel
    breaking_changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    migration_needed: bool = False
    migration_steps: List[str] = field(default_factory=list)


class VersionManager:
    """
    版本管理器
    
    提供版本控制、迁移和兼容性检查功能。
    
    Attributes:
        version_registry: 版本注册表
        migration_registry: 迁移函数注册表
    """
    
    def __init__(self):
        """初始化版本管理器"""
        self.version_registry: Dict[str, PersonalityConfig] = {}
        self.migration_registry: Dict[str, List[MigrationStep]] = {}
        self.history: List[VersionHistory] = []
        
        # 注册内置迁移
        self._register_builtin_migrations()
    
    def _register_builtin_migrations(self) -> None:
        """注册内置迁移"""
        # 0.x.x -> 1.0.0 迁移
        self.register_migration(
            from_version="0.9.0",
            to_version="1.0.0",
            description="Migrate from pre-1.0 to stable 1.0.0",
            migrate_func=self._migrate_0x_to_1_0,
            rollback_func=None
        )
    
    def register_migration(
        self,
        from_version: str,
        to_version: str,
        description: str,
        migrate_func: Callable[[PersonalityConfig], PersonalityConfig],
        rollback_func: Optional[Callable[[PersonalityConfig], PersonalityConfig]] = None
    ) -> None:
        """
        注册迁移函数
        
        Args:
            from_version: 源版本
            to_version: 目标版本
            description: 迁移描述
            migrate_func: 迁移函数
            rollback_func: 回滚函数
        """
        key = f"{from_version}->{to_version}"
        
        if key not in self.migration_registry:
            self.migration_registry[key] = []
        
        step = MigrationStep(
            from_version=from_version,
            to_version=to_version,
            description=description,
            migrate_func=migrate_func,
            rollback_func=rollback_func
        )
        
        self.migration_registry[key].append(step)
        logger.debug(f"Registered migration: {key}")
    
    def parse_version(self, version_str: str) -> Version:
        """
        解析版本字符串
        
        Args:
            version_str: 版本字符串
            
        Returns:
            Version对象
        """
        return Version.parse(version_str)
    
    def check_compatibility(
        self,
        current_version: str,
        target_version: str
    ) -> CompatibilityResult:
        """
        检查版本兼容性
        
        Args:
            current_version: 当前版本
            target_version: 目标版本
            
        Returns:
            CompatibilityResult对象
        """
        current = Version.parse(current_version)
        target = Version.parse(target_version)
        
        breaking_changes = []
        warnings = []
        
        # 检查主版本号差异
        if current.major != target.major:
            # 主版本号不同 - 不兼容
            breaking_changes.append(
                f"Major version mismatch: {current.major} != {target.major}"
            )
            
            if current.major < target.major:
                return CompatibilityResult(
                    is_compatible=False,
                    compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                    breaking_changes=breaking_changes,
                    warnings=["Major version upgrade required"],
                    migration_needed=True,
                    migration_steps=self._get_migration_path(current_version, target_version)
                )
            else:
                return CompatibilityResult(
                    is_compatible=False,
                    compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                    breaking_changes=breaking_changes,
                    warnings=["Cannot downgrade to older major version"]
                )
        
        # 主版本号相同，检查次版本号
        if current.minor < target.minor:
            # 次版本号增加 - 向后兼容
            return CompatibilityResult(
                is_compatible=True,
                compatibility_level=CompatibilityLevel.BACKWARD,
                warnings=["Minor version upgrade - new features may be unavailable"]
            )
        elif current.minor > target.minor:
            # 次版本号减少 - 向前兼容
            warnings.append("Downgrading minor version")
            return CompatibilityResult(
                is_compatible=True,
                compatibility_level=CompatibilityLevel.FORWARD,
                warnings=warnings
            )
        
        # 补丁版本号检查
        if current.patch < target.patch:
            return CompatibilityResult(
                is_compatible=True,
                compatibility_level=CompatibilityLevel.FULL
            )
        elif current.patch > target.patch:
            warnings.append("Downgrading patch version")
            return CompatibilityResult(
                is_compatible=True,
                compatibility_level=CompatibilityLevel.FULL,
                warnings=warnings
            )
        
        # 完全相同
        return CompatibilityResult(
            is_compatible=True,
            compatibility_level=CompatibilityLevel.FULL
        )
    
    def _get_migration_path(
        self,
        from_version: str,
        to_version: str
    ) -> List[str]:
        """获取迁移路径"""
        path = []
        
        current = Version.parse(from_version)
        target = Version.parse(to_version)
        
        # 生成简单的版本路径
        while current < target:
            if current.minor == 0 and current.patch == 0:
                current = current.bump_major()
            elif current.patch == 0:
                current = current.bump_minor()
            else:
                current = current.bump_patch()
            
            path.append(str(current))
        
        return path
    
    def migrate(
        self,
        config: PersonalityConfig,
        from_version: str,
        to_version: str,
        strict: bool = True
    ) -> PersonalityConfig:
        """
        执行版本迁移
        
        Args:
            config: 待迁移的配置
            from_version: 源版本
            to_version: 目标版本
            strict: 是否严格模式
            
        Returns:
            迁移后的配置
            
        Raises:
            VersionError: 迁移失败
        """
        if from_version == to_version:
            return config
        
        # 检查兼容性
        compat = self.check_compatibility(from_version, to_version)
        
        if not compat.is_compatible and strict:
            raise VersionError(
                f"Cannot migrate from {from_version} to {to_version}: "
                f"{', '.join(compat.breaking_changes)}"
            )
        
        # 查找迁移函数
        key = f"{from_version}->{to_version}"
        
        if key in self.migration_registry:
            for step in self.migration_registry[key]:
                try:
                    config = step.migrate_func(config)
                    self._record_migration(from_version, to_version, step.description)
                    logger.info(f"Migrated {from_version} -> {to_version}: {step.description}")
                except Exception as e:
                    raise VersionError(f"Migration failed: {e}")
        else:
            # 尝试自动迁移
            config = self._auto_migrate(config, from_version, to_version)
        
        # 更新配置版本
        config.version = to_version
        
        return config
    
    def _auto_migrate(
        self,
        config: PersonalityConfig,
        from_version: str,
        to_version: str
    ) -> PersonalityConfig:
        """
        自动迁移配置
        
        Args:
            config: 配置
            from_version: 源版本
            to_version: 目标版本
            
        Returns:
            迁移后的配置
        """
        current = Version.parse(from_version)
        target = Version.parse(to_version)
        
        migrated = config
        
        # 按步骤迁移
        while current < target:
            next_version = self._get_next_version(current, target)
            
            key = f"{current}->{next_version}"
            if key in self.migration_registry:
                for step in self.migration_registry[key]:
                    migrated = step.migrate_func(migrated)
                    self._record_migration(str(current), str(next_version), step.description)
            else:
                # 使用通用迁移
                migrated = self._generic_migrate(migrated, current, next_version)
            
            current = next_version
        
        return migrated
    
    def _get_next_version(
        self,
        current: Version,
        target: Version
    ) -> Version:
        """获取下一步版本"""
        if current.major < target.major:
            return current.bump_major()
        elif current.minor < target.minor:
            return current.bump_minor()
        else:
            return current.bump_patch()
    
    def _generic_migrate(
        self,
        config: PersonalityConfig,
        from_ver: Version,
        to_ver: Version
    ) -> PersonalityConfig:
        """
        通用迁移逻辑
        
        Args:
            config: 配置
            from_ver: 源版本
            to_ver: 目标版本
            
        Returns:
            迁移后的配置
        """
        migrated = config
        
        # 1.0.0 之前的版本迁移到 1.0.0
        if from_ver.major == 0 and to_ver.major == 1:
            migrated = self._migrate_0x_to_1_0(migrated)
        
        return migrated
    
    def _migrate_0x_to_1_0(self, config: PersonalityConfig) -> PersonalityConfig:
        """
        0.x.x 到 1.0.0 的迁移
        
        Args:
            config: 原始配置
            
        Returns:
            迁移后的配置
        """
        migrated = PersonalityConfig(
            name=config.name,
            version="1.0.0",
            traits=config.traits or [],
            values=config.values or [],
            behaviors=config.behaviors or [],
            constraints=config.constraints or [],
            communication_style=config.communication_style or CommunicationStyle(
                tone=config.communication_style.tone if config.communication_style else CommunicationTone.PROFESSIONAL,
                length=config.communication_style.length if config.communication_style else ResponseLength.MODERATE
            ),
            domain_expertise=config.domain_expertise or [],
            metadata=config.metadata,
            compatibility={}
        )
        
        return migrated
    
    def _record_migration(
        self,
        from_version: str,
        to_version: str,
        description: str
    ) -> None:
        """记录迁移历史"""
        history = VersionHistory(
            version=to_version,
            changed_at=datetime.now(),
            change_description=description,
            migration_path=f"{from_version} -> {to_version}"
        )
        self.history.append(history)
    
    def rollback(
        self,
        config: PersonalityConfig,
        to_version: str
    ) -> PersonalityConfig:
        """
        回滚配置
        
        Args:
            config: 当前配置
            to_version: 目标版本
            
        Returns:
            回滚后的配置
            
        Raises:
            VersionError: 回滚失败
        """
        current_version = config.version
        
        # 查找回滚函数
        key = f"{current_version}->{to_version}"
        
        if key not in self.migration_registry:
            raise VersionError(f"No rollback path from {current_version} to {to_version}")
        
        for step in self.migration_registry[key]:
            if step.rollback_func:
                try:
                    config = step.rollback_func(config)
                    self._record_migration(current_version, to_version, f"Rollback: {step.description}")
                except Exception as e:
                    raise VersionError(f"Rollback failed: {e}")
            else:
                raise VersionError(f"No rollback function for {key}")
        
        return config
    
    def register_version(
        self,
        config: PersonalityConfig,
        description: Optional[str] = None
    ) -> None:
        """
        注册版本
        
        Args:
            config: 人格配置
            description: 版本描述
        """
        key = f"{config.name}:{config.version}"
        self.version_registry[key] = config
        
        history = VersionHistory(
            version=config.version,
            changed_at=datetime.now(),
            change_description=description
        )
        self.history.append(history)
        
        logger.info(f"Registered version: {key}")
    
    def get_version(
        self,
        name: str,
        version: str
    ) -> Optional[PersonalityConfig]:
        """
        获取指定版本
        
        Args:
            name: 人格名称
            version: 版本号
            
        Returns:
            配置或None
        """
        key = f"{name}:{version}"
        return self.version_registry.get(key)
    
    def get_latest_version(self, name: str) -> Optional[PersonalityConfig]:
        """
        获取最新版本
        
        Args:
            name: 人格名称
            
        Returns:
            最新配置或None
        """
        candidates = [
            (v, c) for k, c in self.version_registry.items()
            if k.startswith(f"{name}:")
            for v in [Version.parse(c.version)]
        ]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x[0])[1]
    
    def list_versions(self, name: str) -> List[str]:
        """
        列出所有版本
        
        Args:
            name: 人格名称
            
        Returns:
            版本列表
        """
        versions = []
        prefix = f"{name}:"
        
        for key in self.version_registry:
            if key.startswith(prefix):
                version = key[len(prefix):]
                versions.append(version)
        
        return sorted(versions, key=lambda v: Version.parse(v))
    
    def get_history(self, limit: Optional[int] = None) -> List[VersionHistory]:
        """
        获取版本历史
        
        Args:
            limit: 限制返回数量
            
        Returns:
            历史记录列表
        """
        if limit:
            return self.history[-limit:]
        return self.history
    
    def export_version_info(self) -> str:
        """
        导出版本信息
        
        Returns:
            JSON格式的版本信息
        """
        info = {
            "registered_versions": [
                {"key": k, "version": v.version}
                for k, v in self.version_registry.items()
            ],
            "migration_paths": list(self.migration_registry.keys()),
            "history": [
                {
                    "version": h.version,
                    "changed_at": h.changed_at.isoformat(),
                    "description": h.change_description
                }
                for h in self.history
            ]
        }
        
        return json.dumps(info, indent=2)


def create_version_manager() -> VersionManager:
    """
    工厂函数：创建版本管理器
    
    Returns:
        VersionManager实例
    """
    return VersionManager()
