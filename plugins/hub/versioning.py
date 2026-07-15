"""
语义版本控制模块

提供SemVer解析、版本兼容性、升级路径和回滚支持功能。
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict
import copy


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class SemVer:
    """语义化版本
    
    遵循 SemVer 2.0.0 规范。
    """
    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: str = ""
    build: str = ""
    
    def __post_init__(self):
        # 确保整数类型
        self.major = int(self.major)
        self.minor = int(self.minor)
        self.patch = int(self.patch)
    
    @classmethod
    def parse(cls, version_str: str) -> "SemVer":
        """解析版本字符串
        
        Args:
            version_str: 版本字符串，如 "1.2.3-alpha+build123"
            
        Returns:
            SemVer对象
            
        Raises:
            ValueError: 解析失败
        """
        if not version_str:
            raise ValueError("Version string cannot be empty")
        
        # 移除前缀 'v' 或 'V'
        version_str = version_str.lstrip('vV')
        
        # 正则匹配 SemVer
        pattern = r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$'
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version_str}")
        
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        patch = int(match.group(3)) if match.group(3) else 0
        prerelease = match.group(4) or ""
        build = match.group(5) or ""
        
        return cls(major, minor, patch, prerelease, build)
    
    def __str__(self) -> str:
        """转换为字符串"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __eq__(self, other: object) -> bool:
        """相等比较（忽略build元数据）"""
        if not isinstance(other, SemVer):
            return False
        return (self.major, self.minor, self.patch, self.prerelease) == \
               (other.major, other.minor, other.patch, other.prerelease)
    
    def __lt__(self, other: "SemVer") -> bool:
        """小于比较"""
        # 比较主版本号
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # 预发布版本比较
        if self.prerelease and not other.prerelease:
            return True  # 预发布版本 < 正式版本
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0
        
        return False
    
    def __le__(self, other: "SemVer") -> bool:
        return self == other or self < other
    
    def __gt__(self, other: "SemVer") -> bool:
        return not self <= other
    
    def __ge__(self, other: "SemVer") -> bool:
        return not self < other
    
    def _compare_prerelease(self, pre1: str, pre2: str) -> int:
        """比较预发布版本标识符"""
        parts1 = pre1.split('.')
        parts2 = pre2.split('.')
        
        for p1, p2 in zip(parts1, parts2):
            # 数字标识符比较
            is_num1 = p1.isdigit()
            is_num2 = p2.isdigit()
            
            if is_num1 and is_num2:
                n1, n2 = int(p1), int(p2)
                if n1 != n2:
                    return -1 if n1 < n2 else 1
            elif is_num1:
                return -1  # 数字 < 非数字
            elif is_num2:
                return 1   # 非数字 > 数字
            else:
                if p1 != p2:
                    return -1 if p1 < p2 else 1
        
        # 长度不同
        if len(parts1) < len(parts2):
            return -1
        elif len(parts1) > len(parts2):
            return 1
        
        return 0
    
    def is_prerelease(self) -> bool:
        """是否为预发布版本"""
        return bool(self.prerelease)
    
    def is_stable(self) -> bool:
        """是否为稳定版本"""
        return self.major > 0 and not self.prerelease
    
    def bump_major(self) -> "SemVer":
        """增加主版本号"""
        return SemVer(self.major + 1, 0, 0)
    
    def bump_minor(self) -> "SemVer":
        """增加次版本号"""
        return SemVer(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> "SemVer":
        """增加修订版本号"""
        return SemVer(self.major, self.minor, self.patch + 1)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemVer":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VersionConstraint:
    """版本约束"""
    operator: str  # =, >, >=, <, <=, ~, ^, *
    version: SemVer
    
    @classmethod
    def parse(cls, constraint_str: str) -> List["VersionConstraint"]:
        """解析约束字符串
        
        支持格式:
        - "1.2.3" - 精确匹配
        - ">=1.2.3" - 大于等于
        - "^1.2.3" - 兼容版本（不改变主版本号）
        - "~1.2.3" - 近似版本（不改变次版本号）
        - ">=1.2.3,<2.0.0" - 范围
        - "1.x" 或 "1.X" 或 "1.*" - 通配符
        
        Args:
            constraint_str: 约束字符串
            
        Returns:
            约束列表
        """
        constraints = []
        
        # 分割多个约束
        parts = constraint_str.split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 检查操作符
            if part.startswith('>='):
                version = SemVer.parse(part[2:])
                constraints.append(cls('>=', version))
            elif part.startswith('>'):
                version = SemVer.parse(part[1:])
                constraints.append(cls('>', version))
            elif part.startswith('<='):
                version = SemVer.parse(part[2:])
                constraints.append(cls('<=', version))
            elif part.startswith('<'):
                version = SemVer.parse(part[1:])
                constraints.append(cls('<', version))
            elif part.startswith('^'):
                version = SemVer.parse(part[1:])
                constraints.append(cls('^', version))
            elif part.startswith('~'):
                version = SemVer.parse(part[1:])
                constraints.append(cls('~', version))
            elif part.startswith('='):
                version = SemVer.parse(part[1:])
                constraints.append(cls('=', version))
            elif 'x' in part.lower() or '*' in part:
                # 通配符处理
                constraints.append(cls('*', SemVer.parse(part.replace('x', '0').replace('X', '0').replace('*', '0'))))
            else:
                # 默认为精确匹配
                version = SemVer.parse(part)
                constraints.append(cls('=', version))
        
        return constraints
    
    def is_satisfied_by(self, version: SemVer) -> bool:
        """检查版本是否满足约束"""
        if self.operator == '=':
            return version == self.version
        elif self.operator == '>':
            return version > self.version
        elif self.operator == '>=':
            return version >= self.version
        elif self.operator == '<':
            return version < self.version
        elif self.operator == '<=':
            return version <= self.version
        elif self.operator == '^':
            # 兼容版本: ^1.2.3 表示 >=1.2.3 <2.0.0
            if version < self.version:
                return False
            if self.version.major == 0:
                # 0.x.x 特殊处理
                return version.major == 0 and version.minor == self.version.minor
            return version.major == self.version.major
        elif self.operator == '~':
            # 近似版本: ~1.2.3 表示 >=1.2.3 <1.3.0
            if version < self.version:
                return False
            return version.major == self.version.major and version.minor == self.version.minor
        elif self.operator == '*':
            return True
        
        return False
    
    def __str__(self) -> str:
        return f"{self.operator}{self.version}"


@dataclass
class VersionCompatibility:
    """版本兼容性报告"""
    is_compatible: bool
    breaking_changes: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    deprecations: List[str] = field(default_factory=list)
    migration_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpgradePath:
    """升级路径"""
    from_version: SemVer
    to_version: SemVer
    intermediate_versions: List[SemVer] = field(default_factory=list)
    estimated_risk: str = "low"  # low, medium, high
    estimated_time: str = ""  # e.g., "30 minutes"
    required_actions: List[str] = field(default_factory=list)
    can_rollback: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_version': str(self.from_version),
            'to_version': str(self.to_version),
            'intermediate_versions': [str(v) for v in self.intermediate_versions],
            'estimated_risk': self.estimated_risk,
            'estimated_time': self.estimated_time,
            'required_actions': self.required_actions,
            'can_rollback': self.can_rollback,
        }


# ---------------------------------------------------------------------------
# 版本兼容性检查器
# ---------------------------------------------------------------------------

class VersionCompatibility:
    """版本兼容性检查器"""
    
    def __init__(self):
        self._breaking_change_patterns = [
            r'BREAKING CHANGE',
            r'\[BREAKING\]',
            r'⚠️\s*BREAKING',
        ]
        self._feature_patterns = [
            r'feat\([^)]*\):',
            r'feature:',
            r'\[FEATURE\]',
        ]
        self._deprecation_patterns = [
            r'deprecat',
            r'\[DEPRECATED\]',
        ]
    
    def check_compatibility(self, current: SemVer, target: SemVer,
                            changelog: str = "") -> VersionCompatibility:
        """检查版本兼容性
        
        Args:
            current: 当前版本
            target: 目标版本
            changelog: 更新日志
            
        Returns:
            兼容性报告
        """
        breaking_changes = []
        new_features = []
        deprecations = []
        migration_notes = []
        
        # 根据版本号变化判断
        if target.major > current.major:
            # 主版本号变化，必然有不兼容变更
            breaking_changes.append(f"Major version change: {current.major} -> {target.major}")
        
        if target.major == 0 or current.major == 0:
            # 0.x.x 版本不稳定，任何变化都可能不兼容
            if target.minor > current.minor:
                breaking_changes.append("Minor version change in 0.x.x (unstable)")
            if target.patch > current.patch:
                breaking_changes.append("Patch version change in 0.x.x (unstable)")
        
        # 解析更新日志
        if changelog:
            breaking_changes.extend(self._extract_breaking_changes(changelog))
            new_features.extend(self._extract_features(changelog))
            deprecations.extend(self._extract_deprecations(changelog))
        
        is_compatible = len(breaking_changes) == 0
        
        # 生成迁移建议
        if breaking_changes:
            migration_notes.append("Review all breaking changes before upgrading")
            migration_notes.append("Test in staging environment first")
        
        if target.major > current.major:
            migration_notes.append(f"Major upgrade: check migration guide from v{current.major} to v{target.major}")
        
        return VersionCompatibility(
            is_compatible=is_compatible,
            breaking_changes=breaking_changes,
            new_features=new_features,
            deprecations=deprecations,
            migration_notes=migration_notes,
        )
    
    def _extract_breaking_changes(self, changelog: str) -> List[str]:
        """提取破坏性变更"""
        changes = []
        for pattern in self._breaking_change_patterns:
            matches = re.finditer(pattern, changelog, re.IGNORECASE)
            for match in matches:
                # 提取整行
                start = changelog.rfind('\n', 0, match.start()) + 1
                end = changelog.find('\n', match.end())
                if end == -1:
                    end = len(changelog)
                line = changelog[start:end].strip()
                if line and line not in changes:
                    changes.append(line)
        return changes
    
    def _extract_features(self, changelog: str) -> List[str]:
        """提取新特性"""
        features = []
        for pattern in self._feature_patterns:
            matches = re.finditer(pattern, changelog, re.IGNORECASE)
            for match in matches:
                start = changelog.rfind('\n', 0, match.start()) + 1
                end = changelog.find('\n', match.end())
                if end == -1:
                    end = len(changelog)
                line = changelog[start:end].strip()
                if line and line not in features:
                    features.append(line)
        return features
    
    def _extract_deprecations(self, changelog: str) -> List[str]:
        """提取弃用警告"""
        deprecations = []
        for pattern in self._deprecation_patterns:
            matches = re.finditer(pattern, changelog, re.IGNORECASE)
            for match in matches:
                start = changelog.rfind('\n', 0, match.start()) + 1
                end = changelog.find('\n', match.end())
                if end == -1:
                    end = len(changelog)
                line = changelog[start:end].strip()
                if line and line not in deprecations:
                    deprecations.append(line)
        return deprecations


# ---------------------------------------------------------------------------
# 版本解析器
# ---------------------------------------------------------------------------

class VersionResolver:
    """版本解析器
    
    解析版本约束，选择最佳匹配版本。
    """
    
    def __init__(self):
        self._version_cache: Dict[str, SemVer] = {}
    
    def resolve(self, constraint_str: str,
                available_versions: List[str]) -> Optional[SemVer]:
        """解析最佳版本
        
        Args:
            constraint_str: 版本约束字符串
            available_versions: 可用版本列表
            
        Returns:
            最佳匹配的版本，无匹配返回None
        """
        constraints = VersionConstraint.parse(constraint_str)
        
        # 解析所有可用版本
        versions = []
        for v_str in available_versions:
            try:
                v = SemVer.parse(v_str)
                versions.append(v)
            except ValueError:
                continue
        
        # 按版本排序（降序）
        versions.sort(reverse=True)
        
        # 找到满足所有约束的最新版本
        for version in versions:
            if all(c.is_satisfied_by(version) for c in constraints):
                return version
        
        return None
    
    def find_latest_compatible(self, current: SemVer,
                                available_versions: List[str]) -> Optional[SemVer]:
        """查找最新的兼容版本
        
        根据 SemVer，如果主版本号相同，则向后兼容。
        
        Args:
            current: 当前版本
            available_versions: 可用版本列表
            
        Returns:
            最新兼容版本
        """
        constraint = f"^{current}"
        return self.resolve(constraint, available_versions)
    
    def find_all_satisfying(self, constraint_str: str,
                            available_versions: List[str]) -> List[SemVer]:
        """查找所有满足约束的版本
        
        Args:
            constraint_str: 版本约束字符串
            available_versions: 可用版本列表
            
        Returns:
            满足约束的版本列表
        """
        constraints = VersionConstraint.parse(constraint_str)
        
        versions = []
        for v_str in available_versions:
            try:
                v = SemVer.parse(v_str)
                if all(c.is_satisfied_by(v) for c in constraints):
                    versions.append(v)
            except ValueError:
                continue
        
        versions.sort(reverse=True)
        return versions


# ---------------------------------------------------------------------------
# 升级路径规划器
# ---------------------------------------------------------------------------

class UpgradePathPlanner:
    """升级路径规划器"""
    
    def __init__(self):
        self._compatibility = VersionCompatibility()
    
    def plan_upgrade(self, current: SemVer, target: SemVer,
                     available_versions: List[str],
                     changelog_provider: Optional[Callable[[SemVer, SemVer], str]] = None) -> UpgradePath:
        """规划升级路径
        
        Args:
            current: 当前版本
            target: 目标版本
            available_versions: 可用版本列表
            changelog_provider: 更新日志提供函数
            
        Returns:
            升级路径
        """
        # 解析所有版本
        versions = []
        for v_str in available_versions:
            try:
                v = SemVer.parse(v_str)
                if current < v <= target:
                    versions.append(v)
            except ValueError:
                continue
        
        versions.sort()
        
        # 确定中间版本
        intermediates = []
        
        # 跨主版本升级时，建议经过每个主版本的最新版
        if target.major > current.major:
            for major in range(current.major + 1, target.major + 1):
                major_versions = [v for v in versions if v.major == major]
                if major_versions:
                    intermediates.append(max(major_versions))
        
        # 评估风险
        risk = self._assess_risk(current, target, versions)
        
        # 生成所需操作
        actions = []
        if target.major > current.major:
            actions.append("Review breaking changes in major version upgrade")
            actions.append("Update configuration files")
        
        if changelog_provider:
            for i in range(len(intermediates)):
                from_v = intermediates[i - 1] if i > 0 else current
                to_v = intermediates[i]
                changelog = changelog_provider(from_v, to_v)
                compat = self._compatibility.check_compatibility(from_v, to_v, changelog)
                if not compat.is_compatible:
                    actions.append(f"Address breaking changes in {from_v} -> {to_v}")
        
        can_rollback = target.major == current.major or len(intermediates) > 0
        
        return UpgradePath(
            from_version=current,
            to_version=target,
            intermediate_versions=intermediates,
            estimated_risk=risk,
            estimated_time=self._estimate_time(current, target, versions),
            required_actions=actions,
            can_rollback=can_rollback,
        )
    
    def _assess_risk(self, current: SemVer, target: SemVer,
                     versions: List[SemVer]) -> str:
        """评估升级风险"""
        if target.major > current.major:
            return "high"
        
        if target.minor > current.minor + 5:
            return "medium"
        
        if len(versions) > 10:
            return "medium"
        
        return "low"
    
    def _estimate_time(self, current: SemVer, target: SemVer,
                       versions: List[SemVer]) -> str:
        """估计升级时间"""
        version_diff = len(versions)
        
        if version_diff <= 1:
            return "5 minutes"
        elif version_diff <= 5:
            return "15 minutes"
        elif version_diff <= 10:
            return "30 minutes"
        elif target.major > current.major:
            return "1-2 hours"
        else:
            return "1 hour"


# ---------------------------------------------------------------------------
# 回滚管理器
# ---------------------------------------------------------------------------

class RollbackManager:
    """回滚管理器
    
    管理版本回滚操作。
    """
    
    def __init__(self, backup_dir: Optional[str] = None):
        """
        Args:
            backup_dir: 备份目录
        """
        self._backup_dir = backup_dir or os.path.join(
            os.path.expanduser("~"), ".clawhub", "backups"
        )
        self._backup_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        os.makedirs(self._backup_dir, exist_ok=True)
    
    def create_backup(self, plugin_id: str, version: SemVer,
                      plugin_path: str) -> str:
        """创建备份
        
        Args:
            plugin_id: 插件ID
            version: 版本
            plugin_path: 插件路径
            
        Returns:
            备份ID
        """
        import shutil
        
        backup_id = f"{plugin_id}_{version}_{int(datetime.now().timestamp())}"
        backup_path = os.path.join(self._backup_dir, backup_id)
        
        with self._lock:
            if os.path.isdir(plugin_path):
                shutil.copytree(plugin_path, backup_path)
            else:
                os.makedirs(backup_path, exist_ok=True)
                shutil.copy2(plugin_path, backup_path)
            
            self._backup_history.append({
                'backup_id': backup_id,
                'plugin_id': plugin_id,
                'version': str(version),
                'path': backup_path,
                'created_at': datetime.now().isoformat(),
            })
        
        return backup_id
    
    def rollback(self, backup_id: str, restore_path: str) -> bool:
        """回滚到备份
        
        Args:
            backup_id: 备份ID
            restore_path: 恢复路径
            
        Returns:
            是否成功
        """
        import shutil
        
        with self._lock:
            backup = None
            for b in self._backup_history:
                if b['backup_id'] == backup_id:
                    backup = b
                    break
            
            if not backup or not os.path.exists(backup['path']):
                return False
            
            try:
                # 删除当前版本
                if os.path.exists(restore_path):
                    if os.path.isdir(restore_path):
                        shutil.rmtree(restore_path)
                    else:
                        os.remove(restore_path)
                
                # 恢复备份
                if os.path.isdir(backup['path']):
                    shutil.copytree(backup['path'], restore_path)
                else:
                    shutil.copy2(backup['path'], restore_path)
                
                return True
            except Exception:
                return False
    
    def list_backups(self, plugin_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出备份"""
        with self._lock:
            if plugin_id:
                return [b for b in self._backup_history if b['plugin_id'] == plugin_id]
            return self._backup_history.copy()
    
    def delete_backup(self, backup_id: str) -> bool:
        """删除备份"""
        import shutil
        
        with self._lock:
            for i, backup in enumerate(self._backup_history):
                if backup['backup_id'] == backup_id:
                    try:
                        if os.path.exists(backup['path']):
                            shutil.rmtree(backup['path'])
                    except Exception:
                        pass
                    
                    self._backup_history.pop(i)
                    return True
            
            return False
    
    def cleanup_old_backups(self, max_age_days: int = 30) -> int:
        """清理旧备份
        
        Args:
            max_age_days: 最大保留天数
            
        Returns:
            删除的备份数量
        """
        import shutil
        
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        deleted = 0
        
        with self._lock:
            to_delete = []
            for backup in self._backup_history:
                created = datetime.fromisoformat(backup['created_at']).timestamp()
                if created < cutoff:
                    to_delete.append(backup)
            
            for backup in to_delete:
                try:
                    if os.path.exists(backup['path']):
                        shutil.rmtree(backup['path'])
                    self._backup_history.remove(backup)
                    deleted += 1
                except Exception:
                    pass
        
        return deleted


# ---------------------------------------------------------------------------
# 版本管理器
# ---------------------------------------------------------------------------

class VersionManager:
    """版本管理器
    
    整合所有版本管理功能的主类。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".clawhub", "versions"
        )
        
        self._resolver = VersionResolver()
        self._compatibility = VersionCompatibility()
        self._path_planner = UpgradePathPlanner()
        self._rollback = RollbackManager()
        
        self._version_history: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        
        os.makedirs(self._storage_path, exist_ok=True)
        self._load_from_disk()
    
    def parse_version(self, version_str: str) -> SemVer:
        """解析版本字符串"""
        return SemVer.parse(version_str)
    
    def check_constraint(self, version: SemVer, constraint_str: str) -> bool:
        """检查版本是否满足约束"""
        constraints = VersionConstraint.parse(constraint_str)
        return all(c.is_satisfied_by(version) for c in constraints)
    
    def resolve_version(self, constraint_str: str,
                        available_versions: List[str]) -> Optional[SemVer]:
        """解析最佳版本"""
        return self._resolver.resolve(constraint_str, available_versions)
    
    def check_compatibility(self, current: str, target: str,
                            changelog: str = "") -> VersionCompatibility:
        """检查版本兼容性"""
        current_v = SemVer.parse(current)
        target_v = SemVer.parse(target)
        return self._compatibility.check_compatibility(current_v, target_v, changelog)
    
    def plan_upgrade(self, current: str, target: str,
                     available_versions: List[str]) -> UpgradePath:
        """规划升级路径"""
        current_v = SemVer.parse(current)
        target_v = SemVer.parse(target)
        return self._path_planner.plan_upgrade(current_v, target_v, available_versions)
    
    def create_backup(self, plugin_id: str, version: str,
                      plugin_path: str) -> str:
        """创建备份"""
        version_v = SemVer.parse(version)
        return self._rollback.create_backup(plugin_id, version_v, plugin_path)
    
    def rollback(self, backup_id: str, restore_path: str) -> bool:
        """回滚到备份"""
        return self._rollback.rollback(backup_id, restore_path)
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """比较两个版本
        
        Returns:
            -1: version1 < version2
             0: version1 == version2
             1: version1 > version2
        """
        v1 = SemVer.parse(version1)
        v2 = SemVer.parse(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0
    
    def is_upgrade(self, current: str, target: str) -> bool:
        """检查是否为升级"""
        return self.compare_versions(current, target) < 0
    
    def is_downgrade(self, current: str, target: str) -> bool:
        """检查是否为降级"""
        return self.compare_versions(current, target) > 0
    
    def get_next_version(self, current: str, bump_type: str = "patch") -> str:
        """获取下一个版本号
        
        Args:
            current: 当前版本
            bump_type: 增加类型 (major, minor, patch)
            
        Returns:
            下一个版本号
        """
        v = SemVer.parse(current)
        
        if bump_type == "major":
            return str(v.bump_major())
        elif bump_type == "minor":
            return str(v.bump_minor())
        else:
            return str(v.bump_patch())
    
    def record_version_change(self, plugin_id: str, from_version: str,
                              to_version: str) -> None:
        """记录版本变更"""
        with self._lock:
            self._version_history[plugin_id].append({
                'from': from_version,
                'to': to_version,
                'timestamp': datetime.now().isoformat(),
            })
            self._save_to_disk()
    
    def get_version_history(self, plugin_id: str) -> List[Dict[str, Any]]:
        """获取版本历史"""
        with self._lock:
            return self._version_history.get(plugin_id, [])
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        try:
            file_path = os.path.join(self._storage_path, 'history.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self._version_history), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        file_path = os.path.join(self._storage_path, 'history.json')
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._version_history = defaultdict(list, data)
        except Exception:
            pass
