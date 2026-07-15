"""
版本兼容性检查
确保调用方与被调用方API版本一致
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class VersionCompatibility(Enum):
    """版本兼容性级别"""
    FULLY_COMPATIBLE = "fully_compatible"       # 完全兼容
    BACKWARD_COMPATIBLE = "backward_compatible" # 向后兼容
    FORWARD_COMPATIBLE = "forward_compatible"   # 向前兼容
    INCOMPATIBLE = "incompatible"               # 不兼容
    UNKNOWN = "unknown"                         # 未知


@dataclass(frozen=True)
class SemanticVersion:
    """
    语义化版本
    
    格式: MAJOR.MINOR.PATCH[-prerelease][+build]
    例如: 1.2.3, 2.0.0-beta, 1.0.0+build.123
    """
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: SemanticVersion) -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # 预发布版本小于正式版本
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self._compare_prerelease(self.prerelease, other.prerelease) < 0
        
        return False

    def __le__(self, other: SemanticVersion) -> bool:
        return self == other or self < other

    def __gt__(self, other: SemanticVersion) -> bool:
        return other < self

    def __ge__(self, other: SemanticVersion) -> bool:
        return self == other or self > other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    @classmethod
    def parse(cls, version_str: str) -> SemanticVersion:
        """
        解析版本字符串
        
        Args:
            version_str: 版本字符串
            
        Returns:
            SemanticVersion对象
            
        Raises:
            ValueError: 格式错误
        """
        # 移除前缀v或V
        version_str = version_str.lstrip('vV')
        
        # 分离build metadata
        build = None
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)
        
        # 分离prerelease
        prerelease = None
        if '-' in version_str:
            version_str, prerelease = version_str.split('-', 1)
        
        # 解析主版本号
        parts = version_str.split('.')
        if len(parts) < 3:
            # 补全版本号
            parts.extend(['0'] * (3 - len(parts)))
        
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version_str}") from e
        
        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build
        )

    def is_prerelease(self) -> bool:
        """是否为预发布版本"""
        return self.prerelease is not None

    def is_stable(self) -> bool:
        """是否为稳定版本"""
        return self.major >= 1 and self.prerelease is None

    def get_base_version(self) -> SemanticVersion:
        """获取基础版本（不含prerelease和build）"""
        return SemanticVersion(self.major, self.minor, self.patch)

    def bump_major(self) -> SemanticVersion:
        """增加主版本号"""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        """增加次版本号"""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        """增加补丁版本号"""
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    @staticmethod
    def _compare_prerelease(a: str, b: str) -> int:
        """比较预发布版本"""
        a_parts = a.split('.')
        b_parts = b.split('.')
        
        for i in range(max(len(a_parts), len(b_parts))):
            if i >= len(a_parts):
                return -1
            if i >= len(b_parts):
                return 1
            
            a_part = a_parts[i]
            b_part = b_parts[i]
            
            # 数字标识符比较
            try:
                a_num = int(a_part)
                b_num = int(b_part)
                if a_num != b_num:
                    return -1 if a_num < b_num else 1
            except ValueError:
                # 字符串比较
                if a_part != b_part:
                    return -1 if a_part < b_part else 1
        
        return 0


class VersionConstraint:
    """
    版本约束
    
    支持多种约束格式:
    - =1.2.3: 精确版本
    - >1.2.3: 大于指定版本
    - >=1.2.3: 大于等于指定版本
    - <1.2.3: 小于指定版本
    - <=1.2.3: 小于等于指定版本
    - ~1.2.3: 兼容版本（同major.minor）
    - ^1.2.3: 兼容版本（同major）
    - 1.2.x: 通配符
    - >=1.0.0,<2.0.0: 范围
    """

    def __init__(self, constraint_str: str):
        """
        初始化版本约束
        
        Args:
            constraint_str: 约束字符串
        """
        self._original = constraint_str
        self._constraints: List[Tuple[str, SemanticVersion]] = []
        self._parse(constraint_str)

    def _parse(self, constraint_str: str) -> None:
        """解析约束字符串"""
        # 处理逗号分隔的多个约束
        parts = [p.strip() for p in constraint_str.split(',')]
        
        for part in parts:
            if not part:
                continue
            
            # 识别操作符
            op = '='
            version_str = part
            
            for possible_op in ['>=', '<=', '>', '<', '~', '^', '=']:
                if part.startswith(possible_op):
                    op = possible_op
                    version_str = part[len(possible_op):].strip()
                    break
            
            # 处理通配符
            if 'x' in version_str.lower() or '*' in version_str:
                self._parse_wildcard(op, version_str)
            else:
                version = SemanticVersion.parse(version_str)
                self._constraints.append((op, version))

    def _parse_wildcard(self, op: str, version_str: str) -> None:
        """解析通配符版本"""
        version_str = version_str.lower().replace('x', '*')
        parts = version_str.split('.')
        
        # 确定通配符位置
        if parts[-1] == '*':
            # 1.2.* 格式
            base = '.'.join(parts[:-1])
            min_ver = SemanticVersion.parse(f"{base}.0")
            max_ver = min_ver.bump_minor()
            self._constraints.append(('>=', min_ver))
            self._constraints.append(('<', max_ver))

    def matches(self, version: Union[str, SemanticVersion]) -> bool:
        """
        检查版本是否满足约束
        
        Args:
            version: 要检查的版本
            
        Returns:
            是否满足约束
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        
        for op, constraint_version in self._constraints:
            if not self._check_constraint(version, op, constraint_version):
                return False
        
        return True

    def _check_constraint(
        self,
        version: SemanticVersion,
        op: str,
        constraint: SemanticVersion
    ) -> bool:
        """检查单个约束"""
        if op == '=':
            return version == constraint
        elif op == '>':
            return version > constraint
        elif op == '>=':
            return version >= constraint
        elif op == '<':
            return version < constraint
        elif op == '<=':
            return version <= constraint
        elif op == '~':
            # ~1.2.3 表示 >=1.2.3,<1.3.0
            return version >= constraint and version < constraint.bump_minor()
        elif op == '^':
            # ^1.2.3 表示 >=1.2.3,<2.0.0
            return version >= constraint and version < constraint.bump_major()
        
        return False

    def __str__(self) -> str:
        return self._original


class VersionCompatibilityChecker:
    """
    版本兼容性检查器
    
    检查调用方与被调用方的API版本兼容性
    """

    def __init__(self):
        self._api_versions: Dict[str, Dict[str, SemanticVersion]] = {}
        self._compatibility_matrix: Dict[str, Dict[str, VersionCompatibility]] = {}

    def register_api_version(
        self,
        api_name: str,
        version: Union[str, SemanticVersion],
        agent_id: Optional[str] = None
    ) -> None:
        """
        注册API版本
        
        Args:
            api_name: API名称
            version: 版本号
            agent_id: Agent ID（可选）
        """
        if isinstance(version, str):
            version = SemanticVersion.parse(version)
        
        if api_name not in self._api_versions:
            self._api_versions[api_name] = {}
        
        key = agent_id or "_default"
        self._api_versions[api_name][key] = version

    def check_compatibility(
        self,
        caller_version: Union[str, SemanticVersion],
        callee_version: Union[str, SemanticVersion]
    ) -> VersionCompatibility:
        """
        检查两个版本的兼容性
        
        Args:
            caller_version: 调用方版本
            callee_version: 被调用方版本
            
        Returns:
            兼容性级别
        """
        if isinstance(caller_version, str):
            caller_version = SemanticVersion.parse(caller_version)
        if isinstance(callee_version, str):
            callee_version = SemanticVersion.parse(callee_version)
        
        # 相同版本：完全兼容
        if caller_version == callee_version:
            return VersionCompatibility.FULLY_COMPATIBLE
        
        # 主版本不同：不兼容
        if caller_version.major != callee_version.major:
            return VersionCompatibility.INCOMPATIBLE
        
        # 主版本相同，检查次版本
        if caller_version.minor == callee_version.minor:
            # 只有patch不同：完全兼容
            return VersionCompatibility.FULLY_COMPATIBLE
        
        # 调用方版本较新
        if caller_version > callee_version:
            # 检查是否向后兼容
            if self._is_backward_compatible(callee_version, caller_version):
                return VersionCompatibility.BACKWARD_COMPATIBLE
            return VersionCompatibility.INCOMPATIBLE
        
        # 被调用方版本较新
        if self._is_forward_compatible(caller_version, callee_version):
            return VersionCompatibility.FORWARD_COMPATIBLE
        
        return VersionCompatibility.INCOMPATIBLE

    def _is_backward_compatible(
        self,
        old_version: SemanticVersion,
        new_version: SemanticVersion
    ) -> bool:
        """检查新版本是否向后兼容旧版本"""
        # 相同主版本，次版本增加通常向后兼容
        return (
            new_version.major == old_version.major and
            new_version.minor >= old_version.minor
        )

    def _is_forward_compatible(
        self,
        old_version: SemanticVersion,
        new_version: SemanticVersion
    ) -> bool:
        """检查旧版本是否向前兼容新版本"""
        # 相同主版本，次版本差异在1以内通常向前兼容
        return (
            new_version.major == old_version.major and
            new_version.minor - old_version.minor <= 1
        )

    def can_call(
        self,
        caller_version: Union[str, SemanticVersion],
        callee_version: Union[str, SemanticVersion]
    ) -> bool:
        """
        检查调用方是否可以调用被调用方
        
        Args:
            caller_version: 调用方版本
            callee_version: 被调用方版本
            
        Returns:
            是否可以调用
        """
        compatibility = self.check_compatibility(caller_version, callee_version)
        return compatibility in (
            VersionCompatibility.FULLY_COMPATIBLE,
            VersionCompatibility.BACKWARD_COMPATIBLE,
            VersionCompatibility.FORWARD_COMPATIBLE
        )

    def find_compatible_versions(
        self,
        target_version: Union[str, SemanticVersion],
        available_versions: List[Union[str, SemanticVersion]]
    ) -> List[Tuple[SemanticVersion, VersionCompatibility]]:
        """
        查找兼容的版本
        
        Args:
            target_version: 目标版本
            available_versions: 可用版本列表
            
        Returns:
            兼容版本及兼容性级别列表
        """
        if isinstance(target_version, str):
            target_version = SemanticVersion.parse(target_version)
        
        compatible = []
        for version in available_versions:
            if isinstance(version, str):
                version = SemanticVersion.parse(version)
            
            compat = self.check_compatibility(target_version, version)
            if compat != VersionCompatibility.INCOMPATIBLE:
                compatible.append((version, compat))
        
        # 按兼容性级别排序
        order = {
            VersionCompatibility.FULLY_COMPATIBLE: 0,
            VersionCompatibility.BACKWARD_COMPATIBLE: 1,
            VersionCompatibility.FORWARD_COMPATIBLE: 2
        }
        compatible.sort(key=lambda x: order.get(x[1], 3))
        
        return compatible

    def get_best_compatible_version(
        self,
        target_version: Union[str, SemanticVersion],
        available_versions: List[Union[str, SemanticVersion]]
    ) -> Optional[Tuple[SemanticVersion, VersionCompatibility]]:
        """
        获取最佳兼容版本
        
        Args:
            target_version: 目标版本
            available_versions: 可用版本列表
            
        Returns:
            最佳兼容版本及兼容性级别
        """
        compatible = self.find_compatible_versions(target_version, available_versions)
        return compatible[0] if compatible else None


class APIVersionNegotiator:
    """
    API版本协商器
    
    在调用方和被调用方之间协商共同支持的API版本
    """

    def __init__(self):
        self._supported_versions: Dict[str, Set[SemanticVersion]] = {}

    def register_supported_versions(
        self,
        agent_id: str,
        versions: List[Union[str, SemanticVersion]]
    ) -> None:
        """
        注册Agent支持的版本
        
        Args:
            agent_id: Agent ID
            versions: 支持的版本列表
        """
        parsed_versions = set()
        for v in versions:
            if isinstance(v, str):
                v = SemanticVersion.parse(v)
            parsed_versions.add(v)
        
        self._supported_versions[agent_id] = parsed_versions

    def negotiate_version(
        self,
        caller_id: str,
        callee_id: str
    ) -> Optional[SemanticVersion]:
        """
        协商双方共同支持的版本
        
        Args:
            caller_id: 调用方ID
            callee_id: 被调用方ID
            
        Returns:
            协商后的版本，无法协商则返回None
        """
        caller_versions = self._supported_versions.get(caller_id, set())
        callee_versions = self._supported_versions.get(callee_id, set())
        
        if not caller_versions or not callee_versions:
            return None
        
        # 找交集
        common = caller_versions & callee_versions
        if common:
            # 返回最高版本
            return max(common)
        
        # 没有共同版本，尝试找兼容版本
        checker = VersionCompatibilityChecker()
        
        for caller_ver in sorted(caller_versions, reverse=True):
            for callee_ver in sorted(callee_versions, reverse=True):
                if checker.can_call(caller_ver, callee_ver):
                    return callee_ver
        
        return None

    def negotiate_with_preference(
        self,
        caller_id: str,
        callee_id: str,
        preference: List[Union[str, SemanticVersion]]
    ) -> Optional[SemanticVersion]:
        """
        按偏好顺序协商版本
        
        Args:
            caller_id: 调用方ID
            callee_id: 被调用方ID
            preference: 偏好版本列表（按优先级排序）
            
        Returns:
            协商后的版本
        """
        callee_versions = self._supported_versions.get(callee_id, set())
        
        for pref in preference:
            if isinstance(pref, str):
                pref = SemanticVersion.parse(pref)
            
            if pref in callee_versions:
                return pref
            
            # 检查兼容性
            checker = VersionCompatibilityChecker()
            for callee_ver in callee_versions:
                if checker.can_call(pref, callee_ver):
                    return callee_ver
        
        # 回退到默认协商
        return self.negotiate_version(caller_id, callee_id)


def parse_version(version_str: str) -> SemanticVersion:
    """解析版本字符串"""
    return SemanticVersion.parse(version_str)


def check_version_range(
    version: Union[str, SemanticVersion],
    min_version: Optional[Union[str, SemanticVersion]] = None,
    max_version: Optional[Union[str, SemanticVersion]] = None
) -> bool:
    """
    检查版本是否在范围内
    
    Args:
        version: 要检查的版本
        min_version: 最小版本（包含）
        max_version: 最大版本（包含）
        
    Returns:
        是否在范围内
    """
    if isinstance(version, str):
        version = SemanticVersion.parse(version)
    
    if min_version:
        if isinstance(min_version, str):
            min_version = SemanticVersion.parse(min_version)
        if version < min_version:
            return False
    
    if max_version:
        if isinstance(max_version, str):
            max_version = SemanticVersion.parse(max_version)
        if version > max_version:
            return False
    
    return True
