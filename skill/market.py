#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 市场后端模块

本模块提供技能注册表和市场功能，包括技能的 CRUD 操作、版本管理、
评分评论、安装卸载、更新检查和搜索过滤等功能。

作者: AGI Framework Team
版本: 1.0.0
"""

from __future__ import annotations

import os
import re
import json
import shutil
import hashlib
import asyncio
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging

# 配置日志
logger = logging.getLogger(__name__)


class MarketError(Exception):
    """市场错误基类"""
    pass


class SkillNotFoundError(MarketError):
    """技能未找到错误"""
    pass


class SkillAlreadyExistsError(MarketError):
    """技能已存在错误"""
    pass


class VersionConflictError(MarketError):
    """版本冲突错误"""
    pass


class InstallationError(MarketError):
    """安装错误"""
    pass


class SearchError(MarketError):
    """搜索错误"""
    pass


class SkillStatus(Enum):
    """技能状态枚举"""
    ACTIVE = auto()       # 活跃
    DEPRECATED = auto()   # 已弃用
    SUSPENDED = auto()    # 已暂停
    PENDING = auto()      # 待审核
    REJECTED = auto()     # 已拒绝


@dataclass
class SkillReview:
    """
    技能评论
    
    属性:
        id: 评论 ID
        skill_id: 技能 ID
        user_id: 用户 ID
        rating: 评分（1-5）
        comment: 评论内容
        created_at: 创建时间
        updated_at: 更新时间
        helpful_count: 有用票数
    """
    id: str
    skill_id: str
    user_id: str
    rating: int
    comment: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    helpful_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'skill_id': self.skill_id,
            'user_id': self.user_id,
            'rating': self.rating,
            'comment': self.comment,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'helpful_count': self.helpful_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillReview:
        """从字典创建"""
        data = data.copy()
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class SkillVersion:
    """
    技能版本信息
    
    属性:
        version: 版本号
        release_date: 发布日期
        changelog: 变更日志
        download_url: 下载 URL
        checksum: 文件校验和
        min_platform_version: 最低平台版本要求
        deprecated: 是否已弃用
        deprecated_reason: 弃用原因
    """
    version: str
    release_date: datetime = field(default_factory=datetime.now)
    changelog: str = ""
    download_url: str = ""
    checksum: str = ""
    min_platform_version: str = "1.0.0"
    deprecated: bool = False
    deprecated_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': self.version,
            'release_date': self.release_date.isoformat(),
            'changelog': self.changelog,
            'download_url': self.download_url,
            'checksum': self.checksum,
            'min_platform_version': self.min_platform_version,
            'deprecated': self.deprecated,
            'deprecated_reason': self.deprecated_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillVersion:
        """从字典创建"""
        data = data.copy()
        if 'release_date' in data and isinstance(data['release_date'], str):
            data['release_date'] = datetime.fromisoformat(data['release_date'])
        return cls(**data)


@dataclass
class SkillEntry:
    """
    市场技能条目
    
    属性:
        id: 技能 ID
        name: 技能名称
        description: 技能描述
        author: 作者
        category: 分类
        tags: 标签列表
        icon: 图标 URL
        status: 状态
        versions: 版本列表
        reviews: 评论列表
        download_count: 下载次数
        created_at: 创建时间
        updated_at: 更新时间
        repository_url: 代码仓库 URL
        homepage: 主页 URL
        license: 许可证
    """
    id: str
    name: str
    description: str = ""
    author: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    icon: str = ""
    status: SkillStatus = SkillStatus.ACTIVE
    versions: List[SkillVersion] = field(default_factory=list)
    reviews: List[SkillReview] = field(default_factory=list)
    download_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    repository_url: str = ""
    homepage: str = ""
    license: str = "MIT"
    
    @property
    def latest_version(self) -> Optional[SkillVersion]:
        """获取最新版本"""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: self._version_key(v.version))
    
    @property
    def average_rating(self) -> float:
        """获取平均评分"""
        if not self.reviews:
            return 0.0
        return sum(r.rating for r in self.reviews) / len(self.reviews)
    
    @property
    def review_count(self) -> int:
        """获取评论数量"""
        return len(self.reviews)
    
    def _version_key(self, version: str) -> Tuple[int, ...]:
        """将版本字符串转换为排序键"""
        parts = version.split('.')
        return tuple(int(p) for p in parts if p.isdigit())
    
    def get_version(self, version: str) -> Optional[SkillVersion]:
        """获取指定版本"""
        for v in self.versions:
            if v.version == version:
                return v
        return None
    
    def has_version(self, version: str) -> bool:
        """检查是否有指定版本"""
        return any(v.version == version for v in self.versions)
    
    def add_version(self, version: SkillVersion) -> None:
        """添加版本"""
        if self.has_version(version.version):
            raise VersionConflictError(f"版本 {version.version} 已存在")
        self.versions.append(version)
        self.versions.sort(key=lambda v: self._version_key(v.version), reverse=True)
        self.updated_at = datetime.now()
    
    def add_review(self, review: SkillReview) -> None:
        """添加评论"""
        # 检查用户是否已评论
        existing = next((r for r in self.reviews if r.user_id == review.user_id), None)
        if existing:
            # 更新现有评论
            existing.rating = review.rating
            existing.comment = review.comment
            existing.updated_at = datetime.now()
        else:
            self.reviews.append(review)
        self.updated_at = datetime.now()
    
    def remove_review(self, review_id: str) -> bool:
        """删除评论"""
        for i, review in enumerate(self.reviews):
            if review.id == review_id:
                self.reviews.pop(i)
                self.updated_at = datetime.now()
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'author': self.author,
            'category': self.category,
            'tags': self.tags,
            'icon': self.icon,
            'status': self.status.name,
            'versions': [v.to_dict() for v in self.versions],
            'reviews': [r.to_dict() for r in self.reviews],
            'download_count': self.download_count,
            'average_rating': self.average_rating,
            'review_count': self.review_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'repository_url': self.repository_url,
            'homepage': self.homepage,
            'license': self.license,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillEntry:
        """从字典创建"""
        data = data.copy()
        
        # 转换状态
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = SkillStatus[data['status']]
        
        # 转换版本
        if 'versions' in data:
            data['versions'] = [SkillVersion.from_dict(v) for v in data['versions']]
        
        # 转换评论
        if 'reviews' in data:
            data['reviews'] = [SkillReview.from_dict(r) for r in data['reviews']]
        
        # 转换时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


@dataclass
class InstalledSkill:
    """
    已安装技能
    
    属性:
        skill_id: 技能 ID
        version: 安装版本
        install_path: 安装路径
        installed_at: 安装时间
        updated_at: 更新时间
        enabled: 是否启用
        config: 配置
    """
    skill_id: str
    version: str
    install_path: Path
    installed_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'skill_id': self.skill_id,
            'version': self.version,
            'install_path': str(self.install_path),
            'installed_at': self.installed_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'enabled': self.enabled,
            'config': self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InstalledSkill:
        """从字典创建"""
        data = data.copy()
        if 'install_path' in data and isinstance(data['install_path'], str):
            data['install_path'] = Path(data['install_path'])
        if 'installed_at' in data and isinstance(data['installed_at'], str):
            data['installed_at'] = datetime.fromisoformat(data['installed_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class SkillRegistry:
    """技能注册表"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        初始化注册表
        
        参数:
            storage_path: 存储路径
        """
        self.storage_path = storage_path or Path.home() / '.agi_skills' / 'registry.json'
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._skills: Dict[str, SkillEntry] = {}
        self._lock = threading.RLock()
        
        self._load()
    
    def _load(self) -> None:
        """从存储加载"""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding='utf-8'))
                for skill_data in data.get('skills', []):
                    skill = SkillEntry.from_dict(skill_data)
                    self._skills[skill.id] = skill
                logger.info(f"从 {self.storage_path} 加载了 {len(self._skills)} 个技能")
            except Exception as e:
                logger.error(f"加载注册表失败: {e}")
    
    def _save(self) -> None:
        """保存到存储"""
        with self._lock:
            data = {
                'skills': [skill.to_dict() for skill in self._skills.values()],
                'updated_at': datetime.now().isoformat(),
            }
            self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def register(self, skill: SkillEntry) -> None:
        """
        注册技能
        
        参数:
            skill: 技能条目
            
        抛出:
            SkillAlreadyExistsError: 技能已存在
        """
        with self._lock:
            if skill.id in self._skills:
                raise SkillAlreadyExistsError(f"技能 '{skill.id}' 已存在")
            
            self._skills[skill.id] = skill
            self._save()
            logger.info(f"注册技能: {skill.id}")
    
    def update(self, skill: SkillEntry) -> None:
        """
        更新技能
        
        参数:
            skill: 技能条目
        """
        with self._lock:
            self._skills[skill.id] = skill
            self._save()
            logger.info(f"更新技能: {skill.id}")
    
    def unregister(self, skill_id: str) -> bool:
        """
        注销技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否成功
        """
        with self._lock:
            if skill_id in self._skills:
                del self._skills[skill_id]
                self._save()
                logger.info(f"注销技能: {skill_id}")
                return True
            return False
    
    def get(self, skill_id: str) -> Optional[SkillEntry]:
        """
        获取技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            技能条目或 None
        """
        return self._skills.get(skill_id)
    
    def get_all(self) -> List[SkillEntry]:
        """
        获取所有技能
        
        返回:
            技能列表
        """
        return list(self._skills.values())
    
    def exists(self, skill_id: str) -> bool:
        """
        检查技能是否存在
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否存在
        """
        return skill_id in self._skills
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SkillEntry]:
        """
        搜索技能
        
        参数:
            query: 搜索关键词
            filters: 过滤条件
            
        返回:
            匹配的技能列表
        """
        results = []
        query_lower = query.lower()
        
        for skill in self._skills.values():
            # 关键词匹配
            match = (
                query_lower in skill.id.lower() or
                query_lower in skill.name.lower() or
                query_lower in skill.description.lower() or
                any(query_lower in tag.lower() for tag in skill.tags)
            )
            
            if not match:
                continue
            
            # 应用过滤器
            if filters:
                if 'category' in filters and skill.category != filters['category']:
                    continue
                if 'author' in filters and skill.author != filters['author']:
                    continue
                if 'status' in filters and skill.status != filters['status']:
                    continue
                if 'min_rating' in filters and skill.average_rating < filters['min_rating']:
                    continue
            
            results.append(skill)
        
        # 按相关性和评分排序
        results.sort(key=lambda s: (s.average_rating, s.download_count), reverse=True)
        
        return results
    
    def get_by_category(self, category: str) -> List[SkillEntry]:
        """
        按分类获取技能
        
        参数:
            category: 分类名称
            
        返回:
            技能列表
        """
        return [s for s in self._skills.values() if s.category == category]
    
    def get_by_author(self, author: str) -> List[SkillEntry]:
        """
        按作者获取技能
        
        参数:
            author: 作者名称
            
        返回:
            技能列表
        """
        return [s for s in self._skills.values() if s.author == author]
    
    def get_categories(self) -> List[str]:
        """
        获取所有分类
        
        返回:
            分类列表
        """
        categories = set(s.category for s in self._skills.values())
        return sorted(list(categories))
    
    def get_tags(self) -> List[str]:
        """
        获取所有标签
        
        返回:
            标签列表
        """
        tags = set()
        for skill in self._skills.values():
            tags.update(skill.tags)
        return sorted(list(tags))


class SkillMarket:
    """技能市场"""
    
    def __init__(self, registry: Optional[SkillRegistry] = None,
                 install_dir: Optional[Path] = None):
        """
        初始化市场
        
        参数:
            registry: 技能注册表
            install_dir: 安装目录
        """
        self.registry = registry or SkillRegistry()
        self.install_dir = install_dir or Path.home() / '.agi_skills' / 'installed'
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        self._installed: Dict[str, InstalledSkill] = {}
        self._load_installed()
    
    def _load_installed(self) -> None:
        """加载已安装技能"""
        manifest_path = self.install_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding='utf-8'))
                for skill_data in data.get('installed', []):
                    skill = InstalledSkill.from_dict(skill_data)
                    self._installed[skill.skill_id] = skill
                logger.info(f"加载了 {len(self._installed)} 个已安装技能")
            except Exception as e:
                logger.error(f"加载已安装技能失败: {e}")
    
    def _save_installed(self) -> None:
        """保存已安装技能"""
        manifest_path = self.install_dir / 'manifest.json'
        data = {
            'installed': [skill.to_dict() for skill in self._installed.values()],
            'updated_at': datetime.now().isoformat(),
        }
        manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    
    def _download_and_install(self, url: str, install_path: Path, 
                               skill_id: str, version: str) -> None:
        """
        从URL下载并安装技能
        
        参数:
            url: 下载URL
            install_path: 安装路径
            skill_id: 技能ID
            version: 版本号
        """
        import urllib.request
        import zipfile
        import io
        
        logger.info(f"正在下载技能: {skill_id}@{version} 从 {url}")
        
        try:
            # 下载文件
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            
            # 检查是否为ZIP文件
            if url.endswith('.zip') or data[:4] == b'PK\x03\x04':
                # 解压ZIP
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    zf.extractall(install_path)
                logger.info(f"解压完成: {install_path}")
            else:
                # 直接保存为SKILL.md
                skill_file = install_path / 'SKILL.md'
                skill_file.write_bytes(data)
                logger.info(f"保存完成: {skill_file}")
                
        except urllib.error.URLError as e:
            raise InstallationError(f"下载失败: {e}")
        except zipfile.BadZipFile as e:
            raise InstallationError(f"解压失败: {e}")
    
    def _local_install(self, skill: 'SkillEntry', version: str, 
                       install_path: Path) -> None:
        """
        本地安装（从registry复制）
        
        参数:
            skill: 技能条目
            version: 版本号
            install_path: 安装路径
        """
        # 创建默认的SKILL.md文件
        skill_content = f"""---
id: {skill.skill_id}
name: {skill.name}
version: {version}
description: {skill.description}
author: {skill.author}
tags: {skill.tags}
---

# {skill.name}

{skill.description}

## 使用方法

该技能已成功安装。请参考技能文档了解如何使用。
"""
        skill_file = install_path / 'SKILL.md'
        skill_file.write_text(skill_content, encoding='utf-8')
        logger.info(f"创建技能文件: {skill_file}")
    
    def install(self, skill_id: str, version: Optional[str] = None,
                force: bool = False) -> InstalledSkill:
        """
        安装技能
        
        参数:
            skill_id: 技能 ID
            version: 版本号（None 表示最新版本）
            force: 是否强制重新安装
            
        返回:
            已安装技能信息
            
        抛出:
            SkillNotFoundError: 技能未找到
            InstallationError: 安装失败
        """
        # 检查是否已安装
        if skill_id in self._installed and not force:
            installed = self._installed[skill_id]
            if version is None or installed.version == version:
                logger.info(f"技能 '{skill_id}' 已安装")
                return installed
        
        # 获取技能信息
        skill = self.registry.get(skill_id)
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未找到")
        
        # 确定版本
        if version is None:
            version_info = skill.latest_version
            if not version_info:
                raise InstallationError(f"技能 '{skill_id}' 没有可用版本")
            version = version_info.version
        else:
            version_info = skill.get_version(version)
            if not version_info:
                raise InstallationError(f"版本 '{version}' 不存在")
        
        # 创建安装目录
        install_path = self.install_dir / skill_id
        if install_path.exists():
            shutil.rmtree(install_path)
        install_path.mkdir(parents=True)
        
        try:
            # 下载并安装
            if version_info and version_info.download_url:
                # 从URL下载
                self._download_and_install(version_info.download_url, install_path, skill_id, version)
            else:
                # 本地安装（从registry复制文件）
                self._local_install(skill, version, install_path)
            
            # 创建已安装记录
            installed = InstalledSkill(
                skill_id=skill_id,
                version=version,
                install_path=install_path,
            )
            
            self._installed[skill_id] = installed
            self._save_installed()
            
            # 更新下载计数
            skill.download_count += 1
            self.registry.update(skill)
            
            logger.info(f"安装技能: {skill_id}@{version}")
            return installed
            
        except Exception as e:
            # 清理安装目录
            if install_path.exists():
                shutil.rmtree(install_path)
            raise InstallationError(f"安装失败: {e}")
    
    def uninstall(self, skill_id: str) -> bool:
        """
        卸载技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否成功
        """
        if skill_id not in self._installed:
            return False
        
        installed = self._installed[skill_id]
        
        # 删除安装目录
        if installed.install_path.exists():
            shutil.rmtree(installed.install_path)
        
        # 移除记录
        del self._installed[skill_id]
        self._save_installed()
        
        logger.info(f"卸载技能: {skill_id}")
        return True
    
    def update(self, skill_id: str) -> Optional[InstalledSkill]:
        """
        更新技能到最新版本
        
        参数:
            skill_id: 技能 ID
            
        返回:
            更新后的技能信息，如果没有更新则返回 None
        """
        if skill_id not in self._installed:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未安装")
        
        installed = self._installed[skill_id]
        skill = self.registry.get(skill_id)
        
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 在市场中未找到")
        
        latest = skill.latest_version
        if not latest or latest.version == installed.version:
            logger.info(f"技能 '{skill_id}' 已是最新版本")
            return None
        
        # 重新安装
        return self.install(skill_id, latest.version, force=True)
    
    def check_updates(self) -> List[Tuple[str, str, str]]:
        """
        检查可更新的技能
        
        返回:
            [(skill_id, current_version, latest_version), ...]
        """
        updates = []
        
        for skill_id, installed in self._installed.items():
            skill = self.registry.get(skill_id)
            if skill:
                latest = skill.latest_version
                if latest and latest.version != installed.version:
                    updates.append((skill_id, installed.version, latest.version))
        
        return updates
    
    def get_installed(self, skill_id: str) -> Optional[InstalledSkill]:
        """
        获取已安装技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            已安装技能信息或 None
        """
        return self._installed.get(skill_id)
    
    def get_all_installed(self) -> List[InstalledSkill]:
        """
        获取所有已安装技能
        
        返回:
            已安装技能列表
        """
        return list(self._installed.values())
    
    def is_installed(self, skill_id: str) -> bool:
        """
        检查技能是否已安装
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否已安装
        """
        return skill_id in self._installed
    
    def enable(self, skill_id: str) -> bool:
        """
        启用技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否成功
        """
        if skill_id not in self._installed:
            return False
        
        self._installed[skill_id].enabled = True
        self._installed[skill_id].updated_at = datetime.now()
        self._save_installed()
        return True
    
    def disable(self, skill_id: str) -> bool:
        """
        禁用技能
        
        参数:
            skill_id: 技能 ID
            
        返回:
            是否成功
        """
        if skill_id not in self._installed:
            return False
        
        self._installed[skill_id].enabled = False
        self._installed[skill_id].updated_at = datetime.now()
        self._save_installed()
        return True
    
    def add_review(self, skill_id: str, review: SkillReview) -> None:
        """
        添加评论
        
        参数:
            skill_id: 技能 ID
            review: 评论
        """
        skill = self.registry.get(skill_id)
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未找到")
        
        skill.add_review(review)
        self.registry.update(skill)
    
    def get_reviews(self, skill_id: str) -> List[SkillReview]:
        """
        获取评论列表
        
        参数:
            skill_id: 技能 ID
            
        返回:
            评论列表
        """
        skill = self.registry.get(skill_id)
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未找到")
        
        return skill.reviews


class SkillSearchEngine:
    """技能搜索引擎"""
    
    def __init__(self, registry: SkillRegistry):
        """
        初始化搜索引擎
        
        参数:
            registry: 技能注册表
        """
        self.registry = registry
        self._index: Dict[str, Set[str]] = defaultdict(set)
        self._build_index()
    
    def _build_index(self) -> None:
        """构建搜索索引"""
        for skill in self.registry.get_all():
            self._index_skill(skill)
    
    def _index_skill(self, skill: SkillEntry) -> None:
        """索引技能"""
        # 索引 ID
        self._index['id:' + skill.id.lower()].add(skill.id)
        
        # 索引名称
        for word in skill.name.lower().split():
            self._index['name:' + word].add(skill.id)
        
        # 索引描述
        for word in skill.description.lower().split():
            self._index['desc:' + word].add(skill.id)
        
        # 索引标签
        for tag in skill.tags:
            self._index['tag:' + tag.lower()].add(skill.id)
        
        # 索引分类
        self._index['cat:' + skill.category.lower()].add(skill.id)
        
        # 索引作者
        self._index['author:' + skill.author.lower()].add(skill.id)
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[SkillEntry]:
        """
        搜索技能
        
        参数:
            query: 搜索查询
            filters: 过滤条件
            
        返回:
            匹配的技能列表
        """
        query_lower = query.lower()
        
        # 收集匹配的 ID
        matched_ids: Set[str] = set()
        
        # 在索引中搜索
        for key, ids in self._index.items():
            if query_lower in key:
                matched_ids.update(ids)
        
        # 获取技能对象
        results = []
        for skill_id in matched_ids:
            skill = self.registry.get(skill_id)
            if skill:
                results.append(skill)
        
        # 应用过滤器
        if filters:
            results = self._apply_filters(results, filters)
        
        # 排序
        results.sort(key=lambda s: s.average_rating, reverse=True)
        
        return results
    
    def _apply_filters(self, skills: List[SkillEntry], 
                       filters: Dict[str, Any]) -> List[SkillEntry]:
        """应用过滤器"""
        results = skills
        
        if 'category' in filters:
            results = [s for s in results if s.category == filters['category']]
        
        if 'author' in filters:
            results = [s for s in results if s.author == filters['author']]
        
        if 'status' in filters:
            results = [s for s in results if s.status == filters['status']]
        
        if 'min_rating' in filters:
            results = [s for s in results if s.average_rating >= filters['min_rating']]
        
        if 'tags' in filters:
            required_tags = set(filters['tags'])
            results = [s for s in results if required_tags.issubset(set(s.tags))]
        
        return results
    
    def advanced_search(self, query: str, search_fields: Optional[List[str]] = None,
                        filters: Optional[Dict[str, Any]] = None,
                        sort_by: str = 'rating',
                        limit: int = 50) -> List[SkillEntry]:
        """
        高级搜索
        
        参数:
            query: 搜索查询
            search_fields: 搜索字段列表
            filters: 过滤条件
            sort_by: 排序方式
            limit: 返回数量限制
            
        返回:
            匹配的技能列表
        """
        all_skills = self.registry.get_all()
        results = []
        
        query_lower = query.lower()
        
        for skill in all_skills:
            score = 0
            
            # 计算匹配分数
            if 'id' in (search_fields or ['id']) and query_lower in skill.id.lower():
                score += 10
            
            if 'name' in (search_fields or ['name']) and query_lower in skill.name.lower():
                score += 8
            
            if 'description' in (search_fields or ['description']) and query_lower in skill.description.lower():
                score += 5
            
            if 'tags' in (search_fields or ['tags']):
                for tag in skill.tags:
                    if query_lower in tag.lower():
                        score += 3
            
            if score > 0:
                results.append((skill, score))
        
        # 应用过滤器
        if filters:
            filtered = []
            for skill, score in results:
                if self._matches_filters(skill, filters):
                    filtered.append((skill, score))
            results = filtered
        
        # 排序
        if sort_by == 'rating':
            results.sort(key=lambda x: (x[0].average_rating, x[1]), reverse=True)
        elif sort_by == 'downloads':
            results.sort(key=lambda x: (x[0].download_count, x[1]), reverse=True)
        elif sort_by == 'relevance':
            results.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == 'newest':
            results.sort(key=lambda x: x[0].created_at, reverse=True)
        
        # 限制数量
        results = results[:limit]
        
        return [skill for skill, _ in results]
    
    def _matches_filters(self, skill: SkillEntry, filters: Dict[str, Any]) -> bool:
        """检查技能是否匹配过滤器"""
        if 'category' in filters and skill.category != filters['category']:
            return False
        if 'author' in filters and skill.author != filters['author']:
            return False
        if 'status' in filters and skill.status != filters['status']:
            return False
        if 'min_rating' in filters and skill.average_rating < filters['min_rating']:
            return False
        if 'tags' in filters and not set(filters['tags']).issubset(set(skill.tags)):
            return False
        return True


class SkillVersionManager:
    """技能版本管理器"""
    
    def __init__(self, registry: SkillRegistry):
        """
        初始化版本管理器
        
        参数:
            registry: 技能注册表
        """
        self.registry = registry
    
    def compare_versions(self, v1: str, v2: str) -> int:
        """
        比较两个版本
        
        参数:
            v1: 版本 1
            v2: 版本 2
            
        返回:
            -1: v1 < v2
             0: v1 = v2
             1: v1 > v2
        """
        def parse_version(v: str) -> List[int]:
            parts = v.split('.')
            return [int(p) for p in parts if p.isdigit()]
        
        parts1 = parse_version(v1)
        parts2 = parse_version(v2)
        
        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            if p1 > p2:
                return 1
        
        if len(parts1) < len(parts2):
            return -1
        if len(parts1) > len(parts2):
            return 1
        
        return 0
    
    def is_compatible(self, required_version: str, actual_version: str,
                      constraint: str = ">=") -> bool:
        """
        检查版本兼容性
        
        参数:
            required_version: 要求的版本
            actual_version: 实际版本
            constraint: 约束条件
            
        返回:
            是否兼容
        """
        comparison = self.compare_versions(actual_version, required_version)
        
        if constraint == "=":
            return comparison == 0
        elif constraint == ">=":
            return comparison >= 0
        elif constraint == ">":
            return comparison > 0
        elif constraint == "<=":
            return comparison <= 0
        elif constraint == "<":
            return comparison < 0
        
        return False
    
    def get_changelog(self, skill_id: str, from_version: str,
                      to_version: str) -> List[str]:
        """
        获取版本变更日志
        
        参数:
            skill_id: 技能 ID
            from_version: 起始版本
            to_version: 目标版本
            
        返回:
            变更日志列表
        """
        skill = self.registry.get(skill_id)
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未找到")
        
        changelogs = []
        collecting = False
        
        for version in skill.versions:
            if version.version == from_version:
                collecting = True
            
            if collecting:
                changelogs.append(f"## {version.version}\n{version.changelog}")
            
            if version.version == to_version:
                break
        
        return changelogs
    
    def deprecate_version(self, skill_id: str, version: str,
                          reason: str = "") -> None:
        """
        弃用版本
        
        参数:
            skill_id: 技能 ID
            version: 版本号
            reason: 弃用原因
        """
        skill = self.registry.get(skill_id)
        if not skill:
            raise SkillNotFoundError(f"技能 '{skill_id}' 未找到")
        
        version_info = skill.get_version(version)
        if not version_info:
            raise VersionConflictError(f"版本 '{version}' 不存在")
        
        version_info.deprecated = True
        version_info.deprecated_reason = reason
        self.registry.update(skill)


# 便捷函数
def create_market(storage_path: Optional[Path] = None,
                  install_dir: Optional[Path] = None) -> SkillMarket:
    """
    创建技能市场实例
    
    参数:
        storage_path: 存储路径
        install_dir: 安装目录
        
    返回:
        技能市场实例
    """
    registry = SkillRegistry(storage_path)
    return SkillMarket(registry, install_dir)


def search_skills(query: str, **filters) -> List[SkillEntry]:
    """
    便捷函数：搜索技能
    
    参数:
        query: 搜索查询
        **filters: 过滤条件
        
    返回:
        匹配的技能列表
    """
    market = create_market()
    return market.registry.search(query, filters if filters else None)


def install_skill(skill_id: str, version: Optional[str] = None) -> InstalledSkill:
    """
    便捷函数：安装技能
    
    参数:
        skill_id: 技能 ID
        version: 版本号
        
    返回:
        已安装技能信息
    """
    market = create_market()
    return market.install(skill_id, version)


def uninstall_skill(skill_id: str) -> bool:
    """
    便捷函数：卸载技能
    
    参数:
        skill_id: 技能 ID
        
    返回:
        是否成功
    """
    market = create_market()
    return market.uninstall(skill_id)


# 单元测试存根
class TestSkillMarket:
    """SkillMarket 单元测试"""
    
    def test_register_skill(self) -> None:
        """测试注册技能"""
        registry = SkillRegistry()
        skill = SkillEntry(
            id="test-skill",
            name="Test Skill",
            description="A test skill",
            author="Test Author",
        )
        skill.add_version(SkillVersion(version="1.0.0"))
        
        registry.register(skill)
        
        retrieved = registry.get("test-skill")
        assert retrieved is not None
        assert retrieved.name == "Test Skill"
    
    def test_search_skills(self) -> None:
        """测试搜索技能"""
        registry = SkillRegistry()
        
        skill1 = SkillEntry(id="skill-1", name="Python Helper", tags=["python", "dev"])
        skill2 = SkillEntry(id="skill-2", name="JavaScript Tool", tags=["js", "dev"])
        
        registry.register(skill1)
        registry.register(skill2)
        
        results = registry.search("python")
        assert len(results) == 1
        assert results[0].id == "skill-1"
    
    def test_install_uninstall(self, tmp_path) -> None:
        """测试安装和卸载"""
        registry = SkillRegistry(storage_path=tmp_path / 'registry.json')
        market = SkillMarket(registry, install_dir=tmp_path / 'installed')
        
        skill = SkillEntry(id="test-skill", name="Test")
        skill.add_version(SkillVersion(version="1.0.0"))
        registry.register(skill)
        
        # 安装
        installed = market.install("test-skill")
        assert installed.skill_id == "test-skill"
        assert market.is_installed("test-skill")
        
        # 卸载
        assert market.uninstall("test-skill")
        assert not market.is_installed("test-skill")
    
    def test_version_comparison(self) -> None:
        """测试版本比较"""
        manager = SkillVersionManager(SkillRegistry())
        
        assert manager.compare_versions("1.0.0", "1.0.0") == 0
        assert manager.compare_versions("1.0.0", "1.0.1") == -1
        assert manager.compare_versions("1.1.0", "1.0.9") == 1
        assert manager.compare_versions("2.0.0", "1.9.9") == 1
    
    def test_reviews(self) -> None:
        """测试评论功能"""
        registry = SkillRegistry()
        skill = SkillEntry(id="test-skill", name="Test")
        registry.register(skill)
        
        review = SkillReview(
            id="review-1",
            skill_id="test-skill",
            user_id="user-1",
            rating=5,
            comment="Great skill!"
        )
        
        skill.add_review(review)
        
        assert skill.average_rating == 5.0
        assert skill.review_count == 1
    
    def test_skill_entry_dict(self) -> None:
        """测试技能条目序列化"""
        skill = SkillEntry(
            id="test",
            name="Test Skill",
            versions=[SkillVersion(version="1.0.0")],
        )
        
        data = skill.to_dict()
        restored = SkillEntry.from_dict(data)
        
        assert restored.id == skill.id
        assert restored.name == skill.name
        assert len(restored.versions) == 1


# 全局市场实例
_default_market: Optional[SkillMarket] = None


def get_default_market() -> SkillMarket:
    """获取默认市场实例"""
    global _default_market
    if _default_market is None:
        _default_market = create_market()
    return _default_market
