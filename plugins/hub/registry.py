"""
插件注册表模块

提供插件元数据管理、版本追踪、依赖图管理和搜索索引功能。
"""

import json
import os
import re
import time
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Iterator
from collections import defaultdict
import fnmatch


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class PluginAuthor:
    """插件作者信息"""
    name: str
    email: str = ""
    url: str = ""
    avatar: str = ""
    bio: str = ""
    verified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginAuthor":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginVersion:
    """插件版本信息"""
    version: str
    release_date: Optional[datetime] = None
    changelog: str = ""
    download_url: str = ""
    checksum: str = ""
    size: int = 0
    min_platform_version: str = ""
    max_platform_version: str = ""
    deprecated: bool = False
    deprecated_reason: str = ""
    
    def __post_init__(self):
        if isinstance(self.release_date, str):
            try:
                self.release_date = datetime.fromisoformat(self.release_date)
            except ValueError:
                self.release_date = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.release_date:
            result['release_date'] = self.release_date.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginStats:
    """插件统计信息"""
    downloads: int = 0
    installs: int = 0
    uninstalls: int = 0
    updates: int = 0
    views: int = 0
    last_download: Optional[datetime] = None
    last_install: Optional[datetime] = None
    daily_downloads: Dict[str, int] = field(default_factory=dict)
    weekly_downloads: Dict[str, int] = field(default_factory=dict)
    monthly_downloads: Dict[str, int] = field(default_factory=dict)
    
    def record_download(self) -> None:
        """记录下载"""
        self.downloads += 1
        self.last_download = datetime.now()
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_downloads[today] = self.daily_downloads.get(today, 0) + 1
        
        week = datetime.now().strftime("%Y-W%U")
        self.weekly_downloads[week] = self.weekly_downloads.get(week, 0) + 1
        
        month = datetime.now().strftime("%Y-%m")
        self.monthly_downloads[month] = self.monthly_downloads.get(month, 0) + 1
    
    def record_install(self) -> None:
        """记录安装"""
        self.installs += 1
        self.last_install = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.last_download:
            result['last_download'] = self.last_download.isoformat()
        if self.last_install:
            result['last_install'] = self.last_install.isoformat()
        return result


@dataclass
class PluginMetadata:
    """插件元数据"""
    # 基本信息
    plugin_id: str
    name: str
    version: str
    description: str = ""
    short_description: str = ""
    
    # 作者信息
    author: PluginAuthor = field(default_factory=lambda: PluginAuthor(""))
    
    # 分类和标签
    category: str = "general"
    subcategory: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # 版本信息
    versions: List[PluginVersion] = field(default_factory=list)
    latest_version: str = ""
    
    # 依赖
    dependencies: Dict[str, str] = field(default_factory=dict)
    optional_dependencies: Dict[str, str] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    
    # 平台要求
    min_platform_version: str = ""
    max_platform_version: str = ""
    supported_platforms: List[str] = field(default_factory=lambda: ["linux", "windows", "macos"])
    python_version: str = ">=3.8"
    
    # 链接
    homepage: str = ""
    repository: str = ""
    documentation: str = ""
    bug_tracker: str = ""
    
    # 许可证
    license: str = "MIT"
    license_url: str = ""
    
    # 媒体
    icon: str = ""
    screenshots: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    
    # 统计
    stats: PluginStats = field(default_factory=PluginStats)
    
    # 评分
    rating: float = 0.0
    rating_count: int = 0
    
    # 状态
    status: str = "active"  # active, deprecated, suspended, pending_review
    verified: bool = False
    featured: bool = False
    
    # 时间戳
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    
    # 额外信息
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.author, dict):
            self.author = PluginAuthor.from_dict(self.author)
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if isinstance(self.versions, list) and self.versions and isinstance(self.versions[0], dict):
            self.versions = [PluginVersion.from_dict(v) for v in self.versions]
        if isinstance(self.stats, dict):
            self.stats = PluginStats(**self.stats)
        
        for attr in ['created_at', 'updated_at', 'published_at']:
            val = getattr(self, attr)
            if isinstance(val, str):
                try:
                    setattr(self, attr, datetime.fromisoformat(val))
                except ValueError:
                    setattr(self, attr, None)
    
    @property
    def unique_id(self) -> str:
        """生成唯一标识符"""
        return f"{self.plugin_id}@{self.version}"
    
    def get_version_info(self, version: str) -> Optional[PluginVersion]:
        """获取特定版本信息"""
        for v in self.versions:
            if v.version == version:
                return v
        return None
    
    def add_version(self, version: PluginVersion) -> None:
        """添加版本"""
        # 检查是否已存在
        for i, v in enumerate(self.versions):
            if v.version == version.version:
                self.versions[i] = version
                return
        self.versions.append(version)
        self.versions.sort(key=lambda x: x.version, reverse=True)
        if not self.latest_version or version.version > self.latest_version:
            self.latest_version = version.version
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'short_description': self.short_description,
            'author': self.author.to_dict(),
            'category': self.category,
            'subcategory': self.subcategory,
            'tags': list(self.tags),
            'versions': [v.to_dict() for v in self.versions],
            'latest_version': self.latest_version,
            'dependencies': self.dependencies,
            'optional_dependencies': self.optional_dependencies,
            'conflicts': self.conflicts,
            'min_platform_version': self.min_platform_version,
            'max_platform_version': self.max_platform_version,
            'supported_platforms': self.supported_platforms,
            'python_version': self.python_version,
            'homepage': self.homepage,
            'repository': self.repository,
            'documentation': self.documentation,
            'bug_tracker': self.bug_tracker,
            'license': self.license,
            'license_url': self.license_url,
            'icon': self.icon,
            'screenshots': self.screenshots,
            'videos': self.videos,
            'stats': self.stats.to_dict(),
            'rating': self.rating,
            'rating_count': self.rating_count,
            'status': self.status,
            'verified': self.verified,
            'featured': self.featured,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'extra': self.extra,
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RegistrySearchResult:
    """注册表搜索结果"""
    plugins: List[PluginMetadata]
    total: int
    page: int
    page_size: int
    query: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    
    def has_next(self) -> bool:
        """是否有下一页"""
        return self.page * self.page_size < self.total
    
    def has_previous(self) -> bool:
        """是否有上一页"""
        return self.page > 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plugins': [p.to_dict() for p in self.plugins],
            'total': self.total,
            'page': self.page,
            'page_size': self.page_size,
            'query': self.query,
            'filters': self.filters,
            'facets': self.facets,
            'execution_time_ms': self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# 依赖图
# ---------------------------------------------------------------------------

class DependencyGraph:
    """依赖图管理器"""
    
    def __init__(self):
        self._nodes: Dict[str, Set[str]] = {}  # plugin_id -> set of dependency_ids
        self._reverse: Dict[str, Set[str]] = {}  # plugin_id -> set of dependent_ids
        self._metadata: Dict[str, PluginMetadata] = {}
        self._lock = threading.RLock()
    
    def add_plugin(self, metadata: PluginMetadata) -> None:
        """添加插件到依赖图"""
        with self._lock:
            plugin_id = metadata.plugin_id
            self._metadata[plugin_id] = metadata
            
            # 添加依赖关系
            deps = set(metadata.dependencies.keys())
            self._nodes[plugin_id] = deps
            
            # 更新反向依赖
            for dep_id in deps:
                if dep_id not in self._reverse:
                    self._reverse[dep_id] = set()
                self._reverse[dep_id].add(plugin_id)
    
    def remove_plugin(self, plugin_id: str) -> None:
        """从依赖图移除插件"""
        with self._lock:
            if plugin_id in self._nodes:
                # 从反向依赖中移除
                for dep_id in self._nodes[plugin_id]:
                    if dep_id in self._reverse:
                        self._reverse[dep_id].discard(plugin_id)
                del self._nodes[plugin_id]
            
            if plugin_id in self._reverse:
                del self._reverse[plugin_id]
            
            if plugin_id in self._metadata:
                del self._metadata[plugin_id]
    
    def get_dependencies(self, plugin_id: str, recursive: bool = False) -> Set[str]:
        """获取插件依赖"""
        with self._lock:
            if not recursive:
                return self._nodes.get(plugin_id, set()).copy()
            
            # 递归获取所有依赖
            visited = set()
            stack = list(self._nodes.get(plugin_id, set()))
            
            while stack:
                dep = stack.pop()
                if dep in visited:
                    continue
                visited.add(dep)
                stack.extend(self._nodes.get(dep, set()) - visited)
            
            return visited
    
    def get_dependents(self, plugin_id: str, recursive: bool = False) -> Set[str]:
        """获取依赖于该插件的插件"""
        with self._lock:
            if not recursive:
                return self._reverse.get(plugin_id, set()).copy()
            
            visited = set()
            stack = list(self._reverse.get(plugin_id, set()))
            
            while stack:
                dep = stack.pop()
                if dep in visited:
                    continue
                visited.add(dep)
                stack.extend(self._reverse.get(dep, set()) - visited)
            
            return visited
    
    def detect_cycles(self) -> List[List[str]]:
        """检测依赖循环"""
        with self._lock:
            cycles = []
            visited = set()
            rec_stack = set()
            
            def dfs(node: str, path: List[str]) -> None:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                
                for neighbor in self._nodes.get(node, set()):
                    if neighbor not in visited:
                        dfs(neighbor, path)
                    elif neighbor in rec_stack:
                        # 发现循环
                        cycle_start = path.index(neighbor)
                        cycles.append(path[cycle_start:] + [neighbor])
                
                path.pop()
                rec_stack.remove(node)
            
            for node in self._nodes:
                if node not in visited:
                    dfs(node, [])
            
            return cycles
    
    def get_installation_order(self, plugin_ids: List[str]) -> List[str]:
        """获取安装顺序（拓扑排序）"""
        with self._lock:
            # 构建子图
            subgraph = {}
            for pid in plugin_ids:
                subgraph[pid] = self._nodes.get(pid, set()) & set(plugin_ids)
            
            # Kahn算法
            in_degree = {pid: 0 for pid in plugin_ids}
            for deps in subgraph.values():
                for dep in deps:
                    if dep in in_degree:
                        in_degree[dep] += 1
            
            queue = [pid for pid, degree in in_degree.items() if degree == 0]
            result = []
            
            while queue:
                node = queue.pop(0)
                result.append(node)
                
                for neighbor in subgraph.get(node, set()):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            if len(result) != len(plugin_ids):
                raise ValueError("依赖图中存在循环，无法确定安装顺序")
            
            return result
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        with self._lock:
            return {
                'nodes': {k: list(v) for k, v in self._nodes.items()},
                'reverse': {k: list(v) for k, v in self._reverse.items()},
            }


# ---------------------------------------------------------------------------
# 搜索索引
# ---------------------------------------------------------------------------

class SearchIndex:
    """搜索索引管理器"""
    
    def __init__(self):
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)  # term -> plugin_ids
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> plugin_ids
        self._category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> plugin_ids
        self._author_index: Dict[str, Set[str]] = defaultdict(set)  # author -> plugin_ids
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._lock = threading.RLock()
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
        # 转换为小写，提取单词
        text = text.lower()
        words = re.findall(r'\b[a-z0-9]+\b', text)
        return words
    
    def index_plugin(self, metadata: PluginMetadata) -> None:
        """索引插件"""
        with self._lock:
            plugin_id = metadata.plugin_id
            self._metadata_cache[plugin_id] = metadata
            
            # 索引名称和描述
            text = f"{metadata.name} {metadata.description} {metadata.short_description}"
            for term in self._tokenize(text):
                self._inverted_index[term].add(plugin_id)
            
            # 索引标签
            for tag in metadata.tags:
                self._tag_index[tag.lower()].add(plugin_id)
            
            # 索引分类
            self._category_index[metadata.category.lower()].add(plugin_id)
            if metadata.subcategory:
                self._category_index[metadata.subcategory.lower()].add(plugin_id)
            
            # 索引作者
            self._author_index[metadata.author.name.lower()].add(plugin_id)
    
    def remove_plugin(self, plugin_id: str) -> None:
        """从索引中移除插件"""
        with self._lock:
            # 从倒排索引中移除
            for term, plugin_ids in list(self._inverted_index.items()):
                plugin_ids.discard(plugin_id)
                if not plugin_ids:
                    del self._inverted_index[term]
            
            # 从其他索引中移除
            for idx in [self._tag_index, self._category_index, self._author_index]:
                for key, plugin_ids in list(idx.items()):
                    plugin_ids.discard(plugin_id)
                    if not plugin_ids:
                        del idx[key]
            
            self._metadata_cache.pop(plugin_id, None)
    
    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "desc"
    ) -> RegistrySearchResult:
        """搜索插件"""
        with self._lock:
            start_time = time.time()
            filters = filters or {}
            
            # 解析查询
            terms = self._tokenize(query)
            if not terms:
                # 返回所有结果
                candidate_ids = set(self._metadata_cache.keys())
            else:
                # 获取每个词的候选
                candidate_sets = []
                for term in terms:
                    matching_ids = set()
                    # 精确匹配
                    if term in self._inverted_index:
                        matching_ids.update(self._inverted_index[term])
                    # 前缀匹配
                    for idx_term, ids in self._inverted_index.items():
                        if idx_term.startswith(term):
                            matching_ids.update(ids)
                    candidate_sets.append(matching_ids)
                
                # 交集
                if candidate_sets:
                    candidate_ids = set.intersection(*candidate_sets)
                else:
                    candidate_ids = set()
            
            # 应用过滤器
            if 'category' in filters:
                cat = filters['category'].lower()
                candidate_ids &= self._category_index.get(cat, set())
            
            if 'tags' in filters:
                tags = filters['tags']
                if isinstance(tags, str):
                    tags = [tags]
                for tag in tags:
                    candidate_ids &= self._tag_index.get(tag.lower(), set())
            
            if 'author' in filters:
                author = filters['author'].lower()
                candidate_ids &= self._author_index.get(author, set())
            
            if 'status' in filters:
                status = filters['status']
                candidate_ids = {
                    pid for pid in candidate_ids
                    if self._metadata_cache.get(pid, PluginMetadata("", "", "")).status == status
                }
            
            if 'verified' in filters:
                verified = filters['verified']
                candidate_ids = {
                    pid for pid in candidate_ids
                    if self._metadata_cache.get(pid, PluginMetadata("", "", "")).verified == verified
                }
            
            # 获取完整元数据
            results = [self._metadata_cache[pid] for pid in candidate_ids if pid in self._metadata_cache]
            
            # 排序
            if sort_by == "relevance":
                # 基于匹配词数计算相关性
                def relevance_score(p):
                    score = 0
                    p_text = f"{p.name} {p.description}".lower()
                    for term in terms:
                        score += p_text.count(term)
                    return score
                results.sort(key=relevance_score, reverse=(sort_order == "desc"))
            elif sort_by == "name":
                results.sort(key=lambda p: p.name, reverse=(sort_order == "desc"))
            elif sort_by == "rating":
                results.sort(key=lambda p: p.rating, reverse=(sort_order == "desc"))
            elif sort_by == "downloads":
                results.sort(key=lambda p: p.stats.downloads, reverse=(sort_order == "desc"))
            elif sort_by == "updated":
                results.sort(key=lambda p: p.updated_at or datetime.min, reverse=(sort_order == "desc"))
            
            # 分页
            total = len(results)
            start = (page - 1) * page_size
            end = start + page_size
            paginated = results[start:end]
            
            # 计算分面
            facets = self._compute_facets(candidate_ids)
            
            execution_time = (time.time() - start_time) * 1000
            
            return RegistrySearchResult(
                plugins=paginated,
                total=total,
                page=page,
                page_size=page_size,
                query=query,
                filters=filters,
                facets=facets,
                execution_time_ms=execution_time
            )
    
    def _compute_facets(self, plugin_ids: Set[str]) -> Dict[str, List[Tuple[str, int]]]:
        """计算分面统计"""
        facets = {
            'category': defaultdict(int),
            'tags': defaultdict(int),
            'license': defaultdict(int),
            'verified': defaultdict(int),
        }
        
        for pid in plugin_ids:
            if pid not in self._metadata_cache:
                continue
            meta = self._metadata_cache[pid]
            facets['category'][meta.category] += 1
            for tag in meta.tags:
                facets['tags'][tag] += 1
            facets['license'][meta.license] += 1
            facets['verified']['verified' if meta.verified else 'unverified'] += 1
        
        # 转换为排序后的列表
        result = {}
        for key, counts in facets.items():
            result[key] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return result
    
    def get_suggestions(self, prefix: str, limit: int = 10) -> List[str]:
        """获取搜索建议"""
        with self._lock:
            prefix = prefix.lower()
            suggestions = []
            
            # 从倒排索引中找匹配
            for term in self._inverted_index:
                if term.startswith(prefix) and term not in suggestions:
                    suggestions.append(term)
                    if len(suggestions) >= limit:
                        break
            
            # 从标签中找匹配
            for tag in self._tag_index:
                if tag.startswith(prefix) and tag not in suggestions:
                    suggestions.append(tag)
                    if len(suggestions) >= limit:
                        break
            
            return suggestions[:limit]


# ---------------------------------------------------------------------------
# 插件注册表
# ---------------------------------------------------------------------------

class PluginRegistry:
    """插件注册表
    
    管理插件元数据、版本追踪、依赖图和搜索索引。
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Args:
            storage_path: 存储路径
        """
        self._storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".clawhub", "registry"
        )
        self._plugins: Dict[str, PluginMetadata] = {}
        self._name_index: Dict[str, str] = {}  # name -> plugin_id
        self._dependency_graph = DependencyGraph()
        self._search_index = SearchIndex()
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, PluginMetadata], None]] = []
        
        os.makedirs(self._storage_path, exist_ok=True)
        self._load_from_disk()
    
    def add_listener(self, callback: Callable[[str, PluginMetadata], None]) -> None:
        """添加变更监听器"""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[str, PluginMetadata], None]) -> None:
        """移除变更监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify(self, event: str, metadata: PluginMetadata) -> None:
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(event, metadata)
            except Exception:
                pass
    
    def register(self, metadata: PluginMetadata) -> bool:
        """注册插件
        
        Args:
            metadata: 插件元数据
            
        Returns:
            是否成功
        """
        with self._lock:
            plugin_id = metadata.plugin_id
            
            # 检查名称冲突
            if metadata.name in self._name_index:
                existing_id = self._name_index[metadata.name]
                if existing_id != plugin_id:
                    return False
            
            # 更新时间戳
            now = datetime.now()
            if metadata.plugin_id not in self._plugins:
                metadata.created_at = metadata.created_at or now
            metadata.updated_at = now
            
            # 存储
            self._plugins[plugin_id] = metadata
            self._name_index[metadata.name] = plugin_id
            
            # 更新依赖图
            self._dependency_graph.add_plugin(metadata)
            
            # 更新搜索索引
            self._search_index.index_plugin(metadata)
            
            # 持久化
            self._save_to_disk()
            
            # 通知
            event = "updated" if metadata.created_at != now else "created"
            self._notify(event, metadata)
            
            return True
    
    def unregister(self, plugin_id: str) -> bool:
        """注销插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if plugin_id not in self._plugins:
                return False
            
            metadata = self._plugins[plugin_id]
            
            # 检查是否有其他插件依赖它
            dependents = self._dependency_graph.get_dependents(plugin_id)
            if dependents:
                return False
            
            # 移除
            del self._plugins[plugin_id]
            del self._name_index[metadata.name]
            
            self._dependency_graph.remove_plugin(plugin_id)
            self._search_index.remove_plugin(plugin_id)
            
            # 删除文件
            file_path = os.path.join(self._storage_path, f"{plugin_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            self._notify("deleted", metadata)
            
            return True
    
    def get(self, plugin_id: str) -> Optional[PluginMetadata]:
        """获取插件元数据"""
        with self._lock:
            return self._plugins.get(plugin_id)
    
    def get_by_name(self, name: str) -> Optional[PluginMetadata]:
        """通过名称获取插件"""
        with self._lock:
            plugin_id = self._name_index.get(name)
            if plugin_id:
                return self._plugins.get(plugin_id)
            return None
    
    def list_plugins(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        author: Optional[str] = None,
        status: Optional[str] = None,
        verified_only: bool = False,
        featured_only: bool = False
    ) -> List[PluginMetadata]:
        """列出插件"""
        with self._lock:
            results = list(self._plugins.values())
            
            if category:
                results = [p for p in results if p.category == category]
            
            if tag:
                results = [p for p in results if tag in p.tags]
            
            if author:
                results = [p for p in results if p.author.name == author]
            
            if status:
                results = [p for p in results if p.status == status]
            
            if verified_only:
                results = [p for p in results if p.verified]
            
            if featured_only:
                results = [p for p in results if p.featured]
            
            return results
    
    def search(
        self,
        query: str,
        **kwargs
    ) -> RegistrySearchResult:
        """搜索插件"""
        return self._search_index.search(query, **kwargs)
    
    def get_dependencies(self, plugin_id: str, recursive: bool = False) -> Set[str]:
        """获取插件依赖"""
        return self._dependency_graph.get_dependencies(plugin_id, recursive)
    
    def get_dependents(self, plugin_id: str, recursive: bool = False) -> Set[str]:
        """获取依赖该插件的插件"""
        return self._dependency_graph.get_dependents(plugin_id, recursive)
    
    def check_compatibility(
        self,
        plugin_id: str,
        platform_version: str,
        installed_plugins: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, List[str]]:
        """检查兼容性
        
        Returns:
            (是否兼容, 错误信息列表)
        """
        with self._lock:
            metadata = self._plugins.get(plugin_id)
            if not metadata:
                return False, ["Plugin not found"]
            
            errors = []
            
            # 检查平台版本
            if metadata.min_platform_version:
                if platform_version < metadata.min_platform_version:
                    errors.append(f"Requires platform version >= {metadata.min_platform_version}")
            
            if metadata.max_platform_version:
                if platform_version > metadata.max_platform_version:
                    errors.append(f"Requires platform version <= {metadata.max_platform_version}")
            
            # 检查依赖
            installed = installed_plugins or {}
            for dep_id, version_constraint in metadata.dependencies.items():
                if dep_id not in installed:
                    errors.append(f"Missing dependency: {dep_id}")
                elif version_constraint:
                    installed_version = installed[dep_id]
                    # 简化版本检查
                    if not self._check_version_constraint(installed_version, version_constraint):
                        errors.append(f"Dependency {dep_id} version mismatch: need {version_constraint}, have {installed_version}")
            
            # 检查冲突
            for conflict in metadata.conflicts:
                if conflict in installed:
                    errors.append(f"Conflicts with: {conflict}")
            
            return len(errors) == 0, errors
    
    def _check_version_constraint(self, version: str, constraint: str) -> bool:
        """检查版本约束"""
        # 简化实现
        if constraint.startswith(">="):
            return version >= constraint[2:]
        elif constraint.startswith(">"):
            return version > constraint[1:]
        elif constraint.startswith("<="):
            return version <= constraint[2:]
        elif constraint.startswith("<"):
            return version < constraint[1:]
        elif constraint.startswith("=="):
            return version == constraint[2:]
        return True
    
    def get_installation_order(self, plugin_ids: List[str]) -> List[str]:
        """获取安装顺序"""
        return self._dependency_graph.get_installation_order(plugin_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_downloads = sum(p.stats.downloads for p in self._plugins.values())
            total_installs = sum(p.stats.installs for p in self._plugins.values())
            
            categories = defaultdict(int)
            licenses = defaultdict(int)
            
            for p in self._plugins.values():
                categories[p.category] += 1
                licenses[p.license] += 1
            
            return {
                'total_plugins': len(self._plugins),
                'total_downloads': total_downloads,
                'total_installs': total_installs,
                'categories': dict(categories),
                'licenses': dict(licenses),
                'verified_count': sum(1 for p in self._plugins.values() if p.verified),
                'featured_count': sum(1 for p in self._plugins.values() if p.featured),
            }
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        for plugin_id, metadata in self._plugins.items():
            file_path = os.path.join(self._storage_path, f"{plugin_id}.json")
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        if not os.path.exists(self._storage_path):
            return
        
        for filename in os.listdir(self._storage_path):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(self._storage_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = PluginMetadata.from_dict(data)
                    self._plugins[metadata.plugin_id] = metadata
                    self._name_index[metadata.name] = metadata.plugin_id
                    self._dependency_graph.add_plugin(metadata)
                    self._search_index.index_plugin(metadata)
            except Exception:
                pass
    
    def export(self, file_path: str) -> bool:
        """导出注册表"""
        try:
            with self._lock:
                data = {
                    'version': '1.0',
                    'exported_at': datetime.now().isoformat(),
                    'plugins': [p.to_dict() for p in self._plugins.values()],
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
        except Exception:
            return False
    
    def import_registry(self, file_path: str, merge: bool = True) -> Tuple[int, List[str]]:
        """导入注册表
        
        Returns:
            (成功导入数量, 错误列表)
        """
        errors = []
        count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not merge:
                # 清空现有数据
                self._plugins.clear()
                self._name_index.clear()
            
            for plugin_data in data.get('plugins', []):
                try:
                    metadata = PluginMetadata.from_dict(plugin_data)
                    self.register(metadata)
                    count += 1
                except Exception as e:
                    errors.append(f"Failed to import {plugin_data.get('name', 'unknown')}: {e}")
            
            return count, errors
        except Exception as e:
            return 0, [str(e)]
