"""
插件市场客户端模块

提供插件搜索、元数据获取、包下载和评分聚合功能。
仅使用 Python 标准库。
"""

import json
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 数据结构定义
# ---------------------------------------------------------------------------

@dataclass
class PluginRating:
    """插件评分"""
    average: float = 0.0
    count: int = 0
    distribution: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "average": self.average,
            "count": self.count,
            "distribution": self.distribution,
            "sources": self.sources,
        }


@dataclass
class PluginMetadata:
    """插件元数据"""
    plugin_id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    author_url: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    download_url: str = ""
    checksum: str = ""
    signature_url: str = ""
    categories: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    rating: Optional[PluginRating] = None
    downloads: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    size: int = 0
    python_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    requirements: Dict[str, str] = field(default_factory=dict)
    supported_platforms: List[str] = field(default_factory=list)
    min_agi_version: str = ""
    compatibility: Dict[str, str] = field(default_factory=dict)
    screenshots: List[str] = field(default_factory=list)
    changelog: str = ""
    readme: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "author_url": self.author_url,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "download_url": self.download_url,
            "checksum": self.checksum,
            "signature_url": self.signature_url,
            "categories": self.categories,
            "tags": sorted(self.tags),
            "rating": self.rating.to_dict() if self.rating else None,
            "downloads": self.downloads,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "size": self.size,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
            "requirements": self.requirements,
            "supported_platforms": self.supported_platforms,
            "min_agi_version": self.min_agi_version,
            "compatibility": self.compatibility,
            "screenshots": self.screenshots,
            "changelog": self.changelog,
            "readme": self.readme,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginMetadata":
        """从字典创建"""
        rating = None
        if data.get("rating"):
            rating = PluginRating(**data["rating"])

        def parse_date(d):
            if d is None:
                return None
            if isinstance(d, datetime):
                return d
            if isinstance(d, str):
                try:
                    return datetime.fromisoformat(d)
                except ValueError:
                    return None
            return None

        return cls(
            plugin_id=data.get("plugin_id", ""),
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            author_url=data.get("author_url", ""),
            license=data.get("license", ""),
            homepage=data.get("homepage", ""),
            repository=data.get("repository", ""),
            download_url=data.get("download_url", ""),
            checksum=data.get("checksum", ""),
            signature_url=data.get("signature_url", ""),
            categories=data.get("categories", []),
            tags=set(data.get("tags", [])),
            rating=rating,
            downloads=data.get("downloads", 0),
            created_at=parse_date(data.get("created_at")),
            updated_at=parse_date(data.get("updated_at")),
            published_at=parse_date(data.get("published_at")),
            size=data.get("size", 0),
            python_version=data.get("python_version", ""),
            dependencies=data.get("dependencies", []),
            requirements=data.get("requirements", {}),
            supported_platforms=data.get("supported_platforms", []),
            min_agi_version=data.get("min_agi_version", ""),
            compatibility=data.get("compatibility", {}),
            screenshots=data.get("screenshots", []),
            changelog=data.get("changelog", ""),
            readme=data.get("readme", ""),
            extra=data.get("extra", {}),
        )


@dataclass
class SearchResult:
    """搜索结果"""
    plugins: List[PluginMetadata]
    total: int
    page: int
    page_size: int
    query: str
    filters: Dict[str, Any]
    facets: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    related_queries: List[str] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plugins": [p.to_dict() for p in self.plugins],
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "query": self.query,
            "filters": self.filters,
            "facets": self.facets,
            "suggestions": self.suggestions,
            "related_queries": self.related_queries,
            "execution_time": self.execution_time,
        }

    def has_next(self) -> bool:
        """是否有下一页"""
        return self.page * self.page_size < self.total

    def has_previous(self) -> bool:
        """是否有上一页"""
        return self.page > 1


# ---------------------------------------------------------------------------
# 评分聚合器
# ---------------------------------------------------------------------------

class RatingAggregator:
    """评分聚合器

    从多个来源聚合插件评分，支持加权平均和置信度计算。
    """

    def __init__(self):
        self._source_weights: Dict[str, float] = {
            "official": 1.0,
            "github": 0.8,
            "npm": 0.7,
            "pypi": 0.6,
            "community": 0.5,
        }
        self._cache: Dict[str, PluginRating] = {}
        self._cache_ttl: float = 3600  # 1小时

    def set_source_weight(self, source: str, weight: float) -> None:
        """设置来源权重"""
        self._source_weights[source] = max(0.0, min(1.0, weight))

    def get_source_weight(self, source: str) -> float:
        """获取来源权重"""
        return self._source_weights.get(source, 0.5)

    def aggregate(self, plugin_id: str,
                  ratings: List[Dict[str, Any]]) -> PluginRating:
        """聚合多个来源的评分

        Args:
            plugin_id: 插件ID
            ratings: 评分列表 [{"source": str, "average": float,
                                 "count": int, "distribution": dict}]

        Returns:
            聚合后的评分
        """
        # 检查缓存
        if plugin_id in self._cache:
            return self._cache[plugin_id]

        if not ratings:
            return PluginRating()

        total_weight = 0.0
        weighted_sum = 0.0
        total_count = 0
        combined_distribution: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        sources: List[str] = []

        for rating in ratings:
            source = rating.get("source", "unknown")
            weight = self.get_source_weight(source)

            average = rating.get("average", 0.0)
            count = rating.get("count", 0)

            if count <= 0:
                continue

            weighted_sum += average * weight * count
            total_count += count
            total_weight += weight * count

            # 聚合分布
            distribution = rating.get("distribution", {})
            for star, cnt in distribution.items():
                star_int = int(star) if isinstance(star, str) else star
                if 1 <= star_int <= 5:
                    combined_distribution[star_int] += int(cnt)

            if source not in sources:
                sources.append(source)

        # 计算加权平均
        if total_weight > 0:
            average = weighted_sum / total_weight
        else:
            average = 0.0

        result = PluginRating(
            average=round(average, 2),
            count=total_count,
            distribution=combined_distribution,
            sources=sources,
        )

        # 缓存结果
        self._cache[plugin_id] = result

        return result

    def confidence(self, rating: PluginRating) -> float:
        """计算评分的置信度

        基于评分数量和来源多样性计算置信度。
        """
        # 基础置信度基于评分数量
        count_confidence = min(1.0, rating.count / 100)

        # 来源多样性加成
        diversity_bonus = min(0.3, len(rating.sources) * 0.1)

        # 分布均匀性加成
        distribution_values = list(rating.distribution.values())
        if sum(distribution_values) > 0:
            expected = sum(distribution_values) / 5
            variance = sum((v - expected) ** 2 for v in distribution_values) / 5
            max_variance = expected ** 2 * 4
            uniformity = 1.0 - (variance / max_variance) if max_variance > 0 else 0.0
        else:
            uniformity = 0.0

        return round(count_confidence + diversity_bonus + uniformity * 0.2, 3)

    def invalidate_cache(self, plugin_id: Optional[str] = None) -> None:
        """使缓存失效"""
        if plugin_id:
            self._cache.pop(plugin_id, None)
        else:
            self._cache.clear()


# ---------------------------------------------------------------------------
# 插件搜索器
# ---------------------------------------------------------------------------

class PluginSearch:
    """插件搜索器

    提供插件搜索功能，支持关键词搜索、过滤、排序和分页。
    """

    def __init__(self, marketplace_url: str, api_key: Optional[str] = None):
        """
        Args:
            marketplace_url: 市场API基础URL
            api_key: API密钥
        """
        self._base_url = marketplace_url.rstrip("/")
        self._api_key = api_key
        self._default_headers: Dict[str, str] = {}
        if api_key:
            self._default_headers["Authorization"] = f"Bearer {api_key}"

    def search(self, query: str,
              page: int = 1,
              page_size: int = 20,
              filters: Optional[Dict[str, Any]] = None,
              sort_by: str = "relevance",
              sort_order: str = "desc",
              categories: Optional[List[str]] = None,
              tags: Optional[List[str]] = None,
              min_rating: Optional[float] = None,
              min_downloads: Optional[int] = None,
              author: Optional[str] = None,
              version: Optional[str] = None,
              platform: Optional[str] = None) -> SearchResult:
        """搜索插件

        Args:
            query: 搜索关键词
            page: 页码
            page_size: 每页数量
            filters: 其他过滤器
            sort_by: 排序字段 (relevance, downloads, rating, updated, name)
            sort_order: 排序方向 (asc, desc)
            categories: 分类过滤
            tags: 标签过滤
            min_rating: 最低评分
            min_downloads: 最低下载量
            author: 作者过滤
            version: 版本过滤
            platform: 平台过滤

        Returns:
            搜索结果
        """
        start_time = time.time()

        # 构建查询参数
        params: Dict[str, Any] = {
            "q": query,
            "page": page,
            "size": page_size,
            "sort": sort_by,
            "order": sort_order,
        }

        # 添加过滤器
        if categories:
            params["categories"] = ",".join(categories)
        if tags:
            params["tags"] = ",".join(tags)
        if min_rating is not None:
            params["min_rating"] = min_rating
        if min_downloads is not None:
            params["min_downloads"] = min_downloads
        if author:
            params["author"] = author
        if version:
            params["version"] = version
        if platform:
            params["platform"] = platform
        if filters:
            params.update(filters)

        # 发送请求
        url = f"{self._base_url}/api/v1/plugins/search"
        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"

        try:
            response_data = self._fetch_json(full_url)
            return self._parse_search_response(response_data, query, params, start_time)
        except Exception as e:
            # 返回空结果
            return SearchResult(
                plugins=[],
                total=0,
                page=page,
                page_size=page_size,
                query=query,
                filters=params,
                execution_time=time.time() - start_time,
            )

    def _fetch_json(self, url: str) -> dict:
        """获取JSON数据"""
        request = urllib.request.Request(url, headers=self._default_headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_search_response(self, data: dict, query: str,
                               params: Dict[str, Any],
                               start_time: float) -> SearchResult:
        """解析搜索响应"""
        plugins = []
        for item in data.get("results", []):
            try:
                plugins.append(PluginMetadata.from_dict(item))
            except Exception:
                continue

        # 提取分面
        facets = {}
        for facet_name, facet_data in data.get("facets", {}).items():
            facets[facet_name] = [
                (item["key"], item["count"]) for item in facet_data
            ]

        return SearchResult(
            plugins=plugins,
            total=data.get("total", 0),
            page=data.get("page", params.get("page", 1)),
            page_size=data.get("page_size", params.get("size", 20)),
            query=query,
            filters=params,
            facets=facets,
            suggestions=data.get("suggestions", []),
            related_queries=data.get("related_queries", []),
            execution_time=time.time() - start_time,
        )

    def get_categories(self) -> List[Dict[str, Any]]:
        """获取所有分类"""
        url = f"{self._base_url}/api/v1/categories"
        try:
            return self._fetch_json(url)
        except Exception:
            return []

    def get_tags(self, prefix: Optional[str] = None) -> List[str]:
        """获取标签列表"""
        url = f"{self._base_url}/api/v1/tags"
        if prefix:
            url = f"{url}?prefix={urllib.parse.quote(prefix)}"

        try:
            data = self._fetch_json(url)
            return data.get("tags", [])
        except Exception:
            return []


# ---------------------------------------------------------------------------
# 包下载器（市场客户端版）
# ---------------------------------------------------------------------------

class PackageDownloader:
    """市场包下载器

    从市场下载插件包。
    """

    def __init__(self, timeout: int = 300,
                 max_retries: int = 3,
                 chunk_size: int = 8192):
        """
        Args:
            timeout: 下载超时（秒）
            max_retries: 最大重试次数
            chunk_size: 每次读取的块大小
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._chunk_size = chunk_size
        self._download_cache: Dict[str, str] = {}
        self._download_dir: str = os.path.join(os.getcwd(), ".plugin_cache")
        os.makedirs(self._download_dir, exist_ok=True)

    def set_download_dir(self, directory: str) -> None:
        """设置下载目录"""
        self._download_dir = directory
        os.makedirs(self._download_dir, exist_ok=True)

    def get_download_dir(self) -> str:
        """获取下载目录"""
        return self._download_dir

    def download(self, url: str,
                filename: Optional[str] = None,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> Tuple[bool, str]:
        """下载包

        Args:
            url: 下载URL
            filename: 文件名（可选）
            progress_callback: 进度回调

        Returns:
            (是否成功, 文件路径或错误信息)
        """
        # 确定文件名
        if not filename:
            parsed = urllib.parse.urlparse(url)
            filename = os.path.basename(parsed.path) or "plugin.zip"

        dest_path = os.path.join(self._download_dir, filename)

        # 检查缓存
        if os.path.exists(dest_path):
            return True, dest_path

        # 下载
        for attempt in range(self._max_retries):
            try:
                success, result = self._download_file(url, dest_path, progress_callback)
                if success:
                    self._download_cache[url] = dest_path
                    return True, result
            except Exception as e:
                if attempt == self._max_retries - 1:
                    return False, str(e)

        return False, f"达到最大重试次数 ({self._max_retries})"

    def _download_file(self, url: str, dest_path: str,
                      progress_callback: Optional[Callable[[int, int], None]]) -> Tuple[bool, str]:
        """下载文件"""
        request = urllib.request.Request(url)

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                # 确保目录存在
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                with open(dest_path, "wb") as f:
                    while True:
                        chunk = response.read(self._chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)

                return True, dest_path

        except urllib.error.HTTPError as e:
            return False, f"HTTP错误: {e.code} {e.reason}"
        except urllib.error.URLError as e:
            return False, f"URL错误: {str(e.reason)}"

    def get_cached_path(self, url: str) -> Optional[str]:
        """获取缓存路径"""
        return self._download_cache.get(url)

    def clear_cache(self) -> int:
        """清除缓存"""
        count = 0
        for path in self._download_cache.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
                    count += 1
            except OSError:
                pass

        self._download_cache.clear()
        return count


# ---------------------------------------------------------------------------
# 市场客户端
# ---------------------------------------------------------------------------

class MarketplaceClient:
    """插件市场客户端

    提供与插件市场交互的完整功能，包括搜索、获取元数据、下载等。
    """

    DEFAULT_MARKETPLACE_URL = "https://marketplace.agi-unified.example.com"

    def __init__(self,
                 marketplace_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 cache_dir: Optional[str] = None):
        """
        Args:
            marketplace_url: 市场API基础URL
            api_key: API密钥
            timeout: 请求超时
            cache_dir: 缓存目录
        """
        self._base_url = (marketplace_url or self.DEFAULT_MARKETPLACE_URL).rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}

        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._search = PluginSearch(self._base_url, api_key)
        self._downloader = PackageDownloader(timeout=300)

        if cache_dir:
            self._downloader.set_download_dir(cache_dir)

        # 缓存
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._cache_ttl: float = 3600  # 1小时

    def get_base_url(self) -> str:
        """获取基础URL"""
        return self._base_url

    def set_api_key(self, api_key: str) -> None:
        """设置API密钥"""
        self._api_key = api_key
        self._headers["Authorization"] = f"Bearer {api_key}"

    def search(self, query: str, **kwargs) -> SearchResult:
        """搜索插件

        Args:
            query: 搜索关键词
            **kwargs: 传递给搜索的参数

        Returns:
            搜索结果
        """
        return self._search.search(query, **kwargs)

    def get_plugin(self, plugin_id: str, use_cache: bool = True) -> Optional[PluginMetadata]:
        """获取插件元数据

        Args:
            plugin_id: 插件ID
            use_cache: 是否使用缓存

        Returns:
            插件元数据，失败返回None
        """
        if use_cache and plugin_id in self._metadata_cache:
            return self._metadata_cache[plugin_id]

        url = f"{self._base_url}/api/v1/plugins/{plugin_id}"
        try:
            data = self._fetch_json(url)
            metadata = PluginMetadata.from_dict(data)
            self._metadata_cache[plugin_id] = metadata
            return metadata
        except Exception:
            return None

    def get_plugin_by_name(self, name: str) -> Optional[PluginMetadata]:
        """通过名称获取插件

        Args:
            name: 插件名称

        Returns:
            插件元数据
        """
        # 先搜索
        result = self.search(name, page_size=1)
        for plugin in result.plugins:
            if plugin.name.lower() == name.lower():
                return plugin
        return None

    def get_featured_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """获取精选插件

        Args:
            limit: 返回数量

        Returns:
            插件列表
        """
        url = f"{self._base_url}/api/v1/plugins/featured?limit={limit}"
        try:
            data = self._fetch_json(url)
            plugins = []
            for item in data.get("plugins", []):
                try:
                    plugins.append(PluginMetadata.from_dict(item))
                except Exception:
                    continue
            return plugins
        except Exception:
            return []

    def get_trending_plugins(self, limit: int = 10,
                            time_range: str = "week") -> List[PluginMetadata]:
        """获取热门插件

        Args:
            limit: 返回数量
            time_range: 时间范围 (day, week, month)

        Returns:
            插件列表
        """
        url = f"{self._base_url}/api/v1/plugins/trending?limit={limit}&range={time_range}"
        try:
            data = self._fetch_json(url)
            plugins = []
            for item in data.get("plugins", []):
                try:
                    plugins.append(PluginMetadata.from_dict(item))
                except Exception:
                    continue
            return plugins
        except Exception:
            return []

    def get_new_plugins(self, limit: int = 10) -> List[PluginMetadata]:
        """获取最新插件

        Args:
            limit: 返回数量

        Returns:
            插件列表
        """
        url = f"{self._base_url}/api/v1/plugins/new?limit={limit}"
        try:
            data = self._fetch_json(url)
            plugins = []
            for item in data.get("plugins", []):
                try:
                    plugins.append(PluginMetadata.from_dict(item))
                except Exception:
                    continue
            return plugins
        except Exception:
            return []

    def get_recommended_plugins(self,
                               user_id: Optional[str] = None,
                               limit: int = 10) -> List[PluginMetadata]:
        """获取推荐插件

        Args:
            user_id: 用户ID
            limit: 返回数量

        Returns:
            插件列表
        """
        url = f"{self._base_url}/api/v1/plugins/recommended?limit={limit}"
        if user_id:
            url += f"&user_id={user_id}"

        try:
            data = self._fetch_json(url)
            plugins = []
            for item in data.get("plugins", []):
                try:
                    plugins.append(PluginMetadata.from_dict(item))
                except Exception:
                    continue
            return plugins
        except Exception:
            return []

    def get_plugin_versions(self, plugin_id: str) -> List[Dict[str, Any]]:
        """获取插件的所有版本

        Args:
            plugin_id: 插件ID

        Returns:
            版本列表
        """
        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/versions"
        try:
            data = self._fetch_json(url)
            return data.get("versions", [])
        except Exception:
            return []

    def get_plugin_changelog(self, plugin_id: str,
                            version: Optional[str] = None) -> str:
        """获取插件更新日志

        Args:
            plugin_id: 插件ID
            version: 版本号（可选）

        Returns:
            更新日志
        """
        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/changelog"
        if version:
            url += f"?version={version}"

        try:
            data = self._fetch_json(url)
            return data.get("changelog", "")
        except Exception:
            return ""

    def get_plugin_readme(self, plugin_id: str) -> str:
        """获取插件README

        Args:
            plugin_id: 插件ID

        Returns:
            README内容
        """
        plugin = self.get_plugin(plugin_id)
        if plugin and plugin.readme:
            return plugin.readme

        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/readme"
        try:
            data = self._fetch_json(url)
            return data.get("readme", "")
        except Exception:
            return ""

    def download_plugin(self, plugin_id: str,
                       dest_path: Optional[str] = None,
                       version: Optional[str] = None,
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> Tuple[bool, str]:
        """下载插件包

        Args:
            plugin_id: 插件ID
            dest_path: 目标路径（可选）
            version: 版本号（可选）
            progress_callback: 进度回调

        Returns:
            (是否成功, 文件路径或错误信息)
        """
        # 获取插件元数据
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False, f"插件不存在: {plugin_id}"

        download_url = plugin.download_url
        if version:
            # 获取特定版本
            versions = self.get_plugin_versions(plugin_id)
            for v in versions:
                if v.get("version") == version:
                    download_url = v.get("download_url", download_url)
                    break

        if not download_url:
            return False, "下载链接不可用"

        # 确定文件名
        filename = f"{plugin.name}-{plugin.version}.zip"
        if version:
            filename = f"{plugin.name}-{version}.zip"

        if dest_path:
            if os.path.isdir(dest_path):
                filename = os.path.join(dest_path, filename)
            else:
                filename = dest_path

        # 下载
        return self._downloader.download(download_url, filename, progress_callback)

    def get_ratings(self, plugin_id: str,
                   source: Optional[str] = None) -> PluginRating:
        """获取插件评分

        Args:
            plugin_id: 插件ID
            source: 评分来源（可选）

        Returns:
            聚合评分
        """
        aggregator = RatingAggregator()

        if source:
            url = f"{self._base_url}/api/v1/plugins/{plugin_id}/ratings?source={source}"
        else:
            url = f"{self._base_url}/api/v1/plugins/{plugin_id}/ratings"

        try:
            data = self._fetch_json(url)
            ratings = data.get("ratings", [])
            return aggregator.aggregate(plugin_id, ratings)
        except Exception:
            return PluginRating()

    def submit_rating(self, plugin_id: str,
                     rating: int,
                     review: Optional[str] = None,
                     source: str = "community") -> bool:
        """提交评分

        Args:
            plugin_id: 插件ID
            rating: 评分 (1-5)
            review: 评论（可选）
            source: 来源

        Returns:
            是否成功
        """
        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/ratings"

        payload = {
            "rating": rating,
            "source": source,
        }
        if review:
            payload["review"] = review

        try:
            self._post_json(url, payload)
            return True
        except Exception:
            return False

    def report_plugin(self, plugin_id: str,
                     reason: str,
                     details: Optional[str] = None) -> bool:
        """举报插件

        Args:
            plugin_id: 插件ID
            reason: 举报原因
            details: 详细说明

        Returns:
            是否成功
        """
        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/report"

        payload = {
            "reason": reason,
        }
        if details:
            payload["details"] = details

        try:
            self._post_json(url, payload)
            return True
        except Exception:
            return False

    def get_related_plugins(self, plugin_id: str,
                           limit: int = 5) -> List[PluginMetadata]:
        """获取相关插件

        Args:
            plugin_id: 插件ID
            limit: 返回数量

        Returns:
            插件列表
        """
        url = f"{self._base_url}/api/v1/plugins/{plugin_id}/related?limit={limit}"
        try:
            data = self._fetch_json(url)
            plugins = []
            for item in data.get("plugins", []):
                try:
                    plugins.append(PluginMetadata.from_dict(item))
                except Exception:
                    continue
            return plugins
        except Exception:
            return []

    def check_compatibility(self, plugin_id: str,
                           agi_version: str) -> Tuple[bool, str]:
        """检查插件兼容性

        Args:
            plugin_id: 插件ID
            agi_version: AGI版本

        Returns:
            (是否兼容, 错误信息)
        """
        plugin = self.get_plugin(plugin_id)
        if not plugin:
            return False, "插件不存在"

        min_version = plugin.min_agi_version
        if min_version and agi_version < min_version:
            return False, f"需要AGI版本 {min_version} 或更高"

        compatibility = plugin.compatibility.get(agi_version)
        if compatibility == "unsupported":
            return False, f"AGI版本 {agi_version} 不受支持"

        return True, ""

    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._metadata_cache.clear()
        self._downloader.clear_cache()

    def _fetch_json(self, url: str) -> dict:
        """获取JSON数据"""
        request = urllib.request.Request(url, headers=self._headers)

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                raise Exception(error_data.get("message", str(e)))
            except json.JSONDecodeError:
                raise Exception(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"网络错误: {str(e.reason)}")

    def _post_json(self, url: str, payload: dict) -> dict:
        """发送JSON数据"""
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers=self._headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                raise Exception(error_data.get("message", str(e)))
            except json.JSONDecodeError:
                raise Exception(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise Exception(f"网络错误: {str(e.reason)}")
