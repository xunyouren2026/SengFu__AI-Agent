"""
内置网页搜索工具模块
聚合DuckDuckGo和Bing搜索结果，支持速率限制、结果去重、缓存、搜索历史、安全搜索过滤
"""

import os
import re
import json
import time
import hashlib
import urllib.parse
import urllib.request
import urllib.error
import threading
import logging
from typing import (
    Optional, Union, List, Dict, Any, Callable, Tuple,
    Set, Iterator, Generator
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from html.parser import HTMLParser
import http.client

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """搜索提供者枚举"""
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    AGGREGATED = "aggregated"


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    url: str
    snippet: str
    provider: SearchProvider
    rank: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """搜索响应数据类"""
    query: str
    results: List[SearchResult]
    total_results: int
    provider: SearchProvider
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    cached: bool = False


class HTMLTextExtractor(HTMLParser):
    """HTML文本提取器"""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.skip_tags: Set[str] = {'script', 'style', 'noscript'}
        self.current_skip: Optional[str] = None
        self.link_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self.skip_tags:
            self.current_skip = tag
        elif tag == 'a':
            for attr_name, attr_value in attrs:
                if attr_name == 'href' and attr_value:
                    self.link_text.append(attr_value)

    def handle_endtag(self, tag: str) -> None:
        if tag == self.current_skip:
            self.current_skip = None

    def handle_data(self, data: str) -> None:
        if self.current_skip is None:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self.text_parts)


class SafeSearchFilter:
    """安全搜索过滤器"""

    # 成人内容关键词
    ADULT_PATTERNS: List[str] = [
        r'\bxxx\b', r'\bporn\b', r'\bnude\b', r'\bnaked\b',
        r'\bnsfw\b', r'\b成人\b', r'\b色情\b', r'\b激情\b',
    ]

    # 暴力内容关键词
    VIOLENCE_PATTERNS: List[str] = [
        r'\bgore\b', r'\bgraphic\s+violence\b', r'\bbrutal\b',
    ]

    # 危险内容关键词
    DANGER_PATTERNS: List[str] = [
        r'\bhack\s+tutorial\b', r'\bhow\s+to\s+make\s+bomb\b',
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        block_adult: bool = True,
        block_violence: bool = False,
        block_danger: bool = True,
        custom_blocked_terms: Optional[List[str]] = None,
    ):
        """
        初始化安全过滤器

        Args:
            strict_mode: 严格模式
            block_adult: 阻止成人内容
            block_violence: 阻止暴力内容
            block_danger: 阻止危险内容
            custom_blocked_terms: 自定义阻止词
        """
        self.strict_mode = strict_mode
        self.block_adult = block_adult
        self.block_violence = block_violence
        self.block_danger = block_danger
        self.custom_blocked_terms = custom_blocked_terms or []

        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """编译过滤模式"""
        self._adult_regex = re.compile(
            '|'.join(self.ADULT_PATTERNS), re.IGNORECASE
        ) if self.block_adult else None
        self._violence_regex = re.compile(
            '|'.join(self.VIOLENCE_PATTERNS), re.IGNORECASE
        ) if self.block_violence else None
        self._danger_regex = re.compile(
            '|'.join(self.DANGER_PATTERNS), re.IGNORECASE
        ) if self.block_danger else None
        self._custom_regex = re.compile(
            '|'.join(re.escape(term) for term in self.custom_blocked_terms),
            re.IGNORECASE
        ) if self.custom_blocked_terms else None

    def is_safe(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        检查内容是否安全

        Args:
            text: 待检查文本

        Returns:
            (是否安全, 阻止原因)
        """
        text_lower = text.lower()

        if self._adult_regex and self._adult_regex.search(text):
            return False, "成人内容"

        if self._violence_regex and self._violence_regex.search(text):
            return False, "暴力内容"

        if self._danger_regex and self._danger_regex.search(text):
            return False, "危险内容"

        if self._custom_regex and self._custom_regex.search(text):
            return False, "自定义过滤"

        return True, None

    def filter_results(
        self,
        results: List[SearchResult],
        aggressive: bool = False,
    ) -> List[SearchResult]:
        """
        过滤搜索结果

        Args:
            results: 搜索结果列表
            aggressive: 激进模式 - 检查所有字段

        Returns:
            过滤后的结果
        """
        filtered = []

        for result in results:
            text_to_check = f"{result.title} {result.snippet}"

            if aggressive:
                text_to_check += f" {result.url}"

            is_safe, reason = self.is_safe(text_to_check)

            if is_safe:
                filtered.append(result)
            else:
                logger.debug(f"过滤结果: {result.title[:50]}... - {reason}")

        return filtered

    def add_blocked_term(self, term: str) -> None:
        """添加阻止词"""
        self.custom_blocked_terms.append(term)
        self._compile_patterns()

    def remove_blocked_term(self, term: str) -> None:
        """移除阻止词"""
        if term in self.custom_blocked_terms:
            self.custom_blocked_terms.remove(term)
            self._compile_patterns()


class RateLimiter:
    """速率限制器"""

    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_hour: int = 500,
        requests_per_day: int = 5000,
        burst_limit: int = 5,
    ):
        """
        初始化速率限制器

        Args:
            requests_per_minute: 每分钟请求数
            requests_per_hour: 每小时请求数
            requests_per_day: 每天请求数
            burst_limit: 突发限制
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit

        self._minute_window: List[float] = []
        self._hour_window: List[float] = []
        self._day_window: List[float] = []
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        获取请求许可

        Args:
            timeout: 超时时间

        Returns:
            是否获得许可
        """
        start_time = time.time()

        while True:
            with self._lock:
                now = time.time()
                self._cleanup_windows(now)

                # 检查各层级限制
                minute_count = len(self._minute_window)
                hour_count = len(self._hour_window)
                day_count = len(self._day_window)

                if minute_count >= self.requests_per_minute:
                    wait_time = 60 - (now - self._minute_window[0])
                    if wait_time > 0:
                        if time.time() - start_time + wait_time > timeout:
                            return False
                        time.sleep(min(wait_time, 1.0))
                        continue

                if hour_count >= self.requests_per_hour:
                    wait_time = 3600 - (now - self._hour_window[0])
                    if wait_time > 0:
                        if time.time() - start_time + wait_time > timeout:
                            return False
                        time.sleep(min(wait_time, 1.0))
                        continue

                if day_count >= self.requests_per_day:
                    wait_time = 86400 - (now - self._day_window[0])
                    if wait_time > 0:
                        if time.time() - start_time + wait_time > timeout:
                            return False
                        time.sleep(min(wait_time, 1.0))
                        continue

                # 检查突发限制
                recent_count = sum(
                    1 for t in self._minute_window
                    if now - t < 1.0
                )
                if recent_count >= self.burst_limit:
                    time.sleep(0.2)
                    continue

                # 记录请求
                self._minute_window.append(now)
                self._hour_window.append(now)
                self._day_window.append(now)
                return True

    def _cleanup_windows(self, now: float) -> None:
        """清理过期记录"""
        self._minute_window = [
            t for t in self._minute_window
            if now - t < 60
        ]
        self._hour_window = [
            t for t in self._hour_window
            if now - t < 3600
        ]
        self._day_window = [
            t for t in self._day_window
            if now - t < 86400
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            now = time.time()
            return {
                'requests_last_minute': len([
                    t for t in self._minute_window if now - t < 60
                ]),
                'requests_last_hour': len([
                    t for t in self._hour_window if now - t < 3600
                ]),
                'requests_last_day': len([
                    t for t in self._day_window if now - t < 86400
                ]),
                'limits': {
                    'per_minute': self.requests_per_minute,
                    'per_hour': self.requests_per_hour,
                    'per_day': self.requests_per_day,
                }
            }


class ResultCache:
    """搜索结果缓存"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
    ):
        """
        初始化缓存

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存有效期(秒)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[SearchResponse, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _generate_key(self, query: str, provider: SearchProvider) -> str:
        """生成缓存键"""
        normalized = query.lower().strip()
        key_string = f"{provider.value}:{normalized}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        query: str,
        provider: SearchProvider,
    ) -> Optional[SearchResponse]:
        """
        获取缓存的搜索结果

        Args:
            query: 搜索查询
            provider: 搜索提供者

        Returns:
            缓存的搜索响应,无则返回None
        """
        key = self._generate_key(query, provider)

        with self._lock:
            if key not in self._cache:
                return None

            response, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None

            # 移到末尾(最近使用)
            self._cache.move_to_end(key)

            # 返回缓存副本
            cached_response = SearchResponse(
                query=response.query,
                results=response.results.copy(),
                total_results=response.total_results,
                provider=response.provider,
                execution_time=response.execution_time,
                timestamp=response.timestamp,
                error=response.error,
                cached=True,
            )
            return cached_response

    def put(
        self,
        query: str,
        provider: SearchProvider,
        response: SearchResponse,
    ) -> None:
        """
        存储搜索结果到缓存

        Args:
            query: 搜索查询
            provider: 搜索提供者
            response: 搜索响应
        """
        key = self._generate_key(query, provider)

        with self._lock:
            # 清理过期和超出大小的条目
            self._cleanup()

            self._cache[key] = (response, time.time())
            self._cache.move_to_end(key)

    def _cleanup(self) -> None:
        """清理过期和超出大小的缓存"""
        now = time.time()
        expired_keys = [
            k for k, (_, timestamp) in self._cache.items()
            if now - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
            }


class SearchHistory:
    """搜索历史记录"""

    def __init__(
        self,
        max_history: int = 1000,
        persist_file: Optional[str] = None,
    ):
        """
        初始化搜索历史

        Args:
            max_history: 最大历史记录数
            persist_file: 持久化文件路径
        """
        self.max_history = max_history
        self.persist_file = persist_file
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        if persist_file and os.path.exists(persist_file):
            self._load()

    def add(
        self,
        query: str,
        provider: SearchProvider,
        result_count: int,
        execution_time: float,
    ) -> None:
        """
        添加搜索历史

        Args:
            query: 搜索查询
            provider: 搜索提供者
            result_count: 结果数量
            execution_time: 执行时间
        """
        with self._lock:
            entry = {
                'query': query,
                'provider': provider.value,
                'result_count': result_count,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
            }

            self._history.append(entry)

            # 限制大小
            while len(self._history) > self.max_history:
                self._history.pop(0)

            # 持久化
            if self.persist_file:
                self._save()

    def get_recent(
        self,
        limit: int = 10,
        provider: Optional[SearchProvider] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取最近的搜索记录

        Args:
            limit: 返回数量
            provider: 过滤特定提供者

        Returns:
            搜索历史列表
        """
        with self._lock:
            history = self._history

            if provider:
                history = [
                    h for h in history
                    if h['provider'] == provider.value
                ]

            return history[-limit:]

    def search_queries(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的历史记录

        Args:
            query: 搜索关键词
            limit: 返回数量

        Returns:
            匹配的搜索历史
        """
        with self._lock:
            query_lower = query.lower()
            matches = [
                h for h in self._history
                if query_lower in h['query'].lower()
            ]
            return matches[-limit:]

    def clear(self) -> None:
        """清空历史"""
        with self._lock:
            self._history.clear()
            if self.persist_file and os.path.exists(self.persist_file):
                os.remove(self.persist_file)

    def _load(self) -> None:
        """加载历史"""
        try:
            with open(self.persist_file, 'r', encoding='utf-8') as f:
                self._history = json.load(f)
        except Exception as e:
            logger.warning(f"加载搜索历史失败: {e}")
            self._history = []

    def _save(self) -> None:
        """保存历史"""
        try:
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存搜索历史失败: {e}")


class DuckDuckGoSearch:
    """DuckDuckGo搜索客户端"""

    BASE_URL = "https://duckduckgo.com/html/"
    API_URL = "https://api.duckduckgo.com/"

    def __init__(
        self,
        timeout: int = 10,
        user_agent: Optional[str] = None,
        safe_search: bool = True,
    ):
        """
        初始化DuckDuckGo搜索

        Args:
            timeout: 超时时间
            user_agent: 用户代理
            safe_search: 安全搜索
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.safe_search = safe_search

    def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> SearchResponse:
        """
        执行搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数

        Returns:
            搜索响应
        """
        start_time = time.time()

        try:
            # URL编码
            encoded_query = urllib.parse.quote_plus(query)
            url = f"{self.BASE_URL}?q={encoded_query}&kl=wt-wt"

            if self.safe_search:
                url += "&ia=web"

            # 构建请求
            request = urllib.request.Request(url)
            request.add_header('User-Agent', self.user_agent)
            request.add_header('Accept', 'text/html,application/xhtml+xml')
            request.add_header('Accept-Language', 'en-US,en;q=0.9')

            # 发送请求
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # 解析结果
            results = self._parse_results(html, num_results)
            execution_time = time.time() - start_time

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                provider=SearchProvider.DUCKDUCKGO,
                execution_time=execution_time,
            )

        except urllib.error.HTTPError as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=SearchProvider.DUCKDUCKGO,
                execution_time=time.time() - start_time,
                error=f"HTTP错误: {e.code}",
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=SearchProvider.DUCKDUCKGO,
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def _parse_results(self, html: str, max_results: int) -> List[SearchResult]:
        """解析搜索结果"""
        results = []

        # DuckDuckGo HTML结果模式
        pattern = re.compile(
            r'<a class="result__a" href="([^"]+)">([^<]+)</a>.*?'
            r'<a class="result__snippet" href="[^"]*">([^<]+)</a>',
            re.DOTALL
        )

        for i, match in enumerate(pattern.finditer(html)):
            if i >= max_results:
                break

            url = match.group(1)
            title = self._clean_html(match.group(2))
            snippet = self._clean_html(match.group(3))

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                provider=SearchProvider.DUCKDUCKGO,
                rank=i + 1,
            ))

        return results

    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 解码HTML实体
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text.strip()


class BingSearch:
    """Bing搜索客户端"""

    # 免费Bing搜索API端点(用于演示)
    API_URL = "https://www.bing.com/search"

    def __init__(
        self,
        timeout: int = 10,
        user_agent: Optional[str] = None,
        api_key: Optional[str] = None,
        safe_search: bool = True,
    ):
        """
        初始化Bing搜索

        Args:
            timeout: 超时时间
            user_agent: 用户代理
            api_key: API密钥(可选)
            safe_search: 安全搜索
        """
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.api_key = api_key
        self.safe_search = safe_search

    def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> SearchResponse:
        """
        执行搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数

        Returns:
            搜索响应
        """
        start_time = time.time()

        try:
            # URL编码
            encoded_query = urllib.parse.quote_plus(query)
            url = f"{self.API_URL}?q={encoded_query}&count={num_results}"

            # 构建请求
            request = urllib.request.Request(url)
            request.add_header('User-Agent', self.user_agent)
            request.add_header('Accept', 'text/html,application/xhtml+xml')
            request.add_header('Accept-Language', 'en-US,en;q=0.9')

            # 发送请求
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                html = response.read().decode('utf-8', errors='ignore')

            # 解析结果
            results = self._parse_results(html, num_results)
            execution_time = time.time() - start_time

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                provider=SearchProvider.BING,
                execution_time=execution_time,
            )

        except urllib.error.HTTPError as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=SearchProvider.BING,
                execution_time=time.time() - start_time,
                error=f"HTTP错误: {e.code}",
            )
        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=SearchProvider.BING,
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def _parse_results(self, html: str, max_results: int) -> List[SearchResult]:
        """解析搜索结果"""
        results = []

        # Bing结果模式
        pattern = re.compile(
            r'<li class="b_algo">.*?'
            r'<a href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'<p>([^<]+)</p>',
            re.DOTALL
        )

        for i, match in enumerate(pattern.finditer(html)):
            if i >= max_results:
                break

            url = match.group(1)
            title = self._clean_html(match.group(2))
            snippet = self._clean_html(match.group(3))

            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                provider=SearchProvider.BING,
                rank=i + 1,
            ))

        return results

    def _clean_html(self, text: str) -> str:
        """清理HTML标签"""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text.strip()


class SearchAggregator:
    """搜索结果聚合器"""

    def __init__(
        self,
        providers: Optional[List[Any]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ResultCache] = None,
        safe_search_filter: Optional[SafeSearchFilter] = None,
        deduplicate: bool = True,
        max_results: int = 20,
    ):
        """
        初始化搜索聚合器

        Args:
            providers: 搜索提供者列表
            rate_limiter: 速率限制器
            cache: 结果缓存
            safe_search_filter: 安全过滤器
            deduplicate: 是否去重
            max_results: 最大结果数
        """
        self.providers = providers or [
            DuckDuckGoSearch(),
            BingSearch(),
        ]
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache or ResultCache()
        self.safe_search_filter = safe_search_filter or SafeSearchFilter()
        self.deduplicate = deduplicate
        self.max_results = max_results

    def search(
        self,
        query: str,
        providers: Optional[List[SearchProvider]] = None,
        use_cache: bool = True,
        apply_filter: bool = True,
    ) -> SearchResponse:
        """
        执行聚合搜索

        Args:
            query: 搜索查询
            providers: 指定使用的提供者
            use_cache: 是否使用缓存
            apply_filter: 是否应用安全过滤

        Returns:
            聚合后的搜索响应
        """
        start_time = time.time()

        # 确定使用的提供者
        if providers:
            provider_map = {
                SearchProvider.DUCKDUCKGO: self.providers[0] if len(self.providers) > 0 else DuckDuckGoSearch(),
                SearchProvider.BING: self.providers[1] if len(self.providers) > 1 else BingSearch(),
            }
            search_providers = [
                provider_map.get(p)
                for p in providers
                if p in provider_map
            ]
        else:
            search_providers = self.providers

        # 获取所有提供者的结果
        all_results: List[SearchResult] = []
        errors: List[str] = []

        for provider in search_providers:
            # 检查速率限制
            if not self.rate_limiter.acquire():
                errors.append(f"速率限制: {provider}")
                continue

            # 检查缓存
            if use_cache:
                cached = self.cache.get(query, provider.provider)
                if cached:
                    all_results.extend(cached.results)
                    continue

            # 执行搜索
            response = provider.search(query)

            if response.error:
                errors.append(f"{provider.provider}: {response.error}")
                continue

            # 缓存结果
            if use_cache and response.results:
                self.cache.put(query, provider.provider, response)

            all_results.extend(response.results)

        # 去重
        if self.deduplicate:
            all_results = self._deduplicate_results(all_results)

        # 安全过滤
        if apply_filter:
            all_results = self.safe_search_filter.filter_results(all_results)

        # 限制结果数
        all_results = all_results[:self.max_results]

        # 重新排序
        for i, result in enumerate(all_results):
            result.rank = i + 1

        execution_time = time.time() - start_time

        return SearchResponse(
            query=query,
            results=all_results,
            total_results=len(all_results),
            provider=SearchProvider.AGGREGATED,
            execution_time=execution_time,
            error='; '.join(errors) if errors else None,
        )

    def _deduplicate_results(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """去重搜索结果"""
        seen_urls: Set[str] = set()
        seen_titles: Set[str] = set()
        deduplicated: List[SearchResult] = []

        for result in results:
            # 规范化URL
            normalized_url = self._normalize_url(result.url)
            # 规范化标题
            normalized_title = result.title.lower().strip()

            # 检查是否重复
            if normalized_url in seen_urls:
                continue
            if normalized_title in seen_titles:
                continue

            seen_urls.add(normalized_url)
            seen_titles.add(normalized_title)
            deduplicated.append(result)

        return deduplicated

    def _normalize_url(self, url: str) -> str:
        """规范化URL"""
        parsed = urllib.parse.urlparse(url)
        # 移除跟踪参数
        return urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            '',  # 移除查询参数
            '',  # 移除片段
        ))


class WebSearch:
    """统一网页搜索接口"""

    def __init__(
        self,
        default_provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        rate_limit: int = 30,
        cache_enabled: bool = True,
        safe_search: bool = True,
        history_enabled: bool = True,
    ):
        """
        初始化网页搜索

        Args:
            default_provider: 默认搜索提供者
            rate_limit: 每分钟请求数
            cache_enabled: 是否启用缓存
            safe_search: 安全搜索
            history_enabled: 是否记录历史
        """
        self.default_provider = default_provider

        # 初始化组件
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit)
        self.cache = ResultCache() if cache_enabled else None
        self.safe_filter = SafeSearchFilter() if safe_search else None
        self.history = SearchHistory() if history_enabled else None

        # 初始化提供者
        self.duckduckgo = DuckDuckGoSearch()
        self.bing = BingSearch()

        # 初始化聚合器
        self.aggregator = SearchAggregator(
            providers=[self.duckduckgo, self.bing],
            rate_limiter=self.rate_limiter,
            cache=self.cache,
            safe_search_filter=self.safe_filter,
        )

    def search(
        self,
        query: str,
        provider: Optional[SearchProvider] = None,
        num_results: int = 10,
        use_cache: bool = True,
        safe_search: bool = True,
    ) -> SearchResponse:
        """
        执行网页搜索

        Args:
            query: 搜索查询
            provider: 搜索提供者
            num_results: 返回结果数
            use_cache: 是否使用缓存
            safe_search: 是否启用安全过滤

        Returns:
            搜索响应
        """
        provider = provider or self.default_provider

        # 检查速率限制
        if not self.rate_limiter.acquire():
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                provider=provider,
                execution_time=0,
                error="速率限制",
            )

        # 根据提供者执行搜索
        if provider == SearchProvider.DUCKDUCKGO:
            response = self.duckduckgo.search(query, num_results)
        elif provider == SearchProvider.BING:
            response = self.bing.search(query, num_results)
        else:
            response = self.aggregator.search(
                query,
                providers=[provider] if provider != SearchProvider.AGGREGATED else None,
                use_cache=use_cache,
                apply_filter=safe_search,
            )

        # 安全过滤
        if safe_search and self.safe_filter and response.results:
            response.results = self.safe_filter.filter_results(response.results)

        # 记录历史
        if self.history:
            self.history.add(
                query=query,
                provider=provider,
                result_count=response.total_results,
                execution_time=response.execution_time,
            )

        return response

    def batch_search(
        self,
        queries: List[str],
        provider: Optional[SearchProvider] = None,
        num_results: int = 10,
    ) -> List[SearchResponse]:
        """
        批量搜索

        Args:
            queries: 搜索查询列表
            provider: 搜索提供者
            num_results: 返回结果数

        Returns:
            搜索响应列表
        """
        results = []
        for query in queries:
            response = self.search(
                query=query,
                provider=provider,
                num_results=num_results,
            )
            results.append(response)

            # 避免过快请求
            time.sleep(0.5)

        return results

    def get_history(
        self,
        limit: int = 10,
        provider: Optional[SearchProvider] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取搜索历史

        Args:
            limit: 返回数量
            provider: 过滤特定提供者

        Returns:
            历史记录列表
        """
        if not self.history:
            return []

        return self.history.get_recent(limit=limit, provider=provider)

    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'rate_limiter': self.rate_limiter.get_stats(),
        }

        if self.cache:
            stats['cache'] = self.cache.get_stats()

        return stats
