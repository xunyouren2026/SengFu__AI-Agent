"""
REST API服务模块

提供插件CRUD、搜索API、下载统计和认证接口。
"""

import json
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import logging


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class APIResponse:
    """API响应"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'success': self.success,
            'data': self.data,
            'meta': self.meta,
        }
        if self.error:
            result['error'] = self.error
        return result


@dataclass
class APIRequest:
    """API请求"""
    method: str
    path: str
    query_params: Dict[str, List[str]]
    headers: Dict[str, str]
    body: Optional[bytes] = None
    user_id: Optional[str] = None
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """获取查询参数"""
        values = self.query_params.get(name, [])
        return values[0] if values else default
    
    def get_json(self) -> Optional[Dict[str, Any]]:
        """获取JSON请求体"""
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode('utf-8'))
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# 路由系统
# ---------------------------------------------------------------------------

class Router:
    """API路由器"""
    
    def __init__(self):
        self._routes: Dict[str, Dict[str, Callable]] = {}
        self._middleware: List[Callable] = []
    
    def add_route(self, method: str, path: str, handler: Callable) -> None:
        """添加路由"""
        if method not in self._routes:
            self._routes[method] = {}
        self._routes[method][path] = handler
        logger.info(f"Added route: {method} {path}")
    
    def get(self, path: str):
        """GET路由装饰器"""
        def decorator(handler):
            self.add_route('GET', path, handler)
            return handler
        return decorator
    
    def post(self, path: str):
        """POST路由装饰器"""
        def decorator(handler):
            self.add_route('POST', path, handler)
            return handler
        return decorator
    
    def put(self, path: str):
        """PUT路由装饰器"""
        def decorator(handler):
            self.add_route('PUT', path, handler)
            return handler
        return decorator
    
    def delete(self, path: str):
        """DELETE路由装饰器"""
        def decorator(handler):
            self.add_route('DELETE', path, handler)
            return handler
        return decorator
    
    def add_middleware(self, middleware: Callable) -> None:
        """添加中间件"""
        self._middleware.append(middleware)
    
    def match(self, method: str, path: str) -> Optional[Callable]:
        """匹配路由"""
        routes = self._routes.get(method, {})
        
        # 精确匹配
        if path in routes:
            return routes[path]
        
        # 模式匹配
        for route_path, handler in routes.items():
            if self._match_pattern(route_path, path):
                return handler
        
        return None
    
    def _match_pattern(self, pattern: str, path: str) -> bool:
        """匹配路径模式"""
        # 简单实现：支持 :param 格式的参数
        pattern_parts = pattern.split('/')
        path_parts = path.split('/')
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for p_part, path_part in zip(pattern_parts, path_parts):
            if p_part.startswith(':'):
                continue
            if p_part != path_part:
                return False
        
        return True
    
    def extract_params(self, pattern: str, path: str) -> Dict[str, str]:
        """提取路径参数"""
        params = {}
        pattern_parts = pattern.split('/')
        path_parts = path.split('/')
        
        for p_part, path_part in zip(pattern_parts, path_parts):
            if p_part.startswith(':'):
                params[p_part[1:]] = path_part
        
        return params


# ---------------------------------------------------------------------------
# 插件API
# ---------------------------------------------------------------------------

class PluginAPI:
    """插件API处理器"""
    
    def __init__(self, registry):
        """
        Args:
            registry: 插件注册表实例
        """
        self._registry = registry
        self._router = Router()
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置路由"""
        self._router.get('/api/v1/plugins')(self.list_plugins)
        self._router.post('/api/v1/plugins')(self.create_plugin)
        self._router.get('/api/v1/plugins/:id')(self.get_plugin)
        self._router.put('/api/v1/plugins/:id')(self.update_plugin)
        self._router.delete('/api/v1/plugins/:id')(self.delete_plugin)
        self._router.get('/api/v1/plugins/:id/versions')(self.list_versions)
        self._router.post('/api/v1/plugins/:id/versions')(self.add_version)
        self._router.get('/api/v1/plugins/:id/download')(self.download_plugin)
        self._router.post('/api/v1/plugins/:id/rate')(self.rate_plugin)
        self._router.get('/api/v1/plugins/:id/reviews')(self.get_reviews)
        self._router.post('/api/v1/plugins/:id/reviews')(self.add_review)
    
    def list_plugins(self, request: APIRequest) -> APIResponse:
        """列出插件"""
        try:
            category = request.get_param('category')
            tag = request.get_param('tag')
            author = request.get_param('author')
            page = int(request.get_param('page', 1))
            page_size = int(request.get_param('page_size', 20))
            
            plugins = self._registry.list_plugins(
                category=category,
                tag=tag,
                author=author,
            )
            
            # 分页
            total = len(plugins)
            start = (page - 1) * page_size
            end = start + page_size
            paginated = plugins[start:end]
            
            return APIResponse(
                success=True,
                data=[p.to_dict() for p in paginated],
                meta={
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': (total + page_size - 1) // page_size,
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def create_plugin(self, request: APIRequest) -> APIResponse:
        """创建插件"""
        try:
            data = request.get_json()
            if not data:
                return APIResponse(success=False, error="Invalid JSON body")
            
            from ..registry import PluginMetadata, PluginAuthor
            
            metadata = PluginMetadata(
                plugin_id=data.get('plugin_id'),
                name=data.get('name'),
                version=data.get('version', '0.1.0'),
                description=data.get('description', ''),
                author=PluginAuthor.from_dict(data.get('author', {'name': 'Unknown'})),
                category=data.get('category', 'general'),
                tags=set(data.get('tags', [])),
            )
            
            success = self._registry.register(metadata)
            
            return APIResponse(
                success=success,
                data=metadata.to_dict() if success else None,
                error=None if success else "Plugin already exists"
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_plugin(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """获取插件详情"""
        try:
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return APIResponse(success=False, error="Plugin not found")
            
            return APIResponse(success=True, data=plugin.to_dict())
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def update_plugin(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """更新插件"""
        try:
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return APIResponse(success=False, error="Plugin not found")
            
            data = request.get_json()
            if not data:
                return APIResponse(success=False, error="Invalid JSON body")
            
            # 更新字段
            for key, value in data.items():
                if hasattr(plugin, key):
                    setattr(plugin, key, value)
            
            success = self._registry.register(plugin)
            
            return APIResponse(
                success=success,
                data=plugin.to_dict() if success else None
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def delete_plugin(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """删除插件"""
        try:
            success = self._registry.unregister(plugin_id)
            return APIResponse(
                success=success,
                error=None if success else "Plugin not found or has dependents"
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def list_versions(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """列出插件版本"""
        try:
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return APIResponse(success=False, error="Plugin not found")
            
            return APIResponse(
                success=True,
                data=[v.to_dict() for v in plugin.versions]
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def add_version(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """添加版本"""
        try:
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return APIResponse(success=False, error="Plugin not found")
            
            data = request.get_json()
            if not data:
                return APIResponse(success=False, error="Invalid JSON body")
            
            from ..registry import PluginVersion
            
            version = PluginVersion.from_dict(data)
            plugin.add_version(version)
            
            self._registry.register(plugin)
            
            return APIResponse(success=True, data=version.to_dict())
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def download_plugin(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """下载插件"""
        try:
            version = request.get_param('version')
            
            plugin = self._registry.get(plugin_id)
            if not plugin:
                return APIResponse(success=False, error="Plugin not found")
            
            # 更新下载统计
            plugin.stats.record_download()
            self._registry.register(plugin)
            
            version_info = plugin.get_version_info(version) if version else None
            if version and not version_info:
                return APIResponse(success=False, error="Version not found")
            
            return APIResponse(
                success=True,
                data={
                    'plugin_id': plugin_id,
                    'version': version or plugin.latest_version,
                    'download_url': version_info.download_url if version_info else '',
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def rate_plugin(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """评分插件"""
        try:
            data = request.get_json()
            if not data:
                return APIResponse(success=False, error="Invalid JSON body")
            
            rating = data.get('rating')
            if not rating or not 1 <= rating <= 5:
                return APIResponse(success=False, error="Rating must be between 1 and 5")
            
            # 这里应该调用评分系统
            # self._rating_system.submit_rating(request.user_id, plugin_id, rating)
            
            return APIResponse(success=True, data={'rating': rating})
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_reviews(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """获取评论"""
        try:
            page = int(request.get_param('page', 1))
            page_size = int(request.get_param('page_size', 10))
            sort_by = request.get_param('sort_by', 'newest')
            
            # 这里应该调用评分系统
            # reviews, total = self._rating_system.get_plugin_reviews(plugin_id, sort_by, page, page_size)
            
            return APIResponse(
                success=True,
                data=[],  # [r.to_dict() for r in reviews]
                meta={
                    'total': 0,  # total
                    'page': page,
                    'page_size': page_size,
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def add_review(self, request: APIRequest, plugin_id: str) -> APIResponse:
        """添加评论"""
        try:
            data = request.get_json()
            if not data:
                return APIResponse(success=False, error="Invalid JSON body")
            
            # 这里应该调用评分系统
            # review = Review(...)
            # self._rating_system.submit_review(review)
            
            return APIResponse(success=True, data={})
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def handle(self, request: APIRequest, path: str) -> Optional[APIResponse]:
        """处理请求"""
        handler = self._router.match(request.method, path)
        if handler:
            # 提取路径参数
            for route_path in self._router._routes.get(request.method, {}):
                if self._router._match_pattern(route_path, path):
                    params = self._router.extract_params(route_path, path)
                    return handler(request, **params)
        return None


# ---------------------------------------------------------------------------
# 搜索API
# ---------------------------------------------------------------------------

class SearchAPI:
    """搜索API处理器"""
    
    def __init__(self, indexer):
        """
        Args:
            indexer: 全文索引器实例
        """
        self._indexer = indexer
        self._router = Router()
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置路由"""
        self._router.get('/api/v1/search')(self.search)
        self._router.get('/api/v1/search/suggestions')(self.get_suggestions)
        self._router.get('/api/v1/search/trending')(self.get_trending)
        self._router.get('/api/v1/search/recommendations')(self.get_recommendations)
    
    def search(self, request: APIRequest) -> APIResponse:
        """搜索"""
        try:
            query = request.get_param('q', '')
            page = int(request.get_param('page', 1))
            page_size = int(request.get_param('page_size', 20))
            category = request.get_param('category')
            
            from ..indexer import SearchQuery
            
            search_query = SearchQuery(
                query=query,
                filters={'category': category} if category else {},
                page=page,
                page_size=page_size,
            )
            
            result = self._indexer.search(search_query)
            
            return APIResponse(
                success=True,
                data=result.to_dict()
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_suggestions(self, request: APIRequest) -> APIResponse:
        """获取搜索建议"""
        try:
            prefix = request.get_param('q', '')
            limit = int(request.get_param('limit', 10))
            
            suggestions = self._indexer.get_suggestions(prefix, limit)
            
            return APIResponse(success=True, data=suggestions)
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_trending(self, request: APIRequest) -> APIResponse:
        """获取热门"""
        try:
            limit = int(request.get_param('limit', 10))
            
            trending = self._indexer.get_trending(limit)
            
            return APIResponse(success=True, data=trending)
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_recommendations(self, request: APIRequest) -> APIResponse:
        """获取推荐"""
        try:
            user_id = request.get_param('user_id')
            limit = int(request.get_param('limit', 10))
            
            if user_id:
                recommendations = self._indexer.recommend_for_user(user_id, limit)
            else:
                recommendations = self._indexer.get_trending(limit)
            
            return APIResponse(success=True, data=recommendations)
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def handle(self, request: APIRequest, path: str) -> Optional[APIResponse]:
        """处理请求"""
        handler = self._router.match(request.method, path)
        if handler:
            return handler(request)
        return None


# ---------------------------------------------------------------------------
# 统计API
# ---------------------------------------------------------------------------

class StatsAPI:
    """统计API处理器"""
    
    def __init__(self, registry, indexer):
        """
        Args:
            registry: 插件注册表实例
            indexer: 全文索引器实例
        """
        self._registry = registry
        self._indexer = indexer
        self._router = Router()
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """设置路由"""
        self._router.get('/api/v1/stats')(self.get_stats)
        self._router.get('/api/v1/stats/downloads')(self.get_download_stats)
        self._router.get('/api/v1/stats/categories')(self.get_category_stats)
        self._router.get('/api/v1/stats/tags')(self.get_tag_stats)
    
    def get_stats(self, request: APIRequest) -> APIResponse:
        """获取总体统计"""
        try:
            registry_stats = self._registry.get_stats()
            index_stats = self._indexer.get_stats()
            
            return APIResponse(
                success=True,
                data={
                    'registry': registry_stats,
                    'index': index_stats,
                }
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_download_stats(self, request: APIRequest) -> APIResponse:
        """获取下载统计"""
        try:
            period = request.get_param('period', 'daily')  # daily, weekly, monthly
            
            plugins = self._registry.list_plugins()
            
            stats = {
                'total_downloads': sum(p.stats.downloads for p in plugins),
                'total_installs': sum(p.stats.installs for p in plugins),
            }
            
            if period == 'daily':
                # 合并所有插件的日下载数据
                daily = {}
                for p in plugins:
                    for date, count in p.stats.daily_downloads.items():
                        daily[date] = daily.get(date, 0) + count
                stats['by_date'] = daily
            
            return APIResponse(success=True, data=stats)
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_category_stats(self, request: APIRequest) -> APIResponse:
        """获取分类统计"""
        try:
            registry_stats = self._registry.get_stats()
            
            return APIResponse(
                success=True,
                data=registry_stats.get('categories', {})
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def get_tag_stats(self, request: APIRequest) -> APIResponse:
        """获取标签统计"""
        try:
            index_stats = self._indexer.get_stats()
            
            return APIResponse(
                success=True,
                data=index_stats.get('popular_tags', [])
            )
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def handle(self, request: APIRequest, path: str) -> Optional[APIResponse]:
        """处理请求"""
        handler = self._router.match(request.method, path)
        if handler:
            return handler(request)
        return None


# ---------------------------------------------------------------------------
# HTTP请求处理器
# ---------------------------------------------------------------------------

class HubHTTPHandler(BaseHTTPRequestHandler):
    """Hub HTTP请求处理器"""
    
    server_version = "ClawHub-API/1.0"
    
    def __init__(self, api_server, *args, **kwargs):
        self._api_server = api_server
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """自定义日志"""
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        """处理GET请求"""
        self._handle_request('GET')
    
    def do_POST(self):
        """处理POST请求"""
        self._handle_request('POST')
    
    def do_PUT(self):
        """处理PUT请求"""
        self._handle_request('PUT')
    
    def do_DELETE(self):
        """处理DELETE请求"""
        self._handle_request('DELETE')
    
    def _handle_request(self, method: str) -> None:
        """处理请求"""
        try:
            # 解析URL
            parsed = urlparse(self.path)
            path = parsed.path
            query_params = parse_qs(parsed.query)
            
            # 读取请求体
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            
            # 构建请求对象
            request = APIRequest(
                method=method,
                path=path,
                query_params=query_params,
                headers=dict(self.headers),
                body=body,
            )
            
            # 处理请求
            response = self._api_server.process_request(request)
            
            # 发送响应
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self._send_json_response(APIResponse(success=False, error=str(e)))
    
    def _send_json_response(self, response: APIResponse, status_code: int = 200) -> None:
        """发送JSON响应"""
        data = json.dumps(response.to_dict()).encode('utf-8')
        
        self.send_response(status_code if response.success else 400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# API服务器
# ---------------------------------------------------------------------------

class HubAPIServer:
    """Hub API服务器
    
    整合所有API功能的主类。
    """
    
    def __init__(self, registry, indexer, host: str = 'localhost', port: int = 8080):
        """
        Args:
            registry: 插件注册表实例
            indexer: 全文索引器实例
            host: 主机地址
            port: 端口
        """
        self._registry = registry
        self._indexer = indexer
        self._host = host
        self._port = port
        
        self._plugin_api = PluginAPI(registry)
        self._search_api = SearchAPI(indexer)
        self._stats_api = StatsAPI(registry, indexer)
        
        self._server: Optional[HTTPServer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def process_request(self, request: APIRequest) -> APIResponse:
        """处理API请求"""
        path = request.path
        
        # 尝试各个API处理器
        response = self._plugin_api.handle(request, path)
        if response:
            return response
        
        response = self._search_api.handle(request, path)
        if response:
            return response
        
        response = self._stats_api.handle(request, path)
        if response:
            return response
        
        return APIResponse(success=False, error="Not found", data=None)
    
    def start(self, blocking: bool = False) -> None:
        """启动服务器"""
        if self._running:
            return
        
        def handler_factory(*args, **kwargs):
            return HubHTTPHandler(self, *args, **kwargs)
        
        self._server = HTTPServer((self._host, self._port), handler_factory)
        self._running = True
        
        logger.info(f"Starting Hub API server on {self._host}:{self._port}")
        
        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """停止服务器"""
        if not self._running:
            return
        
        logger.info("Stopping Hub API server")
        
        if self._server:
            self._server.shutdown()
            self._server = None
        
        self._running = False
    
    def is_running(self) -> bool:
        """检查服务器是否运行中"""
        return self._running
