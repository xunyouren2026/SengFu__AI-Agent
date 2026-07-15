"""
HTTP降级通信
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import json
import urllib.parse


class HTTPMethod(Enum):
    """HTTP方法"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class HTTPRequest:
    """HTTP请求"""
    
    def __init__(
        self,
        method: HTTPMethod,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None
    ):
        self.method = method
        self.path = path
        self.headers = headers or {}
        self.body = body or {}
        self.query_params = query_params or {}
        self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'path': self.path,
            'headers': self.headers,
            'body': self.body,
            'query_params': self.query_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HTTPRequest':
        """从字典创建"""
        return cls(
            method=HTTPMethod(data['method']),
            path=data['path'],
            headers=data.get('headers'),
            body=data.get('body'),
            query_params=data.get('query_params')
        )


class HTTPResponse:
    """HTTP响应"""
    
    def __init__(
        self,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body or {}
        self.timestamp = datetime.now().timestamp()
    
    @property
    def is_success(self) -> bool:
        """是否成功"""
        return 200 <= self.status_code < 300
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'status_code': self.status_code,
            'headers': self.headers,
            'body': self.body
        }


class Route:
    """路由"""
    
    def __init__(
        self,
        path: str,
        method: HTTPMethod,
        handler: Callable[[HTTPRequest], HTTPResponse]
    ):
        self.path = path
        self.method = method
        self.handler = handler


class HTTPFallbackService:
    """
    HTTP降级通信服务
    
    当gRPC不可用时的降级方案
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: float = 30.0
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._routes: Dict[str, Dict[HTTPMethod, Route]] = {}
        self._middleware: List[Callable] = []
        self._is_running = False
        
        # 统计
        self._total_requests = 0
        self._total_errors = 0
    
    def start(self) -> None:
        """启动服务"""
        self._is_running = True
    
    def stop(self) -> None:
        """停止服务"""
        self._is_running = False
    
    def route(
        self,
        path: str,
        method: HTTPMethod = HTTPMethod.GET
    ) -> Callable:
        """
        路由装饰器
        
        @service.route("/api/model", HTTPMethod.POST)
        def handle_model(request):
            return HTTPResponse(200, body={"status": "ok"})
        """
        def decorator(handler: Callable) -> Callable:
            self.add_route(path, method, handler)
            return handler
        return decorator
    
    def add_route(
        self,
        path: str,
        method: HTTPMethod,
        handler: Callable[[HTTPRequest], HTTPResponse]
    ) -> None:
        """添加路由"""
        if path not in self._routes:
            self._routes[path] = {}
        
        self._routes[path][method] = Route(path, method, handler)
    
    def remove_route(self, path: str, method: HTTPMethod) -> None:
        """移除路由"""
        if path in self._routes and method in self._routes[path]:
            del self._routes[path][method]
    
    def add_middleware(
        self,
        middleware: Callable[[HTTPRequest, Callable], HTTPResponse]
    ) -> None:
        """添加中间件"""
        self._middleware.append(middleware)
    
    def handle_request(self, request: HTTPRequest) -> HTTPResponse:
        """
        处理请求
        
        Args:
            request: HTTP请求
        
        Returns:
            HTTP响应
        """
        self._total_requests += 1
        
        if not self._is_running:
            return HTTPResponse(503, body={'error': 'Service unavailable'})
        
        # 查找路由
        path = request.path
        method = request.method
        
        if path not in self._routes:
            return HTTPResponse(404, body={'error': 'Not found'})
        
        if method not in self._routes[path]:
            return HTTPResponse(405, body={'error': 'Method not allowed'})
        
        route = self._routes[path][method]
        
        # 应用中间件
        handler = route.handler
        for middleware in reversed(self._middleware):
            handler = lambda req, h=handler, m=middleware: m(req, h)
        
        try:
            response = handler(request)
            return response
        except Exception as e:
            self._total_errors += 1
            return HTTPResponse(500, body={'error': str(e)})
    
    def get(self, path: str) -> Callable:
        """GET路由快捷方式"""
        return self.route(path, HTTPMethod.GET)
    
    def post(self, path: str) -> Callable:
        """POST路由快捷方式"""
        return self.route(path, HTTPMethod.POST)
    
    def put(self, path: str) -> Callable:
        """PUT路由快捷方式"""
        return self.route(path, HTTPMethod.PUT)
    
    def delete(self, path: str) -> Callable:
        """DELETE路由快捷方式"""
        return self.route(path, HTTPMethod.DELETE)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'is_running': self._is_running,
            'host': self.host,
            'port': self.port,
            'total_routes': sum(len(methods) for methods in self._routes.values()),
            'total_requests': self._total_requests,
            'total_errors': self._total_errors,
            'error_rate': self._total_errors / self._total_requests if self._total_requests > 0 else 0
        }


class HTTPClient:
    """
    HTTP客户端
    
    用于与HTTP服务通信
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._headers: Dict[str, str] = {
            'Content-Type': 'application/json'
        }
        
        # 统计
        self._total_requests = 0
        self._total_errors = 0
    
    def set_header(self, key: str, value: str) -> None:
        """设置请求头"""
        self._headers[key] = value
    
    def set_auth_token(self, token: str) -> None:
        """设置认证令牌"""
        self._headers['Authorization'] = f'Bearer {token}'
    
    def request(
        self,
        method: HTTPMethod,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """
        发送请求
        
        Args:
            method: HTTP方法
            path: 路径
            body: 请求体
            query_params: 查询参数
        
        Returns:
            HTTP响应
        """
        self._total_requests += 1
        
        # 构建URL
        url = f"{self.base_url}{path}"
        if query_params:
            url += '?' + urllib.parse.urlencode(query_params)
        
        # 模拟请求（实际实现需要使用http.client或urllib）
        # 这里返回模拟响应
        return HTTPResponse(
            status_code=200,
            body={'message': 'Request processed (simulated)'}
        )
    
    def get(
        self,
        path: str,
        query_params: Optional[Dict[str, str]] = None
    ) -> HTTPResponse:
        """GET请求"""
        return self.request(HTTPMethod.GET, path, query_params=query_params)
    
    def post(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None
    ) -> HTTPResponse:
        """POST请求"""
        return self.request(HTTPMethod.POST, path, body=body)
    
    def put(
        self,
        path: str,
        body: Optional[Dict[str, Any]] = None
    ) -> HTTPResponse:
        """PUT请求"""
        return self.request(HTTPMethod.PUT, path, body=body)
    
    def delete(self, path: str) -> HTTPResponse:
        """DELETE请求"""
        return self.request(HTTPMethod.DELETE, path)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'base_url': self.base_url,
            'total_requests': self._total_requests,
            'total_errors': self._total_errors
        }


class CommunicationFallback:
    """
    通信降级管理器
    
    管理gRPC和HTTP之间的降级切换
    """
    
    def __init__(
        self,
        grpc_available: bool = True,
        fallback_threshold: int = 3
    ):
        self.grpc_available = grpc_available
        self.fallback_threshold = fallback_threshold
        
        self._consecutive_failures = 0
        self._using_fallback = False
        self._grpc_service: Optional[Any] = None
        self._http_service: Optional[HTTPFallbackService] = None
    
    def set_grpc_service(self, service: Any) -> None:
        """设置gRPC服务"""
        self._grpc_service = service
    
    def set_http_service(self, service: HTTPFallbackService) -> None:
        """设置HTTP服务"""
        self._http_service = service
    
    def report_success(self) -> None:
        """报告成功"""
        self._consecutive_failures = 0
        if self._using_fallback and self.grpc_available:
            # 尝试切回gRPC
            self._using_fallback = False
    
    def report_failure(self) -> None:
        """报告失败"""
        self._consecutive_failures += 1
        
        if self._consecutive_failures >= self.fallback_threshold:
            self._using_fallback = True
    
    def get_service(self) -> Any:
        """获取当前使用的服务"""
        if self._using_fallback or not self.grpc_available:
            return self._http_service
        return self._grpc_service
    
    def is_using_fallback(self) -> bool:
        """是否使用降级"""
        return self._using_fallback
    
    def force_fallback(self) -> None:
        """强制使用降级"""
        self._using_fallback = True
    
    def force_grpc(self) -> None:
        """强制使用gRPC"""
        if self.grpc_available:
            self._using_fallback = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'grpc_available': self.grpc_available,
            'using_fallback': self._using_fallback,
            'consecutive_failures': self._consecutive_failures,
            'fallback_threshold': self.fallback_threshold
        }
