"""
透明代理网关 - 请求拦截与转发控制
"""
import socket
import threading
import ssl
import re
import json
import time
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import base64
from urllib.parse import urlparse, parse_qs


class InterceptAction(Enum):
    """拦截动作枚举"""
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    REDIRECT = "redirect"
    LOG = "log"
    ALERT = "alert"


@dataclass
class HTTPRequest:
    """HTTP请求封装"""
    method: str
    path: str
    version: str
    headers: Dict[str, str]
    body: bytes
    raw_request: bytes
    source_ip: str = ""
    source_port: int = 0
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = hashlib.sha256(
                f"{self.method}{self.path}{self.timestamp}".encode()
            ).hexdigest()[:16]
    
    @classmethod
    def parse(cls, raw_data: bytes, source_ip: str = "", source_port: int = 0) -> Optional['HTTPRequest']:
        """解析原始HTTP请求"""
        try:
            # 分离头部和body
            if b'\r\n\r\n' in raw_data:
                header_part, body = raw_data.split(b'\r\n\r\n', 1)
            else:
                header_part = raw_data
                body = b''
            
            header_lines = header_part.decode('utf-8', errors='ignore').split('\r\n')
            if not header_lines:
                return None
            
            # 解析请求行
            request_line = header_lines[0]
            parts = request_line.split(' ')
            if len(parts) < 3:
                return None
            
            method, path, version = parts[0], parts[1], parts[2]
            
            # 解析头部
            headers = {}
            for line in header_lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            
            return cls(
                method=method,
                path=path,
                version=version,
                headers=headers,
                body=body,
                raw_request=raw_data,
                source_ip=source_ip,
                source_port=source_port
            )
        except Exception:
            return None
    
    def to_raw(self) -> bytes:
        """转换为原始HTTP请求字节"""
        header_lines = [f"{self.method} {self.path} {self.version}"]
        for key, value in self.headers.items():
            header_lines.append(f"{key}: {value}")
        headers_str = '\r\n'.join(header_lines) + '\r\n\r\n'
        return headers_str.encode() + self.body


@dataclass
class InterceptResult:
    """拦截结果"""
    action: InterceptAction
    request_id: str
    modified_request: Optional[HTTPRequest] = None
    redirect_url: Optional[str] = None
    reason: str = ""
    rule_matched: str = ""
    timestamp: float = field(default_factory=time.time)


class RequestInterceptor:
    """请求拦截器"""
    
    def __init__(self):
        self._intercept_hooks: List[Callable[[HTTPRequest], InterceptResult]] = []
        self._block_patterns: List[re.Pattern] = []
        self._modify_rules: Dict[str, Callable[[HTTPRequest], HTTPRequest]] = {}
        self._log_callback: Optional[Callable[[str, Any], None]] = None
    
    def add_hook(self, hook: Callable[[HTTPRequest], InterceptResult]) -> None:
        """添加拦截钩子"""
        self._intercept_hooks.append(hook)
    
    def add_block_pattern(self, pattern: str) -> None:
        """添加阻止模式"""
        self._block_patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def add_modify_rule(self, name: str, modifier: Callable[[HTTPRequest], HTTPRequest]) -> None:
        """添加修改规则"""
        self._modify_rules[name] = modifier
    
    def set_log_callback(self, callback: Callable[[str, Any], None]) -> None:
        """设置日志回调"""
        self._log_callback = callback
    
    def intercept(self, request: HTTPRequest) -> InterceptResult:
        """执行拦截检查"""
        # 检查阻止模式
        for pattern in self._block_patterns:
            if pattern.search(request.path) or pattern.search(request.body.decode('utf-8', errors='ignore')):
                result = InterceptResult(
                    action=InterceptAction.BLOCK,
                    request_id=request.request_id,
                    reason=f"匹配阻止模式: {pattern.pattern}"
                )
                self._log("block", result)
                return result
        
        # 执行钩子
        for hook in self._intercept_hooks:
            result = hook(request)
            if result.action != InterceptAction.ALLOW:
                self._log("hook", result)
                return result
        
        # 执行修改规则
        modified = request
        for name, modifier in self._modify_rules.items():
            modified = modifier(modified)
        
        if modified.raw_request != request.raw_request:
            result = InterceptResult(
                action=InterceptAction.MODIFY,
                request_id=request.request_id,
                modified_request=modified,
                rule_matched=name
            )
            self._log("modify", result)
            return result
        
        return InterceptResult(
            action=InterceptAction.ALLOW,
            request_id=request.request_id
        )
    
    def _log(self, event_type: str, data: Any) -> None:
        """记录日志"""
        if self._log_callback:
            self._log_callback(event_type, data)


class TransparentProxy:
    """透明代理网关"""
    
    def __init__(
        self,
        listen_port: int = 8080,
        target_host: str = "localhost",
        target_port: int = 80,
        use_ssl: bool = False,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None
    ):
        self.listen_port = listen_port
        self.target_host = target_host
        self.target_port = target_port
        self.use_ssl = use_ssl
        self.cert_file = cert_file
        self.key_file = key_file
        
        self._interceptor = RequestInterceptor()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._active_connections: Dict[str, socket.socket] = {}
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "modified_requests": 0,
            "allowed_requests": 0
        }
        self._lock = threading.Lock()
    
    @property
    def interceptor(self) -> RequestInterceptor:
        """获取拦截器"""
        return self._interceptor
    
    def add_intercept_rule(self, rule: Callable[[HTTPRequest], InterceptResult]) -> None:
        """添加拦截规则"""
        self._interceptor.add_hook(rule)
    
    def block_path_pattern(self, pattern: str) -> None:
        """阻止匹配路径"""
        self._interceptor.add_block_pattern(pattern)
    
    def _create_server_socket(self) -> socket.socket:
        """创建服务端socket"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.listen_port))
        sock.listen(100)
        
        if self.use_ssl and self.cert_file and self.key_file:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(self.cert_file, self.key_file)
            sock = context.wrap_socket(sock, server_side=True)
        
        return sock
    
    def _handle_client(self, client_socket: socket.socket, client_addr: Tuple[str, int]) -> None:
        """处理客户端连接"""
        connection_id = hashlib.md5(f"{client_addr}{time.time()}".encode()).hexdigest()[:8]
        
        with self._lock:
            self._active_connections[connection_id] = client_socket
        
        try:
            # 接收请求数据
            request_data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                request_data += chunk
                if b'\r\n\r\n' in request_data:
                    # 检查Content-Length
                    header_part = request_data.split(b'\r\n\r\n')[0]
                    headers = {}
                    for line in header_part.decode('utf-8', errors='ignore').split('\r\n')[1:]:
                        if ':' in line:
                            k, v = line.split(':', 1)
                            headers[k.strip().lower()] = v.strip()
                    
                    content_length = int(headers.get('content-length', 0))
                    body_start = request_data.find(b'\r\n\r\n') + 4
                    if len(request_data) >= body_start + content_length:
                        break
            
            if not request_data:
                return
            
            # 解析请求
            request = HTTPRequest.parse(
                request_data,
                source_ip=client_addr[0],
                source_port=client_addr[1]
            )
            
            if not request:
                client_socket.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                return
            
            with self._lock:
                self._stats["total_requests"] += 1
            
            # 执行拦截
            result = self._interceptor.intercept(request)
            
            if result.action == InterceptAction.BLOCK:
                with self._lock:
                    self._stats["blocked_requests"] += 1
                response = self._generate_block_response(result.reason)
                client_socket.sendall(response)
                return
            
            if result.action == InterceptAction.MODIFY and result.modified_request:
                with self._lock:
                    self._stats["modified_requests"] += 1
                request = result.modified_request
            
            with self._lock:
                self._stats["allowed_requests"] += 1
            
            # 转发请求到目标
            self._forward_request(client_socket, request)
            
        except Exception as e:
            try:
                client_socket.sendall(f"HTTP/1.1 500 Internal Server Error\r\n\r\nError: {e}".encode())
            except:
                pass
        finally:
            with self._lock:
                self._active_connections.pop(connection_id, None)
            try:
                client_socket.close()
            except:
                pass
    
    def _forward_request(self, client_socket: socket.socket, request: HTTPRequest) -> None:
        """转发请求到目标服务器"""
        try:
            target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_socket.settimeout(30)
            target_socket.connect((self.target_host, self.target_port))
            
            # 发送请求
            target_socket.sendall(request.to_raw())
            
            # 接收响应
            response_data = b''
            while True:
                chunk = target_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
            
            target_socket.close()
            
            # 返回响应给客户端
            client_socket.sendall(response_data)
            
        except Exception as e:
            client_socket.sendall(f"HTTP/1.1 502 Bad Gateway\r\n\r\nGateway Error: {e}".encode())
    
    def _generate_block_response(self, reason: str) -> bytes:
        """生成阻止响应"""
        body = json.dumps({
            "error": "Request blocked",
            "reason": reason,
            "timestamp": time.time()
        })
        return f"HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
    
    def start(self) -> None:
        """启动代理"""
        self._running = True
        self._server_socket = self._create_server_socket()
        
        while self._running:
            try:
                client_socket, client_addr = self._server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, client_addr),
                    daemon=True
                )
                thread.start()
            except Exception:
                if self._running:
                    continue
                break
    
    def stop(self) -> None:
        """停止代理"""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
        
        # 关闭所有活动连接
        with self._lock:
            for conn in self._active_connections.values():
                try:
                    conn.close()
                except:
                    pass
            self._active_connections.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        with self._lock:
            return self._stats.copy()


class ProxyChain:
    """代理链"""
    
    def __init__(self):
        self._proxies: List[TransparentProxy] = []
    
    def add_proxy(self, proxy: TransparentProxy) -> None:
        """添加代理"""
        self._proxies.append(proxy)
    
    def start_all(self) -> None:
        """启动所有代理"""
        for proxy in self._proxies:
            thread = threading.Thread(target=proxy.start, daemon=True)
            thread.start()
    
    def stop_all(self) -> None:
        """停止所有代理"""
        for proxy in self._proxies:
            proxy.stop()
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """获取所有代理统计"""
        return {f"proxy_{i}": p.get_stats() for i, p in enumerate(self._proxies)}


class RateLimiter:
    """请求速率限制器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_history: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def check_rate(self, client_ip: str) -> Tuple[bool, int]:
        """检查速率限制"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        with self._lock:
            # 清理过期记录
            if client_ip in self._request_history:
                self._request_history[client_ip] = [
                    t for t in self._request_history[client_ip] if t > cutoff_time
                ]
            else:
                self._request_history[client_ip] = []
            
            # 检查是否超限
            request_count = len(self._request_history[client_ip])
            if request_count >= self.max_requests:
                return False, self.max_requests - request_count
            
            # 记录新请求
            self._request_history[client_ip].append(current_time)
            return True, self.max_requests - request_count - 1
    
    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """获取客户端统计"""
        with self._lock:
            if client_ip in self._request_history:
                return {
                    "request_count": len(self._request_history[client_ip]),
                    "window_seconds": self.window_seconds,
                    "max_requests": self.max_requests
                }
            return {"request_count": 0}
