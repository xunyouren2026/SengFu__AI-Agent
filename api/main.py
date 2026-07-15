"""
FastAPI主应用模块

提供AGI Unified Framework的REST API服务入口。
包含应用初始化、中间件注册、路由挂载和生命周期管理。

主要组件:
    - AGIAPIApplication: API应用主类
    - create_app: 应用工厂函数
    - lifespan: 应用生命周期管理

使用示例:
    >>> from agi_unified_framework.api import create_app
    >>> app = create_app()
    >>> # 使用uvicorn启动
    >>> # uvicorn main:app --reload
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# 配置日志
logger = logging.getLogger(__name__)

# 全局状态
_app_state: Dict[str, Any] = {
    "started_at": None,
    "ready": False,
    "components": {},
}


class AGIAPIApplication:
    """
    AGI API应用主类
    
    封装FastAPI应用，提供统一的配置和管理接口。
    
    Attributes:
        app: FastAPI应用实例
        config: 应用配置
        middlewares: 已注册的中间件列表
        startup_handlers: 启动处理器列表
        shutdown_handlers: 关闭处理器列表
    
    Example:
        >>> api_app = AGIAPIApplication()
        >>> api_app.setup()
        >>> app = api_app.app
    """
    
    def __init__(
        self,
        title: str = "AGI Unified Framework API",
        version: str = "1.0.0",
        description: str = "Unified API Gateway for AGI System",
        debug: bool = False,
        docs_url: str = "/docs",
        redoc_url: str = "/redoc",
        openapi_url: str = "/openapi.json",
    ):
        """
        初始化API应用
        
        Args:
            title: API标题
            version: API版本
            description: API描述
            debug: 是否启用调试模式
            docs_url: Swagger UI路径
            redoc_url: ReDoc路径
            openapi_url: OpenAPI规范路径
        """
        self.title = title
        self.version = version
        self.description = description
        self.debug = debug
        self.docs_url = docs_url
        self.redoc_url = redoc_url
        self.openapi_url = openapi_url
        
        self.app: Optional[FastAPI] = None
        self.middlewares: List[Dict[str, Any]] = []
        self.startup_handlers: List[Callable] = []
        self.shutdown_handlers: List[Callable] = []
        self._routers: List[Any] = []
        
    def create_app(self) -> FastAPI:
        """
        创建FastAPI应用实例
        
        Returns:
            FastAPI: 配置好的应用实例
        """
        self.app = FastAPI(
            title=self.title,
            version=self.version,
            description=self.description,
            debug=self.debug,
            docs_url=self.docs_url,
            redoc_url=self.redoc_url,
            openapi_url=self.openapi_url,
            lifespan=lifespan,
        )
        return self.app
    
    def setup(self) -> FastAPI:
        """
        完整设置应用（创建 + 配置）
        
        Returns:
            FastAPI: 配置好的应用实例
        """
        self.create_app()
        self._setup_middlewares()
        self._setup_routes()
        self._setup_exception_handlers()
        self._setup_event_handlers()
        return self.app
    
    def _setup_middlewares(self) -> None:
        """设置中间件 - 使用FastAPI兼容方式"""
        if not self.app:
            raise RuntimeError("App not created. Call create_app() first.")
        
        # ============================================
        # 1. CORS中间件 - 使用FastAPI内置的CORSMiddleware
        # ============================================
        allowed_origins = self._get_allowed_origins()
        self.app.add_middleware(
            FastAPICORSMiddleware,
            allow_origins=["*"],  # 开发环境允许所有源
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS中间件已注册")
        
        # ============================================
        # 2. GZip压缩中间件
        # ============================================
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        logger.info("GZip压缩中间件已注册")
        
        # ============================================
        # 注意：自定义认证中间件暂禁用，改用依赖注入方式实现
        # 如需启用认证，请使用 FastAPI 的 Depends 机制
        # ============================================
    
    def _get_jwt_secret(self) -> str:
        """获取JWT密钥"""
        import os
        secret = os.getenv("JWT_SECRET_KEY", "")
        if not secret:
            # 生成一个临时密钥（仅用于开发）
            import secrets
            secret = secrets.token_urlsafe(32)
            logger.warning("JWT_SECRET_KEY未设置，使用临时密钥（仅开发环境）")
        return secret
    
    def _get_allowed_origins(self) -> List[str]:
        """获取允许的CORS来源"""
        import os
        origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
        if origins_str:
            return [o.strip() for o in origins_str.split(",")]
        if self.debug:
            return ["*"]
        # 生产环境默认只允许同源
        return ["http://localhost:8000", "http://localhost:3000"]
    
    def _get_allowed_hosts(self) -> List[str]:
        """获取允许的主机"""
        import os
        hosts_str = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1")
        return [h.strip() for h in hosts_str.split(",")]
    
    def _setup_routes(self) -> None:
        """设置路由"""
        if not self.app:
            raise RuntimeError("App not created. Call create_app() first.")
        
        # 导入并注册路由
        from .routes import (
            personality,
            channel,
            message,
            plugin,
            routing,
            metrics,
            health,
            models,
            orchestration,
            workflows,
            # 新增: 注册缺失的8个路由模块
            dashboard,
            chat,
            chat_integrated,
            agents,
            cognitive,
            training,
            system,
            advanced,
            # 真实功能路由
            generation_real as generation,
            chat_multimodal,
            computer_use_api,
            dashboard_real,
            # 新增: 遥测和硬件监控路由
            telemetry,
            hardware,
            # 算法集成路由
            algorithms,
            training_algorithms,
            multiagent_algorithms,
            swing_layer_algorithms,
            reasoning_algorithms,
            memory_algorithms,
        )
        
        # API版本前缀
        api_prefix = "/api/v1"
        
        # 注册路由
        self.app.include_router(
            personality.router,
            prefix=f"{api_prefix}/personality",
            tags=["Personality"],
        )
        self.app.include_router(
            channel.router,
            prefix=f"{api_prefix}/channels",
            tags=["Channels"],
        )
        self.app.include_router(
            message.router,
            prefix=f"{api_prefix}/messages",
            tags=["Messages"],
        )
        self.app.include_router(
            plugin.router,
            prefix=f"{api_prefix}/plugins",
            tags=["Plugins"],
        )
        self.app.include_router(
            routing.router,
            prefix=f"{api_prefix}/routing",
            tags=["Routing"],
        )
        self.app.include_router(
            metrics.router,
            prefix=f"{api_prefix}/metrics",
            tags=["Metrics"],
        )
        self.app.include_router(
            health.router,
            prefix=f"{api_prefix}/health",
            tags=["Health"],
        )
        self.app.include_router(
            models.router,
            prefix=f"{api_prefix}/models",
            tags=["Models"],
        )
        self.app.include_router(
            orchestration.router,
            prefix=f"{api_prefix}/orchestration",
            tags=["Orchestration"],
        )
        self.app.include_router(
            workflows.router,
            prefix=f"{api_prefix}/workflows",
            tags=["Workflows"],
        )
        
        # 注册新增的8个路由
        self.app.include_router(
            dashboard.router,
            prefix=f"{api_prefix}/dashboard",
            tags=["Dashboard"],
        )
        self.app.include_router(
            chat.router,
            prefix=f"{api_prefix}/chat",
            tags=["Chat"],
        )
        self.app.include_router(
            chat_integrated.router,
            prefix=f"{api_prefix}/chat-integrated",
            tags=["Chat Integrated"],
        )
        self.app.include_router(
            agents.router,
            prefix=f"{api_prefix}/agents",
            tags=["Agents"],
        )
        self.app.include_router(
            cognitive.router,
            prefix=f"{api_prefix}/cognitive",
            tags=["Cognitive"],
        )
        self.app.include_router(
            training.router,
            prefix=f"{api_prefix}/training",
            tags=["Training"],
        )
        self.app.include_router(
            system.router,
            prefix=f"{api_prefix}/system",
            tags=["System"],
        )
        self.app.include_router(
            advanced.router,
            prefix=f"{api_prefix}/advanced",
            tags=["Advanced"],
        )
        self.app.include_router(
            generation.router,
            prefix=f"{api_prefix}/generation",
            tags=["Generation"],
        )
        self.app.include_router(
            chat_multimodal.router,
            prefix=f"{api_prefix}/chat-multimodal",
            tags=["Chat Multimodal"],
        )
        self.app.include_router(
            computer_use_api.router,
            prefix=f"{api_prefix}/computer-use",
            tags=["Computer Use"],
        )
        self.app.include_router(
            dashboard_real.router,
            prefix=f"{api_prefix}/dashboard-real",
            tags=["Dashboard Real"],
        )
        self.app.include_router(
            telemetry.router,
            prefix=api_prefix,
            tags=["Telemetry"],
        )
        self.app.include_router(
            hardware.router,
            prefix=api_prefix,
            tags=["Hardware"],
        )
        
        # 算法集成路由
        self.app.include_router(
            algorithms.router,
            prefix=api_prefix,
            tags=["Algorithms"],
        )
        self.app.include_router(
            training_algorithms.router,
            prefix=api_prefix,
            tags=["Training Algorithms"],
        )
        self.app.include_router(
            multiagent_algorithms.router,
            prefix=api_prefix,
            tags=["Multiagent Algorithms"],
        )
        self.app.include_router(
            swing_layer_algorithms.router,
            prefix=api_prefix,
            tags=["Swing Layer Algorithms"],
        )
        self.app.include_router(
            reasoning_algorithms.router,
            prefix=api_prefix,
            tags=["Reasoning Algorithms"],
        )
        self.app.include_router(
            memory_algorithms.router,
            prefix=api_prefix,
            tags=["Memory Algorithms"],
        )
        
        # API信息 (根路径由静态文件服务提供 index.html)
        @self.app.get("/api", tags=["Root"])
        async def api_info():
            """API信息"""
            return {
                "name": self.title,
                "version": self.version,
                "description": self.description,
                "version_detail": "v1",
                "endpoints": {
                    "personality": "/api/v1/personality",
                    "channels": "/api/v1/channels",
                    "messages": "/api/v1/messages",
                    "plugins": "/api/v1/plugins",
                    "routing": "/api/v1/routing",
                    "metrics": "/api/v1/metrics",
                    "health": "/api/v1/health",
                    "models": "/api/v1/models",
                    "orchestration": "/api/v1/orchestration",
                    "workflows": "/api/v1/workflows",
                    "dashboard": "/api/v1/dashboard",
                    "chat": "/api/v1/chat",
                    "agents": "/api/v1/agents",
                    "cognitive": "/api/v1/cognitive",
                    "training": "/api/v1/training",
                    "system": "/api/v1/system",
                    "advanced": "/api/v1/advanced",
                    "generation": "/api/v1/generation",
                    "chat-multimodal": "/api/v1/chat-multimodal",
                    "computer-use": "/api/v1/computer-use",
                    "dashboard-real": "/api/v1/dashboard-real",
                }
            }
        
        # 挂载静态文件
        import os
        web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
        if os.path.exists(web_dir):
            self.app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")
            logger.info(f"Static files mounted from: {web_dir}")
        else:
            logger.warning(f"Web directory not found: {web_dir}")
        
        # 挂载uploads静态文件目录
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
        if os.path.exists(uploads_dir):
            self.app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
            logger.info(f"Upload files mounted from: {uploads_dir}")
        else:
            # 如果uploads目录不存在，自动创建
            os.makedirs(uploads_dir, exist_ok=True)
            self.app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
            logger.info(f"Upload directory created and mounted from: {uploads_dir}")
        
        logger.debug("Routes configured")
    
    def _setup_exception_handlers(self) -> None:
        """设置异常处理器"""
        if not self.app:
            raise RuntimeError("App not created. Call create_app() first.")
        
        @self.app.exception_handler(AssertionError)
        async def assertion_handler(request: Request, exc: AssertionError):
            """处理StaticFiles的WebSocket AssertionError - 静默忽略"""
            # StaticFiles收到WebSocket请求时会抛出assert scope["type"] == "http"
            # 这是正常行为，静默处理避免日志刷屏
            pass
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """全局异常处理器"""
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "detail": str(exc) if self.debug else "An unexpected error occurred",
                    "path": str(request.url.path),
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        
        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: Any):
            """404处理器"""
            request_id = getattr(request.state, 'request_id', 'unknown')
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "Not found",
                    "path": str(request.url.path),
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id},
            )
        
        logger.debug("Exception handlers configured")
    
    def _setup_event_handlers(self) -> None:
        """设置事件处理器"""
        if not self.app:
            raise RuntimeError("App not created. Call create_app() first.")
        
        @self.app.on_event("startup")
        async def on_startup():
            """启动事件"""
            logger.info(f"Starting {self.title} v{self.version}")
            _app_state["started_at"] = time.time()
            
            for handler in self.startup_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Startup handler failed: {e}")
            
            _app_state["ready"] = True
            logger.info("Application started successfully")
        
        @self.app.on_event("shutdown")
        async def on_shutdown():
            """关闭事件"""
            logger.info("Shutting down application")
            _app_state["ready"] = False
            
            for handler in self.shutdown_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    logger.error(f"Shutdown handler failed: {e}")
            
            logger.info("Application shut down")
    
    def add_startup_handler(self, handler: Callable) -> None:
        """添加启动处理器"""
        self.startup_handlers.append(handler)
    
    def add_shutdown_handler(self, handler: Callable) -> None:
        """添加关闭处理器"""
        self.shutdown_handlers.append(handler)
    
    def get_state(self) -> Dict[str, Any]:
        """获取应用状态"""
        return _app_state.copy()


# =============================================================================
# 安全中间件类
# =============================================================================

class SecurityHeadersMiddleware:
    """
    安全响应头中间件
    
    添加安全相关的HTTP响应头，防止常见的Web攻击。
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                
                # 添加安全响应头
                security_headers = [
                    (b"X-Content-Type-Options", b"nosniff"),
                    (b"X-Frame-Options", b"DENY"),
                    (b"X-XSS-Protection", b"1; mode=block"),
                    (b"Referrer-Policy", b"strict-origin-when-cross-origin"),
                    (b"Permissions-Policy", b"accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"),
                    (b"Strict-Transport-Security", b"max-age=31536000; includeSubDomains"),
                    (b"Content-Security-Policy", b"default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' ws: wss:;"),
                ]
                
                # 合并现有头和新头
                existing_names = {name.lower() for name, _ in headers}
                for name, value in security_headers:
                    if name.lower() not in existing_names:
                        headers.append((name, value))
                
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_with_headers)


class RequestIDMiddleware:
    """
    请求ID中间件
    
    为每个请求生成唯一的请求ID，便于追踪和调试。
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # 生成或获取请求ID
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id", b"").decode() if b"x-request-id" in headers else None
        
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        # 存储在scope中供后续使用
        scope["request_id"] = request_id
        
        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                # 添加请求ID到响应头
                headers.append((b"X-Request-ID", request_id.encode()))
                message["headers"] = headers
            
            await send(message)
        
        await self.app(scope, receive, send_with_request_id)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    应用生命周期管理器
    
    处理应用的启动和关闭逻辑。
    
    Args:
        app: FastAPI应用实例
        
    Yields:
        None
    """
    # 启动
    logger.info("Application starting up...")
    _app_state["started_at"] = time.time()
    
    try:
        # 初始化组件
        await _initialize_components()
        _app_state["ready"] = True
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # 关闭
    logger.info("Application shutting down...")
    _app_state["ready"] = False
    
    try:
        await _cleanup_components()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


async def _initialize_components() -> None:
    """初始化应用组件"""
    components = {}
    
    # 初始化指标收集器
    try:
        from ..telemetry.metrics.collector import MetricsCollector
        from ..telemetry.config import MetricsConfig
        
        metrics_config = MetricsConfig()
        metrics_collector = MetricsCollector(metrics_config)
        metrics_collector.start()
        components["metrics_collector"] = metrics_collector
        logger.debug("Metrics collector initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize metrics collector: {e}")
    
    # 初始化渠道管理器
    try:
        from ..channel.gateway import ChannelGateway
        components["channel_gateway"] = ChannelGateway()
        logger.debug("Channel gateway initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize channel gateway: {e}")
    
    # 初始化人格引擎
    try:
        from ..memory.personality.personality_engine import PersonalityEngine
        components["personality_engine"] = PersonalityEngine()
        logger.debug("Personality engine initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize personality engine: {e}")
    
    # 初始化插件管理器
    try:
        from ..core.plugins import PluginLifecycleManager
        components["plugin_manager"] = PluginLifecycleManager()
        logger.debug("Plugin manager initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize plugin manager: {e}")
    
    _app_state["components"] = components


async def _cleanup_components() -> None:
    """清理应用组件"""
    components = _app_state.get("components", {})
    
    # 关闭指标收集器
    if "metrics_collector" in components:
        try:
            components["metrics_collector"].shutdown()
            logger.debug("Metrics collector shut down")
        except Exception as e:
            logger.warning(f"Error shutting down metrics collector: {e}")
    
    # 清理其他组件
    for name, component in components.items():
        if hasattr(component, 'close'):
            try:
                if asyncio.iscoroutinefunction(component.close):
                    await component.close()
                else:
                    component.close()
                logger.debug(f"Component {name} closed")
            except Exception as e:
                logger.warning(f"Error closing component {name}: {e}")
    
    _app_state["components"] = {}


def create_app(
    title: str = "AGI Unified Framework API",
    version: str = "1.0.0",
    description: str = "Unified API Gateway for AGI System",
    debug: bool = False,
    **kwargs: Any
) -> FastAPI:
    """
    创建FastAPI应用工厂函数
    
    创建并配置完整的FastAPI应用实例。
    
    Args:
        title: API标题
        version: API版本
        description: API描述
        debug: 是否启用调试模式
        **kwargs: 其他配置参数
        
    Returns:
        FastAPI: 配置好的应用实例
        
    Example:
        >>> app = create_app(debug=True)
        >>> # 启动服务
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    api_app = AGIAPIApplication(
        title=title,
        version=version,
        description=description,
        debug=debug,
        **kwargs
    )
    return api_app.setup()


def get_application() -> Optional[FastAPI]:
    """
    获取当前应用实例
    
    Returns:
        Optional[FastAPI]: 应用实例或None
    """
    # 返回全局状态中的应用引用
    return _app_state.get("app")


def get_app_state() -> Dict[str, Any]:
    """
    获取应用状态
    
    Returns:
        Dict: 应用状态字典
    """
    return _app_state.copy()


def is_ready() -> bool:
    """
    检查应用是否就绪
    
    Returns:
        bool: 是否就绪
    """
    return _app_state.get("ready", False)


def get_uptime() -> float:
    """
    获取应用运行时间
    
    Returns:
        float: 运行时间（秒）
    """
    started_at = _app_state.get("started_at")
    if started_at:
        return time.time() - started_at
    return 0.0


# 导出
__all__ = [
    "AGIAPIApplication",
    "create_app",
    "get_application",
    "lifespan",
    "get_app_state",
    "is_ready",
    "get_uptime",
    "app",
    # 安全中间件
    "SecurityHeadersMiddleware",
    "RequestIDMiddleware",
]

# 创建应用实例（用于uvicorn启动）
app = create_app(debug=False)
