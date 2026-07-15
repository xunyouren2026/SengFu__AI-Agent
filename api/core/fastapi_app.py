"""
API服务模块 - FastAPI Service

提供完整的REST API服务：
- 认证授权
- 请求限流
- WebSocket支持
- 模型推理接口
- 批处理队列
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Tuple
from contextlib import asynccontextmanager
import asyncio
import time
import json
import hashlib
import jwt
from datetime import datetime, timedelta
from collections import defaultdict
import logging


# ============================================================================
# 数据模型
# ============================================================================

class InferenceRequest(BaseModel):
    """推理请求"""
    inputs: str = Field(..., description="输入文本或数据")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="推理参数")
    request_id: Optional[str] = Field(None, description="请求ID")


class InferenceResponse(BaseModel):
    """推理响应"""
    outputs: Any = Field(..., description="输出结果")
    request_id: str = Field(..., description="请求ID")
    latency_ms: float = Field(..., description="延迟(ms)")
    model_version: str = Field(..., description="模型版本")


class BatchRequest(BaseModel):
    """批量请求"""
    requests: List[InferenceRequest] = Field(..., description="请求列表")
    parallel: bool = Field(True, description="是否并行处理")


class TokenRequest(BaseModel):
    """令牌请求"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """令牌响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ============================================================================
# 认证
# ============================================================================

class AuthManager:
    """认证管理器"""
    
    def __init__(
        self,
        secret_key: str = "your-secret-key",
        algorithm: str = "HS256",
        token_expire_hours: int = 24
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_hours = token_expire_hours
        
        # 用户存储（实际应用应使用数据库）
        self.users: Dict[str, Dict] = {}
        
        # 令牌黑名单
        self.blacklist: set = set()
    
    def hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(
        self,
        username: str,
        password: str,
        roles: List[str] = None
    ) -> None:
        """注册用户"""
        self.users[username] = {
            'password_hash': self.hash_password(password),
            'roles': roles or ['user'],
            'created_at': datetime.now().isoformat()
        }
    
    def verify_password(self, username: str, password: str) -> bool:
        """验证密码"""
        if username not in self.users:
            return False
        return self.users[username]['password_hash'] == self.hash_password(password)
    
    def create_token(self, username: str) -> str:
        """创建令牌"""
        expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
        payload = {
            'sub': username,
            'roles': self.users.get(username, {}).get('roles', []),
            'exp': expire
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[Dict]:
        """解码令牌"""
        if token in self.blacklist:
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str) -> None:
        """撤销令牌"""
        self.blacklist.add(token)
    
    def has_role(self, token: str, role: str) -> bool:
        """检查角色"""
        payload = self.decode_token(token)
        if not payload:
            return False
        return role in payload.get('roles', [])


# ============================================================================
# 限流
# ============================================================================

class RateLimiter:
    """请求限流器"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        
        # 请求计数
        self.minute_counts: Dict[str, List[float]] = defaultdict(list)
        self.hour_counts: Dict[str, List[float]] = defaultdict(list)
        
        # 清理任务
        self._last_cleanup = time.time()
    
    def _cleanup(self) -> None:
        """清理过期记录"""
        now = time.time()
        
        if now - self._last_cleanup < 60:
            return
        
        for key in list(self.minute_counts.keys()):
            self.minute_counts[key] = [
                t for t in self.minute_counts[key]
                if now - t < 60
            ]
        
        for key in list(self.hour_counts.keys()):
            self.hour_counts[key] = [
                t for t in self.hour_counts[key]
                if now - t < 3600
            ]
        
        self._last_cleanup = now
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        检查是否允许请求
        
        Returns:
            (是否允许, 状态信息)
        """
        self._cleanup()
        
        now = time.time()
        
        # 检查每分钟限制
        minute_requests = len(self.minute_counts[client_id])
        if minute_requests >= self.rpm:
            retry_after = 60 - (now - self.minute_counts[client_id][0])
            return False, {
                'reason': 'rate_limit_exceeded',
                'limit': 'per_minute',
                'retry_after': retry_after
            }
        
        # 检查每小时限制
        hour_requests = len(self.hour_counts[client_id])
        if hour_requests >= self.rph:
            retry_after = 3600 - (now - self.hour_counts[client_id][0])
            return False, {
                'reason': 'rate_limit_exceeded',
                'limit': 'per_hour',
                'retry_after': retry_after
            }
        
        # 记录请求
        self.minute_counts[client_id].append(now)
        self.hour_counts[client_id].append(now)
        
        return True, {
            'minute_remaining': self.rpm - minute_requests - 1,
            'hour_remaining': self.rph - hour_requests - 1
        }


# ============================================================================
# 推理引擎
# ============================================================================

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model: Any, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model_version = "1.0.0"
        
        # 批处理队列
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.batch_size = 8
        self.batch_timeout = 0.1  # 100ms
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'total_latency': 0.0,
            'errors': 0
        }
    
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理"""
        start_time = time.time()
        
        try:
            # 调用模型
            if hasattr(self.model, 'generate'):
                output = self.model.generate(request.inputs, **request.parameters)
            elif hasattr(self.model, 'forward'):
                output = self.model.forward(request.inputs, **request.parameters)
            else:
                output = self.model(request.inputs, **request.parameters)
            
            latency = (time.time() - start_time) * 1000
            
            self.stats['total_requests'] += 1
            self.stats['total_latency'] += latency
            
            return InferenceResponse(
                outputs=output,
                request_id=request.request_id or self._generate_id(),
                latency_ms=latency,
                model_version=self.model_version
            )
        
        except Exception as e:
            self.stats['errors'] += 1
            raise HTTPException(status_code=500, detail=str(e))
    
    async def batch_infer(self, batch: BatchRequest) -> List[InferenceResponse]:
        """批量推理"""
        if batch.parallel:
            tasks = [self.infer(req) for req in batch.requests]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for req in batch.requests:
                results.append(await self.infer(req))
            return results
    
    def _generate_id(self) -> str:
        """生成请求ID"""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]


# ============================================================================
# WebSocket管理
# ============================================================================

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """接受连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            'connected_at': datetime.now().isoformat(),
            'message_count': 0
        }
    
    def disconnect(self, client_id: str) -> None:
        """断开连接"""
        self.active_connections.pop(client_id, None)
        self.connection_metadata.pop(client_id, None)
    
    async def send_message(self, client_id: str, message: Dict) -> None:
        """发送消息"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
            self.connection_metadata[client_id]['message_count'] += 1
    
    async def broadcast(self, message: Dict) -> None:
        """广播消息"""
        for connection in self.active_connections.values():
            await connection.send_json(message)
    
    async def send_stream(
        self,
        client_id: str,
        generator: AsyncGenerator
    ) -> None:
        """发送流式数据"""
        if client_id not in self.active_connections:
            return
        
        websocket = self.active_connections[client_id]
        async for chunk in generator:
            await websocket.send_json({'type': 'chunk', 'data': chunk})
        
        await websocket.send_json({'type': 'done'})


# ============================================================================
# FastAPI应用
# ============================================================================

def create_app(
    model: Any = None,
    auth_manager: Optional[AuthManager] = None,
    rate_limiter: Optional[RateLimiter] = None
) -> FastAPI:
    """
    创建FastAPI应用
    
    Args:
        model: 推理模型
        auth_manager: 认证管理器
        rate_limiter: 限流器
        
    Returns:
        FastAPI应用
    """
    # 生命周期管理
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动
        logging.info("Starting API server...")
        yield
        # 关闭
        logging.info("Shutting down API server...")
    
    app = FastAPI(
        title="AGI Unified Framework API",
        description="REST API for AGI model inference",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 组件
    auth = auth_manager or AuthManager()
    limiter = rate_limiter or RateLimiter()
    engine = InferenceEngine(model) if model else None
    ws_manager = ConnectionManager()
    
    security = HTTPBearer()

    # 开发模式开关（生产环境改为False）
    DEV_MODE = True

    # 依赖注入
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict:
        # 开发模式：直接返回admin，跳过认证
        if DEV_MODE:
            return {"sub": "admin", "roles": ["*"], "permissions": ["*"]}
        token = credentials.credentials
        payload = auth.decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    
    async def check_rate_limit(request: Request) -> None:
        client_id = request.client.host if request.client else "unknown"
        allowed, info = limiter.is_allowed(client_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=info,
                headers={"Retry-After": str(int(info.get('retry_after', 60)))}
            )
    
    # ============================================================================
    # 路由
    # ============================================================================
    
    @app.get("/api")
    async def root():
        return {"message": "AGI Unified Framework API", "version": "1.0.0"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    # 认证
    @app.post("/auth/token", response_model=TokenResponse)
    async def get_token(request: TokenRequest):
        if not auth.verify_password(request.username, request.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = auth.create_token(request.username)
        return TokenResponse(
            access_token=token,
            expires_in=auth.token_expire_hours * 3600
        )
    
    @app.post("/auth/logout")
    async def logout(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        auth.revoke_token(credentials.credentials)
        return {"message": "Logged out"}
    
    # 推理
    @app.post("/inference", response_model=InferenceResponse)
    async def inference(
        request: InferenceRequest,
        user: Dict = Depends(get_current_user),
        _: None = Depends(check_rate_limit)
    ):
        if not engine:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return await engine.infer(request)
    
    @app.post("/inference/batch", response_model=List[InferenceResponse])
    async def batch_inference(
        batch: BatchRequest,
        user: Dict = Depends(get_current_user),
        _: None = Depends(check_rate_limit)
    ):
        if not engine:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return await engine.batch_infer(batch)
    
    @app.post("/inference/stream")
    async def stream_inference(
        request: InferenceRequest,
        user: Dict = Depends(get_current_user)
    ):
        """流式推理"""
        async def generate():
            if not engine:
                yield json.dumps({"error": "Model not loaded"})
                return
            
            # 模拟流式输出
            for i in range(10):
                chunk = {"index": i, "partial": f"chunk_{i}"}
                yield json.dumps(chunk) + "\n"
                await asyncio.sleep(0.1)
        
        return StreamingResponse(generate(), media_type="application/json")
    
    # 统计
    @app.get("/stats")
    async def get_stats(user: Dict = Depends(get_current_user)):
        if not auth.has_role(user.get('sub', ''), 'admin'):
            raise HTTPException(status_code=403, detail="Admin only")
        
        return {
            'inference': engine.stats if engine else {},
            'rate_limiter': {
                'active_clients': len(limiter.minute_counts)
            }
        }
    
    # WebSocket
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await ws_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_json()
                
                # 处理消息
                if data.get('type') == 'inference':
                    if engine:
                        request = InferenceRequest(
                            inputs=data.get('inputs', ''),
                            parameters=data.get('parameters', {})
                        )
                        response = await engine.infer(request)
                        await ws_manager.send_message(client_id, {
                            'type': 'inference_result',
                            'data': response.dict()
                        })
                elif data.get('type') == 'ping':
                    await ws_manager.send_message(client_id, {'type': 'pong'})
        
        except WebSocketDisconnect:
            ws_manager.disconnect(client_id)
    
    return app
