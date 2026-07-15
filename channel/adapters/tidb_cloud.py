"""
TiDB Cloud适配器模块

该模块提供TiDB Cloud分布式SQL数据库的通道适配器实现，支持：
- 集群管理（创建、查询、删除）
- SQL执行（同步/异步）
- 连接池管理
- 数据导出
- MySQL兼容协议

API文档: https://docs.pingcap.com/tidbcloud/api/v1
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import ClientTimeout

from ...base import (
    ChannelAdapter,
    ChannelCapability,
    ChannelConfig,
    ConnectionState,
    MessagePriority,
    ReceiveResult,
    RetryConfig,
    SendResult,
)
from ...universal_message import (
    Attachment,
    AttachmentType,
    ChannelIdentity,
    MessageContent,
    MessageDirection,
    MessageMetadata,
    MessageType,
    UniversalMessage,
    UserIdentity,
)

logger = logging.getLogger(__name__)


class ClusterStatus(Enum):
    """集群状态"""
    CREATING = "CREATING"
    AVAILABLE = "AVAILABLE"
    MODIFYING = "MODIFYING"
    PAUSED = "PAUSED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class ClusterTier(Enum):
    """集群层级"""
    SERVERLESS = "SERVERLESS"
    DEDICATED = "DEDICATED"


@dataclass
class TiDBCloudConfig(ChannelConfig):
    """TiDB Cloud配置类
    
    Attributes:
        public_key: API公钥
        private_key: API私钥
        endpoint: API端点
        region: 默认区域
        connect_timeout: 连接超时时间（秒）
        read_timeout: 读取超时时间（秒）
        max_retry: 最大重试次数
        default_cluster_id: 默认集群ID
        default_project_id: 默认项目ID
    """
    public_key: str = ""
    private_key: str = ""
    endpoint: str = "https://api.tidbcloud.com/v1"
    region: str = "ap-southeast-1"
    connect_timeout: int = 30
    read_timeout: int = 60
    max_retry: int = 3
    default_cluster_id: Optional[str] = None
    default_project_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.public_key or not self.private_key:
            raise TiDBCloudConfigError("Public Key和Private Key不能为空")


class TiDBCloudConfigError(Exception):
    """TiDB Cloud配置错误"""
    pass


class TiDBCloudAPIError(Exception):
    """TiDB Cloud API错误
    
    Attributes:
        code: 错误代码
        message: 错误信息
        request_id: 请求ID
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        request_id: Optional[str] = None,
    ):
        self.code = code
        self.message = message
        self.request_id = request_id
        super().__init__(f"[{code}] {message} (RequestId: {request_id})")
    
    @property
    def is_retryable(self) -> bool:
        """判断错误是否可重试"""
        retryable_codes = ["InternalError", "ServiceUnavailable", "RequestTimeout"]
        return self.code in retryable_codes


@dataclass
class TiDBCluster:
    """TiDB集群信息
    
    Attributes:
        cluster_id: 集群ID
        project_id: 项目ID
        name: 集群名称
        status: 集群状态
        tier: 集群层级
        region: 区域
        create_time: 创建时间
        update_time: 更新时间
    """
    cluster_id: str
    project_id: str
    name: str
    status: ClusterStatus
    tier: ClusterTier
    region: str
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "cluster_id": self.cluster_id,
            "project_id": self.project_id,
            "name": self.name,
            "status": self.status.value,
            "tier": self.tier.value,
            "region": self.region,
            "create_time": self.create_time,
            "update_time": self.update_time,
        }


@dataclass
class SQLResult:
    """SQL执行结果
    
    Attributes:
        columns: 列定义
        rows: 行数据
        row_count: 行数
        execution_time: 执行时间（毫秒）
    """
    columns: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    row_count: int = 0
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "execution_time": self.execution_time,
        }


@dataclass
class ConnectionInfo:
    """连接信息
    
    Attributes:
        host: 主机地址
        port: 端口
        username: 用户名
        password: 密码
        database: 默认数据库
        ssl_mode: SSL模式
    """
    host: str
    port: int
    username: str
    password: str
    database: str = ""
    ssl_mode: str = "REQUIRED"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***",
            "database": self.database,
            "ssl_mode": self.ssl_mode,
        }
    
    @property
    def connection_string(self) -> str:
        """获取连接字符串"""
        return f"mysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class TiDBCloudAdapter(ChannelAdapter):
    """TiDB Cloud适配器
    
    提供TiDB Cloud分布式SQL数据库的统一接口，支持集群管理、
    SQL执行、连接池管理、数据导出等功能。
    
    Example:
        config = TiDBCloudConfig(
            public_key="your-public-key",
            private_key="your-private-key",
        )
        adapter = TiDBCloudAdapter(config)
        await adapter.connect()
        
        # 列出集群
        clusters = await adapter.list_clusters()
        
        # 执行SQL
        result = await adapter.execute_sql(
            "SELECT * FROM users WHERE id = ?",
            [1],
        )
        
        # 获取连接信息
        conn_info = await adapter.get_connection_info("cluster-xxx")
    """
    
    def __init__(self, config: TiDBCloudConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._connection_pool: Dict[str, Any] = {}
    
    def _initialize_capabilities(self) -> None:
        """初始化适配器能力"""
        self._capabilities = {
            ChannelCapability.TEXT_MESSAGES,
            ChannelCapability.DATA_MESSAGES,
            ChannelCapability.CHANNEL_INFO,
        }
    
    def _get_base_url(self) -> str:
        """获取基础URL"""
        return self._cfg.endpoint
    
    def _generate_signature(self, method: str, path: str, timestamp: str) -> str:
        """生成请求签名
        
        Args:
            method: HTTP方法
            path: 请求路径
            timestamp: 时间戳
            
        Returns:
            签名字符串
        """
        # 构建签名字符串
        sign_string = f"{method}\n{path}\n{timestamp}\n"
        
        # HMAC-SHA256签名
        signature = hmac.new(
            self._cfg.private_key.encode("utf-8"),
            sign_string.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        
        return base64.b64encode(signature).decode("utf-8")
    
    def _get_auth_headers(
        self,
        method: str,
        path: str,
    ) -> Dict[str, str]:
        """获取认证头
        
        Args:
            method: HTTP方法
            path: 请求路径
            
        Returns:
            认证头字典
        """
        timestamp = str(int(time.time()))
        signature = self._generate_signature(method, path, timestamp)
        
        return {
            "Authorization": f"Digest username={self._cfg.public_key}, algorithm=SHA-256, signature={signature}",
            "Date": timestamp,
            "Content-Type": "application/json",
        }
    
    async def _make_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送API请求
        
        Args:
            method: HTTP方法
            path: 请求路径
            params: URL参数
            data: 请求数据
            
        Returns:
            API响应
        """
        if not self._session:
            raise TiDBCloudAPIError("NotConnected", "适配器未连接")
        
        url = f"{self._get_base_url()}{path}"
        headers = self._get_auth_headers(method, path)
        
        timeout = ClientTimeout(
            connect=self._cfg.connect_timeout,
            total=self._cfg.read_timeout,
        )
        
        try:
            if method.upper() == "GET":
                async with self._session.get(
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            elif method.upper() == "DELETE":
                async with self._session.delete(
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            else:
                async with self._session.post(
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=timeout,
                ) as response:
                    result = await response.json()
            
            # 检查错误
            if "error" in result:
                error = result["error"]
                raise TiDBCloudAPIError(
                    error.get("code", "UnknownError"),
                    error.get("message", "Unknown error"),
                    error.get("request_id"),
                )
            
            return result
            
        except aiohttp.ClientError as e:
            raise TiDBCloudAPIError("RequestError", str(e))
    
    async def _connect_impl(self) -> bool:
        """实现连接逻辑"""
        try:
            timeout = ClientTimeout(
                connect=self._cfg.connect_timeout,
                total=self._cfg.read_timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            
            # 测试连接：获取项目列表
            await self.list_projects()
            
            self._logger.info("已连接到TiDB Cloud")
            return True
        except Exception as e:
            self._logger.error(f"连接TiDB Cloud失败: {e}")
            return False
    
    async def _disconnect_impl(self) -> None:
        """实现断开连接逻辑"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connection_pool.clear()
        self._logger.info("已断开与TiDB Cloud的连接")
    
    async def _health_check_impl(self) -> bool:
        """实现健康检查逻辑"""
        try:
            await self.list_projects()
            return True
        except Exception:
            return False
    
    async def _send_impl(self, message: UniversalMessage, priority: MessagePriority) -> SendResult:
        """实现发送逻辑（用于执行SQL）"""
        try:
            operation = message.get_context("operation", "execute_sql")
            
            if operation == "execute_sql":
                sql = message.get_context("sql")
                params = message.get_context("params", [])
                cluster_id = message.get_context("cluster_id") or self._cfg.default_cluster_id
                
                if not sql:
                    return SendResult(
                        success=False,
                        error="缺少sql参数",
                        error_code="MISSING_PARAMS",
                    )
                
                if not cluster_id:
                    return SendResult(
                        success=False,
                        error="缺少cluster_id参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.execute_sql(cluster_id, sql, params)
                return SendResult(
                    success=True,
                    message_id=str(time.time()),
                    timestamp=time.time(),
                )
            
            elif operation == "create_cluster":
                project_id = message.get_context("project_id") or self._cfg.default_project_id
                name = message.get_context("name")
                
                if not project_id or not name:
                    return SendResult(
                        success=False,
                        error="缺少project_id或name参数",
                        error_code="MISSING_PARAMS",
                    )
                
                result = await self.create_cluster(project_id, name)
                return SendResult(
                    success=True,
                    message_id=result.get("cluster_id", ""),
                    timestamp=time.time(),
                )
            
            else:
                return SendResult(
                    success=False,
                    error=f"不支持的操作: {operation}",
                    error_code="UNSUPPORTED_OPERATION",
                )
                
        except TiDBCloudAPIError as e:
            return SendResult(
                success=False,
                error=e.message,
                error_code=e.code,
            )
        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
                error_code="UNKNOWN_ERROR",
            )
    
    async def _receive_impl(self, payload: Optional[Dict]) -> ReceiveResult:
        """实现接收逻辑（用于查询数据）"""
        try:
            operation = payload.get("operation", "query") if payload else "query"
            messages = []
            
            if operation == "query_clusters":
                project_id = payload.get("project_id") if payload else self._cfg.default_project_id
                clusters = await self.list_clusters(project_id)
                for cluster in clusters:
                    content = MessageContent(text=f"[Cluster: {cluster.name}]")
                    metadata = MessageMetadata(
                        message_id=cluster.cluster_id,
                        channel_id="tidb_cloud",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.DATA,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("cluster", cluster.to_dict())
                    messages.append(msg)
            
            elif operation == "query_sql":
                cluster_id = payload.get("cluster_id") if payload else self._cfg.default_cluster_id
                sql = payload.get("sql") if payload else None
                
                if cluster_id and sql:
                    result = await self.execute_sql(cluster_id, sql)
                    content = MessageContent(text=f"[SQL Result: {result.row_count} rows]")
                    metadata = MessageMetadata(
                        message_id=str(time.time()),
                        channel_id="tidb_cloud",
                        timestamp=time.time(),
                        direction=MessageDirection.INBOUND,
                        message_type=MessageType.DATA,
                    )
                    msg = UniversalMessage(content=content, metadata=metadata)
                    msg.set_context("result", result.to_dict())
                    messages.append(msg)
            
            return ReceiveResult(success=True, messages=messages)
        except Exception as e:
            return ReceiveResult(success=False, error=str(e))
    
    # ========== 项目管理 ==========
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """获取项目列表
        
        Returns:
            项目列表
        """
        result = await self._make_request("GET", "/projects")
        return result.get("items", [])
    
    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """获取项目详情
        
        Args:
            project_id: 项目ID
            
        Returns:
            项目信息
        """
        result = await self._make_request("GET", f"/projects/{project_id}")
        return result
    
    # ========== 集群管理 ==========
    
    async def list_clusters(
        self,
        project_id: Optional[str] = None,
    ) -> List[TiDBCluster]:
        """获取集群列表
        
        Args:
            project_id: 项目ID
            
        Returns:
            集群列表
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        result = await self._make_request(
            "GET",
            f"/projects/{project_id}/clusters",
        )
        
        clusters = []
        for item in result.get("items", []):
            cluster = TiDBCluster(
                cluster_id=item.get("id", ""),
                project_id=project_id,
                name=item.get("name", ""),
                status=ClusterStatus(item.get("status", "CREATING")),
                tier=ClusterTier(item.get("cluster_type", "SERVERLESS")),
                region=item.get("region", ""),
                create_time=item.get("create_timestamp"),
                update_time=item.get("update_timestamp"),
            )
            clusters.append(cluster)
        
        return clusters
    
    async def get_cluster(
        self,
        cluster_id: str,
        project_id: Optional[str] = None,
    ) -> Optional[TiDBCluster]:
        """获取集群详情
        
        Args:
            cluster_id: 集群ID
            project_id: 项目ID
            
        Returns:
            集群信息
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        result = await self._make_request(
            "GET",
            f"/projects/{project_id}/clusters/{cluster_id}",
        )
        
        if not result:
            return None
        
        return TiDBCluster(
            cluster_id=result.get("id", ""),
            project_id=project_id,
            name=result.get("name", ""),
            status=ClusterStatus(result.get("status", "CREATING")),
            tier=ClusterTier(result.get("cluster_type", "SERVERLESS")),
            region=result.get("region", ""),
            create_time=result.get("create_timestamp"),
            update_time=result.get("update_timestamp"),
        )
    
    async def create_cluster(
        self,
        project_id: str,
        name: str,
        tier: ClusterTier = ClusterTier.SERVERLESS,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建集群
        
        Args:
            project_id: 项目ID
            name: 集群名称
            tier: 集群层级
            region: 区域
            
        Returns:
            创建结果
        """
        data = {
            "name": name,
            "cluster_type": tier.value,
            "region": region or self._cfg.region,
        }
        
        result = await self._make_request(
            "POST",
            f"/projects/{project_id}/clusters",
            data=data,
        )
        
        return {
            "success": True,
            "cluster_id": result.get("id", ""),
            "name": name,
        }
    
    async def delete_cluster(
        self,
        cluster_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        """删除集群
        
        Args:
            cluster_id: 集群ID
            project_id: 项目ID
            
        Returns:
            是否成功
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        await self._make_request(
            "DELETE",
            f"/projects/{project_id}/clusters/{cluster_id}",
        )
        
        return True
    
    async def get_connection_info(
        self,
        cluster_id: str,
        project_id: Optional[str] = None,
    ) -> ConnectionInfo:
        """获取集群连接信息
        
        Args:
            cluster_id: 集群ID
            project_id: 项目ID
            
        Returns:
            连接信息
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        result = await self._make_request(
            "GET",
            f"/projects/{project_id}/clusters/{cluster_id}/connection",
        )
        
        return ConnectionInfo(
            host=result.get("host", ""),
            port=result.get("port", 4000),
            username=result.get("username", ""),
            password=result.get("password", ""),
            database=result.get("database", ""),
            ssl_mode=result.get("ssl_mode", "REQUIRED"),
        )
    
    # ========== SQL执行 ==========
    
    async def execute_sql(
        self,
        cluster_id: str,
        sql: str,
        params: Optional[List[Any]] = None,
        project_id: Optional[str] = None,
    ) -> SQLResult:
        """执行SQL语句
        
        Args:
            cluster_id: 集群ID
            sql: SQL语句
            params: 参数列表
            project_id: 项目ID
            
        Returns:
            SQL执行结果
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        data = {
            "statements": [sql],
        }
        
        if params:
            data["parameters"] = params
        
        start_time = time.time()
        result = await self._make_request(
            "POST",
            f"/projects/{project_id}/clusters/{cluster_id}/sql",
            data=data,
        )
        execution_time = (time.time() - start_time) * 1000
        
        # 解析结果
        columns = []
        rows = []
        row_count = 0
        
        if "results" in result and result["results"]:
            query_result = result["results"][0]
            columns = query_result.get("columns", [])
            rows = query_result.get("rows", [])
            row_count = len(rows)
        
        return SQLResult(
            columns=columns,
            rows=rows,
            row_count=row_count,
            execution_time=execution_time,
        )
    
    async def execute_sql_async(
        self,
        cluster_id: str,
        sql: str,
        project_id: Optional[str] = None,
    ) -> str:
        """异步执行SQL语句
        
        Args:
            cluster_id: 集群ID
            sql: SQL语句
            project_id: 项目ID
            
        Returns:
            任务ID
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        data = {
            "statements": [sql],
            "async": True,
        }
        
        result = await self._make_request(
            "POST",
            f"/projects/{project_id}/clusters/{cluster_id}/sql",
            data=data,
        )
        
        return result.get("job_id", "")
    
    async def get_sql_result(
        self,
        cluster_id: str,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> SQLResult:
        """获取异步SQL执行结果
        
        Args:
            cluster_id: 集群ID
            job_id: 任务ID
            project_id: 项目ID
            
        Returns:
            SQL执行结果
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        result = await self._make_request(
            "GET",
            f"/projects/{project_id}/clusters/{cluster_id}/sql/{job_id}",
        )
        
        columns = []
        rows = []
        row_count = 0
        
        if "results" in result and result["results"]:
            query_result = result["results"][0]
            columns = query_result.get("columns", [])
            rows = query_result.get("rows", [])
            row_count = len(rows)
        
        return SQLResult(
            columns=columns,
            rows=rows,
            row_count=row_count,
        )
    
    # ========== 数据导出 ==========
    
    async def export_data(
        self,
        cluster_id: str,
        sql: str,
        format: str = "csv",
        project_id: Optional[str] = None,
    ) -> str:
        """导出数据
        
        Args:
            cluster_id: 集群ID
            sql: SQL查询语句
            format: 导出格式（csv/json）
            project_id: 项目ID
            
        Returns:
            导出任务ID
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        data = {
            "sql": sql,
            "format": format,
        }
        
        result = await self._make_request(
            "POST",
            f"/projects/{project_id}/clusters/{cluster_id}/export",
            data=data,
        )
        
        return result.get("job_id", "")
    
    async def get_export_status(
        self,
        cluster_id: str,
        job_id: str,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取导出任务状态
        
        Args:
            cluster_id: 集群ID
            job_id: 任务ID
            project_id: 项目ID
            
        Returns:
            导出状态
        """
        project_id = project_id or self._cfg.default_project_id
        
        if not project_id:
            raise TiDBCloudAPIError("MissingProject", "未指定项目ID")
        
        result = await self._make_request(
            "GET",
            f"/projects/{project_id}/clusters/{cluster_id}/export/{job_id}",
        )
        
        return result
    
    # ========== 连接池管理 ==========
    
    def get_connection_pool_config(self, cluster_id: str) -> Dict[str, Any]:
        """获取连接池配置
        
        Args:
            cluster_id: 集群ID
            
        Returns:
            连接池配置
        """
        return {
            "min_size": 5,
            "max_size": 20,
            "max_idle_time": 300,
            "max_lifetime": 3600,
        }
    
    async def test_connection(
        self,
        cluster_id: str,
        project_id: Optional[str] = None,
    ) -> bool:
        """测试连接
        
        Args:
            cluster_id: 集群ID
            project_id: 项目ID
            
        Returns:
            是否连接成功
        """
        try:
            result = await self.execute_sql(
                cluster_id,
                "SELECT 1",
                project_id=project_id,
            )
            return result.row_count == 1
        except Exception:
            return False
    
    # ========== 工具方法 ==========
    
    def get_capabilities(self):
        """获取适配器能力"""
        return self._capabilities
    
    def supports_capability(self, capability) -> bool:
        """检查是否支持特定能力"""
        return capability in self._capabilities
    
    def __repr__(self) -> str:
        return f"TiDBCloudAdapter(endpoint={self._cfg.endpoint})"


# ========== CLI测试代码 ==========

async def test_tidb_cloud():
    """测试TiDB Cloud适配器"""
    import os
    
    # 从环境变量获取配置
    public_key = os.environ.get("TIDB_CLOUD_PUBLIC_KEY", "test-public-key")
    private_key = os.environ.get("TIDB_CLOUD_PRIVATE_KEY", "test-private-key")
    
    config = TiDBCloudConfig(
        public_key=public_key,
        private_key=private_key,
    )
    
    adapter = TiDBCloudAdapter(config)
    
    try:
        # 连接
        print("正在连接TiDB Cloud...")
        connected = await adapter.connect()
        print(f"连接结果: {connected}")
        
        if not connected:
            print("连接失败，请检查配置")
            return
        
        # 健康检查
        healthy = await adapter.health_check()
        print(f"健康检查: {healthy}")
        
        # 获取项目列表
        print("\n获取项目列表:")
        projects = await adapter.list_projects()
        for project in projects[:5]:
            print(f"  - {project.get('name', 'Unknown')} ({project.get('id', '')})")
        
        print("\n测试完成!")
        
    except Exception as e:
        print(f"测试出错: {e}")
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # 运行测试
    asyncio.run(test_tidb_cloud())
